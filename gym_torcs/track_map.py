#!/usr/bin/env python
# track_map.py
#
# Global racing line + min-time speed profile over a whole circuit.
#
# The local controller in torcs_mpc.py can only see ~120 m of track at a time,
# so its racing line is myopic: it cannot set up a corner several turns ahead
# and it re-solves a fresh local line every frame. This module lifts the same
# minimum-curvature idea to the *whole* closed lap, then reweights it by the
# lap-time cost for the learned car.
#
# Pipeline:
#   1. A TrackMap stores, on a uniform arc-length grid s in [0, length):
#        kappa(s) - signed centreline curvature   (1/m, + = left)
#        width(s) - full track width              (m)
#   2. build() solves, over the closed loop:
#        n*(s)  - lateral offset of the racing line from the centreline
#                 (+ = left), the periodic minimum-curvature line, and
#        v*(s)  - the min-time speed profile for that line (grip, steering,
#                 understeer, braking, and speed-dependent acceleration).
#   3. Live, torcs_mpc.py looks up n*(s) and v*(s) by distFromStart and uses
#      them to bias the local line and set the speed target.
#
# A map can be built two ways:
#   * from_centerline(X, Y, width)  - from known geometry (used to validate the
#                                     optimiser in the offline sim), or
#   * from_map_log(csv)             - from a logged warm-up lap (live use).
#
# Everything here is offline / one-shot, so clarity beats micro-optimisation.

import argparse
import json
import math
import os

import numpy as np

try:
    import scipy.sparse as sp
    from scipy.optimize import lsq_linear
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _periodic_smooth(a, smooth_m, ds):
    a = np.asarray(a, dtype=float)
    w = max(3, int(round(smooth_m / ds)) | 1)
    if a.shape[0] < w:
        return a
    k = np.ones(w) / w
    return np.convolve(np.concatenate([a[-w:], a, a[:w]]),
                       k, mode='same')[w:-w]


def _default_model_path():
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(os.getcwd(), 'dynamics_model.json'),
                 os.path.join(here, 'dynamics_model.json')):
        if os.path.exists(path):
            return path
    return None


def load_car_limits(model_path=None, dynamics=None, params=None, base=None):
    """Return the vehicle limits used by the global racing-line builder.

    Values are pulled from torcs_mpc.PARAMS first, then overridden by the learned
    dynamics model when available, mirroring how the live controller plans.
    """
    car = dict(a_lat=16.0, a_brake=11.0, a_accel=7.0,
               accel_a0=None, accel_a1=0.0,
               wheelbase=2.4, wheelbase_eff=2.4, understeer_K=0.0,
               steer_lock=0.366, v_max=95.0, v_min=8.0,
               combined_grip=0.25, source='built-in defaults')
    try:
        from torcs_mpc import PARAMS
        p = dict(PARAMS)
        if params:
            p.update(params)
        car.update(
            a_lat=float(p.get('a_lat', car['a_lat'])),
            a_brake=float(p.get('a_brake', car['a_brake'])),
            a_accel=float(p.get('a_accel', car['a_accel'])),
            accel_a0=float(p.get('a_accel', car['a_accel'])),
            wheelbase=float(p.get('wheelbase', car['wheelbase'])),
            wheelbase_eff=float(p.get('wheelbase', car['wheelbase'])),
            steer_lock=float(p.get('steer_lock', car['steer_lock'])),
            v_max=float(p.get('v_max', car['v_max'])),
            v_min=float(p.get('v_min_corner', car['v_min'])),
            source='torcs_mpc.PARAMS',
        )
    except Exception:
        pass

    if base:
        car.update({k: v for k, v in base.items() if v is not None})

    if dynamics is None and model_path:
        try:
            with open(model_path, 'r') as f:
                dynamics = json.load(f)
        except Exception:
            dynamics = None

    if dynamics:
        car.update(
            a_lat=_clip(float(dynamics.get('a_lat', car['a_lat'])), 6.0, 30.0),
            a_brake=_clip(float(dynamics.get('a_brake', car['a_brake'])), 5.0, 25.0),
            accel_a0=_clip(float(dynamics.get('accel_a0', car['a_accel'])), 1.0, 20.0),
            accel_a1=_clip(float(dynamics.get('accel_a1', car['accel_a1'])), 0.0, 0.3),
            wheelbase_eff=_clip(float(dynamics.get('wheelbase_eff',
                                                   car['wheelbase_eff'])), 1.0, 5.0),
            understeer_K=_clip(float(dynamics.get('understeer_K',
                                                  car['understeer_K'])), 0.0, 0.05),
            source=model_path or 'provided dynamics',
        )
        car['a_accel'] = car['accel_a0']

    car['a_lat'] = _clip(float(car['a_lat']), 6.0, 30.0)
    car['a_brake'] = _clip(float(car['a_brake']), 5.0, 25.0)
    car['a_accel'] = _clip(float(car['a_accel']), 1.0, 20.0)
    car['v_max'] = _clip(float(car['v_max']), 15.0, 120.0)
    car['v_min'] = _clip(float(car['v_min']), 3.0, 30.0)
    return car


# ---------------------------------------------------------------------------
#  Periodic operators
# ---------------------------------------------------------------------------
def _periodic_second_difference(n, ds):
    """Sparse second-difference operator d^2/ds^2 on a periodic uniform grid."""
    I = sp.identity(n, format='csr')
    fwd = sp.csr_matrix((np.ones(n),
                         (np.arange(n), (np.arange(n) + 1) % n)), shape=(n, n))
    bwd = sp.csr_matrix((np.ones(n),
                         (np.arange(n), (np.arange(n) - 1) % n)), shape=(n, n))
    return (fwd - 2 * I + bwd) / (ds * ds)


def _d2_array(x, ds):
    return (np.roll(x, -1) - 2.0 * x + np.roll(x, 1)) / (ds * ds)


# ---------------------------------------------------------------------------
#  Optimisation: global racing line + speed profile
# ---------------------------------------------------------------------------
def _solve_weighted_racing_line(s, kappa, width, margin, weight=None, ridge=1e-6):
    """Periodic minimum-curvature racing line.

    The curvature of a path offset by n(s) from a reference of curvature
    kappa(s) is, to first order, kappa + n''. Minimising the path curvature is
    therefore  min_n || kappa + n'' ||^2  subject to the car staying within the
    track:  -(w/2 - margin) <= n <= (w/2 - margin).  A *tiny* ridge term only
    resolves the constant null-space of the second-difference operator (a
    constant offset does not change curvature); it must stay far smaller than
    the curvature cost or it would pin the line to the centre instead of letting
    it use the full track width.

    Returns (n, kappa_rl) where kappa_rl = kappa + n'' is the racing-line
    curvature used by the speed profile.
    """
    N = s.shape[0]
    ds = float(s[1] - s[0])
    half = np.maximum(width * 0.5 - margin, 0.0)
    lo, hi = -half, half
    # Where the corridor is narrower than 2*margin, pin to the centre.
    bad = lo >= hi
    lo = np.where(bad, -1e-3, lo)
    hi = np.where(bad, 1e-3, hi)

    if weight is None:
        weight = np.ones(N)
    weight = np.maximum(np.asarray(weight, dtype=float), 1e-4)

    if _HAVE_SCIPY:
        D2 = _periodic_second_difference(N, ds)
        W = sp.diags(np.sqrt(weight), format='csr')
        # minimise || W * (D2 n - (-kappa)) ||^2 + ridge || n ||^2
        A = sp.vstack([W @ D2, math.sqrt(ridge) * sp.identity(N)]).tocsr()
        b = np.concatenate([-(np.sqrt(weight) * kappa), np.zeros(N)])
        res = lsq_linear(A, b, bounds=(lo, hi), max_iter=200,
                         tol=1e-4, lsmr_tol='auto')
        return res.x, kappa + (D2 @ res.x)

    # NumPy fallback: projected FISTA on the same convex quadratic. The D2
    # operator is symmetric on a periodic grid, so grad = D2(W^2(D2n+k)) + ridge*n.
    x = np.clip(np.zeros(N), lo, hi)
    y = x.copy()
    t = 1.0
    L = float(np.max(weight)) * 16.0 / max(ds ** 4, 1e-9) + ridge
    step = 1.0 / max(L, 1e-9)
    for _ in range(900):
        residual = _d2_array(y, ds) + kappa
        grad = _d2_array(weight * residual, ds) + ridge * y
        x_new = np.clip(y - step * grad, lo, hi)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        if float(np.max(np.abs(x_new - x))) < 1e-5:
            x = x_new
            break
        x, t = x_new, t_new
    return x, kappa + _d2_array(x, ds)


def build_racing_line_global(s, kappa, width, a_lat=None, margin=1.5, ridge=1e-6):
    """Periodic minimum-curvature racing line.

    The curvature of a path offset by n(s) from a reference of curvature
    kappa(s) is, to first order, kappa + n''. Minimising the path curvature is
    therefore  min_n || kappa + n'' ||^2  subject to the car staying within the
    track:  -(w/2 - margin) <= n <= (w/2 - margin).  A *tiny* ridge term only
    resolves the constant null-space of the second-difference operator (a
    constant offset does not change curvature); it must stay far smaller than
    the curvature cost or it would pin the line to the centre instead of letting
    it use the full track width.

    Returns (n, kappa_rl) where kappa_rl = kappa + n'' is the first-order
    curvature estimate. The TrackMap builder later recomputes exact curvature.
    """
    n, kappa_rl = _solve_weighted_racing_line(s, kappa, width, margin,
                                              weight=None, ridge=ridge)
    return n, kappa_rl


def racing_line_metrics(s, kappa, n):
    """Exact offset-curve curvature and local racing-line segment lengths.

    For r(s)=c(s)+n(s)N(s), with centreline curvature kappa:
      r'  = (1-kappa*n)T + n'N
      r'' = (-kappa'*n - 2*kappa*n')T + (kappa*(1-kappa*n)+n'')N
    so curvature is cross(r', r'') / |r'|^3. This avoids integrating a noisy
    centreline around the loop and handles variable offset n(s), not just a
    constant parallel curve.
    """
    ds = float(s[1] - s[0])
    n1 = (np.roll(n, -1) - np.roll(n, 1)) / (2.0 * ds)
    n2 = (np.roll(n, -1) - 2.0 * n + np.roll(n, 1)) / (ds * ds)
    k1 = (np.roll(kappa, -1) - np.roll(kappa, 1)) / (2.0 * ds)
    a = 1.0 - kappa * n
    b = n1
    c = -k1 * n - 2.0 * kappa * n1
    d = kappa * a + n2
    speed_scale = np.sqrt(np.maximum(a * a + b * b, 1e-6))
    kappa_rl = (a * d - b * c) / (speed_scale ** 3 + 1e-9)
    ds_rl = np.clip(speed_scale * ds, 0.25 * ds, 3.0 * ds)
    return kappa_rl, ds_rl


def racing_line_curvature(s, kappa, n):
    return racing_line_metrics(s, kappa, n)[0]


def _engine_accel(v, accel_a0, accel_a1, a_accel):
    if accel_a0 is None:
        return np.full_like(np.asarray(v, dtype=float), float(a_accel))
    return np.maximum(float(accel_a0) - float(accel_a1) * np.asarray(v, dtype=float),
                      0.5)


def _ellipse_factor(lat_use, strength):
    if strength <= 0.0:
        return 1.0
    lat_use = np.clip(lat_use, 0.0, 0.98)
    pure = np.sqrt(np.maximum(1.0 - lat_use * lat_use, 0.08))
    return (1.0 - strength) + strength * pure


def speed_profile_global(s, kappa_rl, a_lat, a_brake, a_accel,
                         v_max, v_min, n_passes=6, ds_step=None,
                         accel_a0=None, accel_a1=0.0,
                         wheelbase_eff=None, understeer_K=0.0,
                         steer_lock=None, combined_grip=0.0):
    """Min-time speed profile for a closed loop.

    Grip-limited corner speed v = sqrt(a_lat / |kappa|), then alternating
    backward (braking) and forward (traction) passes around the loop until the
    longitudinal acceleration limits are satisfied everywhere. Periodic, so the
    end of the lap constrains the start.
    """
    N = s.shape[0]
    ds = float(s[1] - s[0])
    if ds_step is None:
        ds_step = np.full(N, ds)
    else:
        ds_step = np.asarray(ds_step, dtype=float)
    k = np.abs(kappa_rl) + 1e-5
    v = np.minimum(np.sqrt(a_lat / k), v_max)
    if steer_lock is not None and wheelbase_eff is not None:
        steer_lock = max(float(steer_lock), 1e-6)
        wheelbase_eff = max(float(wheelbase_eff), 1e-6)
        understeer_K = max(float(understeer_K), 0.0)
        v_steer = np.full(N, v_max)
        if understeer_K > 1e-9:
            feasible = (steer_lock / k) - wheelbase_eff
            v_steer = np.sqrt(np.maximum(feasible / understeer_K, 0.0))
        else:
            feasible = k <= steer_lock / wheelbase_eff
            v_steer = np.where(feasible, v_max, v_min)
        v = np.minimum(v, v_steer)
    v = np.clip(v, v_min, v_max)

    for _ in range(n_passes):
        # backward pass: ensure we can brake down to each corner
        for i in range(N - 1, -N - 1, -1):
            j = i % N
            nxt = (i + 1) % N
            lat_use = (v[nxt] * v[nxt] * k[nxt]) / max(a_lat, 1e-6)
            a_eff = a_brake * _ellipse_factor(lat_use, combined_grip)
            v[j] = min(v[j], math.sqrt(v[nxt] ** 2 + 2.0 * a_eff * ds_step[j]))
        # forward pass: ensure we do not exceed what we can accelerate to
        for i in range(0, 2 * N):
            j = i % N
            prv = (i - 1) % N
            lat_use = (v[prv] * v[prv] * k[prv]) / max(a_lat, 1e-6)
            a_drive = float(_engine_accel(np.array([v[prv]]), accel_a0,
                                          accel_a1, a_accel)[0])
            a_eff = a_drive * _ellipse_factor(lat_use, combined_grip)
            v[j] = min(v[j], math.sqrt(v[prv] ** 2 + 2.0 * a_eff * ds_step[prv]))

    return np.clip(v, v_min, v_max)


def speed_profile_for_car(s, kappa_rl, ds_rl, car):
    return speed_profile_global(
        s, kappa_rl,
        car['a_lat'], car['a_brake'], car['a_accel'],
        car['v_max'], car['v_min'],
        ds_step=ds_rl,
        accel_a0=car.get('accel_a0'),
        accel_a1=car.get('accel_a1', 0.0),
        wheelbase_eff=car.get('wheelbase_eff'),
        understeer_K=car.get('understeer_K', 0.0),
        steer_lock=car.get('steer_lock'),
        combined_grip=car.get('combined_grip', 0.0))


def build_time_optimal_racing_line(s, kappa, width, car, margin=2.5,
                                   iterations=6, ridge=1e-6,
                                   time_weight=6.0):
    """Iteratively reweighted global line optimized for this car's lap time.

    Each iteration solves a periodic curvature minimization, computes the exact
    racing-line curvature/length and the learned-car speed profile, then raises
    the curvature weights where the car is actually speed-limited. This keeps
    the robust minimum-curvature backbone, but asks the line to spend track
    width where it buys the most time for this particular car.
    """
    ds = float(s[1] - s[0])
    weight = np.ones_like(s, dtype=float)
    best = None
    iterations = max(1, int(iterations))

    for _ in range(iterations):
        n, _ = _solve_weighted_racing_line(s, kappa, width, margin,
                                           weight=weight, ridge=ridge)
        kappa_rl, ds_rl = racing_line_metrics(s, kappa, n)
        v = speed_profile_for_car(s, kappa_rl, ds_rl, car)
        lap_time = float(np.sum(ds_rl / np.maximum(v, 0.5)))
        if best is None or lap_time < best['lap_time']:
            best = dict(n=n, kappa_rl=kappa_rl, ds_rl=ds_rl,
                        v=v, lap_time=lap_time,
                        line_length=float(np.sum(ds_rl)))

        lat_load = (v * v * np.abs(kappa_rl)) / max(car['a_lat'], 1e-6)
        steer_load = np.zeros_like(lat_load)
        if car.get('steer_lock') and car.get('wheelbase_eff'):
            steer_need = np.abs(kappa_rl) * (
                car['wheelbase_eff'] + car.get('understeer_K', 0.0) * v * v)
            steer_load = steer_need / max(car['steer_lock'], 1e-6)
        load = np.maximum(lat_load, steer_load)
        slow = np.clip((car['v_max'] / np.maximum(v, car['v_min']) - 1.0) / 2.5,
                       0.0, 1.0)
        new_weight = 1.0 + time_weight * np.clip(load, 0.0, 1.4) * slow
        weight = _periodic_smooth(new_weight, smooth_m=18.0, ds=ds)

    return best['n'], best['kappa_rl'], best['ds_rl'], best['v'], best


# ---------------------------------------------------------------------------
#  TrackMap
# ---------------------------------------------------------------------------
class TrackMap:
    def __init__(self, s, kappa, width, length):
        self.s = np.asarray(s, dtype=float)
        self.kappa = np.asarray(kappa, dtype=float)
        self.width = np.asarray(width, dtype=float)
        self.length = float(length)
        self.n_rl = None         # racing-line offset (+ = left of centre)
        self.kappa_rl = None     # racing-line curvature
        self.v_target = None     # min-time speed profile
        self.ds_rl = None        # local racing-line segment lengths
        self.line_length = None
        self.lap_time = None
        self.meta = {}

    def _apply_speed_zones(self, zones):
        """Cap the target speed to a hard limit in specific corners, given by
        their distFromStart - for corners the map gets wrong (e.g. corkscrew's
        downhill, which is slow because of elevation, not curvature, so it is
        not the tightest corner the auto-detector would find). Each zone is
        (s_center_m, max_speed_mps, window_m); the cap tapers smoothly to no
        change at the window edge, so every other corner is untouched.

        Read s_center off the telemetry 'd' column where the car runs wide."""
        if not zones or self.v_target is None:
            return
        s = self.s
        ds = float(s[1] - s[0])
        idx = np.arange(self.v_target.shape[0])
        for z in zones:
            s_c, vmax, window = float(z[0]), float(z[1]), float(z[2])
            ic = int(round((s_c % self.length) / ds)) % self.v_target.shape[0]
            sig = max(window / ds / 2.0, 1.0)
            d = np.abs(idx - ic)
            d = np.minimum(d, self.v_target.shape[0] - d)
            win = np.exp(-0.5 * (d / sig) ** 2)         # 1 at corner, ->0 away
            capped = np.minimum(self.v_target, vmax)
            self.v_target = self.v_target * (1.0 - win) + capped * win

    # ---- build the racing line + speed profile ----
    def build(self, a_lat=None, a_brake=None, a_accel=None,
              v_max=None, v_min=None, margin=2.5,
              dynamics=None, model_path=None, params=None, iterations=6,
              speed_zones=None):
        base = dict(a_lat=a_lat, a_brake=a_brake, a_accel=a_accel,
                    accel_a0=a_accel, v_max=v_max, v_min=v_min)
        car = load_car_limits(model_path=model_path, dynamics=dynamics,
                              params=params, base=base)
        self.n_rl, self.kappa_rl, self.ds_rl, self.v_target, info = \
            build_time_optimal_racing_line(
                self.s, self.kappa, self.width, car, margin=margin,
                iterations=iterations)
        # ONLY the explicit per-corner speed caps are applied (e.g. corkscrew).
        # No automatic slowing of the tightest-curvature corner - every corner
        # except the ones you list in speed_zones is left at full min-time speed.
        self._apply_speed_zones(speed_zones)
        self.line_length = info['line_length']
        self.lap_time = info['lap_time']
        self.meta = dict(
            optimizer='iterative weighted min-time',
            iterations=int(iterations),
            margin=float(margin),
            car={k: v for k, v in car.items()
                 if k not in ('source',) and isinstance(v, (int, float, str))},
            car_source=car.get('source', 'unknown'),
            lap_time=self.lap_time,
            line_length=self.line_length,
        )
        return self

    # ---- periodic lookups (live) ----
    def offset_at(self, s_query):
        return np.interp(np.mod(s_query, self.length), self.s, self.n_rl,
                         period=self.length)

    def speed_at(self, s_query):
        # Works for scalar or array s_query (np.interp returns the same shape).
        return np.interp(np.mod(s_query, self.length), self.s,
                         self.v_target, period=self.length)

    def lookahead_speed(self, s0, a_brake, horizon=200.0, n=32, start=0.0):
        """Speed to hold *now* so we can still brake to every speed the map
        prescribes between `start` and `horizon` metres ahead: min over s' of
        sqrt(v*(s')^2 + 2*a_brake*(s'-s0)).

        `start` > 0 ignores the map right around the car, leaving the near range
        to the (accurate, directly-sensed) local grip limit. This stops a map
        whose curvature is over-estimated at a tight corner from craters-ing the
        speed *under the car* and overriding what the beams can plainly see; the
        map still provides the smooth, full-distance braking target further on."""
        ds = np.linspace(start, horizon, n)
        vfut = self.speed_at(s0 + ds)
        return float(np.min(np.sqrt(vfut * vfut + 2.0 * a_brake * ds)))

    def width_at(self, s_query):
        return float(np.interp(np.mod(s_query, self.length), self.s,
                               self.width, period=self.length))

    # ---- persistence ----
    def save(self, path):
        d = dict(length=self.length,
                 s=self.s.tolist(), kappa=self.kappa.tolist(),
                 width=self.width.tolist(),
                 n_rl=None if self.n_rl is None else self.n_rl.tolist(),
                 kappa_rl=None if self.kappa_rl is None else self.kappa_rl.tolist(),
                 v_target=None if self.v_target is None else self.v_target.tolist(),
                 ds_rl=None if self.ds_rl is None else self.ds_rl.tolist(),
                 line_length=self.line_length,
                 lap_time=self.lap_time,
                 meta=self.meta)
        with open(path, 'w') as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            d = json.load(f)
        m = cls(d['s'], d['kappa'], d['width'], d['length'])
        if d.get('n_rl') is not None:
            m.n_rl = np.asarray(d['n_rl'], dtype=float)
            m.kappa_rl = np.asarray(d['kappa_rl'], dtype=float)
            m.v_target = np.asarray(d['v_target'], dtype=float)
            if d.get('ds_rl') is not None:
                m.ds_rl = np.asarray(d['ds_rl'], dtype=float)
            m.line_length = d.get('line_length')
            m.lap_time = d.get('lap_time')
            m.meta = d.get('meta', {})
        return m


# ---------------------------------------------------------------------------
#  Builders
# ---------------------------------------------------------------------------
def _resample_periodic(s_raw, val_raw, length, ds):
    """Bin/interpolate scattered (s, value) samples onto a uniform periodic
    grid and lightly smooth (periodic moving average)."""
    grid = np.arange(0.0, length, ds)
    # sort by s and make periodic by wrapping a copy on each side
    order = np.argsort(s_raw)
    s_sorted = s_raw[order]
    v_sorted = val_raw[order]
    s_ext = np.concatenate([s_sorted - length, s_sorted, s_sorted + length])
    v_ext = np.concatenate([v_sorted, v_sorted, v_sorted])
    val = np.interp(grid, s_ext, v_ext)
    # periodic smoothing
    w = max(3, int(round(7.0 / ds)) | 1)
    k = np.ones(w) / w
    val = np.convolve(np.concatenate([val[-w:], val, val[:w]]), k, mode='same')[w:-w]
    return grid, val


def from_centerline(X, Y, width, ds=3.0):
    """Build a TrackMap from a known closed centreline polyline (offline/sim).

    width may be a scalar or a per-point array. Curvature is derived from the
    turning of the centreline heading."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    dx = np.diff(X)
    dy = np.diff(Y)
    seg = np.hypot(dx, dy)
    s_pts = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(s_pts[-1])
    th = np.unwrap(np.arctan2(dy, dx))
    th = np.concatenate([th, [th[-1]]])
    # curvature = d(theta)/ds (centred)
    dth = np.gradient(th, s_pts)
    if np.isscalar(width):
        width_arr = np.full(X.shape[0], float(width))
    else:
        width_arr = np.asarray(width, dtype=float)

    grid = np.arange(0.0, length, ds)
    kappa = np.interp(grid, s_pts, dth)
    width_g = np.interp(grid, s_pts, width_arr)
    # light periodic smoothing of curvature
    w = max(3, int(round(9.0 / ds)) | 1)
    k = np.ones(w) / w
    kappa = np.convolve(np.concatenate([kappa[-w:], kappa, kappa[:w]]),
                        k, mode='same')[w:-w]
    return TrackMap(grid, kappa, width_g, length)


def _robust_bin(s_raw, val_raw, length, ds, bin_m=6.0, smooth_m=11.0):
    """Robustly turn scattered (s, value) samples into a smooth periodic profile
    on the output grid (spacing ds).

    The median is taken over *coarse* measurement bins (bin_m wide) so each bin
    has enough samples to reject transient outliers and start/finish-line
    glitches; the coarse medians are then periodically interpolated to the fine
    output grid and lightly smoothed. (Median bins as fine as ds would hold only
    a handful of samples and let outliers through.)"""
    grid = np.arange(0.0, length, ds)
    nb = max(6, int(round(length / bin_m)))
    bi = np.clip((np.mod(s_raw, length) / length * nb).astype(int), 0, nb - 1)
    centers = (np.arange(nb) + 0.5) * (length / nb)
    med = np.full(nb, np.nan)
    for c in range(nb):
        vals = val_raw[bi == c]
        if vals.size:
            med[c] = np.median(vals)
    good = ~np.isnan(med)
    if good.sum() < 3:
        return grid, np.zeros(grid.shape[0])
    # periodic median-of-3 over the coarse bins: removes isolated spikes (e.g.
    # the start/finish-line glitch) without eroding sustained corners.
    filled = med.copy()
    filled[~good] = np.interp(np.where(~good)[0], np.where(good)[0], med[good],
                              period=nb)
    med = np.median(np.stack([np.roll(filled, 1), filled, np.roll(filled, -1)]),
                    axis=0)
    cg, vg = centers, med
    # periodic interpolation to the fine grid
    out = np.interp(grid,
                    np.concatenate([cg - length, cg, cg + length]),
                    np.concatenate([vg, vg, vg]))
    w = max(3, int(round(smooth_m / ds)) | 1)
    k = np.ones(w) / w
    out = np.convolve(np.concatenate([out[-w:], out, out[:w]]), k, mode='same')[w:-w]
    return grid, out


def from_map_log(path, ds=3.0, wheelbase=2.4, steer_lock=0.366,
                 dynamics=None, model_path=None, params=None, car=None):
    """Build a TrackMap from a warm-up-lap log written by torcs_mpc.py.

    Track curvature is reconstructed from the car's motion rather than from
    instantaneous beam geometry. If learned dynamics are supplied, yaw rate uses
    the same understeer model as torcs_mpc.py; otherwise it falls back to the
    nominal bicycle model. A loop-closure rescale forces the lap's total turning
    to the nearest multiple of 2*pi, cancelling systematic yaw-estimate bias.

    Expected columns: t, dfs, v, steer, angle, width, ontrack.
    """
    if car is None:
        car = load_car_limits(model_path=model_path, dynamics=dynamics,
                              params=params,
                              base=dict(wheelbase=wheelbase,
                                        wheelbase_eff=wheelbase,
                                        steer_lock=steer_lock))
    data = np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True))
    t = data['t']; dfs = data['dfs']; v = data['v']
    steer = data['steer']; angle = data['angle']; width = data['width']
    on = data['ontrack'] > 0.5

    length = float(np.max(dfs)) + ds
    dt = np.diff(t)
    dds = np.diff(dfs)
    dang = (np.diff(angle) + np.pi) % (2 * np.pi) - np.pi
    delta = steer[:-1] * car.get('steer_lock', steer_lock)
    if car.get('understeer_K', 0.0) > 0.0 or \
            abs(car.get('wheelbase_eff', wheelbase) - wheelbase) > 1e-6:
        yaw_rate = (v[:-1] * delta) / (
            car.get('wheelbase_eff', wheelbase) +
            car.get('understeer_K', 0.0) * v[:-1] * v[:-1])
    else:
        yaw_rate = (v[:-1] / wheelbase) * np.tan(delta)
    dtheta = yaw_rate * dt + dang
    # valid steps: continuous in time, moving forward along the track (drop the
    # start/finish wrap where dfs resets), on track, and not in the small zone
    # right at the start/finish line (heading/projection glitch there).
    s_all = dfs[:-1]
    edge = 8.0
    ok = (dt > 1e-3) & (dt < 0.1) & (dds > 0.05) & (dds < 15.0) & \
         on[:-1] & on[1:] & (s_all > edge) & (s_all < length - edge)
    s_mid = dfs[:-1][ok]
    kappa_step = dtheta[ok] / dds[ok]
    w_step = 0.5 * (width[:-1] + width[1:])[ok]

    # Loop closure / yaw-bias correction. A simple closed circuit turns exactly
    # +-2*pi per lap, so the measured total turning *must* equal sign*laps*2*pi.
    # The kinematic yaw-from-steer is systematically too large when the car is
    # at the grip limit (commanded steer exceeds the yaw the tyres deliver), so
    # we rescale the whole curvature profile to hit the known total. The lap
    # count comes from distance travelled, not from the (biased) turning sum.
    total = float(np.sum(dtheta[ok]))
    fwd_dist = float(np.sum(dds[ok]))
    laps = max(1, int(round(fwd_dist / length)))
    sign = 1.0 if total >= 0 else -1.0
    total_true = sign * laps * 2 * np.pi
    if abs(total) > 1e-3:
        kappa_step = kappa_step * (total_true / total)

    grid, kappa_g = _robust_bin(s_mid, kappa_step, length, ds, smooth_m=11.0)
    _, width_g = _robust_bin(s_mid, w_step, length, ds, smooth_m=7.0)
    width_g = np.clip(width_g, 4.0, 30.0)
    return TrackMap(grid, kappa_g, width_g, length)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Build a global, car-aware racing line from a mapping lap.')
    parser.add_argument('maplog', help='CSV written by torcs_mpc.py --map-log')
    parser.add_argument('out', nargs='?', default='track_map.json')
    parser.add_argument('--model', default=None,
                        help='learned dynamics JSON; defaults to dynamics_model.json if present')
    parser.add_argument('--no-model', action='store_true',
                        help='ignore dynamics_model.json and use torcs_mpc defaults')
    parser.add_argument('--ds', type=float, default=3.0,
                        help='map grid spacing in metres')
    parser.add_argument('--margin', type=float, default=2.5,
                        help='metres to keep inside each track edge')
    parser.add_argument('--iters', type=int, default=6,
                        help='time-weighted racing-line optimization iterations')
    parser.add_argument('--slow', action='append', default=[],
                        metavar='dist,maxspeed,window',
                        help='cap the speed in one corner: distFromStart(m),'
                             ' max_speed(m/s), window(m). Repeatable. e.g. '
                             '--slow 2500,10,60 for the corkscrew.')
    args = parser.parse_args(argv)

    speed_zones = []
    for z in args.slow:
        try:
            s_c, vmax, win = (float(x) for x in z.split(','))
            speed_zones.append((s_c, vmax, win))
        except Exception:
            print("ignoring bad --slow '%s' (want dist,maxspeed,window)" % z)

    model_path = None
    if not args.no_model:
        model_path = args.model or _default_model_path()
    car = load_car_limits(model_path=model_path)

    m = from_map_log(args.maplog, ds=args.ds, car=car).build(
        margin=args.margin, model_path=model_path, iterations=args.iters,
        speed_zones=speed_zones)
    m.save(args.out)
    for s_c, vmax, win in speed_zones:
        print("  speed zone: <= %.1f m/s at d=%.0f m (+-%.0f m)" % (vmax, s_c, win))
    print("Track length %.0f m, %d nodes" % (m.length, m.s.shape[0]))
    print("car model: %s" % car.get('source', 'unknown'))
    print("limits: a_lat=%.2f a_brake=%.2f accel=%.2f-%.4f*v understeer_K=%.4f"
          % (car['a_lat'], car['a_brake'], car.get('accel_a0') or car['a_accel'],
             car.get('accel_a1', 0.0), car.get('understeer_K', 0.0)))
    print("racing-line offset range: %.2f .. %.2f m"
          % (m.n_rl.min(), m.n_rl.max()))
    print("line length %.1f m, estimated lap %.2f s"
          % (m.line_length or m.length, m.lap_time or 0.0))
    print("target speed range: %.1f .. %.1f m/s"
          % (m.v_target.min(), m.v_target.max()))
    print("saved -> %s" % args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
