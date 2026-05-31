#!/usr/bin/env python
# torcs_mpc.py
#
# A from-scratch SCR/TORCS control client built around:
#   1. Corridor reconstruction from the 19 rangefinder beams.
#   2. A cubic-spline (G2-continuous, locally clothoid-like) reference path
#      fitted through the reconstructed track centreline.
#   3. A speed profile derived from the path curvature + look-ahead braking.
#   4. A sampling-based Model Predictive Controller (MPPI flavour) that rolls
#      out a kinematic bicycle model over a receding horizon and picks the
#      steering that best tracks the path while respecting the track edges.
#
# The control logic (class MPCDriver) is fully decoupled from the UDP
# networking so it can be exercised by the offline simulator in sim_test.py.
#
# Networking boilerplate (Client / ServerState / DriverAction) is adapted from
# the snakeoil client shipped with gym_torcs (torcs_jm_par.py).

import socket
import sys
import getopt
import os
import time
import math
import numpy as np

try:
    from scipy.interpolate import CubicSpline
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy expected to be present
    _HAVE_SCIPY = False

PI = 3.14159265359
data_size = 2 ** 17


def clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


# ===========================================================================
#  CONTROL CORE  (no networking - testable in isolation)
# ===========================================================================

# Rangefinder beam angles requested in the init string, in radians.
TRACK_ANGLES_DEG = np.array(
    [-45, -19, -12, -7, -4, -2.5, -1.7, -1, -.5, 0, .5, 1, 1.7, 2.5, 4, 7, 12, 19, 45],
    dtype=float)
TRACK_ANGLES = np.radians(TRACK_ANGLES_DEG)
BEAM_COS = np.cos(TRACK_ANGLES)
BEAM_SIN = np.sin(TRACK_ANGLES)

# Tunable parameters. Kept in one dict so the simulator can sweep them.
PARAMS = dict(
    # --- vehicle / actuation ---
    wheelbase=2.4,          # m, kinematic bicycle length
    steer_lock=0.366,       # rad, steer command of 1.0 == this front-wheel angle
    # --- grip / speed profile ---
    a_lat=16.5,             # m/s^2, max usable lateral acceleration (cornering)
    a_brake=16.0,           # m/s^2, planning deceleration for look-ahead braking
    a_accel=12.0,            # m/s^2, planning acceleration
    v_max=200.0,             # m/s hard cap on target speed
    v_min_corner=1.0,       # m/s, never plan slower than this
    vis_margin=5.0,         # m kept in hand beyond the furthest thing we can see
    # --- path reconstruction ---
    n_center=18,            # number of centreline samples to fit the spline
    max_lookahead=120.0,    # m, clamp reconstructed depth
    edge_margin=1.4,        # m, keep this far inside each reconstructed edge
    reg_center=0.012,       # very weak pull toward the middle (straight-line tie-break only)
    w_global_bias=0.06,     # pull of the local line toward the global racing-line offset
    w_line_heading=0.6,     # how hard the racing line leaves along the car heading
    map_horizon=70.0,       # m of track-map speed looked at just beyond sight
    map_brake_horizon=200.0,# m of map look-ahead for the speed target (braking distance)
    map_speed=22.0,         # m/s cap while driving a mapping/reconnaissance lap
    grip_lookahead=35.0,    # m: map look-ahead starts beyond this (avoids near-corner crawl)
    grip_range=80.0,        # m: local grip safety range in map mode (catches map errors/blind corners)
    map_speed_safety=1.0,  # margin on map target speed (covers beam-map curvature error)
    fwd_clear_smooth=0.15,  # low-pass on furthest-beam range (anti speed-wobble on straights)
    tv_smooth=0.25,         # temporal low-pass on target speed (kills frame-to-frame wobble)
    tv_snap=5.0,            # m/s: drop bigger than this bypasses the smoothing (real braking)
    # --- MPC ---
    horizon=16,             # steps
    dt_mpc=0.06,            # s per MPC step (~1.0 s horizon)
    n_samples=120,          # control sequences sampled per step
    steer_sigma=0.03,       # rad, exploration noise on front-wheel angle
    mppi_lambda=1.5,        # temperature for the soft-min weighting
    w_cross=0.5,            # cross-track error weight (lower = less twitchy chasing of line wobble)
    w_heading=1.8,          # heading error weight (lean on heading -> smoother straights)
    w_offtrack=120.0,       # penalty for leaving the reconstructed corridor
    w_progress=1.1,         # reward for longitudinal progress (least time)
    w_steer=1.0,            # penalty on steering magnitude (prefer straight; kills left-right wander)
    w_steer_rate=24.0,      # penalty on steering rate (smoothness)
    steer_smooth=0.3,       # output low-pass: fraction of new command per step (lower = smoother)
    # --- longitudinal controller (maps target speed -> pedals) ---
    # Racing style: flat-out whenever we are at/below the speed the track
    # allows, brake only once genuinely over it (no part-throttle cruising,
    # no dragging the brakes).
    # Decisive pedals: flat to the floor below target, a small coast band so it
    # does not chatter between throttle and brake at the target, then hard on
    # the brakes above it.
    coast_band=1.5,         # m/s over target: coast (no throttle, no brake) - kills oscillation
    kp_brake=0.6,           # brake firmness once past the coast band (brakes to the floor)
    brake_max=1.0,          # max brake command (ABS modulates below this)
    # ABS: full brake up to abs_slip, then bleed off; only deep lock cuts hard.
    abs_slip=0.12,          # wheel slip ratio allowed at full brake (~optimal grip)
    abs_gain=3.0,           # how fast brake is released as slip exceeds abs_slip
    abs_min=0.5,            # brake is never cut below this fraction
    # Friction circle: tyres share grip between braking and cornering.
    brake_plan_frac=0.8,   # plan braking at this fraction of max -> brake earlier,
                            #   reach corner speed before turn-in (leaves lateral grip)
    fc_min=0.6,             # min brake factor while cornering hard (bleed for steering)
    # --- anti-spin (power oversteer) ---
    # Conservative: a RWD car with downforce on a downhill (corkscrew) snaps the
    # rear out easily, so throttle is held well back while steering and the
    # reactive traction control bites hard and early.
    steer_throttle_cut=0.75,  # fraction of throttle removed at full steering lock
    min_throttle_factor=0.3,  # never cut throttle below this fraction while turning
    accel_cap_g1=0.55,        # throttle ceiling in 1st gear (most wheelspin)
    accel_cap_g2=0.8,         # throttle ceiling in 2nd gear
    accel_cap_g3=0.92,        # throttle ceiling in 3rd gear
    tc_slip=2.0,              # rad/s rear-front wheel-speed diff that triggers TC
    tc_cut=0.6,               # throttle removed when TC triggers
)


def reconstruct_corridor(track, p):
    """Turn the 19 beam ranges into a drivable corridor in the car frame.

    Returns (xk, yc, yl, yr) sampled at increasing depth x, where yc is the
    centreline lateral offset and yl/yr are the left/right edges, or None only
    when the car is genuinely off-track (almost no valid beams).

    Geometry: x points forward along the car heading, y points left. A beam at
    angle a returning distance d hits the edge at (d*cos a, d*sin a).
    """
    d = np.asarray(track, dtype=float)
    if d.shape[0] != 19:
        return None
    valid = d > 0.0
    if valid.sum() < 4:
        return None  # genuinely off track -> recovery mode

    d = np.clip(d, 0.05, p['max_lookahead'])
    x = d * BEAM_COS
    y = d * BEAM_SIN

    return _centerline(x, y, d, p)


def _est_halfwidth(x, y):
    # Lateral extent of the boundary fan, ignoring the far forward points.
    near = x < 40.0
    if near.sum() >= 2:
        yy = y[near]
    else:
        yy = y
    hw = 0.5 * (float(np.max(yy)) - float(np.min(yy)))
    return clip(hw, 2.5, 8.0)


def _centerline(x, y, d, p):
    """Build the centreline from the boundary fan, split at the deepest beam
    into a right wall (beams up to the peak) and a left wall (peak onward).

    On straights/gentle curves both walls are well sampled and we take their
    midpoint. On sharp turns nearly all beams hit the inner wall and the outer
    wall is sparse; we then follow the dense (inner) wall offset inward by the
    estimated half-width, which hugs the apex. Returns None only if neither
    wall is usable (genuinely off track)."""
    ipeak = int(np.argmax(d))
    xr, yr_ = x[:ipeak + 1], y[:ipeak + 1]
    xl, yl_ = x[ipeak:], y[ipeak:]

    ar = np.argsort(xr)
    xr, yr_ = xr[ar], yr_[ar]
    al = np.argsort(xl)
    xl, yl_ = xl[al], yl_[al]

    hw = _est_halfwidth(x, y)
    right_ok = xr.shape[0] >= 2 and (xr[-1] - xr[0]) > 3.0
    left_ok = xl.shape[0] >= 2 and (xl[-1] - xl[0]) > 3.0

    if right_ok and left_ok:
        x_lo = max(xr[0], xl[0], 1.5)
        x_hi = min(xr[-1], xl[-1], p['max_lookahead'])
        if x_hi - x_lo > 4.0:
            xk = np.linspace(x_lo, x_hi, p['n_center'])
            yr = np.interp(xk, xr, yr_)
            yl = np.interp(xk, xl, yl_)
            return xk, 0.5 * (yl + yr), yl, yr

    # Sharp turn: follow whichever wall is densely sampled, offset inward.
    if right_ok and xr.shape[0] >= xl.shape[0]:
        xk = np.linspace(max(xr[0], 1.5), min(xr[-1], p['max_lookahead']),
                         p['n_center'])
        yr = np.interp(xk, xr, yr_)
        yc = yr + hw
        return xk, yc, yc + hw, yr
    if left_ok:
        xk = np.linspace(max(xl[0], 1.5), min(xl[-1], p['max_lookahead']),
                         p['n_center'])
        yl = np.interp(xk, xl, yl_)
        yc = yl - hw
        return xk, yc, yl, yc - hw
    return None


def curvature_of_spline(cs, xk):
    """Signed curvature kappa(x) of a y(x) spline: y'' / (1 + y'^2)^1.5."""
    d1 = cs(xk, 1)
    d2 = cs(xk, 2)
    return d2 / np.power(1.0 + d1 * d1, 1.5)


def _curvature_operator(xs):
    """Stacked rows of the discrete second derivative y''(x) at each interior
    node, for *non-uniform* spacing, scaled by sqrt(segment length) so that
    minimising ||A y||^2 approximates the curvature integral int y''(x)^2 ds.

    For three points x0<x1<x2 with h1=x1-x0, h2=x2-x1:
        y''(x1) ~ 2*( h2*y0 - (h1+h2)*y1 + h1*y2 ) / ( h1*h2*(h1+h2) )
    The uniform [1,-2,1]/h^2 stencil is the special case h1==h2, but using the
    proper form matters here because the first segment (car -> first beam hit)
    is much shorter than the rest, and a wrong stencil there bends the line.
    """
    n = xs.shape[0]
    A = np.zeros((n - 2, n))
    for k, i in enumerate(range(1, n - 1)):
        h1 = max(xs[i] - xs[i - 1], 1e-3)
        h2 = max(xs[i + 1] - xs[i], 1e-3)
        denom = h1 * h2 * (h1 + h2)
        w = math.sqrt(0.5 * (h1 + h2))            # ds weight for the integral
        A[k, i - 1] = w * 2.0 * h2 / denom
        A[k, i] = -w * 2.0 * (h1 + h2) / denom
        A[k, i + 1] = w * 2.0 * h1 / denom
    return A


def build_racing_line(xk, yl, yr, p, target=None):
    """Minimum-curvature racing line through the reconstructed corridor.

    If `target` is given (a desired lateral offset per xk node, e.g. projected
    from the global racing line), the line is pulled toward it instead of the
    corridor middle - so the local, beam-clamped path follows the globally
    optimal line while the curvature term keeps it smooth and the box bounds
    keep it on track.

    Solves the classic min-curvature racing-line QP:

        min   || y'' ||^2_ds                       (smallest possible curvature)
        s.t.  yr + margin <= y <= yl - margin       (stay on the track)
              y(0) = 0, y'(0) = 0                    (start at the car, tangent
                                                      to its current heading)

    The minimum-curvature solution is the racing line: it runs to the outside
    on entry, clips the inside at the apex and drifts back out on exit, because
    that is the straightest path the corridor permits. A *very* weak pull
    toward the corridor middle only breaks the tie on a dead-straight section
    (where every offset has equal, zero curvature) so the car does not pin
    itself to a wall. Returns (xs, y) or None on failure.
    """
    if not _HAVE_SCIPY:
        return None
    try:
        from scipy.optimize import lsq_linear
    except Exception:
        return None

    margin = p['edge_margin']
    xs = np.concatenate([[0.0], xk])              # node 0 = car position
    low = np.concatenate([[0.0], yr + margin])
    high = np.concatenate([[0.0], yl - margin])
    # If the corridor is narrower than the margins, collapse to its middle.
    bad = low > high
    mid = 0.5 * (low + high)
    low[bad] = mid[bad] - 1e-3
    high[bad] = mid[bad] + 1e-3
    low[0], high[0] = -1e-4, 1e-4                 # pin the start to the car
    mid = 0.5 * (low + high)

    n = xs.shape[0]
    blocks = []
    rhs = []

    # 1. curvature (the racing-line objective)
    blocks.append(_curvature_operator(xs))
    rhs.append(np.zeros(n - 2))

    # 2. heading: leave the car tangent to its current heading -> y'(0)=0
    h0 = max(xs[1] - xs[0], 1e-3)
    hr = np.zeros(n)
    hr[0], hr[1] = -1.0 / h0, 1.0 / h0
    blocks.append(p['w_line_heading'] * hr[None, :])
    rhs.append(np.zeros(1))

    # 3. lateral tie-break. Without a global line this is a *very* weak pull to
    #    the corridor middle (straights only). With one, it is a stronger pull
    #    toward the global racing-line offset, clamped into the visible corridor.
    if target is not None:
        tgt = np.concatenate([[0.0], np.asarray(target, dtype=float)])
        tgt = np.clip(tgt, low, high)
        reg = p['w_global_bias']
    else:
        tgt = mid
        reg = p['reg_center']
    C = np.zeros((n - 1, n))
    for k, i in enumerate(range(1, n)):
        C[k, i] = reg
    blocks.append(C)
    rhs.append(reg * tgt[1:])

    A = np.vstack(blocks)
    b = np.concatenate(rhs)

    try:
        res = lsq_linear(A, b, bounds=(low, high), max_iter=60,
                         tol=1e-3, method='bvls')
        return xs, res.x
    except Exception:
        return None


class MPCDriver:
    """Pure control core: feed it a sensor dict, get an action dict back."""

    def __init__(self, params=None, dynamics=None, track_map=None, mapping=False,
                 car=None):
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        # Physics-exact car profile (speed-dependent grip/brake/accel from the
        # TORCS car+track files; see car_profile.py). When present it replaces
        # the hand-guessed flat a_lat/a_brake/a_accel constants.
        self.car = car
        # Mapping mode: follow the centreline at a steady, moderate pace so the
        # warm-up lap yields a clean track-curvature measurement (no racing-line
        # apex-tucking, which would read as spurious corner-entry curvature).
        self.mapping = mapping
        self.prev_u = np.zeros(self.p['horizon'])  # warm-start steering plan
        self.stuck_timer = 0.0
        self.last_target_v = self.p['v_min_corner']
        self.prev_steer = 0.0  # for output low-pass smoothing
        self.fwd_clear_filt = None  # low-passed furthest-beam range (speed cap input)
        self.last_kappa = 0.0  # signed path curvature at the car (for logging)
        self.last_map_kappa = 0.0   # centreline curvature at car (for mapping)
        self.last_map_width = 0.0   # track width at car (for mapping)
        self.dyn = None        # learned vehicle-dynamics model (or None)
        self.track_map = track_map  # global racing line + speed profile (or None)
        if dynamics:
            self.apply_dynamics(dynamics)

    def apply_dynamics(self, dyn):
        """Install a learned dynamics model (see learn_dynamics.py / load_dynamics).

        The model overrides the hand-tuned grip/accel constants used by the
        speed profile and switches the MPC rollout from the plain kinematic
        bicycle to the learned understeer + drivetrain response, so the MPC
        predicts what the car *actually* does instead of an idealised model.
        Values are clamped to sane ranges so a bad fit cannot make the car
        undriveable.
        """
        self.dyn = dyn
        self.p['a_lat'] = clip(float(dyn.get('a_lat', self.p['a_lat'])), 6.0, 30.0)
        self.p['a_brake'] = clip(float(dyn.get('a_brake', self.p['a_brake'])),
                                 5.0, 25.0)
        a0 = float(dyn.get('accel_a0', self.p['a_accel']))
        self.p['a_accel'] = clip(a0, 1.0, 20.0)

    # ---- public API -----------------------------------------------------
    def control(self, S):
        """S: dict with keys angle, trackPos, speedX, speedY, track(list19),
        rpm, gear, wheelSpinVel(list4). Returns action dict."""
        v = float(S['speedX'])
        corridor = reconstruct_corridor(S['track'], self.p)

        # Stuck detection: genuinely pinned (almost no motion at all) for a
        # sustained time - not just briefly slow mid-spin. A spinning car still
        # has speed, so it stays in normal/off-track handling rather than being
        # thrown into reverse, which is what made it thrash before.
        if abs(v) < 1.5 and (abs(S.get('trackPos', 0)) > 0.9 or corridor is None):
            self.stuck_timer += 0.02
        else:
            self.stuck_timer = max(0.0, self.stuck_timer - 0.06)

        if self.stuck_timer > 2.0:
            return self._recover_stuck(S)

        if corridor is None:
            return self._recover_offtrack(S)

        xk, yc, yl, yr = corridor

        # Record the centreline curvature + width at the car for offline track
        # mapping (used to build the global racing line from a warm-up lap).
        self._record_map_features(xk, yc, yl, yr)

        s0 = S.get('distFromStart', None)
        if self.mapping:
            # Reconnaissance lap: just follow the reconstructed centreline.
            xs, yref = xk, yc
        else:
            # If a global racing line is loaded, project its offset for the
            # upcoming track onto the local nodes; the local solve is then
            # pulled toward the globally optimal line instead of the corridor
            # middle. distFromStart is the arc-length station; x ahead ~
            # arc-length ahead.
            target_off = None
            if self.track_map is not None and s0 is not None:
                goff = self.track_map.offset_at(s0 + xk)
                # clamp into the visible corridor so a map error can't aim off
                target_off = np.clip(yc + goff, yr + self.p['edge_margin'],
                                     yl - self.p['edge_margin'])
            # Minimum-curvature racing line; falls back to centreline on failure.
            rl = build_racing_line(xk, yl, yr, self.p, target=target_off)
            if rl is not None:
                xs, yref = rl
            else:
                xs, yref = xk, yc

        if _HAVE_SCIPY:
            cs = CubicSpline(xs, yref)
            curv = curvature_of_spline(cs, xs)
            ref_eval = lambda xx: (cs(xx), cs(xx, 1))
            xk_curv = xs
        else:
            ref_eval = self._make_linear_ref(xs, yref)
            curv = np.gradient(np.gradient(yref, xs), xs)
            xk_curv = xs

        # Signed path curvature right at the car (~0.6 m): recorded for offline
        # dynamics learning, where it is matched against the measured yaw rate.
        # Kept close to the car (not a lookahead) so it reflects the curvature
        # the car is *currently* turning through, not the corner ahead.
        self.last_kappa = float(np.interp(min(0.6, xs[-1]), xs, curv))

        # Furthest beam range drives the visibility speed cap; it flickers
        # frame-to-frame in live TORCS (the forward beam pops in and out of
        # range), which made the capped target speed - and so the throttle -
        # oscillate on straights. Low-pass it. Sight is long, so the small lag
        # this adds to spotting a corner is harmless.
        fwd_clear_raw = float(np.max(S['track']))
        if self.fwd_clear_filt is None:
            self.fwd_clear_filt = fwd_clear_raw
        else:
            af = self.p['fwd_clear_smooth']
            self.fwd_clear_filt = af * fwd_clear_raw + (1.0 - af) * self.fwd_clear_filt
        fwd_clear = self.fwd_clear_filt
        p = self.p
        # Braking decel for the look-ahead: physics-exact (downforce-boosted) at
        # the current speed if a car profile is loaded, else the flat constant.
        ab_now = (float(self.car.a_brake(v)) * p['brake_plan_frac']
                  if self.car is not None else p['a_brake'])
        if self.track_map is not None and s0 is not None:
            # MAP MODE: the speed target is the map's own look-ahead braking
            # profile - smooth, accurate, with full braking distance. The local
            # beam grip is used only at *near* range as a safety backstop; its
            # noisy long-range curvature (which invented phantom corners on
            # straights and made the throttle oscillate) is no longer trusted.
            grip_v = self._grip_speed(xk_curv, curv, max_x=p['grip_range'])
            # Map governs only beyond the local sensing range; within it, the
            # directly-sensed grip limit wins (so an over-tight map corner can't
            # crawl a corner the beams can see is fine).
            cap = p['map_speed_safety'] * self.track_map.lookahead_speed(
                s0, ab_now, p['map_brake_horizon'],
                start=p['grip_lookahead'])
            # SAFETY FLOOR: never go faster than we can stop within what the
            # beams can see, even when trusting the map. A wrong map corner (or
            # one the beams resolve late) cannot then carry the car in too hot -
            # this is the rule that was missing every time it ran off. On a
            # clear straight fwd is large so this does not slow us.
            vis = self._vis_speed(fwd_clear)
            raw = min(grip_v, cap, vis)
        else:
            # NO MAP: local grip over the full visible range + visibility cap.
            grip_v = self._grip_speed(xk_curv, curv)
            cap = self._vis_speed(fwd_clear)
            raw = min(grip_v, cap)
        if self.mapping:
            raw = min(raw, p['map_speed'])        # steady, moderate recon pace
        raw = clip(raw, p['v_min_corner'], p['v_max'])
        # Temporal low-pass to kill frame-to-frame target wobble, but let a
        # genuine large drop (real braking zone) through immediately for safety.
        prev = self.last_target_v
        if raw < prev - p['tv_snap']:
            target_v = raw
        else:
            target_v = p['tv_smooth'] * raw + (1.0 - p['tv_smooth']) * prev
        self.last_target_v = target_v

        steer = self._mpc_steer(v, xk, yref, yl, yr, ref_eval, target_v)
        # Output low-pass: blend the new command with the last one to damp the
        # residual jitter from the sampling-based optimiser.
        a = self.p['steer_smooth']
        steer = a * steer + (1.0 - a) * self.prev_steer
        self.prev_steer = steer
        accel, brake = self._long_control(v, target_v, S, steer)
        gear, clutch = self._gear_clutch(S, v)

        map_off = float(self.track_map.offset_at(s0)) \
            if (self.track_map is not None and s0 is not None) else 0.0
        return dict(steer=steer, accel=accel, brake=brake,
                    gear=gear, clutch=clutch, meta=0,
                    _target_v=target_v, _kappa=self.last_kappa,
                    _map_kappa=self.last_map_kappa,
                    _map_width=self.last_map_width,
                    _mode='map' if self.track_map is not None else
                          ('map-lap' if self.mapping else 'drive'),
                    _grip_v=grip_v, _cap=cap, _fwd_clear=fwd_clear,
                    _map_off=map_off, _trackpos_des=0.0)

    # ---- speed profile ---------------------------------------------------
    def _grip_speed(self, xk, curv, max_x=None):
        """Fastest speed allowed by the *visible* track: grip-limited corner
        speed sqrt(a_lat/kappa) plus look-ahead braking so we can still slow to
        each corner within what the beams resolve.

        The curvature is smoothed first: a single noisy beam frame can otherwise
        produce a one-node curvature spike that briefly craters the target speed.
        `max_x` limits the look-ahead to the near, reliably-reconstructed range:
        beyond ~40-50 m the beam fan is sparse and invents phantom corners, so
        when a (trustworthy, long-range) global map is available we only use the
        local grip for near safety and let the map handle far braking."""
        p = self.p
        a = np.abs(curv)
        if a.shape[0] >= 3:
            k = np.ones(3) / 3.0
            a = np.convolve(np.concatenate([a[:1], a, a[-1:]]), k, 'same')[1:-1]
        if max_x is not None:
            m = xk <= max_x
            if m.sum() >= 2:
                xk, a = xk[m], a[m]
        kappa = a + 1e-5
        if self.car is not None:
            v_curve = self.car.max_corner_speed(kappa)        # downforce-aware
            # plan braking conservatively so we slow *before* turn-in (friction
            # circle: can't brake at the limit while also cornering)
            ab = self.car.a_brake(v_curve) * p['brake_plan_frac']
        else:
            v_curve = np.sqrt(p['a_lat'] / kappa)
            ab = p['a_brake']
        v_allow = np.sqrt(np.maximum(v_curve ** 2 + 2.0 * ab * xk, 0.0))
        return float(np.min(v_allow))

    def _vis_speed(self, fwd_clear):
        """Visibility cap (used only without a global map): never go faster than
        we can bleed down to v_min_corner within what we can see, so the car
        backs off into blind corners the rangefinders cannot resolve yet."""
        p = self.p
        return math.sqrt(p['v_min_corner'] ** 2 +
                         2.0 * p['a_brake'] * max(fwd_clear - p['vis_margin'], 1.0))

    def _map_vis_speed(self, s0, fwd_clear):
        """Map-informed visibility cap: instead of assuming the worst hidden
        corner needs v_min_corner, use the global profile to find the slowest
        the track actually gets just beyond sight, and require we can brake down
        to it within the distance we can see."""
        p = self.p
        look0 = max(fwd_clear - p['vis_margin'], 1.0)
        ss = s0 + np.linspace(look0, look0 + p['map_horizon'], 12)
        v_future = float(np.min(self.track_map.speed_at(ss)))
        return math.sqrt(v_future ** 2 + 2.0 * p['a_brake'] * look0)

    def _record_map_features(self, xk, yc, yl, yr):
        """Estimate centreline curvature + track width at the car, for building
        the global map from a warm-up lap. Curvature from a quadratic fit to the
        near centreline (kappa ~ 2*c2 when the local slope is small); width from
        the median corridor width near the car."""
        near = xk < 30.0
        if near.sum() >= 3:
            xx, yy = xk[near], yc[near]
        else:
            xx, yy = xk, yc
        try:
            c = np.polyfit(xx, yy, 2)
            slope = 2.0 * c[0] * xx[0] + c[1]
            self.last_map_kappa = float(2.0 * c[0] / (1.0 + slope * slope) ** 1.5)
        except Exception:
            self.last_map_kappa = 0.0
        self.last_map_width = float(np.median(yl - yr))

    # ---- MPC -------------------------------------------------------------
    def _mpc_steer(self, v, xk, yref, yl, yr, ref_eval, target_v):
        p = self.p
        H, K = p['horizon'], p['n_samples']
        dt = p['dt_mpc']
        L = p['wheelbase']
        lock = p['steer_lock']

        # Warm-started nominal plan, shifted one step.
        nominal = np.empty(H)
        nominal[:-1] = self.prev_u[1:]
        nominal[-1] = self.prev_u[-1]

        # Sample K steering sequences as nominal + correlated noise.
        noise = np.random.randn(K, H) * p['steer_sigma']
        noise = np.cumsum(noise, axis=1) * 0.6 + noise * 0.4  # smooth/random-walk
        u = np.clip(nominal[None, :] + noise, -lock, lock)
        u[0] = np.clip(nominal, -lock, lock)  # keep the nominal as a candidate

        # Vectorised rollout for all K samples. Uses the learned dynamics model
        # when one is installed, otherwise the plain kinematic bicycle.
        dyn = self.dyn
        if dyn is not None:
            wb_eff = dyn['wheelbase_eff']
            understeer_K = dyn['understeer_K']
            acc_a0 = dyn['accel_a0']
            acc_a1 = dyn['accel_a1']
        x = np.zeros(K)
        y = np.zeros(K)
        psi = np.zeros(K)
        vel = np.full(K, v)
        x_lo, x_hi = xk[0], xk[-1]

        cost = np.zeros(K)
        prev_delta = np.zeros(K)
        for t in range(H):
            delta = u[:, t]
            xc = np.clip(x, x_lo, x_hi)
            yr_t = np.interp(xc, xk, yr)
            yl_t = np.interp(xc, xk, yl)
            yref_t, slope = ref_eval(xc)

            ct = y - yref_t                              # cross-track error
            he = _wrap(psi - np.arctan(slope))           # heading error
            off = (np.maximum(0.0, (yr_t + p['edge_margin']) - y) +
                   np.maximum(0.0, y - (yl_t - p['edge_margin'])))
            cost += (p['w_cross'] * ct * ct +
                     p['w_heading'] * he * he +
                     p['w_offtrack'] * off * off +
                     p['w_steer'] * delta * delta +
                     p['w_steer_rate'] * (delta - prev_delta) ** 2)
            prev_delta = delta

            # yaw model: learned understeer if available, else kinematic bicycle
            if dyn is not None:
                yaw_rate = vel * delta / (wb_eff + understeer_K * vel * vel)
            else:
                yaw_rate = vel / L * np.tan(delta)
            # longitudinal + grip limits: physics-exact (speed-dependent) car
            # profile if present, else learned drivetrain, else flat constants.
            if self.car is not None:
                a_acc_lim = self.car.a_accel(vel)
                a_brk_lim = self.car.a_brake(vel)
                lat_lim = self.car.a_lat(vel)
            elif dyn is not None:
                a_acc_lim = np.maximum(acc_a0 - acc_a1 * vel, 0.5)
                a_brk_lim = p['a_brake']
                lat_lim = p['a_lat']
            else:
                a_acc_lim = p['a_accel']
                a_brk_lim = p['a_brake']
                lat_lim = p['a_lat']
            acc = np.clip((target_v - vel) * 1.8, -a_brk_lim, a_acc_lim)
            vel = np.maximum(vel + acc * dt, 0.0)
            # Lateral grip limit: the car cannot out-turn its tyres. Modelling
            # this here stops the MPC from believing it can hold an arc that
            # reality would wash wide on.
            rate_cap = lat_lim / np.maximum(vel, 1.0)
            yaw_rate = np.clip(yaw_rate, -rate_cap, rate_cap)
            psi = psi + yaw_rate * dt
            x = x + vel * np.cos(psi) * dt
            y = y + vel * np.sin(psi) * dt

        cost -= p['w_progress'] * x  # reward longitudinal progress (least time)

        # MPPI soft-min weighting over sampled plans.
        beta = cost.min()
        w = np.exp(-(cost - beta) / p['mppi_lambda'])
        w /= w.sum() + 1e-9
        u_opt = (w[:, None] * u).sum(axis=0)
        self.prev_u = u_opt
        return clip(u_opt[0] / lock, -1.0, 1.0)

    # ---- longitudinal & gearbox -----------------------------------------
    def _long_control(self, v, target_v, S, steer=0.0):
        p = self.p
        err = target_v - v
        # Decisive, bang-bang style longitudinal control:
        #   below target            -> full throttle (flat to the floor)
        #   within the coast band    -> coast (no pedal) so it cannot oscillate
        #                               between throttle and brake at the target
        #   above target+coast_band  -> hard on the brakes (ABS keeps it stable)
        if err >= 0.0:
            accel = 1.0
            brake = 0.0
        elif err > -p['coast_band']:
            accel = 0.0
            brake = 0.0
        else:
            accel = 0.0
            brake = clip((-err - p['coast_band']) * p['kp_brake'],
                         0.3, p['brake_max'])

        # Physics brake cap: the car's brakes are far stronger than tyre grip,
        # so commanding full brake locks the (first-to-lock) front wheels - that
        # kills braking grip AND steering, and the car ploughs straight off.
        # Cap the command at the computed lock threshold (speed-dependent: higher
        # at speed as downforce loads the tyres) so braking stays grip-limited
        # and the fronts keep rolling and steering.
        if brake > 0.0 and self.car is not None:
            brake = min(brake, float(self.car.max_brake_cmd(v)))
            # Friction circle: how much lateral grip the current corner is using
            # (v^2*kappa / a_lat). Bleed the brake by what's left, so braking +
            # cornering together stay inside the tyre and it does not wash wide.
            lat_acc = v * v * abs(self.last_kappa)
            lat_frac = min(lat_acc / max(float(self.car.a_lat(v)), 1.0), 1.0)
            brake *= math.sqrt(max(p['fc_min'], 1.0 - lat_frac * lat_frac))

        # Power-oversteer guard: a slow corner taken in 1st/2nd at full throttle
        # spins the rear. Bleed the throttle the harder we are steering, and cap
        # it outright in the low gears where there is the most torque to spin up
        # the wheels. This is what stops the flat-out logic from snapping the
        # car around on corner exit.
        cut = 1.0 - p['steer_throttle_cut'] * abs(steer)
        accel *= clip(cut, p['min_throttle_factor'], 1.0)
        gear_now = int(S.get('gear', 1)) or 1
        if gear_now <= 1:
            accel = min(accel, p['accel_cap_g1'])
        elif gear_now == 2:
            accel = min(accel, p['accel_cap_g2'])
        elif gear_now == 3:
            accel = min(accel, p['accel_cap_g3'])

        # Traction control: ease off if the driven (rear) wheels are spinning
        # faster than the fronts (i.e. faster than the ground) - the early
        # warning of a power-on slide.
        w = S.get('wheelSpinVel', [0, 0, 0, 0])
        if (w[2] + w[3]) - (w[0] + w[1]) > p['tc_slip']:
            accel = max(0.0, accel - p['tc_cut'])
        # ABS: hold the wheels near their optimal braking slip (~12%) rather
        # than cutting hard the instant they slip at all. Under genuine
        # threshold braking the tyres always slip ~10-15% (that is where peak
        # grip is), so the old 'cut to 40% past 15% slip' throttled every hard
        # stop and the car could not slow for corners. Here the brake stays full
        # up to the optimal slip and only bleeds off proportionally once the
        # slip is genuinely deep (a wheel actually locking).
        if brake > 0 and v > 5 and w:
            r = self.car.wheel_radius if self.car is not None else 0.3276
            wheel_surf = min(w) * r                  # slowest wheel surface speed
            slip = (v - wheel_surf) / max(v, 1.0)
            if slip > p['abs_slip']:
                fac = 1.0 - (slip - p['abs_slip']) * p['abs_gain']
                brake *= clip(fac, p['abs_min'], 1.0)
        return accel, brake

    # Upper bound on gear for a given speed (m/s). Stops the box from lugging
    # along in 6th at 40 km/h: a high-rpm blip during acceleration used to
    # upshift and then nothing ever pulled the gear back down.
    _GEAR_SPEED_CAP = [13.0, 24.0, 37.0, 52.0, 70.0]  # max v for gears 1..5

    def _gear_clutch(self, S, v):
        rpm = S.get('rpm', 0)
        gear = int(S.get('gear', 1)) or 1
        if gear < 1:
            gear = 1

        # Speed-derived ceiling: highest gear whose speed window we are within.
        cap = 6
        for i, vmax in enumerate(self._GEAR_SPEED_CAP):
            if v < vmax:
                cap = i + 1
                break

        if rpm > 9300 and gear < cap:
            gear += 1
        elif gear > cap:
            gear = cap            # forced downshift when we have slowed down
        elif rpm < 4500 and gear > 1:
            gear -= 1
        gear = max(1, min(gear, cap))

        clutch = 0.0
        if v < 5.0:
            gear = max(gear, 1)
            clutch = clip(0.4 - v * 0.08, 0.0, 0.5)
        return gear, clutch

    # ---- recovery --------------------------------------------------------
    def _recover_offtrack(self, S):
        # Calmly rejoin: align the car with the track axis and steer back toward
        # the centre, but never add power while we are still carrying speed the
        # wrong way (that is what spins a slide into a full pirouette). Scrub
        # speed first, then feed in gentle, wheelspin-limited throttle.
        v = float(S['speedX'])
        angle = S.get('angle', 0.0)
        tp = clip(S.get('trackPos', 0.0), -3.0, 3.0)
        steer = clip(angle * 0.6 - tp * 0.35, -1, 1)
        if v > 12.0:
            accel, brake = 0.0, 0.45      # too fast off line -> brake
        elif v < -0.5:
            accel, brake = 0.0, 0.35      # sliding/rolling backwards -> arrest
        else:
            # Gentle, anti-spin throttle to crawl back on track.
            accel = clip(0.4 * (1.0 - 0.7 * abs(steer)), 0.15, 0.4)
            brake = 0.0
        gear = 1 if v < 3.0 else self._gear_clutch(S, v)[0]
        clutch = 0.2 if v < 3.0 else 0.0
        return dict(steer=steer, accel=accel, brake=brake,
                    gear=gear, clutch=clutch, meta=0, _target_v=12.0,
                    _mode='OFFTRACK')

    def _recover_stuck(self, S):
        # Genuinely pinned (e.g. nose against a wall): reverse straight out,
        # steering gently to bring the nose back toward the track centre.
        tp = S.get('trackPos', 0.0)
        if tp != 0:
            steer = clip(math.copysign(0.5, tp), -1, 1)
        else:
            steer = clip(-S.get('angle', 0.0), -1, 1)
        return dict(steer=steer, accel=0.4, brake=0.0,
                    gear=-1, clutch=0.0, meta=0, _target_v=8.0,
                    _mode='STUCK')

    @staticmethod
    def _make_linear_ref(xk, yref):
        def ref(xx):
            yy = np.interp(xx, xk, yref)
            slope = np.gradient(yref, xk)
            ss = np.interp(xx, xk, slope)
            return yy, ss
        return ref


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ===========================================================================
#  LEARNED DYNAMICS  (produced by learn_dynamics.py)
# ===========================================================================

# Keys a valid dynamics-model file must contain to be usable by the MPC.
_DYN_REQUIRED = ('wheelbase_eff', 'understeer_K', 'a_lat',
                 'a_brake', 'accel_a0', 'accel_a1')


def load_dynamics(path):
    """Load a learned vehicle-dynamics model written by learn_dynamics.py.

    Returns the model dict, or None if the file is missing/invalid (in which
    case the controller just runs on its hand-tuned defaults).
    """
    import json
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            dyn = json.load(f)
    except Exception as e:
        print("Could not read dynamics model %s: %s" % (path, e))
        return None
    missing = [k for k in _DYN_REQUIRED if k not in dyn]
    if missing:
        print("Dynamics model %s missing keys %s - ignoring." % (path, missing))
        return None
    return dyn


class DataLogger:
    """Appends one CSV row per control step so the driving can be replayed
    offline by learn_dynamics.py to fit the vehicle model. Speeds are stored in
    m/s (already converted), angles in rad."""

    COLUMNS = ['t', 'dist', 'v', 'vy', 'angle', 'trackPos', 'rpm', 'gear',
               'wsv0', 'wsv1', 'wsv2', 'wsv3',
               'steer', 'accel', 'brake', 'kappa', 'ontrack']

    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w')
        self.f.write(','.join(self.COLUMNS) + '\n')
        self.n = 0

    def log(self, S_raw, sensors, act):
        w = sensors.get('wheelSpinVel', [0, 0, 0, 0]) or [0, 0, 0, 0]
        w = list(w) + [0, 0, 0, 0]
        tp = float(S_raw.get('trackPos', 0.0))
        row = [
            float(S_raw.get('curLapTime', 0.0)),
            float(S_raw.get('distRaced', 0.0)),
            float(sensors.get('speedX', 0.0)),        # m/s
            float(sensors.get('speedY', 0.0)),        # m/s
            float(S_raw.get('angle', 0.0)),
            tp,
            float(S_raw.get('rpm', 0.0)),
            float(S_raw.get('gear', 0.0)),
            float(w[0]), float(w[1]), float(w[2]), float(w[3]),
            float(act.get('steer', 0.0)),
            float(act.get('accel', 0.0)),
            float(act.get('brake', 0.0)),
            float(act.get('_kappa', 0.0)),
            1.0 if abs(tp) <= 1.0 else 0.0,
        ]
        self.f.write(','.join('%.6g' % x for x in row) + '\n')
        self.n += 1
        if self.n % 50 == 0:
            self.f.flush()

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass
        print("Logged %d samples to %s" % (self.n, self.path))


class MapLogger:
    """Records one row per step of a warm-up lap so track_map.from_map_log can
    build the global racing line. Track curvature is reconstructed offline from
    the car's *motion* (steering -> yaw, plus heading change over distance),
    which is far smoother than instantaneous beam curvature; the width is taken
    from the beams. Columns: time, station, speed, steer, heading-error, width."""

    COLUMNS = ['t', 'dfs', 'v', 'steer', 'angle', 'width', 'trackPos', 'ontrack']

    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w')
        self.f.write(','.join(self.COLUMNS) + '\n')
        self.n = 0

    def log(self, S_raw, sensors, act):
        tp = float(S_raw.get('trackPos', 0.0))
        row = [
            float(S_raw.get('curLapTime', 0.0)),
            float(S_raw.get('distFromStart', 0.0)),
            float(sensors.get('speedX', 0.0)),       # m/s
            float(act.get('steer', 0.0)),
            float(S_raw.get('angle', 0.0)),
            float(act.get('_map_width', 0.0)),
            tp,
            1.0 if abs(tp) <= 1.0 else 0.0,
        ]
        self.f.write(','.join('%.6g' % x for x in row) + '\n')
        self.n += 1
        if self.n % 50 == 0:
            self.f.flush()

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass
        print("Logged %d map samples to %s" % (self.n, self.path))


# ===========================================================================
#  NETWORKING  (adapted from snakeoil / torcs_jm_par.py)
# ===========================================================================

def destringify(s):
    if not s:
        return s
    if isinstance(s, str):
        try:
            return float(s)
        except ValueError:
            return s
    if isinstance(s, list):
        if len(s) < 2:
            return destringify(s[0])
        return [destringify(i) for i in s]


class ServerState():
    def __init__(self):
        self.d = dict()

    def parse_server_str(self, server_string):
        self.servstr = server_string.strip()[:-1]
        sslisted = self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = destringify(w[1:])


class DriverAction():
    def __init__(self):
        self.d = {'accel': 0.2, 'brake': 0, 'clutch': 0, 'gear': 1,
                  'steer': 0, 'focus': [-90, -45, 0, 45, 90], 'meta': 0}

    def clip_to_limits(self):
        self.d['steer'] = clip(self.d['steer'], -1, 1)
        self.d['brake'] = clip(self.d['brake'], 0, 1)
        self.d['accel'] = clip(self.d['accel'], 0, 1)
        self.d['clutch'] = clip(self.d['clutch'], 0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear'] = 0
        if self.d['meta'] not in [0, 1]:
            self.d['meta'] = 0
        if (not isinstance(self.d['focus'], list) or
                min(self.d['focus']) < -180 or max(self.d['focus']) > 180):
            self.d['focus'] = 0

    def __repr__(self):
        self.clip_to_limits()
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not isinstance(v, list):
                out += '%.3f' % v
            else:
                out += ' '.join([str(x) for x in v])
            out += ')'
        return out


class Client():
    def __init__(self, H=None, p=None, i=None, e=None, t=None, s=None, d=None):
        self.host = 'localhost'
        self.port = 3001
        self.sid = 'SCR'
        self.maxEpisodes = 1
        self.trackname = 'unknown'
        self.stage = 3
        self.debug = False
        self.maxSteps = 100000
        self.logpath = None      # --log FILE : record telemetry for learning
        self.modelpath = None    # --model FILE : load a learned dynamics model
        self.maplogpath = None   # --map-log FILE : record a warm-up mapping lap
        self.trackmappath = None # --track-map FILE : load a global racing line
        self.parse_the_command_line()
        if H: self.host = H
        if p: self.port = p
        if i: self.sid = i
        if e: self.maxEpisodes = e
        if t: self.trackname = t
        if s: self.stage = s
        if d: self.debug = d
        self.S = ServerState()
        self.R = DriverAction()
        self.setup_connection()

    def setup_connection(self):
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error:
            print('Error: Could not create socket...')
            sys.exit(-1)
        self.so.settimeout(1)
        n_fail = 5
        while True:
            a = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
            initmsg = '%s(init %s)' % (self.sid, a)
            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error:
                sys.exit(-1)
            sockdata = str()
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error:
                print("Waiting for server on %d............" % self.port)
                print("Count Down : " + str(n_fail))
                if n_fail < 0:
                    n_fail = 5
                n_fail -= 1
            if '***identified***' in sockdata:
                print("Client connected on %d.............." % self.port)
                break

    def parse_the_command_line(self):
        try:
            (opts, args) = getopt.getopt(
                sys.argv[1:], 'H:p:i:m:e:t:s:l:dhv',
                ['host=', 'port=', 'id=', 'steps=', 'episodes=', 'track=',
                 'stage=', 'log=', 'model=', 'map-log=', 'track-map=',
                 'debug', 'help', 'version'])
        except getopt.error as why:
            print('getopt error: %s' % why)
            sys.exit(-1)
        for opt in opts:
            if opt[0] in ('-h', '--help'):
                print('see torcs_jm_par.py options'); sys.exit(0)
            if opt[0] in ('-d', '--debug'): self.debug = True
            if opt[0] in ('-H', '--host'): self.host = opt[1]
            if opt[0] in ('-i', '--id'): self.sid = opt[1]
            if opt[0] in ('-t', '--track'): self.trackname = opt[1]
            if opt[0] in ('-s', '--stage'): self.stage = int(opt[1])
            if opt[0] in ('-p', '--port'): self.port = int(opt[1])
            if opt[0] in ('-e', '--episodes'): self.maxEpisodes = int(opt[1])
            if opt[0] in ('-m', '--steps'): self.maxSteps = int(opt[1])
            if opt[0] in ('-l', '--log'): self.logpath = opt[1]
            if opt[0] == '--model': self.modelpath = opt[1]
            if opt[0] == '--map-log': self.maplogpath = opt[1]
            if opt[0] == '--track-map': self.trackmappath = opt[1]

    def get_servers_input(self):
        if not self.so:
            return
        sockdata = str()
        while True:
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error:
                print('.', end=' ')
            if '***identified***' in sockdata:
                continue
            elif '***shutdown***' in sockdata:
                print("Server shut down race on %d." % self.port)
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                print("Server restarted race on %d." % self.port)
                self.shutdown()
                return
            elif not sockdata:
                continue
            else:
                self.S.parse_server_str(sockdata)
                break

    def respond_to_server(self):
        if not self.so:
            return
        try:
            self.so.sendto(repr(self.R).encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s" % str(emsg))
            sys.exit(-1)

    def shutdown(self):
        if not self.so:
            return
        print("Shutting down %d." % self.port)
        self.so.close()
        self.so = None


# The live SCR server reports the 19 track beams in the opposite left/right
# handedness from the car-frame convention the planner uses (y positive = left).
# Reversing the array mirrors it back so the reconstructed corridor matches
# reality. (The offline sim is self-consistent and does not need this.)
MIRROR_BEAMS = True


def drive_mpc(c, driver):
    S, R = c.S.d, c.R.d
    # The SCR server reports speeds in km/h; the controller works in m/s.
    sensors = dict(S)
    sensors['speedX'] = S.get('speedX', 0.0) / 3.6
    sensors['speedY'] = S.get('speedY', 0.0) / 3.6
    if MIRROR_BEAMS and isinstance(S.get('track'), list):
        sensors['track'] = list(reversed(S['track']))
    act = driver.control(sensors)
    R['steer'] = act['steer']
    R['accel'] = act['accel']
    R['brake'] = act['brake']
    R['gear'] = act['gear']
    R['clutch'] = act['clutch']
    R['meta'] = act['meta']
    return sensors, act


# Default locations alongside this script (auto-loaded if present).
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(_HERE, 'dynamics_model.json')
DEFAULT_CAR_PROFILE = os.path.join(_HERE, 'car_profile.json')


if __name__ == "__main__":
    C = Client(p=3001)

    # Load a learned dynamics model if one is available.
    model_path = C.modelpath or DEFAULT_MODEL
    dyn = load_dynamics(model_path)

    # Load the physics-exact car profile if available (preferred over the
    # learned dynamics / hand-tuned constants for the grip & accel limits).
    car = None
    try:
        from car_profile import CarProfile
        car = CarProfile.load(DEFAULT_CAR_PROFILE)
        if car is not None:
            v = car.v
            print("Loaded car profile %s: a_lat %.1f..%.1f m/s^2, top %.0f km/h"
                  % (DEFAULT_CAR_PROFILE, car.alat.min(), car.alat.max(),
                     car.top_speed * 3.6))
    except Exception as e:
        print("Car profile not loaded: %s" % e)

    # Load a global racing line (track map) if requested.
    track_map = None
    if C.trackmappath:
        try:
            from track_map import TrackMap
            track_map = TrackMap.load(C.trackmappath)
            print("Loaded global racing line from %s "
                  "(len=%.0f m, target speed %.1f..%.1f m/s)"
                  % (C.trackmappath, track_map.length,
                     float(track_map.v_target.min()),
                     float(track_map.v_target.max())))
        except Exception as e:
            print("Could not load track map %s: %s" % (C.trackmappath, e))

    driver = MPCDriver(dynamics=dyn, track_map=track_map,
                       mapping=bool(C.maplogpath), car=car)
    if dyn is not None:
        print("Loaded learned dynamics from %s "
              "(a_lat=%.1f a_brake=%.1f understeer_K=%.4f)"
              % (model_path, driver.p['a_lat'], driver.p['a_brake'],
                 dyn['understeer_K']))
    else:
        print("No learned dynamics model - using hand-tuned defaults.")

    # Optional telemetry logging for offline learning, and mapping-lap logging.
    logger = DataLogger(C.logpath) if C.logpath else None
    if logger is not None:
        print("Logging telemetry to %s" % C.logpath)
    maplogger = MapLogger(C.maplogpath) if C.maplogpath else None
    if maplogger is not None:
        print("Recording mapping lap to %s "
              "(drive one clean lap, then: python track_map.py %s)"
              % (C.maplogpath, C.maplogpath))

    # Telemetry print rate (steps between lines). ~50 = 1 Hz. Set env
    # TORCS_TELEM_EVERY to a smaller number for denser logs to share for tuning.
    telem_every = int(os.environ.get('TORCS_TELEM_EVERY', '25'))
    print("# cols: t v(km/h) v(m/s) tv grip cap fwd | acc brk steer | "
          "tp ang gear rpm | dist mode last")

    n = 0
    last_dist = 0.0
    try:
        for step in range(C.maxSteps, 0, -1):
            C.get_servers_input()
            if not C.so:
                break
            sensors, act = drive_mpc(C, driver)
            C.respond_to_server()
            if logger is not None:
                logger.log(C.S.d, sensors, act)
            if maplogger is not None:
                maplogger.log(C.S.d, sensors, act)
            n += 1
            if n % telem_every == 0:
                S = C.S.d
                last_dist = S.get('distRaced', last_dist)
                vms = S.get('speedX', 0.0) / 3.6
                print("t=%6.1f v=%5.0f %5.1f tv=%5.1f grip=%5.1f cap=%5.1f "
                      "fwd=%5.1f | acc=%.2f brk=%.2f str=%+.2f | "
                      "tp=%+.2f ang=%+.2f g%d rpm=%4.0f | "
                      "d=%5.0f %-8s last=%s"
                      % (S.get('curLapTime', 0), S.get('speedX', 0), vms,
                         act.get('_target_v', 0), act.get('_grip_v', 0),
                         act.get('_cap', 0), act.get('_fwd_clear', 0),
                         act.get('accel', 0), act.get('brake', 0),
                         act.get('steer', 0),
                         S.get('trackPos', 0), S.get('angle', 0),
                         int(S.get('gear', 0)), S.get('rpm', 0),
                         last_dist, act.get('_mode', '?'),
                         S.get('lastLapTime', 0)))
    finally:
        if logger is not None:
            logger.close()
        if maplogger is not None:
            maplogger.close()
    C.shutdown()
