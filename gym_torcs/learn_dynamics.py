#!/usr/bin/env python
# learn_dynamics.py
#
# Fits a compact, physically-interpretable vehicle-dynamics model from the
# telemetry logged by torcs_mpc.py (run it with `--log FILE` to produce the
# CSV), then saves the fitted parameters to dynamics_model.json. torcs_mpc.py
# loads that file automatically and uses it inside the MPC rollout + speed
# profile, so the controller plans against what the car *actually* does rather
# than an idealised kinematic bicycle.
#
# What is learned (all from data, no hand tuning):
#   * accel_a0, accel_a1 : full-throttle acceleration vs speed,  a = a0 - a1*v
#                          (captures engine power fall-off + aero drag)
#   * a_brake            : achievable braking deceleration (m/s^2)
#   * coast_decel        : deceleration when coasting (drag + engine braking)
#   * a_lat              : grip-limited lateral acceleration (m/s^2)
#   * wheelbase_eff,     : steering response  yaw = v*delta / (L_eff + K*v^2)
#     understeer_K         (K>0 => the car understeers more the faster it goes)
#
# Why a parametric model and not a neural net: with the handful of signals SCR
# exposes, these few physical parameters capture essentially all of the
# behaviour that matters to the planner, are robust to noisy/limited data, and
# stay fully interpretable. The fitting below is ordinary least squares +
# robust percentiles; the same data/JSON interface would let a NN be dropped in
# later if ever needed.
#
# Usage:
#   python learn_dynamics.py LOG.csv [MORE.csv ...]        # -> dynamics_model.json
#   python learn_dynamics.py --out my_model.json LOG.csv
#   python learn_dynamics.py "logs/*.csv"

import os
import sys
import glob
import json
import time
import math

import numpy as np

try:
    from torcs_mpc import PARAMS
    _WHEELBASE = PARAMS['wheelbase']
    _STEER_LOCK = PARAMS['steer_lock']
except Exception:
    _WHEELBASE = 2.4
    _STEER_LOCK = 0.366

COLUMNS = ['t', 'dist', 'v', 'vy', 'angle', 'trackPos', 'rpm', 'gear',
           'wsv0', 'wsv1', 'wsv2', 'wsv3',
           'steer', 'accel', 'brake', 'kappa', 'ontrack']
WHEEL_R = 0.3179
PLANNER_SAFETY_VERSION = 1


# ---------------------------------------------------------------------------
#  Loading & transition extraction
# ---------------------------------------------------------------------------
def load_logs(paths):
    rows = []
    for p in paths:
        for fn in sorted(glob.glob(p)) or [p]:
            if not os.path.exists(fn):
                print("  (skip, not found: %s)" % fn)
                continue
            data = np.genfromtxt(fn, delimiter=',', names=True)
            if data.size == 0:
                continue
            data = np.atleast_1d(data)
            rows.append(data)
            print("  loaded %5d rows from %s" % (data.shape[0], fn))
    if not rows:
        return None
    # All files share the header, so fields line up.
    return np.concatenate(rows)


def _smooth(a, w=9):
    """Centred moving average; tamps down sensor/finite-difference noise (e.g.
    the quantised track heading in the offline sim) before differentiating."""
    a = np.asarray(a, dtype=float)
    if w < 3 or a.shape[0] < w:
        return a
    k = np.ones(w) / w
    return np.convolve(a, k, mode='same')


def build_transitions(d):
    """Return a dict of per-transition arrays (row i -> row i+1), keeping only
    physically continuous, on-track samples."""
    t = d['t']
    dt = np.diff(t)
    # Valid step: small positive time gap (drop lap-time resets and stalls),
    # and on track at both ends.
    ok = (dt > 1e-3) & (dt < 0.1)
    ok &= (d['ontrack'][:-1] > 0.5) & (d['ontrack'][1:] > 0.5)

    v = d['v']
    # smooth heading & curvature before differencing (kills finite-difference
    # noise; the offline sim's track heading is quantised by nearest-sample)
    angle_s = _smooth(d['angle'], 9)
    kappa_s = _smooth(d['kappa'], 5)
    # angle change between consecutive samples (wrapped to [-pi, pi])
    dang = np.diff(angle_s)
    dang = (dang + np.pi) % (2 * np.pi) - np.pi
    out = dict(
        dt=dt[ok],
        v=v[:-1][ok],
        v_next=v[1:][ok],
        accel=d['accel'][:-1][ok],
        brake=d['brake'][:-1][ok],
        steer=d['steer'][:-1][ok],
        kappa=kappa_s[:-1][ok],
        gear=d['gear'][:-1][ok],
        rpm=d['rpm'][:-1][ok],
        dangle=dang[ok],
    )
    # measured longitudinal acceleration (m/s^2)
    out['a_long'] = (out['v_next'] - out['v']) / out['dt']
    # Actual yaw rate. The car's heading error vs the track changes as
    #   d(angle)/dt = v*kappa_track - yaw_rate, so  yaw_rate = v*kappa - dangle/dt.
    # Using kappa of the line the car is on (~kappa_track) this recovers the
    # *real* yaw rate, which automatically drops below v*kappa when the car is
    # washing wide at the grip limit - exactly what we need to measure grip.
    out['yaw_rate'] = out['kappa'] * out['v'] - out['dangle'] / out['dt']
    # wheel slip ratio: driven (rear) wheel surface speed vs ground speed
    wsv_rear = 0.5 * (d['wsv2'] + d['wsv3'])[:-1][ok]
    out['slip'] = wsv_rear * WHEEL_R - out['v']
    return out


# ---------------------------------------------------------------------------
#  Fits
# ---------------------------------------------------------------------------
def fit_longitudinal(tr, info):
    v = tr['v']
    a = tr['a_long']

    # --- full-throttle acceleration vs speed: a = a0 - a1 v ---
    m = (tr['accel'] > 0.85) & (tr['brake'] < 0.05) & (v > 3.0) & \
        (np.abs(tr['slip']) < 8.0) & (a > -2.0)
    n_acc = int(m.sum())
    if n_acc >= 20:
        A = np.vstack([np.ones(n_acc), v[m]]).T
        coef, *_ = np.linalg.lstsq(A, a[m], rcond=None)
        a0 = float(coef[0])
        a1 = float(-coef[1])           # a = a0 - a1 v  => slope is -a1
        # guard against nonsense fits
        a0 = float(np.clip(a0, 2.0, 20.0))
        a1 = float(np.clip(a1, 0.0, 0.2))
    else:
        a0, a1 = PARAMS_DEFAULT('a_accel'), 0.05
    info['accel_samples'] = n_acc

    # --- braking deceleration ---
    mb = (tr['brake'] > 0.6) & (v > 5.0)
    decel = -a[mb]
    decel = decel[(decel > 1.0) & (decel < 45.0)]
    n_brk = int(decel.shape[0])
    if n_brk >= 15:
        default = PARAMS_DEFAULT('a_brake')
        raw_p55 = float(np.percentile(decel, 55))
        # The planner needs repeatable braking, not the best spike in the log.
        # Cap the estimate relative to the known-good default and blend back
        # toward that default until there is enough braking data.
        repeatable = min(raw_p55, default * 1.35)
        confidence = float(np.clip((n_brk - 15) / 80.0, 0.0, 1.0))
        a_brake = (1.0 - confidence) * default + confidence * repeatable
        a_brake = float(np.clip(a_brake, 8.0, 16.0))
        info['brake_raw_p55'] = round(raw_p55, 2)
        info['brake_confidence'] = round(confidence, 3)
    else:
        a_brake = PARAMS_DEFAULT('a_brake')
    info['brake_samples'] = n_brk

    # --- coast deceleration (drag + engine braking) ---
    mc = (tr['accel'] < 0.1) & (tr['brake'] < 0.1) & (v > 5.0)
    coast = float(np.clip(np.median(-a[mc]), 0.0, 8.0)) if mc.sum() >= 10 else 1.0

    return dict(accel_a0=a0, accel_a1=a1, a_brake=a_brake, coast_decel=coast)


def fit_lateral(tr, info):
    v = tr['v']
    kappa = tr['kappa']
    r = tr['yaw_rate']
    delta = tr['steer'] * _STEER_LOCK

    # Lateral grip = the largest sustained lateral acceleration |v * yaw_rate|.
    # Using the *measured* yaw rate (not the planned curvature) means samples
    # where the car washed wide contribute their true, lower value, so the
    # percentile reflects real grip instead of the planner's intent. Require
    # genuine cornering and roughly steady speed (so longitudinal load transfer
    # is not eating the lateral budget).
    a_long = tr['a_long']
    mk = (np.abs(kappa) > 3e-3) & (v > 6.0) & (np.abs(r) > 0.02) & \
         (np.abs(a_long) < 4.0)
    n_lat = int(mk.sum())
    if n_lat >= 20:
        # Two independent estimates of sustained lateral grip:
        #   yaw-based   |v * yaw_rate|  - correct even when the car understeers
        #               wide of its line, but sensitive to heading noise.
        #   curvature   v^2 * |kappa|   - robust/clean, but a mild over-estimate
        #               at the limit (assumes the car held the planned line).
        # Taking the lower of the two is robust to both failure modes; a small
        # safety factor keeps us just inside the real limit (over-estimating
        # grip is what sends the car off; under-estimating only costs a little).
        yaw_est = float(np.percentile(np.abs(v[mk] * r[mk]), 70))
        curv_est = float(np.percentile((v[mk] ** 2) * np.abs(kappa[mk]), 75))
        a_lat = 0.90 * min(yaw_est, curv_est)
        a_lat = float(np.clip(a_lat, 8.0, 20.0))
        info['lat_yaw_est'] = round(yaw_est, 2)
        info['lat_curv_est'] = round(curv_est, 2)
    else:
        a_lat = PARAMS_DEFAULT('a_lat')
    info['lat_samples'] = n_lat

    # Understeer: model  yaw_rate = v*delta / (L + K v^2)  =>
    #   v*delta / yaw_rate = L + K v^2.
    # Regress that on v^2 over consistent-sign, decent-signal cornering samples.
    mu = (np.abs(kappa) > 3e-3) & (v > 6.0) & (np.sign(delta) == np.sign(r)) & \
         (np.abs(delta) > 0.02) & (np.abs(r) > 0.03)
    ratio = v[mu] * delta[mu] / r[mu]
    vv = v[mu] ** 2
    good = (ratio > 0.5) & (ratio < 200.0)        # plausible radii only
    ratio, vv = ratio[good], vv[good]
    n_us = int(ratio.shape[0])
    if n_us >= 30:
        A = np.vstack([np.ones(n_us), vv]).T
        coef, *_ = np.linalg.lstsq(A, ratio, rcond=None)
        L_eff = float(np.clip(coef[0], 1.5, 4.0))
        K = float(np.clip(coef[1], 0.0, 0.03))
    else:
        L_eff, K = _WHEELBASE, 0.0
    info['understeer_samples'] = n_us

    return dict(a_lat=a_lat, wheelbase_eff=L_eff, understeer_K=K)


def PARAMS_DEFAULT(key):
    try:
        from torcs_mpc import PARAMS
        return float(PARAMS[key])
    except Exception:
        return {'a_accel': 7.0, 'a_brake': 12.0, 'a_lat': 16.5}[key]


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main(argv):
    out_path = 'dynamics_model.json'
    paths = []
    i = 0
    while i < len(argv):
        if argv[i] in ('--out', '-o'):
            out_path = argv[i + 1]; i += 2
        else:
            paths.append(argv[i]); i += 1
    if not paths:
        paths = ['torcs_log.csv']

    print("Loading logs...")
    d = load_logs(paths)
    if d is None:
        print("No usable log data found in: %s" % paths)
        return 1

    tr = build_transitions(d)
    n = tr['dt'].shape[0]
    print("Usable transitions: %d (%.1f s of driving)"
          % (n, float(np.sum(tr['dt']))))
    if n < 100:
        print("WARNING: very little data - drive a lap or two with --log first.")
        if n < 20:
            return 1

    info = {}
    lon = fit_longitudinal(tr, info)
    lat = fit_lateral(tr, info)

    model = {}
    model.update(lon)
    model.update(lat)
    model['_meta'] = dict(
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        sources=paths,
        n_transitions=int(n),
        samples=info,
        wheelbase_nominal=_WHEELBASE,
        steer_lock=_STEER_LOCK,
        planner_safety_version=PLANNER_SAFETY_VERSION,
    )

    with open(out_path, 'w') as f:
        json.dump(model, f, indent=2)

    print("\nLearned vehicle dynamics  ->  %s" % out_path)
    print("  acceleration   a(v) = %.2f - %.4f*v   m/s^2   (n=%d)"
          % (lon['accel_a0'], lon['accel_a1'], info.get('accel_samples', 0)))
    print("  braking        a_brake = %.2f m/s^2          (n=%d)"
          % (lon['a_brake'], info.get('brake_samples', 0)))
    print("  coasting       coast   = %.2f m/s^2" % lon['coast_decel'])
    print("  lateral grip   a_lat   = %.2f m/s^2          (n=%d)"
          % (lat['a_lat'], info.get('lat_samples', 0)))
    print("  steering       yaw = v*delta/(%.2f + %.4f*v^2)  (n=%d)"
          % (lat['wheelbase_eff'], lat['understeer_K'],
             info.get('understeer_samples', 0)))
    print("\nRun torcs_mpc.py and it will load this automatically.")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
