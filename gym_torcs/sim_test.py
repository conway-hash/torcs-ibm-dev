#!/usr/bin/env python
# sim_test.py
#
# Offline test harness for torcs_mpc.MPCDriver. It builds synthetic tracks,
# simulates a kinematic-bicycle car with a lateral-grip limit, ray-casts the
# 19 SCR rangefinder beams against the track edges, runs the driver, and
# reports lap time / off-track behaviour. This lets us iterate the controller
# without the TORCS GUI.

import math
import numpy as np

from torcs_mpc import MPCDriver, PARAMS

WHEEL_R = 0.3179


# ---------------------------------------------------------------------------
#  Track construction
# ---------------------------------------------------------------------------
def build_track(segments, width=12.0, ds=1.0):
    """segments: list of ('straight', length) or ('turn', radius, angle_deg).
    Returns a dict with centreline samples and left/right edge polylines."""
    X, Y, H = [0.0], [0.0], [0.0]
    x, y, h = 0.0, 0.0, 0.0
    for seg in segments:
        if seg[0] == 'straight':
            length = seg[1]
            n = max(1, int(length / ds))
            for _ in range(n):
                x += ds * math.cos(h)
                y += ds * math.sin(h)
                X.append(x); Y.append(y); H.append(h)
        else:  # turn: radius, signed angle (deg, +left)
            radius, ang = seg[1], math.radians(seg[2])
            arc = abs(ang) * radius
            n = max(1, int(arc / ds))
            dh = ang / n
            for _ in range(n):
                h += dh
                x += ds * math.cos(h)
                y += ds * math.sin(h)
                X.append(x); Y.append(y); H.append(h)
    X = np.array(X); Y = np.array(Y); H = np.array(H)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    # normals point left of travel direction
    nx, ny = -np.sin(H), np.cos(H)
    hw = width * 0.5
    left = np.column_stack([X + nx * hw, Y + ny * hw])
    right = np.column_stack([X - nx * hw, Y - ny * hw])
    return dict(X=X, Y=Y, H=H, s=s, left=left, right=right,
                width=width, length=float(s[-1]))


TRACKS = {
    # A "maze" of mixed corners: sweepers, a chicane, and a hairpin.
    'maze': [
        ('straight', 120),
        ('turn', 45, -80),
        ('straight', 40),
        ('turn', 35, 90),
        ('straight', 30),
        ('turn', 20, -150),   # hairpin
        ('straight', 60),
        ('turn', 60, 70),
        ('straight', 25),
        ('turn', 25, -60),    # chicane part 1
        ('turn', 25, 60),     # chicane part 2
        ('straight', 80),
        ('turn', 50, -90),
        ('straight', 100),
    ],
    'sweepers': [
        ('straight', 200), ('turn', 80, -90), ('straight', 150),
        ('turn', 90, 90), ('straight', 200), ('turn', 70, -120),
        ('straight', 120),
    ],
    'hairpins': [
        ('straight', 80), ('turn', 15, -170), ('straight', 80),
        ('turn', 15, 170), ('straight', 80), ('turn', 18, -160),
        ('straight', 80),
    ],
    # Closed loops (return to start) - needed for the global racing line, which
    # optimises over a periodic circuit. Built symmetrically so they close.
    'square': [('straight', 200), ('turn', 30, 90)] * 4,   # isolated 90s
    'oval': [('straight', 220), ('turn', 55, 180),
             ('straight', 220), ('turn', 55, 180)],
}

# Which tracks are closed loops (one lap returns to the start).
CLOSED_TRACKS = ('square', 'oval')


# ---------------------------------------------------------------------------
#  Geometry helpers
# ---------------------------------------------------------------------------
def nearest_index(track, x, y, hint, window=60):
    lo = max(0, hint - window)
    hi = min(len(track['X']), hint + window)
    dx = track['X'][lo:hi] - x
    dy = track['Y'][lo:hi] - y
    return lo + int(np.argmin(dx * dx + dy * dy))


def ray_cast(ox, oy, ang, segs_a, segs_b, max_d=200.0):
    """Nearest intersection distance of ray (ox,oy)+t*(cos,sin) with the
    segment set [segs_a -> segs_b]. Vectorised over all segments."""
    ux, uy = math.cos(ang), math.sin(ang)
    ax, ay = segs_a[:, 0], segs_a[:, 1]
    bx, by = segs_b[:, 0], segs_b[:, 1]
    ex, ey = bx - ax, by - ay
    denom = ux * ey - uy * ex
    nz = np.abs(denom) > 1e-9
    t = np.full(ax.shape, np.inf)
    wx = ax - ox
    wy = ay - oy
    s = (wx * uy - wy * ux)
    s = np.where(nz, s / np.where(nz, denom, 1.0), -1.0)
    tt = np.where(nz, (wx * ey - wy * ex) / np.where(nz, denom, 1.0), -1.0)
    good = nz & (s >= 0.0) & (s <= 1.0) & (tt >= 0.0)
    t = np.where(good, tt, np.inf)
    d = float(np.min(t))
    return min(d, max_d) if np.isfinite(d) else max_d


# ---------------------------------------------------------------------------
#  Simulator
# ---------------------------------------------------------------------------
def simulate(driver, track, max_time=120.0, dt=0.02, a_lat_phys=17.0,
             verbose=False, laps=1, logger=None, maplogger=None, car=None):
    x, y = track['X'][0], track['Y'][0]
    yaw = track['H'][0]
    v = 0.0
    hint = 0
    t = 0.0
    p = PARAMS
    offtrack_steps = 0
    total_progress = 0.0
    prev_s = 0.0
    lap_target = track['length'] * laps
    wrap_offset = 0.0
    steps = 0

    while t < max_time:
        idx = nearest_index(track, x, y, hint)
        hint = idx
        th0 = track['H'][idx]
        # lateral offset (left positive) and continuous longitudinal station.
        # Real SCR reports continuous distFromStart and a smooth heading;
        # project the car onto the local centreline tangent and interpolate the
        # centreline heading at that station (rather than snapping to the
        # nearest 1 m sample), so angle and d(distFromStart) are smooth.
        dx, dy = x - track['X'][idx], y - track['Y'][idx]
        offset = -math.sin(th0) * dx + math.cos(th0) * dy
        along = math.cos(th0) * dx + math.sin(th0) * dy
        dist_from_start = float(track['s'][idx] + along)
        th = float(np.interp(dist_from_start, track['s'], track['H']))
        hw = track['width'] * 0.5
        trackpos = offset / hw
        angle = math.atan2(math.sin(th - yaw), math.cos(th - yaw))

        # build beam readings
        lo = max(0, idx - 80)
        hi = min(len(track['X']) - 1, idx + 220)
        la = track['left'][lo:hi]
        lb = track['left'][lo + 1:hi + 1]
        ra = track['right'][lo:hi]
        rb = track['right'][lo + 1:hi + 1]
        seg_a = np.vstack([la, ra])
        seg_b = np.vstack([lb, rb])

        beams = []
        if abs(trackpos) > 1.0:
            beams = [-1.0] * 19
        else:
            from torcs_mpc import TRACK_ANGLES
            for ba in TRACK_ANGLES:
                beams.append(ray_cast(x, y, yaw + ba, seg_a, seg_b))

        gear = 1 + int(v > 12) + int(v > 22) + int(v > 35) + int(v > 50) + int(v > 70)
        rpm = 2000 + (v / max(gear, 1)) * 320
        wsv = [v / WHEEL_R] * 4

        S = dict(angle=angle, trackPos=trackpos, speedX=v, speedY=0.0,
                 speedZ=0.0, track=beams, rpm=rpm, gear=gear,
                 wheelSpinVel=wsv, distFromStart=dist_from_start)
        act = driver.control(S)

        if logger is not None:
            S_raw = dict(S)
            S_raw['curLapTime'] = t
            S_raw['distRaced'] = total_progress
            logger.log(S_raw, S, act)
        if maplogger is not None:
            S_rawm = dict(S)
            S_rawm['curLapTime'] = t
            maplogger.log(S_rawm, S, act)

        # --- vehicle dynamics ---
        delta = act['steer'] * p['steer_lock']
        # If a physics-exact car profile is supplied, use ITS speed-dependent
        # accel/brake/grip so the sim car matches car1-trb1; otherwise the old
        # generic constants.
        if car is not None:
            a_drive = act['accel'] * float(car.a_accel(v))
            # Brake force is proportional to command up to the lock threshold,
            # where it reaches the grip-limited deceleration (matches TORCS).
            bc = max(float(car.max_brake_cmd(v)), 0.05)
            a_brk = min(act['brake'] / bc, 1.0) * float(car.a_brake(v))
            grip_lat = float(car.a_lat(v))
        else:
            a_drive = act['accel'] * max(2.0, 9.0 - v * 0.05)
            a_brk = act['brake'] * 13.0
            grip_lat = a_lat_phys
        v = max(0.0, v + (a_drive - a_brk) * dt)
        if act['gear'] == -1:
            v = -max(0.0, a_drive * dt) * 4 + min(v, 0)  # crude reverse

        yaw_rate = v / p['wheelbase'] * math.tan(delta)
        # lateral grip limit -> understeer when over the limit. The grip is
        # shared with longitudinal load (friction circle): hard braking or
        # accelerating leaves less for cornering, so trail-braking into a corner
        # washes wide if the combined demand exceeds the tyre.
        if abs(v) > 0.1:
            a_long = a_drive - a_brk
            a_lat_avail = math.sqrt(max(0.0, grip_lat * grip_lat -
                                        a_long * a_long))
            max_rate = a_lat_avail / abs(v)
            yaw_rate = max(-max_rate, min(max_rate, yaw_rate))
        yaw += yaw_rate * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt

        if abs(trackpos) > 1.05:
            offtrack_steps += 1

        # progress tracking with lap wrap
        s_here = track['s'][idx] + wrap_offset
        if track['s'][idx] + wrap_offset < prev_s - track['length'] * 0.5:
            wrap_offset += track['length']
            s_here = track['s'][idx] + wrap_offset
        total_progress = max(total_progress, s_here)
        prev_s = s_here

        t += dt
        steps += 1
        if verbose and steps % 50 == 0:
            print("t=%5.1f s=%7.1f v=%5.1f tp=%+.2f steer=%+.2f tv=%.1f"
                  % (t, total_progress, v, trackpos, act['steer'],
                     act.get('_target_v', 0)))

        if total_progress >= lap_target:
            return dict(finished=True, time=t, progress=total_progress,
                        offtrack=offtrack_steps, avg_speed=total_progress / t)
        if offtrack_steps > 250:  # spun off and not recovering
            break

    return dict(finished=False, time=t, progress=total_progress,
                offtrack=offtrack_steps,
                avg_speed=total_progress / max(t, 1e-3))


def ground_truth_map(track, a_lat=16.5, a_brake=12.0, a_accel=7.0,
                     margin=2.5, dynamics=None):
    """Build a TrackMap straight from a sim track's known centreline geometry.
    Lets us validate the global racing line independent of beam-mapping noise."""
    import track_map as tm
    return tm.from_centerline(track['X'], track['Y'], track['width'],
                              ds=2.0).build(a_lat=a_lat, a_brake=a_brake,
                                            a_accel=a_accel, margin=margin,
                                            dynamics=dynamics)


def run_all(params=None, tracks=('maze', 'sweepers', 'hairpins'), verbose=False,
            logpath=None, dynamics=None, use_map=False, car=None):
    logger = None
    if logpath:
        from torcs_mpc import DataLogger
        logger = DataLogger(logpath)
    results = {}
    for name in tracks:
        track = build_track(TRACKS[name])
        tmap = ground_truth_map(track, dynamics=dynamics) if use_map else None
        driver = MPCDriver(params, dynamics=dynamics, track_map=tmap, car=car)
        np.random.seed(0)
        r = simulate(driver, track, verbose=verbose, logger=logger, car=car)
        results[name] = r
        status = 'FIN' if r['finished'] else 'DNF'
        print("%-9s %s len=%6.1fm  prog=%6.1fm  t=%6.2fs  "
              "avg=%5.1f m/s  offtrack=%d"
              % (name, status, track['length'], r['progress'], r['time'],
                 r['avg_speed'], r['offtrack']))
    if logger is not None:
        logger.close()
    return results


if __name__ == "__main__":
    import sys
    verbose = '-v' in sys.argv
    logpath = None
    dynamics = None
    use_map = '--map' in sys.argv
    tracks = ('maze', 'sweepers', 'hairpins')
    if '--closed' in sys.argv:
        tracks = CLOSED_TRACKS
    if '--log' in sys.argv:
        logpath = sys.argv[sys.argv.index('--log') + 1]
    if '--model' in sys.argv:
        from torcs_mpc import load_dynamics
        dynamics = load_dynamics(sys.argv[sys.argv.index('--model') + 1])
        print("dynamics model loaded:", dynamics is not None)
    car = None
    if '--car' in sys.argv:
        from car_profile import CarProfile
        car = CarProfile.load(sys.argv[sys.argv.index('--car') + 1])
        print("car profile loaded:", car is not None)
    run_all(verbose=verbose, logpath=logpath, dynamics=dynamics,
            tracks=tracks, use_map=use_map, car=car)
