#!/usr/bin/env python
# car_profile.py
#
# Build a *physics-exact* performance profile for a TORCS car straight from its
# XML spec + the track surface friction, reproducing the simuv2 engine formulas
# (read from src/modules/simu/simuv2: aero.cpp, wheel.cpp). This replaces the
# hand-guessed a_lat / a_brake / a_accel constants with the car's real,
# speed-dependent limits.
#
# Physics reproduced (exact TORCS formulas):
#   Aero drag      : F_drag = 0.645*Cx*FrontArea * v^2          (0.645 = 0.5*rho)
#   Body downforce : F = Clift * v^2 * hm   (hm = ground-effect from ride height)
#   Wing downforce : F = 4*1.23*area*sin(angle) * v^2           (1.23 = rho)
#   Tyre grip      : Fmax = Fz * mu_eff * kFriction * camber
#                    mu_eff = mu*(lfMin + (lfMax-lfMin)*exp(lfK*Fz/opLoad))
#                    lfK = ln((1-lfMin)/(lfMax-lfMin)), opLoad = 1.2*staticLoad
#   Engine         : torque curve * gear * finaldrive * eff / wheelRadius
#
# Output: car_profile.json with a_lat(v), a_brake(v), a_accel(v), shift points,
# top speed - loaded by torcs_mpc.py for the speed profile / MPC / map.
#
# Usage:
#   python car_profile.py CAR.xml [--kfric 1.1] [--fuel 40] [--out car_profile.json]

import sys
import json
import math
import xml.etree.ElementTree as ET

import numpy as np

RHO = 1.23
G = 9.81


# ---------------------------------------------------------------------------
#  XML helpers (TORCS params: nested <section name=..> with <attnum/attstr>)
# ---------------------------------------------------------------------------
def _sec(node, *names):
    for name in names:
        nxt = None
        for s in node.findall('section'):
            if s.get('name') == name:
                nxt = s
                break
        if nxt is None:
            return None
        node = nxt
    return node


def _num(node, name, default=None):
    if node is None:
        return default
    for a in node.findall('attnum'):
        if a.get('name') == name:
            return float(a.get('val'))
    return default


def _str(node, name, default=None):
    if node is None:
        return default
    for a in node.findall('attstr'):
        if a.get('name') == name:
            return a.get('val')
    return default


# ---------------------------------------------------------------------------
#  Profile computation
# ---------------------------------------------------------------------------
def build_profile(car_xml, kfriction=1.1, fuel_kg=40.0, v_max=120.0, dv=1.0):
    root = ET.parse(car_xml).getroot()

    car = _sec(root, 'Car')
    mass = _num(car, 'mass', 1150.0) + fuel_kg
    frep = _num(car, 'front-rear weight repartition', 0.5)   # front fraction

    aero = _sec(root, 'Aerodynamics')
    Cx = _num(aero, 'Cx', 0.35)
    frontArea = _num(aero, 'front area', 2.0)
    cl_f = _num(aero, 'front Clift', 0.0)
    cl_r = _num(aero, 'rear Clift', 0.0)
    SCx2 = 0.645 * Cx * frontArea                  # drag coeff (incl 0.5*rho)

    # ground-effect height factor hm (uses nominal ride heights)
    rh = []
    for wn in ('Front Right Wheel', 'Front Left Wheel',
               'Rear Right Wheel', 'Rear Left Wheel'):
        rh.append(_num(_sec(root, wn), 'ride height', 100.0) / 1000.0)
    hm = 1.5 * sum(rh)
    hm = hm * hm
    hm = hm * hm
    hm = 2.0 * math.exp(-3.0 * hm)

    # wing downforce coefficients: F = 4*1.23*area*sin(angle) * v^2
    def wing_coeff(name):
        w = _sec(root, name)
        if w is None:
            return 0.0
        area = _num(w, 'area', 0.0)
        ang = math.radians(_num(w, 'angle', 0.0))
        return 4.0 * RHO * area * math.sin(ang)
    kz_front = wing_coeff('Front Wing')
    kz_rear = wing_coeff('Rear Wing')

    # downforce per axle as a function of v^2
    df_front = cl_f * hm + kz_front     # N per (m/s)^2 on front axle
    df_rear = cl_r * hm + kz_rear       # N per (m/s)^2 on rear axle

    # static per-wheel load and load-sensitivity params (per wheel)
    W = mass * G
    static = {
        'f': 0.5 * frep * W,            # one front wheel
        'r': 0.5 * (1.0 - frep) * W,    # one rear wheel
    }
    wheel_secs = {'f': 'Front Right Wheel', 'r': 'Rear Right Wheel'}
    tyre = {}
    for k, wn in wheel_secs.items():
        ws = _sec(root, wn)
        mu = _num(ws, 'mu', 1.0)
        camber = math.radians(_num(ws, 'camber', 0.0))
        lfMin, lfMax = 0.8, 1.6         # TORCS clamps: min<=0.8, max>=1.6
        opLoad = 1.2 * static[k]
        lfK = math.log((1.0 - lfMin) / (lfMax - lfMin))
        cam = 1.0 + 0.05 * math.sin(-camber * 18.0)
        tyre[k] = dict(mu=mu, lfMin=lfMin, lfMax=lfMax, opLoad=opLoad, lfK=lfK,
                       cam=cam, static=static[k])

    def wheel_grip(k, Fz):
        t = tyre[k]
        mu_eff = t['mu'] * (t['lfMin'] + (t['lfMax'] - t['lfMin']) *
                            math.exp(t['lfK'] * Fz / t['opLoad']))
        return Fz * mu_eff * kfriction * t['cam']

    # engine torque curve
    eng = _sec(root, 'Engine', 'data points')
    rpms, tqs = [], []
    i = 1
    while True:
        pt = _sec(_sec(root, 'Engine', 'data points'), str(i))
        if pt is None:
            break
        rpms.append(_num(pt, 'rpm'))
        tqs.append(_num(pt, 'Tq'))
        i += 1
    rpms, tqs = np.array(rpms), np.array(tqs)
    rev_limit = _num(_sec(root, 'Engine'), 'revs limiter', rpms[-1])

    # gearbox + final drive
    gb = _sec(root, 'Gearbox', 'gears')
    ratios, effs = [], []
    for gi in ('1', '2', '3', '4', '5', '6'):
        gs = _sec(gb, gi)
        if gs is None:
            break
        ratios.append(_num(gs, 'ratio'))
        effs.append(_num(gs, 'efficiency', 0.95))
    fd = _sec(root, 'Rear Differential')
    final_ratio = _num(fd, 'ratio', 1.0)
    final_eff = _num(fd, 'efficiency', 0.95)

    # wheel radii: rim/2 + tyre sidewall (front & rear differ slightly)
    def wheel_radius(name):
        ws = _sec(root, name)
        rim_m = _num(ws, 'rim diameter', 18.0) * 0.0254
        tw = _num(ws, 'tire width', 300.0) / 1000.0
        rhw = _num(ws, 'tire height-width ratio', 0.3)
        return rim_m / 2.0 + tw * rhw
    wheel_r = wheel_radius('Rear Right Wheel')      # driven wheels
    wheel_r_f = wheel_radius('Front Right Wheel')

    # brake system: torque per wheel = coeff * pressure,
    #   coeff = diam*0.5*area*mu ; pressure = cmd * maxpress * repartition.
    # The front wheels (bigger pistons + more bias) lock first, so they set the
    # usable brake command. Commanding more than the lock threshold just locks
    # the fronts -> no braking grip AND no steering (the car ploughs off).
    bsys = _sec(root, 'Brake System')
    brk_press = _num(bsys, 'max pressure', 20000.0) * 1000.0      # kPa -> Pa
    brk_rep = _num(bsys, 'front-rear brake repartition', 0.5)

    def brake_coeff(name):
        b = _sec(root, name)
        diam = _num(b, 'disk diameter', 300.0) / 1000.0
        area = _num(b, 'piston area', 25.0) / 1e4                 # cm^2 -> m^2
        mub = _num(b, 'mu', 0.3)
        return diam * 0.5 * area * mub
    bc_f = brake_coeff('Front Right Brake')
    bc_r = brake_coeff('Rear Right Brake')

    # ---- sweep speed ----
    vs = np.arange(1.0, v_max + dv, dv)
    a_lat = np.zeros_like(vs)
    a_brake = np.zeros_like(vs)
    a_accel = np.zeros_like(vs)
    brake_cmd_max = np.zeros_like(vs)
    best_gear = np.zeros_like(vs, dtype=int)
    for j, v in enumerate(vs):
        v2 = v * v
        Fz_f = static['f'] + 0.5 * df_front * v2
        Fz_r = static['r'] + 0.5 * df_rear * v2
        gf = wheel_grip('f', Fz_f)
        gr = wheel_grip('r', Fz_r)
        grip = 2.0 * gf + 2.0 * gr
        drag = SCx2 * v2
        a_lat[j] = grip / mass                       # pure cornering
        a_brake[j] = grip / mass + drag / mass       # braking: tyres + drag

        # brake command at which each axle's brake force == its grip (lock):
        #   brake_force = coeff * (cmd * maxpress * rep) / wheel_radius
        cf = gf * wheel_r_f / (bc_f * brk_press * brk_rep + 1e-9)
        cr = gr * wheel_r / (bc_r * brk_press * (1.0 - brk_rep) + 1e-9)
        brake_cmd_max[j] = float(np.clip(min(cf, cr), 0.05, 1.0))

        # acceleration: best gear's wheel force, traction-limited at the rear
        fbest = 0.0
        gbest = 1
        for gi, (gr, ef) in enumerate(zip(ratios, effs), start=1):
            rpm = (v / wheel_r) * gr * final_ratio * 60.0 / (2.0 * math.pi)
            if rpm > rev_limit * 1.02:
                continue
            tq = float(np.interp(rpm, rpms, tqs))
            Fdrive = tq * gr * final_ratio * ef * final_eff / wheel_r
            if Fdrive > fbest:
                fbest = Fdrive
                gbest = gi
        traction = 2.0 * wheel_grip('r', Fz_r)       # RWD: rear tyres
        Fnet = min(fbest, traction) - drag
        a_accel[j] = max(Fnet / mass, 0.0)
        best_gear[j] = gbest

    # top speed: where drive force == drag
    top = float(vs[np.argmax(a_accel <= 0.02)]) if np.any(a_accel <= 0.02) else v_max

    # optimal up-shift speeds: where best gear index increments
    shift_speeds = []
    for gi in range(1, len(ratios)):
        idx = np.where(best_gear >= gi + 1)[0]
        if idx.size:
            shift_speeds.append(round(float(vs[idx[0]]), 1))

    profile = dict(
        car=_str(car, 'category', 'unknown') or 'car',
        source=car_xml, kfriction=kfriction, fuel_kg=fuel_kg,
        mass=mass, wheel_radius=round(wheel_r, 4),
        final_ratio=final_ratio, gear_ratios=ratios,
        hm=round(hm, 4), df_front=round(df_front, 4), df_rear=round(df_rear, 4),
        SCx2=round(SCx2, 4),
        v=vs.tolist(),
        a_lat=np.round(a_lat, 3).tolist(),
        a_brake=np.round(a_brake, 3).tolist(),
        a_accel=np.round(a_accel, 3).tolist(),
        brake_cmd_max=np.round(brake_cmd_max, 3).tolist(),
        shift_speeds=shift_speeds, top_speed=round(top, 1),
    )
    return profile


class CarProfile:
    """Loads a car_profile.json and serves the physics-exact, speed-dependent
    limits. `max_corner_speed(kappa)` inverts v^2 = a_lat(v)/kappa (a_lat grows
    with v via downforce, so this is solved through a lookup table)."""

    def __init__(self, d):
        self.v = np.asarray(d['v'], dtype=float)
        self.alat = np.asarray(d['a_lat'], dtype=float)
        self.abrake = np.asarray(d['a_brake'], dtype=float)
        self.aaccel = np.asarray(d['a_accel'], dtype=float)
        bcm = d.get('brake_cmd_max')
        self.bcmax = np.asarray(bcm, dtype=float) if bcm is not None \
            else np.ones_like(self.v)
        self.top_speed = float(d.get('top_speed', self.v[-1]))
        self.shift_speeds = d.get('shift_speeds', [])
        self.wheel_radius = float(d.get('wheel_radius', 0.3276))
        self.meta = d
        # kappa a car can hold at speed v: kappa_of_v = a_lat(v)/v^2 (decreasing
        # in v); reverse so it is increasing for np.interp.
        kov = self.alat / (self.v * self.v)
        self._kv = kov[::-1].copy()        # increasing kappa
        self._vv = self.v[::-1].copy()     # decreasing v

    @classmethod
    def load(cls, path):
        import os
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return cls(json.load(f))
        except Exception as e:
            print("Could not load car profile %s: %s" % (path, e))
            return None

    def a_lat(self, v):
        return np.interp(v, self.v, self.alat)

    def a_brake(self, v):
        return np.interp(v, self.v, self.abrake)

    def a_accel(self, v):
        return np.interp(v, self.v, self.aaccel)

    def max_brake_cmd(self, v):
        """Largest brake command that does not lock the (first-to-lock) wheels,
        so braking stays grip-limited and the fronts keep steering."""
        return float(np.interp(v, self.v, self.bcmax))

    def max_corner_speed(self, kappa):
        """Grip-limited corner speed for curvature kappa (scalar or array),
        including the downforce that makes fast corners grippier."""
        k = np.abs(kappa) + 1e-9
        return np.interp(k, self._kv, self._vv)   # clamps outside the table


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    car_xml = argv[0]
    kfric, fuel, out = 1.1, 40.0, 'car_profile.json'
    i = 1
    while i < len(argv):
        if argv[i] == '--kfric':
            kfric = float(argv[i + 1]); i += 2
        elif argv[i] == '--fuel':
            fuel = float(argv[i + 1]); i += 2
        elif argv[i] == '--out':
            out = argv[i + 1]; i += 2
        else:
            i += 1
    p = build_profile(car_xml, kfriction=kfric, fuel_kg=fuel)
    with open(out, 'w') as f:
        json.dump(p, f)
    v = np.array(p['v'])
    print("Car: %s  mass=%.0f kg (fuel %.0f)  wheel_r=%.3f m  kFriction=%.2f"
          % (p['car'], p['mass'], p['fuel_kg'], p['wheel_radius'], p['kfriction']))
    print("Lateral grip a_lat:   %.1f m/s^2 @ 10 m/s   %.1f @ 30   %.1f @ 50   %.1f @ 70"
          % tuple(np.interp([10, 30, 50, 70], v, p['a_lat'])))
    print("Braking   a_brake:    %.1f m/s^2 @ 10        %.1f @ 30   %.1f @ 50   %.1f @ 70"
          % tuple(np.interp([10, 30, 50, 70], v, p['a_brake'])))
    print("Accel     a_accel:    %.1f m/s^2 @ 10        %.1f @ 30   %.1f @ 50   %.1f @ 70"
          % tuple(np.interp([10, 30, 50, 70], v, p['a_accel'])))
    print("Up-shift speeds (m/s):", p['shift_speeds'])
    print("Top speed: %.1f m/s (%.0f km/h)" % (p['top_speed'], p['top_speed'] * 3.6))
    print("saved -> %s" % out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
