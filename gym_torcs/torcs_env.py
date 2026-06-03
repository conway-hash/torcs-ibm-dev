# torcs_env.py — random_test
# ---------------------------------------------------------------------------
# Minimal, robust TORCS env for Evolution Strategies. No shaped reward (ES
# scores rollouts by actual lap time / distance), so there is nothing to
# reward-hack. Responsibilities:
#   * launch / reset / relaunch TORCS (crash + hang recovery)
#   * 33-dim normalized observation incl. MPC target-speed lookahead
#   * frame-skip + automatic RPM gearbox
#   * lap detection via distRaced (reliable in SCR mode), self-timed
#   * off-track / stall / backwards / timeout termination
# ---------------------------------------------------------------------------
import json
import math
import os
import time
import numpy as np

from config import CFG
from scr_client import ScrClient
from launcher import TorcsInstance


def _load_mpc_speed_profile(cfg):
    try:
        if not cfg.mpc_track_map or not os.path.exists(cfg.mpc_track_map):
            return None, None
        with open(cfg.mpc_track_map) as f:
            m = json.load(f)
        return (np.array(m['s'], dtype=np.float32),
                np.array(m['v_target'], dtype=np.float32))
    except Exception:
        return None, None


def _load_racing_line(cfg):
    """Load the racing-line arrays (station s, target speed, signed curvature,
    lateral offset, track width) used for the anticipatory lookahead features."""
    try:
        if not cfg.mpc_track_map or not os.path.exists(cfg.mpc_track_map):
            return None
        with open(cfg.mpc_track_map) as f:
            m = json.load(f)
        return dict(
            s=np.array(m['s'], dtype=np.float32),
            v=np.array(m['v_target'], dtype=np.float32),
            kappa=np.array(m['kappa_rl'], dtype=np.float32),
            n=np.array(m['n_rl'], dtype=np.float32),
            width=np.array(m['width'], dtype=np.float32),
        )
    except Exception:
        return None


class TorcsEnv:
    def __init__(self, idx, cfg=CFG, gui=False, attach=False):
        self.cfg = cfg
        self.idx = idx
        self.port = cfg.port_for(idx)
        self.gui = gui
        # attach mode: connect to an externally-launched TORCS; never launch,
        # relaunch, or kill the process (the user owns it).
        self.attach = attach
        self.instance = None if attach else TorcsInstance(idx, cfg, gui=gui)
        self.client = ScrClient(cfg.host, self.port, cfg.track_sensor_angles)
        self.obs_dim = cfg.obs_dim
        self.act_dim = cfg.act_dim

        self._mpc_s, self._mpc_v = _load_mpc_speed_profile(cfg)
        self._mpc_lookahead = list(cfg.mpc_lookahead_m)
        # Wrap the speed-profile lookup over the track_map's OWN length (its
        # last station), not track_length_m -- the two differ by ~41 m and the
        # speed array is indexed by the map's stations.
        self._mpc_L = (float(self._mpc_s[-1]) if self._mpc_s is not None
                       else cfg.track_length_m)
        # Racing-line lookahead (anticipatory features appended to the obs).
        self._rl = _load_racing_line(cfg)
        self._map_la_dists = list(cfg.map_lookahead_m)   # NOT _map_lookahead:
                                                         # that name is the method

        self._gear = 1
        self._clutch = 0.0
        self._step = 0
        self._stall_ctr = 0
        self._back_ctr  = 0
        self._last_dist = 0.0
        self._sim_time  = 0.0
        self._steer_prev = 0.0
        self._launched  = False
        self._episode_count = 0

    # ---------------------------------------------------------------- obs
    def _mpc_speeds(self, dist):
        n = len(self._mpc_lookahead)
        if self._mpc_s is None:
            return np.zeros(n, dtype=np.float32)
        L, norm = self._mpc_L, self.cfg.norm_mpc_speed
        pos = float(dist) % L
        out = np.empty(n, dtype=np.float32)
        for i, la in enumerate(self._mpc_lookahead):
            out[i] = float(np.interp((pos + la) % L,
                                     self._mpc_s, self._mpc_v)) / norm
        return out

    def _map_lookahead(self, dist):
        """Anticipatory racing-line features at each map_lookahead_m distance
        ahead: (target speed, signed curvature, lateral offset as a fraction of
        half-width). 3 features per point. This is the MPC's plan, fed to the
        reactive net so it can brake/turn ahead of time."""
        m = 3 * len(self._map_la_dists)
        if self._rl is None:
            return np.zeros(m, dtype=np.float32)
        c = self.cfg
        L = self._mpc_L
        s, v, kap, n, w = (self._rl['s'], self._rl['v'], self._rl['kappa'],
                           self._rl['n'], self._rl['width'])
        pos = float(dist) % L
        out = np.empty(m, dtype=np.float32)
        for i, la in enumerate(self._map_la_dists):
            p = (pos + la) % L
            vt = float(np.interp(p, s, v)) / c.norm_mpc_speed
            kt = np.clip(float(np.interp(p, s, kap)) / c.norm_kappa, -1.5, 1.5)
            half = max(1.0, 0.5 * float(np.interp(p, s, w)))
            nt = np.clip(float(np.interp(p, s, n)) / half, -1.5, 1.5)
            out[3*i:3*i+3] = (vt, kt, nt)
        return out

    def _normalize(self, S):
        c = self.cfg
        track = np.asarray(S.get('track', [0.0]*19), dtype=np.float32)
        if track.shape[0] != 19:
            track = np.zeros(19, dtype=np.float32)
        wsv = np.asarray(S.get('wheelSpinVel', [0.0]*4), dtype=np.float32)
        if wsv.shape[0] != 4:
            wsv = np.zeros(4, dtype=np.float32)
        dist = float(S.get('distRaced', self._last_dist))
        obs = np.concatenate([
            track / c.norm_track,
            np.array([np.clip(S.get('trackPos', 0.0), -2.0, 2.0)], np.float32),
            np.array([S.get('angle', 0.0) / c.norm_angle], np.float32),
            np.array([S.get('speedX', 0.0) / c.norm_speed,
                      S.get('speedY', 0.0) / c.norm_speed,
                      S.get('speedZ', 0.0) / c.norm_speed], np.float32),
            np.array([S.get('rpm', 0.0) / c.norm_rpm], np.float32),
            np.array([float(self._gear) / c.norm_gear], np.float32),
            wsv / c.norm_wheelspin,
            self._mpc_speeds(dist),        # ---- first 33 dims (unchanged) ----
            self._map_lookahead(dist),     # ---- 18 NEW anticipatory dims -----
        ]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------- gearbox
    def _auto_gear(self, S):
        """Ported verbatim from torcs_mpc (copy) _gear_clutch so the env's
        gearbox + launch clutch match what the MPC uses to lap ~89-100s --
        otherwise a weak shifter caps the lap time the policy can reach."""
        c = self.cfg
        rpm = S.get('rpm', 0.0)
        v = S.get('speedX', 0.0) / 3.6
        gear = int(S.get('gear', self._gear)) or 1
        if gear < 1:
            gear = 1
        # speed-derived ceiling: highest gear whose speed window we're within
        cap = c.gear_max
        for i, vmax in enumerate(c.gear_speed_cap):
            if v < vmax:
                cap = i + 1
                break
        if rpm > c.rpm_up and gear < cap:
            gear += 1
        elif gear > cap:
            gear = cap                       # forced downshift when slowed
        elif rpm < c.rpm_down and gear > 1:
            gear -= 1
        gear = max(1, min(gear, cap))
        clutch = 0.0
        if v < 5.0:                          # launch clutch ramp
            clutch = max(0.0, min(0.5, 0.4 - v * 0.08))
        self._gear = gear
        self._clutch = clutch
        return gear

    # --------------------------------------------------------------- reset
    def reset(self, relaunch=False):
        c = self.cfg
        self._gear = 1
        self._clutch = 0.0
        self._step = 0
        self._stall_ctr = 0
        self._back_ctr  = 0
        self._last_dist = 0.0
        self._sim_time  = 0.0
        self._steer_prev = 0.0
        if self.attach:
            return self._attach_reset()
        # Periodic hard relaunch to clear long-run TORCS state drift / leaks.
        self._episode_count += 1
        if (c.relaunch_every_episodes and
                self._episode_count % c.relaunch_every_episodes == 0):
            relaunch = True
        if relaunch:
            self.instance.kill()
            self.client.shutdown()
            self._launched = False

        startup_sleep = 10.0 if self.gui else 3.5   # more cold-start margin
        for attempt in range(c.max_relaunch_tries):
            try:
                if not self._launched or not self.instance.is_alive():
                    self.instance.kill()
                    self.client.shutdown()
                    self.instance.launch(wait_handshake=False)
                    self._launched = True
                    time.sleep(startup_sleep)
                else:
                    # reuse running TORCS: ask scr_server to restart the race
                    if self.client.so:
                        self.client.send_restart()
                        for _ in range(30):
                            st = self.client.get_servers_input(0.5, 4.0)
                            if st in ('restart', 'shutdown'):
                                break
                    time.sleep(0.8)

                if not self.client.connect(c.handshake_timeout_s,
                                           c.handshake_attempts):
                    raise ConnectionError("handshake failed")

                for _ in range(40):
                    st = self.client.get_servers_input(c.socket_timeout_s,
                                                       c.stale_relaunch_s)
                    if st == 'ok':
                        S = self.client.S.d
                        self.client.R.d['gear'] = 1
                        self.client.respond_to_server()
                        self._last_dist = S.get('distRaced', 0.0)
                        return self._normalize(S)
                    elif st in ('shutdown', 'restart'):
                        break
            except Exception:
                self.instance.kill()
                self.client.shutdown()
                self._launched = False
                time.sleep(1.5)
        raise RuntimeError("TorcsEnv idx=%d failed to reset" % self.idx)

    def _attach_reset(self):
        """Connect to an externally-launched TORCS race (we never own/launch the
        process). Retries patiently so you can start the race after launching
        the script, and uses meta=1 to restart between episodes."""
        c = self.cfg
        deadline = time.time() + 90.0     # generous: time to start the race
        announced = False
        while time.time() < deadline:
            try:
                # If already connected (e.g. a previous episode), ask TORCS to
                # restart the race for a fresh lap.
                if self.client.so:
                    self.client.send_restart()
                    for _ in range(30):
                        st = self.client.get_servers_input(0.5, 4.0)
                        if st in ('restart', 'shutdown'):
                            break
                    time.sleep(0.5)
                if self.client.connect(c.handshake_timeout_s,
                                       c.handshake_attempts):
                    for _ in range(60):
                        st = self.client.get_servers_input(c.socket_timeout_s,
                                                           c.stale_relaunch_s)
                        if st == 'ok':
                            S = self.client.S.d
                            self.client.R.d['gear'] = 1
                            self.client.respond_to_server()
                            self._last_dist = S.get('distRaced', 0.0)
                            return self._normalize(S)
                        elif st in ('shutdown', 'restart'):
                            break
            except Exception:
                pass
            if not announced:
                print('[attach] waiting for TORCS on port %d — start the race '
                      'now (scr_server, corkscrew)...' % self.port, flush=True)
                announced = True
            self.client.shutdown()
            time.sleep(1.0)
        raise RuntimeError(
            "attach: no TORCS race answered on port %d within 90s. Make sure "
            "you started a race with the scr_server driver on that port."
            % self.port)

    # ---------------------------------------------------------------- step
    def _decode(self, action):
        a = np.asarray(action, dtype=np.float32).ravel()
        steer = float(np.clip(a[0], -1.0, 1.0))
        accel = float(np.clip((a[1] + 1.0) * 0.5, 0.0, 1.0))
        brake = float(np.clip((a[2] + 1.0) * 0.5, 0.0, 1.0))
        return steer, accel, brake

    def step(self, action, action_fn=None):
        """action in [-1,1]^3. If action_fn(S)->action is given, it re-decides
        every tick (tight control for a high-rate expert under frame-skip);
        the stored `action` still labels the step for any recorder."""
        c = self.cfg
        steer, accel, brake = self._decode(action)
        done = False
        info = {}
        last_obs = self._normalize(self.client.S.d)

        for _ in range(c.frame_skip):
            S = self.client.S.d
            if action_fn is not None:
                fresh = action_fn(S)
                if fresh is not None:
                    steer, accel, brake = self._decode(fresh)
            self._auto_gear(S)
            # Steering low-pass (the MPC's anti-oscillation trick). a=0 -> off.
            a = getattr(c, 'steer_lp_alpha', 0.0)
            if a > 0.0:
                steer = a * self._steer_prev + (1.0 - a) * steer
            self._steer_prev = steer
            R = self.client.R.d
            R['steer'], R['accel'], R['brake'] = steer, accel, brake
            R['gear'], R['clutch'] = self._gear, self._clutch
            self.client.respond_to_server()
            st = self.client.get_servers_input(c.socket_timeout_s,
                                               c.stale_relaunch_s)
            if st in ('shutdown', 'restart', 'timeout'):
                info['crashed'] = True
                self._launched = False
                if self.instance is not None:      # not in attach mode
                    self.instance.kill()
                done = True
                break
            S = self.client.S.d
            term, tinfo = self._terminal(S)
            info.update(tinfo)
            last_obs = self._normalize(S)
            if term:
                done = True
                break

        self._step += 1
        if not done and self._step >= c.max_steps:
            done = True
            info['timeout'] = True
        info['dist_raced'] = float(self._last_dist)
        return last_obs, 0.0, done, info

    # ------------------------------------------------------ terminal/timing
    def _terminal(self, S):
        c = self.cfg
        self._sim_time += c.sim_dt
        angle    = S.get('angle', 0.0)
        trackPos = S.get('trackPos', 0.0)
        track    = S.get('track', [0.0]*19)
        track_min = min(track) if isinstance(track, list) and track else 0.0
        dist = S.get('distRaced', self._last_dist)
        self._last_dist = dist
        info = {}
        term = False

        if track_min < 0.0 or abs(trackPos) > c.offtrack_trackpos:
            term = True
            info['offtrack'] = True

        if not term and dist >= c.track_length_m:
            info['lap_time'] = float(self._sim_time)
            info['lap_done'] = True
            term = True

        if self._step > c.start_grace_steps:
            if abs(S.get('speedX', 0.0)) < c.stall_speed_kmh:
                self._stall_ctr += 1
            else:
                self._stall_ctr = 0
            if math.cos(angle) < 0.0:
                self._back_ctr += 1
            else:
                self._back_ctr = 0
            if (self._stall_ctr >= c.stall_steps or
                    self._back_ctr >= c.backward_steps):
                term = True
                info['stall'] = True

        return term, info

    def close(self):
        self.client.shutdown()
        if self.instance is not None:      # leave an attached TORCS running
            self.instance.kill()
