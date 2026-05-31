import subprocess
import pyautogui
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import snakeoil3_gym as snakeoil3
import copy
import collections as col
import os
import time

TORCS_EXE = r'C:\torcs\torcs\torcs\wtorcs.exe'
TORCS_LOAD_WAIT = 10  # seconds to wait for main menu
TORCS_RACE_WAIT = 8   # seconds to wait for track to load after menu


def _launch_torcs():
    subprocess.Popen(TORCS_EXE, cwd=os.path.dirname(TORCS_EXE))
    time.sleep(TORCS_LOAD_WAIT)
    pyautogui.press('enter')
    time.sleep(0.3)
    pyautogui.press('enter')
    time.sleep(0.3)
    pyautogui.press('enter')
    time.sleep(TORCS_RACE_WAIT)


def _kill_torcs():
    os.system('taskkill /f /im wtorcs.exe >nul 2>&1')
    time.sleep(1.0)


class TorcsEnv:
    terminal_judge_start = 150
    termination_limit_progress = 2
    default_speed = 50

    initial_reset = True

    def __init__(self, vision=False, throttle=False, gear_change=False):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.initial_run = True

        _kill_torcs()
        _launch_torcs()

        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf], dtype=np.float32)
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf], dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255], dtype=np.float32)
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0], dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, u):
        client = self.client

        this_action = self.agent_to_torcs(u)
        action_torcs = client.R.d

        action_torcs['steer'] = this_action['steer']  # [-1, 1]

        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer'] * 50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1 / (client.S.d['speedX'] + .1)

            if ((client.S.d['wheelSpinVel'][2] + client.S.d['wheelSpinVel'][3]) -
                    (client.S.d['wheelSpinVel'][0] + client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = (this_action['accel'] + 1.0) / 2.0
            action_torcs['brake'] = max(0.0, this_action['brake'])

        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            action_torcs['gear'] = 1
            if client.S.d['speedX'] > 35:
                action_torcs['gear'] = 2
            if client.S.d['speedX'] > 60:
                action_torcs['gear'] = 3
            if client.S.d['speedX'] > 90:
                action_torcs['gear'] = 4
            if client.S.d['speedX'] > 120:
                action_torcs['gear'] = 5
            if client.S.d['speedX'] > 150:
                action_torcs['gear'] = 6

        prev_steer = float(self.last_u[0]) if self.last_u is not None else float(u[0])
        self.last_u = u

        obs_pre = copy.deepcopy(client.S.d)

        client.respond_to_server()
        client.get_servers_input()

        if client.so is None:
            self.observation = self.make_observaton(obs_pre)
            finish_bonus = 5000.0 + (50000.0 / max(self.time_step, 1))
            race_time = obs_pre.get('curLapTime', 0.0)
            dist = float(obs_pre.get('distFromStart', 0.0))
            return self.get_obs(), finish_bonus, True, False, {
                'term_reason': 'finished', 'time': race_time, 'dist_from_start': dist}

        obs = client.S.d
        self.observation = self.make_observaton(obs)
        dist = float(obs.get('distFromStart', 0.0))

        if obs.get('lastLapTime', 0) > 0:
            lap_time = obs['lastLapTime']
            print(f"### LAP FINISHED in {lap_time:.1f}s ###")
            # Strong time-based finish bonus: faster laps earn far more.
            # Rewards optimizing lap TIME, not just completion.
            finish_bonus = 5000.0 + (80000.0 / max(self.time_step, 1))
            client.R.d['meta'] = True
            client.respond_to_server()
            return self.get_obs(), finish_bonus, True, False, {
                'term_reason': 'finished', 'time': lap_time, 'dist_from_start': dist}

        track    = np.array(obs['track'])
        sp       = np.array(obs['speedX'])
        progress = sp * np.cos(obs['angle'])

        # ── Reward: go FAST, go forward, don't drift ─────────────────
        # No trackPos penalty — the car should use the full track width.
        # Good racing lines hug the edge; only punish leaving the track entirely.
        # Speed term weighted up (1.5x) to push the car to maximize pace and
        # break out of the "cozy slow lap" attractor. The drift penalty
        # (lateral velocity) discourages sliding/scrubbing speed in corners.
        reward = 1.5 * sp * np.cos(obs['angle']) - sp * abs(np.sin(obs['angle']))

        # Removed the flat per-step survival bonus — it rewarded plodding
        # slowly to rack up steps. Replaced with a speed-proportional on-track
        # bonus so staying on track AND being fast is what pays.
        if track.min() > 0:
            reward += 0.5 + 0.04 * sp

        # Edge-correction nudge: only when very close to leaving the track,
        # gently reward steering/being back toward center. This targets the
        # recurring off_track failure without penalizing racing-line use.
        tp = abs(float(obs['trackPos']))
        if tp > 0.85:
            reward -= (tp - 0.85) * 20.0

        # Steering-smoothness penalty: punish abrupt steering changes to
        # reduce wobble/oscillation. Scaled by speed so it matters most at
        # high speed where wobble causes off_track. Slightly reduced base
        # coefficient so it does not suppress legitimate fast racing-line
        # corrections needed for an aggressive line.
        steer_delta = abs(float(u[0]) - prev_steer)
        reward -= steer_delta * (2.0 + 0.05 * sp)

        # Discourage near-zero speed (prevents stall attractor)
        if sp < 0.3:
            reward -= 4.0

        # Clip to keep Q-values stable
        reward = float(np.clip(reward, -50.0, 60.0))

        term_reason = None

        if obs['damage'] - obs_pre['damage'] > 0:
            reward -= 40
            client.R.d['meta'] = True
            term_reason = 'damage'

        episode_terminate = False
        if track.min() < 0:
            reward -= 60
            episode_terminate = True
            client.R.d['meta'] = True
            term_reason = 'off_track'

        if self.terminal_judge_start < self.time_step:
            if progress < self.termination_limit_progress:
                reward -= 40
                episode_terminate = True
                client.R.d['meta'] = True
                term_reason = 'too_slow'

        if np.cos(obs['angle']) < 0:
            reward -= 60
            episode_terminate = True
            client.R.d['meta'] = True
            term_reason = 'backwards'

        if client.R.d['meta'] is True:
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], False, {
            'term_reason': term_reason, 'time': obs.get('curLapTime', 0.0),
            'dist_from_start': dist}

    def reset(self, relaunch=False, seed=None, options=None):
        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            if relaunch is True:
                self.reset_torcs()
                print("### TORCS RELAUNCHED ###")

        self.client = snakeoil3.Client(p=3001, vision=self.vision)
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()

        obs = client.S.d
        self.observation = self.make_observaton(obs)
        self.last_u = None
        self.initial_reset = False

        return self.get_obs(), {}

    def end(self):
        if hasattr(self, 'client') and self.client:
            self.client.shutdown()
        _kill_torcs()

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        _kill_torcs()
        _launch_torcs()

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}
        if self.throttle is True:
            torcs_action.update({'accel': u[1], 'brake': u[2]})
        if self.gear_change is True:
            torcs_action.update({'gear': u[3]})
        return torcs_action

    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec = obs_image_vec
        rgb = []
        temp = []
        for i in range(0, 12286, 3):
            temp.append(image_vec[i])
            temp.append(image_vec[i + 1])
            temp.append(image_vec[i + 2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus', 'speedX', 'speedY', 'speedZ', 'opponents', 'rpm', 'track', 'wheelSpinVel', 'angle', 'trackPos']
            Observation = col.namedtuple('Observaion', names)
            return Observation(
                focus=np.array(raw_obs['focus'], dtype=np.float32) / 200.,
                speedX=np.array(raw_obs['speedX'], dtype=np.float32) / self.default_speed,
                speedY=np.array(raw_obs['speedY'], dtype=np.float32) / self.default_speed,
                speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / self.default_speed,
                opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
                rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                track=np.array(raw_obs['track'], dtype=np.float32) / 200.,
                wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                angle=np.array(raw_obs['angle'], dtype=np.float32) / 3.14159,
                trackPos=np.array(raw_obs['trackPos'], dtype=np.float32))
        else:
            names = ['focus', 'speedX', 'speedY', 'speedZ', 'opponents', 'rpm', 'track', 'wheelSpinVel', 'img']
            Observation = col.namedtuple('Observaion', names)
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])
            return Observation(
                focus=np.array(raw_obs['focus'], dtype=np.float32) / 200.,
                speedX=np.array(raw_obs['speedX'], dtype=np.float32) / self.default_speed,
                speedY=np.array(raw_obs['speedY'], dtype=np.float32) / self.default_speed,
                speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / self.default_speed,
                opponents=np.array(raw_obs['opponents'], dtype=np.float32) / 200.,
                rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                track=np.array(raw_obs['track'], dtype=np.float32) / 200.,
                wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                img=image_rgb)