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
    # Navigate: Enter (Race) -> Enter (Quick Race) -> Enter (New Race)
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
    terminal_judge_start = 100
    termination_limit_progress = 5   # [km/h]
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
            # actor outputs [-1,1] via Tanh; remap to [0,1] for TORCS accel/brake
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

        # TORCS sent ***shutdown*** — server ended race
        if client.so is None:
            self.observation = self.make_observaton(obs_pre)
            finish_bonus = 50000.0 + (5000000.0 / self.time_step)
            race_time = obs_pre.get('curLapTime', 0.0)
            return self.get_obs(), finish_bonus, True, False, {'term_reason': 'finished', 'time': race_time}

        obs = client.S.d
        self.observation = self.make_observaton(obs)

        # Lap complete via lastLapTime sensor
        if obs.get('lastLapTime', 0) > 0:
            lap_time = obs['lastLapTime']
            print(f"### LAP FINISHED in {lap_time:.1f}s ###")
            finish_bonus = 50000.0 + (5000000.0 / self.time_step)
            client.R.d['meta'] = True
            client.respond_to_server()
            return self.get_obs(), finish_bonus, True, False, {'term_reason': 'finished', 'time': lap_time}

        track = np.array(obs['track'])
        sp = np.array(obs['speedX'])
        sp_y = np.array(obs['speedY'])
        progress = sp * np.cos(obs['angle'])

        # Gentle smoothness nudge — discourages oscillation without blocking cornering
        steer_delta = abs(float(u[0]) - prev_steer)
        smoothness_penalty = 0.5 * steer_delta

        reward = progress + 0.1 * sp - 0.5 * abs(sp_y) - 2.0 - smoothness_penalty
        term_reason = None

        if obs['damage'] - obs_pre['damage'] > 0:
            reward -= 1000
            client.R.d['meta'] = True
            term_reason = 'damage'

        episode_terminate = False
        if track.min() < 0:
            reward -= 1000
            episode_terminate = True
            client.R.d['meta'] = True
            term_reason = 'off_track'

        if self.terminal_judge_start < self.time_step:
            if progress < self.termination_limit_progress:
                reward -= 1000
                episode_terminate = True
                client.R.d['meta'] = True
                term_reason = 'too_slow'

        if np.cos(obs['angle']) < 0:
            reward -= 1000
            episode_terminate = True
            client.R.d['meta'] = True
            term_reason = 'backwards'

        if client.R.d['meta'] is True:
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], False, {'term_reason': term_reason, 'time': obs.get('curLapTime', 0.0)}

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
