import argparse
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import random
import os
import time as _time

from gym_torcs import TorcsEnv

# ── Hyperparameters ────────────────────────────────────────────────
STATE_DIM      = 29
ACTION_DIM     = 3
MAX_EPISODES   = 100_000
MAX_STEPS      = 14_000
BATCH_SIZE     = 256
BUFFER_SIZE    = 500_000
GAMMA          = 0.99
TAU            = 0.005
ACTOR_LR       = 2e-5
CRITIC_LR      = 2e-5
POLICY_NOISE   = 0.10
NOISE_CLIP     = 0.20
POLICY_DELAY   = 2
WARMUP_STEPS   = 5_000
SEED_STEPS     = 8_000
EXPL_NOISE     = 0.03
RELAUNCH_EVERY = 20
TRAIN_FREQ     = 2
ACCEL_FLOOR    = 0.15   # TORCS accel min → actor-space: 2*0.15-1 = -0.70

# Episode model storage
EPISODE_DIR    = 'models/episodes'
SAVE_EVERY     = 1          # save every N episodes (1 = every episode)
KEEP_LAST_N    = 0          # 0 = keep all; N > 0 = delete oldest beyond N

# Control flags written by auto_train.py
LOG_FILE       = 'training_log.txt'
STOP_FLAG      = 'stop.flag'
LOAD_FLAG      = 'load.flag'   # auto_train writes episode path here to hot-swap model

# Prioritized Experience Replay
PER_ALPHA          = 0.6
PER_BETA_START     = 0.4
PER_BETA_INCREMENT = 0.0001

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Noise ──────────────────────────────────────────────────────────
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.mu    = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state


def obs_to_state(obs):
    return np.hstack([
        np.atleast_1d(obs.track),
        np.atleast_1d(obs.speedX),
        np.atleast_1d(obs.speedY),
        np.atleast_1d(obs.speedZ),
        np.atleast_1d(obs.angle),
        np.atleast_1d(obs.trackPos),
        np.atleast_1d(obs.rpm) / 10000.0,
        np.atleast_1d(obs.wheelSpinVel) / 100.0,
    ]).astype(np.float32)


# ── Prioritized Replay Buffer ──────────────────────────────────────
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.size = 0
        self.ptr  = 0

    def _propagate(self, idx, delta):
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def _retrieve(self, idx, s):
        while True:
            left, right = 2 * idx + 1, 2 * idx + 2
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    def total(self):  return self.tree[0]

    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        self._propagate(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def get(self, s):
        s   = max(0.0, min(s, self.total() - 1e-8))
        idx = self._retrieve(0, s)
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]

    def clear(self):
        self.tree[:] = 0.0
        self.data = [None] * self.capacity
        self.size = 0
        self.ptr  = 0


class PrioritizedReplayBuffer:
    def __init__(self, max_size):
        self.tree         = SumTree(max_size)
        self.max_size     = max_size
        self.beta         = PER_BETA_START
        self.max_priority = 1.0
        self.epsilon      = 1e-6

    def add(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + PER_BETA_INCREMENT)
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(max(priority, self.epsilon))
        probs   = np.array(priorities) / self.tree.total()
        weights = (self.tree.size * probs) ** (-self.beta)
        weights /= weights.max()
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(DEVICE),
            torch.FloatTensor(np.array(a)).to(DEVICE),
            torch.FloatTensor(np.array(r)).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(np.array(s2)).to(DEVICE),
            torch.FloatTensor(np.array(d)).unsqueeze(1).to(DEVICE),
            idxs,
            torch.FloatTensor(weights).unsqueeze(1).to(DEVICE),
        )

    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            priority = (float(abs(td_error)) + self.epsilon) ** PER_ALPHA
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def clear(self):
        self.tree.clear()
        self.max_priority = 1.0
        self.beta = PER_BETA_START

    def __len__(self):
        return self.tree.size


# ── Networks ───────────────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400), nn.ReLU(),
            nn.Linear(400, 300),       nn.ReLU(),
            nn.Linear(300, action_dim), nn.Tanh(),
        )
    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400), nn.ReLU(),
            nn.Linear(400, 300),                    nn.ReLU(),
            nn.Linear(300, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400), nn.ReLU(),
            nn.Linear(400, 300),                    nn.ReLU(),
            nn.Linear(300, 1),
        )
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)
    def q1_only(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)


# ── TD3 Agent ──────────────────────────────────────────────────────
class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor        = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt    = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic        = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt    = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.total_it = 0

    def select_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        return self.actor(s).detach().cpu().numpy()[0]

    def train_step(self, replay_buffer):
        if len(replay_buffer) < BATCH_SIZE:
            return
        self.total_it += 1
        s, a, r, s2, d, idxs, weights = replay_buffer.sample(BATCH_SIZE)
        with torch.no_grad():
            noise  = (torch.randn_like(a) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_a = (self.actor_target(s2) + noise).clamp(-1, 1)
            q1_t, q2_t = self.critic_target(s2, next_a)
            q_target = r + GAMMA * (1 - d) * torch.min(q1_t, q2_t)
        q1, q2 = self.critic(s, a)
        td_errors = (q1.detach() - q_target).abs().cpu().numpy().flatten()
        replay_buffer.update_priorities(idxs, td_errors)
        critic_loss = (weights * (q1 - q_target).pow(2)).mean() + \
                      (weights * (q2 - q_target).pow(2)).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()
        if self.total_it % POLICY_DELAY == 0:
            actor_loss = -self.critic.q1_only(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_opt.step()
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),  f'{path}/actor.pth')
        torch.save(self.critic.state_dict(), f'{path}/critic.pth')

    def load(self, path):
        self.actor.load_state_dict(torch.load(f'{path}/actor.pth',  map_location=DEVICE))
        self.critic.load_state_dict(torch.load(f'{path}/critic.pth', map_location=DEVICE))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        print(f'  [loaded] {path}')


# ── Episode storage helpers ────────────────────────────────────────
import re as _re
import shutil as _shutil

STATE_FILE = os.path.join(EPISODE_DIR, '_state.json')


def _ep_name(ep_num, branch):
    return f'ep_{ep_num:06d}' if branch == 0 else f'ep_{ep_num:06d}.{branch}'


def _parse_ep_name(dirname):
    m = _re.match(r'^ep_(\d{6})(?:\.(\d+))?$', dirname)
    if not m:
        return None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else 0)


def _read_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {'next_ep': 0, 'branch': 0, 'last_dir': None}


def _write_state(s):
    os.makedirs(EPISODE_DIR, exist_ok=True)
    tmp = STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, STATE_FILE)


def _save_episode(agent, st, reward, avg10, term_reason, total_steps):
    ep_num = st['next_ep']
    branch = st['branch']
    if ep_num % SAVE_EVERY != 0:
        st['next_ep'] += 1
        return None
    name = _ep_name(ep_num, branch)
    path = os.path.join(EPISODE_DIR, name)
    agent.save(path)
    with open(f'{path}/info.json', 'w') as f:
        json.dump({'ep_num': ep_num, 'branch': branch,
                   'reward': round(reward, 2), 'avg10': round(avg10, 2),
                   'term_reason': term_reason, 'total_steps': total_steps}, f)
    st['next_ep'] += 1
    st['last_dir'] = path
    _write_state(st)
    if KEEP_LAST_N > 0:
        dirs = sorted(
            (p for d in os.listdir(EPISODE_DIR)
             if _parse_ep_name(d) and (p := os.path.join(EPISODE_DIR, d))),
            key=os.path.getmtime
        )
        for old in dirs[:-KEEP_LAST_N]:
            _shutil.rmtree(old, ignore_errors=True)
    return path


# ── Training Loop ──────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',      action='store_true')
    parser.add_argument('--load-from', type=str, default=None,
                        help='Episode dir to load, e.g. models/episodes/ep_000350')
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    if args.cuda and torch.cuda.is_available():
        DEVICE = torch.device('cuda')

    print(f'  [device] {DEVICE}')
    env    = TorcsEnv(throttle=True, gear_change=False)
    agent  = TD3(STATE_DIM, ACTION_DIM)
    buffer = PrioritizedReplayBuffer(BUFFER_SIZE)

    st          = _read_state()
    loaded_from = None

    if args.load_from:
        lp = args.load_from
        if os.path.exists(f'{lp}/actor.pth'):
            agent.load(lp)
            loaded_from = lp
            last = st.get('last_dir') or ''
            if os.path.normpath(lp) != os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), last)):
                st['branch'] += 1
            parsed = _parse_ep_name(os.path.basename(lp))
            if parsed:
                st['next_ep'] = parsed[0] + 1
            st['last_dir'] = lp
        else:
            print(f'  [warning] --load-from "{lp}" not found, falling through')

    if loaded_from is None and st.get('last_dir'):
        lp = st['last_dir']
        full = lp if os.path.isabs(lp) else os.path.join(WORK_DIR, lp)
        if os.path.exists(f'{full}/actor.pth'):
            agent.load(full)
            loaded_from = full

    session_line = (f'=== Session {_time.strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'loaded: {loaded_from or "fresh"} | '
                    f'next: {_ep_name(st["next_ep"], st["branch"])} ===')
    print(session_line)
    with open(LOG_FILE, 'a') as f:
        f.write(session_line + '\n')

    ou_steer = OUNoise(1, mu=0.0,  theta=0.15, sigma=0.10)
    ou_accel = OUNoise(1, mu=0.4,  theta=0.5,  sigma=0.10)
    ou_brake = OUNoise(1, mu=-0.9, theta=1.0,  sigma=0.02)

    accel_floor_raw  = 2.0 * ACCEL_FLOOR - 1.0
    total_steps      = 0
    episode_rewards  = []
    TRACK_LENGTH     = 3200.0
    track_len_locked = False

    for episode in range(MAX_EPISODES):

        if os.path.exists(STOP_FLAG):
            print('  [auto] stop.flag — saving and exiting')
            break

        if os.path.exists(LOAD_FLAG):
            with open(LOAD_FLAG) as _f:
                _swap_path = _f.read().strip()
            os.remove(LOAD_FLAG)
            if os.path.exists(f'{_swap_path}/actor.pth'):
                agent.load(_swap_path)
                buffer.clear()
                st['branch'] += 1
                parsed = _parse_ep_name(os.path.basename(_swap_path))
                if parsed:
                    st['next_ep'] = parsed[0] + 1
                st['last_dir'] = _swap_path
                episode_rewards = []
                swap_line = (f'  [ai-swap] loaded {_swap_path} | '
                             f'new branch={st["branch"]} | '
                             f'next={_ep_name(st["next_ep"], st["branch"])}')
                print(swap_line)
                with open(LOG_FILE, 'a') as f:
                    f.write(swap_line + '\n')
            else:
                print(f'  [ai-swap] path not found: {_swap_path}')

        relaunch = (episode % RELAUNCH_EVERY == 0)
        obs, _   = env.reset(relaunch=relaunch)
        state    = obs_to_state(obs)
        ou_steer.reset()
        ou_accel.reset()
        ou_brake.reset()

        episode_reward = 0
        step           = 0
        term_reason    = 'max_steps'
        ep_time        = 0.0
        dist_start     = None
        dist_covered   = 0.0

        for step in range(MAX_STEPS):
            if total_steps < WARMUP_STEPS:
                action = np.array([ou_steer.sample()[0],
                                   ou_accel.sample()[0],
                                   ou_brake.sample()[0]])
            else:
                action = agent.select_action(state)
                action[0] = np.clip(action[0] + ou_steer.sample()[0] * EXPL_NOISE, -1, 1)
                action[1] = np.clip(action[1] + ou_accel.sample()[0] * EXPL_NOISE, -1, 1)
                action[2] = np.clip(action[2] + ou_brake.sample()[0] * EXPL_NOISE, -1, 1)

            action[1] = max(action[1], accel_floor_raw)
            action    = np.clip(action, -1, 1)

            obs, reward, done, _, info = env.step(action)
            next_state = obs_to_state(obs)

            step_dist = info.get('dist_from_start', 0.0)
            if dist_start is None:
                dist_start = step_dist
            raw = step_dist - dist_start
            if raw < -100:
                raw += TRACK_LENGTH
            dist_covered = max(dist_covered, raw)

            buffer.add(state, action, reward, next_state, float(done))
            if total_steps > SEED_STEPS and total_steps % TRAIN_FREQ == 0:
                agent.train_step(buffer)

            state          = next_state
            episode_reward += reward
            total_steps    += 1

            if done:
                term_reason = info.get('term_reason', 'unknown')
                ep_time     = info.get('time', 0.0)
                break

        if term_reason == 'finished' and not track_len_locked:
            TRACK_LENGTH     = dist_covered
            track_len_locked = True
            tl_line = f'  [track] length measured by TORCS sensor: {TRACK_LENGTH:.0f}m'
            print(tl_line)
            with open(LOG_FILE, 'a') as f:
                f.write(tl_line + '\n')

        pct = dist_covered / TRACK_LENGTH * 100

        episode_rewards.append(episode_reward)
        avg10      = float(np.mean(episode_rewards[-10:]))
        mins, secs = divmod(ep_time, 60)
        buf_pct    = 100.0 * len(buffer) / BUFFER_SIZE
        ep_label   = _ep_name(st['next_ep'], st['branch'])

        line = (f'Ep {ep_label} | '
                f'Time {int(mins)}:{secs:05.2f} | '
                f'Steps {step+1:5d} | '
                f'Reward {episode_reward:9.1f} | '
                f'Avg(10) {avg10:9.1f} | '
                f'Dist {dist_covered:6.0f}m ({pct:5.1f}%) | '
                f'Buf {buf_pct:5.1f}% | '
                f'Total {total_steps:7d} | '
                f'End: {term_reason}')
        print(line)
        with open(LOG_FILE, 'a') as f:
            f.write(line + '\n')

        _save_episode(agent, st, episode_reward, avg10, term_reason, total_steps)

    env.end()
    print('Training complete.')