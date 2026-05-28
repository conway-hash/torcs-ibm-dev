import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os

from gym_torcs import TorcsEnv

# ── Hyperparameters ────────────────────────────────────────────────
STATE_DIM      = 24
ACTION_DIM     = 3
MAX_EPISODES   = 10000
MAX_STEPS      = 10000
BATCH_SIZE     = 256
BUFFER_SIZE    = 500_000
GAMMA          = 0.99
TAU            = 0.001
ACTOR_LR       = 1e-6
CRITIC_LR      = 1e-6
POLICY_NOISE   = 0.08
NOISE_CLIP     = 0.15
POLICY_DELAY   = 3
WARMUP_STEPS   = 0
SEED_STEPS     = 20_000
EXPL_NOISE     = 0.03
RELAUNCH_EVERY = 20
MODEL_DIR      = 'models'
LOG_FILE       = 'training_log.txt'
TRAIN_FREQ     = 4       # train once per N env steps (reduces update rate)
ROLLBACK_SEED  = 20_000  # steps without training after rollback (rebuild buffer)

# Prioritized Experience Replay
PER_ALPHA          = 0.6    # prioritisation (0=uniform, 1=full)
PER_BETA_START     = 0.4    # IS correction start, anneals to 1.0
PER_BETA_INCREMENT = 0.0001 # per training step

# Auto-rollback: if Avg(10) drops this fraction below its peak, reload best-avg checkpoint
ROLLBACK_DROP     = 0.35
ROLLBACK_COOLDOWN = 25      # min episodes between rollbacks

DEVICE = torch.device('cpu')


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
    ]).astype(np.float32)


# ── Prioritized Replay Buffer ──────────────────────────────────────
class SumTree:
    """Binary segment tree for O(log N) priority sampling."""
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

    def total(self):
        return self.tree[0]

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

    def train(self, replay_buffer):
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

        # Update PER priorities before backward pass
        td_errors = (q1.detach() - q_target).abs().cpu().numpy().flatten()
        replay_buffer.update_priorities(idxs, td_errors)

        # Importance-weighted critic loss
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

    def save(self, path=MODEL_DIR, quiet=False):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),  f'{path}/actor.pth')
        torch.save(self.critic.state_dict(), f'{path}/critic.pth')
        if not quiet:
            print(f"  [saved] {path}/")

    def load(self, path=MODEL_DIR, quiet=False):
        self.actor.load_state_dict(torch.load(f'{path}/actor.pth', map_location=DEVICE))
        self.critic.load_state_dict(torch.load(f'{path}/critic.pth', map_location=DEVICE))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        if not quiet:
            print(f"  [loaded] {path}/")


# ── Training Loop ──────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='train on GPU (CUDA)')
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    if args.cuda:
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        else:
            print('  [warning] --cuda specified but CUDA is not available, falling back to CPU')

    print(f"  [device] training on {DEVICE}")
    env    = TorcsEnv(throttle=True, gear_change=False)
    agent  = TD3(STATE_DIM, ACTION_DIM)
    buffer = PrioritizedReplayBuffer(BUFFER_SIZE)

    best_reward          = -np.inf
    best_avg10           = -np.inf
    rollback_cooldown    = 0
    steps_since_rollback = ROLLBACK_SEED  # start ready to train

    # Load best checkpoint
    best_dir = f'{MODEL_DIR}/best'
    if os.path.exists(best_dir):
        scored = []
        for d in os.listdir(best_dir):
            try:
                scored.append((float(d), d))
            except ValueError:
                pass
        if scored:
            top_score, top = max(scored)
            agent.load(f'{best_dir}/{top}')
            best_reward = top_score
            print(f"  [resume] loaded best checkpoint {top} — best_reward={top_score:.1f}")
        elif os.path.exists(f'{MODEL_DIR}/actor.pth'):
            agent.load()
    elif os.path.exists(f'{MODEL_DIR}/actor.pth'):
        agent.load()

    ou_steer = OUNoise(1, mu=0.0,  theta=0.15, sigma=0.15)
    ou_accel = OUNoise(1, mu=0.5,  theta=1.0,  sigma=0.005)
    ou_brake = OUNoise(1, mu=-0.9, theta=1.0,  sigma=0.002)

    total_steps     = 0
    episode_rewards = []

    for episode in range(MAX_EPISODES):
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

        for step in range(MAX_STEPS):
            if total_steps < WARMUP_STEPS:
                action = np.array([ou_steer.sample()[0], ou_accel.sample()[0], ou_brake.sample()[0]])
                action = np.clip(action, -1, 1)
            else:
                action = agent.select_action(state)
                action[0] = np.clip(action[0] + ou_steer.sample()[0] * EXPL_NOISE, -1, 1)
                action[1] = np.clip(action[1] + ou_accel.sample()[0] * EXPL_NOISE, -1, 1)
                action[2] = np.clip(action[2] + ou_brake.sample()[0] * EXPL_NOISE, -1, 1)

            obs, reward, done, _, info = env.step(action)
            next_state = obs_to_state(obs)

            buffer.add(state, action, reward, next_state, float(done))
            steps_since_rollback += 1
            if (total_steps > SEED_STEPS
                    and steps_since_rollback >= ROLLBACK_SEED
                    and total_steps % TRAIN_FREQ == 0):
                agent.train(buffer)

            state          = next_state
            episode_reward += reward
            total_steps    += 1

            if done:
                term_reason = info.get('term_reason', 'unknown')
                ep_time     = info.get('time', 0.0)
                break

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-10:])
        mins, secs = divmod(ep_time, 60)
        buf_pct    = 100.0 * len(buffer) / BUFFER_SIZE

        line = (
            f"Ep {episode:4d} | "
            f"Time {int(mins)}:{secs:05.2f} | "
            f"Steps {step+1:5d} | "
            f"Reward {episode_reward:8.1f} | "
            f"Avg(10) {avg_reward:8.1f} | "
            f"Buf {buf_pct:5.1f}% | "
            f"Total {total_steps:7d} | "
            f"End: {term_reason}"
        )
        print(line)
        with open(LOG_FILE, 'a') as f:
            f.write(line + '\n')

        # Save rollback checkpoint whenever avg10 hits a new high
        if avg_reward > best_avg10:
            best_avg10 = avg_reward
            agent.save(f'{MODEL_DIR}/rollback', quiet=True)

        # Auto-rollback if avg10 collapsed
        rollback_cooldown = max(0, rollback_cooldown - 1)
        if (rollback_cooldown == 0
                and len(episode_rewards) >= 10
                and avg_reward < best_avg10 * (1 - ROLLBACK_DROP)
                and os.path.exists(f'{MODEL_DIR}/rollback/actor.pth')):
            agent.load(f'{MODEL_DIR}/rollback', quiet=True)
            buffer = PrioritizedReplayBuffer(BUFFER_SIZE)  # clear poisoned buffer
            steps_since_rollback = 0                       # pause training for ROLLBACK_SEED steps
            episode_rewards = []                           # reset avg so it reflects new episodes only
            rollback_cooldown = ROLLBACK_COOLDOWN
            rb_line = f"  [rollback] avg {avg_reward:.0f} dropped from peak {best_avg10:.0f} — weights + buffer reset"
            print(rb_line)
            with open(LOG_FILE, 'a') as f:
                f.write(rb_line + '\n')

        # Save best single-episode reward checkpoint
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(f'{MODEL_DIR}/best/{episode_reward:.1f}')
            best_line = f"  [best] ep {episode} reward {episode_reward:.1f}"
            print(best_line)
            with open(LOG_FILE, 'a') as f:
                f.write(best_line + '\n')

        # Save whenever a lap is finished
        if term_reason == 'finished':
            agent.save(f'{MODEL_DIR}/best/finish_latest', quiet=True)

    env.end()
    agent.save()
    print("Training complete.")
