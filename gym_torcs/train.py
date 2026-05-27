import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import random
import os

from gym_torcs import TorcsEnv

# ── Hyperparameters ────────────────────────────────────────────────
STATE_DIM      = 24       # track(19) + speedX + speedY + speedZ + angle + trackPos
ACTION_DIM     = 3        # steer + accel + brake
MAX_EPISODES   = 10000
MAX_STEPS      = 10000    # max steps per episode
BATCH_SIZE     = 64
BUFFER_SIZE    = 100_000
GAMMA          = 0.99
TAU            = 0.005    # soft update rate
ACTOR_LR       = 5e-5
CRITIC_LR      = 5e-5
POLICY_NOISE   = 0.2      # noise added to target actions
NOISE_CLIP     = 0.5      # clip target action noise
POLICY_DELAY   = 2        # update actor every N critic updates
WARMUP_STEPS   = 0        # no random warmup when loading a good checkpoint
EXPL_NOISE     = 0.005    # tiny noise for fine-tuning
SAVE_EVERY     = 1        # save every episode to track best
RELAUNCH_EVERY = 20       # restart TORCS every N episodes (memory leak workaround)
MODEL_DIR      = 'models'
DEVICE         = torch.device('cpu')


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
    """Flatten the TorcsEnv namedtuple observation into a 1-D numpy array."""
    return np.hstack([
        np.atleast_1d(obs.track),    # 19 — distance to track edges (normalised)
        np.atleast_1d(obs.speedX),   # 1  — forward speed (normalised)
        np.atleast_1d(obs.speedY),   # 1  — lateral speed (normalised)
        np.atleast_1d(obs.speedZ),   # 1  — vertical speed (normalised)
        np.atleast_1d(obs.angle),    # 1  — angle to track axis (normalised by pi)
        np.atleast_1d(obs.trackPos), # 1  — lateral position (-1 left, 0 center, 1 right)
    ]).astype(np.float32)


# ── Replay Buffer ──────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(DEVICE),
            torch.FloatTensor(np.array(a)).to(DEVICE),
            torch.FloatTensor(np.array(r)).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(np.array(s2)).to(DEVICE),
            torch.FloatTensor(np.array(d)).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


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
        s, a, r, s2, d = replay_buffer.sample(BATCH_SIZE)

        with torch.no_grad():
            noise  = (torch.randn_like(a) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_a = (self.actor_target(s2) + noise).clamp(-1, 1)
            q1_t, q2_t = self.critic_target(s2, next_a)
            q_target = r + GAMMA * (1 - d) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if self.total_it % POLICY_DELAY == 0:
            actor_loss = -self.critic.q1_only(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

    def save(self, path=MODEL_DIR):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),  f'{path}/actor.pth')
        torch.save(self.critic.state_dict(), f'{path}/critic.pth')
        print(f"  [saved] {path}/")

    def load(self, path=MODEL_DIR):
        self.actor.load_state_dict(torch.load(f'{path}/actor.pth', map_location=DEVICE))
        self.critic.load_state_dict(torch.load(f'{path}/critic.pth', map_location=DEVICE))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        print(f"  [loaded] {path}/")


# ── Training Loop ──────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='train on GPU (CUDA)')
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining  # hide our flags from the TORCS getopt parser

    if args.cuda:
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        else:
            print('  [warning] --cuda specified but CUDA is not available, falling back to CPU')

    print(f"  [device] training on {DEVICE}")
    env    = TorcsEnv(throttle=True, gear_change=False)
    agent  = TD3(STATE_DIM, ACTION_DIM)
    buffer = ReplayBuffer(BUFFER_SIZE)

    best_reward = -np.inf

    # Resume from highest-reward best checkpoint if any exist, otherwise latest
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
            print(f"  [resume] loaded best checkpoint {top} — best_reward set to {top_score:.1f}")
        elif os.path.exists(f'{MODEL_DIR}/actor.pth'):
            agent.load()
    elif os.path.exists(f'{MODEL_DIR}/actor.pth'):
        agent.load()

    ou_steer = OUNoise(1, mu=0.0,  theta=0.6, sigma=0.01)
    ou_accel = OUNoise(1, mu=0.5,  theta=1.0, sigma=0.005)
    ou_brake = OUNoise(1, mu=-0.9, theta=1.0, sigma=0.002)

    total_steps    = 0
    episode_rewards = []

    for episode in range(MAX_EPISODES):
        relaunch = (episode % RELAUNCH_EVERY == 0)
        obs, _   = env.reset(relaunch=relaunch)
        state    = obs_to_state(obs)
        ou_steer.reset()
        ou_accel.reset()
        ou_brake.reset()

        episode_reward = 0
        step = 0
        term_reason = 'max_steps'

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
            agent.train(buffer)

            state          = next_state
            episode_reward += reward
            total_steps    += 1

            if done:
                term_reason = info.get('term_reason', 'unknown')
                break

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-10:])
        print(
            f"Ep {episode:4d} | "
            f"Steps {step+1:5d} | "
            f"Reward {episode_reward:8.1f} | "
            f"Avg(10) {avg_reward:8.1f} | "
            f"Total {total_steps:7d} | "
            f"End: {term_reason}"
        )

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_name = f'{MODEL_DIR}/best/{episode_reward:.1f}'
            agent.save(best_name)
            print(f"  [best] ep {episode} reward {episode_reward:.1f}")

    env.end()
    agent.save()
    print("Training complete.")
