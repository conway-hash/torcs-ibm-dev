# TORCS AI Racing — gym_torcs

A Python/Windows reinforcement learning environment for TORCS, based on the original [gym-torcs](https://github.com/ugo-nama-kun/gym_torcs) by Naoto Yoshida (Preferred Networks, 2016), adapted for Windows with an automatic TORCS launcher.

---

## What's in this folder

| File | Role |
|---|---|
| `gym_torcs.py` | Gymnasium-compatible `TorcsEnv` class — auto-launches TORCS, wraps the scr_server protocol |
| `snakeoil3_gym.py` | UDP client that speaks the TORCS scr_server protocol on port 3001 |
| `train.py` | TD3 + Prioritized Experience Replay training script — the main entry point for RL |
| `plot_training.py` | Live matplotlib dashboard that reads `training_log.txt` and updates every 10 s |
| `torcs_jm_par.py` | Rule-based driver — connects directly to TORCS, no ML required; good for testing the connection |
| `sample_agent.py` | Random-action agent, useful for verifying the environment API works |
| `example_experiment.py` | Minimal example showing how to create and step through `TorcsEnv` |
| `requirements.txt` | Python package dependencies |
| `models/` | Saved actor/critic checkpoints written by `train.py` |

---

## Architecture

```
train.py
  └── TorcsEnv  (gym_torcs.py)
        ├── launches wtorcs.exe via subprocess + pyautogui
        └── Client  (snakeoil3_gym.py)
              └── UDP socket ←→ TORCS scr_server driver (port 3001)
```

`train.py` creates a `TorcsEnv`, which on construction kills any running TORCS instance, launches `wtorcs.exe`, and navigates the menu automatically. The `Client` in `snakeoil3_gym.py` then connects over UDP and exchanges sensor/action packets with the `scr_server` driver inside TORCS.

---

## Quick start

See [SETUP.md](SETUP.md) for full installation instructions. Once set up:

```python
from gym_torcs import TorcsEnv

env = TorcsEnv(throttle=True)           # auto-launches TORCS
obs, _ = env.reset(relaunch=True)

# action: [steer, accel, brake]  — all in [-1, 1]
obs, reward, done, _, info = env.step([0.0, 0.5, 0.0])

env.end()                               # kills TORCS cleanly
```

---

## Observation and action spaces

### State vector (24 dimensions, from `obs_to_state` in `train.py`)

| Sensor | Dims | Range | Description |
|---|---|---|---|
| `track` | 19 | [0, 1] | Distance sensors at 19 angles around the car |
| `speedX` | 1 | [-∞, ∞] | Forward speed (normalised by 50 km/h) |
| `speedY` | 1 | [-∞, ∞] | Lateral speed |
| `speedZ` | 1 | [-∞, ∞] | Vertical speed |
| `angle` | 1 | [-1, 1] | Car angle relative to track axis (÷ π) |
| `trackPos` | 1 | [-1, 1] | Lateral position (0 = centre, ±1 = edge) |

### Action vector (3 dimensions, `throttle=True`)

| Index | Signal | Range | Notes |
|---|---|---|---|
| 0 | `steer` | [-1, 1] | Left/right steering |
| 1 | `accel` | [-1, 1] | Remapped to [0, 1] internally |
| 2 | `brake` | [-1, 1] | Only positive portion used |

---

## Training algorithm

`train.py` implements **TD3 (Twin Delayed DDPG)** with several enhancements:

- **Clipped Double Q-learning** — two critic networks, target uses the minimum
- **Delayed policy updates** — actor updated every `POLICY_DELAY` critic steps
- **Prioritized Experience Replay (PER)** — samples transitions by TD error via a sum-tree
- **Ornstein-Uhlenbeck exploration noise** — separate processes for steer, accel, and brake
- **Auto-rollback** — if the 10-episode rolling average drops more than `ROLLBACK_DROP` below its peak, the best checkpoint is restored automatically
- **Periodic TORCS relaunch** — every `RELAUNCH_EVERY` episodes to avoid the simulator's memory leak

### Key hyperparameters

| Constant | Default | Effect |
|---|---|---|
| `MAX_EPISODES` | 10 000 | Total training episodes |
| `RELAUNCH_EVERY` | 20 | Relaunch TORCS every N episodes |
| `ACTOR_LR` / `CRITIC_LR` | 3e-6 | Learning rates |
| `EXPL_NOISE` | 0.08 | Scale of exploration noise |
| `ROLLBACK_DROP` | 0.60 | Rollback trigger: avg drop fraction below peak |
| `BUFFER_SIZE` | 500 000 | Replay buffer capacity |
| `BATCH_SIZE` | 256 | Minibatch size |

---

## Reward function

At each step:

```
reward = (speedX × cos(angle))   # forward progress
       + 0.3 × speedX            # speed bonus
       - 0.5 × |speedY|          # lateral slip penalty
       - 2.0                     # time penalty
       - 0.5 × |Δsteer|          # smoothness penalty
```

Episode terminates (with –1 000 penalty) if:
- Car goes off-track (`track.min() < 0`)
- Car is going backwards (`cos(angle) < 0`)
- Car is too slow for > 100 steps (`progress < 5 km/h`)
- Car takes damage

A large bonus is awarded for completing a lap.

---

## Checkpoints

`train.py` saves models to `models/`:

| Path | Content |
|---|---|
| `models/actor.pth` / `models/critic.pth` | Latest weights |
| `models/best/<reward>/` | Best single-episode checkpoint |
| `models/best/finish_latest/` | Latest lap-completion checkpoint |
| `models/rollback/` | Best 10-episode rolling average checkpoint (used for auto-rollback) |

Training resumes automatically if any of these exist.

---

## License

Based on [gym-torcs](https://github.com/ugo-nama-kun/gym_torcs) by Naoto Yoshida, developed during spring internship 2016 at Preferred Networks. See [LICENSE](LICENSE).
