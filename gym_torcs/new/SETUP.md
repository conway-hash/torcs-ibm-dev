# Setup Guide — TORCS AI Racing (Windows)

This guide covers everything from a fresh Windows machine to running RL training.

---

## Prerequisites

- Windows 10 or 11 (64-bit)
- Python 3.10–3.13 (must be on your PATH — verify with `python --version`)
- TORCS installed at `C:\torcs\torcs\torcs\` with `wtorcs.exe` present

> If TORCS is installed somewhere else, update `TORCS_EXE` at the top of `gym_torcs.py`.

---

## Step 1 — One-time TORCS configuration

You must configure TORCS once by hand before running any scripts. After this, TORCS remembers the settings and the auto-launcher will work.

1. Double-click `C:\torcs\torcs\torcs\wtorcs.exe`
2. Press **Tab**, then **Enter** — this enters the Race menu
3. Press **Enter** — selects Quick Race
4. Select **Configure Race**
5. Set the **track** to **Corkscrew** (road category)
6. Set the **driver** to **scr_server 1**
7. Press **New Race** — TORCS should show a blue screen with "Waiting for connection…"
8. Close TORCS

> **Why this matters:** `gym_torcs.py` navigates the menu by sending Enter × 3 automatically. For this to reach "New Race" directly, TORCS must already have Corkscrew + scr_server selected.

---

## Step 2 — Python virtual environment

Open a terminal (PowerShell or Command Prompt) in the `new/` folder:

```
python -m venv venv
```

Activate it:

```
venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt. Now install dependencies:

```
pip install -r requirements.txt
```

This installs: `numpy`, `gymnasium`, `pyautogui`, `torch`, `matplotlib`.

---

## Step 3 — Run RL training

With the virtual environment active:

```
python train.py
```

What happens automatically:
1. Any running TORCS instance is killed (`taskkill`)
2. `wtorcs.exe` is launched
3. TORCS is given `TORCS_LOAD_WAIT` seconds (default: 10 s) to reach the main menu
4. Three Enter keypresses navigate to New Race
5. TORCS is given `TORCS_RACE_WAIT` seconds (default: 8 s) to load the track
6. The Python UDP client connects on port 3001
7. Training begins

Training logs are appended to `training_log.txt`. The script resumes from the best saved checkpoint automatically if `models/` contains weights.

To watch training progress live in a second terminal (keep the venv active):

```
python plot_training.py
```

This opens a matplotlib window that refreshes every 10 seconds.

---

## Step 4 — Enable GPU training (optional)

If you have a CUDA-capable GPU:

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
python train.py --cuda
```

CPU training is fully supported and is the default.

---

## Step 5 — Test the connection without RL (optional)

To verify TORCS is reachable before training:

1. Launch TORCS manually and navigate to New Race (see Step 1)
2. In a second terminal with the venv active:

```
python torcs_jm_par.py
```

This connects to TORCS on port 3001 and drives with a handcrafted rule-based controller. If the car moves, the connection is working.

---

## File reference

| File | Purpose |
|---|---|
| `gym_torcs.py` | Gymnasium-compatible `TorcsEnv` — auto-launches TORCS, defines step/reset/end |
| `snakeoil3_gym.py` | UDP client implementing the TORCS scr_server protocol |
| `train.py` | TD3 + PER training loop — primary entry point |
| `plot_training.py` | Real-time training dashboard (reads `training_log.txt`) |
| `torcs_jm_par.py` | Rule-based driver for connection testing and baseline comparison |
| `sample_agent.py` | Random agent for verifying the `TorcsEnv` API |
| `example_experiment.py` | Minimal usage example |
| `requirements.txt` | Python dependencies |
| `models/` | Saved checkpoints — training resumes from here automatically |

---

## Tuning

### gym_torcs.py constants

| Constant | Default | What to change |
|---|---|---|
| `TORCS_EXE` | `C:\torcs\...\wtorcs.exe` | Path to TORCS executable |
| `TORCS_LOAD_WAIT` | 10 s | Increase if TORCS is slow to open on your machine |
| `TORCS_RACE_WAIT` | 8 s | Increase if the track takes longer to load |

### train.py constants

| Constant | Default | What to change |
|---|---|---|
| `MAX_EPISODES` | 10 000 | Total number of training episodes |
| `RELAUNCH_EVERY` | 20 | How often to restart TORCS (avoids memory leak) |
| `ACTOR_LR` / `CRITIC_LR` | 3e-6 | Learning rates — lower = more stable, slower |
| `EXPL_NOISE` | 0.08 | Exploration noise — higher = more random early exploration |
| `ROLLBACK_DROP` | 0.60 | Fraction drop in avg reward that triggers auto-rollback |
| `SEED_STEPS` | 20 000 | Env steps before training begins (buffer warm-up) |

---

## Troubleshooting

### TORCS does not start or menu navigation fails

- Increase `TORCS_LOAD_WAIT` in `gym_torcs.py` (try 15 or 20)
- Make sure no other TORCS window is open before running `train.py`
- Run `taskkill /f /im wtorcs.exe` manually, then try again

### Python client cannot connect

- Confirm TORCS is showing the blue "Waiting for connection…" screen
- Confirm `scr_server` is selected as the driver in TORCS (Step 1)
- Allow `wtorcs.exe` through Windows Defender Firewall on first run (a dialog may appear)
- Check if port 3001 is in use: `netstat -ano | findstr :3001`

### Car does not move / immediately crashes

- Check `PYTHONUTF8=1` is set if you see encoding errors on launch
- The agent starts with random actions — the car will crash early in training; this is normal
- Training needs several thousand steps before the replay buffer is warm enough to improve

### Training reward is stuck very negative

- This is normal for the first 50–100 episodes while the buffer fills
- Check `training_log.txt` — if all episodes end in `too_slow` or `backwards`, the env is connected correctly but the agent has not learned yet
- If `off_track` is the only termination reason, the Corkscrew track is loading correctly

### TORCS freezes or crashes after many episodes

- Stop training with Ctrl+C
- Run `taskkill /f /im wtorcs.exe`
- Re-run `python train.py` — it will resume from the last saved checkpoint

---

## Typical training run

```
Ep    0 | Time 0:12.40 | Steps   620 | Reward  -12543.2 | Avg(10) -12543.2 | Buf   0.1% | Total     620 | End: off_track
Ep    1 | Time 0:08.14 | Steps   407 | Reward   -8921.0 | Avg(10) -10732.1 | Buf   0.2% | Total    1027 | End: too_slow
...
Ep  340 | Time 1:43.22 | Steps  5161 | Reward  142000.0 | Avg(10)  98000.0 | Buf  42.0% | Total  180000 | End: finished
  [best] ep 340 reward 142000.0
```

A lap completion (`End: finished`) usually appears after several hundred episodes.
