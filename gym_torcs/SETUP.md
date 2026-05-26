# TORCS AI Racing — Setup Guide

## Prerequisites

- Windows 10 or 11 (64-bit)
- Python 3.10–3.13
- TORCS installed at `C:\torcs\torcs\torcs\` with `wtorcs.exe` present
- TORCS must be pre-configured with **Corkscrew** track and **scr_server** driver
  (do this once manually — TORCS remembers the settings)

---

## One-Time TORCS Configuration

Do this once before using the automated launcher:

1. Double-click `C:\torcs\torcs\torcs\wtorcs.exe`
2. Press **Tab** then **Enter**
3. Press **Enter** (Quick Race)
4. Select **Configure Race**
5. Set track to **Corkscrew**
6. Set driver to **scr_server 1**
7. Press **New Race** — confirm it loads and waits for the server
8. Close TORCS

After this, TORCS remembers these settings and the auto-launcher will work.

---

## Environment Setup

### Step 1 — Create the virtual environment

Open a terminal in `C:\torcs\torcs\gym_torcs\`:

```
python -m venv venv
```

### Step 2 — Activate the virtual environment

```
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 3 — Install dependencies

```
pip install -r requirements.txt
```

This installs: numpy, gymnasium, pyautogui, torch, matplotlib.

---

## Running the AI Driver

Make sure the venv is activated, then run:

```
python torcs_jm_par.py
```

This uses the rule-based driver to connect to TORCS on port 3001.

---

## Running RL Training

With the venv activated:

```
python train.py
```

`gym_torcs.py` will automatically:
- Kill any running TORCS instance
- Launch `wtorcs.exe`
- Navigate the menu (Tab → Enter → Enter → Enter)
- Wait for the track to load
- Connect the Python client on port 3001

If TORCS becomes unstable after many episodes (memory leak), stop training,
restart TORCS manually through the menu, then run the script again.

---

## File Overview

| File | Purpose |
|---|---|
| `gym_torcs.py` | Gymnasium-compatible TORCS environment (auto-launches TORCS) |
| `snakeoil3_gym.py` | UDP client that talks to TORCS on port 3001 |
| `torcs_jm_par.py` | Rule-based driver — good baseline and starting point |
| `train.py` | TD3 RL training script |
| `sample_agent.py` | Random agent placeholder |
| `requirements.txt` | Python dependencies |

---

## Troubleshooting

**TORCS does not start / menu navigation fails**
- Increase `TORCS_LOAD_WAIT` in `gym_torcs.py` (default: 5s)
- Make sure no other TORCS window is open before running

**Python client cannot connect**
- Check TORCS is in "waiting for server" state (blue loading screen)
- Confirm `scr_server` is selected as the driver in TORCS
- Allow `wtorcs.exe` through Windows Defender Firewall

**Car does not move**
- Check port 3001 is not in use by another process
- Set `PYTHONUTF8=1` environment variable if seeing encoding errors
