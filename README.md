# torcs-ibm-dev

Evolution Strategies AI trained to drive the Corkscrew track in TORCS.

## Setup (once)

Open PowerShell in the project folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Creates a `venv` and installs all dependencies.

---

## Watch the AI drive

```powershell
powershell -ExecutionPolicy Bypass -File run.ps1
```

A TORCS window opens. When it does, click:

**Race → Quick Race → New Race**

The AI connects automatically and starts driving. Lap time prints when it finishes.

---

## Train

```powershell
venv\Scripts\python gym_torcs\train_es.py
```

Runs indefinitely. Press `Ctrl+C` to stop. Saves `runs/corkscrew_evomap/best.npz` whenever a new fastest lap is found.

---

## Requirements

- Windows 10/11
- Python 3.10+
- TORCS Windows binary — download from https://ibm.ent.box.com/v/TORCSdownloadzip and extract into the `torcs/` folder so that `torcs\wtorcs.exe` exists

---

## Troubleshooting

**`cannot bind socket`** — a stale TORCS process is holding the port:
```powershell
venv\Scripts\python gym_torcs\launcher.py --kill
```
Then re-run `run.ps1`.

**No TORCS window** — the graphics DLL only loads when TORCS starts from its own menu. `run.ps1` handles this automatically.
