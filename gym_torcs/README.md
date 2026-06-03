# random_test — overnight fastest-lap trainer (Evolution Strategies)

A self-contained, **unattended overnight** trainer that drives TORCS Corkscrew
and **continuously improves the fastest lap** without falling into the traps
that broke the earlier SAC attempts (catastrophic forgetting, alpha runaway,
reward-hacking, regression-on-resume).

It uses **Evolution Strategies (ES)** on a small neural-net policy instead of
off-policy RL. ES optimises the **true objective** — actual lap time — and
keeps an **elite** (best policy ever found) that it can never regress below.
That combination is what makes it safe to start and walk away.

```
config.py      every tunable in one place
scr_client.py  SCRC UDP client
launcher.py    TORCS process management + watchdog
torcs_env.py   env: 33-dim obs (incl. MPC speed lookahead), gearbox, lap timing
policy.py      tiny tanh MLP, evaluated in pure numpy
es.py          OpenAI-style ES: antithetic, rank-shaped, Adam, adaptive sigma
mpc_bc.py      OPTIONAL warm-start: behavior-clone the MPC into the policy
train_es.py    the overnight trainer (run this)
eval.py        replay the best policy in a window
```

---

## TL;DR — run it overnight

```powershell
cd C:\Users\honza\auticka\random_test

# 1. (optional, ~2 min) warm-start from the MPC so it starts driving well
python mpc_bc.py --attempts 10 --epochs 300

# 2. start the overnight run and walk away  (Ctrl-C anytime; it auto-resumes)
python train_es.py

# 3. next morning: watch the best lap it found
python eval.py --gui
```

That's it. `train_es.py` runs forever until you stop it, checkpoints every
generation, and survives TORCS crashes. Re-running it continues where it left
off. The fastest validated lap is always in `runs/corkscrew_es/best.npz`.

---

## Expectations (honest)

The MPC's *planned optimal* on this car/track is **69 s** (from
`gym_torcs/track_map.json`). Realistic outcomes for this trainer:

| Lap time | Likelihood overnight |
|---|---|
| **~95–110 s** | very likely (a clean completed lap) |
| **~85–95 s** | likely with the MPC warm-start + a full night |
| **~75 s** | stretch goal, needs a good night and some luck |
| **60 s** | almost certainly **below the car's physical limit** here — don't expect it |

The longer it runs, the better — ES improves monotonically, so more hours only
helps. A full 8-hour night is the intended use.

---

## Why this won't fall into a trap

Every failure mode from the SAC runs is structurally prevented:

| Trap (what happened before) | Mechanism that prevents it |
|---|---|
| Catastrophic forgetting / regression | **Elitism**: `best.npz` is only overwritten by a faster, re-validated lap. Cannot get worse. |
| Reward hacking (the brake-penalty bug) | Fitness **is** lap time / distance — there is no proxy reward to game. |
| Lucky fluke crowned as "best" | New records are **re-evaluated K times** (`es_elite_reeval`) and averaged before being accepted. |
| Every candidate stalls → no signal | Fresh policies **floor the throttle** (output bias) so ES always gets distance signal; bad ones die fast on the watchdogs. |
| Stuck in a local optimum | **Adaptive sigma**: exploration ramps up after `es_stagnation_patience` stagnant generations, relaxes on progress. |
| TORCS hangs/crashes at 3 a.m. | Each worker **relaunches** dead TORCS and returns a floor score; a per-generation **timeout** fills missing rollouts so one bad worker never stalls the run. |
| Lost progress on restart | Full ES state (mean, Adam, sigma, RNG, elite) **checkpointed every generation** (atomic write); `train_es.py` **auto-resumes**. |
| Resume re-poisons the policy | ES has no replay buffer and no demonstrator in the loop — resume is exact and inert. |

---

## Reading the log

```
[gen  142] best=96.41s  gen_best=98.30s  mean=101.7s  maxdist=3650m  laps=7/32  sigma=0.052  74s (0.4 ev/s)
```
- **best**     — fastest validated lap so far (this is your result; saved to `best.npz`).
- **gen_best** — fastest lap among this generation's candidates.
- **mean**     — lap time of the current mean policy.
- **maxdist**  — furthest any candidate drove (watch this climb toward 3650 m early on).
- **laps**     — how many of the generation's candidates completed a full lap.
- **sigma**    — current exploration spread (rises if stuck, falls on progress).

`runs/corkscrew_es/log.csv` has the full history for plotting.

Early on (before any candidate completes a lap) `best`/`gen_best` show `-` and
you watch **maxdist** grow. Once it reaches ~3650 m a lap is completed and the
time numbers appear and start dropping.

---

## Commands

```powershell
python train_es.py                 # start / auto-resume (default 6 workers)
python train_es.py --fresh         # ignore saved state, start over
python train_es.py --hours 8       # stop after 8 wall-clock hours
python train_es.py --workers 4     # fewer parallel TORCS (less CPU)

python mpc_bc.py --attempts 10 --epochs 300   # (re)build the warm-start
python mpc_bc.py --gui                          # watch the MPC record

python eval.py --gui               # watch best.npz drive (safe during training)
python eval.py --gui --episodes 5  # repeatability / average over laps
python eval.py --theta runs/corkscrew_es/best.npz

python launcher.py --kill          # clear stray/zombie TORCS processes
python launcher.py --smoke 2       # sanity check: 2 headless TORCS handshake
```

`eval.py` uses a free scr_server slot (above the workers) and never kills other
TORCS, so it is safe to run while training continues.

---

## Key knobs (`config.py`)

| Setting | Default | Effect |
|---|---|---|
| `n_workers` | 6 | parallel TORCS / rollouts. Lower if the machine struggles. |
| `es_popsize` | 32 | candidates per generation. More = smoother gradient, slower generations. |
| `es_sigma` | 0.05 | base exploration spread. |
| `es_lr` | 0.02 | step size on the policy mean. |
| `es_elite_reeval` | 3 | re-eval laps to confirm a new record (noise robustness). |
| `es_stagnation_patience` | 15 | generations before sigma ramps up to escape a plateau. |
| `max_steps` | 4500 | per-lap step budget (~270 s sim). Must exceed a full lap. |
| `track_length_m` | 3649.86 | real Corkscrew length (from `track_map.json`). |

---

## TORCS gotchas (learned the hard way)

- **scr_server binds UDP `3001 + driver_idx`** regardless of config — the
  Python port must match. `base_port = 3001`.
- **Lap detection** uses cumulative `distRaced ≥ track_length_m`, self-timed
  from launch. `lastLapTime` / `distFromStart` are **not** reliably populated
  in this SCR race mode + `-nolaptime`.
- **The real lap is 3649.86 m** (not the ~2057 m you might guess). Too small a
  value mis-fires lap detection; too large means a lap never registers.
- **Determinism is weak** across runs — same policy ≠ identical lap time. ES
  handles this via population averaging and elite re-evaluation.
- Headless (`-T`) starts in ~2.5 s; a GUI window takes ~10 s — the env sleeps
  before the handshake accordingly.

## Versions
Python 3.11, torch 2.x (only needed for the optional `mpc_bc.py`), numpy 2.x,
scipy (only for the MPC). The ES trainer itself needs only numpy.
