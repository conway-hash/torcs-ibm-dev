#!/usr/bin/env python3
"""
TORCS Continuous Training — AI Reviews Every 10 Minutes

  - Every episode is saved as models/episodes/ep_NNNNNN/ (weights + stats)
  - No rollback feature, no best-model logic — just a full history
  - On start: automatically loads the last saved episode
  - Every 10 min: AI reviews training log + episode history and decides:
      ok            — do nothing, keep training
      change        — AI picks any past episode + optionally new params

Usage:
  $env:ANTHROPIC_API_KEY = "sk-ant-..."
  python auto_train.py
  python auto_train.py --check-minutes 10
  python auto_train.py --skip-setup
"""

import argparse, json, os, re, shutil, subprocess, sys
import tempfile, time, traceback, py_compile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import anthropic

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR      = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT  = os.path.join(WORK_DIR, 'train.py')
GYM_SCRIPT    = os.path.join(WORK_DIR, 'gym_torcs.py')
LOG_FILE      = os.path.join(WORK_DIR, 'training_log.txt')
STOP_FLAG     = os.path.join(WORK_DIR, 'stop.flag')
LOAD_FLAG     = os.path.join(WORK_DIR, 'load.flag')
EPISODE_DIR   = os.path.join(WORK_DIR, 'models', 'episodes')
SNAPSHOTS_DIR = os.path.join(WORK_DIR, 'snapshots')   # human-readable check archive
VENV_PYTHON   = os.path.join(WORK_DIR, 'venv', 'Scripts', 'python.exe')

CHECK_MINUTES = 10
LLM_MODEL     = 'claude-opus-4-8'
LOG_TAIL      = 400    # recent log lines sent to LLM
EP_HISTORY    = 60     # last N episode stats shown to LLM


# ── Utilities ──────────────────────────────────────────────────────────────────

def python_exe():
    return VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable


def read_file(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        return f.read()


def tail_log(n=LOG_TAIL):
    if not os.path.exists(LOG_FILE):
        return '(no log yet)'
    with open(LOG_FILE, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    return ''.join(lines[-n:])


def validate_python(code, label):
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8') as f:
        f.write(code)
        tmp = f.name
    try:
        py_compile.compile(tmp, doraise=True)
        return True
    except py_compile.PyCompileError as e:
        print(f'  [syntax error in {label}] {e}')
        return False
    finally:
        os.unlink(tmp)


def extract_tag(text, tag):
    m = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    if not m:
        return None
    c = m.group(1).strip()
    c = re.sub(r'^```(?:python)?\s*', '', c)
    c = re.sub(r'\s*```$', '', c)
    return c.strip()


# ── Episode inventory ──────────────────────────────────────────────────────────

_EP_RE = re.compile(r'^ep_(\d{6})(?:\.(\d+))?$')


def list_episodes():
    """Return sorted list of (sort_key, dirname, path, info_dict) for all saved episodes."""
    if not os.path.exists(EPISODE_DIR):
        return []
    eps = []
    for d in os.listdir(EPISODE_DIR):
        m = _EP_RE.match(d)
        if not m:
            continue
        ep_num = int(m.group(1))
        branch = int(m.group(2)) if m.group(2) else 0
        path   = os.path.join(EPISODE_DIR, d)
        info   = {}
        info_p = os.path.join(path, 'info.json')
        if os.path.exists(info_p):
            with open(info_p) as f:
                info = json.load(f)
        # sort by file modification time so order reflects actual training sequence
        mtime = os.path.getmtime(path) if os.path.exists(path) else 0
        eps.append((mtime, d, path, info))
    return sorted(eps)   # oldest first


def last_episode():
    """Return (dirname, path) of most recently saved episode, or (None, None)."""
    eps = list_episodes()
    if not eps:
        return None, None
    _, d, path, _ = eps[-1]
    return d, path


def _read_ep_state():
    state_file = os.path.join(EPISODE_DIR, '_state.json')
    if os.path.exists(state_file):
        with open(state_file) as f:
            return json.load(f)
    return {}


def episode_summary_for_llm():
    """Compact table of the last EP_HISTORY episodes for the LLM prompt."""
    eps = list_episodes()[-EP_HISTORY:]
    if not eps:
        return 'No episodes saved yet.'
    lines = ['episode_name    | reward      | avg10       | term_reason']
    lines.append('-' * 62)
    for _, d, _, info in eps:
        lines.append(
            f"{d:16s} | "
            f"{info.get('reward', 0):>11.1f} | "
            f"{info.get('avg10',  0):>11.1f} | "
            f"{info.get('term_reason', '?')}"
        )
    return '\n'.join(lines)


# ── Process management ─────────────────────────────────────────────────────────

def launch_proc(load_from=None):
    for flag in (STOP_FLAG, LOAD_FLAG):
        if os.path.exists(flag):
            os.remove(flag)
    cmd = [python_exe(), TRAIN_SCRIPT]
    if load_from:
        cmd += ['--load-from', load_from]
    lbl = f'--load-from {load_from}' if load_from else '(auto: last episode)'
    print(f'  Launching train.py {lbl}')
    return subprocess.Popen(cmd, cwd=WORK_DIR)


def stop_proc(proc):
    if proc is None or proc.poll() is not None:
        return
    with open(STOP_FLAG, 'w') as f:
        f.write('stop')
    try:
        proc.wait(timeout=300)
        print('  train.py stopped.')
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        if os.path.exists(STOP_FLAG):
            os.remove(STOP_FLAG)


# ── Snapshot ───────────────────────────────────────────────────────────────────

def save_snapshot(check_n, label=''):
    dest = os.path.join(SNAPSHOTS_DIR, f'check_{check_n:04d}_{label}')
    os.makedirs(dest, exist_ok=True)
    for src in (TRAIN_SCRIPT, GYM_SCRIPT):
        if os.path.exists(src):
            shutil.copy(src, dest)
    with open(os.path.join(dest, 'log_tail.txt'), 'w', encoding='utf-8') as f:
        f.write(tail_log(LOG_TAIL))
    _plot(dest, check_n, label)
    return dest


def _plot(dest, check_n, label):
    ep_re = re.compile(r'Ep\s+(\d+).*?Reward\s+([-\d.]+).*?Avg\(10\)\s+([-\d.]+)')
    episodes, rewards, avgs = [], [], []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = ep_re.search(line)
                if m:
                    episodes.append(int(m.group(1)))
                    rewards.append(float(m.group(2)))
                    avgs.append(float(m.group(3)))
    if not episodes:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(episodes, rewards, alpha=0.25, lw=0.8, color='steelblue', label='Episode reward')
    ax.plot(episodes, avgs,    lw=1.8,     color='darkorange',          label='Avg(10)')
    ax.set_title(f'Check #{check_n} {label} | best_avg={max(avgs):.0f} | {len(episodes)} eps')
    ax.set_xlabel('Episode (global)'); ax.set_ylabel('Reward')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(dest, 'progress.png'), dpi=120)
    plt.close(fig)


# ── LLM ───────────────────────────────────────────────────────────────────────

SYSTEM = """\
You are an expert reinforcement learning engineer optimising a TORCS TD3 car-racing agent.
You make evidence-based decisions. You never break Python syntax.\
"""

SETUP_PROMPT = """\
You are choosing the best starting configuration for a fresh TORCS TD3 training run.

## Episode history (all past training)
{ep_table}

## Current train.py
```python
{train_py}
```

## Current gym_torcs.py
```python
{gym_py}
```

## Task
Pick the hyperparameters and reward function most likely to succeed, based on what the episode
history shows worked and what failed. Key constraints:
- ACCEL_FLOOR = 0.15 prevents zero-throttle death spiral — keep it
- Terminal penalties must be small (< 100). Episode ending is the punishment.
- EXPL_NOISE: 0.10–0.20. Too high = crashes. Too low = no exploration.
- ACTOR_LR / CRITIC_LR: stable range 1e-5 to 1e-4.

If there are past episodes to continue from, specify which one with <load_from>.

Respond with ONLY this XML:

<analysis>Which past config worked best and why.</analysis>
<load_from>models/episodes/ep_NNNNNN</load_from>
<train_py>[complete train.py]</train_py>
<gym_torcs_py>[complete gym_torcs.py]</gym_torcs_py>
"""

REVIEW_PROMPT = """\
## TORCS TD3 training review — check #{check_n}

Training runs CONTINUOUSLY. Episodes are saved individually so you can pick any one to continue from.
There is NO automatic rollback — you decide everything.

## What you can do
- **ok**     — training is healthy, do nothing
- **change** — pick any past episode to continue from (optional) AND/OR change params (optional).
               If only switching episode: hot-swap with no restart (buffer clears, new branch number
               appended to episode names, e.g. ep_002001.2). If also changing params: train.py restarts.

## Saved episode history (last {ep_history} episodes)
{ep_table}

## Recent training log (last ~{log_tail} lines)
```
{log_tail_text}
```

## Current train.py
```python
{train_py}
```

## Current gym_torcs.py
```python
{gym_py}
```

## Decision rules
- Avg(10) going up → **ok**
- Avg(10) was better at an earlier episode → **change**, load that episode
- Policy stuck in a crash loop despite trying different episodes → **change** reward function
- Be conservative. Change 1–3 things max per review.
- DO NOT suggest restart_fresh — only the human can do that.
- Use EXACT episode name from the table for load_from, e.g. models/episodes/ep_002001.1

Respond with ONLY this XML (omit any tag you don't need):

<decision>ok</decision>
<analysis>2-3 sentences on what's happening.</analysis>
<action>What you did and why.</action>
<load_from>models/episodes/ep_NNNNNN</load_from>
<train_py>[complete new train.py — only if changing params]</train_py>
<gym_torcs_py>[complete new gym_torcs.py — only if changing reward]</gym_torcs_py>
"""


def call_llm(prompt):
    client = anthropic.Anthropic()
    print(f'  Calling {LLM_MODEL} ...')
    msg = client.messages.create(
        model=LLM_MODEL, max_tokens=16384,
        system=SYSTEM,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return msg.content[0].text


def apply_files(new_train, new_gym):
    changed = False
    if new_train and validate_python(new_train, 'train.py'):
        shutil.copy(TRAIN_SCRIPT, TRAIN_SCRIPT + '.bak')
        with open(TRAIN_SCRIPT, 'w', encoding='utf-8') as f:
            f.write(new_train)
        print('  [applied] train.py')
        changed = True
    elif new_train:
        print('  [skipped] train.py — syntax error')
    if new_gym and validate_python(new_gym, 'gym_torcs.py'):
        shutil.copy(GYM_SCRIPT, GYM_SCRIPT + '.bak')
        with open(GYM_SCRIPT, 'w', encoding='utf-8') as f:
            f.write(new_gym)
        print('  [applied] gym_torcs.py')
        changed = True
    elif new_gym:
        print('  [skipped] gym_torcs.py — syntax error')
    return changed


# ── Initial setup ──────────────────────────────────────────────────────────────

def initial_setup():
    print('\n── Initial setup ────────────────────────────────────────────')
    ep_table = episode_summary_for_llm()
    prompt   = SETUP_PROMPT.format(
        ep_table=ep_table,
        train_py=read_file(TRAIN_SCRIPT),
        gym_py=read_file(GYM_SCRIPT),
    )
    text      = call_llm(prompt)
    analysis  = extract_tag(text, 'analysis')     or ''
    load_from = (extract_tag(text, 'load_from')   or '').strip() or None
    new_train = extract_tag(text, 'train_py')
    new_gym   = extract_tag(text, 'gym_torcs_py')

    print(f'  Analysis  : {analysis}')
    print(f'  Load from : {load_from or "(last episode)"}')

    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    with open(os.path.join(SNAPSHOTS_DIR, 'setup_response.txt'), 'w', encoding='utf-8') as f:
        f.write(text)

    apply_files(new_train, new_gym)
    print('── Setup complete ───────────────────────────────────────────\n')
    return load_from


# ── 10-minute review ───────────────────────────────────────────────────────────

def do_review(check_n, check_min, proc):
    """Returns (decision, needs_restart, load_from_path)."""
    ep_table  = episode_summary_for_llm()
    log_text  = tail_log(LOG_TAIL)
    train_py  = read_file(TRAIN_SCRIPT)
    gym_py    = read_file(GYM_SCRIPT)

    prompt = REVIEW_PROMPT.format(
        check_n=check_n,
        ep_history=EP_HISTORY,
        ep_table=ep_table,
        log_tail=LOG_TAIL,
        log_tail_text=log_text,
        train_py=train_py,
        gym_py=gym_py,
    )
    text = call_llm(prompt)

    decision  = (extract_tag(text, 'decision')  or 'ok').strip().lower()
    analysis  = extract_tag(text, 'analysis')   or ''
    action    = extract_tag(text, 'action')     or ''
    load_from = (extract_tag(text, 'load_from') or '').strip() or None
    new_train = extract_tag(text, 'train_py')
    new_gym   = extract_tag(text, 'gym_torcs_py')

    print(f'\n  Decision  : {decision.upper()}')
    print(f'  Load from : {load_from or "(no change)"}')
    print(f'  Analysis  : {analysis}')
    print(f'  Action    : {action}')

    # Validate load_from path
    if load_from:
        full = load_from if os.path.isabs(load_from) else os.path.join(WORK_DIR, load_from)
        if not os.path.exists(os.path.join(full, 'actor.pth')):
            print(f'  [warning] load_from "{load_from}" has no actor.pth — ignoring')
            load_from = None

    # Save snapshot
    dest = save_snapshot(check_n, label=decision.upper())
    with open(os.path.join(dest, 'analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Decision : {decision}\nLoad from: {load_from or "none"}\n\n'
                f'Analysis:\n{analysis}\n\nAction:\n{action}\n')
    with open(os.path.join(dest, 'llm_response.txt'), 'w', encoding='utf-8') as f:
        f.write(text)

    needs_restart  = False
    effective_load = load_from

    if decision == 'ok':
        pass   # nothing to do

    elif decision == 'change':
        files_changed  = False
        proc_stopped   = False

        if new_train or new_gym:
            stop_proc(proc)
            proc_stopped  = True
            files_changed = apply_files(new_train, new_gym)

        if load_from:
            if proc_stopped:
                # Already stopped — restart with new params + chosen episode
                needs_restart = True
            else:
                # Params unchanged — hot-swap only, no restart needed
                print(f'  Writing load.flag → {load_from} (hot-swap)')
                with open(LOAD_FLAG, 'w') as f:
                    f.write(load_from)

        elif proc_stopped:
            # Params changed (or attempted), no specific episode chosen — restart from last
            needs_restart = True

        if not load_from and not proc_stopped:
            print('  No changes specified — treating as ok')

    else:
        print(f'  Unknown decision "{decision}" — treating as ok')

    return decision, needs_restart, effective_load


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-minutes', type=int, default=CHECK_MINUTES)
    parser.add_argument('--skip-setup',    action='store_true')
    args = parser.parse_args()

    if not os.environ.get('ANTHROPIC_API_KEY'):
        print('ERROR: set ANTHROPIC_API_KEY first.')
        sys.exit(1)

    check_min = args.check_minutes
    print('=' * 62)
    print('  TORCS — Continuous Training + AI Reviews')
    print(f'  Review interval : {check_min} min  |  LLM: {LLM_MODEL}')
    print('=' * 62)

    # ── Initial setup ──
    setup_load_from = None
    if not args.skip_setup:
        setup_load_from = initial_setup()
    else:
        print('  Skipping initial setup.')

    # ── Launch training ──
    start_load = setup_load_from
    if start_load is None:
        _, last_path = last_episode()
        start_load = last_path   # None = fresh if no episodes yet

    check_n    = 1
    proc       = launch_proc(load_from=start_load)
    next_check = time.time() + check_min * 60

    print(f'  Training started. First review in {check_min} min.\n')

    try:
        while True:
            if proc.poll() is not None:
                print(f'  train.py exited ({proc.returncode}) — restarting from last episode')
                with open(LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(f'=== train.py exited ({proc.returncode}), restarting '
                            f'{time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
                _, last_path = last_episode()
                # Pass last_path so train.py sees same last_dir → no branch increment
                proc = launch_proc(load_from=last_path)

            if time.time() >= next_check:
                print(f'\n{"─"*62}')
                print(f'  REVIEW #{check_n}   {time.strftime("%Y-%m-%d %H:%M:%S")}')
                d, p = last_episode()
                print(f'  Last saved: {d}')
                print(f'{"─"*62}')

                try:
                    decision, needs_restart, load_from = do_review(check_n, check_min, proc)
                    if needs_restart:
                        if load_from is None:
                            _, load_from = last_episode()
                        with open(LOG_FILE, 'a', encoding='utf-8') as f:
                            f.write(f'=== Restarting after {decision} '
                                    f'(load: {load_from or "fresh"}) '
                                    f'{time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
                        proc = launch_proc(load_from=load_from)
                except Exception:
                    print('  [error] Review failed:')
                    traceback.print_exc()
                    save_snapshot(check_n, label='ERROR')

                check_n   += 1
                next_check = time.time() + check_min * 60
                print(f'  Next review: #{check_n} in {check_min} min.\n')

            time.sleep(5)

    except KeyboardInterrupt:
        print('\n  Stopping ...')
        stop_proc(proc)


if __name__ == '__main__':
    main()
