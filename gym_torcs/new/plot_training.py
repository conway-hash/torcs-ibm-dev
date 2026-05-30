import re
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LOG_FILE        = 'training_log.txt'
REFRESH_SECONDS = 10

COLOR_MAP = {
    'off_track': '#e07b00',
    'too_slow':  '#d62728',
    'damage':    '#9467bd',
    'finished':  '#2ca02c',
    'unknown':   '#aaaaaa',
}

GRAY = 0.68   # neutral gray level for faded old dots



def _smooth(vals, window):
    if len(vals) < 2:
        return np.array(vals)
    w = max(1, min(window, len(vals) // 4))
    kernel = np.ones(w) / w
    return np.convolve(vals, kernel, mode='same')


def parse_log():
    episodes    = []
    restart_eps = []   # episode indices (into `episodes`) where a branch/session change happened

    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return [], []

    ep_re = re.compile(
        r'Ep\s+ep_(\d+)(?:\.\d+)?\s*\|'
        r'.*?Time\s+(\d+):(\d+\.\d+)\s*\|'
        r'.*?Reward\s+([-\d.]+)\s*\|'
        r'.*?Avg\(10\)\s+([-\d.]+)\s*\|'
        r'(?:.*?Dist\s+([\d.]+)m\s*\(\s*([\d.]+)%\)\s*\|)?'
        r'.*?End:\s+(\w+)'
    )
    restart_re = re.compile(r'=== Session|ai-swap|ai-rollback')

    for line in lines:
        line = line.strip()

        if restart_re.search(line) and episodes:
            restart_eps.append(len(episodes))
            continue

        m = ep_re.search(line)
        if not m:
            continue

        ep_num   = int(m.group(1))
        mins     = int(m.group(2))
        secs     = float(m.group(3))
        reward   = float(m.group(4))
        avg10    = float(m.group(5))
        dist_m   = float(m.group(6)) if m.group(6) else None
        dist_pct = float(m.group(7)) if m.group(7) else None
        reason   = m.group(8)

        episodes.append({
            'ep':       ep_num,
            'time':     mins * 60 + secs,
            'reward':   reward,
            'avg10':    avg10,
            'dist_m':   dist_m,
            'dist_pct': dist_pct,
            'reason':   reason,
        })

    return episodes, restart_eps


GRAY_HEX = f'#{int(GRAY*255):02x}{int(GRAY*255):02x}{int(GRAY*255):02x}'

def _scatter_with_fade(ax, xs, ys, reasons, is_recent, sizes, alphas):
    """Two vectorized scatter calls: one gray batch, one colored batch."""
    xs, ys = list(xs), list(ys)
    old  = [i for i in range(len(xs)) if not is_recent[i]]
    new  = [i for i in range(len(xs)) if is_recent[i]]
    if old:
        ax.scatter([xs[i] for i in old], [ys[i] for i in old],
                   color=GRAY_HEX, s=float(sizes[old[0]]), alpha=0.35,
                   zorder=2, linewidths=0)
    if new:
        ax.scatter([xs[i] for i in new], [ys[i] for i in new],
                   c=[COLOR_MAP.get(reasons[i], '#aaaaaa') for i in new],
                   s=float(sizes[new[0]]), alpha=0.85,
                   zorder=3, linewidths=0)


def draw(episodes, restart_eps, ax1, ax2, ax3):
    ax1.cla(); ax2.cla(); ax3.cla()

    if not episodes:
        ax1.set_title('No data yet')
        return

    n        = len(episodes)
    RECENT_N = 50   # last N episodes get full termination color; rest are gray
    # Boolean mask: True = recent (colored), False = old (gray)
    is_recent = np.array([i >= n - RECENT_N for i in range(n)])
    sizes     = np.where(is_recent, 30, 14).astype(float)
    alphas    = np.where(is_recent, 0.85, 0.35)

    times   = [d['time']   for d in episodes]
    rewards = [d['reward'] for d in episodes]
    avg10s  = [d['avg10']  for d in episodes]
    eps     = [d['ep']     for d in episodes]
    reasons = [d['reason'] for d in episodes]

    # ── Legend patches (shared) ──────────────────────────────────────
    leg_patches = [mpatches.Patch(color=COLOR_MAP[k], label=k)
                   for k in COLOR_MAP if k != 'unknown']

    # ── Graph 1: Track time × Reward (fade to gray) ──────────────────
    _scatter_with_fade(ax1, times, rewards, reasons, is_recent, sizes, alphas)
    ax1.set_xlabel('Episode track time (seconds)', fontsize=10)
    ax1.set_ylabel('Reward', fontsize=10)
    ax1.set_title('Track time  ×  Reward', fontsize=11)
    ax1.grid(True, alpha=0.20)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f}k'))
    ax1.legend(handles=leg_patches, fontsize=8, loc='upper left')

    # ── Graph 2: continuous line on sequential index (no zigzag from branches) ──
    seq = list(range(n))   # 0,1,2,... — strictly increasing, never zigzags
    cut = max(0, n - RECENT_N)
    if cut > 0:
        ax2.plot(seq[:cut + 1], avg10s[:cut + 1],
                 color='#bbbbbb', linewidth=1.0, alpha=0.55, zorder=2)
    ax2.plot(seq[cut:], avg10s[cut:],
             color='#1f77b4', linewidth=2.0, alpha=0.90, zorder=3, label='Avg(10)')

    # Branch / session-change markers (index into seq is already the right x)
    first_branch = True
    for idx in restart_eps:
        if 0 < idx < n:
            lbl = 'Branch / restart' if first_branch else None
            first_branch = False
            ax2.axvline(x=idx, color='#888888', linestyle=':', linewidth=1.0,
                        alpha=0.55, zorder=2, label=lbl)

    ax2.set_xlabel('Training step (sequential)', fontsize=10)
    ax2.set_ylabel('Avg(10) Reward', fontsize=10)
    ax2.set_title('Avg(10) rolling average', fontsize=11)
    ax2.grid(True, alpha=0.20)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f}k'))
    ax2.legend(fontsize=8, loc='upper left')

    # ── Graph 3: Track time × % of track (fade to gray) ──────────────
    pct_eps = [(i, d) for i, d in enumerate(episodes) if d['dist_pct'] is not None]
    if pct_eps:
        idxs, pdata = zip(*pct_eps)
        _scatter_with_fade(
            ax3,
            [d['time']     for d in pdata],
            [d['dist_pct'] for d in pdata],
            [d['reason']   for d in pdata],
            is_recent[list(idxs)],
            sizes[list(idxs)],
            alphas[list(idxs)],
        )
        ax3.legend(handles=leg_patches, fontsize=8, loc='upper left')
    else:
        ax3.set_title('Track % — no dist data yet')

    ax3.set_xlabel('Episode track time (seconds)', fontsize=10)
    ax3.set_ylabel('Track coverage (%)', fontsize=10)
    ax3.set_title('Track time  ×  % of track covered', fontsize=11)
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.20)


def main():
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle('TORCS TD3 Training', fontsize=13, fontweight='bold')
    fig.tight_layout(pad=3.0)

    _save_tick = 0
    while plt.fignum_exists(fig.number):
        episodes, restart_eps = parse_log()
        draw(episodes, restart_eps, ax1, ax2, ax3)
        _save_tick += 1
        if _save_tick % 3 == 0:   # save PNG every 3rd refresh (every 30s)
            try:
                fig.savefig('training_progress.png', dpi=100, bbox_inches='tight')
            except Exception:
                pass
        plt.pause(REFRESH_SECONDS)


if __name__ == '__main__':
    main()
