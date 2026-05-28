import re
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LOG_FILE = 'training_log.txt'
REFRESH_SECONDS = 10

COLOR_MAP = {
    'off_track': '#e07b00',
    'too_slow':  '#d62728',
    'damage':    '#9467bd',
    'finished':  '#2ca02c',
    'unknown':   '#aaaaaa',
}


def parse_log():
    episodes = []
    rollback_segs = []   # list of (ep_before, low_avg, peak_avg)
    pending_rollback = None

    cumulative_offset = 0
    prev_ep_num = -1

    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return [], []

    for line in lines:
        line = line.strip()

        rb = re.search(r'\[rollback\].*?avg\s+([\d.]+).*?peak\s+([\d.]+)', line)
        if rb:
            pending_rollback = (float(rb.group(1)), float(rb.group(2)))
            continue

        m = re.match(
            r'Ep\s+(\d+)\s*\|'
            r'.*?Time\s+(\d+):(\d+\.\d+)\s*\|'
            r'.*?Reward\s+([-\d.]+)\s*\|'
            r'.*?Avg\(10\)\s+([-\d.]+)\s*\|'
            r'.*?End:\s+(\w+)',
            line
        )
        if not m:
            continue

        ep_num   = int(m.group(1))
        mins     = int(m.group(2))
        secs     = float(m.group(3))
        reward   = float(m.group(4))
        avg10    = float(m.group(5))
        reason   = m.group(6)
        t_sec    = mins * 60 + secs

        if ep_num == 0 and prev_ep_num > 0:
            cumulative_offset += prev_ep_num + 1
        prev_ep_num = ep_num
        abs_ep = cumulative_offset + ep_num

        if pending_rollback is not None:
            # rollback fired after the previous episode
            if episodes:
                low_avg, peak_avg = pending_rollback
                rollback_segs.append((episodes[-1]['ep'], low_avg, peak_avg))
            pending_rollback = None

        episodes.append({
            'ep':     abs_ep,
            'time':   t_sec,
            'reward': reward,
            'avg10':  avg10,
            'reason': reason,
        })

    return episodes, rollback_segs


def draw(episodes, rollback_segs, ax1, ax2):
    ax1.cla()
    ax2.cla()

    if not episodes:
        ax1.set_title('No data yet')
        return

    eps     = [d['ep']     for d in episodes]
    times   = [d['time']   for d in episodes]
    rewards = [d['reward'] for d in episodes]
    avg10s  = [d['avg10']  for d in episodes]
    reasons = [d['reason'] for d in episodes]

    # Determine which episodes belong to the current (latest) segment
    last_rollback_ep = rollback_segs[-1][0] if rollback_segs else -1
    colors = []
    alphas = []
    sizes  = []
    for d in episodes:
        if d['ep'] > last_rollback_ep:
            colors.append(COLOR_MAP.get(d['reason'], '#aaaaaa'))
            alphas.append(0.80)
            sizes.append(28)
        else:
            colors.append('#bbbbbb')   # grayscale for pre-rollback episodes
            alphas.append(0.35)
            sizes.append(18)

    # ── Graph 1: Track time vs Reward ────────────────────────────────
    # Plot old (greyed) dots first, then current segment on top
    old_t = [times[i]   for i, d in enumerate(episodes) if d['ep'] <= last_rollback_ep]
    old_r = [rewards[i] for i, d in enumerate(episodes) if d['ep'] <= last_rollback_ep]
    cur_t = [times[i]   for i, d in enumerate(episodes) if d['ep'] > last_rollback_ep]
    cur_r = [rewards[i] for i, d in enumerate(episodes) if d['ep'] > last_rollback_ep]
    cur_c = [COLOR_MAP.get(d['reason'], '#aaaaaa')
             for d in episodes if d['ep'] > last_rollback_ep]

    if old_t:
        ax1.scatter(old_t, old_r, c='#cccccc', s=18, alpha=0.35, zorder=2, label='before rollback')
    if cur_t:
        ax1.scatter(cur_t, cur_r, c=cur_c, s=28, alpha=0.80, zorder=3)

    ax1.set_xlabel('Episode track time (seconds)', fontsize=10)
    ax1.set_ylabel('Reward', fontsize=10)
    ax1.set_title('Track time  ×  Reward', fontsize=11)
    ax1.grid(True, alpha=0.25)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f}k'))

    handles = [mpatches.Patch(color=c, label=l) for l, c in COLOR_MAP.items() if l != 'unknown']
    if old_t:
        handles.append(mpatches.Patch(color='#cccccc', label='before rollback'))
    ax1.legend(handles=handles, fontsize=8, loc='upper left')

    # ── Graph 2: Avg(10) with rollback segments ───────────────────────
    # Split episodes into segments at each rollback point
    split_eps = {rb[0] for rb in rollback_segs}

    seg_ep, seg_avg = [], []
    segments = []

    for d in episodes:
        seg_ep.append(d['ep'])
        seg_avg.append(d['avg10'])
        if d['ep'] in split_eps:
            segments.append((seg_ep[:], seg_avg[:]))
            seg_ep, seg_avg = [], []

    if seg_ep:
        segments.append((seg_ep, seg_avg))

    for i, (se, sa) in enumerate(segments):
        ax2.plot(se, sa, color='#1f77b4', linewidth=1.6,
                 label='Avg(10)' if i == 0 else None)

    # Draw dashed vertical restore lines at each rollback
    first_rb = True
    for ep_before, low_avg, peak_avg in rollback_segs:
        lbl = 'Rollback restore' if first_rb else None
        first_rb = False
        ax2.plot([ep_before, ep_before], [low_avg, peak_avg],
                 color='#d62728', linestyle='--', linewidth=1.8,
                 label=lbl, zorder=4)
        ax2.axvline(x=ep_before, color='#d62728', linestyle=':', alpha=0.35, zorder=2)

    # Horizontal line at all-time best avg10
    best = max(avg10s)
    ax2.axhline(y=best, color='#2ca02c', linestyle='--', linewidth=1.0,
                alpha=0.6, label=f'Peak avg {best/1000:.0f}k')

    ax2.set_xlabel('Episode', fontsize=10)
    ax2.set_ylabel('Avg(10) Reward', fontsize=10)
    ax2.set_title('Rolling Average (10 eps) — dashed = rollback restore', fontsize=11)
    ax2.grid(True, alpha=0.25)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1000:.0f}k'))
    ax2.legend(fontsize=8)


def main():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('TORCS TD3 Training', fontsize=13, fontweight='bold')
    plt.tight_layout(pad=3.0)

    while True:
        episodes, rollback_segs = parse_log()
        draw(episodes, rollback_segs, ax1, ax2)
        plt.tight_layout(pad=3.0)
        plt.draw()
        plt.pause(0.1)
        fig.savefig('training_progress.png', dpi=130, bbox_inches='tight')

        # wait REFRESH_SECONDS, checking for window close
        for _ in range(REFRESH_SECONDS * 10):
            if not plt.fignum_exists(fig.number):
                return
            plt.pause(0.1)


if __name__ == '__main__':
    main()
