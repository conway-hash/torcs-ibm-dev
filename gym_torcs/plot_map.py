#!/usr/bin/env python
# plot_map.py
#
# Visualise a TrackMap's global racing line and speed profile, so you can eyeball
# how good the line is: where it cuts apexes, how much track width it uses, and
# how the target speed varies around the lap.
#
# The map stores only (kappa(s), width(s), racing-line offset n*(s), speed
# v*(s)) on an arc-length grid - no global X,Y. We rebuild the 2-D geometry by
# integrating the centreline heading from the curvature, then draw:
#   * the track edges (grey),
#   * the centreline (thin dashed),
#   * the racing line, coloured by target speed (blue = slow, red = fast),
#   * a start/finish marker.
#
# Usage:
#   python plot_map.py track_map.json [out.png]      # a built map
#   python plot_map.py lap.csv [out.png]             # build from a mapping lap first
#   python plot_map.py --demo square [out.png]       # ground-truth map of a sim track
#   python plot_map.py --demo oval

import sys
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')                       # save to file, no display needed
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import track_map as tm


def reconstruct_xy(m):
    """Integrate the centreline (X, Y, heading) from the stored curvature.

    This is for *drawing only* - the controller uses arc-length n*(s)/v*(s) and
    never these X,Y. Integrating noisy curvature accumulates drift, so the loop
    usually doesn't land exactly back on its start; we distribute that closure
    error evenly so the picture closes instead of showing a long straight chord
    across the gap. The size of the raw gap is a handy quality indicator."""
    s = m.s
    ds = float(s[1] - s[0])
    th = np.cumsum(m.kappa) * ds
    cx = np.cumsum(np.cos(th)) * ds
    cy = np.cumsum(np.sin(th)) * ds
    n = cx.shape[0]
    gap = float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]))
    # linear closure correction (cx[-1]->cx[0], cy[-1]->cy[0])
    cx = cx - np.linspace(0.0, cx[-1] - cx[0], n)
    cy = cy - np.linspace(0.0, cy[-1] - cy[0], n)
    nx, ny = -np.sin(th), np.cos(th)        # left normal (+ offset = left)
    return cx, cy, nx, ny, gap


def plot_map(m, out_path, title=None):
    cx, cy, nx, ny, gap = reconstruct_xy(m)
    hw = m.width * 0.5
    lx, ly = cx + nx * hw, cy + ny * hw     # left edge
    rx, ry = cx - nx * hw, cy - ny * hw     # right edge
    rlx, rly = cx + nx * m.n_rl, cy + ny * m.n_rl   # racing line

    fig, ax = plt.subplots(figsize=(12, 12))

    # track edges (close the loop visually)
    ax.plot(np.r_[lx, lx[0]], np.r_[ly, ly[0]], color='0.5', lw=1.0)
    ax.plot(np.r_[rx, rx[0]], np.r_[ry, ry[0]], color='0.5', lw=1.0)
    ax.fill(np.r_[lx, lx[0]], np.r_[ly, ly[0]], color='0.93', zorder=0)
    ax.fill(np.r_[rx, rx[0]], np.r_[ry, ry[0]], color='white', zorder=0)
    # centreline
    ax.plot(np.r_[cx, cx[0]], np.r_[cy, cy[0]], '--', color='0.75', lw=0.8)

    # racing line coloured by target speed
    pts = np.column_stack([np.r_[rlx, rlx[0]], np.r_[rly, rly[0]]])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    spd = np.r_[m.v_target, m.v_target[0]]
    lc = LineCollection(segs, cmap='turbo', linewidth=3.0)
    lc.set_array(0.5 * (spd[:-1] + spd[1:]) * 3.6)   # km/h
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('target speed (km/h)')

    # start / finish
    ax.plot(rlx[0], rly[0], 'ko', ms=8)
    ax.annotate('start', (rlx[0], rly[0]), textcoords='offset points',
                xytext=(8, 8))

    ax.set_aspect('equal', 'datalim')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    vmin, vmax = m.v_target.min() * 3.6, m.v_target.max() * 3.6
    ax.set_title(title or ('Racing line  |  length %.0f m  |  speed %.0f-%.0f km/h'
                           % (m.length, vmin, vmax)))
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print("saved -> %s   (length %.0f m, offset %.1f..%.1f m, speed %.0f-%.0f km/h)"
          % (out_path, m.length, m.n_rl.min(), m.n_rl.max(), vmin, vmax))
    print("   reconstruction closure gap: %.0f m (%.1f%% of length) - "
          "drawing-only; large => curvature/loop-closure noise in the lap"
          % (gap, 100.0 * gap / m.length))


def load_or_build(arg):
    if arg.endswith('.json'):
        return tm.TrackMap.load(arg)
    if arg.endswith('.csv'):
        return tm.from_map_log(arg, ds=2.0).build()
    raise ValueError("give a .json map, a .csv mapping-lap, or --demo TRACK")


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    if argv[0] == '--demo':
        from sim_test import build_track, TRACKS, ground_truth_map
        name = argv[1] if len(argv) > 1 else 'square'
        out = argv[2] if len(argv) > 2 else ('map_%s.png' % name)
        trk = build_track(TRACKS[name], width=15.0)
        m = ground_truth_map(trk)
        plot_map(m, out, title='Racing line (demo: %s)' % name)
        return 0
    src = argv[0]
    out = argv[1] if len(argv) > 1 else os.path.splitext(src)[0] + '.png'
    plot_map(load_or_build(src), out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
