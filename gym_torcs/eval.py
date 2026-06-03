# eval.py — random_test
# Replay the best ES policy and report its lap time. Safe to run while
# train_es.py is going (uses a free scr_server slot above the workers).
#
#   python eval.py --gui                 # watch best.npz drive a lap
#   python eval.py                       # headless, just report the time
#   python eval.py --gui --episodes 5    # repeatability check
#   python eval.py --theta runs/corkscrew_es/best.npz
import argparse
import os
import time
import numpy as np

from config import CFG
from policy import Policy
from torcs_env import TorcsEnv
from launcher import ensure_car


def free_port_index():
    return min(CFG.n_workers + 1, 9)


def main():
    ap = argparse.ArgumentParser(description='Replay the best ES policy')
    ap.add_argument('--theta', default=os.path.join(CFG.run_dir, 'best.npz'),
                    help='policy .npz (default best.npz; try es_state.npz too)')
    ap.add_argument('--port-index', type=int, default=None)
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--gui', action='store_true',
                    help='auto-launch a TORCS window (drops -T)')
    ap.add_argument('--attach', action='store_true',
                    help='connect to a TORCS race YOU launched (it is not '
                         'launched/killed here). Use when --gui shows no window.')
    args = ap.parse_args()

    ensure_car(CFG)   # guarantee we evaluate the SAME car training used

    if not os.path.exists(args.theta):
        raise SystemExit('not found: %s  (has training saved a best lap yet?)'
                         % args.theta)
    data = np.load(args.theta, allow_pickle=True)
    theta = data['theta'].astype(np.float32)
    rec_lap = float(data['lap_time'][0]) if 'lap_time' in data else None
    print('loaded %s%s' % (args.theta,
          '  (recorded lap %.3fs)' % rec_lap if rec_lap else ''))

    # In attach mode default to slot 0 (the race you launched uses scr_server
    # idx 0 -> port 3001); otherwise pick a free slot above the workers.
    if args.port_index is not None:
        port_index = args.port_index
    elif args.attach:
        port_index = 0
    else:
        port_index = free_port_index()

    if args.attach:
        print('[attach] will connect to a TORCS race on port %d that YOU '
              'launched. If not running yet, launch it now '
              '(python launcher.py --gui-race).' % CFG.port_for(port_index))

    pol = Policy(CFG).set_flat(theta)
    env = TorcsEnv(port_index, CFG, gui=args.gui, attach=args.attach)

    laps = []
    try:
        for ep in range(args.episodes):
            obs = env.reset()
            done, steps, dist, lap = False, 0, 0.0, None
            while not done:
                obs, _, done, info = env.step(pol.act(obs))
                steps += 1
                dist = info.get('dist_raced', dist)
                if 'lap_time' in info:
                    lap = info['lap_time']
            if lap:
                laps.append(lap)
                print('ep %d: LAP %.3fs  (%d steps)' % (ep, lap, steps))
            else:
                why = ('offtrack' if info.get('offtrack') else
                       'stall' if info.get('stall') else
                       'timeout' if info.get('timeout') else 'ended')
                print('ep %d: NO LAP (%s)  dist=%.0fm' % (ep, why, dist))
    finally:
        time.sleep(20)
        env.close()

    if laps:
        print('\nbest=%.3fs  mean=%.3fs  (%d/%d clean)'
              % (min(laps), float(np.mean(laps)), len(laps), args.episodes))


if __name__ == '__main__':
    main()
