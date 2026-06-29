# eval.py — random_test
# Replay the best ES policy and report its lap time. Safe to run while
# train_es.py is going (uses a free scr_server slot above the workers).
#
#   python eval.py --gui                 # watch best.npz drive a lap (TORCS menu opens)
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
from launcher import ensure_car, kill_all_torcs, TorcsInstance


def free_port_index():
    return min(CFG.n_workers + 1, 9)


def main():
    ap = argparse.ArgumentParser(description='Replay the best ES policy')
    ap.add_argument('--theta', default=os.path.join(CFG.run_dir, 'best.npz'),
                    help='policy .npz (default best.npz; try es_state.npz too)')
    ap.add_argument('--port-index', type=int, default=None)
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--gui', action='store_true',
                    help='open the TORCS game window so you can watch the AI drive')
    ap.add_argument('--attach', action='store_true',
                    help='connect to a TORCS race YOU already launched manually')
    args = ap.parse_args()

    ensure_car(CFG)
    if not args.attach:
        kill_all_torcs()
        time.sleep(1.5)

    if not os.path.exists(args.theta):
        raise SystemExit('not found: %s  (has training saved a best lap yet?)'
                         % args.theta)
    data = np.load(args.theta, allow_pickle=True)
    theta = data['theta'].astype(np.float32)
    rec_lap = float(data['lap_time'][0]) if 'lap_time' in data else None
    print('loaded %s%s' % (args.theta,
          '  (recorded lap %.3fs)' % rec_lap if rec_lap else ''))

    gui_proc = None

    if args.gui and not args.attach:
        # Launch TORCS in menu mode (no -r flag) so the graphics window appears.
        # The user starts the race manually; the agent connects automatically.
        inst = TorcsInstance(0, CFG, gui=True)
        inst.spawn_menu()
        gui_proc = inst.proc
        port_index = 0   # scr_server 0 = port 3001
        print()
        print('=' * 60)
        print('  TORCS window is opening (allow ~10 seconds).')
        print()
        print('  In TORCS:  Race  ->  Quick Race  ->  New Race')
        print()
        print('  The AI connects automatically once the race loads.')
        print('=' * 60)
        print()
        # Use attach mode so TorcsEnv waits for the user-started race
        attach = True
    else:
        attach = args.attach
        if args.port_index is not None:
            port_index = args.port_index
        elif attach:
            port_index = 0
        else:
            port_index = free_port_index()

        if attach:
            print('[attach] waiting for TORCS on port %d...'
                  % CFG.port_for(port_index))

    pol = Policy(CFG).set_flat(theta)
    env = TorcsEnv(port_index, CFG, gui=False, attach=attach)

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
        try:
            env.client.send_restart()
        except Exception:
            pass
        time.sleep(5)
        env.close()
        if gui_proc is not None:
            try:
                gui_proc.kill()
            except Exception:
                pass

    if laps:
        print('\nbest=%.3fs  mean=%.3fs  (%d/%d clean)'
              % (min(laps), float(np.mean(laps)), len(laps), args.episodes))


if __name__ == '__main__':
    main()
