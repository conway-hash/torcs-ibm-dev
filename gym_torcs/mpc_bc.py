# mpc_bc.py — random_test
# ---------------------------------------------------------------------------
# Warm-start the ES policy by behavior-cloning the MPC (../gym_torcs/
# torcs_mpc.py). The MPC is a competent racing-line driver; cloning it gives ES
# a policy that already drives well, cutting overnight convergence from "learn
# to steer from scratch" to "refine an already-good lap".
#
#   python mpc_bc.py                      # record MPC + clone -> init_theta.npz
#   python mpc_bc.py --attempts 6 --epochs 300
#   python mpc_bc.py --gui                # watch the MPC while it records
#
# Output: runs/corkscrew_es/init_theta.npz  (train_es.py auto-loads it).
# Produces a Policy-compatible flat parameter vector; no effect on the ES loop
# beyond initialization, so if the MPC is unavailable the trainer still runs
# from the throttle-floored random init.
# ---------------------------------------------------------------------------
import argparse
import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

from config import CFG
# Import the MPC from whichever dir the config points at (the better "copy").
sys.path.insert(0, CFG.mpc_gym_dir)
from launcher import kill_all_torcs
from torcs_env import TorcsEnv
from policy import Policy


# ---------------------------------------------------------------------------
# MPC expert
# ---------------------------------------------------------------------------

def build_mpc():
    try:
        import torcs_mpc as M
        car = None
        try:
            from car_profile import CarProfile
            if os.path.exists(CFG.mpc_car_profile):
                car = CarProfile.load(CFG.mpc_car_profile)
        except Exception:
            pass
        track_map = None
        try:
            from track_map import TrackMap
            if os.path.exists(CFG.mpc_track_map):
                track_map = TrackMap.load(CFG.mpc_track_map)
        except Exception:
            pass
        dyn = None
        try:
            if getattr(CFG, 'mpc_dynamics', None) and os.path.exists(CFG.mpc_dynamics):
                dyn = M.load_dynamics(CFG.mpc_dynamics)
        except Exception:
            pass
        driver = M.MPCDriver(dynamics=dyn, track_map=track_map, car=car)
        print('[mpc] ready (car_profile=%s track_map=%s dynamics=%s)'
              % (car is not None, track_map is not None, dyn is not None),
              flush=True)
        return driver
    except Exception as e:
        print('[mpc] UNAVAILABLE: %s' % e, flush=True)
        return None


def mpc_action(driver, S):
    Sc = dict(S)
    Sc['speedX'] = S.get('speedX', 0.0) / 3.6
    Sc['speedY'] = S.get('speedY', 0.0) / 3.6
    tr = S.get('track')
    if isinstance(tr, list):
        Sc['track'] = list(reversed(tr))
    act = driver.control(Sc)
    st = float(act.get('steer', 0.0))
    ac = float(act.get('accel', 0.0))
    br = float(act.get('brake', 0.0))
    return np.array([st, 2.0 * ac - 1.0, 2.0 * br - 1.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Record demonstrations
# ---------------------------------------------------------------------------

def record(driver, attempts, gui, min_dist=500.0):
    """Record MPC laps, KEEPING ONLY attempts that drove >= min_dist. The MPC
    is high-variance through this interface (some laps reach ~2400m, others die
    at the first corner ~40m); cloning the failures would teach the policy to
    drive off, so we discard them. Each attempt relaunches a fresh TORCS to cut
    cross-episode state drift."""
    kill_all_torcs(); time.sleep(1.0)
    env = TorcsEnv(0, CFG, gui=gui)
    kept_obs, kept_act = [], []
    kept, best = 0, 0.0
    try:
        for k in range(attempts):
            obs = env.reset(relaunch=(k > 0))    # fresh TORCS each attempt
            a_obs, a_act = [], []
            steps, dist, lap = 0, 0.0, None
            while True:
                raw = env.client.S.d
                a = mpc_action(driver, raw)
                a_obs.append(obs.copy())
                a_act.append(a.copy())
                obs, _, done, info = env.step(
                    a, action_fn=lambda S: mpc_action(driver, S))
                steps += 1
                dist = info.get('dist_raced', dist)
                if 'lap_time' in info:
                    lap = info['lap_time']
                if done:
                    break
            best = max(best, dist)
            keep = dist >= min_dist
            if keep:
                kept_obs.extend(a_obs)
                kept_act.extend(a_act)
                kept += 1
            tag = ('LAP %.1fs' % lap if lap else 'dist %.0fm' % dist)
            print('[mpc] attempt %d/%d: %s (%d steps) -> %s'
                  % (k + 1, attempts, tag, steps,
                     'KEPT' if keep else 'discarded (<%.0fm)' % min_dist),
                  flush=True)
    finally:
        env.close(); kill_all_torcs()
    print('[mpc] kept %d/%d attempts (best %.0fm) -> %d clean samples'
          % (kept, attempts, best, len(kept_obs)), flush=True)
    if kept == 0:
        print('[mpc] WARNING: no attempt reached %.0fm. The MPC is failing '
              'early -- try more --attempts or lower --min-dist.' % min_dist,
              flush=True)
    return np.array(kept_obs, np.float32), np.array(kept_act, np.float32)


# ---------------------------------------------------------------------------
# Behavior cloning (torch model with arch matching policy.Policy)
# ---------------------------------------------------------------------------

def clone(obs, act, epochs, out_path):
    import torch
    import torch.nn as nn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden = list(CFG.hidden_sizes)
    layers, prev = [], CFG.obs_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers += [nn.Linear(prev, CFG.act_dim)]
    net = nn.Sequential(*layers).to(device)

    def forward(x):
        return torch.tanh(net(x))      # match Policy.act final tanh

    X = torch.tensor(obs, device=device)
    Y = torch.tensor(act, device=device)
    n_val = max(1, int(0.1 * len(X)))
    perm = torch.randperm(len(X))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    print('\n[bc] training on %d samples (%d val)  epochs=%d  device=%s'
          % (len(X), n_val, epochs, device))
    best_val, best_state = float('inf'), None
    for ep in range(1, epochs + 1):
        net.train()
        idx = tr_idx[torch.randperm(len(tr_idx))]
        tot = 0.0
        for i in range(0, len(idx), CFG.es_popsize * 8):
            b = idx[i:i + CFG.es_popsize * 8]
            pred = forward(X[b])
            loss = ((pred - Y[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(b)
        net.eval()
        with torch.no_grad():
            vloss = float(((forward(X[val_idx]) - Y[val_idx]) ** 2).mean())
        if vloss < best_val:
            best_val = vloss
            best_state = [p.detach().cpu().clone() for p in net.parameters()]
        if ep % 25 == 0 or ep == 1:
            print('[bc] epoch %3d  train=%.5f  val=%.5f  best=%.5f'
                  % (ep, tot / len(tr_idx), vloss, best_val), flush=True)

    # restore best, export flat theta in Policy order (per layer: W=weight.T, b)
    params = best_state
    flat = []
    # params come as [w0,b0,w1,b1,...]; weight shape (out,in) -> .T (in,out)
    for li in range(0, len(params), 2):
        W = params[li].numpy().T.astype(np.float32)     # (in, out)
        b = params[li + 1].numpy().astype(np.float32)
        flat.append(W.ravel()); flat.append(b)
    theta = np.concatenate(flat).astype(np.float32)

    # sanity: must match Policy size
    pol = Policy(CFG)
    assert theta.size == pol.n_params(), \
        'param mismatch %d != %d' % (theta.size, pol.n_params())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + '.tmp.npz'         # .npz so np.savez doesn't re-append it
    np.savez(tmp, theta=theta, val_loss=np.array([best_val]))
    os.replace(tmp, out_path)
    print('\n[bc] saved warm-start -> %s  (val_loss=%.5f)' % (out_path, best_val))
    return theta


def main():
    ap = argparse.ArgumentParser(description='MPC behavior-cloning warm-start')
    ap.add_argument('--attempts', type=int, default=15,
                    help='MPC laps to TRY (failures are discarded; the MPC is '
                         'high-variance so try plenty)')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--min-dist', type=float, default=500.0,
                    help='discard attempts that drove less than this (m); keeps '
                         'first-corner crashes out of the clone')
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--out', type=str,
                    default=os.path.join(CFG.run_dir, 'init_theta.npz'))
    args = ap.parse_args()

    driver = build_mpc()
    if driver is None:
        sys.exit('MPC unavailable — cannot warm-start. Train from scratch with '
                 'python train_es.py (slower but works).')

    obs, act = record(driver, args.attempts, args.gui, args.min_dist)
    if len(obs) < 100:
        sys.exit('Too few clean demo samples (%d). The MPC kept failing early; '
                 'rerun with more --attempts or a lower --min-dist.' % len(obs))
    print('[bc] recorded %d state-action pairs' % len(obs))
    clone(obs, act, args.epochs, args.out)
    print('\nNext: python train_es.py     (auto-loads init_theta.npz)')


if __name__ == '__main__':
    main()
