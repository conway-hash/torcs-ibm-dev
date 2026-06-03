# grow_model.py — random_test
# ---------------------------------------------------------------------------
# Function-preserving network growth (Net2Wider). Widens the policy's single
# hidden layer so the BIGGER network reproduces the current best lap EXACTLY at
# initialization, then ES has extra capacity to recruit.
#
# New hidden units get random INPUT weights (so they compute new features) but
# ZERO OUTPUT weights (so they contribute nothing to the action yet) -> the
# function is identical to the old policy. ES grows their output weights to
# "wake them up", exploring directions the smaller net could not represent.
#
#   python grow_model.py --hidden 64        # [32] -> [64], builds on best.npz
#   python grow_model.py --hidden 96
#
# Afterwards set CFG.hidden_sizes = [<new>] (done for you if you use the
# default 64) and run:  python train_es.py --fresh
# ---------------------------------------------------------------------------
import argparse
import os
import shutil
import stat
import time
import numpy as np

from config import CFG

OBS, ACT = CFG.obs_dim, CFG.act_dim
PER_H = OBS + 1 + ACT          # params contributed per hidden unit (single layer)


def infer_hidden(n_params):
    return int(round((n_params - ACT) / PER_H))


def _forward(theta, H, obs):
    """Manual tanh-MLP forward for a single-hidden-layer policy (verification)."""
    k = 0
    W0 = theta[k:k + OBS * H].reshape(OBS, H); k += OBS * H
    b0 = theta[k:k + H]; k += H
    W1 = theta[k:k + H * ACT].reshape(H, ACT); k += H * ACT
    b1 = theta[k:k + ACT]
    h = np.tanh(obs @ W0 + b0)
    return np.tanh(h @ W1 + b1)


def grow(theta, Ho, Hn, rng):
    assert Hn >= Ho, "new width must be >= old width"
    k = 0
    W0 = theta[k:k + OBS * Ho].reshape(OBS, Ho); k += OBS * Ho
    b0 = theta[k:k + Ho]; k += Ho
    W1 = theta[k:k + Ho * ACT].reshape(Ho, ACT); k += Ho * ACT
    b1 = theta[k:k + ACT]

    W0n = np.zeros((OBS, Hn), np.float32)
    W0n[:, :Ho] = W0
    W0n[:, Ho:] = (rng.randn(OBS, Hn - Ho) / np.sqrt(OBS)).astype(np.float32)
    b0n = np.zeros(Hn, np.float32); b0n[:Ho] = b0
    W1n = np.zeros((Hn, ACT), np.float32)
    W1n[:Ho, :] = W1                       # new rows = 0 -> output unchanged
    b1n = b1.copy()
    return np.concatenate([W0n.ravel(), b0n, W1n.ravel(), b1n]).astype(np.float32)


def main():
    runs_parent = os.path.dirname(CFG.run_dir)
    ap = argparse.ArgumentParser(description="Function-preserving policy growth")
    ap.add_argument('--hidden', type=int, default=64, help='new hidden width')
    ap.add_argument('--in', dest='inp',
                    default=os.path.join(runs_parent, 'corkscrew_es', 'best.npz'),
                    help='source checkpoint to grow FROM (default: the old '
                         '[32] run best.npz)')
    ap.add_argument('--out-dir', default=CFG.run_dir,
                    help='where to write the grown checkpoint (default: '
                         'CFG.run_dir = the new dir)')
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise SystemExit('not found: %s' % args.inp)
    # IMPORTANT: np.load on an .npz returns a lazy NpzFile that holds the file
    # OPEN. Use a with-block and copy the arrays out, so the handle is closed
    # before we (possibly) overwrite the same file -- otherwise the process
    # blocks itself (WinError 32).
    with np.load(args.inp, allow_pickle=True) as d:
        theta = d['theta'].astype(np.float32).copy()
        lap = float(d['lap_time'][0]) if 'lap_time' in d else float('nan')
    Ho = infer_hidden(theta.size)
    Hn = args.hidden
    print('growing [%d] -> [%d]  (was %.3fs, %d params -> %d params)'
          % (Ho, Hn, lap, theta.size, Hn * PER_H + ACT))

    rng = np.random.RandomState(0)
    big = grow(theta, Ho, Hn, rng)

    # verify EXACT function preservation
    obs = rng.randn(200, OBS).astype(np.float32)
    a_old = _forward(theta, Ho, obs)
    a_new = _forward(big, Hn, obs)
    max_err = float(np.max(np.abs(a_old - a_new)))
    print('function-preservation max action error: %.2e %s'
          % (max_err, 'OK' if max_err < 1e-5 else 'FAILED'))
    if max_err >= 1e-5:
        raise SystemExit('growth did not preserve the function — aborting.')

    # Write into a FRESH directory (default CFG.run_dir). Brand-new files can't
    # be held open by the indexer/AV, so this sidesteps the lock on the old
    # run's best.npz entirely. The original [%d] run is left untouched.
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    def _robust_save(name, **arrays):
        dst = os.path.join(out, name)
        last = None
        for attempt in range(10):
            try:
                if os.path.exists(dst):
                    try:
                        os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)
                    except Exception:
                        pass
                    os.remove(dst)
                np.savez(dst, **arrays)      # dst ends in .npz -> exact name
                return
            except PermissionError as e:
                last = e
                time.sleep(0.5 * (attempt + 1))
        raise SystemExit('\nERROR: could not write %s (%s).' % (dst, last))

    _robust_save('best.npz', theta=big, lap_time=np.array([lap]),
                 generation=np.array([0]))
    _robust_save('init_theta.npz', theta=big)

    print('\nwrote grown best.npz + init_theta.npz [%d] -> %s' % (Hn, out))
    print('(original [%d] run preserved untouched in %s)'
          % (Ho, os.path.dirname(args.inp)))
    print('\nNext:  python train_es.py --fresh')


if __name__ == '__main__':
    main()
