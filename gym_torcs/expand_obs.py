# expand_obs.py — evo-with-map
# ---------------------------------------------------------------------------
# Function-preserving OBSERVATION expansion. Takes a trained policy from the
# 33-dim-obs pipeline and lifts it into this project's 51-dim obs (the same 33
# first, plus 18 new racing-line lookahead features APPENDED at the end).
#
# The new input weights are set to ZERO, so the bigger network reproduces the
# old policy EXACTLY at init (the 18 new features are dormant). ES then grows
# those weights to recruit the anticipatory features and push past 75.33s.
#
#   python expand_obs.py --in ../random_test-77secs/runs/corkscrew_es_mpc89/best.npz
#
# Writes best.npz + init_theta.npz (51-dim) into CFG.run_dir, then:
#   python train_es.py --fresh
# ---------------------------------------------------------------------------
import argparse
import os
import stat
import time
import numpy as np

from config import CFG

ACT = CFG.act_dim
H = CFG.hidden_sizes[0]                     # single hidden layer assumed
PER_OUT = H + H * ACT + ACT                 # params not in the input layer


def infer_in(n_params):
    return int(round((n_params - PER_OUT) / H))


def _unpack(theta, in_dim):
    k = 0
    W0 = theta[k:k + in_dim * H].reshape(in_dim, H); k += in_dim * H
    b0 = theta[k:k + H]; k += H
    W1 = theta[k:k + H * ACT].reshape(H, ACT); k += H * ACT
    b1 = theta[k:k + ACT]
    return W0, b0, W1, b1


def expand(theta, old_in, new_in):
    assert new_in >= old_in
    W0, b0, W1, b1 = _unpack(theta, old_in)
    W0n = np.zeros((new_in, H), dtype=np.float32)
    W0n[:old_in] = W0                       # new rows (appended obs) = 0 -> dormant
    return np.concatenate([W0n.ravel(), b0, W1.ravel(), b1]).astype(np.float32)


def _fwd(theta, in_dim, obs):
    W0, b0, W1, b1 = _unpack(theta, in_dim)
    return np.tanh(np.tanh(obs @ W0 + b0) @ W1 + b1)


def main():
    ap = argparse.ArgumentParser(description="Function-preserving obs expansion")
    ap.add_argument('--in', dest='inp', required=True,
                    help='source best.npz from the 33-dim-obs run')
    ap.add_argument('--out-dir', default=CFG.run_dir)
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise SystemExit('not found: %s' % args.inp)
    with np.load(args.inp, allow_pickle=True) as d:     # close handle (lock-safe)
        theta = d['theta'].astype(np.float32).copy()
        lap = float(d['lap_time'][0]) if 'lap_time' in d else float('nan')

    old_in = infer_in(theta.size)
    new_in = CFG.obs_dim
    print('expanding obs %d -> %d  (was %.3fs, %d -> %d params, hidden=%d)'
          % (old_in, new_in, lap, theta.size, new_in * H + PER_OUT, H))

    big = expand(theta, old_in, new_in)

    # verify EXACT preservation: the 18 appended features must be dormant, i.e.
    # any values there leave the action unchanged.
    rng = np.random.RandomState(0)
    base = rng.randn(300, old_in).astype(np.float32)
    extra = rng.randn(300, new_in - old_in).astype(np.float32)   # arbitrary
    a_old = _fwd(theta, old_in, base)
    a_new = _fwd(big, new_in, np.concatenate([base, extra], axis=1))
    err = float(np.max(np.abs(a_old - a_new)))
    print('function-preservation max action error: %.2e %s'
          % (err, 'OK' if err < 1e-6 else 'FAILED'))
    if err >= 1e-6:
        raise SystemExit('expansion did not preserve the function -- aborting.')

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    def _save(name, **arr):
        dst = os.path.join(out, name)
        for attempt in range(10):
            try:
                if os.path.exists(dst):
                    try:
                        os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)
                    except Exception:
                        pass
                    os.remove(dst)
                np.savez(dst, **arr)
                return
            except PermissionError as e:
                last = e
                time.sleep(0.5 * (attempt + 1))
        raise SystemExit('could not write %s (%s)' % (dst, last))

    _save('best.npz', theta=big, lap_time=np.array([lap]),
          generation=np.array([0]))
    _save('init_theta.npz', theta=big)
    print('\nwrote 51-dim best.npz + init_theta.npz -> %s' % out)
    print('the policy reproduces %.3fs exactly; 18 lookahead features dormant.' % lap)
    print('\nNext:  python train_es.py --fresh')


if __name__ == '__main__':
    main()
