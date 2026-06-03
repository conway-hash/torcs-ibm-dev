# policy.py — random_test
# A tiny tanh MLP evaluated in pure numpy (no torch needed in the rollout
# workers). Maps the 33-dim observation to a 3-dim action in [-1, 1]
# (steer, accel_n, brake_n). Parameters are stored/loaded as one flat vector
# so Evolution Strategies can perturb them directly.
import numpy as np

from config import CFG


class Policy:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        sizes = [cfg.obs_dim] + list(cfg.hidden_sizes) + [cfg.act_dim]
        self.layers = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.out_bias = np.asarray(cfg.out_bias, dtype=np.float32)
        self.W = None
        self.b = None

    # ---- parameter bookkeeping ----------------------------------------------
    def n_params(self):
        return int(sum(i * o + o for i, o in self.layers))

    def init_theta(self, rng):
        """Sensible init: small weights, last-layer bias set to out_bias so a
        fresh policy floors the throttle and drives straight."""
        parts = []
        n = len(self.layers)
        for li, (i, o) in enumerate(self.layers):
            last = (li == n - 1)
            scale = (1.0 / np.sqrt(i)) * (0.1 if last else 1.0)
            W = (rng.randn(i, o) * scale).astype(np.float32)
            b = np.zeros(o, dtype=np.float32)
            if last:
                b[:] = self.out_bias
            parts.append(W.ravel())
            parts.append(b)
        return np.concatenate(parts).astype(np.float32)

    def set_flat(self, theta):
        theta = np.asarray(theta, dtype=np.float32)
        self.W, self.b, k = [], [], 0
        for (i, o) in self.layers:
            self.W.append(theta[k:k + i * o].reshape(i, o)); k += i * o
            self.b.append(theta[k:k + o]);                   k += o
        return self

    # ---- forward -------------------------------------------------------------
    def act(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        for li in range(len(self.W)):
            x = x @ self.W[li] + self.b[li]
            if li < len(self.W) - 1:
                x = np.tanh(x)
        return np.tanh(x)   # output in [-1, 1]^act_dim
