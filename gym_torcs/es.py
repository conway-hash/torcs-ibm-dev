# es.py — random_test
# OpenAI-style Evolution Strategies with antithetic sampling, centered-rank
# fitness shaping, an Adam update on the mean, weight decay, and adaptive
# sigma for escaping local optima. Pure numpy; fully checkpointable.
#
# Maximises fitness. One generation:
#   cand = es.ask()                  # (popsize, dim) candidate parameter vecs
#   fit  = [evaluate(c) for c in cand]
#   es.tell(fit)                     # natural-gradient ascent on the mean
import numpy as np


def centered_ranks(x):
    """Map raw fitnesses to centered ranks in [-0.5, 0.5] (robust to scale)."""
    x = np.asarray(x, dtype=np.float64)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[np.argsort(x)] = np.arange(len(x))
    if len(x) > 1:
        ranks /= (len(x) - 1)
    ranks -= 0.5
    return ranks


class ES:
    def __init__(self, theta0, cfg):
        self.cfg = cfg
        self.theta = np.asarray(theta0, dtype=np.float32).copy()
        self.dim = self.theta.size
        self.popsize = int(cfg.es_popsize)
        if self.popsize % 2 != 0:
            self.popsize += 1                 # antithetic needs even count
        self.sigma = float(cfg.es_sigma)
        self.lr = float(cfg.es_lr)
        self.weight_decay = float(cfg.es_weight_decay)
        self.rng = np.random.RandomState(cfg.es_seed)

        # Adam state for the mean
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.adam_t = 0
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8

        self.generation = 0
        self._eps = None                       # last-drawn perturbations

        # anti-stagnation bookkeeping
        self.best_fitness = -np.inf
        self.stagnation = 0

    # ---- sampling ------------------------------------------------------------
    def ask(self):
        half = self.popsize // 2
        eps = self.rng.randn(half, self.dim).astype(np.float32)
        self._eps = eps
        full = np.concatenate([eps, -eps], axis=0)        # antithetic
        return self.theta[None, :] + self.sigma * full    # (popsize, dim)

    # ---- update --------------------------------------------------------------
    def tell(self, fitnesses):
        fitnesses = np.asarray(fitnesses, dtype=np.float64)
        half = self.popsize // 2
        ranks = centered_ranks(fitnesses)
        # antithetic gradient estimate: sum (r+ - r-) * eps
        diff = ranks[:half] - ranks[half:]
        grad = (self._eps.T @ diff) / half
        grad = grad / self.sigma
        # we MAXIMISE -> ascend. Add weight decay (pulls theta toward 0).
        g = grad - self.weight_decay * self.theta

        # Adam ascent
        self.adam_t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g * g)
        mhat = self.m / (1 - self.b1 ** self.adam_t)
        vhat = self.v / (1 - self.b2 ** self.adam_t)
        self.theta = (self.theta + self.lr * mhat /
                      (np.sqrt(vhat) + self.eps)).astype(np.float32)

        self.generation += 1
        return float(np.max(fitnesses)), float(np.mean(fitnesses))

    # ---- regime-aware adaptive sigma -----------------------------------------
    def adapt(self, elite_fitness, completion_rate):
        """Adapt exploration to the current regime, called once per generation.

        * Laps reliably completing (completion_rate high)  -> we are REFINING
          speed; DECAY sigma toward sigma_min for finer search. (Big noise here
          just knocks fast candidates off the track.)
        * Stuck *before* reliably completing                -> EXPLORE harder;
          ramp sigma up to find policies that finish the lap.
        * Deep stall in either regime                        -> a one-off sigma
          bump to escape a local optimum.
        """
        c = self.cfg
        if elite_fitness > self.best_fitness + 1e-9:
            self.best_fitness = elite_fitness
            self.stagnation = 0
        else:
            self.stagnation += 1

        if self.stagnation >= 2 * c.es_stagnation_patience:
            self.sigma = min(c.es_sigma_max, self.sigma * c.es_sigma_ramp)
            self.stagnation = 0
        elif completion_rate >= c.es_refine_completion:
            self.sigma = max(c.es_sigma_min, self.sigma * c.es_sigma_decay)
        elif self.stagnation >= c.es_stagnation_patience:
            self.sigma = min(c.es_sigma_max, self.sigma * c.es_sigma_ramp)
            self.stagnation = 0

    # ---- checkpoint I/O ------------------------------------------------------
    def state(self):
        return dict(
            theta=self.theta, m=self.m, v=self.v,
            adam_t=np.array([self.adam_t]),
            sigma=np.array([self.sigma]),
            generation=np.array([self.generation]),
            best_fitness=np.array([self.best_fitness]),
            stagnation=np.array([self.stagnation]),
            rng_state=np.array(self.rng.get_state()[1]),
            rng_pos=np.array([self.rng.get_state()[2]]),
        )

    def load_state(self, s):
        self.theta = s['theta'].astype(np.float32)
        self.m = s['m'].astype(np.float32)
        self.v = s['v'].astype(np.float32)
        self.adam_t = int(s['adam_t'][0])
        # clamp to current config bounds so a tuned-down sigma_max takes effect
        # on resume (otherwise a runaway sigma would reload unchanged)
        self.sigma = float(np.clip(s['sigma'][0],
                                   self.cfg.es_sigma_min, self.cfg.es_sigma_max))
        self.generation = int(s['generation'][0])
        self.best_fitness = float(s['best_fitness'][0])
        self.stagnation = int(s['stagnation'][0])
        try:
            st = self.rng.get_state()
            self.rng.set_state((st[0], s['rng_state'].astype(np.uint32),
                                int(s['rng_pos'][0]), st[3], st[4]))
        except Exception:
            pass
