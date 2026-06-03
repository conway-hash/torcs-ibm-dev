# train_es.py — random_test
# ---------------------------------------------------------------------------
# Overnight, unattended Evolution-Strategies trainer for the fastest Corkscrew
# lap. Designed to run for hours with ZERO input and to MONOTONICALLY improve
# the best lap without falling into traps.
#
#   python train_es.py                 # start / auto-resume (recommended)
#   python train_es.py --fresh         # ignore any saved state, start over
#   python train_es.py --hours 8       # stop after 8 wall-clock hours
#   python train_es.py --workers 4     # fewer parallel TORCS
#
# ROBUSTNESS MECHANISMS (why it won't fall into a trap):
#   * Elitism      — best-ever policy saved to best.npz; only ever overwritten
#                    by a genuinely faster, RE-VALIDATED lap. Cannot regress.
#   * Re-eval      — a new record is confirmed over K extra laps (averaged)
#                    before being accepted, so TORCS noise can't crown a fluke.
#   * True objective — fitness IS lap time / distance; nothing to reward-hack.
#   * Anti-stall   — fresh policies floor the throttle (drive forward) so ES
#                    always has a gradient; bad candidates die fast on
#                    off-track / stall / backward / timeout watchdogs.
#   * Anti-stagnation — sigma ramps up after N stagnant generations to escape
#                    local optima, then relaxes when progress resumes.
#   * Crash recovery — each worker relaunches a dead/hung TORCS and returns a
#                    floor fitness; a per-generation timeout fills any missing
#                    rollouts so one bad worker never stalls the whole run.
#   * Checkpoint/resume — full ES state saved every generation (atomic write);
#                    restarting continues exactly where it left off.
# ---------------------------------------------------------------------------
import argparse
import csv
import os
import sys
import time
import numpy as np
import multiprocessing as mp

from config import CFG
from policy import Policy
from es import ES


# ---------------------------------------------------------------------------
# Fitness from a rollout outcome
# ---------------------------------------------------------------------------

def compute_fitness(lap_time, dist, cfg):
    if lap_time is not None:
        return (cfg.lap_done_bonus
                + (cfg.lap_time_cap - lap_time) * cfg.lap_time_weight)
    return float(dist)        # incomplete: metres covered (dense gradient)


# ---------------------------------------------------------------------------
# Worker: owns one persistent TORCS, evaluates candidate params on demand
# ---------------------------------------------------------------------------

def worker_loop(idx, task_q, result_q, stop_event):
    # Imports inside the process (spawn start method).
    import numpy as _np
    from torcs_env import TorcsEnv
    from policy import Policy as _Policy

    policy = _Policy(CFG)
    env = None

    def ensure_env():
        nonlocal env
        if env is None:
            env = TorcsEnv(idx, CFG, gui=False)
        return env

    def rollout(theta):
        e = ensure_env()
        policy.set_flat(theta)
        obs = e.reset()
        lap_time, dist = None, 0.0
        steps = 0
        while True:
            a = policy.act(obs)
            obs, _, done, info = e.step(a)
            steps += 1
            dist = info.get('dist_raced', dist)
            if 'lap_time' in info:
                lap_time = info['lap_time']
            if done:
                break
        return lap_time, dist, info

    while not stop_event.is_set():
        try:
            task = task_q.get(timeout=1.0)
        except Exception:
            continue
        if task is None:
            break
        eval_id, theta = task
        try:
            lap_time, dist, info = rollout(theta)
            fitness = compute_fitness(lap_time, dist, CFG)
            result_q.put((eval_id, fitness, lap_time, dist))
        except Exception as e:
            # Hard failure: relaunch TORCS and return a floor result so the
            # generation still completes.
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            env = None
            result_q.put((eval_id, 0.0, None, 0.0))
            time.sleep(1.0)

    try:
        if env is not None:
            env.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, args):
        self.cfg = CFG
        if args.workers:
            self.cfg.n_workers = args.workers
        self.run_dir = self.cfg.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.state_path = os.path.join(self.run_dir, 'es_state.npz')
        self.best_path  = os.path.join(self.run_dir, 'best.npz')
        self.log_path   = os.path.join(self.run_dir, 'log.csv')

        self.policy = Policy(self.cfg)
        self.es = self._init_es(args)

        # elite tracking
        self.best_lap = float('inf')
        self.gens_since_best = 0
        self.burst_remaining = 0          # exploration-burst countdown
        self.best_theta = self.es.theta.copy()
        if os.path.exists(self.best_path):
            try:
                with np.load(self.best_path) as b:   # close handle before any
                    self.best_lap = float(b['lap_time'][0])   # later overwrite
                    self.best_theta = b['theta'].astype(np.float32).copy()
            except Exception:
                pass

        # Optional: re-center the ES mean on the elite policy so refinement
        # happens AROUND the best lap found, not the diffuse population mean.
        if getattr(args, 'recenter', False) and self.best_lap < 1e8:
            self.es.theta = self.best_theta.copy()
            self.es.m[:] = 0.0
            self.es.v[:] = 0.0
            self.es.adam_t = 0
            self.es.sigma = min(self.es.sigma, self.cfg.es_sigma)
            print('[es] re-centered on elite (best_lap=%.3fs); sigma=%.3f'
                  % (self.best_lap, self.es.sigma), flush=True)

        self.max_seconds = (args.hours or self.cfg.max_hours) * 3600.0
        self.t_start = time.time()

    def _init_es(self, args):
        es = ES(self.policy.init_theta(np.random.RandomState(self.cfg.es_seed)),
                self.cfg)
        if not args.fresh and os.path.exists(self.state_path):
            try:
                s = np.load(self.state_path, allow_pickle=True)
                es.load_state(s)
                print('[es] resumed at generation %d (sigma=%.3f, best_fit=%.0f)'
                      % (es.generation, es.sigma, es.best_fitness), flush=True)
            except Exception as ex:
                print('[es] could not resume (%s) — starting fresh' % ex, flush=True)
        else:
            # Fresh start (either --fresh, or no saved state): begin from the
            # MPC warm-start if present, else the random floored init. NOTE:
            # this runs for --fresh too -- --fresh discards in-progress training
            # STATE, not the warm-start starting policy.
            init_path = os.path.join(self.run_dir, 'init_theta.npz')
            if os.path.exists(init_path):
                try:
                    w = np.load(init_path)['theta'].astype(np.float32)
                    if w.size == es.theta.size:
                        es.theta = w.copy()
                        print('[es] warm-started from MPC init_theta.npz '
                              '(gen 0)', flush=True)
                    else:
                        print('[es] init_theta.npz size %d != policy %d — '
                              'random init' % (w.size, es.theta.size), flush=True)
                except Exception as e:
                    print('[es] could not load init_theta.npz (%s) — random '
                          'init' % e, flush=True)
            else:
                print('[es] no warm-start found — random init (gen 0)', flush=True)
        return es

    # ---- parallel evaluation -------------------------------------------------
    def _dispatch(self, thetas):
        """Evaluate a list of parameter vectors across the worker pool.
        Returns parallel lists (fitness, lap_time, dist) indexed like thetas.
        Missing results (hung worker) get floor fitness after gen_timeout_s."""
        n = len(thetas)
        # Drain any late results from a previous (timed-out) dispatch so they
        # can't be mis-attributed to this batch's eval_ids.
        while True:
            try:
                self.result_q.get_nowait()
            except Exception:
                break
        for i, th in enumerate(thetas):
            self.task_q.put((i, th))
        fits  = [0.0] * n
        laps  = [None] * n
        dists = [0.0] * n
        got = 0
        deadline = time.time() + self.cfg.gen_timeout_s
        while got < n and time.time() < deadline:
            try:
                eid, fit, lap, dist = self.result_q.get(timeout=2.0)
            except Exception:
                continue
            fits[eid], laps[eid], dists[eid] = fit, lap, dist
            got += 1
        if got < n:
            print('[warn] generation timeout: %d/%d rollouts returned' % (got, n),
                  flush=True)
        return fits, laps, dists

    # ---- main loop -----------------------------------------------------------
    def run(self):
        ctx = mp.get_context('spawn')
        self.stop_event = ctx.Event()
        self.task_q   = ctx.Queue()
        self.result_q = ctx.Queue()

        from launcher import kill_all_torcs, ensure_car
        ensure_car(self.cfg)          # all workers run the SAME car (self-heal)
        kill_all_torcs()
        time.sleep(1.0)

        # spawn workers (staggered so simultaneous TORCS launches don't thrash)
        self.workers = []
        for i in range(self.cfg.n_workers):
            p = ctx.Process(target=worker_loop,
                            args=(i, self.task_q, self.result_q, self.stop_event),
                            daemon=False)
            p.start()
            self.workers.append(p)
            time.sleep(self.cfg.launch_stagger_s)

        self._log_header()
        print('[train] %d workers up. Optimising...  (Ctrl-C to stop & save)'
              % self.cfg.n_workers, flush=True)

        try:
            while True:
                if self.max_seconds and (time.time() - self.t_start) > self.max_seconds:
                    print('[train] wall-clock budget reached.', flush=True)
                    break
                self._generation()
        except KeyboardInterrupt:
            print('\n[train] Ctrl-C — saving and shutting down...', flush=True)
        finally:
            self._save_state()
            self.stop_event.set()
            for _ in self.workers:
                try:
                    self.task_q.put(None)
                except Exception:
                    pass
            for p in self.workers:
                p.join(timeout=10)
            kill_all_torcs()
            bl = '%.3fs' % self.best_lap if self.best_lap < 1e8 else 'none'
            print('[train] done. best_lap=%s  (resume: python train_es.py)' % bl,
                  flush=True)

    # ---- one generation ------------------------------------------------------
    def _generation(self):
        gen = self.es.generation
        t0 = time.time()

        cand = self.es.ask()                       # (pop, dim)
        fits, laps, dists = self._dispatch([c for c in cand])
        gen_best_fit, gen_mean_fit = self.es.tell(fits)

        # progress stats
        completed = [lt for lt in laps if lt is not None]
        gen_best_lap = min(completed) if completed else None
        max_dist = max(dists) if dists else 0.0

        # evaluate the new mean (the "current policy") once for tracking and as
        # the elite candidate
        mean_fit, mean_lap, mean_dist = self._eval_theta(self.es.theta)

        # candidate elite: the better of (best generation member) and (mean)
        # — re-validate before committing to best.npz
        elite_lap, elite_theta = self._pick_elite(cand, laps, mean_lap)
        improved = False
        if elite_lap is not None and elite_lap < self.best_lap - 1e-6:
            confirmed = self._revalidate(elite_theta, elite_lap)
            if confirmed is not None and confirmed < self.best_lap - 1e-6:
                self.best_lap = confirmed
                self.best_theta = elite_theta.copy()
                self._save_best()
                improved = True
                self.gens_since_best = 0
                print('[train] *** NEW BEST LAP %.3fs *** (gen %d)'
                      % (confirmed, gen), flush=True)
        if not improved:
            self.gens_since_best += 1

        # Exploit<->explore cycling (iterated local search). Refine the elite
        # most of the time; periodically fire a bold, elite-anchored burst to
        # probe for a faster basin. Orchestrated here (we hold best_theta).
        completion_rate = len(completed) / max(1, len(cand))
        c = self.cfg
        if self.burst_remaining > 0:
            self.burst_remaining -= 1
            self.es.sigma = c.es_sigma_explore       # hold wide during burst
            if self.burst_remaining == 0:
                self.gens_since_best = 0             # cooldown: refine next
                print('[burst] ended -> refining', flush=True)
        elif self.gens_since_best >= c.es_burst_patience:
            # start a burst: anchor on the best lap, reset momentum, go wide
            self.es.theta = self.best_theta.copy()
            self.es.m[:] = 0.0
            self.es.v[:] = 0.0
            self.es.adam_t = 0
            self.es.sigma = c.es_sigma_explore
            self.burst_remaining = c.es_burst_gens - 1
            print('[burst] exploring around elite %.2fs (sigma=%.2f, %d gens)'
                  % (self.best_lap, c.es_sigma_explore, c.es_burst_gens),
                  flush=True)
        elif completion_rate >= c.es_refine_completion:
            self.es.sigma = max(c.es_sigma_min, self.es.sigma * c.es_sigma_decay)

        if gen % self.cfg.ckpt_every_gens == 0:
            self._save_state()

        dt = time.time() - t0
        evals = len(cand) + 1
        bl = '%.2f' % self.best_lap if self.best_lap < 1e8 else '-'
        gl = '%.2f' % gen_best_lap if gen_best_lap else '-'
        ml = '%.2f' % mean_lap if mean_lap else '-'
        nlap = len(completed)
        print('[gen %4d] best=%ss  gen_best=%ss  mean=%ss  maxdist=%4.0fm  '
              'laps=%d/%d  sigma=%.3f  stale=%d  %.0fs (%.1f ev/s)%s'
              % (gen, bl, gl, ml, max_dist, nlap, len(cand),
                 self.es.sigma, self.gens_since_best, dt,
                 evals / max(dt, 1e-6),
                 '  <improved' if improved else ''), flush=True)
        self._log_row(gen, gen_best_lap, mean_lap, max_dist, nlap, len(cand), dt)

    def _eval_theta(self, theta):
        f, l, d = self._dispatch([theta])
        return f[0], l[0], d[0]

    def _pick_elite(self, cand, laps, mean_lap):
        """Choose the most promising completed-lap policy this generation."""
        best_lap, best_theta = None, None
        for i, lt in enumerate(laps):
            if lt is not None and (best_lap is None or lt < best_lap):
                best_lap, best_theta = lt, cand[i]
        # the mean policy competes too
        if mean_lap is not None and (best_lap is None or mean_lap < best_lap):
            best_lap, best_theta = mean_lap, self.es.theta
        return best_lap, (best_theta.copy() if best_theta is not None else None)

    def _revalidate(self, theta, first_lap):
        """Re-run the candidate K times; return the MEAN lap over all completed
        runs (incl. the first), or None if it fails to complete on re-eval."""
        laps = [first_lap]
        for _ in range(max(0, self.cfg.es_elite_reeval)):
            _, l, _ = self._eval_theta(theta)
            if l is not None:
                laps.append(l)
        if len(laps) < 2:           # never reproduced -> reject as a fluke
            return None
        return float(np.mean(laps))

    # ---- persistence ---------------------------------------------------------
    def _save_state(self):
        s = self.es.state()
        s['best_lap'] = np.array([self.best_lap])
        s['best_theta'] = self.best_theta
        # tmp must end in .npz (np.savez appends .npz otherwise) for the atomic
        # os.replace to find the file it just wrote.
        tmp = self.state_path + '.tmp.npz'
        np.savez(tmp, **s)
        os.replace(tmp, self.state_path)

    def _save_best(self):
        tmp = self.best_path + '.tmp.npz'
        np.savez(tmp, theta=self.best_theta,
                 lap_time=np.array([self.best_lap]),
                 generation=np.array([self.es.generation]))
        os.replace(tmp, self.best_path)

    def _log_header(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                csv.writer(f).writerow(
                    ['time', 'generation', 'best_lap', 'gen_best_lap',
                     'mean_lap', 'max_dist', 'laps_completed', 'popsize',
                     'gen_seconds', 'sigma'])

    def _log_row(self, gen, gbl, ml, md, nlap, pop, dt):
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow(
                ['%.0f' % time.time(), gen,
                 '%.3f' % self.best_lap if self.best_lap < 1e8 else '',
                 '%.3f' % gbl if gbl else '',
                 '%.3f' % ml if ml else '',
                 '%.1f' % md, nlap, pop, '%.1f' % dt, '%.4f' % self.es.sigma])


def main():
    ap = argparse.ArgumentParser(description='Overnight ES fastest-lap trainer')
    ap.add_argument('--fresh',   action='store_true', help='ignore saved state')
    ap.add_argument('--hours',   type=float, default=0.0, help='wall-clock budget (0=forever)')
    ap.add_argument('--workers', type=int, default=0, help='parallel TORCS (default cfg)')
    ap.add_argument('--recenter', action='store_true',
                    help='on resume, re-center the search on the elite policy '
                         '(refine around the best lap found)')
    args = ap.parse_args()
    Trainer(args).run()


if __name__ == '__main__':
    main()
