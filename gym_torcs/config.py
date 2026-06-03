# config.py — random_test (Evolution Strategies overnight trainer)
# Single source of truth. Every tunable lives here.
import os
from dataclasses import dataclass, field, asdict
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_TORCS_DIR = os.path.join(_ROOT, "torcs")
_GYM_DIR   = os.path.join(_ROOT, "gym_torcs copy")  # the BETTER MPC (1:29 lap,
                                                    # has a learned dynamics model)


@dataclass
class Config:
    # ---- TORCS install / launch ---------------------------------------------
    torcs_dir:   str = _TORCS_DIR
    torcs_exe:   str = os.path.join(_TORCS_DIR, "wtorcs.exe")
    torcs_flags: List[str] = field(
        default_factory=lambda: ["-T", "-nofuel", "-nodamage", "-nolaptime"])
    track_name:     str = "corkscrew"
    track_category: str = "road"
    # Car every scr_server slot must use. ensure_car() (called at train/eval
    # startup) self-heals scr_server.xml back to this if anything reverts it,
    # so all workers + eval always run the SAME car. car1-ow1 = open-wheel.
    car_name:       str = "car1-ow1"
    car_setup_src:  str = os.path.join(_HERE, "car1-ow1_default.xml")
    laps_per_race:  int = 50
    raceconfig_dir: str = os.path.join(_TORCS_DIR, "config", "raceman")
    raceconfig_prefix: str = "_rt_es_"

    # ---- Networking ----------------------------------------------------------
    # TORCS scr_server binds UDP 3001 + driver_idx; the Python port MUST match.
    base_port: int = 3001
    host:      str = "127.0.0.1"
    track_sensor_angles: str = (
        "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45")

    # ---- Parallelism ---------------------------------------------------------
    n_workers:          int   = 6      # parallel TORCS / rollout workers
    launch_stagger_s:   float = 3.0
    handshake_timeout_s: float = 1.0
    handshake_attempts: int   = 8
    socket_timeout_s:   float = 1.0
    stale_relaunch_s:   float = 8.0
    max_relaunch_tries: int   = 6
    # Proactively kill+relaunch each worker's TORCS every N episodes to clear
    # any state drift / slow leak that accumulates over a long (24h) run.
    # ~10 s relaunch amortised over N laps is negligible. 0 disables.
    relaunch_every_episodes: int = 150

    # ---- Observation (33 dims) ----------------------------------------------
    # track[19] + trackPos + angle + speedX/Y/Z + rpm + gear + wheelSpin[4]
    # + MPC target-speed lookahead[3]  = 33.
    obs_dim: int = 33
    norm_track:    float = 200.0
    norm_speed:    float = 300.0       # km/h
    norm_rpm:      float = 10000.0
    norm_gear:     float = 7.0
    norm_wheelspin: float = 300.0
    norm_angle:    float = 3.14159265359
    mpc_lookahead_m: List[float] = field(default_factory=lambda: [0.0, 50.0, 150.0])
    norm_mpc_speed: float = 120.0      # m/s
    mpc_track_map:  str = os.path.join(_GYM_DIR, "track_map.json")
    mpc_gym_dir:    str = _GYM_DIR
    mpc_car_profile: str = os.path.join(_GYM_DIR, "car_profile.json")
    mpc_dynamics:   str = os.path.join(_GYM_DIR, "dynamics_model.json")  # learned

    # ---- Action / gearbox ----------------------------------------------------
    act_dim:    int = 3                 # steer, accel, brake (gear auto)
    # frame_skip 1 = the network decides EVERY tick (same control rate as the
    # MPC). Nearly free in wall-time (cost is set by TORCS ticks, not steps);
    # only the cheap numpy forward pass triples. The step-based watchdogs below
    # are scaled x3 to keep the same wall-time behaviour.
    frame_skip: int = 1
    rpm_up:     float = 9300.0          # match torcs_mpc copy's _gear_clutch
    rpm_down:   float = 4500.0          # (was 3000 -> revs dropped too low out
                                        # of corners, ~11s slower than the MPC)
    gear_max:   int   = 6
    gear_speed_cap: List[float] = field(
        default_factory=lambda: [13.0, 24.0, 37.0, 52.0, 70.0])

    # ---- Policy network ------------------------------------------------------
    # Grown 32 -> 64 via grow_model.py (function-preserving: reproduces the 98s
    # lap exactly, with 32 dormant units ES can recruit for faster lines).
    hidden_sizes: List[int] = field(default_factory=lambda: [64])
    # Output-layer bias so a fresh (near-zero-weight) policy floors the throttle
    # and drives straight -> guarantees ES gets distance signal from gen 0
    # instead of every candidate stalling. (steer 0, accel high, brake low.)
    out_bias: List[float] = field(default_factory=lambda: [0.0, 2.0, -2.0])

    # ---- Lap detection / timing ----------------------------------------------
    # TORCS's OWN reported Corkscrew length (the loader prints "Track Length
    # 3608.45 m"). Lap fires at distRaced >= this. The MPC track_map.json
    # "length" (3649.86, its driven centerline) was ~41 m too long and inflated
    # every lap time by ~1 s -- use the real value.
    track_length_m: float = 3608.45
    sim_dt:         float = 0.02        # 50 Hz tick

    # ---- Episode termination (step counts at frame_skip=1, i.e. = ticks) -----
    max_steps:        int   = 7500      # ticks (~150 s) -> timeout (lap ~100 s)
    offtrack_trackpos: float = 1.0
    stall_speed_kmh:  float = 5.0
    stall_steps:      int   = 75        # ~75 ticks ~= 1.5 s below 5 km/h
    backward_steps:   int   = 60        # ~1.2 s facing backwards
    start_grace_steps: int  = 180       # ~3.6 s launch grace

    # ---- Fitness (ES maximises this; higher = better) ------------------------
    # Incomplete lap  -> fitness = metres covered (0..track_length). Dense, so
    #                    there is always a gradient toward going further/faster.
    # Completed lap   -> fitness = lap_done_bonus + (lap_time_cap - lap_time)
    #                    * lap_time_weight. Any completed lap >> any incomplete,
    #                    and faster laps strictly dominate slower ones.
    lap_done_bonus:  float = 1.0e6
    lap_time_cap:    float = 300.0      # s
    lap_time_weight: float = 1000.0

    # ---- Evolution Strategies ------------------------------------------------
    es_popsize:     int   = 48          # candidates per generation (even; antithetic)
                                        # bumped for the bigger net + smoother grad
    es_sigma:       float = 0.04        # refine baseline perturbation std
    es_sigma_min:   float = 0.015       # floor for fine speed refinement
    es_sigma_max:   float = 0.20        # clamp ceiling (bursts use es_sigma_explore)
    es_lr:          float = 0.02        # Adam step on the mean
    es_weight_decay: float = 0.002
    es_seed:        int   = 0
    es_refine_completion: float = 0.30  # frac completing -> refine (decay sigma)
    # Slower decay so sigma stays meaningful for ~30 gens before hitting the
    # floor (was 0.88 = floored in 7 gens, killing the gradient signal).
    es_sigma_decay: float = 0.97        # per-gen sigma shrink while refining

    # ---- Exploit <-> Explore cycling (iterated local search) -----------------
    es_burst_patience: int  = 25        # stagnant gens before an exploration burst
    es_burst_gens:     int  = 6         # how many gens a burst lasts (was 12)
    # Burst sigma must scale DOWN with the 2371-dim weight space. 0.18 gave a
    # ~8.8 L2 displacement (~60% of the policy norm) = near-random networks
    # (every burst drove 150-190s). 0.05 gives ~2.4 L2 displacement -- enough
    # to probe a neighbouring basin while candidates can still actually drive.
    es_sigma_explore:  float = 0.05     # sigma during a burst (was 0.18)

    # Elite re-evaluation: a new record is confirmed with this many EXTRA laps
    # before committing. At this stage improvements are 0.1-0.5s -- smaller than
    # TORCS lap-to-lap noise (~±1s) -- so averaging 3 extra runs was averaging
    # genuine improvements away (gen 13 found 98.42s but it got rejected).
    # 1 extra lap = one confirmation run; still filters flukes but accepts real
    # small gains. Raise back to 3 once you're consistently sub-90s.
    es_elite_reeval: int = 1

    # ---- Checkpointing / logging ---------------------------------------------
    # The [64]-grown run lives in its own dir (the old [32] run's best.npz was
    # held open by a system service/indexer, blocking in-place overwrite, and a
    # fresh dir sidesteps that entirely). grow_model writes the grown checkpoint
    # here. The original [32] 98.59s run is preserved untouched in corkscrew_es.
    run_dir:        str = os.path.join(_HERE, "runs", "corkscrew_es_mpc89")
    ckpt_every_gens: int = 1            # save ES state every N generations
    # Hard wall-clock budget (hours). 0 = run forever until Ctrl-C.
    max_hours:      float = 0.0
    # Per-generation safety timeout (s): if some rollouts never return (a hung
    # worker), fill them with floor fitness and continue rather than hang.
    gen_timeout_s:  float = 420.0

    # ---- Misc ----------------------------------------------------------------
    seed: int = 0

    def port_for(self, idx: int) -> int:
        return self.base_port + idx

    def to_dict(self):
        return asdict(self)


CFG = Config()
