"""
Migrate a 24-dim TD3 checkpoint to 29-dim.

Old state: track(19) + speedX + speedY + speedZ + angle + trackPos  = 24
New state: track(19) + speedX + speedY + speedZ + angle + trackPos
           + rpm(1) + wheelSpinVel(4)                               = 29

The 5 new dimensions are appended at the END, so:
  Actor  first layer : (400, 24) -> (400, 29)  — zero-pad last 5 cols
  Critic first layer : (400, 27) -> (400, 32)  — zero-pad state cols 24-28,
                                                  shift action cols 24-26 -> 29-31

Usage (run from the project directory):
  python migrate_model.py --src PATH_TO_OLD_EP_DIR --dst PATH_TO_NEW_EP_DIR

Examples:
  # Migrate from a Downloads folder into the episode store
  python migrate_model.py \
    --src "C:/Users/user/Downloads/torcs-ibm-dev-main/torcs-ibm-dev-main/gym_torcs/new/models/episodes/ep_000000" \
    --dst "models/episodes/ep_000000"
"""

import argparse, os, json
import torch
import torch.nn as nn


OLD_STATE  = 24
NEW_STATE  = 29
ACTION_DIM = 3


def migrate_actor(old_sd):
    """(400,24) -> (400,29) with zero-padded last 5 columns."""
    w_old = old_sd['net.0.weight']          # (400, 24)
    b_old = old_sd['net.0.bias']            # (400,)
    assert w_old.shape == (400, OLD_STATE), f"Unexpected actor shape: {w_old.shape}"

    w_new = torch.zeros(400, NEW_STATE)
    w_new[:, :OLD_STATE] = w_old            # copy old 24 cols; cols 24-28 stay 0

    new_sd = dict(old_sd)
    new_sd['net.0.weight'] = w_new
    new_sd['net.0.bias']   = b_old          # bias unchanged
    return new_sd


def migrate_critic_q(old_sd, prefix):
    """
    Critic Q-network first layer: (400, state+action) layout is [state | action].
    Old: (400, 27) = [state(0-23) | action(24-26)]
    New: (400, 32) = [state(0-28) | action(29-31)]
    """
    key_w = f'{prefix}.0.weight'
    key_b = f'{prefix}.0.bias'
    w_old = old_sd[key_w]                   # (400, 27)
    assert w_old.shape == (400, OLD_STATE + ACTION_DIM), \
        f"Unexpected critic shape: {w_old.shape}"

    w_new = torch.zeros(400, NEW_STATE + ACTION_DIM)
    # state portion
    w_new[:, :OLD_STATE]                        = w_old[:, :OLD_STATE]
    # cols OLD_STATE .. NEW_STATE-1 stay zero  (new state features)
    # action portion — shifted right by (NEW_STATE - OLD_STATE) = 5
    w_new[:, NEW_STATE:NEW_STATE + ACTION_DIM]  = w_old[:, OLD_STATE:OLD_STATE + ACTION_DIM]

    new_sd = dict(old_sd)
    new_sd[key_w] = w_new
    new_sd[key_b] = old_sd[key_b]          # bias unchanged
    return new_sd


def migrate(src, dst):
    src = os.path.normpath(src)
    dst = os.path.normpath(dst)

    actor_path  = os.path.join(src, 'actor.pth')
    critic_path = os.path.join(src, 'critic.pth')

    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"actor.pth not found in {src}")
    if not os.path.exists(critic_path):
        raise FileNotFoundError(f"critic.pth not found in {src}")

    actor_sd  = torch.load(actor_path,  map_location='cpu')
    critic_sd = torch.load(critic_path, map_location='cpu')

    # Verify dimensions
    actual_dim = actor_sd['net.0.weight'].shape[1]
    if actual_dim == NEW_STATE:
        print(f"  Model is already {NEW_STATE}-dim — nothing to migrate.")
        return
    if actual_dim != OLD_STATE:
        raise ValueError(f"Expected {OLD_STATE}-dim actor, got {actual_dim}-dim")

    print(f"  Actor  : {actor_sd['net.0.weight'].shape}  ->  (400, {NEW_STATE})")
    print(f"  Critic : {critic_sd['q1.0.weight'].shape}  ->  (400, {NEW_STATE + ACTION_DIM})")

    new_actor_sd  = migrate_actor(actor_sd)
    new_critic_sd = migrate_critic_q(critic_sd, 'q1')
    new_critic_sd = migrate_critic_q(new_critic_sd, 'q2')

    os.makedirs(dst, exist_ok=True)
    torch.save(new_actor_sd,  os.path.join(dst, 'actor.pth'))
    torch.save(new_critic_sd, os.path.join(dst, 'critic.pth'))
    print(f"  Saved migrated weights -> {dst}")

    # Copy info.json if present, updating state_dim
    info_src = os.path.join(src, 'info.json')
    info_dst = os.path.join(dst, 'info.json')
    if os.path.exists(info_src):
        with open(info_src) as f:
            info = json.load(f)
        info['migrated_from_state_dim'] = OLD_STATE
        info['state_dim'] = NEW_STATE
        with open(info_dst, 'w') as f:
            json.dump(info, f, indent=2)
    else:
        with open(info_dst, 'w') as f:
            json.dump({'migrated_from_state_dim': OLD_STATE, 'state_dim': NEW_STATE,
                       'ep_num': 0, 'branch': 0, 'reward': 0, 'avg10': 0,
                       'term_reason': 'migrated', 'total_steps': 0}, f, indent=2)

    print("  Done. The first 24 input weights are preserved.")
    print("  The 5 new dims (rpm, wheelSpinVel×4) start at zero and will be")
    print("  learned from experience as training continues.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True,
                        help='Source episode directory (24-dim model)')
    parser.add_argument('--dst', required=True,
                        help='Destination episode directory (29-dim model)')
    args = parser.parse_args()
    migrate(args.src, args.dst)
