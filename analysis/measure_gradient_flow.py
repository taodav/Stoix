"""Temporal gradient-flow diagnostic for the recurrent value head.

Question
--------
On a Markov task with a reactive policy, the true value V^pi(S) depends only on
the current state S -- yet our recurrent critic develops strong history-use (see
measure_history_ablation.py). This script examines the *learning-dynamics
mechanism*: how far back in time do gradients flow from the value at S through
the GRU hidden-state chain?

Method
------
The recurrent critic reads a sequence o_0..o_{L-1}, o_L (=S) from a zero initial
hidden state and outputs a value at each step. Take the value at the LAST step,
V(S), as a scalar function of the whole input sequence, and compute

    g_k = || d V(S) / d o_{L-k} ||_2        (lag k = 0, 1, ..., L)

This temporal input-Jacobian IS the gradient that flows backward, through the
hidden state h_L <- h_{L-1} <- ... , from the value at S to the input k steps
earlier. (In this codebase the value REGRESSION TARGET is a stored constant, so
the only cross-timestep gradient path during an update is exactly this one: the
fresh forward pass of the critic within the BPTT chunk.)

Interpretation:
  * lag 0 = instantaneous sensitivity to the current observation (Markov part).
  * lag > 0 = sensitivity routed through memory. A purely Markov value function
    would have ~all mass at lag 0.
  * We summarise each curve by (i) the fraction of gradient mass beyond lag 0,
    and (ii) a "memory horizon" = mass-weighted mean lag (centroid). Growth of
    these across checkpoints shows history-use emerging during training.

Forward+reverse autodiff only, no Monte-Carlo, no retrain -- fast. Sweeps a set
of checkpoints to show how the temporal gradient profile evolves.

Run (in tmux):

    .venv/bin/python analysis/measure_gradient_flow.py --run-uid acrobot_nonrec_actor_h64 \
        --model-name rec_ppo_nonrec_actor
    .venv/bin/python analysis/measure_gradient_flow.py --run-uid 20260708154913   # CartPole
"""

import argparse
import json
import os
import sys

import chex
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import measure_value_drift as m  # noqa: E402
from stoix.networks.base import ScannedRNN  # noqa: E402

REPO_ROOT = m.REPO_ROOT
CKPT_STRIDE = 9728

# When True, probe the critic with its hidden state zeroed every step, matching a
# Markov-value baseline trained with reset_critic_hidden_state_every_step=True.
# Auto-set from checkpoint metadata in run(); default False = recurrent critic.
_CRITIC_RESET_EVERY_STEP = False


def temporal_grad_profile(critic_apply, critic_params, obs_seq, final_obs):
    """Mean per-lag gradient-norm profile g_k = ||dV(S)/d o_{L-k}|| over states.

    obs_seq:   (N, L, obs_dim)  prefixes o_0..o_{L-1}
    final_obs: (N, obs_dim)     o_L = S
    Returns (L+1,) array: index 0 = lag 0 (current obs S), index k = lag k back.
    """
    h_dim, cell = m._critic_hidden_dim, m._critic_cell_type

    def value_at_S(full_seq):
        # full_seq: (L+1, obs_dim).
        obs_in = full_seq[:, jnp.newaxis, :]          # (L+1, 1, obs_dim)
        # Match TRAINING's critic reset pattern. Normally the recurrent critic
        # carries memory across the window (done=zeros). For a Markov-value
        # baseline (reset_critic_hidden_state_every_step=True) the critic hidden
        # state is zeroed EVERY step during training, so we must probe it the same
        # way -- otherwise the GRU's train-unused recurrent weights leak past obs
        # into V(S) and we measure latent, not deployed, history-dependence.
        if _CRITIC_RESET_EVERY_STEP:
            done_in = jnp.ones((full_seq.shape[0], 1), dtype=bool)
        else:
            done_in = jnp.zeros((full_seq.shape[0], 1), dtype=bool)
        init_h = ScannedRNN(h_dim, cell).initialize_carry(1)
        _, values = critic_apply(critic_params, init_h, (obs_in, done_in))
        return values[-1, 0]                          # scalar V(S)

    def profile_one(prefix_obs, o_final):
        full_seq = jnp.concatenate([prefix_obs, o_final[jnp.newaxis, :]], axis=0)  # (L+1,d)
        grad = jax.grad(value_at_S)(full_seq)          # (L+1, obs_dim)
        per_step = jnp.linalg.norm(grad, axis=-1)      # (L+1,)  norm over obs features
        # Reverse so index 0 = last step (lag 0 = S), index k = k steps back.
        return per_step[::-1]

    profiles = jax.vmap(profile_one)(obs_seq, final_obs)  # (N, L+1)
    return jnp.mean(profiles, axis=0)


def summarise(profile: chex.Array) -> dict:
    """Summary stats of a per-lag gradient-norm profile (index 0 = lag 0)."""
    total = float(jnp.sum(profile)) + 1e-8
    lag0_frac = float(profile[0]) / total
    beyond_lag0_frac = 1.0 - lag0_frac
    lags = jnp.arange(profile.shape[0])
    centroid = float(jnp.sum(lags * profile) / total)  # mass-weighted mean lag
    # Effective horizon: smallest lag capturing 90% of cumulative mass.
    cumure = jnp.cumsum(profile) / total
    horizon90 = int(jnp.argmax(cumure >= 0.90))
    return {
        "grad_norm_total": total,
        "lag0_fraction": lag0_frac,
        "beyond_lag0_fraction": beyond_lag0_frac,
        "memory_centroid_lag": centroid,
        "horizon_90pct_lag": horizon90,
    }


def analyse_checkpoint(args, eval_env, actor, critic, nr_critic, actor_rnn,
                       actor_apply, critic_apply, dummy_params, obs_dim, step, key):
    params = m.restore_checkpoint(args.run_uid, step, dummy_params, model_name=args.model_name)

    key, sample_key = jax.random.split(key)
    traj = jax.jit(lambda k: m.sample_prefix_trajectories(
        eval_env, actor_apply, actor_rnn, params, k,
        args.num_traj, args.prefix_len, obs_dim,
    ))(sample_key)
    jax.block_until_ready(traj)

    keep = jnp.where(traj["survived"])[0]
    obs_seq = traj["obs_seq"][keep]
    final_obs = traj["final_obs"][keep]
    n_used = int(keep.shape[0])
    if n_used < 1:
        return {"step": step, "error": "no survived trajectories"}

    profile = jax.jit(lambda: temporal_grad_profile(
        critic_apply, params.critic_params, obs_seq, final_obs
    ))()
    jax.block_until_ready(profile)

    row = {
        "step": step,
        "ckpt_index": step // CKPT_STRIDE,
        "num_traj_used": n_used,
        "profile": [float(x) for x in profile],  # full per-lag curve (index 0 = lag 0)
        **summarise(profile),
    }
    return row


def run(args):
    cfg = m.load_config(args.run_uid, overrides=tuple(args.overrides or ()),
                        model_name=args.model_name)
    cfg.num_devices = 1
    cfg.arch.num_envs = 1

    print("Building environment...")
    _, eval_env = m.environments.make(cfg)
    num_actions, is_continuous, a_min, a_max = m.describe_action_space(eval_env)
    obs_dim = int(eval_env.observation_space().generate_value().shape[-1])
    cfg.system.obs_dim = obs_dim
    cfg.system.action_dim = num_actions

    m._critic_hidden_dim = int(cfg.network.critic_network.rnn_layer.hidden_state_dim)
    m._critic_cell_type = str(cfg.network.critic_network.rnn_layer.cell_type)
    m._actor_hidden_dim = int(cfg.network.actor_network.rnn_layer.hidden_state_dim)
    m._actor_cell_type = str(cfg.network.actor_network.rnn_layer.cell_type)
    meta = m.read_checkpoint_metadata(args.run_uid, model_name=args.model_name)
    m._actor_nonrecurrent = "nonrec_actor" in meta.get("system", {}).get("system_name", "")
    # Match training's critic reset pattern: a Markov-value baseline zeroes the
    # critic hidden state every step, so probe it the same way (else the GRU's
    # unused recurrent weights leak past obs into V(S)).
    global _CRITIC_RESET_EVERY_STEP
    _CRITIC_RESET_EVERY_STEP = bool(
        meta.get("system", {}).get("reset_critic_hidden_state_every_step", False)
    )
    if _CRITIC_RESET_EVERY_STEP:
        print("Critic trained Markov (reset every step); probing with per-step reset.")
    # Appended previous-action features (ActionConcatWrapper); used to strip the
    # Markov actor during trajectory sampling. The recurrent critic (whose gradient
    # profile we measure) sees the FULL obs, so value_at_S is unchanged.
    m._action_feature_dim = int(getattr(eval_env, "action_feature_dim", 0))

    actor, critic, nr_critic, actor_rnn = m.build_networks(
        cfg, num_actions, is_continuous, a_min, a_max
    )
    actor_apply, critic_apply, _ = m.make_apply_fns(actor, critic, nr_critic)
    dummy_params = m.init_dummy_params(cfg, actor, critic, nr_critic)

    available = set(_available_steps(args.run_uid, args.model_name))
    if args.ckpt_steps:
        steps = [int(s) for s in args.ckpt_steps]
    else:
        indices = args.ckpt_indices or [1, 3, 5, 10, 20, 30, 45]
        steps = [int(i) * CKPT_STRIDE for i in indices]
    steps = [s for s in steps if s in available]
    if not steps:
        raise SystemExit(f"None of the requested checkpoints exist for {args.run_uid}. "
                         f"Available: {sorted(available)}")
    print(f"Sweeping {len(steps)} checkpoints: {steps}")

    from tqdm import tqdm
    key = jax.random.PRNGKey(args.seed)
    rows = []
    for step in tqdm(steps, desc="checkpoints"):
        key, ck = jax.random.split(key)
        rows.append(analyse_checkpoint(
            args, eval_env, actor, critic, nr_critic, actor_rnn,
            actor_apply, critic_apply, dummy_params, obs_dim, step, ck,
        ))
    return {
        "meta": {
            "run_uid": args.run_uid,
            "model_name": args.model_name,
            "env": f"{cfg.env.env_name}/{cfg.env.scenario.task_name}",
            "hidden_state_dim": m._critic_hidden_dim,
            "prefix_len": args.prefix_len,
            "num_traj": args.num_traj,
            "actor_nonrecurrent": bool(m._actor_nonrecurrent),
        },
        "checkpoints": rows,
    }


def _available_steps(run_uid, model_name):
    d = os.path.join(REPO_ROOT, "checkpoints", model_name, run_uid)
    return sorted(int(x) for x in os.listdir(d)
                  if x.isdigit() and os.path.isdir(os.path.join(d, x)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-uid", default=None)
    p.add_argument("--model-name", default="rec_ppo_dual_value")
    p.add_argument("--ckpt-indices", nargs="*", type=int, default=None,
                   help="Checkpoint indices to sweep (step = index * 9728). Default 1 3 5 10 20 30 45.")
    p.add_argument("--ckpt-steps", nargs="*", type=int, default=None)
    p.add_argument("--num-traj", type=int, default=128)
    p.add_argument("--prefix-len", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--overrides", nargs="*", default=[])
    args = p.parse_args()

    if args.run_uid is None:
        ckpt_root = os.path.join(REPO_ROOT, "checkpoints", args.model_name)
        runs = sorted(d for d in os.listdir(ckpt_root)
                      if os.path.isdir(os.path.join(ckpt_root, d)))
        if len(runs) != 1:
            raise SystemExit(f"Found {len(runs)} runs in {ckpt_root}: {runs}. Pass --run-uid.")
        args.run_uid = runs[0]
        print(f"Using run_uid={args.run_uid}")

    results = run(args)
    mt = results["meta"]
    print("\n" + "=" * 82)
    print(f"TEMPORAL GRADIENT FLOW through the recurrent value head  [{mt['env']}, "
          f"hidden={mt['hidden_state_dim']}, run={mt['run_uid']}]")
    print("=" * 82)
    print("  g_k = ||dV(S)/d o_{L-k}||: gradient from V(S) back to the obs k steps earlier.")
    print("  lag0_frac = share of gradient mass at the CURRENT obs (Markov part; ~1 => no memory)")
    print("  centroid  = mass-weighted mean lag (nats-free); horizon90 = lag capturing 90% of mass")
    print("-" * 82)
    print(f"{'ckpt':>5}{'step':>9}{'lag0_frac':>11}{'beyond_lag0':>13}"
          f"{'centroid':>10}{'horizon90':>11}")
    for r in results["checkpoints"]:
        if "error" in r:
            print(f"{r.get('ckpt_index','?'):>5}{r['step']:>9}   {r['error']}")
            continue
        print(f"{r['ckpt_index']:>5}{r['step']:>9}{r['lag0_fraction']:>11.3f}"
              f"{r['beyond_lag0_fraction']:>13.3f}{r['memory_centroid_lag']:>10.2f}"
              f"{r['horizon_90pct_lag']:>11d}")
    print("=" * 82)
    print("  lag0_frac -> 1 (all mass at current obs) = Markov / no gradient flow from the past.")
    print("  lag0_frac dropping + centroid/horizon growing = gradients flow further back over training.")

    out = args.out or f"analysis/gradient_flow_{args.run_uid}.json"
    out_path = os.path.join(REPO_ROOT, out) if not os.path.isabs(out) else out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results (incl. full per-lag profiles) to {out_path}")


if __name__ == "__main__":
    main()
