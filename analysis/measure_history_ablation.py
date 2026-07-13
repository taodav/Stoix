"""History-ablation diagnostic: does the recurrent value head actually USE history?

Motivation
----------
On fully-observable *Markov* tasks (CartPole, Acrobot) the optimal value is a
function of the current observation alone, so a recurrent critic has nothing to
gain from its memory -- and may learn to ignore it entirely, collapsing to the
same function as a non-recurrent head. This script tests that directly.

Method
------
For a checkpoint's policy pi we sample states S (final state of a length-L prefix
tau ~ pi). We then read the recurrent critic's value AT S under histories that ALL
end at the same S but differ in what precedes it:

  * real      -- the true prefix o_0..o_{L-1} the policy actually experienced,
                 then o_L = S.
  * swap      -- a DIFFERENT real trajectory's prefix (a derangement over the
                 batch), then o_L = S. This is the clean test: the GRU is warmed
                 up on a real, dynamically-coherent history (in-distribution), so
                 any change in value reflects genuine dependence on WHICH history
                 preceded S -- not an out-of-distribution artifact.
  * scrambled -- a random permutation of the SAME prefix observations, then
                 o_L = S. Destroys temporal ORDER while keeping the content;
                 secondary signal.

Why not compare against a ZEROED hidden state? Because the recurrent critic is
never trained on a zero hidden state mid-episode -- it always acts warmed-up.
Feeding it a zeroed state at S is out-of-distribution, so |V_real - V_zero|
measures that OOD artifact, not history use. The swap test avoids this by always
warming up on a real prefix.

A recurrent value that genuinely uses history changes when the history changes:
|V_real - V_swap| is large relative to how much the value varies across states.
A value that has become effectively non-recurrent (Markov task) gives ~identical
numbers regardless of the preceding history.

To compare across tasks/checkpoints we report the sensitivity normalised by the
spread (std over states) of the real value:

    hist_sensitivity = mean_S |V_real(S) - V_swap(S)|  /  std_S V_real(S)

We also report the raw value-unit differences and the spread itself, since the
normalised number is unstable when the critic is still near-constant across
states (very early training).

We run the SAME swap ablation on the POLICY (the recurrent actor), measuring the
KL divergence of its action distribution at S when the history is swapped:
KL(pi(.|S, real) || pi(.|S, swap)) in nats. This is the load-bearing check for
the "recurrent policy => history-dependent value" story: if the policy is
reactive (pi_KL ~ 0), the value has no legitimate reason to depend on history; if
the policy uses memory (pi_KL > 0), the critic SHOULD track it.

This is forward-passes only -- no Monte-Carlo rollouts -- so it is fast, and it
sweeps a set of checkpoints to show how history use evolves during training.

Run (in tmux):

    .venv/bin/python analysis/measure_history_ablation.py --run-uid acrobot_h64
    .venv/bin/python analysis/measure_history_ablation.py --run-uid <cartpole_uid>

By default it sweeps checkpoints 1,3,5,10,20,30,45 (by index; step = 9728*index),
using whichever of those exist. Override with --ckpt-indices or --ckpt-steps.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import chex
import jax
import jax.numpy as jnp

# Make sibling analysis modules importable whether run as a script
# (python analysis/measure_history_ablation.py) or as a module.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the machinery from the value-drift script (same env/network/restore path).
import measure_value_drift as m  # noqa: E402
from stoix.networks.base import ScannedRNN  # noqa: E402

REPO_ROOT = m.REPO_ROOT
CKPT_STRIDE = 9728  # steps between saved checkpoints (ckpt index k -> step k*STRIDE)


def recurrent_value_under_histories(
    critic_apply,
    critic_params,
    obs_seq: chex.Array,    # (N, L, obs_dim)  real prefixes o_0..o_{L-1}
    final_obs: chex.Array,  # (N, obs_dim)     o_L = S
    swap_idx: chex.Array,   # (N,)  which OTHER trajectory's prefix to warm up on
    perm: chex.Array,       # (N, L)  per-trajectory permutation of its own prefix
) -> Dict[str, chex.Array]:
    """Recurrent critic value at S under real / swap / scrambled history.

    All conditions feed the SAME final observation o_L = S as the last scan step;
    they differ only in the L observations that precede it. The swap condition
    warms the GRU on a DIFFERENT trajectory's (real, in-distribution) prefix.
    """
    h_dim, cell = m._critic_hidden_dim, m._critic_cell_type

    def value_for_sequence(obs_full):
        # No resets within the window: warm up over the prefix, read value at S.
        done_full = jnp.zeros((obs_full.shape[0], 1), dtype=bool)
        obs_in = obs_full[:, jnp.newaxis, :]   # (T, 1, obs_dim)
        init_h = ScannedRNN(h_dim, cell).initialize_carry(1)
        _, values = critic_apply(critic_params, init_h, (obs_in, done_in := done_full))
        return values[-1, 0]                   # value at the last step (= S)

    def value_at_S(prefix_obs, o_final):
        seq = jnp.concatenate([prefix_obs, o_final[jnp.newaxis, :]], axis=0)
        return value_for_sequence(seq)

    # real: each trajectory's own prefix.
    v_real = jax.vmap(value_at_S)(obs_seq, final_obs)
    # swap: warm up on another trajectory's prefix, but end at THIS trajectory's S.
    v_swap = jax.vmap(value_at_S)(obs_seq[swap_idx], final_obs)
    # scrambled: permute each trajectory's own prefix in time.
    scrambled_prefix = jax.vmap(lambda o, p: o[p])(obs_seq, perm)
    v_scr = jax.vmap(value_at_S)(scrambled_prefix, final_obs)

    return {"real": v_real, "swap": v_swap, "scrambled": v_scr}


def nonrecurrent_value_at_S(nr_critic_apply, nr_critic_params, final_obs):
    """Control: the non-recurrent head at S (reset every step). Reused as-is."""
    return m.compute_nonrecurrent_value_at_S(nr_critic_apply, nr_critic_params, final_obs)


def policy_kl_under_histories(
    actor_apply,
    actor_params,
    obs_seq: chex.Array,    # (N, L, obs_dim)  real prefixes o_0..o_{L-1}
    final_obs: chex.Array,  # (N, obs_dim)     o_L = S
    swap_idx: chex.Array,   # (N,)  which OTHER trajectory's prefix to warm up on
    perm: chex.Array,       # (N, L)  per-trajectory permutation of its own prefix
    key: chex.PRNGKey,
    n_kl_samples: int = 128,
) -> Dict[str, chex.Array]:
    """How much does the POLICY's action distribution at S depend on history?

    Runs the recurrent actor over a prefix and reads its action distribution at S,
    under real / swap / scrambled histories (the conditions used for the value
    ablation). Reports the KL of the action distribution away from the
    real-history one:

        KL( pi(.|S, real_history) || pi(.|S, alt_history) )

    This is the load-bearing check for the "recurrent policy => history-dependent
    value" story: if the policy is reactive (Markov), swapping the history leaves
    pi unchanged (KL ~ 0) and the value has no legitimate reason to depend on
    history. If the policy uses memory, KL > 0 and the critic SHOULD track it.

    KL is estimated by Monte-Carlo — KL(p||q) = E_{a~p}[log p(a) - log q(a)] —
    which works for BOTH discrete (CartPole/Acrobot) and continuous
    (tanh-squashed Normal, e.g. HalfCheetah) action heads. Entropy is likewise
    -E_{a~p}[log p(a)]. Everything is in nats.
    """
    h_dim, cell = m._actor_hidden_dim, m._actor_cell_type

    def dist_at_S(prefix_obs, o_final):
        seq = jnp.concatenate([prefix_obs, o_final[jnp.newaxis, :]], axis=0)
        # Match training: non-recurrent actor always resets and never sees the
        # appended previous action; recurrent actor carries memory and sees full obs.
        if m._actor_nonrecurrent:
            done_full = jnp.ones((seq.shape[0], 1), dtype=bool)
            seq = m.strip_action_features(seq, m._action_feature_dim)
        else:
            done_full = jnp.zeros((seq.shape[0], 1), dtype=bool)
        obs_in = seq[:, jnp.newaxis, :]
        init_h = ScannedRNN(h_dim, cell).initialize_carry(1)
        _, pi = actor_apply(actor_params, init_h, (obs_in, done_full))
        return pi  # batched over the (T,1) scan; we index the last step below

    def kl_and_entropy(prefix_p, prefix_q, o_final, k):
        # Distributions at S under history p and history q. We evaluate log-probs at
        # the LAST scan step (= S). Sampling from p and scoring under both gives an
        # MC estimate of KL(p||q) and of H(p).
        pi_p = dist_at_S(prefix_p, o_final)
        pi_q = dist_at_S(prefix_q, o_final)
        # Sample n actions from p at S. pi_p.sample(sample_shape) prepends a dim.
        samples = pi_p.sample(seed=k, sample_shape=(n_kl_samples,))  # (n, T, 1, ...)
        logp = pi_p.log_prob(samples)[:, -1, 0]  # (n,)
        logq = pi_q.log_prob(samples)[:, -1, 0]  # (n,)
        kl = jnp.mean(logp - logq)
        entropy = -jnp.mean(logp)
        return kl, entropy

    keys = jax.random.split(key, obs_seq.shape[0])
    scrambled_prefix = jax.vmap(lambda o, p: o[p])(obs_seq, perm)

    kl_swap, entropy = jax.vmap(kl_and_entropy)(
        obs_seq, obs_seq[swap_idx], final_obs, keys
    )
    kl_scr, _ = jax.vmap(kl_and_entropy)(obs_seq, scrambled_prefix, final_obs, keys)

    return {"kl_swap": kl_swap, "kl_scrambled": kl_scr, "entropy": entropy}


def analyse_checkpoint(
    args, cfg, eval_env, actor, critic, nr_critic, actor_rnn,
    actor_apply, critic_apply, nr_critic_apply, dummy_params,
    obs_dim: int, step: int, key: chex.PRNGKey,
) -> Dict[str, Any]:
    """Run the ablation for one checkpoint's policy."""
    params = m.restore_checkpoint(args.run_uid, step, dummy_params, model_name=args.model_name)

    # Sample states S from THIS policy (its own on-distribution states).
    key, sample_key, perm_key, kl_key = jax.random.split(key, 4)
    traj = jax.jit(lambda k: m.sample_prefix_trajectories(
        eval_env, actor_apply, actor_rnn, params, k,
        args.num_traj, args.prefix_len, obs_dim,
    ))(sample_key)
    jax.block_until_ready(traj)

    keep = jnp.where(traj["survived"])[0]
    obs_seq = traj["obs_seq"][keep]
    final_obs = traj["final_obs"][keep]
    n_used = int(keep.shape[0])
    if n_used < 2:
        return {"step": step, "error": f"only {n_used} survived trajectories (need >=2)"}

    # Swap derangement: pair each trajectory with a DIFFERENT one's prefix. A cyclic
    # shift by 1 is a guaranteed derangement (no index maps to itself).
    swap_idx = (jnp.arange(n_used) + 1) % n_used
    # Per-trajectory time-permutation for the scrambled condition.
    perm = jax.vmap(lambda k: jax.random.permutation(k, args.prefix_len))(
        jax.random.split(perm_key, n_used)
    )

    rec = jax.jit(lambda: recurrent_value_under_histories(
        critic_apply, params.critic_params, obs_seq, final_obs, swap_idx, perm
    ))()
    v_nr = jax.jit(lambda: nonrecurrent_value_at_S(
        nr_critic_apply, params.nr_critic_params, final_obs
    ))()
    pol = jax.jit(lambda: policy_kl_under_histories(
        actor_apply, params.actor_params, obs_seq, final_obs, swap_idx, perm, kl_key
    ))()
    jax.block_until_ready((rec, v_nr, pol))

    v_real, v_swap, v_scr = rec["real"], rec["swap"], rec["scrambled"]
    spread = float(jnp.std(v_real))  # value spread across states

    def norm(x):
        return float(jnp.mean(jnp.abs(x))) / (spread + 1e-8)

    return {
        "step": step,
        "ckpt_index": step // CKPT_STRIDE,
        "num_traj_used": n_used,
        "value_spread_std": spread,
        # --- Value head: raw mean absolute differences (in value units). ---
        "abs_real_minus_swap": float(jnp.mean(jnp.abs(v_real - v_swap))),
        "abs_real_minus_scrambled": float(jnp.mean(jnp.abs(v_real - v_scr))),
        # Normalised by the across-state spread (unitless; comparable across tasks).
        # Unstable when spread ~ 0 (near-constant critic); read alongside the raw
        # differences and the spread itself.
        "hist_sensitivity_vs_swap": norm(v_real - v_swap),
        "hist_sensitivity_vs_scrambled": norm(v_real - v_scr),
        # --- Policy: KL divergence of pi(.|S) when history is swapped (nats). ---
        # This is the load-bearing check: if the policy is history-dependent, the
        # value legitimately should be too.
        "policy_kl_swap": float(jnp.mean(pol["kl_swap"])),
        "policy_kl_scrambled": float(jnp.mean(pol["kl_scrambled"])),
        "policy_entropy": float(jnp.mean(pol["entropy"])),
        # Diagnostics.
        "mean_v_real": float(jnp.mean(v_real)),
        "mean_v_swap": float(jnp.mean(v_swap)),
        "mean_v_scrambled": float(jnp.mean(v_scr)),
        "mean_v_nonrecurrent": float(jnp.mean(v_nr)),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = m.load_config(args.run_uid, overrides=tuple(args.overrides or ()), model_name=args.model_name)
    cfg.num_devices = 1
    cfg.arch.num_envs = 1

    print("Building environment...")
    _, eval_env = m.environments.make(cfg)
    num_actions, is_continuous, a_min, a_max = m.describe_action_space(eval_env)
    obs_dim = int(eval_env.observation_space().generate_value().shape[-1])
    cfg.system.obs_dim = obs_dim
    cfg.system.action_dim = num_actions

    # Populate the module globals the reused value fns read.
    m._critic_hidden_dim = int(cfg.network.critic_network.rnn_layer.hidden_state_dim)
    m._critic_cell_type = str(cfg.network.critic_network.rnn_layer.cell_type)
    m._actor_hidden_dim = int(cfg.network.actor_network.rnn_layer.hidden_state_dim)
    m._actor_cell_type = str(cfg.network.actor_network.rnn_layer.cell_type)
    # Detect non-recurrent actor from the system_name in the checkpoint metadata.
    meta = m.read_checkpoint_metadata(args.run_uid, model_name=args.model_name)
    system_name = meta.get("system", {}).get("system_name", "")
    m._actor_nonrecurrent = "nonrec_actor" in system_name
    if m._actor_nonrecurrent:
        print("Detected non-recurrent actor (policy is reactive/Markov).")
    # Appended previous-action features (ActionConcatWrapper). The recurrent critic
    # sees the full obs; the Markov actor / memory-free critic strip these.
    m._action_feature_dim = int(getattr(eval_env, "action_feature_dim", 0))
    if m._action_feature_dim:
        print(f"Detected ActionConcatWrapper: {m._action_feature_dim} appended action features.")

    actor, critic, nr_critic, actor_rnn = m.build_networks(
        cfg, num_actions, is_continuous, a_min, a_max
    )
    actor_apply, critic_apply, nr_critic_apply = m.make_apply_fns(actor, critic, nr_critic)
    dummy_params = m.init_dummy_params(cfg, actor, critic, nr_critic)

    # Which checkpoints to sweep.
    available = set(_available_steps(args.run_uid, args.model_name))
    if args.ckpt_steps:
        steps = [int(s) for s in args.ckpt_steps]
    else:
        indices = args.ckpt_indices or [1, 3, 5, 10, 20, 30, 45]
        steps = [int(i) * CKPT_STRIDE for i in indices]
    steps = [s for s in steps if s in available]
    if not steps:
        raise SystemExit(
            f"None of the requested checkpoints exist for run {args.run_uid}. "
            f"Available steps: {sorted(available)}"
        )
    print(f"Sweeping {len(steps)} checkpoints: {steps}")

    key = jax.random.PRNGKey(args.seed)
    rows: List[Dict[str, Any]] = []
    from tqdm import tqdm
    for step in tqdm(steps, desc="checkpoints"):
        key, ck = jax.random.split(key)
        row = analyse_checkpoint(
            args, cfg, eval_env, actor, critic, nr_critic, actor_rnn,
            actor_apply, critic_apply, nr_critic_apply, dummy_params,
            obs_dim, step, ck,
        )
        rows.append(row)

    return {
        "meta": {
            "run_uid": args.run_uid,
            "num_traj": args.num_traj,
            "prefix_len": args.prefix_len,
            "seed": args.seed,
            "env": f"{cfg.env.env_name}/{cfg.env.scenario.task_name}",
            "hidden_state_dim": m._critic_hidden_dim,
        },
        "checkpoints": rows,
    }


def _available_steps(run_uid: str, model_name: str = "rec_ppo_dual_value") -> List[int]:
    d = os.path.join(REPO_ROOT, "checkpoints", model_name, run_uid)
    return sorted(
        int(x) for x in os.listdir(d)
        if x.isdigit() and os.path.isdir(os.path.join(d, x))
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-uid", default=None,
                   help="Checkpoint run folder under checkpoints/<model-name>/.")
    p.add_argument("--model-name", default="rec_ppo_dual_value",
                   help="Model/system name (checkpoint subdirectory). "
                        "Use 'rec_ppo_nonrec_actor' for the non-recurrent-actor variant.")
    p.add_argument("--ckpt-indices", nargs="*", type=int, default=None,
                   help="Checkpoint indices to sweep (step = index * 9728). "
                        "Default: 1 3 5 10 20 30 45.")
    p.add_argument("--ckpt-steps", nargs="*", type=int, default=None,
                   help="Explicit checkpoint steps to sweep (overrides --ckpt-indices).")
    p.add_argument("--num-traj", type=int, default=256)
    p.add_argument("--prefix-len", type=int, default=64,
                   help="Prefix length L; S is the state after L steps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="Output JSON. Default: analysis/history_ablation_<run_uid>.json")
    p.add_argument("--overrides", nargs="*", default=[],
                   help="Extra Hydra overrides (rarely needed; env+arch auto-matched).")
    args = p.parse_args()

    if args.run_uid is None:
        ckpt_root = os.path.join(REPO_ROOT, "checkpoints", args.model_name)
        runs = sorted(d for d in os.listdir(ckpt_root)
                      if os.path.isdir(os.path.join(ckpt_root, d)))
        if len(runs) != 1:
            raise SystemExit(
                f"Found {len(runs)} runs in {ckpt_root}: {runs}. Pass --run-uid."
            )
        args.run_uid = runs[0]
        print(f"Using run_uid={args.run_uid}")

    results = run(args)

    m_ = results["meta"]
    print("\n" + "=" * 84)
    print(f"HISTORY-ABLATION  [{m_['env']}, hidden={m_['hidden_state_dim']}, "
          f"run={m_['run_uid']}]")
    print("=" * 84)
    print("  Does V_rec(S) and pi(.|S) change when a DIFFERENT (real) history precedes S?")
    print("  VALUE:  sens/swap = mean_S |V_real - V_swap| / std_S V_real  (~0 => ignores hist)")
    print("  POLICY: pi_KL = mean_S KL(pi_real || pi_swap) in nats        (~0 => reactive)")
    print("-" * 84)
    print(f"{'ckpt':>5}{'step':>9}{'|real-swap|':>13}{'sens/swap':>11}{'spread':>9}"
          f"{'pi_KL_swap':>12}{'pi_entropy':>12}{'V_rec':>9}")
    for r in results["checkpoints"]:
        if "error" in r:
            print(f"{r.get('ckpt_index','?'):>5}{r['step']:>9}   {r['error']}")
            continue
        print(f"{r['ckpt_index']:>5}{r['step']:>9}"
              f"{r['abs_real_minus_swap']:>13.4f}{r['hist_sensitivity_vs_swap']:>11.4f}"
              f"{r['value_spread_std']:>9.3f}{r['policy_kl_swap']:>12.4f}"
              f"{r['policy_entropy']:>12.4f}{r['mean_v_real']:>9.2f}")
    print("=" * 84)
    print("  Interpretation: |real-swap| tiny AND << spread across checkpoints =>")
    print("  the recurrent critic ignores history (effectively non-recurrent, as")
    print("  expected on a Markov task). |real-swap| comparable to spread => history")
    print("  genuinely informs the value.")

    out = args.out or f"analysis/history_ablation_{args.run_uid}.json"
    out_path = os.path.join(REPO_ROOT, out) if not os.path.isabs(out) else out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
