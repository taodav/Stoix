"""Value-observation temporal sensitivity: WHAT function of the past the critic computes.

The gradient-flow diagnostic collapsed dV(S)/d o_{t-w} to a NORM per lag, telling us
memory is used but not what for. This keeps the full SIGNED vector

    s_w = d V_t / d o_{t-w}   in  R^{obs_dim},   for lag w = 0, 1, ..., W

evaluated at the readout state S = final prefix step t = L. Because s_w lives in
OBSERVATION space (fixed physical meaning per component, unlike hidden units), it
can be read directly and averaged over a batch.

Two views:
  * COMPONENT view (fix w): which physical quantity the value pulls from w steps
    back. Acrobot obs = [cos th1, sin th1, cos th2, sin th2, w1, w2, (prev_action)].
  * LAG view (fix a component): the shape of s_w over w decodes the temporal
    operation the recurrence implements -- the architectural takeaway:
        flat, same sign         -> accumulation / running integral
        exp decay, same sign    -> leaky integration (EMA)
        sign alternates in w    -> finite difference (e.g. velocity/acceleration)
        peak at a specific w*    -> pure delay line of length w*

We report the signed per-lag-per-component tensor (in the JSON), plus, per obs
component, a coarse pattern classification and the dominant lag. Swept across
begin/mid/end checkpoints so the FORMATION of the operation over training is
visible (does it sharpen from noise into a clean integrator/difference?).

Reuses env/checkpoint/network machinery from measure_value_drift. Forward+autodiff
only, no training. Run (in tmux), e.g.:

    .venv/bin/python analysis/measure_value_obs_sensitivity.py \\
        --run-uid acrobot_dual_value_ac_h64 --model-name rec_ppo_dual_value \\
        --ckpt-steps 29184 194560 437760
"""

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import measure_value_drift as m  # noqa: E402
from stoix.networks.base import ScannedRNN  # noqa: E402

REPO_ROOT = m.REPO_ROOT

# Physical labels for the gymnax Acrobot observation (+ optional appended action).
ACROBOT_OBS_LABELS = ["cos_th1", "sin_th1", "cos_th2", "sin_th2", "omega1", "omega2"]


def signed_sensitivity(critic_apply, critic_params, obs_seq, final_obs, max_lag):
    """s_w = d V(S) / d o_{L-w} as a SIGNED (obs_dim,) vector, for w=0..max_lag.

    obs_seq: (N, L, obs_dim) prefixes o_0..o_{L-1}; final_obs: (N, obs_dim) = o_L = S.
    Returns (N, max_lag+1, obs_dim): axis 1 index w = lag w back from S (0 = current).
    """
    h_dim, cell = m._critic_hidden_dim, m._critic_cell_type

    def value_at_S(full_seq):
        obs_in = full_seq[:, jnp.newaxis, :]  # (L+1, 1, obs_dim)
        done_in = jnp.zeros((full_seq.shape[0], 1), dtype=bool)
        init_h = ScannedRNN(h_dim, cell).initialize_carry(1)
        _, values = critic_apply(critic_params, init_h, (obs_in, done_in))
        return values[-1, 0]  # scalar V(S)

    def per_traj(prefix_obs, o_final):
        full_seq = jnp.concatenate([prefix_obs, o_final[jnp.newaxis, :]], axis=0)  # (L+1, d)
        grad = jax.grad(value_at_S)(full_seq)  # (L+1, obs_dim), SIGNED
        # Reverse over time so index 0 = last step (lag 0 = S), index w = w back.
        grad_by_lag = grad[::-1]  # (L+1, obs_dim)
        return grad_by_lag[: max_lag + 1]  # (max_lag+1, obs_dim)

    return jax.vmap(per_traj)(obs_seq, final_obs)  # (N, max_lag+1, obs_dim)


def classify_lag_pattern(curve):
    """Coarse label for a single obs-component's signed s_w curve over lag w.

    curve: (max_lag+1,) signed values, index 0 = lag 0. Heuristics only -- a
    starting point to eyeball against the raw curve in the JSON, not ground truth.
    """
    import numpy as np

    c = np.asarray(curve)
    a = np.abs(c)
    tot = a.sum() + 1e-8
    peak_lag = int(np.argmax(a))
    lag0_frac = float(a[0] / tot)
    # Sign-alternation score over lags with non-trivial magnitude.
    sig = c[a > 0.15 * a.max()] if a.max() > 0 else c
    flips = int(np.sum(np.sign(sig[1:]) * np.sign(sig[:-1]) < 0)) if sig.size > 1 else 0
    # Decay ratio: how fast magnitude falls after the peak.
    beyond = a[1:]
    decay = float(beyond[0] / (a[0] + 1e-8)) if a[0] > 0 else 0.0

    if lag0_frac > 0.75:
        label = "current-frame (Markov)"
    elif flips >= 2:
        label = "finite-difference (derivative-like)"
    elif peak_lag >= 2 and a[peak_lag] > 1.5 * a[0]:
        label = f"delay (peak@lag{peak_lag})"
    elif decay > 0.6:
        label = "accumulation / slow integral"
    else:
        label = "leaky-integration (decaying)"
    return {"label": label, "peak_lag": peak_lag, "lag0_frac": lag0_frac,
            "sign_flips": flips, "decay_ratio": decay}


def run(args):
    cfg = m.load_config(args.run_uid, model_name=args.model_name)
    cfg.num_devices = 1
    cfg.arch.num_envs = 1

    print("Building environment + networks...")
    _, eval_env = m.environments.make(cfg)
    num_actions, is_cont, a_min, a_max = m.describe_action_space(eval_env)
    obs_dim = int(eval_env.observation_space().generate_value().shape[-1])
    cfg.system.obs_dim = obs_dim
    cfg.system.action_dim = num_actions
    m._critic_hidden_dim = int(cfg.network.critic_network.rnn_layer.hidden_state_dim)
    m._critic_cell_type = str(cfg.network.critic_network.rnn_layer.cell_type)
    m._actor_hidden_dim = int(cfg.network.actor_network.rnn_layer.hidden_state_dim)
    m._actor_cell_type = str(cfg.network.actor_network.rnn_layer.cell_type)
    m._action_feature_dim = int(getattr(eval_env, "action_feature_dim", 0))
    meta = m.read_checkpoint_metadata(args.run_uid, model_name=args.model_name)
    m._actor_nonrecurrent = "nonrec_actor" in meta.get("system", {}).get("system_name", "")

    actor, critic, nr_critic, actor_rnn = m.build_networks(
        cfg, num_actions, is_cont, a_min, a_max
    )
    actor_apply, critic_apply, _ = m.make_apply_fns(actor, critic, nr_critic)
    dummy = m.init_dummy_params(cfg, actor, critic, nr_critic)

    # Obs-component labels (append action feature labels if the wrapper is on).
    labels = list(ACROBOT_OBS_LABELS) if obs_dim >= len(ACROBOT_OBS_LABELS) else \
        [f"obs{i}" for i in range(obs_dim)]
    for i in range(len(labels), obs_dim):
        labels.append(f"prev_action{i - len(ACROBOT_OBS_LABELS)}")

    steps = [int(s) for s in args.ckpt_steps]
    key = jax.random.PRNGKey(args.seed)
    rows = []
    from tqdm import tqdm
    for step in tqdm(steps, desc="checkpoints"):
        params = m.restore_checkpoint(args.run_uid, step, dummy, model_name=args.model_name)
        key, sk = jax.random.split(key)
        traj = jax.jit(lambda k: m.sample_prefix_trajectories(
            eval_env, actor_apply, actor_rnn, params, k,
            args.num_traj, args.prefix_len, obs_dim,
        ))(sk)
        keep = jnp.where(traj["survived"])[0]
        obs_seq = traj["obs_seq"][keep]
        final_obs = traj["final_obs"][keep]
        n = int(keep.shape[0])

        s = jax.jit(lambda: signed_sensitivity(
            critic_apply, params.critic_params, obs_seq, final_obs, args.max_lag
        ))()  # (N, W+1, obs_dim)
        s = jax.block_until_ready(s)
        s_mean = jnp.mean(s, axis=0)        # (W+1, obs_dim) batch-averaged signed
        s_absmean = jnp.mean(jnp.abs(s), axis=0)  # magnitude (for classification)

        # Per-component pattern classification on the magnitude-averaged curve
        # (sign taken from the batch-mean, which is meaningful in obs space).
        per_comp = {}
        for j, lab in enumerate(labels):
            per_comp[lab] = classify_lag_pattern([float(s_mean[w, j]) for w in range(args.max_lag + 1)])

        rows.append({
            "step": step,
            "num_traj_used": n,
            "obs_labels": labels,
            "s_mean_signed": [[float(v) for v in s_mean[w]] for w in range(args.max_lag + 1)],
            "s_absmean": [[float(v) for v in s_absmean[w]] for w in range(args.max_lag + 1)],
            "per_component_pattern": per_comp,
        })

    return {
        "meta": {
            "run_uid": args.run_uid, "model_name": args.model_name,
            "env": f"{cfg.env.env_name}/{cfg.env.scenario.task_name}",
            "hidden_state_dim": m._critic_hidden_dim, "obs_dim": obs_dim,
            "max_lag": args.max_lag, "num_traj": args.num_traj,
            "prefix_len": args.prefix_len, "seed": args.seed,
        },
        "checkpoints": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-uid", required=True)
    p.add_argument("--model-name", default="rec_ppo_dual_value")
    p.add_argument("--ckpt-steps", nargs="+", type=int, required=True,
                   help="Checkpoint steps to compare, e.g. begin/mid/end: 29184 194560 437760.")
    p.add_argument("--num-traj", type=int, default=64,
                   help="Batch of states S. Use 1 to inspect a single raw trajectory.")
    p.add_argument("--prefix-len", type=int, default=32)
    p.add_argument("--max-lag", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = run(args)
    mt = results["meta"]
    print("\n" + "=" * 90)
    print(f"VALUE-OBS TEMPORAL SENSITIVITY  s_w = dV(S)/d o_(S-w)   "
          f"[{mt['env']}, run={mt['run_uid']}]")
    print("=" * 90)
    print(f"  batch={mt['num_traj']} traj, prefix_len={mt['prefix_len']}, lags 0..{mt['max_lag']}")
    for r in results["checkpoints"]:
        print("\n" + "-" * 90)
        print(f"  checkpoint step {r['step']}  (num_traj_used={r['num_traj_used']})")
        print(f"  per-component temporal pattern (signed batch-mean s_w over lag):")
        for lab, pat in r["per_component_pattern"].items():
            print(f"    {lab:14s} {pat['label']:32s} peak@lag{pat['peak_lag']:<2d}"
                  f" lag0_frac={pat['lag0_frac']:.2f} flips={pat['sign_flips']}")
        # Compact signed table for the two velocity components (Acrobot's momentum).
        labels = r["obs_labels"]
        vel_idx = [i for i, l in enumerate(labels) if l in ("omega1", "omega2")]
        if vel_idx:
            print("  signed s_w for velocity components (lag: value):")
            for j in vel_idx:
                curve = [r["s_mean_signed"][w][j] for w in range(mt["max_lag"] + 1)]
                print(f"    {labels[j]:8s} " + " ".join(f"{c:+.3f}" for c in curve))

    out = args.out or f"analysis/value_obs_sensitivity_{args.run_uid}.json"
    out_path = os.path.join(REPO_ROOT, out) if not os.path.isabs(out) else out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote full signed s_w tensors to {out_path}")


if __name__ == "__main__":
    main()
