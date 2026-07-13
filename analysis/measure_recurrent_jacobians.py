"""Recurrent-Jacobian analysis: WHY does the GRU use its memory?

The gradient-flow diagnostic (measure_gradient_flow.py) shows the value at S
depends on far-back observations. This script opens up the mechanism by studying
the per-step hidden-to-hidden Jacobian of the GRU along real trajectories:

    J_t = d h_t / d h_{t-1}    (holding the step-t input embedding fixed)

and the k-step product that BPTT actually pushes gradients through:

    d h_t / d h_{t-k} = J_t J_{t-1} ... J_{t-k+1}

We parse these Jacobians several ways (all beyond just the norm):

  1. NORMS vs lag k: spectral (top singular value) and Frobenius norm of the
     k-step product, averaged over t and trajectories. Decay rate => effective
     memory timescale.
  2. PER-STEP EIGENVALUES of J_t: spectral radius, and the count of modes with
     |lambda| near 1 (long-lived); fraction complex (oscillatory memory).
  3. SINGULAR-VALUE SPECTRUM of the k-step product: effective rank of gradient
     flow (how many hidden directions carry information back k steps).
  4. TOP LYAPUNOV EXPONENT via QR iteration over the product: the rigorous
     vanishing(<0)/exploding(>0)-gradient rate, stable over long chains.
  5. RECURRENT vs INPUT Jacobian: ||d h_t/d h_{t-1}|| against ||d h_t/d x_t||
     -- is the state integrated from the past or refreshed by new input?
  6. GRU UPDATE GATE z_t: since h_t = z_t*h_{t-1} + (1-z_t)*n_t, the mean and
     high-retention fraction of z_t directly show whether training carved out
     integrator (memory) units. This is the most interpretable "why".

All quantities are evaluated at the REAL operating points (h_{t-1}, x_t) visited
by the trained policy along sampled trajectories -- not random states -- because
the Jacobian is state-dependent and only the visited region matters.

This reuses the env / checkpoint / network machinery in measure_value_drift.
Forward + autodiff only; no training. Run (in tmux), e.g.:

    .venv/bin/python analysis/measure_recurrent_jacobians.py \\
        --run-uid acrobot_dual_value_ac_h64 --model-name rec_ppo_dual_value
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
from stoix.networks.utils import parse_rnn_cell  # noqa: E402

REPO_ROOT = m.REPO_ROOT
CKPT_STRIDE = 9728


# --------------------------------------------------------------------------- #
# Extracting the GRU cell and pre-torso from restored critic params
# --------------------------------------------------------------------------- #
def make_gru_step_fns(cfg, net_params, hidden_dim, cell_type, network="critic"):
    """Return (embed_fn, cell_step_fn) operating on the chosen recurrent network.

    network: "critic" (recurrent value) or "actor" (e.g. recurrent Direct Backprop
    policy). embed_fn(obs) -> input embedding x_t the GRU sees (pre_torso applied to
    the observation). cell_step_fn(h_prev, x) -> h_t via the GRU cell. Both use the
    restored parameters, so Jacobians are for the trained network. We rebuild the
    sub-modules from the checkpoint's param subtree.
    """
    # Params tree: {'params': {'pre_torso': ..., 'ScannedRNN_0': {'GRUCell_1':
    # {...}}, 'post_torso': ..., <head>: ...}}.
    params = net_params["params"]
    gru_params = {"params": params["ScannedRNN_0"]["GRUCell_1"]}
    pre_params = {"params": params["pre_torso"]}

    # Rebuild the pre_torso module from config to apply it with restored params.
    import hydra
    torso_cfg = (
        cfg.network.actor_network.pre_torso
        if network == "actor"
        else cfg.network.critic_network.pre_torso
    )
    pre_torso = hydra.utils.instantiate(torso_cfg)

    def embed_fn(obs):
        # obs: (obs_dim,) -> (embed_dim,). MLPTorso expects a batch-free vector fine.
        return pre_torso.apply(pre_params, obs)

    cell = parse_rnn_cell(cell_type)(features=hidden_dim)

    def cell_step_fn(h_prev, x):
        # h_prev: (H,), x: (embed_dim,). GRUCell operates unbatched here.
        new_h, _ = cell.apply(gru_params, h_prev, x)
        return new_h

    return embed_fn, cell_step_fn, gru_params, cell


def gru_update_gate(gru_params, cell, h_prev, x):
    """Compute the GRU update gate z_t = sigmoid(iz(x) + hz(h_prev)) per unit.

    Mirrors flax GRUCell internals: h_t = (1-z)*n + z*h_prev, so z is the
    per-unit memory-retention coefficient (diagonal of the Jacobian's linear part).
    """
    p = gru_params["params"]
    # Dense layers: iz/hz kernels; iz has bias, hz does not (flax GRU convention).
    iz = x @ p["iz"]["kernel"] + p["iz"]["bias"]
    hz = h_prev @ p["hz"]["kernel"]
    return jax.nn.sigmoid(iz + hz)


# --------------------------------------------------------------------------- #
# Trajectory collection: real (h_{t-1}, x_t) operating points
# --------------------------------------------------------------------------- #
def collect_operating_points(embed_fn, cell_step_fn, obs_seq, done_seq, hidden_dim):
    """Roll the GRU over a real observation sequence to get (h_{t-1}, x_t, h_t).

    obs_seq: (L, obs_dim), done_seq: (L,). Returns dict of stacked (L, ...) with
    the embeddings x_t and the hidden states h_t (h_0 = zeros). We reset the
    hidden state on done flags exactly like ScannedRNN.
    """
    def scan_step(h_prev, od):
        obs, done = od
        h_prev = jnp.where(done, jnp.zeros_like(h_prev), h_prev)
        x = embed_fn(obs)
        h_t = cell_step_fn(h_prev, x)
        return h_t, {"h_prev": h_prev, "x": x, "h_t": h_t}

    h0 = jnp.zeros((hidden_dim,))
    _, rec = jax.lax.scan(scan_step, h0, (obs_seq, done_seq))
    return rec  # each (L, ...)


# --------------------------------------------------------------------------- #
# Jacobian metrics
# --------------------------------------------------------------------------- #
def analyse_trajectory(cell_step_fn, gru_params, cell, embed_fn,
                       h_prev_seq, x_seq, obs_seq, max_lag):
    """Compute per-step Jacobian metrics along one trajectory.

    Returns per-lag k=1..max_lag: spectral & frobenius norm of the k-step product
    d h_t/d h_{t-k} (evaluated at the LAST step t=L-1), plus per-step eigenvalue /
    gate stats aggregated over the trajectory.
    """
    L = x_seq.shape[0]
    H = h_prev_seq.shape[-1]

    # Per-step recurrent Jacobian J_t = d h_t / d h_{t-1} at fixed input x_t.
    def jac_at(h_prev, x):
        return jax.jacobian(lambda h: cell_step_fn(h, x))(h_prev)  # (H, H)

    J_all = jax.vmap(jac_at)(h_prev_seq, x_seq)  # (L, H, H)

    # Per-step input Jacobian d h_t / d x_t (for the refresh-vs-integrate ratio).
    def input_jac_at(h_prev, x):
        return jax.jacobian(lambda xx: cell_step_fn(h_prev, xx))(x)  # (H, embed)

    Jx_all = jax.vmap(input_jac_at)(h_prev_seq, x_seq)  # (L, H, embed)

    # --- k-step products ending at the final step t = L-1 ---
    # P_k = J_{L-1} J_{L-2} ... J_{L-k}. Build cumulatively backwards.
    def scan_prod(carry, J_km1):
        P = carry @ J_km1  # extend the product by one earlier step
        return P, {"spec": _spectral_norm(P), "frob": jnp.linalg.norm(P)}

    # J ordered from t=L-1 backwards: J_all[L-1], J_all[L-2], ...
    J_rev = J_all[::-1]  # (L, H, H): index 0 = J_{L-1}
    P0 = J_rev[0]
    _, norm_seq = jax.lax.scan(scan_prod, P0, J_rev[1:max_lag])
    # Prepend k=1 (just J_{L-1}).
    spec_k = jnp.concatenate([jnp.array([_spectral_norm(P0)]), norm_seq["spec"]])
    frob_k = jnp.concatenate([jnp.array([jnp.linalg.norm(P0)]), norm_seq["frob"]])

    # --- Per-step eigenvalues of J_t: spectral radius, near-unit count, complex frac ---
    eig = jax.vmap(lambda J: jnp.linalg.eigvals(J))(J_all)  # (L, H) complex
    eig_mag = jnp.abs(eig)
    spectral_radius = jnp.mean(jnp.max(eig_mag, axis=-1))
    near_unit = jnp.mean(jnp.sum(eig_mag > 0.9, axis=-1))  # avg # modes |lambda|>0.9
    complex_frac = jnp.mean(jnp.mean(jnp.abs(eig.imag) > 1e-4, axis=-1))

    # --- Recurrent vs input Jacobian magnitude (per step, averaged) ---
    rec_norm = jnp.mean(jax.vmap(_spectral_norm)(J_all))
    inp_norm = jnp.mean(jax.vmap(_spectral_norm)(Jx_all))

    # --- GRU update gate z_t (memory retention) ---
    z_all = jax.vmap(lambda h, x: gru_update_gate(gru_params, cell, h, x))(
        h_prev_seq, x_seq
    )  # (L, H)
    z_mean = jnp.mean(z_all)
    z_hi_frac = jnp.mean(z_all > 0.8)  # fraction of (step, unit) acting as integrators

    return {
        "spec_k": spec_k, "frob_k": frob_k,
        "spectral_radius": spectral_radius, "near_unit_modes": near_unit,
        "complex_frac": complex_frac, "rec_norm": rec_norm, "inp_norm": inp_norm,
        "z_mean": z_mean, "z_hi_frac": z_hi_frac,
    }


def _spectral_norm(A):
    """Top singular value of A (spectral / operator 2-norm)."""
    return jnp.linalg.svd(A, compute_uv=False)[0]


def top_lyapunov(J_rev, n_steps):
    """Top Lyapunov exponent of the product J_{L-1}...  via QR reorthonormalisation.

    J_rev: (n, H, H) ordered from the last step backwards. Returns (1/n) * sum
    log(diag R) for the leading direction -- the mean log-growth per step of the
    dominant gradient mode. Negative => gradients vanish; positive => explode.
    """
    H = J_rev.shape[-1]
    q0 = jnp.eye(H)[:, :1]  # track the single leading direction

    def step(carry, J):
        q, log_sum = carry
        v = J @ q
        qn, r = jnp.linalg.qr(v)
        log_sum = log_sum + jnp.log(jnp.abs(r[0, 0]) + 1e-30)
        return (qn, log_sum), None

    (q, log_sum), _ = jax.lax.scan(step, (q0, 0.0), J_rev[:n_steps])
    return log_sum / n_steps


def build_critic_apply_pieces(cfg, run_uid, model_name, step):
    """Restore critic params and build the GRU step fns + config dims."""
    # Rebuild networks only to discover dims / restore into the right pytree.
    _, eval_env = m.environments.make(cfg)
    num_actions, is_cont, a_min, a_max = m.describe_action_space(eval_env)
    obs_dim = int(eval_env.observation_space().generate_value().shape[-1])
    cfg.system.obs_dim = obs_dim
    cfg.system.action_dim = num_actions
    actor, critic, nr_critic, actor_rnn = m.build_networks(
        cfg, num_actions, is_cont, a_min, a_max
    )
    dummy = m.init_dummy_params(cfg, actor, critic, nr_critic)
    params = m.restore_checkpoint(run_uid, step, dummy, model_name=model_name)
    return params.critic_params, eval_env, obs_dim, actor, actor_rnn


def run(args):
    cfg = m.load_config(args.run_uid, model_name=args.model_name)
    cfg.num_devices = 1
    cfg.arch.num_envs = 1
    m._critic_hidden_dim = int(cfg.network.critic_network.rnn_layer.hidden_state_dim)
    m._critic_cell_type = str(cfg.network.critic_network.rnn_layer.cell_type)
    m._actor_hidden_dim = int(cfg.network.actor_network.rnn_layer.hidden_state_dim)
    m._actor_cell_type = str(cfg.network.actor_network.rnn_layer.cell_type)
    meta = m.read_checkpoint_metadata(args.run_uid, model_name=args.model_name)
    m._actor_nonrecurrent = "nonrec_actor" in meta.get("system", {}).get("system_name", "")

    hidden_dim = m._critic_hidden_dim
    cell_type = m._critic_cell_type

    # Build env + actor once (actor only to sample trajectories); reuse across ckpts.
    print("Building environment + networks...")
    cfg0 = m.load_config(args.run_uid, model_name=args.model_name)
    cfg0.num_devices = 1
    cfg0.arch.num_envs = 1
    _, eval_env = m.environments.make(cfg0)
    num_actions, is_cont, a_min, a_max = m.describe_action_space(eval_env)
    obs_dim = int(eval_env.observation_space().generate_value().shape[-1])
    cfg0.system.obs_dim = obs_dim
    cfg0.system.action_dim = num_actions
    m._action_feature_dim = int(getattr(eval_env, "action_feature_dim", 0))
    actor, critic, nr_critic, actor_rnn = m.build_networks(
        cfg0, num_actions, is_cont, a_min, a_max
    )
    actor_apply, critic_apply, _ = m.make_apply_fns(actor, critic, nr_critic)
    dummy = m.init_dummy_params(cfg0, actor, critic, nr_critic)

    available = sorted(int(x) for x in os.listdir(
        os.path.join(REPO_ROOT, "checkpoints", args.model_name, args.run_uid)) if x.isdigit())
    if args.ckpt_steps:
        steps = [int(s) for s in args.ckpt_steps]
    else:
        idx = jnp.linspace(0, len(available) - 1, args.max_ckpts).round().astype(int)
        steps = [available[int(i)] for i in idx]
    print(f"Sweeping {len(steps)} checkpoints: {steps}")

    from tqdm import tqdm
    key = jax.random.PRNGKey(args.seed)
    rows = []
    for step in tqdm(steps, desc="checkpoints"):
        params = m.restore_checkpoint(args.run_uid, step, dummy, model_name=args.model_name)
        embed_fn, cell_step_fn, gru_params, cell = make_gru_step_fns(
            cfg0, params.critic_params, hidden_dim, cell_type
        )

        # Sample trajectories with THIS checkpoint's policy (on-distribution states).
        key, sk = jax.random.split(key)
        traj = jax.jit(lambda k: m.sample_prefix_trajectories(
            eval_env, actor_apply, actor_rnn, params, k,
            args.num_traj, args.prefix_len, obs_dim,
        ))(sk)
        keep = jnp.where(traj["survived"])[0]
        obs_seq_b = traj["obs_seq"][keep]          # (N, L, obs_dim)
        done_seq_b = traj["done_seq"][keep]        # (N, L)

        def per_traj(obs_seq, done_seq):
            rec = collect_operating_points(embed_fn, cell_step_fn, obs_seq, done_seq, hidden_dim)
            metrics = analyse_trajectory(
                cell_step_fn, gru_params, cell, embed_fn,
                rec["h_prev"], rec["x"], obs_seq, args.max_lag,
            )
            # Top Lyapunov over the whole trajectory product.
            def jac_at(h, x):
                return jax.jacobian(lambda hh: cell_step_fn(hh, x))(h)
            J_all = jax.vmap(jac_at)(rec["h_prev"], rec["x"])
            lyap = top_lyapunov(J_all[::-1], args.prefix_len)
            metrics["lyapunov_top"] = lyap
            return metrics

        res = jax.jit(jax.vmap(per_traj))(obs_seq_b, done_seq_b)
        res = jax.block_until_ready(res)
        agg = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), res)
        rows.append({
            "step": step,
            "num_traj_used": int(keep.shape[0]),
            "spec_norm_by_lag": [float(v) for v in agg["spec_k"]],
            "frob_norm_by_lag": [float(v) for v in agg["frob_k"]],
            "spectral_radius": float(agg["spectral_radius"]),
            "near_unit_modes": float(agg["near_unit_modes"]),
            "complex_frac": float(agg["complex_frac"]),
            "recurrent_jac_norm": float(agg["rec_norm"]),
            "input_jac_norm": float(agg["inp_norm"]),
            "gate_z_mean": float(agg["z_mean"]),
            "gate_z_high_frac": float(agg["z_hi_frac"]),
            "lyapunov_top": float(agg["lyapunov_top"]),
        })

    return {
        "meta": {
            "run_uid": args.run_uid, "model_name": args.model_name,
            "env": f"{cfg0.env.env_name}/{cfg0.env.scenario.task_name}",
            "hidden_state_dim": hidden_dim, "cell_type": cell_type,
            "num_traj": args.num_traj, "prefix_len": args.prefix_len,
            "max_lag": args.max_lag, "seed": args.seed,
        },
        "checkpoints": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-uid", required=True)
    p.add_argument("--model-name", default="rec_ppo_dual_value")
    p.add_argument("--ckpt-steps", nargs="*", type=int, default=None)
    p.add_argument("--max-ckpts", type=int, default=6)
    p.add_argument("--num-traj", type=int, default=64)
    p.add_argument("--prefix-len", type=int, default=32)
    p.add_argument("--max-lag", type=int, default=16,
                   help="Max lag k for the k-step product norms d h_t/d h_{t-k}.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = run(args)
    mt = results["meta"]
    print("\n" + "=" * 92)
    print(f"RECURRENT JACOBIAN ANALYSIS  [{mt['env']}, hidden={mt['hidden_state_dim']}, "
          f"cell={mt['cell_type']}, run={mt['run_uid']}]")
    print("=" * 92)
    print("  J_t = d h_t/d h_{t-1}.  Per-step / product metrics along real trajectories.")
    print("  rho=spectral radius of J_t | z_mean=mean GRU update gate (memory retention)")
    print("  lyap=top Lyapunov exp (mean log-growth/step; <0 vanish, >0 explode)")
    print("  rec/inp = ||d h/d h_prev|| vs ||d h/d x|| (integrate vs refresh)")
    print("-" * 92)
    print(f"{'step':>9}{'rho':>7}{'#|l|>.9':>9}{'cplx':>7}{'z_mean':>8}{'z>0.8':>7}"
          f"{'rec':>7}{'inp':>7}{'lyap':>8}{'||dh_t/dh_{t-8}||':>18}")
    for r in results["checkpoints"]:
        spec8 = r["spec_norm_by_lag"][min(7, len(r["spec_norm_by_lag"]) - 1)]
        print(f"{r['step']:>9}{r['spectral_radius']:>7.3f}{r['near_unit_modes']:>9.1f}"
              f"{r['complex_frac']:>7.2f}{r['gate_z_mean']:>8.3f}{r['gate_z_high_frac']:>7.2f}"
              f"{r['recurrent_jac_norm']:>7.2f}{r['input_jac_norm']:>7.2f}"
              f"{r['lyapunov_top']:>8.3f}{spec8:>18.4f}")
    print("=" * 92)
    print("  Full per-lag norm curves (spec_norm_by_lag / frob_norm_by_lag) are in the JSON.")

    out = args.out or f"analysis/recurrent_jacobians_{args.run_uid}.json"
    out_path = os.path.join(REPO_ROOT, out) if not os.path.isabs(out) else out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
