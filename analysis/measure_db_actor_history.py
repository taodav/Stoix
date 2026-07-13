"""Actor history-ablation for recurrent Direct Backprop.

The value-side ablation (measure_history_ablation.py) asks how much a recurrent
VALUE function depends on history. This is the ACTOR-side analog for the recurrent
Direct Backprop policy: how much does the (deterministic) action the policy takes
at a state S depend on the history that preceded S?

Why a separate script? The DB system differs from the PPO dual-value systems:
  * The policy is DETERMINISTIC (DeterministicHead + ScalePostProcessor), so there
    is no action distribution to take a KL of. We instead measure how much the
    CHOSEN ACTION moves when the preceding history is swapped -- the action-space
    analog of the value swap-ablation |V_real - V_swap|.
  * The learner_state carries ActorCriticParams with a feed-forward (or absent)
    critic; there is no recurrent critic / nr_critic. This is a pure actor probe.
  * The actor is `MemoryToggleRecurrentActor` (GRU with an optional reset-every-step
    memory kill), rebuilt here to match the checkpoint.

Method (mirrors the value swap-ablation)
----------------------------------------
Sample states S = final state of a length-L prefix rolled out by the trained
deterministic policy (plus exploration noise, matching training). Then read the
policy's action at S under histories that all end at the same S:
  * real  -- the true prefix o_0..o_{L-1}, then o_L = S
  * swap  -- a DIFFERENT trajectory's real prefix (batch derangement), then o_L = S
  * zero  -- reset the hidden state right at S (memoryless action at S)

We report, per checkpoint:
  * |a_real - a_swap|  : mean L2 action change under a swapped real history
  * action_spread      : std of a_real across states (how much the action varies
                         across S at all) -- the natural normaliser
  * sens_swap          : |a_real - a_swap| / action_spread  (~0 => ignores history)
  * |a_real - a_zero|  : action change when memory is killed at S (secondary; the
                         zero-hidden state is in-distribution here because the
                         MEMORYLESS DB variant is trained exactly that way)

For the memoryless checkpoint (reset_hidden_state_every_step=True) every metric is
~0 by construction -- a useful correctness control.

Forward passes only, no differentiable rollout, no training. Run (in tmux):

    .venv/bin/python analysis/measure_db_actor_history.py \\
        --run-uid acrobot_db_memory_ac
    .venv/bin/python analysis/measure_db_actor_history.py \\
        --run-uid acrobot_db_memfree     # control: should read ~0
"""

import argparse
import json
import os
import sys

import chex
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import measure_value_drift as m  # noqa: E402  (env/checkpoint/config helpers)
from stoix.networks.base import ScannedRNN  # noqa: E402
from stoix.systems.direct_backprop.rec_direct_backprop import (  # noqa: E402
    MemoryToggleRecurrentActor,
)
from stoix.networks.base import CompositeNetwork  # noqa: E402
from stoix.networks.postprocessors import tanh_to_spec  # noqa: E402

REPO_ROOT = m.REPO_ROOT
MODEL_NAME = "rec_direct_backprop"


def load_db_config(run_uid: str):
    """Compose the DB config aligned to the checkpoint (env + arch + action head).

    We reuse measure_value_drift.load_config's metadata-driven overrides but point
    it at the DB default config. It restores the env (incl. the ActionConcatWrapper),
    the GRU hidden dim / cell type, and gamma from the checkpoint metadata.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    meta = m.read_checkpoint_metadata(run_uid, model_name=MODEL_NAME)
    meta_net = meta.get("network", {})
    meta_env = meta.get("env", {})
    meta_sys = meta.get("system", {})

    overrides = []
    env_name = meta_env.get("env_name")
    task = meta_env.get("scenario", {}).get("task_name")
    # Map task_name back to the env config group. differentiable_acrobot lives at
    # env/differentiable/acrobot(.yaml); other suites use <env_name>/<task>.
    if env_name == "differentiable":
        env_group = "differentiable/acrobot"
    else:
        env_group = f"{env_name}/{task}"
    overrides.append(f"env={env_group}")
    wrapper = meta_env.get("wrapper", {})
    wrapper = wrapper.get("_target_") if isinstance(wrapper, dict) else None
    if wrapper:
        overrides.append(f"+env.wrapper._target_={wrapper}")
    rnn = meta_net.get("actor_network", {}).get("rnn_layer", {})
    if "hidden_state_dim" in rnn:
        overrides.append(
            f"network.actor_network.rnn_layer.hidden_state_dim={rnn['hidden_state_dim']}"
        )
    if "cell_type" in rnn:
        overrides.append(f"network.actor_network.rnn_layer.cell_type={rnn['cell_type']}")

    config_dir = os.path.join(REPO_ROOT, "stoix/configs/default/anakin")
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="default_rec_direct_backprop", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    # Carry the memory mode from the checkpoint (True => memoryless control).
    cfg.system.reset_hidden_state_every_step = bool(
        meta_sys.get("reset_hidden_state_every_step", False)
    )
    return cfg, int(meta_sys.get("action_feature_dim", 0))


def build_db_actor(cfg, action_dim, a_min, a_max):
    """Rebuild the MemoryToggleRecurrentActor exactly as the DB system does."""
    import hydra

    pre_torso = hydra.utils.instantiate(cfg.network.actor_network.pre_torso)
    post_torso = hydra.utils.instantiate(cfg.network.actor_network.post_torso)
    action_head = hydra.utils.instantiate(
        cfg.network.actor_network.action_head, action_dim=action_dim
    )
    post_processor = hydra.utils.instantiate(
        cfg.network.actor_network.post_processor,
        minimum=a_min,
        maximum=a_max,
        scale_fn=tanh_to_spec,
    )
    action_head = CompositeNetwork([action_head, post_processor])
    actor = MemoryToggleRecurrentActor(
        pre_torso=pre_torso,
        hidden_state_dim=cfg.network.actor_network.rnn_layer.hidden_state_dim,
        cell_type=cfg.network.actor_network.rnn_layer.cell_type,
        post_torso=post_torso,
        action_head=action_head,
        reset_hidden_state_every_step=cfg.system.reset_hidden_state_every_step,
    )
    actor_rnn = ScannedRNN(
        hidden_state_dim=cfg.network.actor_network.rnn_layer.hidden_state_dim,
        cell_type=cfg.network.actor_network.rnn_layer.cell_type,
    )
    return actor, actor_rnn


def restore_actor_params(run_uid, step):
    """Restore just the actor_params from a DB checkpoint (critic may be None)."""
    import orbax.checkpoint as ocp

    directory = os.path.join(REPO_ROOT, "checkpoints", MODEL_NAME, run_uid)
    mgr = ocp.CheckpointManager(directory, ocp.PyTreeCheckpointer())
    available = sorted(int(x) for x in os.listdir(directory) if x.isdigit())
    if step not in available:
        raise SystemExit(f"Step {step} not in {directory}. Available: {available}")
    return mgr.restore(step)["learner_state"]["params"]["actor_params"]


def _available_steps(run_uid):
    d = os.path.join(REPO_ROOT, "checkpoints", MODEL_NAME, run_uid)
    return sorted(int(x) for x in os.listdir(d) if x.isdigit())


def sample_prefix_states(env, actor_apply, actor_params, actor_rnn, key,
                         num_traj, prefix_len, obs_dim, action_dim,
                         action_min, action_max, exploration_noise, actor_strip_dim):
    """Roll the deterministic policy (+ training-style noise) for prefix_len steps.

    Returns obs_seq (num_traj, L, obs_dim), final_obs (num_traj, obs_dim),
    survived (num_traj,). The actor sees the FULL obs when recurrent, or the
    stripped obs when memoryless (actor_strip_dim>0).
    """
    keys = jax.random.split(key, num_traj)
    init_h = actor_rnn.initialize_carry(1)

    def rollout_one(traj_key):
        reset_key, scan_key = jax.random.split(traj_key)
        env_state, ts = env.reset(reset_key)

        def step_fn(carry, step_key):
            env_state, obs, hstate, done, alive = carry
            actor_obs = m.strip_action_features(obs, actor_strip_dim)
            ac_in = (actor_obs[jnp.newaxis, jnp.newaxis, :], done.reshape(1, 1))
            new_h, pi = actor_apply(actor_params, hstate, ac_in)
            action = pi.mode()[0, 0]  # deterministic action, drop (time,batch)
            step_key, nkey = jax.random.split(step_key)
            action = action + jax.random.normal(nkey, action.shape) * exploration_noise
            action = jnp.clip(action, action_min, action_max)
            new_env_state, new_ts = env.step(env_state, action)
            record = {"obs": obs}
            new_done = new_ts.last().reshape(())
            new_alive = jnp.logical_and(alive, jnp.logical_not(done))
            return (new_env_state, new_ts.observation, new_h, new_done, new_alive), record

        init_carry = (env_state, ts.observation, init_h, jnp.array(False), jnp.array(True))
        step_keys = jax.random.split(scan_key, prefix_len)
        (fes, final_obs, _, final_done, alive_through), records = jax.lax.scan(
            step_fn, init_carry, step_keys
        )
        survived = jnp.logical_and(alive_through, jnp.logical_not(final_done))
        return {"obs_seq": records["obs"], "final_obs": final_obs, "survived": survived}

    return jax.vmap(rollout_one)(keys)


def action_under_histories(actor_apply, actor_params, obs_seq, final_obs,
                           swap_idx, actor_rnn, actor_strip_dim, action_min, action_max):
    """Deterministic action at S under real / swap / zero-hidden history.

    All feed the same o_L = S as the last step. The action is pi.mode() (post-
    processed, i.e. in true action units). No exploration noise here -- we probe
    the deterministic policy map.
    """
    h_dim = actor_rnn.hidden_state_dim
    cell = actor_rnn.cell_type

    def action_for_prefix(prefix_obs, o_final, reset_at_S):
        seq = jnp.concatenate([prefix_obs, o_final[jnp.newaxis, :]], axis=0)  # (L+1, d)
        seq = m.strip_action_features(seq, actor_strip_dim)
        done = jnp.zeros((seq.shape[0], 1), dtype=bool)
        done = done.at[-1, 0].set(reset_at_S)  # reset the hidden state right at S
        init_h = ScannedRNN(h_dim, cell).initialize_carry(1)
        _, pi = actor_apply(actor_params, init_h, (seq[:, jnp.newaxis, :], done))
        return pi.mode()[-1, 0]  # action at S, shape (action_dim,)

    a_real = jax.vmap(lambda o, f: action_for_prefix(o, f, False))(obs_seq, final_obs)
    a_swap = jax.vmap(lambda o, f: action_for_prefix(o, f, False))(obs_seq[swap_idx], final_obs)
    a_zero = jax.vmap(lambda o, f: action_for_prefix(o, f, True))(obs_seq, final_obs)
    return a_real, a_swap, a_zero


def analyse_checkpoint(args, env, actor, actor_apply, actor_rnn, cfg,
                       obs_dim, action_dim, a_min, a_max, actor_strip_dim, step, key):
    params = restore_actor_params(args.run_uid, step)
    key, sk = jax.random.split(key)
    traj = jax.jit(lambda k: sample_prefix_states(
        env, actor_apply, params, actor_rnn, k, args.num_traj, args.prefix_len,
        obs_dim, action_dim, a_min, a_max, cfg.system.exploration_noise, actor_strip_dim,
    ))(sk)
    jax.block_until_ready(traj)

    keep = jnp.where(traj["survived"])[0]
    if int(keep.shape[0]) < 2:
        return {"step": step, "error": f"only {int(keep.shape[0])} survived"}
    obs_seq = traj["obs_seq"][keep]
    final_obs = traj["final_obs"][keep]
    n = int(keep.shape[0])
    swap_idx = (jnp.arange(n) + 1) % n  # cyclic derangement

    a_real, a_swap, a_zero = jax.jit(lambda: action_under_histories(
        actor_apply, params, obs_seq, final_obs, swap_idx, actor_rnn,
        actor_strip_dim, a_min, a_max,
    ))()
    jax.block_until_ready((a_real, a_swap, a_zero))

    # L2 over the action dimension, then mean over states.
    d_swap = float(jnp.mean(jnp.linalg.norm(a_real - a_swap, axis=-1)))
    d_zero = float(jnp.mean(jnp.linalg.norm(a_real - a_zero, axis=-1)))
    # Action spread: std of each action component across states, averaged.
    spread = float(jnp.mean(jnp.std(a_real, axis=0))) + 1e-8
    return {
        "step": step,
        "num_traj_used": n,
        "abs_action_real_minus_swap": d_swap,
        "abs_action_real_minus_zero": d_zero,
        "action_spread": spread,
        "sens_swap": d_swap / spread,
        "sens_zero": d_zero / spread,
        "mean_action": float(jnp.mean(a_real)),
    }


def run(args):
    cfg, action_feature_dim = load_db_config(args.run_uid)
    cfg.num_devices = 1
    cfg.arch.num_envs = 1

    print("Building environment...")
    _, eval_env = m.environments.make(cfg)
    action_dim, is_cont, a_min, a_max = m.describe_action_space(eval_env)
    obs_dim = int(eval_env.observation_space().generate_value().shape[-1])
    memoryless = bool(cfg.system.reset_hidden_state_every_step)
    # The actor strips the appended action only when it is memoryless.
    actor_strip_dim = action_feature_dim if memoryless else 0
    print(f"obs_dim={obs_dim} action_dim={action_dim} "
          f"action_feature_dim={action_feature_dim} memoryless={memoryless}")

    actor, actor_rnn = build_db_actor(cfg, action_dim, a_min, a_max)
    actor_apply = actor.apply

    steps = ([int(s) for s in args.ckpt_steps] if args.ckpt_steps
             else _available_steps(args.run_uid))
    if args.max_ckpts and len(steps) > args.max_ckpts:
        # Evenly subsample to at most max_ckpts (keep first and last).
        idx = jnp.linspace(0, len(steps) - 1, args.max_ckpts).round().astype(int)
        steps = [steps[int(i)] for i in idx]
    print(f"Sweeping {len(steps)} checkpoints.")

    from tqdm import tqdm
    key = jax.random.PRNGKey(args.seed)
    rows = []
    for step in tqdm(steps, desc="checkpoints"):
        key, ck = jax.random.split(key)
        rows.append(analyse_checkpoint(
            args, eval_env, actor, actor_apply, actor_rnn, cfg,
            obs_dim, action_dim, a_min, a_max, actor_strip_dim, step, ck,
        ))
    return {
        "meta": {
            "run_uid": args.run_uid, "model_name": MODEL_NAME,
            "env": f"{cfg.env.env_name}/{cfg.env.scenario.task_name}",
            "memoryless": memoryless, "action_feature_dim": action_feature_dim,
            "hidden_state_dim": int(cfg.network.actor_network.rnn_layer.hidden_state_dim),
            "num_traj": args.num_traj, "prefix_len": args.prefix_len, "seed": args.seed,
        },
        "checkpoints": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-uid", required=True,
                   help="Checkpoint folder under checkpoints/rec_direct_backprop/.")
    p.add_argument("--ckpt-steps", nargs="*", type=int, default=None)
    p.add_argument("--max-ckpts", type=int, default=10,
                   help="Subsample the sweep to at most this many checkpoints.")
    p.add_argument("--num-traj", type=int, default=256)
    p.add_argument("--prefix-len", type=int, default=32,
                   help="Prefix length L; S is the state after L steps (DB rollout_length=32).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = run(args)
    mt = results["meta"]
    print("\n" + "=" * 84)
    print(f"DB ACTOR HISTORY-ABLATION  [{mt['env']}, hidden={mt['hidden_state_dim']}, "
          f"run={mt['run_uid']}, memoryless={mt['memoryless']}]")
    print("=" * 84)
    print("  Does the DETERMINISTIC action at S change when the preceding history changes?")
    print("  sens_swap = mean_S ||a_real - a_swap|| / std_S a_real   (~0 => actor ignores history)")
    print("-" * 84)
    print(f"{'step':>9}{'|a_real-a_swap|':>17}{'|a_real-a_zero|':>17}{'spread':>9}"
          f"{'sens_swap':>11}{'sens_zero':>11}")
    for r in results["checkpoints"]:
        if "error" in r:
            print(f"{r['step']:>9}   {r['error']}")
            continue
        print(f"{r['step']:>9}{r['abs_action_real_minus_swap']:>17.4f}"
              f"{r['abs_action_real_minus_zero']:>17.4f}{r['action_spread']:>9.3f}"
              f"{r['sens_swap']:>11.4f}{r['sens_zero']:>11.4f}")
    print("=" * 84)
    print("  Memoryless control (memoryless=True) should read ~0 everywhere.")

    out = args.out or f"analysis/db_actor_history_{args.run_uid}.json"
    out_path = os.path.join(REPO_ROOT, out) if not os.path.isabs(out) else out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
