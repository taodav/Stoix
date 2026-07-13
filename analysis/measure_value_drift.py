"""Measure recurrent vs. non-recurrent "value drift" between two PPO checkpoints.

Background
----------
We train recurrent PPO on CartPole with an auxiliary NON-recurrent value head
(see ``stoix/systems/ppo/anakin/rec_ppo_dual_value.py``). Given two checkpoints
from the same run -- an early policy ``pi_u`` (update ``u``) and a much later
policy ``pi_u'`` (update ``u'``, with ``u << u'``) -- we ask how off-policy the
later value functions are with respect to the earlier policy's trajectories.

Concretely, we sample a set of trajectories ``tau ~ pi_u``. For each trajectory
we take its final state ``S`` (the last state of a fixed-length prefix) and
compute four mean-squared errors:

    Recurrent value head V_hat (a function of the whole trajectory tau):
        E_tau [ V_hat^{u'}(tau) - V_{pi_u}(S)  ]^2      (drift vs OLD policy)
        E_tau [ V_hat^{u'}(tau) - V_{pi_u'}(S) ]^2      (agreement w/ NEW policy)

    Non-recurrent value head V_hat_nr (a function of the state S only):
        E_tau [ V_hat_nr^{u'}(S) - V_{pi_u}(S)  ]^2
        E_tau [ V_hat_nr^{u'}(S) - V_{pi_u'}(S) ]^2

``V_{pi}(S)`` is the ground-truth value of state ``S`` under policy ``pi``,
estimated by Monte-Carlo: we branch the (functional) gymnax environment at ``S``
and roll out many discounted continuations under ``pi``.

The gap between the "vs OLD policy" and "vs NEW policy" errors tells us how much
"value iteration" has happened between ``u`` and ``u'`` -- i.e. how much the
value function has moved on from the state distribution / returns of ``pi_u``.

Notes on faithfulness to training
----------------------------------
* Networks are rebuilt exactly as in the training system and parameters are
  restored from the checkpoint. The architecture (GRU hidden dim, cell type) and
  gamma are auto-read from the checkpoint metadata so they always match the run.
* Policies act stochastically (sample from the categorical), matching training.
* The recurrent hidden state is carried across steps and reset on episode
  boundaries, exactly as in ``rec_ppo_dual_value``. The non-recurrent head is
  always fed a zeroed hidden state (reset flag ``True``), matching how it is
  trained.
* FAITHFUL POLICY MEMORY: each policy's Monte-Carlo continuations from S start
  from the hidden state THAT policy would actually hold at S, reconstructed by
  replaying the prefix o_0..o_{L-1} through that policy's own actor
  (``replay_policy_hstate``). pi_u and pi_u' therefore each get their own memory
  -- we do not reset the GRU at the branch point.
* The recurrent value head is read AT S = s_L (the prefix's last obs o_L is
  appended before reading the value), so all four quantities refer to the same
  state S that the Monte-Carlo ground truth branches from.
* CartPole auto-resets on termination inside gymnax, so every Monte-Carlo
  rollout carries an ``alive`` mask that zeroes out any reward earned after the
  episode has ended.

Cost & caching
--------------
The expensive part is the Monte-Carlo ground truth: num_traj * mc_rollouts *
max_horizon env steps, done for BOTH policies. This is independent of which
value head we score, so it is computed once and cached to a .npz. Re-scoring
(e.g. different heads, plotting, re-analysis) reads the cache and is instant.
We also report an MC-noise-CORRECTED MSE (subtracting the MC estimator's mean
squared standard error), which is an unbiased estimate of the error vs the TRUE
value -- letting you trust a modest ``mc_rollouts`` instead of a huge one.

This script only MEASURES; it does not train. Run it yourself (e.g. in tmux):

    .venv/bin/python analysis/measure_value_drift.py            # compute + score
    .venv/bin/python analysis/measure_value_drift.py --from-cache  # re-score only

Common overrides:

    --run-uid 20260708154913 --ckpt-u 29184 --ckpt-uprime 437760 \
    --num-traj 256 --prefix-len 64 --mc-rollouts 64 --max-horizon 300 \
    --seed 0 --out analysis/value_drift_results.json

Defaults: u = checkpoint 3 (step 29184), u' = checkpoint 45 (step 437760),
max_horizon = 300. (Checkpoints are 9728 steps apart: ckpt k = 9728 * k.)
"""

import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from jax_tqdm import scan_tqdm
from omegaconf import OmegaConf
from tqdm import tqdm

from stoix.networks.base import RecurrentActor, RecurrentCritic, ScannedRNN
from stoix.systems.ppo.anakin.rec_ppo_dual_value import (
    DualValueHiddenStates,
    DualValueParams,
)
from stoix.utils import make_env as environments

# Repo root (one level up from this file's directory).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Config + network / checkpoint loading
# --------------------------------------------------------------------------- #
def read_checkpoint_metadata(run_uid: str, model_name: str = "rec_ppo_dual_value") -> Dict[str, Any]:
    """Read the training config stored in a checkpoint's metadata.

    Stoix saves the full resolved config as ``custom_metadata`` in the orbax
    checkpoint. We use it to auto-match the network architecture (hidden dim,
    cell type) and gamma to whatever the run was actually trained with, rather
    than relying on the current config defaults.
    """
    import orbax.checkpoint as ocp

    directory = os.path.join(REPO_ROOT, "checkpoints", model_name, run_uid)
    manager = ocp.CheckpointManager(directory, ocp.PyTreeCheckpointer())
    md = manager.metadata()
    # In this orbax version metadata() returns a RootMetadata whose dict is under
    # .custom_metadata; older/newer versions may return a plain dict.
    custom = getattr(md, "custom_metadata", None)
    if custom is None and isinstance(md, dict):
        custom = md.get("custom_metadata", md)
    return dict(custom)


def load_config(run_uid: str, overrides: Tuple[str, ...] = (), model_name: str = "rec_ppo_dual_value") -> Any:
    """Compose the training config and align it to the checkpoint's architecture.

    Reads the environment, network hidden-state dims / cell types, and gamma from
    the checkpoint metadata and applies them as overrides, so the rebuilt env and
    networks exactly match the restored parameters regardless of current config
    defaults. This is what lets a single script handle CartPole, Acrobot, etc.
    without per-run edits.
    """
    meta = read_checkpoint_metadata(run_uid, model_name=model_name)
    meta_net = meta.get("network", {})
    meta_sys = meta.get("system", {})
    meta_env = meta.get("env", {})

    auto_overrides = []
    # Select the environment config group the run was trained on. The group name
    # is "<env_name>/<task_name>" (e.g. gymnax/acrobot), matching the training
    # default composition.
    env_name = meta_env.get("env_name")
    task_name = meta_env.get("scenario", {}).get("task_name")
    if env_name and task_name:
        env_group = f"{env_name}/{task_name}"
        env_cfg_path = os.path.join(REPO_ROOT, "stoix/configs/env", f"{env_group}.yaml")
        if not os.path.exists(env_cfg_path):
            raise SystemExit(
                f"Checkpoint was trained on env '{env_group}' but no config exists "
                f"at {env_cfg_path}. Pass the env explicitly via --overrides env=<group>."
            )
        auto_overrides.append(f"env={env_group}")
        # Re-attach any optional env wrapper the run was trained with (e.g. the
        # ActionConcatWrapper). The env config group is named by task_name and does
        # not encode the wrapper, so restore it explicitly from metadata; otherwise
        # the analysis rebuilds an UNWRAPPED env and the observation dims won't match
        # the trained networks.
        wrapper_target = meta_env.get("wrapper", {})
        wrapper_target = wrapper_target.get("_target_") if isinstance(wrapper_target, dict) else None
        if wrapper_target:
            # '+' appends the key: the base env config has no 'wrapper' entry.
            auto_overrides.append(f"+env.wrapper._target_={wrapper_target}")

    for role in ("actor_network", "critic_network"):
        rnn = meta_net.get(role, {}).get("rnn_layer", {})
        if "hidden_state_dim" in rnn:
            auto_overrides.append(
                f"network.{role}.rnn_layer.hidden_state_dim={rnn['hidden_state_dim']}"
            )
        if "cell_type" in rnn:
            auto_overrides.append(
                f"network.{role}.rnn_layer.cell_type={rnn['cell_type']}"
            )
        # Match the pre/post torso widths too (discrete runs used [128], continuous
        # uses [256]); the base config would otherwise impose its own default.
        for torso in ("pre_torso", "post_torso"):
            sizes = meta_net.get(role, {}).get(torso, {}).get("layer_sizes")
            if sizes is not None:
                # Hydra list override syntax: key=[a,b,c] with no spaces.
                auto_overrides.append(
                    f"network.{role}.{torso}.layer_sizes=[{','.join(str(s) for s in sizes)}]"
                )
    # Match the ACTION HEAD to the trained one (discrete CategoricalHead vs a
    # continuous head like NormalAffineTanhDistributionHead). The base config used
    # for composition defaults to the discrete head, so we override its target from
    # the checkpoint metadata; otherwise continuous runs fail to instantiate.
    head_target = meta_net.get("actor_network", {}).get("action_head", {}).get("_target_")
    if head_target:
        auto_overrides.append(f"network.actor_network.action_head._target_={head_target}")
    if "gamma" in meta_sys:
        auto_overrides.append(f"system.gamma={meta_sys['gamma']}")

    all_overrides = auto_overrides + list(overrides)
    config_dir = os.path.join(REPO_ROOT, "stoix/configs/default/anakin")
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="default_rec_ppo_dual_value", overrides=all_overrides)
    OmegaConf.set_struct(cfg, False)
    if auto_overrides:
        print("Auto-matched to checkpoint (env + architecture):")
        for o in auto_overrides:
            print(f"  {o}")
    return cfg


def describe_action_space(env) -> Tuple[int, bool, Optional[float], Optional[float]]:
    """Return (action_dim, is_continuous, minimum, maximum) for an env.

    Discrete spaces expose ``num_values``; continuous (box) spaces expose a
    trailing ``shape`` plus ``minimum``/``maximum`` bounds.
    """
    space = env.action_space()
    if hasattr(space, "num_values"):
        return int(space.num_values), False, None, None
    action_dim = int(space.shape[-1])
    return action_dim, True, float(space.minimum), float(space.maximum)


def build_networks(
    cfg: Any,
    num_actions: int,
    is_continuous: bool = False,
    action_minimum: Optional[float] = None,
    action_maximum: Optional[float] = None,
) -> Tuple[RecurrentActor, RecurrentCritic, RecurrentCritic, ScannedRNN]:
    """Rebuild the actor + recurrent critic + non-recurrent critic (same arch).

    For continuous action heads (e.g. NormalAffineTanhDistributionHead) the
    ``minimum``/``maximum`` bounds are passed through, matching how the training
    system instantiates the head.
    """
    import hydra

    action_head_kwargs = {"action_dim": num_actions}
    if is_continuous:
        action_head_kwargs["minimum"] = action_minimum
        action_head_kwargs["maximum"] = action_maximum

    actor = RecurrentActor(
        pre_torso=hydra.utils.instantiate(cfg.network.actor_network.pre_torso),
        hidden_state_dim=cfg.network.critic_network.rnn_layer.hidden_state_dim,
        cell_type=cfg.network.critic_network.rnn_layer.cell_type,
        post_torso=hydra.utils.instantiate(cfg.network.actor_network.post_torso),
        action_head=hydra.utils.instantiate(
            cfg.network.actor_network.action_head, **action_head_kwargs
        ),
    )
    critic = RecurrentCritic(
        pre_torso=hydra.utils.instantiate(cfg.network.critic_network.pre_torso),
        hidden_state_dim=cfg.network.critic_network.rnn_layer.hidden_state_dim,
        cell_type=cfg.network.critic_network.rnn_layer.cell_type,
        post_torso=hydra.utils.instantiate(cfg.network.critic_network.post_torso),
        critic_head=hydra.utils.instantiate(cfg.network.critic_network.critic_head),
    )
    # Non-recurrent critic: identical architecture, separate params.
    nr_critic = RecurrentCritic(
        pre_torso=hydra.utils.instantiate(cfg.network.critic_network.pre_torso),
        hidden_state_dim=cfg.network.critic_network.rnn_layer.hidden_state_dim,
        cell_type=cfg.network.critic_network.rnn_layer.cell_type,
        post_torso=hydra.utils.instantiate(cfg.network.critic_network.post_torso),
        critic_head=hydra.utils.instantiate(cfg.network.critic_network.critic_head),
    )
    actor_rnn = ScannedRNN(
        hidden_state_dim=cfg.network.actor_network.rnn_layer.hidden_state_dim,
        cell_type=cfg.network.actor_network.rnn_layer.cell_type,
    )
    return actor, critic, nr_critic, actor_rnn


def init_dummy_params(
    cfg: Any, actor: RecurrentActor, critic: RecurrentCritic, nr_critic: RecurrentCritic
) -> DualValueParams:
    """Build a params pytree with the right structure to restore into."""
    key = jax.random.PRNGKey(0)
    n_envs = 1
    # A single (time=1, batch=1) input, matching the training init.
    dummy_obs = jnp.zeros((1, n_envs, cfg.system.obs_dim), dtype=jnp.float32)
    dummy_done = jnp.zeros((1, n_envs), dtype=bool)
    dummy_x = (dummy_obs, dummy_done)
    h_actor = actor.hidden_state_dim
    h_critic = critic.hidden_state_dim
    init_h_actor = ScannedRNN(h_actor, actor.cell_type).initialize_carry(n_envs)
    init_h_critic = ScannedRNN(h_critic, critic.cell_type).initialize_carry(n_envs)
    k1, k2, k3 = jax.random.split(key, 3)
    actor_params = actor.init(k1, init_h_actor, dummy_x)
    critic_params = critic.init(k2, init_h_critic, dummy_x)
    nr_critic_params = nr_critic.init(k3, init_h_critic, dummy_x)
    return DualValueParams(actor_params, critic_params, nr_critic_params)


def restore_checkpoint(run_uid: str, timestep: int, dummy_params: DualValueParams, model_name: str = "rec_ppo_dual_value") -> DualValueParams:
    """Restore a specific checkpoint's params.

    NOTE: we restore directly via orbax rather than through Stoix's
    ``Checkpointer.restore_params``. The latter asserts on
    ``manager.metadata()["checkpointer_version"]``, but in the installed orbax
    version ``metadata()`` returns a ``RootMetadata`` object (dict lives under
    ``.custom_metadata``), so that subscript raises ``TypeError``. Doing the
    restore here keeps this analysis independent of that latent core bug. The
    logic below mirrors ``restore_params`` (minus the version assertion).
    """
    import orbax.checkpoint as ocp

    directory = os.path.join(REPO_ROOT, "checkpoints", model_name, run_uid)
    manager = ocp.CheckpointManager(directory, ocp.PyTreeCheckpointer())
    step = timestep if timestep is not None else manager.latest_step()
    available = manager.all_steps()
    if step not in available:
        raise SystemExit(
            f"Checkpoint step {step} not found in {directory}. "
            f"Available steps: {sorted(available)}"
        )
    restored = manager.restore(step)
    raw_params = restored["learner_state"]["params"]
    # Rebuild the DualValueParams pytree (same field names as the training system).
    return DualValueParams(**raw_params)


# --------------------------------------------------------------------------- #
# Rollout / value helpers
# --------------------------------------------------------------------------- #
def make_apply_fns(actor: RecurrentActor, critic: RecurrentCritic, nr_critic: RecurrentCritic):
    """Bundle the network apply functions."""
    return actor.apply, critic.apply, nr_critic.apply


def sample_prefix_trajectories(
    env,
    actor_apply,
    actor_rnn: ScannedRNN,
    params_u: DualValueParams,
    key: chex.PRNGKey,
    num_traj: int,
    prefix_len: int,
    obs_dim: int,
):
    """Roll pi_u on the eval env for exactly ``prefix_len`` steps per trajectory.

    Returns a dict with, per trajectory:
      * obs_seq:      (num_traj, prefix_len, obs_dim)  observations o_0..o_{L-1}
      * done_seq:     (num_traj, prefix_len)           done flag BEFORE each step
      * final_state:  env state at S (after L steps) -- pytree, leading dim num_traj
      * final_obs:    (num_traj, obs_dim)              observation at S (o_L)
      * survived:     (num_traj,) bool, True if no termination within the prefix
    The recurrent hidden state is carried and reset on episode boundaries,
    exactly as in the training rollout. We do not return the policy hidden state
    here: it is reconstructed downstream (per policy) by replaying obs_seq, so
    that pi_u and pi_u' each get the memory THEY would have at S.
    """
    keys = jax.random.split(key, num_traj)

    init_hstate = actor_rnn.initialize_carry(1)  # (1, H) per env

    def rollout_one(traj_key: chex.PRNGKey):
        reset_key, scan_key = jax.random.split(traj_key)
        env_state, ts = env.reset(reset_key)

        def step_fn(carry, step_key):
            env_state, obs, hstate, done, alive = carry
            # Network expects (time=1, batch=1, ...) and a done/reset flag.
            batched_obs = obs[jnp.newaxis, jnp.newaxis, :]
            # If the actor is non-recurrent, always reset its hidden state (matching
            # how it was trained in the rec_ppo_nonrec_actor system).
            actor_done = (
                jnp.ones_like(done.reshape(1, 1))
                if _actor_nonrecurrent
                else done.reshape(1, 1)
            )
            # A reactive (non-recurrent) actor is Markov and never sees the appended
            # previous action; a recurrent actor (dual_value) sees the full obs.
            actor_obs = (
                strip_action_features(batched_obs, _action_feature_dim)
                if _actor_nonrecurrent
                else batched_obs
            )
            ac_in = (actor_obs, actor_done)
            new_hstate, pi = actor_apply(params_u.actor_params, hstate, ac_in)
            # Network output has leading (time=1, batch=1) dims. Drop them; the env
            # wants a scalar (discrete) or an (action_dim,) vector (continuous).
            action = pi.sample(seed=step_key)[0, 0]
            new_env_state, new_ts = env.step(env_state, action)

            # Record the (obs, done) that FED this step.
            record = {"obs": obs, "done": done.reshape(())}

            # gymnax auto-resets; track whether this episode terminated at/after
            # this step so we know if S is a genuine non-terminal state.
            new_done = new_ts.last().reshape(())
            new_alive = jnp.logical_and(alive, jnp.logical_not(done))

            new_carry = (
                new_env_state,
                new_ts.observation,
                new_hstate,
                new_done,
                new_alive,
            )
            return new_carry, record

        init_carry = (
            env_state,
            ts.observation,
            init_hstate,
            jnp.array(False),
            jnp.array(True),
        )
        step_keys = jax.random.split(scan_key, prefix_len)
        final_carry, records = jax.lax.scan(step_fn, init_carry, step_keys)
        final_env_state, final_obs, _, final_done, alive_through = final_carry

        # "survived" == no termination happened during the whole prefix, so the
        # state S reached after L steps is a genuine non-terminal state.
        survived = jnp.logical_and(alive_through, jnp.logical_not(final_done))

        return {
            "obs_seq": records["obs"],          # (L, obs_dim)
            "done_seq": records["done"],        # (L,)
            "final_state": final_env_state,     # env state at S
            "final_obs": final_obs,             # obs at S
            "survived": survived,               # ()
        }

    return jax.vmap(rollout_one)(keys)


def replay_policy_hstate(
    actor_apply, actor_params, obs_seq: chex.Array, done_seq: chex.Array
) -> chex.Array:
    """Reconstruct a policy's actor hidden state AT S by replaying the prefix.

    A recurrent policy's memory when it acts at S = s_L is a deterministic
    function of the observation history o_0..o_{L-1}: it is the GRU carry after
    processing that history from a zero initial state. Replaying the SAME prefix
    through a given policy's actor therefore yields exactly the hidden state that
    policy would hold at S -- so pi_u and pi_u' each get their own faithful memory.

    obs_seq:  (num_traj, L, obs_dim)   observations o_0..o_{L-1}
    done_seq: (num_traj, L)            reset flags fed with each obs
    returns:  hidden state pytree with leading dim num_traj (h_L per trajectory)
    """
    def hstate_one(obs_s, done_s):
        obs_in = obs_s[:, jnp.newaxis, :]        # (L, 1, obs_dim)
        # For a non-recurrent actor, the hidden state is always zeroed — the
        # "hidden state at S" is just the zero carry regardless of history. We
        # still run the scan (to keep shapes consistent) but with all-True resets,
        # and strip the appended previous action (the reactive actor never sees it).
        if _actor_nonrecurrent:
            done_in = jnp.ones((obs_s.shape[0], 1), dtype=bool)
            obs_in = strip_action_features(obs_in, _action_feature_dim)
        else:
            done_in = done_s[:, jnp.newaxis]     # (L, 1)
        init_h = ScannedRNN(
            hidden_state_dim=_actor_hidden_dim, cell_type=_actor_cell_type
        ).initialize_carry(1)
        final_h, _ = actor_apply(actor_params, init_h, (obs_in, done_in))
        return final_h  # (1, H) -- the memory used to act at S

    return jax.vmap(hstate_one)(obs_seq, done_seq)


def compute_recurrent_value_at_S(
    critic_apply,
    critic_params,
    obs_seq: chex.Array,
    done_seq: chex.Array,
    final_obs: chex.Array,
    final_done: chex.Array,
) -> chex.Array:
    """V_hat^{u'}(tau): replay o_0..o_{L-1}, o_L through the recurrent critic from a
    zero hidden state and read the value AT the final state S = s_L.

    The final observation o_L (``final_obs``) is appended so the value is read at
    S itself -- the same state the Monte-Carlo ground truth branches from and the
    same obs the non-recurrent head uses. (Reading obs_seq[-1] alone would give
    the value of s_{L-1}, a different, off-by-one state.)

    obs_seq:   (num_traj, L, obs_dim)   observations o_0..o_{L-1}
    done_seq:  (num_traj, L)            reset flags for the prefix
    final_obs: (num_traj, obs_dim)      observation at S (o_L)
    final_done:(num_traj,)              reset flag at S (False for survived S)
    returns:   (num_traj,) recurrent value at S given the full history.
    """
    def value_one(obs_s, done_s, o_final, d_final):
        # Append o_L / done_L so the last scan step is exactly state S.
        obs_full = jnp.concatenate([obs_s, o_final[jnp.newaxis, :]], axis=0)  # (L+1, obs_dim)
        done_full = jnp.concatenate([done_s, d_final[jnp.newaxis]], axis=0)   # (L+1,)
        obs_in = obs_full[:, jnp.newaxis, :]     # (L+1, 1, obs_dim)
        done_in = done_full[:, jnp.newaxis]      # (L+1, 1)
        init_h = ScannedRNN(
            hidden_state_dim=_critic_hidden_dim, cell_type=_critic_cell_type
        ).initialize_carry(1)
        _, values = critic_apply(critic_params, init_h, (obs_in, done_in))
        # values: (L+1, 1) -> value at S is the last timestep (o_L).
        return values[-1, 0]

    return jax.vmap(value_one)(obs_seq, done_seq, final_obs, final_done)


def compute_nonrecurrent_value_at_S(
    nr_critic_apply, nr_critic_params, final_obs: chex.Array
) -> chex.Array:
    """V_hat_nr^{u'}(S): non-recurrent critic on the single state S (hidden zeroed).

    final_obs: (num_traj, obs_dim)
    returns:   (num_traj,)
    """

    def value_one(obs):
        # The memory-free critic is always Markov: strip the appended prev action.
        obs = strip_action_features(obs, _action_feature_dim)
        obs_in = obs[jnp.newaxis, jnp.newaxis, :]  # (1, 1, obs_dim)
        done_in = jnp.ones((1, 1), dtype=bool)     # reset -> zero hidden state
        init_h = ScannedRNN(
            hidden_state_dim=_critic_hidden_dim, cell_type=_critic_cell_type
        ).initialize_carry(1)
        _, value = nr_critic_apply(nr_critic_params, init_h, (obs_in, done_in))
        return value[0, 0]

    return jax.vmap(value_one)(final_obs)


def monte_carlo_value(
    env,
    actor_apply,
    actor_params,
    final_states,
    final_obs: chex.Array,
    init_hstates,
    mc_rollouts: int,
    max_horizon: int,
    gamma: float,
    key: chex.PRNGKey,
    desc: str,
):
    """Ground-truth V_pi(S) by branching the env at each S and rolling out pi.

    For each trajectory's final state S we run ``mc_rollouts`` continuations of
    up to ``max_horizon`` steps, accumulate gamma-discounted rewards (masking any
    reward earned after the episode ends), and average over rollouts.

    Each continuation starts the policy from ``init_hstates`` for that state --
    the ACTUAL hidden state pi would hold at S (reconstructed by replaying the
    prefix through pi's own actor; see ``replay_policy_hstate``). This makes the
    rollouts genuine samples of V_pi(S) for the recurrent policy, not an
    approximation that resets memory at the branch point.

    final_states: env states at S, leading dim num_traj.
    final_obs:    (num_traj, obs_dim) observations at S.
    init_hstates: pi's hidden state at S, leading dim num_traj (matches final_*).
    Returns: (mc_mean, mc_var), each (num_traj,). ``mc_var`` is the variance of
    the per-rollout returns for that state; ``mc_var / mc_rollouts`` is the
    squared standard error of the estimate, used downstream to debias the MSE.
    """
    num_traj = jax.tree_util.tree_leaves(final_states)[0].shape[0]

    @scan_tqdm(max_horizon, desc=desc)
    def horizon_step(carry, t):
        env_state, obs, hstate, alive, disc_return, discount, step_key = carry
        step_key, act_key = jax.random.split(step_key)
        batched_obs = obs[jnp.newaxis, jnp.newaxis, :]
        # Within a continuation we never reset the recurrent actor's hidden state
        # (single episode branch); the alive-mask handles termination. BUT if the
        # actor is non-recurrent, we always zero it (matching how it was trained).
        done_flag = (
            jnp.ones((1, 1), dtype=bool)
            if _actor_nonrecurrent
            else jnp.zeros((1, 1), dtype=bool)
        )
        # Reactive actor is Markov and never sees the appended previous action.
        actor_obs = (
            strip_action_features(batched_obs, _action_feature_dim)
            if _actor_nonrecurrent
            else batched_obs
        )
        new_hstate, pi = actor_apply(actor_params, hstate, (actor_obs, done_flag))
        # Drop leading (time=1, batch=1) dims; scalar (discrete) or vector (cont.).
        action = pi.sample(seed=act_key)[0, 0]
        new_env_state, ts = env.step(env_state, action)

        reward = ts.reward.reshape(())
        disc_return = disc_return + discount * reward * alive
        # Update alive AFTER counting this step's reward: once the episode has
        # ended (gymnax auto-resets), subsequent rewards are masked out. Keep
        # ``alive`` as a float so the scan carry dtype stays consistent.
        newly_done = ts.last().reshape(()).astype(jnp.float32)
        new_alive = alive * (1.0 - newly_done)
        new_discount = discount * gamma
        new_carry = (
            new_env_state,
            ts.observation,
            new_hstate,
            new_alive,
            disc_return,
            new_discount,
            step_key,
        )
        return new_carry, None

    def value_from_state(single_state, single_obs, single_hstate, rollout_key):
        def one_rollout(rk):
            init_carry = (
                single_state,
                single_obs,
                single_hstate,    # pi's true memory at S
                jnp.array(1.0),   # alive
                jnp.array(0.0),   # discounted return
                jnp.array(1.0),   # discount factor (gamma^0)
                rk,
            )
            final_carry, _ = jax.lax.scan(
                horizon_step, init_carry, jnp.arange(max_horizon)
            )
            return final_carry[4]  # discounted return

        rollout_keys = jax.random.split(rollout_key, mc_rollouts)
        returns = jax.vmap(one_rollout)(rollout_keys)
        # ddof=1 sample variance; guard the mc_rollouts==1 case.
        var = jnp.where(mc_rollouts > 1, returns.var(ddof=1), 0.0)
        return returns.mean(), var

    keys = jax.random.split(key, num_traj)
    # vmap over the num_traj branch points.
    mc_mean, mc_var = jax.vmap(value_from_state)(
        final_states, final_obs, init_hstates, keys
    )
    return mc_mean, mc_var


# Module-level values populated in run() so jitted closures can read the
# (static) hidden dims / cell types without threading them everywhere.
_critic_hidden_dim: int = 0
_critic_cell_type: str = "gru"
_actor_hidden_dim: int = 0
_actor_cell_type: str = "gru"
# When True, the actor is treated as non-recurrent (hidden state zeroed every
# step) during trajectory sampling and Monte-Carlo rollouts. Auto-set from the
# checkpoint metadata's system_name.
_actor_nonrecurrent: bool = False
# Number of trailing observation features that are the appended previous action
# (ActionConcatWrapper). The RECURRENT critic sees the full obs; the memory-free
# critic always strips these; the actor strips them iff it is non-recurrent.
# Auto-set from the env in precompute/score. 0 => no wrapper (all strips are no-ops).
_action_feature_dim: int = 0


def strip_action_features(observation: chex.Array, action_feature_dim: int) -> chex.Array:
    """Drop the trailing previous-action features from an observation (no-op if 0)."""
    if action_feature_dim <= 0:
        return observation
    return observation[..., :-action_feature_dim]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _cache_path(args: argparse.Namespace) -> str:
    """Path for the cached ground-truth (state set + Monte-Carlo values)."""
    if args.cache:
        p = args.cache
    else:
        p = (
            f"analysis/gt_cache_{args.run_uid}_u{args.ckpt_u}_up{args.ckpt_uprime}"
            f"_n{args.num_traj}_L{args.prefix_len}_mc{args.mc_rollouts}"
            f"_h{args.max_horizon}_seed{args.seed}.npz"
        )
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def precompute_ground_truth(args: argparse.Namespace) -> Dict[str, Any]:
    """EXPENSIVE phase: sample the fixed state set S ~ pi_u and Monte-Carlo the
    ground-truth values V_pi_u(S) and V_pi_u'(S). Everything here is independent
    of which value HEAD we later score, so we compute it once and cache it.

    The states S, their observations, the prefixes (needed to reconstruct value
    heads' inputs), and the MC values+variances are saved to a .npz so the cheap
    scoring phase -- and any re-analysis -- never re-pays the rollout cost.
    """
    global _critic_hidden_dim, _critic_cell_type, _actor_hidden_dim, _actor_cell_type, _actor_nonrecurrent, _action_feature_dim

    cfg = load_config(args.run_uid, overrides=tuple(args.overrides or ()), model_name=args.model_name)
    cfg.num_devices = 1
    cfg.arch.num_envs = 1

    print("Building environment...")
    _, eval_env = environments.make(cfg)

    num_actions, is_continuous, a_min, a_max = describe_action_space(eval_env)
    obs_dim = int(eval_env.observation_space().generate_value().shape[-1])
    cfg.system.obs_dim = obs_dim
    cfg.system.action_dim = num_actions
    print(f"obs_dim={obs_dim}  action_dim={num_actions}  "
          f"{'continuous' if is_continuous else 'discrete'}")

    actor, critic, nr_critic, actor_rnn = build_networks(
        cfg, num_actions, is_continuous, a_min, a_max
    )
    actor_apply, _, _ = make_apply_fns(actor, critic, nr_critic)

    _critic_hidden_dim = int(cfg.network.critic_network.rnn_layer.hidden_state_dim)
    _critic_cell_type = str(cfg.network.critic_network.rnn_layer.cell_type)
    _actor_hidden_dim = int(cfg.network.actor_network.rnn_layer.hidden_state_dim)
    _actor_cell_type = str(cfg.network.actor_network.rnn_layer.cell_type)
    # Detect whether the actor was trained as non-recurrent (hidden zeroed every step).
    meta = read_checkpoint_metadata(args.run_uid, model_name=args.model_name)
    system_name = meta.get("system", {}).get("system_name", "")
    _actor_nonrecurrent = "nonrec_actor" in system_name
    if _actor_nonrecurrent:
        print("Detected non-recurrent actor (policy is reactive/Markov).")
    # How many trailing obs features are the appended previous action (0 if none).
    _action_feature_dim = int(getattr(eval_env, "action_feature_dim", 0))
    if _action_feature_dim:
        print(f"Detected ActionConcatWrapper: {_action_feature_dim} appended action features.")

    if args.gamma is not None:
        cfg.system.gamma = args.gamma
    gamma = float(cfg.system.gamma)

    dummy_params = init_dummy_params(cfg, actor, critic, nr_critic)
    print(f"Restoring u  (ckpt step {args.ckpt_u}) ...")
    params_u = restore_checkpoint(args.run_uid, args.ckpt_u, dummy_params, model_name=args.model_name)
    print(f"Restoring u' (ckpt step {args.ckpt_uprime}) ...")
    params_uprime = restore_checkpoint(args.run_uid, args.ckpt_uprime, dummy_params, model_name=args.model_name)

    key = jax.random.PRNGKey(args.seed)
    key, traj_key = jax.random.split(key)

    # 1) Sample trajectories from pi_u and get S (final state of the prefix).
    print(f"Sampling {args.num_traj} trajectories from pi_u (prefix_len={args.prefix_len}) ...")
    sample_fn = jax.jit(
        lambda k: sample_prefix_trajectories(
            eval_env, actor_apply, actor_rnn, params_u, k,
            args.num_traj, args.prefix_len, obs_dim,
        )
    )
    traj = sample_fn(traj_key)
    jax.block_until_ready(traj)

    survived = traj["survived"]
    n_survived = int(jnp.sum(survived))
    print(f"{n_survived}/{args.num_traj} trajectories survived the full prefix "
          f"(S is non-terminal for these).")
    if n_survived == 0:
        raise SystemExit("No trajectories survived the prefix; lower --prefix-len.")

    # Keep only survived trajectories so S is a genuine non-terminal state.
    keep = jnp.where(survived)[0]
    obs_seq = traj["obs_seq"][keep]
    done_seq = traj["done_seq"][keep]
    final_obs = traj["final_obs"][keep]
    final_done = jnp.zeros((keep.shape[0],), dtype=bool)  # survived => S non-terminal
    final_states = jax.tree_util.tree_map(lambda x: x[keep], traj["final_state"])

    # 2) Reconstruct EACH policy's true hidden state at S by replaying the prefix
    #    through that policy's own actor.
    print("Reconstructing faithful policy hidden states at S ...")
    h_u = jax.jit(lambda: replay_policy_hstate(
        actor_apply, params_u.actor_params, obs_seq, done_seq))()
    h_uprime = jax.jit(lambda: replay_policy_hstate(
        actor_apply, params_uprime.actor_params, obs_seq, done_seq))()
    jax.block_until_ready((h_u, h_uprime))

    # 3) Ground-truth Monte-Carlo values under pi_u and pi_u', each starting from
    #    that policy's faithful hidden state at S.
    key, mc_u_key, mc_uprime_key = jax.random.split(key, 3)
    print(f"Monte-Carlo V_pi_u(S)  ({args.mc_rollouts} rollouts x {args.max_horizon} steps)...")
    v_gt_u, var_u = jax.jit(lambda k: monte_carlo_value(
        eval_env, actor_apply, params_u.actor_params,
        final_states, final_obs, h_u,
        args.mc_rollouts, args.max_horizon, gamma, k, desc="MC pi_u",
    ))(mc_u_key)
    jax.block_until_ready((v_gt_u, var_u))

    print(f"Monte-Carlo V_pi_u'(S) ({args.mc_rollouts} rollouts x {args.max_horizon} steps)...")
    v_gt_uprime, var_uprime = jax.jit(lambda k: monte_carlo_value(
        eval_env, actor_apply, params_uprime.actor_params,
        final_states, final_obs, h_uprime,
        args.mc_rollouts, args.max_horizon, gamma, k, desc="MC pi_u'",
    ))(mc_uprime_key)
    jax.block_until_ready((v_gt_uprime, var_uprime))

    gt = {
        "obs_seq": np.asarray(obs_seq),
        "done_seq": np.asarray(done_seq),
        "final_obs": np.asarray(final_obs),
        "final_done": np.asarray(final_done),
        "v_gt_u": np.asarray(v_gt_u),
        "v_gt_uprime": np.asarray(v_gt_uprime),
        "var_u": np.asarray(var_u),
        "var_uprime": np.asarray(var_uprime),
        # meta (stored as 0-d arrays so np.savez round-trips cleanly)
        "run_uid": np.asarray(args.run_uid),
        "ckpt_u": np.asarray(args.ckpt_u),
        "ckpt_uprime": np.asarray(args.ckpt_uprime),
        "num_traj_used": np.asarray(n_survived),
        "num_traj_requested": np.asarray(args.num_traj),
        "prefix_len": np.asarray(args.prefix_len),
        "mc_rollouts": np.asarray(args.mc_rollouts),
        "max_horizon": np.asarray(args.max_horizon),
        "gamma": np.asarray(gamma),
        "seed": np.asarray(args.seed),
    }

    cache_path = _cache_path(args)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, **gt)
    print(f"Cached ground truth to {cache_path}")
    return gt


def score_from_ground_truth(args: argparse.Namespace, gt: Dict[str, Any]) -> Dict[str, Any]:
    """CHEAP phase: given the cached state set + MC ground truth, run the two
    value heads of u' and compute the four MSEs. No environment rollouts here.

    We also report an MC-noise-corrected MSE. The naive E[(V_hat - V_mc)^2]
    over-counts by the mean squared standard error of the MC estimate, since
    V_mc = V_true + noise. Subtracting E[var/mc_rollouts] gives an unbiased
    estimate of E[(V_hat - V_true)^2] -- which is what lets us trust results
    from a modest number of rollouts.
    """
    global _critic_hidden_dim, _critic_cell_type, _actor_hidden_dim, _actor_cell_type, _actor_nonrecurrent, _action_feature_dim

    cfg = load_config(args.run_uid, overrides=tuple(args.overrides or ()), model_name=args.model_name)
    cfg.num_devices = 1
    cfg.arch.num_envs = 1
    _, eval_env = environments.make(cfg)
    num_actions, is_continuous, a_min, a_max = describe_action_space(eval_env)
    obs_dim = int(eval_env.observation_space().generate_value().shape[-1])
    cfg.system.obs_dim = obs_dim
    cfg.system.action_dim = num_actions

    actor, critic, nr_critic, actor_rnn = build_networks(
        cfg, num_actions, is_continuous, a_min, a_max
    )
    _, critic_apply, nr_critic_apply = make_apply_fns(actor, critic, nr_critic)
    _critic_hidden_dim = int(cfg.network.critic_network.rnn_layer.hidden_state_dim)
    _critic_cell_type = str(cfg.network.critic_network.rnn_layer.cell_type)
    _actor_hidden_dim = int(cfg.network.actor_network.rnn_layer.hidden_state_dim)
    _actor_cell_type = str(cfg.network.actor_network.rnn_layer.cell_type)
    meta = read_checkpoint_metadata(args.run_uid, model_name=args.model_name)
    system_name = meta.get("system", {}).get("system_name", "")
    _actor_nonrecurrent = "nonrec_actor" in system_name
    _action_feature_dim = int(getattr(eval_env, "action_feature_dim", 0))

    dummy_params = init_dummy_params(cfg, actor, critic, nr_critic)
    ckpt_uprime = int(gt["ckpt_uprime"])
    print(f"Restoring u' (ckpt step {ckpt_uprime}) for value heads ...")
    params_uprime = restore_checkpoint(args.run_uid, ckpt_uprime, dummy_params, model_name=args.model_name)

    obs_seq = jnp.asarray(gt["obs_seq"])
    done_seq = jnp.asarray(gt["done_seq"])
    final_obs = jnp.asarray(gt["final_obs"])
    final_done = jnp.asarray(gt["final_done"])
    v_gt_u = jnp.asarray(gt["v_gt_u"])
    v_gt_uprime = jnp.asarray(gt["v_gt_uprime"])
    var_u = jnp.asarray(gt["var_u"])
    var_uprime = jnp.asarray(gt["var_uprime"])
    mc_rollouts = int(gt["mc_rollouts"])

    # Value-head predictions at S (evaluated at S = s_L, matching the MC branch).
    print("Computing recurrent value V_hat^{u'}(tau) at S ...")
    v_rec = jax.jit(lambda: compute_recurrent_value_at_S(
        critic_apply, params_uprime.critic_params, obs_seq, done_seq, final_obs, final_done
    ))()
    print("Computing non-recurrent value V_hat_nr^{u'}(S) ...")
    v_nr = jax.jit(lambda: compute_nonrecurrent_value_at_S(
        nr_critic_apply, params_uprime.nr_critic_params, final_obs
    ))()
    jax.block_until_ready((v_rec, v_nr))

    def mse(a, b):
        return float(jnp.mean((a - b) ** 2))

    # Mean squared standard error of each MC ground-truth estimate.
    sem2_u = float(jnp.mean(var_u / mc_rollouts))
    sem2_uprime = float(jnp.mean(var_uprime / mc_rollouts))

    def corrected(raw_mse, sem2):
        # Unbiased estimate of E[(V_hat - V_true)^2]; clamp at 0 for readability.
        return max(raw_mse - sem2, 0.0)

    rec_old, rec_new = mse(v_rec, v_gt_u), mse(v_rec, v_gt_uprime)
    nr_old, nr_new = mse(v_nr, v_gt_u), mse(v_nr, v_gt_uprime)

    results = {
        "meta": {
            "run_uid": str(gt["run_uid"]),
            "ckpt_u": int(gt["ckpt_u"]),
            "ckpt_uprime": int(gt["ckpt_uprime"]),
            "num_traj_requested": int(gt["num_traj_requested"]),
            "num_traj_used": int(gt["num_traj_used"]),
            "prefix_len": int(gt["prefix_len"]),
            "mc_rollouts": mc_rollouts,
            "max_horizon": int(gt["max_horizon"]),
            "gamma": float(gt["gamma"]),
            "seed": int(gt["seed"]),
            "mc_sem2_pi_u": sem2_u,
            "mc_sem2_pi_uprime": sem2_uprime,
        },
        "recurrent": {
            "mse_vs_old_policy_V_pi_u": rec_old,
            "mse_vs_new_policy_V_pi_uprime": rec_new,
            "mse_vs_old_policy_V_pi_u_mc_corrected": corrected(rec_old, sem2_u),
            "mse_vs_new_policy_V_pi_uprime_mc_corrected": corrected(rec_new, sem2_uprime),
        },
        "non_recurrent": {
            "mse_vs_old_policy_V_pi_u": nr_old,
            "mse_vs_new_policy_V_pi_uprime": nr_new,
            "mse_vs_old_policy_V_pi_u_mc_corrected": corrected(nr_old, sem2_u),
            "mse_vs_new_policy_V_pi_uprime_mc_corrected": corrected(nr_new, sem2_uprime),
        },
        "diagnostics": {
            "mean_V_hat_recurrent": float(jnp.mean(v_rec)),
            "mean_V_hat_nonrecurrent": float(jnp.mean(v_nr)),
            "mean_V_gt_pi_u": float(jnp.mean(v_gt_u)),
            "mean_V_gt_pi_uprime": float(jnp.mean(v_gt_uprime)),
        },
    }
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-uid", default=None,
                   help="Checkpoint run folder under checkpoints/<model-name>/. "
                        "Defaults to the single run present if unambiguous.")
    p.add_argument("--model-name", default="rec_ppo_dual_value",
                   help="Model/system name (checkpoint subdirectory). "
                        "Use 'rec_ppo_nonrec_actor' for the non-recurrent-actor variant.")
    p.add_argument("--ckpt-u", type=int, default=29184,
                   help="Timestep of checkpoint u (early policy). Default: ckpt 3 = 29184.")
    p.add_argument("--ckpt-uprime", type=int, default=437760,
                   help="Timestep of checkpoint u' (late policy). Default: ckpt 45 = 437760.")
    p.add_argument("--num-traj", type=int, default=256)
    p.add_argument("--prefix-len", type=int, default=64,
                   help="Length L of each trajectory prefix; S is the state after L steps.")
    p.add_argument("--mc-rollouts", type=int, default=64,
                   help="Monte-Carlo continuations per state per policy.")
    p.add_argument("--max-horizon", type=int, default=300,
                   help="Max steps per Monte-Carlo continuation. With gamma=0.99 this "
                        "captures ~95%% of the discounted mass (0.99^300 ~= 0.05); "
                        "CartPole caps episodes at 500.")
    p.add_argument("--gamma", type=float, default=None,
                   help="Discount; defaults to the training gamma from config.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="analysis/value_drift_results.json")
    p.add_argument("--cache", default=None,
                   help="Path to the ground-truth .npz cache. Default: auto-named "
                        "from the run/checkpoint/MC settings under analysis/.")
    p.add_argument("--from-cache", action="store_true",
                   help="Skip the expensive Monte-Carlo phase and score directly "
                        "from an existing ground-truth cache. Errors if absent.")
    p.add_argument("--recompute", action="store_true",
                   help="Force recomputation of the ground truth even if a cache exists.")
    p.add_argument("--overrides", nargs="*", default=[],
                   help="Extra Hydra overrides, e.g. network.actor_network.rnn_layer.hidden_state_dim=32")
    args = p.parse_args()

    # Resolve the run uid if not given.
    if args.run_uid is None:
        ckpt_root = os.path.join(REPO_ROOT, "checkpoints", args.model_name)
        if not os.path.isdir(ckpt_root):
            raise SystemExit(
                f"No checkpoint directory at {ckpt_root}. "
                f"Check --model-name (currently '{args.model_name}')."
            )
        runs = sorted(d for d in os.listdir(ckpt_root)
                      if os.path.isdir(os.path.join(ckpt_root, d)))
        if len(runs) != 1:
            raise SystemExit(
                f"Found {len(runs)} runs in {ckpt_root}: {runs}. "
                f"Pass --run-uid to disambiguate."
            )
        args.run_uid = runs[0]
        print(f"Using run_uid={args.run_uid}")

    # --- Ground truth: reuse cache if possible, else Monte-Carlo it once. ---
    cache_path = _cache_path(args)
    if args.from_cache or (os.path.exists(cache_path) and not args.recompute):
        if not os.path.exists(cache_path):
            raise SystemExit(f"--from-cache set but no cache at {cache_path}")
        print(f"Loading cached ground truth from {cache_path}")
        gt = dict(np.load(cache_path, allow_pickle=True))
    else:
        gt = precompute_ground_truth(args)

    # --- Cheap scoring phase (no rollouts). ---
    results = score_from_ground_truth(args, gt)

    m = results["meta"]
    print("\n" + "=" * 74)
    print("VALUE DRIFT RESULTS  (mean-squared error)")
    print("=" * 74)
    print(f"  u  = ckpt step {m['ckpt_u']}   u' = ckpt step {m['ckpt_uprime']}")
    print(f"  trajectories used: {m['num_traj_used']} / {m['num_traj_requested']}"
          f"   | MC: {m['mc_rollouts']} rollouts x {m['max_horizon']} steps")
    print("-" * 74)
    col_old = "vs OLD policy V_pi_u"
    col_new = "vs NEW policy V_pi_u'"
    print(f"{'':<30}{col_old:>22}{col_new:>22}")
    r, nr = results["recurrent"], results["non_recurrent"]
    print(f"{'Recurrent  V_hat(tau)':<30}"
          f"{r['mse_vs_old_policy_V_pi_u']:>22.4f}"
          f"{r['mse_vs_new_policy_V_pi_uprime']:>22.4f}")
    print(f"{'Non-recur. V_hat_nr(S)':<30}"
          f"{nr['mse_vs_old_policy_V_pi_u']:>22.4f}"
          f"{nr['mse_vs_new_policy_V_pi_uprime']:>22.4f}")
    print("-" * 74)
    print("  MC-noise-corrected (unbiased estimate of MSE vs TRUE value):")
    print(f"{'Recurrent  V_hat(tau)':<30}"
          f"{r['mse_vs_old_policy_V_pi_u_mc_corrected']:>22.4f}"
          f"{r['mse_vs_new_policy_V_pi_uprime_mc_corrected']:>22.4f}")
    print(f"{'Non-recur. V_hat_nr(S)':<30}"
          f"{nr['mse_vs_old_policy_V_pi_u_mc_corrected']:>22.4f}"
          f"{nr['mse_vs_new_policy_V_pi_uprime_mc_corrected']:>22.4f}")
    print("=" * 74)
    print(f"  MC standard-error^2:  pi_u={m['mc_sem2_pi_u']:.4f}  "
          f"pi_u'={m['mc_sem2_pi_uprime']:.4f}  (subtracted above)")
    print("Diagnostics (means):")
    for k, v in results["diagnostics"].items():
        print(f"  {k:<28} {v:>10.3f}")

    out_path = os.path.join(REPO_ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
