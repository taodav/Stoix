"""Recurrent Direct Backprop — GRU policy, with optional memory.

This is the recurrent counterpart to ff_direct_backprop.py. The policy is a
GRU-based recurrent network, and gradients flow both through the differentiable
environment dynamics AND through the recurrent hidden state (backprop through time).

A single config flag, `system.reset_hidden_state_every_step`, controls whether the
GRU hidden state persists across steps:
    - False -> fully recurrent policy (memory persists across the rollout).
    - True  -> the hidden state is zeroed at every step, making the policy
               effectively memoryless (a feed-forward function of the current
               observation) while keeping the EXACT SAME architecture and
               parameter count as the recurrent variant.

This enables a clean, controlled comparison: same GRU network, same number of
parameters, toggling only whether temporal memory is available to the analytic
policy gradient.

The critic is kept feed-forward in both variants so that the ablation isolates
the effect of the actor's memory.
"""

import copy
import time
from typing import Any, NamedTuple, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from stoa import Environment, TimeStep, WrapperState
from stoa.env_types import StepType

from stoix.base_types import (
    ActorCriticOptStates,
    ActorCriticParams,
    AnakinExperimentOutput,
    LearnerFn,
)
from stoix.evaluator import evaluator_setup, get_rec_distribution_act_fn
from stoix.networks.base import CompositeNetwork
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.networks.base import ScannedRNN
from stoix.networks.inputs import ArrayInput
from stoix.networks.postprocessors import tanh_to_spec
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate


class MemoryToggleRecurrentActor(nn.Module):
    """Recurrent actor for Direct Backprop with optionally-disabled memory.

    Behaves like `stoix.networks.base.RecurrentActor`, but when
    `reset_hidden_state_every_step` is True the GRU hidden state is reset to
    zeros at every timestep. This makes the policy memoryless while retaining
    the identical architecture and parameter count as the recurrent variant.
    """

    action_head: nn.Module
    post_torso: nn.Module
    hidden_state_dim: int
    cell_type: str
    pre_torso: nn.Module
    reset_hidden_state_every_step: bool = False
    input_layer: nn.Module = ArrayInput()

    @nn.compact
    def __call__(
        self,
        policy_hidden_state: chex.Array,
        observation_done: Tuple[chex.Array, chex.Array],
    ) -> Tuple[chex.Array, Any]:
        observation, done = observation_done

        # Force a reset at every step to disable memory (identical architecture).
        if self.reset_hidden_state_every_step:
            done = jnp.ones_like(done)

        observation = self.input_layer(observation)
        policy_embedding = self.pre_torso(observation)
        policy_rnn_input = (policy_embedding, done)
        policy_hidden_state, policy_embedding = ScannedRNN(
            self.hidden_state_dim, self.cell_type
        )(policy_hidden_state, policy_rnn_input)
        actor_logits = self.post_torso(policy_embedding)
        pi = self.action_head(actor_logits)

        return policy_hidden_state, pi


def strip_action_features(observation: chex.Array, action_feature_dim: int) -> chex.Array:
    """Drop the trailing previous-action features from an observation.

    The ActionConcatWrapper appends the previous action to every observation so a
    RECURRENT actor (which processes trajectories) can accumulate an action
    history. In Direct Backprop the actor is the trajectory-processing network, so
    the previous action is only appropriate for the RECURRENT variant
    (``reset_hidden_state_every_step=False``). The memory-free actor variant and
    the feed-forward terminal-value critic must stay Markov and never see it.
    Removes the last ``action_feature_dim`` features; a dim of 0 is a no-op.
    """
    if action_feature_dim <= 0:
        return observation
    return observation[..., :-action_feature_dim]


class RecDBLearnerState(NamedTuple):
    """Learner state for recurrent Direct Backprop.

    Carries the actor's policy hidden state and the last done flag through the
    training loop so memory persists across consecutive rollouts.
    """

    params: ActorCriticParams
    opt_states: ActorCriticOptStates
    key: chex.PRNGKey
    env_state: WrapperState
    timestep: TimeStep
    done: chex.Array
    hstate: chex.Array


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[Any, Any],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[RecDBLearnerState]:
    """Get the learner function.

    Rolls out the recurrent policy for rollout_length steps through the
    differentiable environment and backpropagates the negative return through
    both the dynamics and the recurrent hidden state.
    """

    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(
        learner_state: RecDBLearnerState, _: Any
    ) -> Tuple[RecDBLearnerState, Tuple]:

        params, opt_states, key, env_state, last_timestep, last_done, hstate = learner_state

        def _differentiable_rollout_loss(
            actor_params: FrozenDict,
            critic_params: FrozenDict,
            env_state: Any,
            observation: chex.Array,
            done: chex.Array,
            hstate: chex.Array,
            key: chex.PRNGKey,
        ) -> Tuple[chex.Array, Tuple]:
            """The H-step recurrent rollout IS the loss computation.

            env.step is called inside this function, so gradients flow through
            the dynamics; the hidden state is threaded through the scan carry,
            so gradients also flow through time (for the recurrent variant).

            Shapes: observation (num_envs, obs_dim), done (num_envs,),
            hstate (num_envs, hidden_dim).
            """
            num_envs = observation.shape[0]

            def _scan_step(carry: Tuple, _: Any) -> Tuple[Tuple, chex.Array]:
                state, obs, done, cumulative_discount, hstate, step_key = carry

                # Add a time dimension for the ScannedRNN (time=1). Strip the
                # appended previous action for the memory-free actor variant only.
                actor_obs = strip_action_features(obs, config.system.actor_strip_dim)
                batched_obs = actor_obs[jnp.newaxis]  # (1, num_envs, obs_dim)
                ac_in = (batched_obs, done[jnp.newaxis])  # done: (1, num_envs)

                new_hstate, pi = actor_apply_fn(actor_params, hstate, ac_in)
                action = pi.mode()[0]  # remove time dim -> (num_envs, action_dim)

                # Exploration noise via reparameterization trick.
                step_key, noise_key = jax.random.split(step_key)
                noise = (
                    jax.random.normal(noise_key, action.shape)
                    * config.system.exploration_noise
                )
                action = action + noise
                action = jnp.clip(
                    action, config.system.action_minimum, config.system.action_maximum
                )

                # DIFFERENTIABLE environment step.
                state, timestep = env.step(state, action)

                new_done = timestep.discount == 0.0  # (num_envs,)
                alive = 1.0 - new_done.astype(jnp.float32)
                reward = (
                    timestep.reward * config.system.reward_scale * cumulative_discount
                )
                new_cumulative_discount = cumulative_discount * config.system.gamma * alive

                new_carry = (
                    state,
                    timestep.observation,
                    new_done,
                    new_cumulative_discount,
                    new_hstate,
                    step_key,
                )
                return new_carry, reward

            init_carry = (
                env_state,
                observation,
                done,
                jnp.ones(num_envs, dtype=jnp.float32),
                hstate,
                key,
            )

            (
                final_state,
                final_obs,
                final_done,
                final_discount,
                final_hstate,
                _,
            ), rewards = jax.lax.scan(
                _scan_step, init_carry, None, config.system.rollout_length
            )

            # Sum discounted rewards over time (num_envs,).
            total_return = jnp.sum(rewards, axis=0)

            # Optional (SHAC-style) terminal value bootstrap. Base Direct Backprop
            # skips this entirely (critic_params is None). Critic is feed-forward.
            if config.system.use_terminal_value:
                # Feed-forward critic is Markov: strip the appended previous action.
                critic_final_obs = strip_action_features(
                    final_obs, config.system.action_feature_dim
                )
                terminal_value = critic_apply_fn(critic_params, critic_final_obs)
                terminal_value = terminal_value * final_discount
                total_return = total_return + terminal_value

            loss = -jnp.mean(total_return)
            aux = (final_state, final_obs, final_done, final_hstate, jnp.mean(total_return))
            return loss, aux

        # ---- Actor gradients through dynamics (+ through time if recurrent) ----
        key, rollout_key = jax.random.split(key)
        actor_grad_fn = jax.grad(_differentiable_rollout_loss, argnums=0, has_aux=True)
        actor_grads, (
            new_env_state,
            final_obs,
            final_done,
            final_hstate,
            total_return,
        ) = actor_grad_fn(
            params.actor_params,
            params.critic_params,
            env_state,
            last_timestep.observation,
            last_done,
            hstate,
            rollout_key,
        )

        actor_grads = jax.lax.pmean(actor_grads, axis_name="batch")
        actor_grads = jax.lax.pmean(actor_grads, axis_name="device")

        actor_updates, actor_new_opt_state = actor_update_fn(
            actor_grads, opt_states.actor_opt_state
        )
        actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

        # ---- Optional critic update (only for the SHAC-style terminal value) ----
        # When use_terminal_value is False there is no critic — pure Direct Backprop.
        if config.system.use_terminal_value:

            def _critic_loss_fn(critic_params: FrozenDict) -> Tuple[chex.Array, dict]:
                # Feed-forward critic is Markov: strip the appended previous action.
                critic_obs = strip_action_features(
                    last_timestep.observation, config.system.action_feature_dim
                )
                predicted_values = critic_apply_fn(critic_params, critic_obs)
                target = jax.lax.stop_gradient(total_return)
                value_loss = jnp.mean(0.5 * (predicted_values - target) ** 2)
                return value_loss, {
                    "value_loss": value_loss,
                    "pred_value": jnp.mean(predicted_values),
                }

            critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
            critic_grads, critic_loss_info = critic_grad_fn(params.critic_params)
            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="batch"
            )
            critic_grads, critic_loss_info = jax.lax.pmean(
                (critic_grads, critic_loss_info), axis_name="device"
            )
            critic_updates, critic_new_opt_state = critic_update_fn(
                critic_grads, opt_states.critic_opt_state
            )
            critic_new_params = optax.apply_updates(params.critic_params, critic_updates)
        else:
            critic_new_params = None
            critic_new_opt_state = None
            critic_loss_info = {}

        new_params = ActorCriticParams(actor_new_params, critic_new_params)
        new_opt_states = ActorCriticOptStates(actor_new_opt_state, critic_new_opt_state)

        # Advance environment: reconstruct a mid-episode timestep for the next rollout.
        num_envs = final_obs.shape[0]
        new_timestep = last_timestep.replace(
            observation=final_obs,
            step_type=jnp.full(num_envs, StepType.MID, dtype=jnp.int8),
            reward=jnp.zeros(num_envs, dtype=jnp.float32),
            discount=jnp.ones(num_envs, dtype=jnp.float32),
        )

        learner_state = RecDBLearnerState(
            new_params,
            new_opt_states,
            key,
            new_env_state,
            new_timestep,
            final_done,
            final_hstate,
        )

        actor_grad_norm = optax.global_norm(actor_grads)
        loss_info = {
            "actor_loss": -total_return,
            "mean_return": total_return,
            "grad_norm": actor_grad_norm,
            **critic_loss_info,
        }
        episode_info = {
            "episode_return": total_return,
            "episode_length": jnp.array(
                config.system.rollout_length, dtype=jnp.float32
            ),
            "is_terminal_step": jnp.array(False),
        }

        return learner_state, (episode_info, loss_info)

    def learner_fn(
        learner_state: RecDBLearnerState,
    ) -> AnakinExperimentOutput[RecDBLearnerState]:
        batched_update_step = jax.vmap(
            _update_step, in_axes=(0, None), axis_name="batch"
        )
        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval
        )
        return AnakinExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[RecDBLearnerState], MemoryToggleRecurrentActor, ScannedRNN, RecDBLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    n_devices = len(jax.devices())

    action_dim = int(env.action_space().shape[-1])
    config.system.action_dim = action_dim
    config.system.action_minimum = float(env.action_space().minimum)
    config.system.action_maximum = float(env.action_space().maximum)

    # Previous-action concatenation (ActionConcatWrapper). The RECURRENT actor
    # sees the appended action; the memory-free actor variant and the feed-forward
    # critic stay Markov and strip it.
    action_feature_dim = int(getattr(env, "action_feature_dim", 0))
    config.system.action_feature_dim = action_feature_dim
    # How many trailing features to STRIP from the actor input: none when the actor
    # is recurrent (it keeps the appended previous action), all of them when the
    # actor is memory-free (Markov, must not see the action). The feed-forward
    # critic always strips `action_feature_dim`.
    actor_strip_dim = (
        action_feature_dim if config.system.reset_hidden_state_every_step else 0
    )
    config.system.actor_strip_dim = actor_strip_dim

    key, actor_net_key, critic_net_key = keys

    # ---- Actor network (recurrent, memory optionally disabled) ----
    actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=action_dim
    )
    action_head_post_processor = hydra.utils.instantiate(
        config.network.actor_network.post_processor,
        minimum=config.system.action_minimum,
        maximum=config.system.action_maximum,
        scale_fn=tanh_to_spec,
    )
    actor_action_head = CompositeNetwork([actor_action_head, action_head_post_processor])

    actor_network = MemoryToggleRecurrentActor(
        pre_torso=actor_pre_torso,
        hidden_state_dim=config.network.actor_network.rnn_layer.hidden_state_dim,
        cell_type=config.network.actor_network.rnn_layer.cell_type,
        post_torso=actor_post_torso,
        action_head=actor_action_head,
        reset_hidden_state_every_step=config.system.reset_hidden_state_every_step,
    )
    actor_rnn = ScannedRNN(
        hidden_state_dim=config.network.actor_network.rnn_layer.hidden_state_dim,
        cell_type=config.network.actor_network.rnn_layer.cell_type,
    )

    # ---- Actor optimiser ----
    actor_lr = make_learning_rate(config.system.actor_lr, config, num_epochs=1)
    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )

    # ---- Initialise actor params ----
    # Actor needs (init_hstate, (init_obs, init_done)) with a time dimension.
    init_obs = env.observation_space().generate_value()
    init_obs = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0), init_obs
    )
    init_obs_time = jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], init_obs)
    init_done = jnp.zeros((1, config.arch.num_envs), dtype=bool)
    # Actor init obs: strip the appended action only if the actor is memory-free.
    init_actor_obs = strip_action_features(init_obs_time, actor_strip_dim)
    init_actor_x = (init_actor_obs, init_done)
    init_policy_hstate = actor_rnn.initialize_carry(config.arch.num_envs)

    actor_params = actor_network.init(actor_net_key, init_policy_hstate, init_actor_x)
    actor_opt_state = actor_optim.init(actor_params)

    # ---- Critic (feed-forward), only for the optional SHAC-style terminal value ----
    if config.system.use_terminal_value:
        critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
        critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)
        critic_network = Critic(torso=critic_torso, critic_head=critic_head)
        critic_lr = make_learning_rate(config.system.critic_lr, config, num_epochs=1)
        critic_optim = optax.chain(
            optax.clip_by_global_norm(config.system.max_grad_norm),
            optax.adam(critic_lr, eps=1e-5),
        )
        # Critic is feed-forward and Markov: strip the appended previous action.
        critic_params = critic_network.init(
            critic_net_key, strip_action_features(init_obs, action_feature_dim)
        )
        critic_opt_state = critic_optim.init(critic_params)
        critic_apply = critic_network.apply
        critic_update = critic_optim.update
    else:
        critic_params = None
        critic_opt_state = None
        critic_apply = None
        critic_update = None

    params = ActorCriticParams(actor_params, critic_params)

    apply_fns = (actor_network.apply, critic_apply)
    update_fns = (actor_optim.update, critic_update)

    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # ---- Initialise environment states ----
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = env.reset(jnp.stack(env_keys))
    reshape_states = lambda x: x.reshape(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # Load checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        params = restored_params

    # ---- Replicate learner state ----
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape(
        (n_devices, config.arch.update_batch_size) + x.shape[1:]
    )
    step_keys = reshape_keys(jnp.stack(step_keys))

    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)
    dones = jnp.zeros((config.arch.num_envs,), dtype=bool)

    replicate_learner = (params, opt_states, init_policy_hstate, dones)
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    params, opt_states, hstate, dones = replicate_learner
    init_learner_state = RecDBLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        done=dones,
        hstate=hstate,
    )

    return learn, actor_network, actor_rnn, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    env, eval_env = environments.make(config=config)

    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=4
    )

    learn, actor_network, actor_rnn, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Recurrent evaluator (the custom actor handles the reset-every-step logic
    # internally, so the same evaluator works for both variants). The shared
    # evaluator feeds the FULL env observation; strip the appended previous action
    # for the memory-free actor variant (actor_strip_dim>0), matching training.
    # A no-op for the recurrent variant and when no ActionConcatWrapper is used.
    eval_actor_strip_dim = int(config.system.get("actor_strip_dim", 0))

    def eval_actor_apply(params: FrozenDict, hstate: chex.Array, obs_and_done: Tuple) -> Tuple:
        observation, done = obs_and_done
        observation = strip_action_features(observation, eval_actor_strip_dim)
        return actor_network.apply(params, hstate, (observation, done))

    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        eval_act_fn=get_rec_distribution_act_fn(config, eval_actor_apply),
        params=learner_state.params.actor_params,
        config=config,
        use_recurrent_net=True,
        scanned_rnn=actor_rnn,
    )

    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    logger = StoixLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    print(f"{Fore.YELLOW}{Style.BRIGHT}JAX Global Devices {jax.devices()}{Style.RESET_ALL}")
    memory_status = (
        "MEMORYLESS (hidden state zeroed every step)"
        if config.system.reset_hidden_state_every_step
        else "RECURRENT (hidden state persists)"
    )
    print(f"{Fore.YELLOW}{Style.BRIGHT}Policy mode: {memory_status}{Style.RESET_ALL}")

    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,
        )

    max_episode_return = -jnp.inf
    best_learner_state = unreplicate_batch_dim(learner_state)
    for eval_step in range(config.arch.num_evaluation):
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        # Report mean rollout return as the training "episode" metric.
        train_episode_metrics = {
            "episode_return": float(
                jnp.mean(learner_output.episode_metrics["episode_return"])
            ),
        }
        train_episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        logger.log(train_episode_metrics, t, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        opt_steps_per_eval = config.arch.num_updates_per_eval
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        start_time = time.time()
        trained_params = unreplicate_batch_dim(
            learner_output.learner_state.params.actor_params
        )
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        evaluator_output = evaluator(trained_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])
        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = (
            steps_per_eval / elapsed_time
        )
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            checkpointer.save(
                timestep=int(steps_per_rollout * (eval_step + 1)),
                unreplicated_learner_state=unreplicate_n_dims(
                    learner_output.learner_state
                ),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_learner_state = copy.deepcopy(unreplicate_batch_dim(learner_state))
            max_episode_return = episode_return

        learner_state = learner_output.learner_state

    if config.arch.absolute_metric:
        start_time = time.time()

        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        best_params = best_learner_state.params.actor_params
        evaluator_output = absolute_metric_evaluator(best_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = (
            steps_per_eval / elapsed_time
        )
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    logger.stop()
    eval_performance = float(
        jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric])
    )
    return eval_performance


@hydra.main(
    config_path="../../configs/default/anakin",
    config_name="default_rec_direct_backprop.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    t0 = time.time()
    eval_performance = run_experiment(cfg)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Recurrent Direct Backprop experiment completed in "
        f"{time.time() - t0:.2f} seconds.{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
