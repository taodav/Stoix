"""Direct Backprop — Analytic Policy Gradient through Differentiable Dynamics.

This algorithm directly backpropagates the sum of discounted rewards through
the (differentiable) environment dynamics into the policy parameters, yielding
exact policy gradients rather than estimates (as in REINFORCE/PPO).

Requirements:
    - The environment must be fully differentiable (no stop_gradient, continuous
      actions). Use env_name=differentiable with the DifferentiableAcrobot.

Key differences from PPO:
    - No trajectory buffer, GAE, advantages, minibatches, or epochs.
    - env.step is called INSIDE jax.grad (the rollout IS the loss computation).
    - A critic provides an optional terminal value bootstrap at the horizon end.
    - Exploration via Gaussian noise (reparameterization trick).

Reference:
    Madeka et al., "Deep Inventory Management", arXiv:2210.03137 (2022).
    Also related to SHAC, APG (analytic policy gradients).
"""

import copy
import time
from typing import Any, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from stoa import Environment, get_final_step_metrics
from stoa.env_types import StepType

from stoix.base_types import (
    ActorApply,
    ActorCriticOptStates,
    ActorCriticParams,
    AnakinExperimentOutput,
    CriticApply,
    LearnerFn,
    OnPolicyLearnerState,
)
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.networks.base import CompositeNetwork
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.networks.postprocessors import tanh_to_spec
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import (
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[OnPolicyLearnerState]:
    """Get the learner function.

    The core of Direct Backprop: the policy update differentiates through
    the environment dynamics to compute exact policy gradients.
    """

    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(
        learner_state: OnPolicyLearnerState, _: Any
    ) -> Tuple[OnPolicyLearnerState, Tuple]:
        """A single update of the network.

        Rolls out the policy for rollout_length steps through the differentiable
        environment, computes the negative return as the loss, and backpropagates
        through the dynamics to get exact actor gradients.
        """

        params, opt_states, key, env_state, last_timestep = learner_state

        def _differentiable_rollout_loss(
            actor_params: FrozenDict,
            critic_params: FrozenDict,
            env_state: Any,
            observation: chex.Array,
            key: chex.PRNGKey,
        ) -> Tuple[chex.Array, Tuple]:
            """Roll out H steps through the differentiable env and return negative return.

            This function is the computational graph that jax.grad differentiates through.
            env.step is called INSIDE this function, so gradients flow through dynamics.

            Note: observation has shape (num_envs, obs_dim) because the env is vmapped.
            All per-env quantities (discount, reward) also have shape (num_envs,).
            """
            num_envs = observation.shape[0]

            def _scan_step(carry: Tuple, _: Any) -> Tuple[Tuple, chex.Array]:
                state, obs, cumulative_discount, step_key = carry

                # Policy forward pass (deterministic action + exploration noise)
                # obs shape: (num_envs, obs_dim) — actor expects batch dim
                action_dist = actor_apply_fn(actor_params, obs)
                action = action_dist.mode()  # (num_envs, action_dim)

                # Exploration noise via reparameterization trick
                step_key, noise_key = jax.random.split(step_key)
                noise = (
                    jax.random.normal(noise_key, action.shape)
                    * config.system.exploration_noise
                )
                action = action + noise
                action = jnp.clip(
                    action, config.system.action_minimum, config.system.action_maximum
                )

                # DIFFERENTIABLE environment step
                state, timestep = env.step(state, action)

                # Mask for episode boundaries (shape: (num_envs,))
                alive = 1.0 - (timestep.discount == 0.0).astype(jnp.float32)

                # Discounted reward for this step (shape: (num_envs,))
                reward = timestep.reward * config.system.reward_scale * cumulative_discount

                # Update cumulative discount (zeros out after termination)
                new_cumulative_discount = cumulative_discount * config.system.gamma * alive

                new_obs = timestep.observation
                return (state, new_obs, new_cumulative_discount, step_key), reward

            # Initial carry — cumulative_discount has shape (num_envs,)
            init_carry = (
                env_state,
                observation,
                jnp.ones(num_envs, dtype=jnp.float32),
                key,
            )

            # Unroll H steps. rewards shape: (rollout_length, num_envs)
            (final_state, final_obs, final_discount, _), rewards = jax.lax.scan(
                _scan_step, init_carry, None, config.system.rollout_length
            )

            # Sum of discounted rewards: sum over time, mean over envs
            total_return = jnp.sum(rewards, axis=0)  # (num_envs,)

            # Optional: terminal value bootstrap from critic
            if config.system.use_terminal_value:
                # final_obs shape: (num_envs, obs_dim)
                terminal_value = critic_apply_fn(critic_params, final_obs)
                # terminal_value shape: (num_envs,)
                terminal_value = terminal_value * final_discount
            else:
                terminal_value = jnp.zeros(num_envs)

            total_return = total_return + terminal_value  # (num_envs,)

            # Average over environments, negate for loss
            loss = -jnp.mean(total_return)

            aux = (final_state, final_obs, jnp.mean(total_return))
            return loss, aux

        # ---- Compute actor gradients ----
        key, rollout_key = jax.random.split(key)

        # We differentiate w.r.t. actor_params (argnums=0).
        # critic_params are passed but NOT differentiated here (actor loss only).
        actor_grad_fn = jax.grad(_differentiable_rollout_loss, argnums=0, has_aux=True)
        actor_grads, (new_env_state, final_obs, total_return) = actor_grad_fn(
            params.actor_params,
            params.critic_params,
            env_state,
            last_timestep.observation,
            rollout_key,
        )

        # Average gradients across batch and devices
        actor_grads = jax.lax.pmean(actor_grads, axis_name="batch")
        actor_grads = jax.lax.pmean(actor_grads, axis_name="device")

        # Update actor parameters
        actor_updates, actor_new_opt_state = actor_update_fn(
            actor_grads, opt_states.actor_opt_state
        )
        actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

        # ---- Update critic (supervised, not through dynamics) ----
        def _critic_loss_fn(critic_params: FrozenDict) -> Tuple[chex.Array, dict]:
            """MSE loss between critic prediction and observed mean return."""
            # last_timestep.observation shape: (num_envs, obs_dim)
            predicted_values = critic_apply_fn(
                critic_params, last_timestep.observation
            )  # (num_envs,)
            # Target is the observed (stopped) mean total return (scalar)
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

        # ---- Pack new params and state ----
        new_params = ActorCriticParams(actor_new_params, critic_new_params)
        new_opt_states = ActorCriticOptStates(actor_new_opt_state, critic_new_opt_state)

        # Advance the environment: the final_state from the rollout IS the new env_state.
        # Construct a mid-episode timestep with the final observation for the next rollout.
        num_envs = final_obs.shape[0]
        new_timestep = last_timestep.replace(
            observation=final_obs,
            step_type=jnp.full(num_envs, StepType.MID, dtype=jnp.int8),
            reward=jnp.zeros(num_envs, dtype=jnp.float32),
            discount=jnp.ones(num_envs, dtype=jnp.float32),
        )

        learner_state = OnPolicyLearnerState(
            new_params, new_opt_states, key, new_env_state, new_timestep
        )

        # ---- Metrics ----
        actor_grad_norm = optax.global_norm(actor_grads)
        loss_info = {
            "actor_loss": -total_return,
            "mean_return": total_return,
            "grad_norm": actor_grad_norm,
            **critic_loss_info,
        }
        # Episode metrics: since we differentiate through the env (no
        # RecordEpisodeMetrics inside the grad), we report rollout return
        # as episode metrics. is_terminal_step=False means the logger won't
        # try to extract per-episode stats from training (eval handles that).
        episode_info = {
            "episode_return": total_return,
            "episode_length": jnp.array(
                config.system.rollout_length, dtype=jnp.float32
            ),
            "is_terminal_step": jnp.array(False),
        }

        return learner_state, (episode_info, loss_info)

    def learner_fn(
        learner_state: OnPolicyLearnerState,
    ) -> AnakinExperimentOutput[OnPolicyLearnerState]:
        """Learner function.

        Vectorizes the update step over the batch dimension and scans
        over num_updates_per_eval iterations.
        """

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
) -> Tuple[LearnerFn[OnPolicyLearnerState], Actor, OnPolicyLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available devices.
    n_devices = len(jax.devices())

    # Get action space info.
    action_dim = int(env.action_space().shape[-1])
    config.system.action_dim = action_dim
    config.system.action_minimum = float(env.action_space().minimum)
    config.system.action_maximum = float(env.action_space().maximum)

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define actor network with DeterministicHead + tanh_to_spec postprocessor.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
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
    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)

    # Define critic network.
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)
    critic_network = Critic(torso=critic_torso, critic_head=critic_head)

    # Optimizers (no epochs/minibatches in Direct Backprop, so pass 1).
    actor_lr = make_learning_rate(config.system.actor_lr, config, num_epochs=1)
    critic_lr = make_learning_rate(config.system.critic_lr, config, num_epochs=1)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation for network init.
    init_x = env.observation_space().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state.
    critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params.
    params = ActorCriticParams(actor_params, critic_params)

    actor_network_apply_fn = actor_network.apply
    critic_network_apply_fn = critic_network.apply

    # Pack apply and update functions.
    apply_fns = (actor_network_apply_fn, critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = env.reset(jnp.stack(env_keys))
    reshape_states = lambda x: x.reshape(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update_batch_size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape(
        (n_devices, config.arch.update_batch_size) + x.shape[1:]
    )
    step_keys = reshape_keys(jnp.stack(step_keys))
    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states = replicate_learner
    init_learner_state = OnPolicyLearnerState(
        params, opt_states, step_keys, env_states, timesteps
    )

    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Create the environments for train and eval.
    env, eval_env = environments.make(config=config)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        eval_act_fn=get_distribution_act_fn(config, actor_network.apply),
        params=learner_state.params.actor_params,
        config=config,
    )

    # Calculate environment steps per evaluation.
    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup.
    logger = StoixLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}JAX Global Devices {jax.devices()}{Style.RESET_ALL}"
    )

    # Set up checkpointer.
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_learner_state = unreplicate_batch_dim(learner_state)
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(
            learner_output.episode_metrics
        )
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        # One gradient step per rollout (no epochs/minibatches).
        opt_steps_per_eval = config.arch.num_updates_per_eval
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = unreplicate_batch_dim(
            learner_output.learner_state.params.actor_params
        )
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        evaluator_output = evaluator(trained_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(
            evaluator_output.episode_metrics["episode_return"]
        )

        steps_per_eval = int(
            jnp.sum(evaluator_output.episode_metrics["episode_length"])
        )
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

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
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
        steps_per_eval = int(
            jnp.sum(evaluator_output.episode_metrics["episode_length"])
        )
        evaluator_output.episode_metrics["steps_per_second"] = (
            steps_per_eval / elapsed_time
        )
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()
    # Record the performance for the final evaluation run.
    eval_performance = float(
        jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric])
    )
    return eval_performance


@hydra.main(
    config_path="../../configs/default/anakin",
    config_name="default_ff_direct_backprop.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    t0 = time.time()
    eval_performance = run_experiment(cfg)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Direct Backprop experiment completed in "
        f"{time.time() - t0:.2f} seconds.{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
