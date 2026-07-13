import copy
import time
from typing import Any, Dict, Tuple

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

from optax import OptState
from typing_extensions import NamedTuple

from stoix.base_types import (
    Action,
    AnakinExperimentOutput,
    Done,
    HiddenState,
    LearnerFn,
    RecActorApply,
    RecCriticApply,
    RNNLearnerState,
    Truncated,
    Value,
)
from stoix.evaluator import evaluator_setup, get_rec_distribution_act_fn
from stoix.networks.base import RecurrentActor, RecurrentCritic, ScannedRNN
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.loss import clipped_value_loss, ppo_clip_loss
from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate


# --- Dual-value data structures ---
# This system trains the usual recurrent actor + recurrent critic, plus an
# additional NON-recurrent value head that rides along on the same data stream.
# The non-recurrent head uses the identical architecture to the recurrent critic
# but has its hidden state zeroed at every step (implemented by always passing
# `reset=True` to the ScannedRNN), so its output is a pure function of the current
# observation. It is trained with its own independent GAE bootstrap and never
# influences the policy. This lets us compare recurrent vs. non-recurrent
# "value drift" as described in the experiment.


class DualValueParams(NamedTuple):
    """Parameters of the actor, recurrent critic and non-recurrent critic."""

    actor_params: FrozenDict
    critic_params: FrozenDict
    nr_critic_params: FrozenDict


class DualValueOptStates(NamedTuple):
    """Optimiser states for the actor, recurrent critic and non-recurrent critic."""

    actor_opt_state: OptState
    critic_opt_state: OptState
    nr_critic_opt_state: OptState


class DualValueHiddenStates(NamedTuple):
    """Hidden states for the actor, recurrent critic and non-recurrent critic."""

    policy_hidden_state: HiddenState
    critic_hidden_state: HiddenState
    nr_critic_hidden_state: HiddenState


class RNNDualValueTransition(NamedTuple):
    """Transition tuple for dual-value recurrent PPO."""

    done: Done
    truncated: Truncated
    action: Action
    value: Value
    nr_value: Value
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    hstates: DualValueHiddenStates
    info: Dict


def strip_action_features(observation: chex.Array, action_feature_dim: int) -> chex.Array:
    """Drop the trailing previous-action features from an observation.

    The ActionConcatWrapper appends the previous action to every observation so
    the RECURRENT critic (which processes trajectories) can accumulate an action
    history. Networks that must stay Markov -- the reactive actor and the
    memory-free critic -- must NOT see the previous action, since it is history,
    not current state. This removes the last ``action_feature_dim`` features.
    A dim of 0 (no wrapper) is a no-op.
    """
    if action_feature_dim <= 0:
        return observation
    return observation[..., :-action_feature_dim]


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[RecActorApply, RecCriticApply, RecCriticApply],
    update_fns: Tuple[
        optax.TransformUpdateFn, optax.TransformUpdateFn, optax.TransformUpdateFn
    ],
    config: DictConfig,
) -> LearnerFn[RNNLearnerState]:
    """Get the learner function."""

    actor_apply_fn, critic_apply_fn, nr_critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn, nr_critic_update_fn = update_fns

    def _update_step(learner_state: RNNLearnerState, _: Any) -> Tuple[RNNLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
            learner_state (NamedTuple):
                - params (ActorCriticParams): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
                - dones (bool): Whether the last timestep was a terminal state.
                - hstates (ActorCriticHiddenStates): The current hidden states of the RNN.
            _ (Any): The current metrics info.
        """

        def _env_step(
            learner_state: RNNLearnerState, _: Any
        ) -> Tuple[RNNLearnerState, RNNDualValueTransition]:
            """Step the environment."""
            (
                params,
                opt_states,
                key,
                env_state,
                last_timestep,
                last_done,
                last_truncated,
                hstates,
            ) = learner_state

            key, policy_key = jax.random.split(key)

            # Add a batch dimension to the observation.
            batched_observation = jax.tree_util.tree_map(
                lambda x: x[jnp.newaxis, :], last_timestep.observation
            )
            reset_hidden_state = jnp.logical_or(last_done, last_truncated)
            # Markov observation (previous-action features stripped) for the
            # networks that must not see action history.
            batched_observation_markov = strip_action_features(
                batched_observation, config.system.action_feature_dim
            )
            # The CRITIC normally uses the real reset logic (carries memory) and
            # sees the FULL observation. If reset_critic_hidden_state_every_step is
            # set, its hidden state is zeroed every step too -> a MARKOV value
            # function (the Exp-1 baseline). One knob isolates recurrent-vs-Markov
            # value, holding the reactive actor and Markov targets fixed.
            critic_reset = (
                jnp.ones_like(reset_hidden_state)
                if config.system.reset_critic_hidden_state_every_step
                else reset_hidden_state
            )
            critic_ac_in = (
                batched_observation,
                critic_reset[jnp.newaxis, :],
            )

            # The ACTOR is made non-recurrent by always zeroing its hidden state
            # (reset flag = True every step). Its output is a pure function of the
            # current observation (action features stripped), so pi is reactive / Markov.
            actor_ac_in = (
                batched_observation_markov,
                jnp.ones_like(reset_hidden_state[jnp.newaxis, :]),
            )

            # The memory-free value head is also Markov: stripped obs, zeroed hidden.
            nr_ac_in = (
                batched_observation_markov,
                jnp.ones_like(reset_hidden_state[jnp.newaxis, :]),
            )

            # Run the network.
            policy_hidden_state, actor_policy = actor_apply_fn(
                params.actor_params, hstates.policy_hidden_state, actor_ac_in
            )
            critic_hidden_state, value = critic_apply_fn(
                params.critic_params, hstates.critic_hidden_state, critic_ac_in
            )
            nr_critic_hidden_state, nr_value = nr_critic_apply_fn(
                params.nr_critic_params, hstates.nr_critic_hidden_state, nr_ac_in
            )

            # Sample action from the policy and squeeze out the batch dimension.
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)
            value, nr_value, action, log_prob = (
                value.squeeze(0),
                nr_value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # Step the environment.
            env_state, timestep = env.step(env_state, action)

            # log episode return and length
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]

            hstates = DualValueHiddenStates(
                policy_hidden_state, critic_hidden_state, nr_critic_hidden_state
            )
            transition = RNNDualValueTransition(
                last_done,
                last_truncated,
                action,
                value,
                nr_value,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                hstates,
                info,
            )
            learner_state = RNNLearnerState(
                params,
                opt_states,
                key,
                env_state,
                timestep,
                done,
                truncated,
                hstates,
            )
            return learner_state, transition

        # INITIALISE RNN STATE
        initial_hstates = learner_state.hstates

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # CALCULATE ADVANTAGE
        (
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            last_truncated,
            hstates,
        ) = learner_state

        # Add a batch dimension to the observation.
        batched_last_observation = jax.tree_util.tree_map(
            lambda x: x[jnp.newaxis, :], last_timestep.observation
        )
        reset_hidden_state = jnp.logical_or(last_done, last_truncated)
        batched_last_observation_markov = strip_action_features(
            batched_last_observation, config.system.action_feature_dim
        )
        # Recurrent critic bootstrap: full obs. Memory-free critic: stripped obs.
        # Zero the critic hidden state too if running the Markov-value baseline.
        critic_reset = (
            jnp.ones_like(reset_hidden_state)
            if config.system.reset_critic_hidden_state_every_step
            else reset_hidden_state
        )
        ac_in = (
            batched_last_observation,
            critic_reset[jnp.newaxis, :],
        )
        nr_ac_in = (
            batched_last_observation_markov,
            jnp.ones_like(reset_hidden_state[jnp.newaxis, :]),
        )

        # Run the network.
        _, last_val = critic_apply_fn(params.critic_params, hstates.critic_hidden_state, ac_in)
        _, nr_last_val = nr_critic_apply_fn(
            params.nr_critic_params, hstates.nr_critic_hidden_state, nr_ac_in
        )
        # Squeeze out the batch dimension and mask out the value of terminal states.
        last_val = last_val.squeeze(0)
        last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)
        nr_last_val = nr_last_val.squeeze(0)
        nr_last_val = jnp.where(last_done, jnp.zeros_like(nr_last_val), nr_last_val)

        r_t = traj_batch.reward
        d_t = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (d_t * config.system.gamma).astype(jnp.float32)

        # Recurrent critic advantages/targets (used for the policy update).
        v_t = jnp.concatenate([traj_batch.value, last_val[None, ...]], axis=0)
        advantages, targets = batch_truncated_generalized_advantage_estimation(
            r_t,
            d_t,
            config.system.gae_lambda,
            values=v_t,
            time_major=True,
            standardize_advantages=config.system.standardize_advantages,
        )

        # Non-recurrent critic targets: an independent GAE bootstrap using the
        # non-recurrent head's own values. These are only used to train the
        # non-recurrent value head and never touch the policy. We don't need its
        # advantages, so we never standardize them.
        nr_v_t = jnp.concatenate([traj_batch.nr_value, nr_last_val[None, ...]], axis=0)
        _, nr_targets = batch_truncated_generalized_advantage_estimation(
            r_t,
            d_t,
            config.system.gae_lambda,
            values=nr_v_t,
            time_major=True,
            standardize_advantages=False,
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                params, opt_states = train_state
                (
                    traj_batch,
                    advantages,
                    targets,
                    nr_targets,
                ) = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: RNNDualValueTransition,
                    gae: chex.Array,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN NETWORK — actor is non-recurrent: always reset hidden
                    # state, and use the Markov obs (previous-action features stripped).
                    actor_reset = jnp.ones_like(
                        jnp.logical_or(traj_batch.done, traj_batch.truncated)
                    )
                    obs_markov = strip_action_features(
                        traj_batch.obs, config.system.action_feature_dim
                    )
                    obs_and_done = (obs_markov, actor_reset)
                    policy_hidden_state = jax.tree_util.tree_map(
                        lambda x: x[0], traj_batch.hstates.policy_hidden_state
                    )
                    _, actor_policy = actor_apply_fn(
                        actor_params, policy_hidden_state, obs_and_done
                    )
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    loss_actor = ppo_clip_loss(
                        log_prob, traj_batch.log_prob, gae, config.system.clip_eps
                    )
                    entropy = actor_policy.entropy().mean()

                    total_loss = loss_actor - config.system.ent_coef * entropy
                    loss_info = {
                        "actor_loss": loss_actor,
                        "entropy": entropy,
                    }
                    return total_loss, loss_info

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: RNNDualValueTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # RERUN NETWORK. Zero the critic hidden state every step for the
                    # Markov-value baseline; else use the real episode-boundary resets.
                    reset_hidden_state = jnp.logical_or(traj_batch.done, traj_batch.truncated)
                    if config.system.reset_critic_hidden_state_every_step:
                        reset_hidden_state = jnp.ones_like(reset_hidden_state)
                    obs_and_done = (traj_batch.obs, reset_hidden_state)
                    critic_hidden_state = jax.tree_util.tree_map(
                        lambda x: x[0], traj_batch.hstates.critic_hidden_state
                    )
                    _, value = critic_apply_fn(critic_params, critic_hidden_state, obs_and_done)

                    # CALCULATE VALUE LOSS
                    value_loss = clipped_value_loss(
                        value, traj_batch.value, targets, config.system.clip_eps
                    )

                    total_loss = config.system.vf_coef * value_loss
                    loss_info = {
                        "value_loss": value_loss,
                    }
                    return total_loss, loss_info

                def _nr_critic_loss_fn(
                    nr_critic_params: FrozenDict,
                    traj_batch: RNNDualValueTransition,
                    nr_targets: chex.Array,
                ) -> Tuple:
                    """Calculate the non-recurrent critic loss.

                    Identical architecture and loss to the recurrent critic, but the
                    hidden state is zeroed at every step (reset flag always True), so the
                    value is a pure function of the current observation.
                    """
                    # RERUN NETWORK with the hidden state reset at every step and
                    # the Markov obs (previous-action features stripped).
                    reset_hidden_state = jnp.ones_like(
                        jnp.logical_or(traj_batch.done, traj_batch.truncated)
                    )
                    obs_markov = strip_action_features(
                        traj_batch.obs, config.system.action_feature_dim
                    )
                    obs_and_done = (obs_markov, reset_hidden_state)
                    nr_critic_hidden_state = jax.tree_util.tree_map(
                        lambda x: x[0], traj_batch.hstates.nr_critic_hidden_state
                    )
                    _, nr_value = nr_critic_apply_fn(
                        nr_critic_params, nr_critic_hidden_state, obs_and_done
                    )

                    # CALCULATE VALUE LOSS
                    nr_value_loss = clipped_value_loss(
                        nr_value, traj_batch.nr_value, nr_targets, config.system.clip_eps
                    )

                    total_loss = config.system.vf_coef * nr_value_loss
                    loss_info = {
                        "nr_value_loss": nr_value_loss,
                    }
                    return total_loss, loss_info

                # CALCULATE ACTOR LOSS
                actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
                actor_grads, actor_loss_info = actor_grad_fn(
                    params.actor_params, traj_batch, advantages
                )

                # CALCULATE CRITIC LOSS
                # --- Memory-free-target intervention ---
                # The recurrent critic is regressed toward `nr_targets` (the GAE
                # targets bootstrapped from the MEMORY-FREE nr_critic head) instead
                # of `targets` (bootstrapped from its own history-conditioned
                # values). This severs the self-reinforcing bootstrap loop while
                # leaving the recurrent critic architecturally identical, isolating
                # whether that loop is what drives spurious history-use. The policy
                # still uses `advantages` (recurrent GAE), so learning is otherwise
                # unchanged.
                critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
                critic_grads, critic_loss_info = critic_grad_fn(
                    params.critic_params, traj_batch, nr_targets
                )

                # CALCULATE NON-RECURRENT CRITIC LOSS
                nr_critic_grad_fn = jax.grad(_nr_critic_loss_fn, has_aux=True)
                nr_critic_grads, nr_critic_loss_info = nr_critic_grad_fn(
                    params.nr_critic_params, traj_batch, nr_targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="batch"
                )
                # pmean over devices.
                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="device"
                )

                nr_critic_grads, nr_critic_loss_info = jax.lax.pmean(
                    (nr_critic_grads, nr_critic_loss_info), axis_name="batch"
                )
                # pmean over devices.
                nr_critic_grads, nr_critic_loss_info = jax.lax.pmean(
                    (nr_critic_grads, nr_critic_loss_info), axis_name="device"
                )

                # UPDATE ACTOR PARAMS AND OPTIMISER STATE
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                # UPDATE CRITIC PARAMS AND OPTIMISER STATE
                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                # UPDATE NON-RECURRENT CRITIC PARAMS AND OPTIMISER STATE
                nr_critic_updates, nr_critic_new_opt_state = nr_critic_update_fn(
                    nr_critic_grads, opt_states.nr_critic_opt_state
                )
                nr_critic_new_params = optax.apply_updates(
                    params.nr_critic_params, nr_critic_updates
                )

                new_params = DualValueParams(
                    actor_new_params, critic_new_params, nr_critic_new_params
                )
                new_opt_state = DualValueOptStates(
                    actor_new_opt_state, critic_new_opt_state, nr_critic_new_opt_state
                )

                # PACK LOSS INFO
                loss_info = {
                    **actor_loss_info,
                    **critic_loss_info,
                    **nr_critic_loss_info,
                }

                return (new_params, new_opt_state), loss_info

            (
                params,
                opt_states,
                init_hstates,
                traj_batch,
                advantages,
                targets,
                nr_targets,
                key,
            ) = update_state
            key, shuffle_key = jax.random.split(key)

            # SHUFFLE MINIBATCHES
            batch = (traj_batch, advantages, targets, nr_targets)
            num_recurrent_chunks = (
                config.system.rollout_length // config.system.recurrent_chunk_size
            )
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape(
                    config.system.recurrent_chunk_size,
                    config.arch.num_envs * num_recurrent_chunks,
                    *x.shape[2:],
                ),
                batch,
            )
            permutation = jax.random.permutation(
                shuffle_key, config.arch.num_envs * num_recurrent_chunks
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )
            reshaped_batch = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, (x.shape[0], config.system.num_minibatches, -1, *x.shape[2:])
                ),
                shuffled_batch,
            )
            minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 0), reshaped_batch)

            # UPDATE MINIBATCHES
            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states), minibatches
            )

            update_state = (
                params,
                opt_states,
                init_hstates,
                traj_batch,
                advantages,
                targets,
                nr_targets,
                key,
            )
            return update_state, loss_info

        init_hstates = jax.tree_util.tree_map(lambda x: x[None, :], initial_hstates)
        update_state = (
            params,
            opt_states,
            init_hstates,
            traj_batch,
            advantages,
            targets,
            nr_targets,
            key,
        )

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, _, traj_batch, advantages, targets, nr_targets, key = update_state
        learner_state = RNNLearnerState(
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            last_truncated,
            hstates,
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: RNNLearnerState) -> AnakinExperimentOutput[RNNLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            learner_state (NamedTuple):
                - params (ActorCriticParams): The initial model parameters.
                - opt_states (OptStates): The initial optimizer states.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (WrapperState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
                - dones (bool): Whether the initial timestep was a terminal state.
                - hstateS (ActorCriticHiddenStates): The initial hidden states of the RNN.
        """

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

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
) -> Tuple[LearnerFn[RNNLearnerState], RecurrentActor, ScannedRNN, RNNLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number/dimension of actions.
    num_actions = int(env.action_space().num_values)
    config.system.action_dim = num_actions

    # If the env appends the previous action to the observation (ActionConcatWrapper),
    # discover how many trailing features that is. The RECURRENT critic sees the
    # full augmented observation; the reactive actor and memory-free critic have
    # these features stripped so they stay Markov.
    action_feature_dim = int(getattr(env, "action_feature_dim", 0))
    config.system.action_feature_dim = action_feature_dim

    # PRNG keys.
    key, actor_net_key, critic_net_key, nr_critic_net_key = keys

    # Define network and optimisers.
    actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=num_actions
    )
    critic_pre_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_post_torso = hydra.utils.instantiate(config.network.critic_network.post_torso)
    critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)
    # The non-recurrent critic uses the SAME architecture as the recurrent critic.
    # We instantiate independent module instances so it has its own parameters.
    nr_critic_pre_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    nr_critic_post_torso = hydra.utils.instantiate(config.network.critic_network.post_torso)
    nr_critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)

    actor_network = RecurrentActor(
        pre_torso=actor_pre_torso,
        hidden_state_dim=config.network.critic_network.rnn_layer.hidden_state_dim,
        cell_type=config.network.critic_network.rnn_layer.cell_type,
        post_torso=actor_post_torso,
        action_head=actor_action_head,
    )
    critic_network = RecurrentCritic(
        pre_torso=critic_pre_torso,
        hidden_state_dim=config.network.critic_network.rnn_layer.hidden_state_dim,
        cell_type=config.network.critic_network.rnn_layer.cell_type,
        post_torso=critic_post_torso,
        critic_head=critic_head,
    )
    nr_critic_network = RecurrentCritic(
        pre_torso=nr_critic_pre_torso,
        hidden_state_dim=config.network.critic_network.rnn_layer.hidden_state_dim,
        cell_type=config.network.critic_network.rnn_layer.cell_type,
        post_torso=nr_critic_post_torso,
        critic_head=nr_critic_head,
    )
    actor_rnn = ScannedRNN(
        hidden_state_dim=config.network.actor_network.rnn_layer.hidden_state_dim,
        cell_type=config.network.actor_network.rnn_layer.cell_type,
    )
    critic_rnn = ScannedRNN(
        hidden_state_dim=config.network.critic_network.rnn_layer.hidden_state_dim,
        cell_type=config.network.critic_network.rnn_layer.cell_type,
    )

    actor_lr = make_learning_rate(
        config.system.actor_lr, config, config.system.epochs, config.system.num_minibatches
    )
    critic_lr = make_learning_rate(
        config.system.critic_lr, config, config.system.epochs, config.system.num_minibatches
    )

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )
    nr_critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation
    init_obs = env.observation_space().generate_value()
    init_obs = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
        init_obs,
    )
    init_obs = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
    init_done = jnp.zeros((1, config.arch.num_envs), dtype=bool)
    # Full (possibly action-augmented) obs for the recurrent critic; Markov obs
    # (previous-action features stripped) for the reactive actor and memory-free
    # critic. When no ActionConcatWrapper is used, the two are identical.
    init_x = (init_obs, init_done)
    init_obs_markov = strip_action_features(init_obs, action_feature_dim)
    init_x_markov = (init_obs_markov, init_done)

    # Initialise hidden states.
    init_policy_hstate = actor_rnn.initialize_carry(config.arch.num_envs)
    init_critic_hstate = critic_rnn.initialize_carry(config.arch.num_envs)
    init_nr_critic_hstate = critic_rnn.initialize_carry(config.arch.num_envs)

    # initialise params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_policy_hstate, init_x_markov)
    actor_opt_state = actor_optim.init(actor_params)
    critic_params = critic_network.init(critic_net_key, init_critic_hstate, init_x)
    critic_opt_state = critic_optim.init(critic_params)
    nr_critic_params = nr_critic_network.init(
        nr_critic_net_key, init_nr_critic_hstate, init_x_markov
    )
    nr_critic_opt_state = nr_critic_optim.init(nr_critic_params)

    actor_network_apply_fn = actor_network.apply
    critic_network_apply_fn = critic_network.apply
    nr_critic_network_apply_fn = nr_critic_network.apply

    # Get network apply functions and optimiser updates.
    apply_fns = (actor_network_apply_fn, critic_network_apply_fn, nr_critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update, nr_critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Pack params and initial states.
    params = DualValueParams(actor_params, critic_params, nr_critic_params)
    hstates = DualValueHiddenStates(
        init_policy_hstate, init_critic_hstate, init_nr_critic_hstate
    )

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, restored_hstates = loaded_checkpoint.restore_params(
            input_params=params, restore_hstates=True
        )
        # Update the params and hstates
        params = restored_params
        hstates = restored_hstates if restored_hstates else hstates

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = env.reset(jnp.stack(env_keys))
    reshape_states = lambda x: x.reshape(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # Define params to be replicated across devices and batches.
    dones = jnp.zeros(
        (config.arch.num_envs,),
        dtype=bool,
    )
    truncated = jnp.zeros(
        (config.arch.num_envs,),
        dtype=bool,
    )
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    opt_states = DualValueOptStates(actor_opt_state, critic_opt_state, nr_critic_opt_state)
    replicate_learner = (params, opt_states, hstates, dones, truncated)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, hstates, dones, truncated = replicate_learner
    init_learner_state = RNNLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        done=dones,
        truncated=truncated,
        hstates=hstates,
    )
    return learn, actor_network, actor_rnn, init_learner_state


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

    # Set recurrent chunk size.
    if config.system.recurrent_chunk_size is None:
        config.system.recurrent_chunk_size = config.system.rollout_length
    else:
        assert (
            config.system.rollout_length % config.system.recurrent_chunk_size == 0
        ), "Rollout length must be divisible by recurrent chunk size."

    # Create the environments for train and eval.
    env, eval_env = environments.make(config)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key, nr_critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=5
    )

    # Setup learner.
    learn, actor_network, actor_rnn, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key, nr_critic_net_key), config
    )

    # Setup evaluator. The shared evaluator feeds the FULL env observation to the
    # actor, but the reactive actor was built on the Markov (action-stripped) obs.
    # Wrap the actor apply to strip the previous-action features first, so eval
    # matches training. A no-op when action_feature_dim == 0.
    eval_action_feature_dim = int(config.system.get("action_feature_dim", 0))

    def eval_actor_apply(params: FrozenDict, hstate: chex.Array, obs_and_done: Tuple) -> Tuple:
        observation, done = obs_and_done
        observation = strip_action_features(observation, eval_action_feature_dim)
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

    # Calculate number of updates per evaluation.
    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = StoixLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    print(f"{Fore.YELLOW}{Style.BRIGHT}JAX Global Devices {jax.devices()}{Style.RESET_ALL}")

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()
        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        # Calculate the number of optimiser steps per second. Since gradients are aggregated
        # across the device and batch axis, we don't consider updates per device/batch as part of
        # the SPS for the learner.
        opt_steps_per_eval = config.arch.num_updates_per_eval * (
            config.system.epochs * config.system.num_minibatches
        )
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = unreplicate_batch_dim(learner_output.learner_state.params.actor_params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        evaluator_output = evaluator(trained_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])

        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=int(steps_per_rollout * (eval_step + 1)),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        evaluator_output = absolute_metric_evaluator(best_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time

        t = int(steps_per_rollout * (eval_step + 1))
        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()
    # Record the performance for the final evaluation run. If the absolute metric is not
    # calculated, this will be the final evaluation run.
    eval_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))
    return eval_performance


@hydra.main(
    config_path="../../../configs/default/anakin",
    config_name="default_rec_ppo_nonrec_actor_mftarget.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}Non-recurrent actor + memory-free-target critic experiment completed"
        f"{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
