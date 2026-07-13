"""Previous-action concatenation wrapper.

Appends the PREVIOUS action to the current observation, so a recurrent (or even
feedforward) agent can condition on what it just did. At the first step of an
episode the previous action is a zero vector.

This is standard practice for recurrent policies/values: without it, the network
must infer the last action from the observation transition alone. For our
recurrent-value experiments it lets the hidden state accumulate an explicit
action history, which is the missing ingredient the earlier runs lacked.

Action encoding:
  * Discrete action space  -> one-hot of the previous action (dim = num_values).
  * Continuous (box) space -> the raw previous action vector (dim = action_dim).

The wrapper resizes the observation space accordingly and tracks the last action
in its own state so everything stays functional/JAX-pure.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from stoa.env_types import Action, EnvParams, TimeStep
from stoa.core_wrappers.wrapper import Wrapper
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, Space

try:
    from flax.struct import dataclass
except ImportError:  # pragma: no cover
    from dataclasses import dataclass


@dataclass
class ActionConcatState:
    """Wrapper state: the wrapped env's state plus the last-action feature vector."""

    env_state: object
    last_action_feat: Array  # (action_feature_dim,), the encoded previous action


class ActionConcatWrapper(Wrapper):
    """Concatenate the previous action (zero at t=0) onto the observation."""

    def __init__(self, env: Environment):
        super().__init__(env)
        action_space = self._env.action_space()
        if hasattr(action_space, "num_values"):
            # Discrete: one-hot encode the previous action.
            self._is_discrete = True
            self._action_feature_dim = int(action_space.num_values)
        else:
            # Continuous (box): use the raw action vector.
            self._is_discrete = False
            self._action_feature_dim = int(action_space.shape[-1])
        # Public attribute so callers (systems/analysis) can discover how many
        # trailing observation features are the appended previous action, and
        # strip them for networks that must remain Markov (reactive actor,
        # memory-free critic). Reachable through stoa's wrapper delegation.
        self.action_feature_dim = self._action_feature_dim

    def _encode_action(self, action: Action) -> Array:
        """Encode an action into the feature vector appended to the observation."""
        if self._is_discrete:
            return jax.nn.one_hot(
                jnp.asarray(action).reshape(()), self._action_feature_dim, dtype=jnp.float32
            )
        return jnp.asarray(action, dtype=jnp.float32).reshape(self._action_feature_dim)

    def _augment(self, observation: Array, last_action_feat: Array) -> Array:
        """Concatenate the encoded previous action onto the observation."""
        return jnp.concatenate([observation, last_action_feat], axis=-1)

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[ActionConcatState, TimeStep]:
        env_state, timestep = self._env.reset(rng_key, env_params)
        # No previous action at the first step -> zero vector.
        last_action_feat = jnp.zeros((self._action_feature_dim,), dtype=jnp.float32)
        timestep = timestep.replace(
            observation=self._augment(timestep.observation, last_action_feat)
        )
        return ActionConcatState(env_state=env_state, last_action_feat=last_action_feat), timestep

    def step(
        self,
        state: ActionConcatState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[ActionConcatState, TimeStep]:
        env_state, timestep = self._env.step(state.env_state, action, env_params)
        # The observation returned for the resulting state is augmented with the
        # action that WAS JUST TAKEN (i.e. the previous action relative to the
        # next decision), matching the "last action" convention.
        new_last_action_feat = self._encode_action(action)
        timestep = timestep.replace(
            observation=self._augment(timestep.observation, new_last_action_feat)
        )
        return (
            ActionConcatState(env_state=env_state, last_action_feat=new_last_action_feat),
            timestep,
        )

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Extend the wrapped observation space by the action-feature dimension."""
        base = self._env.observation_space(env_params)
        base_shape = base.shape
        new_shape = (*base_shape[:-1], base_shape[-1] + self._action_feature_dim)

        # Bounds for the appended action features.
        if self._is_discrete:
            act_low = jnp.zeros((self._action_feature_dim,), dtype=jnp.float32)
            act_high = jnp.ones((self._action_feature_dim,), dtype=jnp.float32)
        else:
            action_space = self._env.action_space(env_params)
            act_low = jnp.broadcast_to(
                jnp.asarray(action_space.minimum, dtype=jnp.float32),
                (self._action_feature_dim,),
            )
            act_high = jnp.broadcast_to(
                jnp.asarray(action_space.maximum, dtype=jnp.float32),
                (self._action_feature_dim,),
            )

        base_low = jnp.broadcast_to(jnp.asarray(base.minimum, dtype=jnp.float32), base_shape)
        base_high = jnp.broadcast_to(jnp.asarray(base.maximum, dtype=jnp.float32), base_shape)
        new_low = jnp.concatenate([base_low, act_low], axis=-1)
        new_high = jnp.concatenate([base_high, act_high], axis=-1)

        return BoundedArraySpace(
            shape=new_shape,
            dtype=jnp.float32,
            minimum=new_low,
            maximum=new_high,
            name="observation",
        )
