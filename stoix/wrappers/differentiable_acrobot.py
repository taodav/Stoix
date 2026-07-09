"""Differentiable Acrobot environment for Direct Backprop.

A continuous-action, fully-differentiable version of the classic Acrobot-v1
environment. Unlike the gymnax implementation, this version:
  - Has NO jax.lax.stop_gradient calls (gradients flow through dynamics)
  - Uses a continuous action space (torque in [-1, 1])
  - Provides a smooth, differentiable reward (tip height)
  - Only terminates by step limit (no early termination on goal)

This enables backpropagation of reward gradients through the environment
dynamics into the policy parameters (Direct Backprop / analytic policy gradients).

Physics are identical to gymnax Acrobot (RK4, same constants).
"""

from typing import TYPE_CHECKING, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import BoundedArraySpace, Space

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class AcrobotState:
    """Internal state of the differentiable Acrobot."""

    joint_angle1: Array
    joint_angle2: Array
    velocity_1: Array
    velocity_2: Array
    time: Array  # int32 step counter


# ============================================================================
# Physics (ported from gymnax, all pure jnp — inherently differentiable)
# ============================================================================

# Default physical constants
_LINK_LENGTH_1 = 1.0
_LINK_LENGTH_2 = 1.0
_LINK_MASS_1 = 1.0
_LINK_MASS_2 = 1.0
_LINK_COM_POS_1 = 0.5
_LINK_COM_POS_2 = 0.5
_LINK_MOI = 1.0
_MAX_VEL_1 = 4.0 * jnp.pi
_MAX_VEL_2 = 9.0 * jnp.pi
_GRAVITY = 9.8


def _dsdt(s_augmented: Array) -> Array:
    """Compute time derivative of the augmented state [theta1, theta2, dtheta1, dtheta2, torque]."""
    theta1, theta2, dtheta1, dtheta2, a = (
        s_augmented[0],
        s_augmented[1],
        s_augmented[2],
        s_augmented[3],
        s_augmented[4],
    )

    m1 = _LINK_MASS_1
    m2 = _LINK_MASS_2
    l1 = _LINK_LENGTH_1
    lc1 = _LINK_COM_POS_1
    lc2 = _LINK_COM_POS_2
    i1 = _LINK_MOI
    i2 = _LINK_MOI
    g = _GRAVITY

    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * jnp.cos(theta2)) + i1 + i2
    d2 = m2 * (lc2**2 + l1 * lc2 * jnp.cos(theta2)) + i2
    phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2**2 * jnp.sin(theta2)
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2.0)
        + phi2
    )
    ddtheta2 = (
        a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * jnp.sin(theta2) - phi2
    ) / (m2 * lc2**2 + i2 - d2**2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

    return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0])


def _rk4(y0: Array, dt: float) -> Array:
    """Single RK4 integration step."""
    dt2 = dt / 2.0
    k1 = _dsdt(y0)
    k2 = _dsdt(y0 + dt2 * k1)
    k3 = _dsdt(y0 + dt2 * k2)
    k4 = _dsdt(y0 + dt * k3)
    yout = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


def _wrap(x: Array, m: float, big_m: float) -> Array:
    """Wrap angle x into [m, big_m). Uses differentiable ops."""
    diff = big_m - m
    # Shift into [0, diff) then shift back
    return (x - m) % diff + m


def _get_obs(state: AcrobotState) -> Array:
    """Compute 6D observation from state."""
    return jnp.array([
        jnp.cos(state.joint_angle1),
        jnp.sin(state.joint_angle1),
        jnp.cos(state.joint_angle2),
        jnp.sin(state.joint_angle2),
        state.velocity_1,
        state.velocity_2,
    ])


# ============================================================================
# Environment
# ============================================================================


class DifferentiableAcrobot(Environment):
    """Differentiable Acrobot with continuous torque and smooth reward.

    The reward at each step is the negative cosine-based height of the tip:
        r = -cos(theta1) - cos(theta1 + theta2)
    This ranges from -2 (hanging down) to +2 (tip at top), providing a dense
    differentiable signal for learning.

    Action: continuous torque in [-1, 1] applied to the second joint.
    """

    def __init__(self, max_steps_in_episode: int = 200, dt: float = 0.2):
        super().__init__()
        self._max_steps = max_steps_in_episode
        self._dt = dt

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[AcrobotState, TimeStep]:
        """Reset to a small random state near the bottom."""
        init_state = jax.random.uniform(rng_key, shape=(4,), minval=-0.1, maxval=0.1)
        state = AcrobotState(
            joint_angle1=init_state[0],
            joint_angle2=init_state[1],
            velocity_1=init_state[2],
            velocity_2=init_state[3],
            time=jnp.array(0, dtype=jnp.int32),
        )
        obs = _get_obs(state)
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=obs,
            extras={},
        )
        return state, timestep

    def step(
        self,
        state: AcrobotState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[AcrobotState, TimeStep]:
        """Step the environment with continuous torque action.

        Args:
            state: Current environment state.
            action: Continuous torque, shape (1,), in [-1, 1].
            env_params: Unused (constants are fixed).

        Returns:
            (new_state, timestep) — both fully differentiable w.r.t. action and state.
        """
        # Extract scalar torque from action array
        torque = jnp.squeeze(action)

        # Build augmented state for RK4: [theta1, theta2, dtheta1, dtheta2, torque]
        s_augmented = jnp.array([
            state.joint_angle1,
            state.joint_angle2,
            state.velocity_1,
            state.velocity_2,
            torque,
        ])

        # Integrate one step
        ns = _rk4(s_augmented, self._dt)

        # Wrap angles to [-pi, pi)
        joint_angle1 = _wrap(ns[0], -jnp.pi, jnp.pi)
        joint_angle2 = _wrap(ns[1], -jnp.pi, jnp.pi)

        # Clip velocities
        velocity_1 = jnp.clip(ns[2], -_MAX_VEL_1, _MAX_VEL_1)
        velocity_2 = jnp.clip(ns[3], -_MAX_VEL_2, _MAX_VEL_2)

        # Smooth reward: tip height (differentiable)
        reward = -jnp.cos(joint_angle1) - jnp.cos(joint_angle1 + joint_angle2)

        # Update state
        new_time = state.time + 1
        new_state = AcrobotState(
            joint_angle1=joint_angle1,
            joint_angle2=joint_angle2,
            velocity_1=velocity_1,
            velocity_2=velocity_2,
            time=jnp.int32(new_time),
        )

        # Termination: only by step limit (truncation, not termination)
        done = new_time >= self._max_steps
        step_type = jnp.where(done, StepType.TRUNCATED, StepType.MID)
        discount = jnp.where(done, 0.0, 1.0)

        obs = _get_obs(new_state)

        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            discount=jnp.asarray(discount, dtype=jnp.float32),
            observation=obs,
            extras={},
        )

        return new_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """6D observation: [cos(t1), sin(t1), cos(t2), sin(t2), w1, w2]."""
        high = jnp.array([1.0, 1.0, 1.0, 1.0, _MAX_VEL_1, _MAX_VEL_2], dtype=jnp.float32)
        return BoundedArraySpace(
            shape=(6,), dtype=jnp.float32, minimum=-high, maximum=high, name="observation"
        )

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """Continuous torque in [-1, 1]."""
        return BoundedArraySpace(
            shape=(1,), dtype=jnp.float32, minimum=-1.0, maximum=1.0, name="action"
        )

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        """State space (angles and velocities)."""
        high = jnp.array(
            [jnp.pi, jnp.pi, _MAX_VEL_1, _MAX_VEL_2], dtype=jnp.float32
        )
        return BoundedArraySpace(
            shape=(4,), dtype=jnp.float32, minimum=-high, maximum=high, name="state"
        )
