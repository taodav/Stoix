#!/bin/bash
#
# Direct Backprop (analytic policy gradients, https://arxiv.org/pdf/2210.03137) on
# the differentiable Acrobot, measuring ACTOR dependence on history.
#
# Runs the RECURRENT Direct Backprop system in two modes for comparison:
#   1. Memory (recurrent actor): the GRU carries history across the differentiable
#      rollout. Uses ACTION CONCATENATION (env=differentiable/acrobot_action_concat)
#      so the recurrent actor sees the previous action (obs 6 -> 7).
#   2. Memory-free (reset_hidden_state_every_step=True): identical architecture but
#      the hidden state is zeroed every step, so the actor is Markov. The appended
#      previous action is stripped, so it never sees action history.
#
# The feed-forward terminal-value critic (if use_terminal_value) is always Markov
# and strips the appended action. Set PYTHON=... to override the interpreter.
#
# Checkpoints:
#   ./checkpoints/rec_direct_backprop/acrobot_db_memory_ac/<step>/
#   ./checkpoints/rec_direct_backprop/acrobot_db_memfree/<step>/

set -euo pipefail
PYTHON="${PYTHON:-.venv/bin/python}"

# --- 1. Recurrent actor WITH memory + action concatenation ---
"${PYTHON}" stoix/systems/direct_backprop/rec_direct_backprop.py \
  env=differentiable/acrobot_action_concat \
  system.reset_hidden_state_every_step=False \
  arch.total_num_envs=64 \
  arch.total_timesteps=5e5 \
  arch.num_evaluation=20 \
  system.rollout_length=32 \
  system.actor_lr=1e-4 \
  system.exploration_noise=0.1 \
  logger.checkpointing.save_model=True \
  logger.checkpointing.save_args.checkpoint_uid=acrobot_db_memory_ac \
  "$@"

# --- 2. Memory-free actor (Markov; action features stripped automatically) ---
"${PYTHON}" stoix/systems/direct_backprop/rec_direct_backprop.py \
  env=differentiable/acrobot_action_concat \
  system.reset_hidden_state_every_step=True \
  arch.total_num_envs=64 \
  arch.total_timesteps=5e5 \
  arch.num_evaluation=20 \
  system.rollout_length=32 \
  system.actor_lr=1e-4 \
  system.exploration_noise=0.1 \
  logger.checkpointing.save_model=True \
  logger.checkpointing.save_args.checkpoint_uid=acrobot_db_memfree \
  "$@"
