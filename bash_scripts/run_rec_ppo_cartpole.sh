#!/bin/bash
#
# Recurrent PPO (GRU) on CartPole-v1 — Anakin (fully-JAX, end-to-end) system.
#
# Spec:
#   * 500k environment timesteps
#   * 4 parallel (vectorised) environments
#   * a checkpoint roughly every ~10k steps (coupled to evaluation cadence)
#
# In Stoix's Anakin loop, a checkpoint is written once per evaluation, so
# `arch.num_evaluation` controls checkpoint cadence:
#     checkpoint cadence (steps) ~= total_timesteps / num_evaluation
#     500_000 / 50 = 10_000  ->  num_evaluation=50
#
# With 1 device, update_batch_size=1, num_envs=4 and rollout_length=128, the
# per-eval step count is a multiple of rollout_length*num_envs = 512, so the
# actual cadence lands on 9_728 steps (19 updates/eval) and 50 checkpoints are
# written (~486k actual steps; the checker prints the small rounding delta).
#
# NOTE on minibatches: with only 4 envs the recurrent batch has 4 sequences,
# and `num_minibatches` must divide (num_envs * num_recurrent_chunks) = 4.
# The rec_ppo default of 32 would crash here, so we set it to 1 (full-batch).
# Valid alternatives with 4 envs are 1, 2, or 4.
#
# Uses ACTION CONCATENATION (env=gymnax/cartpole_action_concat): the previous
# action (one-hot, zero at t=0) is appended to each observation. Both the
# recurrent actor and recurrent critic are trajectory-processing networks, so
# both see it (no stripping needed here -- this base system has no Markov head).

set -euo pipefail

# Use the project's uv-managed virtualenv interpreter.
PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" stoix/systems/ppo/anakin/rec_ppo.py \
  env=gymnax/cartpole_action_concat \
  network=rnn \
  network.actor_network.rnn_layer.hidden_state_dim=32 \
  network.critic_network.rnn_layer.hidden_state_dim=32 \
  system=ppo/rec_ppo \
  arch.total_num_envs=4 \
  arch.total_timesteps=5e5 \
  arch.num_evaluation=50 \
  system.num_minibatches=1 \
  logger.checkpointing.save_model=True \
  logger.checkpointing.save_args.save_interval_steps=1 \
  logger.checkpointing.save_args.max_to_keep=50 \
  "$@"

# Checkpoints are written to:  ./checkpoints/rec_ppo/<timestamp>/<step>/
# To resume/load later, re-run with:
#   logger.checkpointing.load_model=True \
#   logger.checkpointing.load_args.checkpoint_uid=<timestamp>
# (Pin a stable name at save time with logger.checkpointing.save_args.checkpoint_uid=my_run)
