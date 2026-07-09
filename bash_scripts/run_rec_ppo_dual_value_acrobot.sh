#!/bin/bash
#
# Recurrent PPO (GRU) on Acrobot-v1 WITH an auxiliary non-recurrent value head.
#
# Same dual-value setup as run_rec_ppo_dual_value_cartpole.sh, but on Acrobot and
# with a larger GRU (hidden_size = 64). Acrobot is fully observable but has richer
# dynamics than CartPole (6-D observation, momentum that accumulates over time),
# making it a test of whether recurrence helps the value function even without
# partial observability.
#
# Setup: 500k steps, 4 envs, 50 checkpoints => a checkpoint every ~9,728 steps,
# so checkpoint indices line up with the CartPole run (ckpt k = 9728 * k).
#
# Acrobot notes:
#   * Reward is -1 per step until the goal, so returns/values are NEGATIVE
#     (roughly -500 for a failing policy up to ~-70 for a good one).
#   * Episodes terminate when the goal is reached and cap at 500 steps.
#   * 3 discrete actions (the CategoricalHead adapts automatically).
#
# Checkpoints are namespaced by a stable UID (acrobot_h64) rather than a
# timestamp, so they land in a dedicated folder and don't collide with other runs:
#   ./checkpoints/rec_ppo_dual_value/acrobot_h64/<step>/
#
# NOTE on minibatches: with only 4 envs the recurrent batch has 4 sequences, so
# num_minibatches must divide 4. We use 1 (full-batch); valid options are 1, 2, 4.

set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" stoix/systems/ppo/anakin/rec_ppo_dual_value.py \
  env=gymnax/acrobot \
  network=rnn \
  network.actor_network.rnn_layer.hidden_state_dim=64 \
  network.critic_network.rnn_layer.hidden_state_dim=64 \
  system=ppo/rec_ppo_dual_value \
  arch.total_num_envs=4 \
  arch.total_timesteps=5e5 \
  arch.num_evaluation=50 \
  system.num_minibatches=1 \
  logger.checkpointing.save_model=True \
  logger.checkpointing.save_args.checkpoint_uid=acrobot_h64 \
  logger.checkpointing.save_args.save_interval_steps=1 \
  logger.checkpointing.save_args.max_to_keep=50 \
  "$@"

# Each checkpoint's learner_state.params contains actor_params, critic_params
# (recurrent value) and nr_critic_params (non-recurrent value), all with GRU
# hidden_size = 64. The analysis script auto-reads the hidden size and gamma from
# the checkpoint metadata, so no analysis-side changes are needed.
