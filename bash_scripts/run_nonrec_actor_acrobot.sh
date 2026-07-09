#!/bin/bash
#
# Non-recurrent ACTOR + recurrent/non-recurrent dual VALUE heads on Acrobot-v1.
#
# The actor's hidden state is zeroed every step (reset flag always True), making
# the policy purely reactive / Markov. The recurrent critic still carries memory.
# This isolates whether a recurrent value uses history when the policy itself
# cannot use history (i.e. V^pi is provably Markov on this fully-observable task).
#
# Same settings as the dual-value Acrobot run so results are comparable:
#   500k steps, 4 envs, GRU hidden 64, 50 checkpoints (every ~9,728 steps).
#
# Checkpoints land in: ./checkpoints/rec_ppo_nonrec_actor/acrobot_nonrec_actor_h64/

set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" stoix/systems/ppo/anakin/rec_ppo_nonrec_actor.py \
  env=gymnax/acrobot \
  network=rnn \
  network.actor_network.rnn_layer.hidden_state_dim=64 \
  network.critic_network.rnn_layer.hidden_state_dim=64 \
  system=ppo/rec_ppo_nonrec_actor \
  arch.total_num_envs=4 \
  arch.total_timesteps=5e5 \
  arch.num_evaluation=50 \
  system.num_minibatches=1 \
  logger.checkpointing.save_model=True \
  logger.checkpointing.save_args.checkpoint_uid=acrobot_nonrec_actor_h64 \
  logger.checkpointing.save_args.save_interval_steps=1 \
  logger.checkpointing.save_args.max_to_keep=50 \
  "$@"
