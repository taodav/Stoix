#!/bin/bash
#
# Recurrent PPO (GRU) on CartPole-v1 WITH an auxiliary non-recurrent value head.
#
# Same setup as run_rec_ppo_cartpole.sh (500k steps, 4 envs, GRU hidden=32,
# 50 checkpoints => a checkpoint roughly every ~10k steps), so checkpoint indices
# line up: checkpoint 5 == u, checkpoint 45 == u'.
#
# In addition to the recurrent actor + recurrent critic, this trains a SECOND
# value function with the identical architecture but its hidden state zeroed at
# every step (so it is non-recurrent / a pure function of the current obs). It
# sees the same data stream, is trained with its own independent GAE bootstrap,
# and never affects the policy. This lets us compare recurrent vs. non-recurrent
# "value drift" between two checkpoints.
#
# NOTE on minibatches: with only 4 envs the recurrent batch has 4 sequences, so
# num_minibatches must divide 4. We use 1 (full-batch); valid options are 1, 2, 4.

set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" stoix/systems/ppo/anakin/rec_ppo_dual_value.py \
  env=gymnax/cartpole \
  network=rnn \
  network.actor_network.rnn_layer.hidden_state_dim=32 \
  network.critic_network.rnn_layer.hidden_state_dim=32 \
  system=ppo/rec_ppo_dual_value \
  arch.total_num_envs=4 \
  arch.total_timesteps=5e5 \
  arch.num_evaluation=50 \
  system.num_minibatches=1 \
  logger.checkpointing.save_model=True \
  logger.checkpointing.save_args.save_interval_steps=1 \
  logger.checkpointing.save_args.max_to_keep=50 \
  "$@"

# Checkpoints are written to:  ./checkpoints/rec_ppo_dual_value/<timestamp>/<step>/
# Each checkpoint's learner_state.params contains: actor_params, critic_params
# (recurrent value), and nr_critic_params (non-recurrent value).
