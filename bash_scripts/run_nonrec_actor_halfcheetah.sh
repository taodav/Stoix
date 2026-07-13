#!/bin/bash
#
# Non-recurrent (reactive) ACTOR + recurrent/non-recurrent dual VALUE heads on
# Brax HalfCheetah — CONTINUOUS control.
#
# Continuous-action counterpart of run_nonrec_actor_acrobot.sh. The actor's hidden
# state is zeroed every step (reactive / Markov policy), while the recurrent critic
# retains memory. HalfCheetah is fully observable (17-D obs, 6-D continuous
# action), so this extends the "does a recurrent value spuriously use history on a
# fully-observable, Markov task?" question to continuous control.
#
# Setup: 5M steps, 4 envs, GRU hidden 256, 50 checkpoints (~99,840 steps apart),
# tanh-squashed Normal action head. Uses lr 3e-4 (continuous-PPO default).
# (HalfCheetah needs far more than 500k steps to learn; 5M is a realistic budget.)
#
# Uses ACTION CONCATENATION (env=brax/halfcheetah_action_concat): the previous
# 6-D action is appended to each observation. Only the RECURRENT critic sees it
# (obs 17 -> 23); the reactive actor and memory-free critic strip it (stay Markov).
#
# Checkpoints -> ./checkpoints/rec_ppo_nonrec_actor_continuous/halfcheetah_nonrec_actor_ac_h256/
#
# NOTE on minibatches: with only 4 envs the recurrent batch has 4 sequences, so
# num_minibatches must divide 4. We use 1 (full-batch); valid options are 1, 2, 4.

set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" stoix/systems/ppo/anakin/rec_ppo_nonrec_actor_continuous.py \
  env=brax/halfcheetah_action_concat \
  network=rnn_continuous \
  network.actor_network.rnn_layer.hidden_state_dim=256 \
  network.critic_network.rnn_layer.hidden_state_dim=256 \
  system=ppo/rec_ppo_nonrec_actor_continuous \
  arch.total_num_envs=4 \
  arch.total_timesteps=5e6 \
  arch.num_evaluation=50 \
  system.num_minibatches=1 \
  logger.checkpointing.save_model=True \
  logger.checkpointing.save_args.checkpoint_uid=halfcheetah_nonrec_actor_ac_h256 \
  logger.checkpointing.save_args.save_interval_steps=1 \
  logger.checkpointing.save_args.max_to_keep=50 \
  "$@"
