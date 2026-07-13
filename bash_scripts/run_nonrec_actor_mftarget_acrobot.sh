#!/bin/bash
#
# Non-recurrent (reactive) ACTOR + recurrent critic trained on MEMORY-FREE TARGETS,
# on Acrobot-v1. Isolates the bootstrap-history phenomenon.
#
# Identical to run_nonrec_actor_acrobot.sh EXCEPT the recurrent critic regresses
# toward the memory-free value head's GAE targets (nr_targets) instead of its own
# history-conditioned bootstrap. If the recurrent critic's spurious history-use
# (seen in the reactive-actor run) collapses here, the self-reinforcing bootstrap
# loop was the cause; if it persists, something else (raw BPTT capacity) is.
#
# Same setup for comparability: 500k steps, 4 envs, GRU hidden 64, 50 checkpoints
# (~9,728 steps apart, so checkpoint indices line up with the other Acrobot runs).
#
# Uses ACTION CONCATENATION (env=gymnax/acrobot_action_concat): only the recurrent
# critic sees the appended previous action; the reactive actor and memory-free
# critic strip it (stay Markov).
#
# Checkpoints -> ./checkpoints/rec_ppo_nonrec_actor_mftarget/acrobot_mftarget_ac_h64/

set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" stoix/systems/ppo/anakin/rec_ppo_nonrec_actor_mftarget.py \
  env=gymnax/acrobot_action_concat \
  network=rnn \
  network.actor_network.rnn_layer.hidden_state_dim=64 \
  network.critic_network.rnn_layer.hidden_state_dim=64 \
  system=ppo/rec_ppo_nonrec_actor_mftarget \
  arch.total_num_envs=4 \
  arch.total_timesteps=5e5 \
  arch.num_evaluation=50 \
  system.num_minibatches=1 \
  logger.checkpointing.save_model=True \
  logger.checkpointing.save_args.checkpoint_uid=acrobot_mftarget_ac_h64 \
  logger.checkpointing.save_args.save_interval_steps=1 \
  logger.checkpointing.save_args.max_to_keep=50 \
  "$@"
