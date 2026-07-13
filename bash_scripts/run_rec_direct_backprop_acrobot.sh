#!/bin/bash
# Recurrent (GRU) Direct Backprop on Acrobot — memory PERSISTS across steps.
set -euo pipefail
PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" stoix/systems/direct_backprop/rec_direct_backprop.py \
  env=differentiable/acrobot \
  arch.total_num_envs=32 \
  arch.total_timesteps=5e5 \
  arch.num_evaluation=20 \
  system.rollout_length=32 \
  system.actor_lr=3e-4 \
  system.exploration_noise=0.05 \
  system.reset_hidden_state_every_step=False \
  "$@"
