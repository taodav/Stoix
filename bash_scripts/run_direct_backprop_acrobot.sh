#!/bin/bash
set -euo pipefail
PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" stoix/systems/direct_backprop/ff_direct_backprop.py \
  env=differentiable/acrobot \
  arch.total_num_envs=64 \
  arch.total_timesteps=5e5 \
  arch.num_evaluation=20 \
  system.rollout_length=32 \
  system.actor_lr=1e-4 \
  system.exploration_noise=0.1 \
  "$@"
