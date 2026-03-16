#!/usr/bin/env bash
set -euo pipefail

BASE="uv run python baselines/single_agent_ppo.py \
  --restrict-actions \
  --num-envs 64 \
  --rollout-steps 128 \
  --total-steps 3000000 \
  --log-interval 10 \
  --use-wandb \
  --wandb-project factoriax-ppo"

$BASE --entropy-coef 0.01 --resource-density 0.08 --wandb-run-name "ent0.01-d0.08"
$BASE --entropy-coef 0.05 --resource-density 0.08 --wandb-run-name "ent0.05-d0.08"
$BASE --entropy-coef 0.10 --resource-density 0.08 --wandb-run-name "ent0.10-d0.08"
$BASE --entropy-coef 0.05 --resource-density 0.02 --wandb-run-name "ent0.05-d0.02"
$BASE --entropy-coef 0.05 --resource-density 0.15 --wandb-run-name "ent0.05-d0.15"
