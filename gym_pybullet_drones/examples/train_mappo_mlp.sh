#!/usr/bin/env bash
set -e

# Edit these values before starting training.
DEVICE="cuda:0"
TOTAL_TIMESTEPS=100000
ROLLOUT_STEPS=512
BATCH_SIZE=512
EPOCHS=5
ACTOR_LEARNING_RATE=1e-4
CRITIC_LEARNING_RATE=3e-4
# 设为 0 时跳过评估，只保存 final_model。
EVAL_FREQ=0
RUN_NAME="mappo_mlp"
OUTPUT_FOLDER="results/$RUN_NAME"
SWANLAB_PROJECT="gym-pybullet-drones"
SWANLAB_WORKSPACE=""
SWANLAB_MODE="online"

cd "$(dirname "$0")/../.."

python -m gym_pybullet_drones.examples.learn \
  --multiagent true \
  --actor_type mlp \
  --device "$DEVICE" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --rollout_steps "$ROLLOUT_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --actor_learning_rate "$ACTOR_LEARNING_RATE" \
  --critic_learning_rate "$CRITIC_LEARNING_RATE" \
  --eval_freq "$EVAL_FREQ" \
  --output_folder "$OUTPUT_FOLDER" \
  --swanlab_project "$SWANLAB_PROJECT" \
  --swanlab_workspace "$SWANLAB_WORKSPACE" \
  --swanlab_mode "$SWANLAB_MODE" \
  --gui false \
  --plot false
