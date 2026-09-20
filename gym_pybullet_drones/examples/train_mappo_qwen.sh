#!/usr/bin/env bash
set -e

# Edit these values before starting training.
MODEL_PATH="/mnt/nfs/jcliu/wp/models/Qwen--Qwen3.5-0.8B/snapshots/master"
DEVICE="cuda:0"
TOTAL_TIMESTEPS=100000
ROLLOUT_STEPS=512
BATCH_SIZE=512
EPOCHS=5
# 设为 0 时跳过评估，只保存 final_model。
EVAL_FREQ=0
UPDATE_MICROBATCH_STEPS=24
BACKBONE_DTYPE="bfloat16"
RUN_NAME="mappo_qwen3.5_0.8b"
OUTPUT_FOLDER="results/$RUN_NAME"
SWANLAB_PROJECT="gym-pybullet-drones"
SWANLAB_WORKSPACE=""
SWANLAB_MODE="online"

cd "$(dirname "$0")/../.."

python -m gym_pybullet_drones.examples.learn \
  --multiagent true \
  --actor_type qwen \
  --model_path "$MODEL_PATH" \
  --device "$DEVICE" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --rollout_steps "$ROLLOUT_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --eval_freq "$EVAL_FREQ" \
  --update_microbatch_steps "$UPDATE_MICROBATCH_STEPS" \
  --backbone_dtype "$BACKBONE_DTYPE" \
  --output_folder "$OUTPUT_FOLDER" \
  --swanlab_project "$SWANLAB_PROJECT" \
  --swanlab_workspace "$SWANLAB_WORKSPACE" \
  --swanlab_mode "$SWANLAB_MODE" \
  --gui false \
  --plot false
