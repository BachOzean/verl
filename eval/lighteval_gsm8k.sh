#!/bin/bash
export HYDRA_FULL_ERROR=1

module load cuda/12.4

MODEL_PATH="/data/home/scyb494/verl/checkpoints/grpo_subquestion_0910_10:48/global_step_20/actor_merged_hf"
OUTPUT_DIR="/data/home/scyb494/verl/eval"

lighteval "vllm" \
    "/data/home/scyb494/verl/eval/model_config/vllm_model_config.yaml" \
    "lighteval|gsm8k|3|1" \
    "--output-dir=${OUTPUT_DIR}"\
    --save-details

# lighteval "vllm" \
#     "/data/home/scyb494/verl/eval/model_config/vllm_base_model_config.yaml" \
#     "lighteval|gsm8k|3|1" \
#     "--output-dir=${OUTPUT_DIR}"
