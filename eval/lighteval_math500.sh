#!/bin/bash
cd /data/home/scyb494/verl/eval

export HYDRA_FULL_ERROR=1

module load cuda/12.4

export HF_ENDPOINT=https://hf-mirror.com

OUTPUT_DIR=./results/math500

MODEL_ARGS="/data/home/scyb494/verl/eval/model_config/vllm_base_model_config.yaml"

TASK=math_500

    # --use-chat-template \

lighteval vllm $MODEL_ARGS \
    --custom-tasks ./custom_tasks/eval_math500.py \
    "custom|$TASK|0|0" \
    --output-dir $OUTPUT_DIR \
    --save-details
