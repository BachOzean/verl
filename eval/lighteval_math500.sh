#!/bin/bash

HOME=/home/ningmiao/ningyuan/verl

cd $HOME/eval

# export HYDRA_FULL_ERROR=1
# export HF_ENDPOINT=https://hf-mirror.com

pip install datasets==3.6.0

OUTPUT_DIR=./results/math500

MODEL_ARGS="$HOME/eval/model_config/vllm_base_model_config.yaml"

TASK=math_500

lighteval vllm $MODEL_ARGS \
    --custom-tasks ./custom_tasks/eval_math500.py \
    "custom|$TASK|0|0" \
    --output-dir $OUTPUT_DIR \
    --save-details
