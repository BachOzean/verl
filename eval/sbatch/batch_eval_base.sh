#!/bin/bash
#sbatch -J eval_step_100
#sbatch -o /data/home/scyb494/verl/eval/slurm-step_100-%j.out
#sbatch -t 04:00:00
#sbatch --gpus=1

set -euo pipefail
cd /data/home/scyb494/verl/eval
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=https://hf-mirror.com
module load cuda/12.4

lighteval vllm "/data/home/scyb494/verl/eval/tmp_model_configs/base_model.yaml" "lighteval|gsm8k|3|1" --output-dir=/data/home/scyb494/verl/eval/results/gsm8k --save-details
lighteval vllm "/data/home/scyb494/verl/eval/tmp_model_configs/base_model.yaml" --custom-tasks custom_tasks/eval_math500.py "custom|math_500|0|0" --output-dir=/data/home/scyb494/verl/eval/results/math500 --save-details
lighteval vllm "/data/home/scyb494/verl/eval/tmp_model_configs/base_model.yaml" --custom-tasks custom_tasks/eval_aime.py "custom|aime24|0|0" --output-dir=/data/home/scyb494/verl/eval/results/aime24 --save-details
