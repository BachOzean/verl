#!/bin/bash
set -euo pipefail
sbatch -t 04:00:00 --gpus=1 /data/home/scyb494/verl/eval/sbatch/batch_eval_step_100.sh
sbatch -t 04:00:00 --gpus=1 /data/home/scyb494/verl/eval/sbatch/batch_eval_step_200.sh
sbatch -t 04:00:00 --gpus=1 /data/home/scyb494/verl/eval/sbatch/batch_eval_step_300.sh
sbatch -t 04:00:00 --gpus=1 /data/home/scyb494/verl/eval/sbatch/batch_eval_step_400.sh