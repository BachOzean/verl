#!/bin/bash
python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /data/home/scyb494/verl/checkpoints/grpo_subquestion_0914_22:27/global_step_400/actor \
    --target_dir /data/home/scyb494/verl/checkpoints/grpo_subquestion_0914_22:27/global_step_400/actor/huggingface