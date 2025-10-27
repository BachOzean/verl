#!/bin/bash
python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_20/actor \
    --target_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_20/actor/huggingface

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_40/actor \
    --target_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_40/actor/huggingface

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_60/actor \
    --target_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_60/actor/huggingface

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_80/actor \
    --target_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_80/actor/huggingface

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_120/actor \
    --target_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_120/actor/huggingface

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_140/actor \
    --target_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_140/actor/huggingface

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_160/actor \
    --target_dir /home/ningmiao/ningyuan/verl/checkpoints/grpo_decompostion_1015_02:13/global_step_160/actor/huggingface