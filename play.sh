#!/bin/sh
set -x
DATE=$(date +%m%d)
TIME_TAG=$(date +%H:%M)
EXP_NAME="verl_demo"
unset ROCR_VISIBLE_DEVICES
# ------------------------------------------------------------------------------------------------
export HF_ENDPOINT=https://hf-mirror.com
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS

export SWANLAB_API_KEY=2a0frfSlXimfedHjAiNa8
export SWANLAB_LOG_DIR=$HOME/swanlab
export SWANLAB_MODE=local

HOME=/data/home/scyb494/verl

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=/data/home/scyb494/models/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=/data/home/scyb494/models/Qwen2.5-0.5B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.logger='["console"]' \
    trainer.project_name=play \
    trainer.experiment_name=verl_demo \
    trainer.default_local_dir=$HOME/checkpoints/$EXP_NAME \
    trainer.log_freq=1 \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log \