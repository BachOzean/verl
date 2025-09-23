#!/bin/bash
set -x
# 设置路径
HOME=/data/home/scyb494/verl

# 获取日期和时间标签
DATE=$(date +%m%d)
TIME_TAG=$(date +%H:%M)
EXP_NAME="grpo_subquestion_${DATE}_${TIME_TAG}"

# 环境配置
unset ROCR_VISIBLE_DEVICES
export HF_ENDPOINT=https://hf-mirror.com
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export HYDRA_FULL_ERROR=1

# 设置XFormers后端避免CUDA错误
export VLLM_ATTENTION_BACKEND=XFORMERS

# SwanLab配置
# export SWANLAB_API_KEY=2a0frfSlXimfedHjAiNa8
export SWANLAB_LOG_DIR=$HOME/checkpoints/$EXP_NAME
export SWANLAB_MODE=local


# 定义数据集路径
# 用户自定义数据集路径（请根据实际路径修改）
CUSTOM_TRAIN_PATH="/data/home/scyb494/Hybrid-FT/grpo_prompt_subq_progressive.parquet"
# CUSTOM_VAL_PATH="$HOME/data/custom_dataset_val.parquet"

# MATH-500数据集路径（用于验证和测试）
MATH500_VAL_PATH="/data/home/scyb494/verl/data/MATH-500/test.parquet"
MATH500_TEST_PATH="/data/home/scyb494/verl/data/MATH-500/test.parquet"

# 组合验证文件
VAL_FILES="['$MATH500_VAL_PATH']"

echo "开始GRPO训练..."
echo "训练数据: $CUSTOM_TRAIN_PATH"
echo "验证数据: $VAL_FILES"
echo "实验名称: $EXP_NAME"

    # data.val_files="$MATH500_VAL_PATH" \
        # trainer.test_freq=0 \


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$CUSTOM_TRAIN_PATH" \
    data.val_files="$MATH500_VAL_PATH" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path=/data/home/scyb494/models/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=42 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=4 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name="grpo_subquestion_dataset" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$HOME/checkpoints/$EXP_NAME" \
    trainer.total_epochs=3 \
    2>&1 | tee "${EXP_NAME}_training.log"

echo "训练完成！检查点保存在: $HOME/checkpoints/$EXP_NAME"
echo "训练日志保存在: ${EXP_NAME}_training.log"
