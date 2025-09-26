#!/bin/bash
set -x

pip install datasets==4.0.0
# 路径前缀
HOME=/home/ningmiao/ningyuan/verl

# 时间标签
DATE=$(date +%m%d)
TIME_TAG=$(date +%H:%M)
EXP_NAME="grpo_steps_enum_${DATE}_${TIME_TAG}"

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_ENDPOINT=https://hf-mirror.com
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# SwanLab
export SWANLAB_LOG_DIR=$HOME/checkpoints/$EXP_NAME
export SWANLAB_MODE=local

# 数据路径（使用我们生成的 steps 构造数据）
TRAIN_PARQUET="/home/ningmiao/ningyuan/verl/data/OpenR1-Math-220k/grpo_steps_head50k.parquet"
# 验证仍用占位 MATH-500（可替换为真实验证集）
MATH500_VAL_PATH="/home/ningmiao/ningyuan/verl/data/MATH-500/test.parquet"

echo "开始GRPO训练(steps-enum)..."
echo "训练数据: $TRAIN_PARQUET"
echo "验证数据: $MATH500_VAL_PATH"
echo "实验名称: $EXP_NAME"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$MATH500_VAL_PATH" \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path=/home/ningmiao/ningyuan/models/DeepSeek-R1-Distill-Qwen-1.5B \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name="grpo_steps_enum" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$HOME/checkpoints/$EXP_NAME" \
    trainer.total_epochs=3 \
    2>&1 | tee "${EXP_NAME}_training.log"

echo "训练完成！检查点保存在: $HOME/checkpoints/$EXP_NAME"
echo "训练日志保存在: ${EXP_NAME}_training.log"
