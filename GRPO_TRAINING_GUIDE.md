# GRPO训练脚本使用指南

## 概述

这个项目包含了基于您的自定义数据集进行GRPO（Group Relative Policy Optimization）训练的完整脚本，同时支持在MATH-500数据集上进行验证和测试。

## 数据集信息

您的数据集包含以下信息：
- **数据形状**: (55746, 6)
- **列名**: ['prompt', 'response', 'group_id', 'turn_index', 'stage', 'meta_json']

## 文件说明

### 1. `preprocess_custom_dataset.py`
数据预处理脚本，用于：
- 将您的数据集转换为GRPO训练所需的格式
- 划分训练集和验证集
- 数据清洗和统计分析
- 创建MATH-500占位符文件

### 2. `train_grpo_custom_dataset.sh`
GRPO训练脚本，配置了：
- 基于GRPO算法的训练设置
- 您的自定义数据集作为训练数据
- MATH-500数据集作为验证数据
- 适合的模型和超参数配置

## 使用步骤

### 步骤1: 数据预处理

假设您的数据集文件名为 `my_dataset.csv` 或 `my_dataset.parquet`：

```bash
# 进入项目目录
cd /data/home/scyb494/verl

# 运行数据预处理（如果是CSV文件）
python3 preprocess_custom_dataset.py \
    --input /path/to/your/my_dataset.csv \
    --output_dir /data/home/scyb494/verl/data \
    --test_size 0.1 \
    --create_math500_placeholder

# 如果是Parquet文件
python3 preprocess_custom_dataset.py \
    --input /path/to/your/my_dataset.parquet \
    --output_dir /data/home/scyb494/verl/data \
    --test_size 0.1 \
    --create_math500_placeholder
```

### 步骤2: 下载真实的MATH-500数据集（可选）

注意：预处理脚本会创建MATH-500的占位符文件。如果您需要真实的MATH-500数据集，请：

1. 下载MATH-500数据集
2. 将其转换为parquet格式，包含 `prompt` 和 `response` 列
3. 替换占位符文件：
   - `/data/home/scyb494/verl/data/math500/val.parquet`
   - `/data/home/scyb494/verl/data/math500/test.parquet`

### 步骤3: 修改训练脚本中的路径

编辑 `train_grpo_custom_dataset.sh`，确保以下路径正确：

```bash
# 检查并修改这些路径
CUSTOM_TRAIN_PATH="$HOME/data/custom_dataset_train.parquet"
CUSTOM_VAL_PATH="$HOME/data/custom_dataset_val.parquet"
MATH500_VAL_PATH="$HOME/data/math500/val.parquet"
MATH500_TEST_PATH="$HOME/data/math500/test.parquet"

# 模型路径（确保模型存在）
actor_rollout_ref.model.path=/data/home/scyb494/models/Qwen2.5-0.5B-Instruct
```

### 步骤4: 开始训练

```bash
# 运行训练脚本
bash train_grpo_custom_dataset.sh
```

## GRPO算法配置说明

本脚本使用了以下关键的GRPO配置：

- **`algorithm.adv_estimator=grpo`**: 使用GRPO算法
- **`actor_rollout_ref.rollout.n=5`**: 每个prompt生成5个响应进行组比较
- **`actor_rollout_ref.actor.use_kl_loss=True`**: 启用KL散度损失
- **`actor_rollout_ref.actor.kl_loss_coef=0.001`**: KL损失系数
- **`algorithm.use_kl_in_reward=False`**: 不在奖励中使用KL惩罚（因为使用了KL损失）

## 训练参数调整建议

根据您的硬件配置，可能需要调整以下参数：

### GPU内存较小时：
```bash
# 减小批次大小
data.train_batch_size=256
actor_rollout_ref.actor.ppo_mini_batch_size=64
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4

# 减少生成数量
actor_rollout_ref.rollout.n=3

# 降低GPU内存使用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.3
```

### GPU内存较大时：
```bash
# 增大批次大小
data.train_batch_size=1024
actor_rollout_ref.actor.ppo_mini_batch_size=256
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16

# 增加生成数量
actor_rollout_ref.rollout.n=8

# 提高GPU内存使用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.7
```

## 多GPU训练

如果您有多个GPU，可以修改以下参数：

```bash
# 设置GPU数量
trainer.n_gpus_per_node=4  # 根据实际GPU数量设置

# 调整tensor并行
actor_rollout_ref.rollout.tensor_model_parallel_size=2  # 根据需要设置
```

## 监控训练进程

训练过程中可以：

1. **查看实时日志**：
   ```bash
   tail -f {实验名称}_training.log
   ```

2. **使用SwanLab监控**：
   - 访问 SwanLab 界面查看训练指标
   - 日志保存在 `$HOME/swanlab`

3. **检查检查点**：
   ```bash
   ls -la $HOME/checkpoints/{实验名称}/
   ```

## 训练输出

训练完成后，您将获得：

- **模型检查点**: 保存在 `$HOME/checkpoints/{实验名称}/`
- **训练日志**: `{实验名称}_training.log`
- **SwanLab日志**: `$HOME/swanlab/`

## 常见问题

### 1. 内存不足 (OOM)
- 减小 `ppo_micro_batch_size_per_gpu`
- 降低 `gpu_memory_utilization`
- 减少 `rollout.n`

### 2. 训练速度慢
- 增大 `ppo_micro_batch_size_per_gpu`（在内存允许范围内）
- 使用多GPU训练
- 启用 `enable_gradient_checkpointing=False`（如果内存充足）

### 3. 数据格式错误
- 确保数据包含 `prompt` 和 `response` 列
- 检查数据是否有空值或空字符串
- 运行预处理脚本进行数据清洗

## 参考资料

- [GRPO算法论文](https://arxiv.org/pdf/2402.03300)
- [VERL官方文档](https://verl.readthedocs.io/)
- [GRPO训练示例](https://github.com/volcengine/verl/tree/main/examples/grpo_trainer)

## 技术支持

如果遇到问题，请检查：
1. 数据格式是否正确
2. 模型路径是否存在
3. GPU内存是否充足
4. Python环境和依赖是否正确安装

