#!/bin/bash

# GRPO训练快速启动脚本
# 使用此脚本可以快速开始训练过程

echo "=========================================="
echo "GRPO训练快速启动脚本"
echo "=========================================="

# 检查是否提供了数据文件参数
if [ $# -eq 0 ]; then
    echo "用法: bash quick_start.sh <数据文件路径>"
    echo "示例: bash quick_start.sh /path/to/your/dataset.csv"
    echo "      bash quick_start.sh /path/to/your/dataset.parquet"
    exit 1
fi

INPUT_FILE="$1"
PROJECT_DIR="/data/home/scyb494/verl"
DATA_DIR="$PROJECT_DIR/data"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 数据文件不存在: $INPUT_FILE"
    exit 1
fi

echo "输入数据文件: $INPUT_FILE"
echo "项目目录: $PROJECT_DIR"
echo "数据输出目录: $DATA_DIR"

# 步骤1: 数据预处理
echo ""
echo "步骤1: 开始数据预处理..."
cd "$PROJECT_DIR"

python3 preprocess_custom_dataset.py \
    --input "$INPUT_FILE" \
    --output_dir "$DATA_DIR" \
    --test_size 0.1 \
    --create_math500_placeholder

if [ $? -ne 0 ]; then
    echo "错误: 数据预处理失败"
    exit 1
fi

echo "数据预处理完成！"

# 步骤2: 检查必要文件
echo ""
echo "步骤2: 检查训练文件..."

TRAIN_FILE="$DATA_DIR/custom_dataset_train.parquet"
VAL_FILE="$DATA_DIR/custom_dataset_val.parquet"
MATH500_VAL="$DATA_DIR/math500/val.parquet"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "错误: 训练文件不存在: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "错误: 验证文件不存在: $VAL_FILE"
    exit 1
fi

if [ ! -f "$MATH500_VAL" ]; then
    echo "错误: MATH-500验证文件不存在: $MATH500_VAL"
    exit 1
fi

echo "所有必要文件检查通过！"

# 步骤3: 检查模型
echo ""
echo "步骤3: 检查模型文件..."
MODEL_PATH="/data/home/scyb494/models/Qwen2.5-0.5B-Instruct"

if [ ! -d "$MODEL_PATH" ]; then
    echo "警告: 模型目录不存在: $MODEL_PATH"
    echo "请确保已下载模型，或修改train_grpo_custom_dataset.sh中的模型路径"
    echo ""
    echo "建议的模型下载命令:"
    echo "huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir $MODEL_PATH"
    echo ""
    read -p "是否继续训练？(y/N): " continue_training
    if [[ ! $continue_training =~ ^[Yy]$ ]]; then
        echo "训练已取消"
        exit 1
    fi
fi

# 步骤4: 开始训练
echo ""
echo "步骤4: 开始GRPO训练..."
echo "注意: 训练可能需要很长时间，请确保有足够的GPU资源"
echo ""

read -p "是否开始训练？(y/N): " start_training
if [[ ! $start_training =~ ^[Yy]$ ]]; then
    echo "训练已取消"
    echo ""
    echo "如需手动开始训练，请运行:"
    echo "bash train_grpo_custom_dataset.sh"
    exit 0
fi

# 运行训练脚本
echo "开始训练..."
bash train_grpo_custom_dataset.sh

echo ""
echo "=========================================="
echo "快速启动脚本完成"
echo "=========================================="

