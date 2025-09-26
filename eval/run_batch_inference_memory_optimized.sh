#!/bin/bash
# 内存优化版批推理启动脚本
# 专门针对大模型内存不足问题优化

PYTHON_PATH="/opt/anaconda3/envs/xny_verl/bin/python3"
SCRIPT_PATH="/home/ningmiao/ningyuan/verl/eval/batch_inference.py"

# 检查Python路径是否存在
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ Python解释器不存在: $PYTHON_PATH"
    echo "请确保已激活正确的conda环境"
    exit 1
fi

# 检查脚本路径是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ 脚本不存在: $SCRIPT_PATH"
    exit 1
fi

# 内存优化默认参数
MODEL=""
DATASET="open-r1/OpenR1-Math-220k"
SPLIT="train"
OUTPUT_DIR="/home/ningmiao/ningyuan/verl/eval/results/OpenR1-Math-220k"
NUM_SAMPLES=16
BATCH_SIZE=8
NUM_GPUS=2
GPU_MEMORY=0.6

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu_memory)
            GPU_MEMORY="$2"
            shift 2
            ;;
        *)
            echo "❌ 未知参数: $1"
            echo "用法: $0 --model <model_path> [其他参数...]"
            exit 1
            ;;
    esac
done

# 检查是否提供了模型路径
if [ -z "$MODEL" ]; then
    echo "❌ 必须提供模型路径"
    echo "用法: $0 --model <model_path> [其他参数...]"
    echo ""
    echo "可选参数（内存优化版默认值）:"
    echo "  --dataset <dataset> (默认: open-r1/OpenR1-Math-220k)"
    echo "  --split <split> (默认: train)"
    echo "  --output_dir <dir> (默认: /home/ningmiao/ningyuan/verl/eval/results/OpenR1-Math-220k)"
    echo "  --num_samples <n> (默认: 16)"
    echo "  --batch_size <n> (默认: 8)"
    echo "  --num_gpus <n> (默认: 2)"
    echo "  --gpu_memory <fraction> (默认: 0.6)"
    exit 1
fi

echo "🚀 启动内存优化版批推理..."
echo "  模型: $MODEL"
echo "  数据集: $DATASET"
echo "  输出目录: $OUTPUT_DIR"
echo "  样本数: $NUM_SAMPLES"
echo "  批大小: $BATCH_SIZE"
echo "  GPU数量: $NUM_GPUS"
echo "  GPU内存利用率: $GPU_MEMORY"
echo ""
echo "💡 内存优化配置:"
echo "  - 减少上下文长度到2048"
echo "  - 减少生成长度到512"
echo "  - 禁用束搜索"
echo "  - 禁用日志统计"

# 运行批推理脚本
exec $PYTHON_PATH $SCRIPT_PATH \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --num_gpus "$NUM_GPUS" \
    --gpu_memory_utilization "$GPU_MEMORY"
