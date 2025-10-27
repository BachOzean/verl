#!/bin/bash
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT="https://hf-mirror.com"

# 批推理启动脚本
# HOME="/data/home/scyb494"
HOME="/home/ningmiao/ningyuan"

SCRIPT_PATH="$HOME/verl/eval/batch_inference.py"

# 检查脚本路径是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ 脚本不存在: $SCRIPT_PATH"
    exit 1
fi

# 默认参数
MODEL="$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET="/home/ningmiao/ningyuan/verl/data/OpenR1-Math-220k/data"
SPLIT="train"
OUTPUT_DIR="$HOME/verl/eval/results/OpenR1-Math-220k"
NUM_SAMPLES=16
SAMPLES_PER_CALL=4
BATCH_SIZE=4
NUM_GPUS=8
GPU_MEMORY=0.75
# 新增：控制长度、精度与并发
MAX_MODEL_LEN=8192
MAX_TOKENS=4096
DTYPE="bfloat16"   # 可选: auto|bfloat16|float16|float32
MAX_NUM_SEQS=1

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
        --samples_per_call)
            SAMPLES_PER_CALL="$2"
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
        --max_model_len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --max_num_seqs)
            MAX_NUM_SEQS="$2"
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
    echo "可选参数:"
    echo "  --dataset <dataset> (默认: open-r1/OpenR1-Math-220k)"
    echo "  --split <split> (默认: train)"
    echo "  --output_dir <dir> (默认: /home/ningmiao/ningyuan/verl/eval/results/OpenR1-Math-220k)"
    echo "  --num_samples <n> (默认: 64)"
    echo "  --batch_size <n> (默认: 32)"
    echo "  --num_gpus <n> (默认: 8)"
    echo "  --gpu_memory <fraction> (默认: 0.75)"
    echo "  --max_model_len <n> (默认: 4096)"
    echo "  --max_tokens <n> (默认: 1024)"
    echo "  --dtype <auto|bfloat16|float16|float32> (默认: bfloat16)"
    echo "  --max_num_seqs <n> (默认: 1)"
    exit 1
fi

echo "🚀 启动批推理..."
echo "  模型: $MODEL"
echo "  数据集: $DATASET"
echo "  输出目录: $OUTPUT_DIR"
echo "  样本数: $NUM_SAMPLES"
echo "  每次调用采样: $SAMPLES_PER_CALL"
echo "  批大小: $BATCH_SIZE"
echo "  GPU数量: $NUM_GPUS"
echo "  GPU内存利用率: $GPU_MEMORY"
echo "  上下文长度: $MAX_MODEL_LEN"
echo "  生成长度: $MAX_TOKENS"
echo "  精度: $DTYPE"
echo "  并发序列上限: $MAX_NUM_SEQS"

# 运行批推理脚本
exec python $SCRIPT_PATH \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --samples_per_call "$SAMPLES_PER_CALL" \
    --batch_size "$BATCH_SIZE" \
    --num_gpus "$NUM_GPUS" \
    --gpu_memory_utilization "$GPU_MEMORY" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_tokens "$MAX_TOKENS" \
    --dtype "$DTYPE" \
    --max_num_seqs "$MAX_NUM_SEQS"
