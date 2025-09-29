#!/bin/bash
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT="https://hf-mirror.com"

# 4x H800 启动脚本（包装现有 run_batch_inference.sh），提供更稳妥的默认值
HOME="/data/home/scyb494"
RUN_SCRIPT="$HOME/verl/eval/run_batch_inference.sh"

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "❌ 启动脚本不存在: $RUN_SCRIPT"
    exit 1
fi

# 针对 4x H800 的推荐默认参数（可被命令行参数覆盖）
MODEL="$HOME/models/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET="/data/home/scyb494/.cache/huggingface/hub/datasets--open-r1--OpenR1-Math-220k/snapshots/e4e141ec9dea9f8326f4d347be56105859b2bd68/data"
SPLIT="train"
OUTPUT_DIR="$HOME/verl/eval/results/OpenR1-Math-220k_H800_4x"
NUM_SAMPLES=64
SAMPLES_PER_CALL=8
BATCH_SIZE=16
NUM_GPUS=4
GPU_MEMORY=0.90
MAX_MODEL_LEN=8192
MAX_TOKENS=4096
DTYPE="bfloat16"   # 可选: auto|bfloat16|float16|float32
MAX_NUM_SEQS=1

# 限制仅使用前 4 张卡（如无需限制可注释掉）
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "🚀 4x H800 启动 vLLM Server + 客户端批推理..."
echo "  模型: $MODEL"
echo "  数据集: $DATASET"
echo "  输出目录: $OUTPUT_DIR"
echo "  样本数: $NUM_SAMPLES"
echo "  每次调用采样: $SAMPLES_PER_CALL"
echo "  并发(客户端线程): $BATCH_SIZE"
echo "  GPU数量: $NUM_GPUS"
echo "  GPU内存利用率: $GPU_MEMORY"
echo "  上下文长度: $MAX_MODEL_LEN"
echo "  生成长度: $MAX_TOKENS"
echo "  精度: $DTYPE"

VLLM_PORT=${VLLM_PORT:-8000}
SERVER_LOG="$OUTPUT_DIR/server.log"
mkdir -p "$OUTPUT_DIR"

# 启动 vLLM Server（TP=4）
echo "🟢 启动 vLLM Server (tensor-parallel-size=4) on port $VLLM_PORT"
CMD="python -m vllm.entrypoints.openai.api_server --model '$MODEL' --tensor-parallel-size 4 --dtype $DTYPE --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEMORY --trust-remote-code --port $VLLM_PORT"
echo "$CMD" > "$OUTPUT_DIR/launch_server.sh"
nohup bash -lc "$CMD" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "vLLM Server PID: $SERVER_PID"

# 等待服务就绪
if ! command -v curl >/dev/null 2>&1; then
  echo "⚠️ 未检测到 curl，将直接等待 10 秒"
  sleep 10
else
  echo -n "⏳ 等待 vLLM Server 就绪"
  for i in {1..120}; do
      sleep 1
      if curl -s "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null; then
          echo " - OK"
          break
      fi
      echo -n "."
  done
fi

# 运行客户端聚合（model name 交由客户端自动解析）
echo "🧩 运行客户端聚合: batch_inference_client.py"
python "$HOME/verl/eval/batch_inference_client.py" \
    --server_url "http://127.0.0.1:$VLLM_PORT" \
    --openai_model_name "auto" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --samples_per_call "$SAMPLES_PER_CALL" \
    --max_tokens "$MAX_TOKENS" \
    --max_concurrency "$BATCH_SIZE" \
    "$@"

# 结束与清理
echo "🛑 停止 vLLM Server (PID=$SERVER_PID)"
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true


