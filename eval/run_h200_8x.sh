#!/bin/bash
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT="https://hf-mirror.com"

# 8x H200 启动脚本（包装现有 run_batch_inference.sh），提供更稳妥的默认值
HOME="/home/ningmiao/ningyuan"
RUN_SCRIPT="$HOME/verl/eval/run_batch_inference.sh"

if [ ! -f "$RUN_SCRIPT" ]; then
    echo "❌ 启动脚本不存在: $RUN_SCRIPT"
    exit 1
fi

# 针对 4x H800 的推荐默认参数（可被命令行参数覆盖）
MODEL="$HOME/models/Qwen3-32B"
DATASET="/home/ningmiao/ningyuan/verl/data/OpenR1-Math-220k/data"
SPLIT="train"
OUTPUT_DIR="$HOME/verl/eval/results/OpenR1-Math-220k_H200_8x_Qwen3-32B"
NUM_SAMPLES=16
SAMPLES_PER_CALL=16
BATCH_SIZE=128
NUM_GPUS=8
GPU_MEMORY=0.90
MAX_MODEL_LEN=4096
MAX_TOKENS=2048
DTYPE="bfloat16"   # 可选: auto|bfloat16|float16|float32
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20
MIN_P=0


echo "🚀 8x H200 启动多vLLM Server + 客户端批推理..."
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

# 启动端口基数
BASE_PORT=${BASE_PORT:-7000}
SERVER_LOGS=()
SERVER_PIDS=()
mkdir -p "$OUTPUT_DIR"

echo "🟢 启动 $NUM_GPUS 个独立 vLLM Server (每卡一个)..."
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$((i))
    PORT=$((BASE_PORT + $((i))))
    SERVER_LOG="$OUTPUT_DIR/server_gpu${GPU_ID}.log"

    echo "  启动 GPU $GPU_ID 的服务器 on port $PORT"
    ENTRY_CMD="export CUDA_VISIBLE_DEVICES=$GPU_ID && /opt/anaconda3/envs/xny_verl/bin/python -m vllm.entrypoints.openai.api_server --model '$MODEL' --tensor-parallel-size 1 --dtype $DTYPE --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEMORY --trust-remote-code --port $PORT"
    SAMPLE_ARGS="--temperature $TEMPERATURE --top_p $TOP_P --top_k $TOP_K --min_p $MIN_P"
    CMD="$ENTRY_CMD $SAMPLE_ARGS"

    # 保存命令到文件
    echo "$CMD" > "$OUTPUT_DIR/launch_server_gpu${GPU_ID}.sh"

    # 启动服务器
    nohup bash -lc "$CMD" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    SERVER_PIDS+=($SERVER_PID)
    SERVER_LOGS+=($SERVER_LOG)

    echo "  GPU $GPU_ID Server PID: $SERVER_PID"
    sleep 2  # 给每个服务器一点启动时间
done

# 等待所有服务就绪（增加等待时间到5分钟）
if ! command -v curl >/dev/null 2>&1; then
  echo "⚠️ 未检测到 curl，将等待 60 秒"
  sleep 60
else
  echo -n "⏳ 等待所有 vLLM Server 就绪 (最多等待 5 分钟)"
  MAX_WAIT=300  # 增加到5分钟
  ALL_READY=false

  for i in $(seq 1 $MAX_WAIT); do
      sleep 1
      READY_COUNT=0

      # 检查所有8个服务器是否就绪
      for port in $(seq $BASE_PORT $((BASE_PORT + $NUM_GPUS - 1))); do
          if curl -s "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
              ((READY_COUNT++))
          fi
      done

      if [ $READY_COUNT -eq $NUM_GPUS ]; then
          echo " - 所有服务器已就绪 ($READY_COUNT/$NUM_GPUS)"
          ALL_READY=true
          break
      fi

      if [ $((i % 30)) -eq 0 ]; then
          echo -n " [$READY_COUNT/$NUM_GPUS ready, $i/$MAX_WAIT]"
      else
          echo -n "."
      fi

      # 检查是否有服务器进程退出
      EXITED=false
      for pid in "${SERVER_PIDS[@]}"; do
          if ! kill -0 $pid 2>/dev/null; then
              EXITED=true
              break
          fi
      done

      if [ "$EXITED" = true ]; then
          echo "❌ 某个服务器进程已退出，查看日志了解详情"
          exit 1
      fi
  done

  if [ "$ALL_READY" != true ]; then
      echo "❌ 等待超时，部分服务器未能就绪"
      exit 1
  fi
fi

# 额外验证：等待几秒钟确保服务器完全稳定
echo "🔍 额外等待 10 秒确保所有服务器完全就绪..."
sleep 10

# 构建多服务器URL列表
SERVER_URLS=""
for port in $(seq $BASE_PORT $((BASE_PORT + $NUM_GPUS - 1))); do
    if [ -n "$SERVER_URLS" ]; then
        SERVER_URLS="$SERVER_URLS,http://127.0.0.1:$port"
    else
        SERVER_URLS="http://127.0.0.1:$port"
    fi
done

echo "🧩 运行客户端聚合，使用多服务器: $SERVER_URLS"
echo "🧩 运行客户端聚合: batch_inference_client.py"
/opt/anaconda3/envs/xny_verl/bin/python "$HOME/verl/eval/batch_inference_client.py" \
    --server_urls "$SERVER_URLS" \
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
echo "🛑 停止所有 vLLM Server..."
for i in "${!SERVER_PIDS[@]}"; do
    pid=${SERVER_PIDS[$i]}
    gpu_id=${i}
    echo "  停止 GPU $gpu_id Server (PID=$pid)"
    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
done
echo "✅ 所有服务器已停止"


