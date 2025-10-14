#!/usr/bin/env bash

# Resolve repo root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"


# Defaults (can be overridden via environment)
SILICONFLOW_API_KEY=${SILICONFLOW_API_KEY:-sk-eejcrxhumrpflxyelcwavqyslrezedxxsmjihwhahiyqcbqa}
INPUT_PATH=${INPUT_PATH:-/data/home/scyb494/.cache/huggingface/hub/datasets--open-r1--OpenR1-Math-220k/snapshots/e4e141ec9dea9f8326f4d347be56105859b2bd68/data}
TARGET_FILE=${TARGET_FILE:-/data/home/scyb494/verl/eval/valid_deduped.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/data/home/scyb494/verl/eval}
MODEL=${MODEL:-deepseek-ai/DeepSeek-V3.2-Exp}
MAX_TOKENS=${MAX_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
TOP_K=${TOP_K:-20}
BATCH_SIZE=${BATCH_SIZE:-10}
MAX_RETRIES=${MAX_RETRIES:-3}
DELAY=${DELAY:-0.1}

mkdir -p "${OUTPUT_DIR}"
TS=$(date +%F_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/subproblems_openr1_2025-10-14.jsonl"

echo "[INFO] Repo root:        ${REPO_ROOT}"
echo "[INFO] Input path:       ${INPUT_PATH}"
echo "[INFO] Output file:      ${OUTPUT_FILE}"
echo "[INFO] Model:            ${MODEL}"
echo "[INFO] Max tokens:       ${MAX_TOKENS}"
echo "[INFO] Temperature:      ${TEMPERATURE}"
echo "[INFO] Top-p:            ${TOP_P}"
echo "[INFO] Top-k:            ${TOP_K}"
echo "[INFO] Batch size:       ${BATCH_SIZE}"
echo "[INFO] Max retries:      ${MAX_RETRIES}"
echo "[INFO] Delay (seconds):  ${DELAY}"

python /data/home/scyb494/verl/eval/decompose_subproblems_api.py \
  --input_path "${INPUT_PATH}" \
  --target_file "${TARGET_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --api_key "${SILICONFLOW_API_KEY}" \
  --model "${MODEL}" \
  --max_tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --batch_size "${BATCH_SIZE}" \
  --max_retries "${MAX_RETRIES}" > "${OUTPUT_FILE}.log" 2>&1

echo "[INFO] Done. Results saved to: ${OUTPUT_FILE}"





