#!/bin/bash

# 数学问题分解脚本 - 调用硅基流动API生成子问题和答案
# 使用示例

# 配置参数
INPUT_PATH="/home/ningmiao/ningyuan/verl/eval/results/OpenR1-Math-220k_H200_8x/valid_deduped.jsonl"
OUTPUT_FILE="/home/ningmiao/ningyuan/verl/eval/results/problem_decomposition_results.jsonl"
API_KEY="sk-eejcrxhumrpflxyelcwavqyslrezedxxsmjihwhahiyqcbqa"  # 请替换为你的真实API密钥

# 可选参数（使用默认值）
MODEL="deepseek-ai/DeepSeek-V3.1-Terminus"
MAX_TOKENS=1024
TEMPERATURE=0.6
TOP_P=0.9
BATCH_SIZE=5
MAX_RETRIES=3
DELAY=0

echo "开始数学问题分解任务..."
echo "输入文件: $INPUT_PATH"
echo "输出文件: $OUTPUT_FILE"
echo "使用模型: $MODEL"


# 运行问题分解脚本
python /home/ningmiao/ningyuan/verl/eval/problem_decomposition.py \
    --input_path "$INPUT_PATH" \
    --output_file "$OUTPUT_FILE" \
    --api_key "$API_KEY" \
    --model "$MODEL" \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --batch_size $BATCH_SIZE \
    --max_retries $MAX_RETRIES \
    --delay $DELAY \
    --enable_resume \
    --progress_file "$OUTPUT_FILE.progress" \
    --save_interval 1

echo "数学问题分解任务完成！"
