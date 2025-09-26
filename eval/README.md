# 批推理脚本使用说明

这个脚本用于对数学问题数据集进行批量推理评估，修复了之前的CUDA初始化问题。

## 主要修复内容

1. **修复CUDA设备分配问题**: 每个进程使用独立的GPU，避免了设备冲突
2. **修复环境变量设置语法**: 正确设置CUDA_VISIBLE_DEVICES
3. **修复torch.distributed环境变量**: 设置MASTER_ADDR和MASTER_PORT等必要环境变量
4. **改进错误处理**: 添加了更详细的错误信息和异常处理
5. **优化vLLM配置**: 添加了适当的模型参数以提高稳定性

## 使用方法

### 基本用法

```bash
cd /home/ningmiao/ningyuan/verl/eval
./run_batch_inference.sh --model <模型路径>
```

### 完整参数

```bash
./run_batch_inference.sh --model <模型路径> \
    --dataset <数据集名称> \
    --split <数据集分割> \
    --output_dir <输出目录> \
    --num_samples <样本数量> \
    --batch_size <批大小> \
    --num_gpus <GPU数量> \
    --gpu_memory <GPU内存利用率>
```

### 参数说明

- `--model`: **必需** - 模型路径或HuggingFace模型ID
- `--dataset`: 数据集名称 (默认: open-r1/OpenR1-Math-220k)
- `--split`: 数据集分割 (默认: default)
- `--output_dir`: 输出目录 (默认: /home/ningmiao/ningyuan/verl/eval/results/OpenR1-Math-220k)
- `--num_samples`: 每个问题生成的样本数量 (默认: 64)
- `--batch_size`: 批处理大小 (默认: 32)
- `--num_gpus`: 使用的GPU数量 (默认: 8)
- `--gpu_memory`: GPU内存利用率 (默认: 0.9)

### 示例

```bash
# 使用本地模型
./run_batch_inference.sh --model /path/to/your/model

# 使用HuggingFace模型
./run_batch_inference.sh --model Qwen/Qwen2.5-7B-Instruct

# 自定义参数
./run_batch_inference.sh --model Qwen/Qwen2.5-7B-Instruct \
    --num_samples 32 \
    --batch_size 16 \
    --num_gpus 4 \
    --gpu_memory 0.8
```

## 输出结果

脚本会在输出目录中生成以下文件：

- `shard_{rank}.jsonl`: 每个GPU进程的结果文件
- `valid.jsonl`: 部分正确的结果
- `all_correct.jsonl`: 全部正确的结果
- `all_wrong.jsonl`: 全部错误的结果

每个结果包含：
- `id`: 问题ID
- `problem`: 问题内容
- `answer`: 标准答案
- `generations`: 生成的多个答案
- `correct_count`: 正确答案数量
- `status`: 状态 (valid/all_correct/all_wrong)

## 故障排除

### 常见问题

1. **CUDA初始化错误**
   - 确保使用正确的Python环境: `/opt/anaconda3/envs/xny_verl/bin/python3`
   - 检查GPU状态: `nvidia-smi`

2. **torch.distributed环境变量错误**
   - 错误信息: `MASTER_ADDR expected, but not set`
   - 解决方案: 脚本会自动设置必要的环境变量，无需手动干预
   - 如果仍有问题，检查防火墙是否阻止了端口12355

3. **内存不足错误**
   - 减少 `--gpu_memory` 参数 (建议: 0.7-0.9)
   - 减少 `--batch_size` (建议: 8-32)
   - 减少 `--num_samples` (建议: 16-64)

4. **模型加载错误**
   - 确保模型路径正确
   - 检查模型文件是否存在且完整
   - 对于大模型，确保有足够的GPU内存

### 测试CUDA状态

运行以下命令测试CUDA是否正常工作：

```bash
cd /home/ningmiao/ningyuan/verl/eval
/opt/anaconda3/envs/xny_verl/bin/python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## 注意事项

1. 脚本会自动检测可用的GPU数量，最多使用指定的GPU数量
2. 每个GPU进程独立运行，互不干扰
3. 结果会自动合并，最终统计信息会在rank=0进程中显示
4. 如果某个批次处理失败，会跳过该批次并继续处理其他批次
