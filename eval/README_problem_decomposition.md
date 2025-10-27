# 数学问题分解脚本

这个脚本用于调用硅基流动API，将数学问题分解成一系列子问题并生成对应答案，帮助理解复杂的数学推理过程。

## 功能特点

1. **智能问题分解**：自动将复杂数学问题分解成逻辑清晰的子问题序列
2. **结构化输出**：每个子问题包含问题描述、解题思路和答案
3. **格式化响应**：使用标准化的Markdown格式，便于后续解析和使用
4. **错误处理**：包含重试机制和完善的错误处理
5. **进度跟踪**：实时显示处理进度和成功率统计

## 使用前提

1. **硅基流动API密钥**：需要在[硅基流动官网](https://siliconflow.cn)注册并获取API密钥
2. **Python环境**：Python 3.7+
3. **依赖包**：requests库（`pip install requests`）

## 文件说明

- `problem_decomposition.py` - 主脚本文件
- `run_problem_decomposition.sh` - 执行脚本示例
- `README_problem_decomposition.md` - 使用说明文档

## 使用步骤

### 1. 配置API密钥

在 `run_problem_decomposition.sh` 文件中，将 `API_KEY` 替换为你的真实API密钥：

```bash
API_KEY="your_actual_api_key_here"
```

### 2. 修改输入输出路径

根据需要修改脚本中的输入和输出文件路径：

```bash
INPUT_FILE="/path/to/your/input/file.jsonl"
OUTPUT_FILE="/path/to/your/output/file.jsonl"
```

### 3. 运行脚本

```bash
chmod +x run_problem_decomposition.sh
./run_problem_decomposition.sh
```

或者直接使用Python脚本：

```bash
python problem_decomposition.py \
    --input_file "/path/to/input.jsonl" \
    --output_file "/path/to/output.jsonl" \
    --api_key "your_api_key" \
    --model "Qwen/Qwen2.5-72B-Instruct"
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_file` | 必填 | 输入的JSONL文件路径 |
| `--output_file` | 必填 | 输出的JSONL文件路径 |
| `--api_key` | 必填 | 硅基流动API密钥 |
| `--model` | `Qwen/Qwen2.5-72B-Instruct` | 使用的模型名称 |
| `--max_tokens` | `2048` | 最大生成token数 |
| `--temperature` | `0.7` | 温度参数，控制生成多样性 |
| `--top_p` | `0.9` | top_p参数，控制生成质量 |
| `--batch_size` | `5` | 批量处理大小 |
| `--max_retries` | `3` | 最大重试次数 |
| `--delay` | `1.0` | 请求间延迟秒数，避免API限流 |

## 输入文件格式

输入文件应为JSONL格式，每行一个JSON对象，包含以下字段：

```json
{
  "id": "问题ID（可选）",
  "problem": "数学问题描述",
  "answer": "标准答案"
}
```

## 输出文件格式

输出文件为JSONL格式，每行包含以下字段：

```json
{
  "id": "原问题ID",
  "original_problem": "原始数学问题",
  "original_answer": "原始答案",
  "decomposition": "完整的分解文本",
  "subproblems": [
    {
      "subproblem_id": "subproblem_1",
      "problem_description": "子问题描述",
      "reasoning": "解题思路说明",
      "answer": "子问题答案"
    }
  ],
  "num_subproblems": 3
}
```

## Prompt设计特点

脚本使用的prompt具有以下特点：

1. **清晰的任务指令**：明确要求分解成子问题并提供答案
2. **结构化格式要求**：指定了严格的输出格式，便于后续解析
3. **逻辑顺序要求**：强调子问题应按逻辑顺序排列
4. **独立性与关联性**：要求子问题相对独立但答案有逻辑关联
5. **详细说明要求**：要求详细说明解题思路

## 示例输出

### 原问题
```
3. (6 points) A construction company was building a tunnel. When 1/3 of the tunnel was completed at the original speed, they started using new equipment, which increased the construction speed by 20% and reduced the working hours to 80% of the original. As a result, it took a total of 185 days to complete the tunnel. If they had not used the new equipment and continued at the original speed, it would have taken __ days to complete the tunnel.
```

### 分解输出
```
## 子问题分解

### 子问题1
**问题描述：** 假设隧道总长度为3个单位，那么在新设备使用前完成了1个单位。设原速度为每天完成x个单位，原工作时间为每天h小时。新设备速度为1.2x，每天工作时间为0.8h。

**解题思路：** 首先建立变量表示隧道长度、工作速度和工作时间的关系。通过比例关系计算出各种参数。

**答案：** 隧道总长度设为3L，原速度每天L，原工作时间每天H小时。新设备速度1.2L，每天工作0.8H小时。

### 子问题2
**问题描述：** 计算在新设备使用前的工作天数。新设备使用前完成了1/3的工程，工作了t天。

**解题思路：** 使用工作量 = 速度 × 时间的关系计算。

**答案：** 新设备使用前工作天数为 t = (L) / (工作量) = 1L / L = 1天。

（更多子问题...）
```

## 注意事项

1. **API费用**：每次调用都会消耗API费用，请合理控制处理数量
2. **速率限制**：脚本已包含延迟机制，如遇速率限制可增加延迟时间
3. **模型选择**：可根据需要选择不同的模型，不同模型效果可能有差异
4. **错误处理**：脚本会跳过处理失败的问题，继续处理下一条
5. **输出解析**：如果需要进一步处理子问题数据，可以解析`subproblems`字段

## 故障排除

1. **API密钥错误**：检查API密钥是否正确配置
2. **网络连接问题**：检查网络连接和防火墙设置
3. **JSON格式错误**：确保输入文件是有效的JSONL格式
4. **权限问题**：确保有读写相关文件的权限

## 扩展使用

如果你需要：

1. **自定义prompt**：可以修改`create_decomposition_prompt`函数
2. **不同的输出格式**：可以修改`parse_decomposition_response`函数
3. **批量处理多个文件**：可以修改脚本支持多个输入文件
4. **不同的API服务**：可以修改`call_siliconcloud_api`函数适配其他服务

## 技术支持

如果遇到问题，请检查：
1. 硅基流动官网的API文档
2. 脚本的错误日志输出
3. 输入文件的格式是否正确

这个脚本旨在帮助更好地理解和学习复杂数学问题的解决过程，通过分解成子问题的方式提供更清晰的学习路径。
