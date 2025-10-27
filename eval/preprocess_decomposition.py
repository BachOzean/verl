import json
import re
import pandas as pd

def process_subproblems(input_file, output_file):
    """
    处理输入JSONL文件，将每个子问题转换为独立的样本，并写入输出文件。
    
    Args:
        input_file (str): 输入JSONL文件路径
        output_file (str): 输出Parquet文件路径
    """
    samples = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析错误: {e}")
                continue
            
            original_uuid = data.get('uuid')
            if not original_uuid:
                print(f"信息: 第{line_num}行UUID为空，已跳过")
                continue
                
            subproblems = data.get('subproblems', [])
            if not subproblems:
                print(f"信息: 第{line_num}行无子问题，已跳过")
                continue
                
            for subproblem in subproblems:
                subproblem_id = subproblem.get('subproblem_id', 'unknown')
                question = subproblem.get('question', '')
                answer = subproblem.get('answer', '')
                
                # 检查问题或答案是否为空
                if not question or not answer:
                    print(f"警告: 第{line_num}行子问题{subproblem_id}问题或答案为空，已跳过")
                    continue
                
                # 检查问题中是否包含"subproblem"关键词（不区分大小写）
                if re.search(r'subproblem', question, re.IGNORECASE):
                    print(f"信息: 第{line_num}行子问题{subproblem_id}问题中包含'subproblem'，已跳过")
                    continue
                
                samples.append({
                    'data_source': 'open-r1/OpenR1-Math-220k',
                    "prompt": [
                        {
                            "role": "user",
                            "content": question,
                        }
                    ],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {
                        "uuid": f"{original_uuid}_{subproblem_id}",
                        "question": question,
                        "answer": answer,
                    },
                })
    
    # 将样本列表转换为DataFrame
    df = pd.DataFrame(samples)
    
    # 保存为Parquet文件
    df.to_parquet(output_file, index=False, engine='pyarrow')
    
    print(f"处理完成。共生成 {len(samples)} 个样本。")
    print(f"输出文件: {output_file}")

if __name__ == '__main__':
    input_filename = '/home/ningmiao/ningyuan/verl/eval/subproblems_openr1_2025-10-14.jsonl'  # 输入文件路径
    output_filename = '/home/ningmiao/ningyuan/verl/eval/subproblems_openr1_2025-10-14_filtered.parquet'  # 输出Parquet文件路径
    process_subproblems(input_filename, output_filename)