#!/usr/bin/env python3
"""
数据预处理脚本：将用户的自定义数据集转换为GRPO训练所需的格式
"""

import pandas as pd
import json
import os
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

def preprocess_custom_dataset(input_file, output_dir, test_size=0.1, random_state=42):
    """
    预处理自定义数据集
    
    Args:
        input_file: 输入文件路径（支持CSV或Parquet格式）
        output_dir: 输出目录
        test_size: 验证集比例
        random_state: 随机种子
    """
    print(f"正在读取数据集: {input_file}")
    
    # 读取数据
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    else:
        raise ValueError("支持的文件格式: .csv 或 .parquet")
    
    print(f"数据集形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 验证必需的列
    required_columns = ['prompt', 'response']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必需的列: {missing_columns}")
    
    # 数据清洗
    print("开始数据清洗...")
    
    # 删除空值
    initial_len = len(df)
    df = df.dropna(subset=['prompt', 'response'])
    print(f"删除空值后: {len(df)} 行 (删除了 {initial_len - len(df)} 行)")
    
    # 删除空字符串
    df = df[(df['prompt'].str.strip() != '') & (df['response'].str.strip() != '')]
    print(f"删除空字符串后: {len(df)} 行")
    
    # 添加GRPO训练所需的reward_model字段
    print("添加reward_model字段...")
    df['reward_model'] = df['response'].apply(lambda x: {
        "style": "rule", 
        "ground_truth": x  # 使用response作为ground_truth
    })
    
    # 如果有stage列，可以根据需要过滤特定阶段的数据
    if 'stage' in df.columns:
        print(f"数据集包含的阶段: {df['stage'].unique()}")
        # 可以根据需要选择特定阶段，这里保留所有阶段
    
    # 如果有group_id和turn_index，显示统计信息
    if 'group_id' in df.columns:
        print(f"对话组数量: {df['group_id'].nunique()}")
    if 'turn_index' in df.columns:
        print(f"轮次分布: {df['turn_index'].value_counts().sort_index()}")
    
    # 创建训练和验证集
    print("划分训练集和验证集...")
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['stage'] if 'stage' in df.columns else None
    )
    
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存处理后的数据
    train_path = os.path.join(output_dir, 'custom_dataset_train.parquet')
    val_path = os.path.join(output_dir, 'custom_dataset_val.parquet')
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    print(f"训练集保存到: {train_path}")
    print(f"验证集保存到: {val_path}")
    
    # 生成数据统计报告
    report = {
        "dataset_info": {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "columns": list(df.columns)
        },
        "text_statistics": {
            "avg_prompt_length": df['prompt'].str.len().mean(),
            "avg_response_length": df['response'].str.len().mean(),
            "max_prompt_length": df['prompt'].str.len().max(),
            "max_response_length": df['response'].str.len().max()
        }
    }
    
    if 'stage' in df.columns:
        report["stage_distribution"] = df['stage'].value_counts().to_dict()
    
    if 'group_id' in df.columns:
        report["conversation_groups"] = df['group_id'].nunique()
    
    # 保存报告
    report_path = os.path.join(output_dir, 'dataset_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"数据统计报告保存到: {report_path}")
    
    return train_path, val_path

def prepare_math500_placeholder(output_dir):
    """
    创建MATH-500数据集的占位符文件
    """
    math500_dir = os.path.join(output_dir, 'math500')
    os.makedirs(math500_dir, exist_ok=True)
    
    # 创建示例MATH-500格式的数据
    # 注意：这里只是占位符，实际使用时需要下载真实的MATH-500数据集
    sample_data = {
        'prompt': [
            "Solve the following math problem: What is 2 + 2?",
            "Calculate the derivative of f(x) = x^2 + 3x - 5",
            "Find the area of a circle with radius 5"
        ],
        'response': [
            "2 + 2 = 4",
            "f'(x) = 2x + 3",
            "Area = π × r² = π × 5² = 25π"
        ]
    }
    
    df_math500 = pd.DataFrame(sample_data)
    
    # 保存占位符文件
    val_path = os.path.join(math500_dir, 'val.parquet')
    test_path = os.path.join(math500_dir, 'test.parquet')
    
    df_math500.to_parquet(val_path, index=False)
    df_math500.to_parquet(test_path, index=False)
    
    print(f"MATH-500占位符文件创建:")
    print(f"  验证集: {val_path}")
    print(f"  测试集: {test_path}")
    print("注意: 这些是占位符文件，请替换为真实的MATH-500数据集")

def main():
    parser = argparse.ArgumentParser(description='预处理自定义数据集用于GRPO训练')
    parser.add_argument('--input', '-i', required=True, 
                       help='输入数据文件路径 (.csv 或 .parquet)')
    parser.add_argument('--output_dir', '-o', default='/data/home/scyb494/verl/data',
                       help='输出目录路径')
    parser.add_argument('--test_size', type=float, default=0.1,
                       help='验证集比例 (默认: 0.1)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--create_math500_placeholder', action='store_true',
                       help='创建MATH-500占位符文件')
    
    args = parser.parse_args()
    
    try:
        # 预处理自定义数据集
        train_path, val_path = preprocess_custom_dataset(
            args.input, 
            args.output_dir, 
            args.test_size, 
            args.random_state
        )
        
        # 如果需要，创建MATH-500占位符
        if args.create_math500_placeholder:
            prepare_math500_placeholder(args.output_dir)
        
        print("\n数据预处理完成！")
        print("接下来可以运行训练脚本:")
        print("bash train_grpo_custom_dataset.sh")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
