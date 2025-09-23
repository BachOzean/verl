#!/usr/bin/env python3
"""
修复现有数据集，添加GRPO训练所需的reward_model字段
"""

import pandas as pd
import json
import sys
import os

def fix_existing_dataset(file_path):
    """
    为现有数据集添加reward_model字段
    """
    print(f"正在处理文件: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        return False
    
    try:
        # 读取数据
        df = pd.read_parquet(file_path)
        print(f"读取到 {len(df)} 行数据")
        print(f"现有列: {list(df.columns)}")
        
        # 检查是否已经有reward_model字段
        if 'reward_model' in df.columns:
            print("数据集已包含reward_model字段，无需修改")
            return True
        
        # 检查必需的列
        if 'response' not in df.columns:
            print("错误: 数据集缺少response列")
            return False
        
        # 添加reward_model字段
        print("添加reward_model字段...")
        df['reward_model'] = df['response'].apply(lambda x: {
            "style": "rule", 
            "ground_truth": str(x)  # 确保是字符串格式
        })
        
        # 备份原文件
        backup_path = file_path + ".backup"
        if not os.path.exists(backup_path):
            print(f"创建备份文件: {backup_path}")
            df_original = pd.read_parquet(file_path)
            df_original.to_parquet(backup_path, index=False)
        
        # 保存修改后的文件
        df.to_parquet(file_path, index=False)
        print(f"文件已更新: {file_path}")
        print(f"新增列: reward_model")
        print(f"总列数: {len(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

def main():
    # 需要修复的文件列表
    files_to_fix = [
        "/data/home/scyb494/Hybrid-FT/grpo_prompt_subq_progressive.parquet",
        "/data/home/scyb494/verl/data/MATH-500/test.parquet",
    ]
    
    # 也可以从命令行参数获取文件路径
    if len(sys.argv) > 1:
        files_to_fix = sys.argv[1:]
    
    print("=== 修复现有数据集，添加reward_model字段 ===")
    print(f"需要处理的文件: {len(files_to_fix)}")
    
    success_count = 0
    for file_path in files_to_fix:
        print(f"\n处理文件 {success_count + 1}/{len(files_to_fix)}: {file_path}")
        if fix_existing_dataset(file_path):
            success_count += 1
        else:
            print(f"处理失败: {file_path}")
    
    print(f"\n=== 处理完成 ===")
    print(f"成功处理: {success_count}/{len(files_to_fix)} 个文件")
    
    if success_count == len(files_to_fix):
        print("所有文件处理成功！现在可以重新运行训练脚本。")
        return 0
    else:
        print("部分文件处理失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    exit(main())
