#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试混合奖励函数 (number_match + substring_match)
"""
import sys
import os

# 添加verl路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from verl.utils.reward_score.gsm8k import compute_score, compute_hybrid_score, compute_number_match_score

def test_hybrid_reward():
    """测试混合奖励函数"""
    print("测试混合奖励函数...")
    print("=" * 60)
    
    # 测试用例1: 完全匹配
    solution1 = "It takes 50/2=25 jellybeans to fill up a small drinking glass."
    ground_truth1 = "It takes 50/2=25 jellybeans to fill up a small drinking glass."
    
    number_score1 = compute_number_match_score(solution1, ground_truth1)
    hybrid_score1 = compute_hybrid_score(solution1, ground_truth1)
    result1 = compute_score(solution1, ground_truth1, method="hybrid")
    
    print(f"测试1 - 完全匹配:")
    print(f"  Solution: {solution1}")
    print(f"  Ground Truth: {ground_truth1}")
    print(f"  数字匹配分数: {number_score1}")
    print(f"  混合分数: {hybrid_score1}")
    print(f"  最终分数: {result1}")
    print()
    
    # 测试用例2: 数字匹配但子字符串不匹配
    solution2 = "The answer is 25 jellybeans. It takes 50/2=25 jellybeans to fill up a small drinking glass."
    ground_truth2 = "It takes 50/2=25 jellybeans to fill up a small drinking glass. The total cost is 100 dollars."
    
    number_score2 = compute_number_match_score(solution2, ground_truth2)
    hybrid_score2 = compute_hybrid_score(solution2, ground_truth2)
    result2 = compute_score(solution2, ground_truth2, method="hybrid")
    
    print(f"测试2 - 数字匹配但子字符串不匹配:")
    print(f"  Solution: {solution2}")
    print(f"  Ground Truth: {ground_truth2}")
    print(f"  数字匹配分数: {number_score2}")
    print(f"  混合分数: {hybrid_score2}")
    print(f"  最终分数: {result2}")
    print()
    
    # 测试用例3: 子字符串匹配但数字不匹配
    solution3 = "It takes 50/2=25 jellybeans to fill up a small drinking glass. But the calculation is wrong."
    ground_truth3 = "It takes 50/2=25 jellybeans to fill up a small drinking glass."
    
    number_score3 = compute_number_match_score(solution3, ground_truth3)
    hybrid_score3 = compute_hybrid_score(solution3, ground_truth3)
    result3 = compute_score(solution3, ground_truth3, method="hybrid")
    
    print(f"测试3 - 子字符串匹配但数字不匹配:")
    print(f"  Solution: {solution3}")
    print(f"  Ground Truth: {ground_truth3}")
    print(f"  数字匹配分数: {number_score3}")
    print(f"  混合分数: {hybrid_score3}")
    print(f"  最终分数: {result3}")
    print()
    
    # 测试用例4: 完全不匹配
    solution4 = "The world is facing increasingly complex challenges, such as climate change, pandemics, technological advancements, and shifts in global values."
    ground_truth4 = "It takes 50/2=25 jellybeans to fill up a small drinking glass."
    
    number_score4 = compute_number_match_score(solution4, ground_truth4)
    hybrid_score4 = compute_hybrid_score(solution4, ground_truth4)
    result4 = compute_score(solution4, ground_truth4, method="hybrid")
    
    print(f"测试4 - 完全不匹配:")
    print(f"  Solution: {solution4}")
    print(f"  Ground Truth: {ground_truth4}")
    print(f"  数字匹配分数: {number_score4}")
    print(f"  混合分数: {hybrid_score4}")
    print(f"  最终分数: {result4}")
    print()
    
    # 测试用例5: 无数字的文本
    solution5 = "The answer is correct and well explained."
    ground_truth5 = "The answer is correct and well explained."
    
    number_score5 = compute_number_match_score(solution5, ground_truth5)
    hybrid_score5 = compute_hybrid_score(solution5, ground_truth5)
    result5 = compute_score(solution5, ground_truth5, method="hybrid")
    
    print(f"测试5 - 无数字的文本:")
    print(f"  Solution: {solution5}")
    print(f"  Ground Truth: {ground_truth5}")
    print(f"  数字匹配分数: {number_score5}")
    print(f"  混合分数: {hybrid_score5}")
    print(f"  最终分数: {result5}")
    print()
    
    # 测试用例6: 自定义权重
    solution6 = "The answer is 25 jellybeans. It takes 50/2=25 jellybeans to fill up a small drinking glass."
    ground_truth6 = "It takes 50/2=25 jellybeans to fill up a small drinking glass. The total cost is 100 dollars."
    
    # 测试不同的权重组合
    weights = [
        (0.7, 0.3),  # 默认权重
        (0.5, 0.5),  # 平衡权重
        (0.3, 0.7),  # 更重视子字符串匹配
        (0.9, 0.1),  # 更重视数字匹配
    ]
    
    print(f"测试6 - 不同权重组合:")
    print(f"  Solution: {solution6}")
    print(f"  Ground Truth: {ground_truth6}")
    print(f"  数字匹配分数: {compute_number_match_score(solution6, ground_truth6)}")
    print(f"  子字符串匹配分数: {1.0 if ground_truth6.strip() in solution6.strip() else 0.0}")
    print()
    
    for num_w, sub_w in weights:
        custom_score = compute_hybrid_score(solution6, ground_truth6, num_w, sub_w)
        print(f"  权重 ({num_w}, {sub_w}): {custom_score}")
    print()

def compare_methods():
    """比较不同方法的结果"""
    print("比较不同方法的结果...")
    print("=" * 60)
    
    test_cases = [
        ("完全匹配", "It takes 50/2=25 jellybeans to fill up a small drinking glass.", "It takes 50/2=25 jellybeans to fill up a small drinking glass."),
        ("部分数字匹配", "The answer is 25 jellybeans. It takes 50/2=25 jellybeans to fill up a small drinking glass.", "It takes 50/2=25 jellybeans to fill up a small drinking glass. The total cost is 100 dollars."),
        ("子字符串匹配", "It takes 50/2=25 jellybeans to fill up a small drinking glass. But the calculation is wrong.", "It takes 50/2=25 jellybeans to fill up a small drinking glass."),
        ("不匹配", "The world is facing increasingly complex challenges.", "It takes 50/2=25 jellybeans to fill up a small drinking glass."),
    ]
    
    methods = ["strict", "substring_match", "number_match", "hybrid"]
    
    for name, solution, ground_truth in test_cases:
        print(f"{name}:")
        print(f"  Solution: {solution}")
        print(f"  Ground Truth: {ground_truth}")
        for method in methods:
            try:
                score = compute_score(solution, ground_truth, method=method)
                print(f"    {method}: {score}")
            except Exception as e:
                print(f"    {method}: Error - {e}")
        print()

if __name__ == "__main__":
    test_hybrid_reward()
    compare_methods()
    print("所有测试完成！")

