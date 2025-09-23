# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import math

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def extract_numbers(text):
    """从文本中提取所有数字（包括整数和小数）"""
    if not text:
        return []
    # 匹配整数和小数，包括负数
    pattern = r'-?\d+\.?\d*'
    numbers = re.findall(pattern, text)
    # 转换为浮点数，过滤掉无效的数字
    valid_numbers = []
    for num_str in numbers:
        try:
            if '.' in num_str:
                valid_numbers.append(float(num_str))
            else:
                valid_numbers.append(int(num_str))
        except ValueError:
            continue
    return valid_numbers


def compute_number_match_score(solution_str, ground_truth):
    """基于数字匹配比例计算分数"""
    if not ground_truth or not solution_str:
        return 0.0
    
    # 提取ground_truth和solution中的数字
    gt_numbers = extract_numbers(ground_truth)
    sol_numbers = extract_numbers(solution_str)
    
    if not gt_numbers:
        # 如果ground_truth中没有数字，使用简单的字符串匹配
        return 1.0 if ground_truth.strip() in solution_str.strip() else 0.0
    
    if not sol_numbers:
        # 如果solution中没有数字，返回0
        return 0.0
    
    # 计算匹配的数字数量
    matched_count = 0
    for gt_num in gt_numbers:
        for sol_num in sol_numbers:
            # 允许一定的数值误差（对于浮点数）
            if isinstance(gt_num, float) and isinstance(sol_num, float):
                if abs(gt_num - sol_num) < 1e-6:
                    matched_count += 1
                    break
            elif gt_num == sol_num:
                matched_count += 1
                break
    
    # 计算匹配比例
    match_ratio = matched_count / len(gt_numbers)
    
    # 如果所有数字都匹配，返回1.0
    # 如果部分匹配，返回匹配比例
    # 如果没有匹配，返回0.0
    return match_ratio


def compute_hybrid_score(solution_str, ground_truth, number_weight=0.7, substring_weight=0.3):
    """混合数字匹配和子字符串匹配的评分函数
    
    Args:
        solution_str: 模型生成的解决方案文本
        ground_truth: 标准答案文本
        number_weight: 数字匹配的权重 (默认0.7)
        substring_weight: 子字符串匹配的权重 (默认0.3)
    
    Returns:
        float: 混合评分结果 (0.0-1.0)
    """
    if not ground_truth or not solution_str:
        return 0.0
    
    # 计算数字匹配分数
    number_score = compute_number_match_score(solution_str, ground_truth)
    
    # 计算子字符串匹配分数
    gt_clean = ground_truth.strip()
    sol_clean = solution_str.strip()
    substring_score = 1.0 if gt_clean in sol_clean else 0.0
    
    # 如果ground_truth中没有数字，只使用子字符串匹配
    gt_numbers = extract_numbers(ground_truth)
    if not gt_numbers:
        return substring_score
    
    # 如果solution中没有数字，数字匹配分数为0
    sol_numbers = extract_numbers(solution_str)
    if not sol_numbers:
        number_score = 0.0
    
    # 计算加权混合分数
    hybrid_score = number_weight * number_score + substring_weight * substring_score
    
    return hybrid_score


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict', 'flexible', 'substring_match', 'number_match', and 'hybrid'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # 如果method是hybrid，使用混合数字匹配和子字符串匹配的评分逻辑
    if method == "hybrid":
        hybrid_score = compute_hybrid_score(solution_str, ground_truth)
        return hybrid_score * score
    
    # 如果method是number_match，使用基于数字匹配比例的评分逻辑
    elif method == "number_match":
        match_ratio = compute_number_match_score(solution_str, ground_truth)
        return match_ratio * score
    
    # 如果method是substring_match，使用简单的子字符串匹配逻辑
    elif method == "substring_match":
        if not ground_truth or not solution_str:
            return 0.0
        # 清理ground_truth和solution_str
        gt_clean = ground_truth.strip()
        sol_clean = solution_str.strip()
        # 检查ground_truth是否在solution中
        if gt_clean in sol_clean:
            return score
        else:
            return 0.0
    
    # 原有的逻辑保持不变
    else:
        answer = extract_solution(solution_str=solution_str, method=method)
        if answer is None:
            return 0
        else:
            if answer == ground_truth:
                return score
            else:
                return format_score
