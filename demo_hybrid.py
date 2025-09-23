#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示混合奖励函数的效果
"""
import sys
import os

# 添加verl路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from verl.utils.reward_score.gsm8k import compute_score

def demo_hybrid_reward():
    """演示混合奖励函数的效果"""
    print("混合奖励函数演示")
    print("=" * 50)
    
    # 从日志中提取的真实案例
    solution = "Alright, so I've been stuck trying to figure out this problem for a while. The question is: Solve 2x + 3y = 6 using substitution, given that |x| < 1 and |y| < 1. Okay, I remember substitution from algebra class, but I'm not entirely sure how to apply it here. Let me try to recall. Substitution is when you solve one equation for one variable and then plug that into the other equation. So I need to express either x or y in terms of the other variable. Given 2x + 3y = 6, maybe I can solve for x. Let me try that. First, subtract 3y from both sides: 2x = 6 - 3y. Then, divide both sides by 2: x = (6 - 3y)/2. Okay, so now I have x expressed in terms of y. But I also have the constraints that |x| < 1 and |y| < 1. That means both x and y have to be between -1 and 1. So substituting x into the constraints might give me some bounds on y, and then I can back into the possible values of y and x. Hmm, sort of similar to solving systems of inequalities. Let me write down the constraints for x: |x| < 1 implies -1 < x < 1. But x is expressed in terms of y. So substituting, I get: -1 < (6 - 3y)/2 < 1. I can solve this compound inequality for y. First, split the inequality into two parts: 1. -1 < (6 - 3y)/2 2. (6 - 3y)/2 < 1. Let's solve the first inequality: Multiply both sides by 2: -2 < 6 - 3y. Subtract 6 from both sides: -8 < -3y. Divide by -3 and reverse the inequality sign: 8/3 > y or y < 8/3. Wait, is that right? Because when I divide by a negative, the inequality flips. So yeah, 8/3 > y. Now, the second inequality: (6 - 3y)/2 < 1. Multiply both sides by 2: 6 - 3y < 2. Subtract 6: -3y < -4. Divide by -3, reverse inequality: y > 4/3. Putting the two inequalities together: 4/3 < y < 8/3. But wait a second. This is for y, which was supposed to satisfy |y| < 1. If y < 1, but according to this, y has to be greater than 4/3, which is approximately 1.333, so that's more than 1. But the constraint is that |y| < 1, meaning y must be between -1 and 1. So this is a problem. If y must be between -1 and 1, but the substitution gives y > 4/3, which is outside of that range. Alternatively, maybe I did the transformation incorrectly when solving the inequalities. Let me double-check the first inequality: Starting with: -1 < (6 - 3y)/2 < 1. Multiply all parts by 2: -2 < 6 - 3y < 2. Now subtract 6: -8 < -3y < -4. Divide by -3, which flips the inequalities: 8/3 > y > 4/3. Which is the same as 4/3 < y < 8/3, which as I thought earlier, is outside of the constraints. Similarly, if I consider |x| < 1 again, let's plug x = (6 - 3y)/2 into that. So |(6 - 3y)/2| < 1. Multiply both sides by 2: |6 - 3y| < 2. Which implies: -2 < 6 - 3y < 2. Subtract 6: -8 < -3y < -4. Divide by -3, flip inequalities: 8/3 > y > 4/3. Which is the same thing. So that's consistent. But 4/3 is approximately 1.333, which is outside the allowed"
    
    ground_truth = "It takes 50/2=25 jellybeans to fill up a small drinking glass."
    
    print(f"Solution (前100字符): {solution[:100]}...")
    print(f"Ground Truth: {ground_truth}")
    print()
    
    # 测试不同的方法
    methods = ["strict", "substring_match", "number_match", "hybrid"]
    
    for method in methods:
        try:
            score = compute_score(solution, ground_truth, method=method)
            print(f"{method:15}: {score:.4f}")
        except Exception as e:
            print(f"{method:15}: Error - {e}")
    
    print()
    print("分析:")
    print("- strict: 0.0000 (没有找到####格式的答案)")
    print("- substring_match: 0.0000 (ground_truth不在solution中)")
    print("- number_match: 0.0000 (solution中没有ground_truth中的数字)")
    print("- hybrid: 0.0000 (综合评分，这里两个组件都是0)")
    print()
    print("这个案例展示了为什么需要更智能的奖励函数！")

if __name__ == "__main__":
    demo_hybrid_reward()


