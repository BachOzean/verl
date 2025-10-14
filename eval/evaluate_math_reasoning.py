import re
import torch
from tqdm import tqdm
from math_verify import parse, verify
from vllm import LLM, SamplingParams


# ======================================================
# 模型配置 - 使用 vLLM
# ======================================================
MODEL_PATH = "/data/home/scyb494/models/DeepSeek-R1-Distill-Qwen-1.5B"
NUM_GENERATIONS = 64
MAX_NEW_TOKENS = 4096

# 使用 vLLM 加载模型 
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    max_model_len=8192,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    dtype="auto",
    tokenizer_mode="auto",
    max_num_seqs=64
)

# 定义采样参数
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.9,
    top_k=20,
    n=NUM_GENERATIONS,
    max_tokens=MAX_NEW_TOKENS,
)

# ======================================================
# 示例数据
# ======================================================
problems = [
{"uuid": "0300afce-a176-556e-a54a-81010f22e5ce", "original_problem": "3. (20 points) The pensioners of one of the planets in the Alpha Centauri system enjoy spending their free time solving cybrow solitaire puzzles: they choose natural numbers from a certain interval $[A, B]$ in such a way that the sum of any two of the chosen numbers is not divisible by a given number $N$. Last week, the newspaper \"Alphacentauri Panorama\" offered its readers a solitaire puzzle with the values $A=1353, B=2134, N=11$. What is the maximum number of numbers that can be a solution to such a solitaire puzzle?", "original_solution": "Answer: 356.\n\nSolution: For $k=0,1, \\ldots, 10$, let $I_{k}$ be the set of all numbers in $[A, B)$ that give a remainder of $k$ when divided by 11. Since $A$ and $B$ are multiples of 11, all sets $I_{k}$ contain an equal number of elements. Therefore, all numbers in $[A, B)$ that are not multiples of 11 can be paired as $(x, y)$, where $x \\in I_{k}, y \\in I_{11-k}$ for some $k$ from $\\{1, \\ldots, 5\\}$. The number of such pairs is $\\frac{2134-1353}{11} \\cdot 5=355$. Clearly, from each pair, at most one number can be included in the final set. Additionally, this set can contain at most one number that is a multiple of 11. Thus, the solution includes no more than 356 numbers.\n\nNow, we will show that a set of 356 numbers is possible. Include $A$ and all numbers that give an odd remainder when divided by 11 in this set. Take two distinct numbers $x$ and $y$ from this set, different from each other and from $A$. Then $(x \\bmod 11) + (y \\bmod 11)$ is an even number between 2 and 18. Therefore, it is not divisible by 11, and thus $x+y$ is also not divisible by 11. Clearly, $A+x$ is also not\ndivisible by 11. This means that such a set satisfies the problem's conditions. It remains to note that this set contains $\\frac{2134-1353}{11} \\cdot 5 + 1 = 356$ numbers.", "original_answer": "356", "decomposition": "### Subproblem 1\n**Question:** For the interval \\([1353, 2134]\\), how many integers are there that are divisible by 11?  \n**Reasoning:** The original problem involves selecting numbers such that no two sum to a multiple of 11. Multiples of 11 require special handling, so determining their count is essential.  \n**Solution:**  \nThe first multiple of 11 in the interval is \\(1353\\) (since \\(1353 \\div 11 = 123\\)), and the last is \\(2134\\) (since \\(2134 \\div 11 = 194\\)). The number of multiples is \\(194 - 123 + 1 = 72\\).  \n**Answer:** 72\n\n### Subproblem 2\n**Question:** How many integers are in the interval \\([1353, 2134]\\)?  \n**Reasoning:** The total number of integers in the interval is needed to determine how many numbers are available for selection.  \n**Solution:**  \nThe count is \\(2134 - 1353 + 1 = 782\\).  \n**Answer:** 782\n\n### Subproblem 3\n**Question:** For each remainder \\(k = 1, 2, \\ldots, 10\\) modulo 11, how many integers in \\([1353, 2134]\\) have a remainder of \\(k\\) when divided by 11?  \n**Reasoning:** The condition that no two numbers sum to a multiple of 11 means that if a number has remainder \\(k\\), no number with remainder \\(11-k\\) can be selected. The counts per remainder group are needed to apply this pairing logic.  \n**Solution:**  \nSince \\(1353\\) and \\(2134\\) are multiples of 11, the interval can be partitioned into complete cycles of 11. The total number of integers is 782, and there are 72 multiples of 11 (from Subproblem 1), so the non-multiples number \\(782 - 72 = 710\\). These 710 numbers are evenly distributed among the 10 remainder classes \\(1\\) through \\(10\\), so each has \\(710 \\div 10 = 71\\) numbers.  \n**Answer:** 71\n\n### Subproblem 4\n**Question:** For each pair of remainders \\((k, 11-k)\\) where \\(k = 1, 2, \\ldots, 5\\), what is the maximum number of numbers that can be selected from the two corresponding remainder groups without having two numbers whose remainders sum to 11?  \n**Reasoning:** To avoid sums divisible by 11, at most one number can be chosen from each pair of groups \\((k, 11-k)\\). This subproblem isolates the constraint for one such pair.  \n**Solution:**  \nEach group has 71 numbers (from Subproblem 3). From the two groups combined, at most 71 numbers can be selected (all from one group, or a mix, but never more than the size of the larger group). Since the groups are equal in size, the maximum is 71.  \n**Answer:** 71\n\n### Subproblem 5\n**Question:** Considering all pairs \\((k, 11-k)\\) for \\(k = 1, 2, \\ldots, 5\\), what is the total maximum number of non-multiples of 11 that can be selected from \\([1353, 2134]\\) under the sum constraint?  \n**Reasoning:** The pairs are disjoint and independent, so the total from non-multiples is the sum of the maximums from each pair.  \n**Solution:**  \nThere are 5 pairs, and each pair contributes at most 71 numbers. Thus, the total is \\(5 \\times 71 = 355\\).  \n**Answer:** 355\n\n### Subproblem 6\n**Question:** What is the maximum number of numbers that can be selected from \\([1353, 2134]\\) such that no two sum to a multiple of 11, considering both multiples and non-multiples of 11?  \n**Reasoning:** The final answer must combine the maximum from non-multiples (Subproblem 5) with the possibility of including multiples of 11. Since any two multiples of 11 sum to a multiple of 11, at most one multiple can be included.  \n**Solution:**  \nFrom Subproblem 5, 355 non-multiples can be selected. Adding one multiple of 11 gives \\(355 + 1 = 356\\). This is achievable by selecting all numbers with odd remainders (1,3,5,7,9) and one multiple of 11.  \n**Answer:** 356", "subproblems": [{"subproblem_id": "subproblem_1", "question": "For the interval \\([1353, 2134]\\), how many integers are there that are divisible by 11?", "reasoning": "The original problem involves selecting numbers such that no two sum to a multiple of 11. Multiples of 11 require special handling, so determining their count is essential.", "solution": "The first multiple of 11 in the interval is \\(1353\\) (since \\(1353 \\div 11 = 123\\)), and the last is \\(2134\\) (since \\(2134 \\div 11 = 194\\)). The number of multiples is \\(194 - 123 + 1 = 72\\).  \n**Answer:** 72", "answer": "72"}, {"subproblem_id": "subproblem_2", "question": "How many integers are in the interval \\([1353, 2134]\\)?", "reasoning": "The total number of integers in the interval is needed to determine how many numbers are available for selection.", "solution": "The count is \\(2134 - 1353 + 1 = 782\\).  \n**Answer:** 782", "answer": "782"}, {"subproblem_id": "subproblem_3", "question": "For each remainder \\(k = 1, 2, \\ldots, 10\\) modulo 11, how many integers in \\([1353, 2134]\\) have a remainder of \\(k\\) when divided by 11?", "reasoning": "The condition that no two numbers sum to a multiple of 11 means that if a number has remainder \\(k\\), no number with remainder \\(11-k\\) can be selected. The counts per remainder group are needed to apply this pairing logic.", "solution": "Since \\(1353\\) and \\(2134\\) are multiples of 11, the interval can be partitioned into complete cycles of 11. The total number of integers is 782, and there are 72 multiples of 11 (from Subproblem 1), so the non-multiples number \\(782 - 72 = 710\\). These 710 numbers are evenly distributed among the 10 remainder classes \\(1\\) through \\(10\\), so each has \\(710 \\div 10 = 71\\) numbers.  \n**Answer:** 71", "answer": "71"}, {"subproblem_id": "subproblem_4", "question": "For each pair of remainders \\((k, 11-k)\\) where \\(k = 1, 2, \\ldots, 5\\), what is the maximum number of numbers that can be selected from the two corresponding remainder groups without having two numbers whose remainders sum to 11?", "reasoning": "To avoid sums divisible by 11, at most one number can be chosen from each pair of groups \\((k, 11-k)\\). This subproblem isolates the constraint for one such pair.", "solution": "Each group has 71 numbers (from Subproblem 3). From the two groups combined, at most 71 numbers can be selected (all from one group, or a mix, but never more than the size of the larger group). Since the groups are equal in size, the maximum is 71.  \n**Answer:** 71", "answer": "71"}, {"subproblem_id": "subproblem_5", "question": "Considering all pairs \\((k, 11-k)\\) for \\(k = 1, 2, \\ldots, 5\\), what is the total maximum number of non-multiples of 11 that can be selected from \\([1353, 2134]\\) under the sum constraint?", "reasoning": "The pairs are disjoint and independent, so the total from non-multiples is the sum of the maximums from each pair.", "solution": "There are 5 pairs, and each pair contributes at most 71 numbers. Thus, the total is \\(5 \\times 71 = 355\\).  \n**Answer:** 355", "answer": "355"}, {"subproblem_id": "subproblem_6", "question": "What is the maximum number of numbers that can be selected from \\([1353, 2134]\\) such that no two sum to a multiple of 11, considering both multiples and non-multiples of 11?", "reasoning": "The final answer must combine the maximum from non-multiples (Subproblem 5) with the possibility of including multiples of 11. Since any two multiples of 11 sum to a multiple of 11, at most one multiple can be included.", "solution": "From Subproblem 5, 355 non-multiples can be selected. Adding one multiple of 11 gives \\(355 + 1 = 356\\). This is achievable by selecting all numbers with odd remainders (1,3,5,7,9) and one multiple of 11.  \n**Answer:** 356", "answer": "356"}], "num_subproblems": 6}]


# ======================================================
# 改进的提示格式化函数
# ======================================================

def format_math_prompt(question: str, problem_type: str = "subproblem") -> str:
    """
    为数学问题创建详细的提示
    """
    if problem_type == "original":
        instruction = """Please solve the following mathematical problem step by step. 

Think carefully and show your reasoning process. At the end, provide your final answer clearly marked.

Problem: {question}

Please follow these steps:
1. Understand the problem and identify what is being asked
2. Break down the problem into logical steps
3. Show your calculations and reasoning clearly
4. Conclude with your final answer in the format: "Final Answer: [your answer]"

Solution:"""
    
    else:  # subproblem
        instruction = """Please answer the following mathematical subproblem. 

Think step by step and provide a clear, concise answer. Make sure to include your final answer at the end.

Subproblem: {question}

Please provide:
- Your reasoning (if needed)
- The final answer in the format: "Answer: [your answer]"

Solution:"""
    
    return instruction.format(question=question)

def format_complex_prompt(question: str, problem_type: str = "subproblem") -> str:
    """
    更复杂的提示格式，包含示例和更详细的指导
    """
    if problem_type == "original":
        base_prompt = """You are a mathematics expert. Solve the following problem carefully.

PROBLEM:
{question}

INSTRUCTIONS:
1. Analyze the problem thoroughly
2. Show your step-by-step reasoning
3. Provide clear explanations for each step
4. Double-check your work
5. End with "Final Answer: [your answer]" in a clear, boxed format

BEGIN SOLUTION:"""
    
    else:
        base_prompt = """Solve this mathematical subproblem:

SUBPROBLEM:
{question}

GUIDELINES:
- Think step by step
- Be precise and accurate
- End with "Answer: [your answer]"

SOLUTION:"""
    
    return base_prompt.format(question=question)

def format_cot_prompt(question: str, problem_type: str = "subproblem") -> str:
    """
    思维链风格的提示，鼓励详细推理
    """
    cot_template = """Let's think through this problem step by step.

Problem: {question}

First, I need to understand what's being asked. 

Now, let me break this down:

Step 1: [First step of reasoning]
Step 2: [Next step]
Step 3: [Continue reasoning]
...

Based on this reasoning, the answer should be:

Final Answer: """
    
    return cot_template.format(question=question)

# ======================================================
# 辅助函数
# ======================================================

def clean_answer(text: str) -> str:
    """从模型输出中提取最终答案"""
    text = text.strip()
    
    # 多种匹配模式，按优先级排序
    patterns = [
        r"(?i)final answer\s*[:：]?\s*\\boxed{([^}]*)}",  # \boxed{answer}
        r"(?i)final answer\s*[:：]?\s*\\boxed{([^}]*)}",  # \boxed{answer}
        r"(?i)answer\s*[:：]?\s*\\boxed{([^}]*)}",       # Answer: \boxed{answer}
        r"(?i)final answer\s*[:：]?\s*([^\n\\]+?)(?:\n|$)",  # Final Answer: answer
        r"(?i)answer\s*[:：]?\s*([^\n\\]+?)(?:\n|$)",        # Answer: answer
        r"\\boxed{([^}]*)}",                                # \boxed{answer}
        r"\[([^\]]*)\]",                                    # [answer]
        r"\(([^)]*)\)",                                     # (answer)
    ]
    
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            ans = m.group(1).strip()
            # 清理答案中的多余字符
            ans = re.sub(r"^\s*\\boxed{\s*|\s*}\s*$", "", ans)
            ans = re.sub(r"^\s*\\[\w]+\s*\{\s*|\s*\}\s*$", "", ans)
            if ans:
                return ans
    
    # 如果没找到特定格式，取最后一行非空文本
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        last_line = lines[-1]
        # 清理最后一行
        last_line = re.sub(r"[^\d\w\.\-\+\*/\(\)\^\{\}\\]+$", "", last_line)
        return last_line.strip()
    
    return text

def generate_multiple_answers(prompt: str, prompt_type: str = "original") -> list:
    """
    使用改进的提示生成多个回答
    """
    # 选择提示格式
    formatted_prompt = format_math_prompt(prompt, prompt_type)
    
    # 单次调用生成多个序列
    outputs = llm.generate([formatted_prompt], sampling_params)
    
    # 提取所有生成的答案
    answers = []
    for output in outputs[0].outputs:
        full_response = output.text.strip()
        answers.append({
            'full_response': full_response,
            'cleaned_answer': clean_answer(full_response)
        })
    
    return answers

def generate_answers_batch(prompts_list: list, prompt_type: str = "subproblem") -> list:
    """
    批量生成回答 - 使用改进的提示
    """
    formatted_prompts = [format_math_prompt(prompt, prompt_type) for prompt in prompts_list]
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    batch_results = []
    for output in outputs:
        prompt_answers = []
        for seq_output in output.outputs:
            full_response = seq_output.text.strip()
            prompt_answers.append({
                'full_response': full_response,
                'cleaned_answer': clean_answer(full_response)
            })
        batch_results.append(prompt_answers)
    
    return batch_results

def verify_answer(pred: str, gold: str) -> bool:
    """验证答案等价性"""
    try:
        pred_parsed = parse(pred)
        gold_parsed = parse(gold)
        return verify(pred_parsed, gold_parsed)
    except Exception:
        return pred.strip().replace(" ", "") == gold.strip().replace(" ", "")

# ======================================================
# 评测函数 - 使用改进的提示
# ======================================================

def evaluate_original_problem(problem: dict) -> float:
    """评估原始问题 - 详细输出"""
    q = problem["original_problem"]
    gold = problem["original_answer"].strip()
    
    print(f"\n{'='*80}")
    print(f"📋 原始问题评估")
    print(f"{'='*80}")
    print(f"问题: {q}")
    print(f"标准答案: {gold}")
    print(f"{'-'*80}")
    
    # 使用改进的提示生成所有答案
    results = generate_multiple_answers(q, "original")
    
    print(f"生成的 {NUM_GENERATIONS} 个答案:")
    correct = 0
    for i, result in enumerate(results, 1):
        pred = result['cleaned_answer']
        full_response = result['full_response']
        
        is_correct = verify_answer(pred, gold)
        status = "✅" if is_correct else "❌"
        if is_correct:
            correct += 1
        
        # 显示简化的答案，如果需要看完整响应可以取消注释下面的行
        print(f"  {i:2d}. {status} {pred}")
        # 如果需要查看完整响应，取消下面的注释
        # print(f"     完整响应: {full_response[:200]}..." if len(full_response) > 200 else f"     完整响应: {full_response}")
    
    acc = correct / NUM_GENERATIONS * 100
    print(f"{'-'*80}")
    print(f"正确数: {correct}/{NUM_GENERATIONS}")
    print(f"✅ UUID {problem['uuid']} — Original Acc: {acc:.2f}%")
    
    return acc

def evaluate_subproblems(problem: dict) -> float:
    """评估所有子问题 - 详细输出"""
    subproblems = problem.get("subproblems", [])
    if not subproblems:
        return 0
        
    total_acc = 0
    
    # 收集所有子问题
    subproblem_questions = [subproblem["question"] for subproblem in subproblems]
    subproblem_gold_answers = [subproblem["answer"].strip() for subproblem in subproblems]
    subproblem_ids = [subproblem["subproblem_id"] for subproblem in subproblems]
    
    # 批量生成所有子问题的答案
    all_results = generate_answers_batch(subproblem_questions, "subproblem")
    
    for i, (idx, question, gold, results) in enumerate(zip(subproblem_ids, subproblem_questions, subproblem_gold_answers, all_results)):
        print(f"\n{'='*80}")
        print(f"🔍 子问题评估: {idx}")
        print(f"{'='*80}")
        print(f"问题: {question}")
        print(f"标准答案: {gold}")
        print(f"{'-'*80}")
        
        correct = 0
        print(f"生成的 {NUM_GENERATIONS} 个答案:")
        for j, result in enumerate(results, 1):
            pred = result['cleaned_answer']
            full_response = result['full_response']
            
            is_correct = verify_answer(pred, gold)
            status = "✅" if is_correct else "❌"
            if is_correct:
                correct += 1
            
            print(f"  {j:2d}. {status} {pred}")
            # 如果需要查看完整响应，取消下面的注释
            # if not is_correct:  # 只显示错误答案的完整响应
            #     print(f"     完整响应: {full_response[:150]}..." if len(full_response) > 150 else f"     完整响应: {full_response}")
        
        acc = correct / NUM_GENERATIONS * 100
        total_acc += acc
        print(f"{'-'*80}")
        print(f"正确数: {correct}/{NUM_GENERATIONS}")
        print(f"   ├─ {idx} acc: {acc:.2f}%")

    return total_acc / len(subproblems)

# ======================================================
# 主评测逻辑
# ======================================================
print(f"Evaluating {len(problems)} problems")
for problem in tqdm(problems, desc="Evaluating problems"):
    print(f"\n{'#'*80}")
    print(f"开始评估问题: {problem['uuid']}")
    print(f"{'#'*80}")
    
    orig_acc = evaluate_original_problem(problem)
    sub_acc = evaluate_subproblems(problem)

    print(f"\n📊 UUID {problem['uuid']} — Original: {orig_acc:.2f}%, SubAvg: {sub_acc:.2f}%\n")