#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decompose math problems into subproblems by calling SiliconFlow API.

Features:
- Read from Parquet shards directory or JSONL file
- English prompt with strict structure and English labels
- Robust parsing of "### Subproblem N" blocks with Question/Reasoning/Solution
- Batched processing with retries, delay, and progress logs

Example:
  python -m verl.eval.decompose_subproblems_api \
    --input_path /data/home/scyb494/.cache/huggingface/hub/datasets--open-r1--OpenR1-Math-220k/snapshots/e4e141ec9dea9f8326f4d347be56105859b2bd68/data \
    --output_file /data/home/scyb494/outputs/subproblems_openr1.jsonl \
    --api_key YOUR_SILICONFLOW_KEY \
    --model Qwen/Qwen2.5-72B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import requests
import pandas as pd

try:
    import pickle
except ImportError:
    pickle = None

try:
    import pyarrow.parquet as pq  # type: ignore
    import pyarrow as pa  # type: ignore
except Exception:
    pq = None
    pa = None


# -------------------------
# Prompt (English + labels)
# -------------------------

def create_decomposition_prompt(problem: str, solution: str, answer: str) -> str:
    """Create an English prompt with English labels and strict formatting."""
    return f"""You are a meticulous math assistant. Decompose the Original Problem into a logically ordered sequence of subproblems. Each subproblem must be self-contained, contribute to the overall solution, and include a concrete solution. The final subproblem's solution must exactly equal or directly yield the Original Answer.

Inputs:
- Original Problem: {problem}
- Original Solution: {solution}
- Original Answer: {answer}

Hard formatting constraints (follow literally):
- Output ONLY the structure below. No extra text, no preface, no summary, no lists, no tables, no code blocks, no backticks.
- Use EXACTLY the following headings and labels (English), and write all content after each label IN ENGLISH.
- Top line: Subproblem Decomposition
- Each subproblem uses a level-3 heading: ### Subproblem N (N is an Arabic numeral, starting from 1 and strictly increasing by 1).
- Under each subproblem, include EXACTLY three bold labels in this exact order, with ASCII colons and non-empty content:
  1) **Problem:**
  2) **Solution:**
  3) **Answer:**
- Do NOT use any other headings (e.g., "## ...").
- Prefer 3–8 subproblems; use fewer only if the problem is truly simple, but ensure the full derivation path is covered.
- The “Solution” field must be a clear, concrete solution for that subproblem. For proof-style tasks, it can be the proved claim for that step.
- The last subproblem’s “Solution” must equal or directly imply the Original Answer (state the equivalence or simplification if needed within the reasoning).

Now produce the output in the exact structure below (do NOT keep placeholders, do NOT add anything else):

Subproblem Decomposition

### Subproblem 1
**Problem:**
**Solution:**
**Answer:**

### Subproblem 2
**Problem:**
**Solution:**
**Answer:**

### Subproblem 3
**Problem:**
**Solution:**
**Answer:**

(Continue with more subproblems if necessary until the solution path is complete)
"""


# -----------------
# Progress management
# -----------------

@dataclass
class ProgressState:
    """进度状态类"""
    processed_count: int = 0
    success_count: int = 0
    current_record_uuid: str = ""
    start_time: float = 0.0

def save_progress(progress_file: str, state: ProgressState) -> None:
    """保存进度到文件"""
    if not pickle:
        print("Warning: pickle not available, cannot save progress", file=sys.stderr)
        return

    try:
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        with open(progress_file, 'wb') as f:
            pickle.dump(state, f)
    except Exception as e:
        print(f"Warning: Failed to save progress: {e}", file=sys.stderr)

def load_progress(progress_file: str) -> Optional[ProgressState]:
    """从文件加载进度"""
    if not pickle:
        print("Warning: pickle not available, cannot load progress", file=sys.stderr)
        return None

    if not os.path.exists(progress_file):
        return None

    try:
        with open(progress_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Failed to load progress: {e}", file=sys.stderr)
        return None

def cleanup_progress(progress_file: str) -> None:
    """清理进度文件"""
    try:
        if os.path.exists(progress_file):
            os.remove(progress_file)
    except Exception as e:
        print(f"Warning: Failed to cleanup progress file: {e}", file=sys.stderr)

# -----------------
# API client helper
# -----------------

SILICONFLOW_URL = "https://api.siliconflow.cn/v1/chat/completions"


def call_siliconcloud_api(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_retries: int,
    timeout: int = 60,
) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(SILICONFLOW_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, dict) and "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"]
            return None
        except Exception as e:
            wait = 2 ** attempt
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retry in {wait}s", file=sys.stderr)
            if attempt < max_retries - 1:
                time.sleep(wait)
    return None


# --------------
# Parse response
# --------------

_SUBPROBLEM_BLOCK_REGEX = re.compile(r"###\s+Subproblem\s+(\d+)(.*?)(?=(?:###\s+Subproblem\s+\d+|##|$))", re.DOTALL)
_PROBLEM_REGEX = re.compile(r"\*\*Problem:\*\*\s*(.*?)(?=\*\*Solution:\*\*)", re.DOTALL)
_SOLUTION_REGEX = re.compile(r"\*\*Solution:\*\*\s*(.*?)(?=\*\*Answer:\*\*)", re.DOTALL)
_ANSWER_REGEX = re.compile(r"\*\*Answer:\*\*\s*(.*?)$", re.DOTALL)


def parse_decomposition_response(response: str) -> List[Dict[str, str]]:
    subproblems: List[Dict[str, str]] = []
    if not response:
        return subproblems

    blocks = _SUBPROBLEM_BLOCK_REGEX.findall(response)
    for num, content in blocks:
        p = _PROBLEM_REGEX.search(content)
        s = _SOLUTION_REGEX.search(content)
        a = _ANSWER_REGEX.search(content)       
        subproblems.append({
            "subproblem_id": f"subproblem_{num}",
            "problem": p.group(1).strip() if p else "",
            "solution": s.group(1).strip() if s else "",
            "answer": a.group(1).strip() if a else "",
        })
    return subproblems


# ---------------------
# Input reading helpers
# ---------------------

def iter_parquet_records(input_path: str) -> Iterator[Dict[str, object]]:
    if pq is None:
        raise RuntimeError("pyarrow is required to read parquet files. Please install pyarrow.")

    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == ".parquet":
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if not files:
            # Also try train-* pattern
            files = sorted(p.glob("train-*.parquet"))
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not files:
        raise FileNotFoundError(f"No parquet files found under: {input_path}")

    for f in files:
        pf = pq.ParquetFile(str(f))
        cols_present = [c for c in ["problem", "answer", "solution", "uuid"] if c in pf.schema.names]
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=cols_present)
            df = table.to_pandas()
            for _, row in df.iterrows():
                yield {c: row.get(c, None) for c in cols_present}


def iter_jsonl_records(input_file: str) -> Iterator[Dict[str, object]]:
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield obj


# --------------
# Core processor
# --------------

@dataclass
class Args:
    input_path: str
    output_file: str
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    batch_size: int
    max_retries: int
    delay: float
    enable_resume: bool
    progress_file: str
    save_interval: int


def process_single(uuid: str, problem: str, solution: str, answer: str, args: Args) -> Optional[Dict[str, object]]:
    print(f"Processing uuid: {uuid}...")
    prompt = create_decomposition_prompt(problem, solution, answer)
    resp = call_siliconcloud_api(
        api_key=args.api_key,
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_retries=args.max_retries,
    )
    if not resp:
        print("API returned empty/failed, skip.", file=sys.stderr)
        return None
    subproblems = parse_decomposition_response(resp)
    return {
        "uuid": uuid,
        "original_problem": problem,
        "original_solution": solution,
        "original_answer": answer,
        "decomposition": resp,
        "subproblems": subproblems,
        "num_subproblems": len(subproblems),
    }


def write_jsonl(path: str, records: Iterable[Dict[str, object]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Subproblem decomposition via SiliconFlow API")
    parser.add_argument("--input_path", type=str, required=True, help="Parquet directory/file or JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--api_key", type=str, required=True, help="SiliconFlow API key")
    parser.add_argument("--model", type=str, default="None", help="Model name")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max generation tokens (recommended: 1024 for 3–8 subproblems)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size (logical loop only)")
    parser.add_argument("--max_retries", type=int, default=3, help="Max API retries")
    parser.add_argument("--delay", type=float, default=0.6, help="Delay seconds between calls")
    parser.add_argument("--enable_resume", action="store_true", help="Enable resume from interruption")
    parser.add_argument("--progress_file", type=str, default=None, help="Progress file path (default: <output_file>.progress)")
    parser.add_argument("--save_interval", type=int, default=10, help="Save progress every N processed records")
    args_ns = parser.parse_args()

    # 设置进度文件路径
    progress_file = args_ns.progress_file or f"{args_ns.output_file}.progress"

    args = Args(
        input_path=args_ns.input_path,
        output_file=args_ns.output_file,
        api_key=args_ns.api_key,
        model=args_ns.model,
        max_tokens=args_ns.max_tokens,
        temperature=args_ns.temperature,
        top_p=args_ns.top_p,
        batch_size=args_ns.batch_size,
        max_retries=args_ns.max_retries,
        delay=args_ns.delay,
        enable_resume=args_ns.enable_resume,
        progress_file=progress_file,
        save_interval=args_ns.save_interval,
    )

    ip = args.input_path
    is_jsonl = ip.lower().endswith(".jsonl")

    if is_jsonl:
        reader = iter_jsonl_records(ip)
    else:
        reader = iter_parquet_records(ip)

    # 加载进度状态
    progress_state = None
    if args.enable_resume:
        progress_state = load_progress(args.progress_file)
        if progress_state:
            print(f"Resuming from previous state: processed={progress_state.processed_count}, success={progress_state.success_count}")
        else:
            print("No previous progress found, starting from beginning")

    processed = progress_state.processed_count if progress_state else 0
    success = progress_state.success_count if progress_state else 0
    out_buffer: List[Dict[str, object]] = []

    # 跳过已处理的记录（仅对JSONL文件有效）
    if progress_state and progress_state.processed_count > 0:
        print(f"Skipping {progress_state.processed_count} already processed records...")
        if is_jsonl:
            # 对于JSONL文件，我们可以跳过前N行
            skipped = 0
            temp_reader = iter_jsonl_records(ip)
            for rec in temp_reader:
                if skipped >= progress_state.processed_count:
                    break
                skipped += 1
            # 重新创建reader从正确位置开始
            reader = iter_jsonl_records(ip)
            for _ in range(progress_state.processed_count):
                next(reader)
        else:
            print("Warning: Resume functionality is limited for Parquet files. Consider using JSONL for full resume support.")
            # 对于Parquet文件，我们无法简单跳过，所以重置计数器但保留输出缓冲区逻辑

    # 初始化进度状态
    if args.enable_resume and not progress_state:
        progress_state = ProgressState(
            processed_count=0,
            success_count=0,
            current_record_uuid="",
            start_time=time.time()
        )

    # 提供源数据地址/home/ningmiao/ningyuan/verl/data/OpenR1-Math-220k/data，找到problem对应的uuid和solution

    source_data_path = "/home/ningmiao/ningyuan/verl/data/OpenR1-Math-220k/data"
    # 预加载源数据，避免每次都重复读取
    full_source_data = pd.read_parquet(source_data_path)

    for rec in reader:
        problem = (rec.get("problem") or "") if isinstance(rec, dict) else ""

        # 为每个problem重新过滤完整的数据，避免DataFrame被修改的问题
        filtered_data = full_source_data[full_source_data["problem"] == problem]

        if filtered_data.empty:
            print(f"Warning: No matching problem found in source data for problem: {problem[:100]}...")
            processed += 1

        uuid = filtered_data["uuid"].values[0]
        solution = filtered_data["solution"].values[0]
        answer = (rec.get("answer") or "") if isinstance(rec, dict) else ""

        result = process_single(uuid, problem, solution, answer, args)
        processed += 1
        if result is not None:
            out_buffer.append({
                "original_problem": result["original_problem"],
                "original_solution": result["original_solution"],
                "original_answer": result["original_answer"],
                "decomposition": result["decomposition"],
                "subproblems": result["subproblems"],
                "num_subproblems": result["num_subproblems"],
            })
            success += 1

        # 定期保存进度
        if processed % args.save_interval == 0 and args.enable_resume:
            progress_state.processed_count = processed
            progress_state.success_count = success
            progress_state.current_record_uuid = uuid
            save_progress(args.progress_file, progress_state)

        # 延迟和进度显示
        if args.delay > 0:
            time.sleep(args.delay)

        if processed % 10 == 0:
            print(f"Progress: processed={processed}, success={success}")

        # 如果有之前的进度，追加到现有文件
        if os.path.exists(args.output_file):
            print(f"Appending to existing output file: {args.output_file}")
            with open(args.output_file, "r", encoding="utf-8") as f:
                existing_data = f.read().strip()
                if existing_data:
                    # 如果文件有内容，追加新内容
                    with open(args.output_file, "a", encoding="utf-8") as f:
                        for record in out_buffer:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    # 如果文件为空，直接写入
                    write_jsonl(args.output_file, out_buffer)
            # 清空缓冲区，避免重复写入
            out_buffer.clear()
        else:
            write_jsonl(args.output_file, out_buffer)
            # 清空缓冲区，避免重复写入
            out_buffer.clear()
    else:
        write_jsonl(args.output_file, out_buffer)
        # 清空缓冲区，避免重复写入
        out_buffer.clear()

    print("Done.")
    print(f"Total processed: {processed}")
    print(f"Total success:   {success}")
    print(f"Total failed:    {processed - success}")

    # 清理进度文件（成功完成时）
    if args.enable_resume:
        cleanup_progress(args.progress_file)
        print("Progress file cleaned up.")


if __name__ == "__main__":
    main()