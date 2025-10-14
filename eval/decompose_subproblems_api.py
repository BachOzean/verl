#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decompose math problems into subproblems by calling SiliconFlow API (async parallel version).

Features:
- Asynchronous API calls via aiohttp (controlled concurrency)
- Resume from checkpoint
- Merge existing outputs
- Target filtering by problem text
- Safe append writes for each record
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
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Set

import asyncio
import aiohttp
from tqdm import tqdm

try:
    import pyarrow.parquet as pq  # type: ignore
    import pyarrow as pa  # type: ignore
except Exception:
    pq = None
    pa = None


# -------------------------
# Prompt Template
# -------------------------

def create_decomposition_prompt(problem: str, solution: str, answer: str) -> str:
    """Create a prompt that emphasizes independent, self-contained subproblems."""
    return f"""You are an expert at breaking down complex math problems into independent, self-contained subproblems. Your goal is to decompose the Original Problem into a sequence of subproblems that can each be solved independently and have clear, definite answers.

CRITICAL REQUIREMENTS:
- Each subproblem must be COMPLETELY SELF-CONTAINED and INDEPENDENT
- No subproblem should depend on the solution of previous subproblems
- Each subproblem should have a CLEAR, DEFINITE ANSWER in the **Answer:** field
- Include intermediate calculations in the **Solution:** field when needed
- The **Answer:** field should contain the final, unambiguous result for that subproblem

Inputs:
- Original Problem: {problem}
- Original Solution: {solution}
- Original Answer: {answer}

Output Format (STRICT):

### Subproblem 1
**Question:** [A completely self-contained question that addresses one aspect of the original problem]
**Reasoning:** [Why this subproblem is relevant to the original problem]
**Solution:** [The answer to this specific subproblem]
**Answer:** [The clear, definite final answer to this subproblem]


### Subproblem 2  
**Question:** [Another self-contained question addressing a different aspect]
**Reasoning:** [Why this subproblem is relevant to the original problem]
**Solution:** [The answer to this specific subproblem]
**Answer:** [The clear, definite final answer to this subproblem]


(Continue with 3-6 subproblems as needed)

KEY GUIDELINES:
- **Solution** field can contain intermediate steps and calculations
- **Answer** field must contain only the final, unambiguous result
- Each question must be answerable independently
- Answers should be numerical values, concise statements, or clear conclusions

Now produce the decomposition:
"""


# -------------------------
# API Configuration
# -------------------------

SILICONFLOW_URL = "https://api.siliconflow.cn/v1/chat/completions"


async def call_siliconcloud_api_async(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_retries: int,
) -> Optional[str]:
    """Asynchronous version of SiliconFlow API call."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": False,
    }

    for attempt in range(max_retries):
        try:
            async with session.post(SILICONFLOW_URL, headers=headers, json=payload, timeout=60) as resp:
                resp.raise_for_status()
                result = await resp.json()
                if isinstance(result, dict) and "choices" in result and result["choices"]:
                    return result["choices"][0]["message"]["content"]
                return None
        except Exception as e:
            wait = 2 ** attempt
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retry in {wait}s")
            await asyncio.sleep(wait)
    return None


# -------------------------
# Response Parsing
# -------------------------

_SUBPROBLEM_BLOCK_REGEX = re.compile(r"###\s+Subproblem\s+(\d+)(.*?)(?=(?:###\s+Subproblem\s+\d+|##|$))", re.DOTALL)
_QUESTION_REGEX = re.compile(r"\*\*Question:\*\*\s*(.*?)(?=\*\*Reasoning:\*\*)", re.DOTALL)
_REASONING_REGEX = re.compile(r"\*\*Reasoning:\*\*\s*(.*?)(?=\*\*Solution:\*\*)", re.DOTALL)
_SOLUTION_REGEX = re.compile(r"\*\*Solution:\*\*\s*(.*?)$", re.DOTALL)
_ANSWER_REGEX = re.compile(r"\*\*Answer:\*\*\s*(.*?)$", re.DOTALL)


def parse_decomposition_response(response: str) -> List[Dict[str, str]]:
    subproblems: List[Dict[str, str]] = []
    if not response:
        return subproblems

    blocks = _SUBPROBLEM_BLOCK_REGEX.findall(response)
    for num, content in blocks:
        q = _QUESTION_REGEX.search(content)
        r = _REASONING_REGEX.search(content)
        s = _SOLUTION_REGEX.search(content)
        a = _ANSWER_REGEX.search(content)
        subproblems.append({
            "subproblem_id": f"subproblem_{num}",
            "question": q.group(1).strip() if q else "",
            "reasoning": r.group(1).strip() if r else "",
            "solution": s.group(1).strip() if s else "",
            "answer": a.group(1).strip() if a else "",
        })
    return subproblems


# -------------------------
# Input Helpers
# -------------------------

def iter_parquet_records(input_path: str) -> Iterator[Dict[str, object]]:
    if pq is None:
        raise RuntimeError("pyarrow required for parquet input. Install pyarrow.")

    p = Path(input_path)
    files = []
    if p.is_file() and p.suffix.lower() == ".parquet":
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.parquet")) or sorted(p.glob("train-*.parquet"))
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

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
                yield json.loads(line)
            except Exception:
                continue


def read_target_problems(target_file: str) -> Set[str]:
    problems = set()
    with open(target_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                problem = obj.get("problem", "").strip()
                if problem:
                    problems.add(problem)
            except Exception:
                continue
    return problems


def get_processed_uuids(output_file: str) -> Set[str]:
    processed_uuids = set()
    if not os.path.exists(output_file):
        return processed_uuids
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                uuid = obj.get("uuid")
                if uuid:
                    processed_uuids.add(uuid)
            except Exception:
                continue
    return processed_uuids


# -------------------------
# Core Async Processing
# -------------------------

@dataclass
class Args:
    input_path: str
    target_file: str
    output_file: str
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    batch_size: int
    max_retries: int


async def process_single_async(session: aiohttp.ClientSession, record: Dict[str, object], args: Args) -> Optional[Dict[str, object]]:
    problem = record.get("problem", "").strip()
    solution = record.get("solution", "").strip()
    answer = record.get("answer", "").strip()
    uuid = record.get("uuid", "")

    if not problem or not answer:
        return None

    prompt = create_decomposition_prompt(problem, solution, answer)
    resp = await call_siliconcloud_api_async(
        session=session,
        api_key=args.api_key,
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_retries=args.max_retries,
    )

    if not resp:
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


def write_single_jsonl(path: str, record: Dict[str, object]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def process_all_async(records_to_process: List[Dict[str, object]], args: Args):
    connector = aiohttp.TCPConnector(limit=args.batch_size)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(args.batch_size)

        async def sem_task(rec):
            async with sem:
                res = await process_single_async(session, rec, args)
                if res:
                    write_single_jsonl(args.output_file, res)
                return res

        tasks = [asyncio.create_task(sem_task(rec)) for rec in records_to_process]

        results = []
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            res = await fut
            if res:
                results.append(res)
        return results


# -------------------------
# Main Function
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Async Subproblem Decomposition via SiliconFlow API")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--target_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--max_retries", type=int, default=3)
    args_ns = parser.parse_args()

    args = Args(
        input_path=args_ns.input_path,
        target_file=args_ns.target_file,
        output_file=args_ns.output_file,
        api_key=args_ns.api_key,
        model=args_ns.model,
        max_tokens=args_ns.max_tokens,
        temperature=args_ns.temperature,
        top_p=args_ns.top_p,
        top_k=args_ns.top_k,
        batch_size=args_ns.batch_size,
        max_retries=args_ns.max_retries,
    )

    print("Reading target problems...")
    target_problems = read_target_problems(args.target_file)
    print(f"Found {len(target_problems)} unique problems.")

    processed_uuids = get_processed_uuids(args.output_file)
    print(f"Found {len(processed_uuids)} already processed UUIDs.")

    # read source
    is_jsonl = args.input_path.lower().endswith(".jsonl")
    reader = iter_jsonl_records(args.input_path) if is_jsonl else iter_parquet_records(args.input_path)

    records_to_process = []
    for rec in reader:
        prob = rec.get("problem", "").strip()
        uid = rec.get("uuid", "")
        if prob in target_problems and uid not in processed_uuids:
            records_to_process.append(rec)

    if not records_to_process:
        print("No new records to process.")
        return

    print(f"Start async processing of {len(records_to_process)} records (parallel={args.batch_size})...")
    results = asyncio.run(process_all_async(records_to_process, args))
    print(f"Done. Success: {len(results)} / {len(records_to_process)}")


if __name__ == "__main__":
    main()
