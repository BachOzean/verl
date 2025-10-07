#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Decompose math problems into subproblems by calling SiliconFlow API.

Enhanced with:
- Matching problems from source data with target JSONL file
- Resume from checkpoint functionality
- Progress saving after each successful processing
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

import requests

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
Subproblem Decomposition

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
    top_k: int,
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
        "top_k": top_k,
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


def read_target_problems(target_file: str) -> Set[str]:
    """Read target JSONL file and extract unique problem texts."""
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
            except Exception as e:
                print(f"Warning: Failed to parse line in target file: {e}", file=sys.stderr)
    return problems


def get_processed_uuids(output_file: str) -> Set[str]:
    """Get UUIDs that have already been processed from output file."""
    processed_uuids = set()
    if not os.path.exists(output_file):
        return processed_uuids
    
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uuid = obj.get("uuid")
                if uuid:
                    processed_uuids.add(uuid)
            except Exception as e:
                print(f"Warning: Failed to parse line in output file: {e}", file=sys.stderr)
    return processed_uuids


def get_existing_output_files(output_dir: str, base_name: str) -> List[str]:
    """Find existing output files that match the pattern."""
    output_path = Path(output_dir)
    pattern = f"{base_name}*.jsonl"
    return sorted(output_path.glob(pattern))


def merge_existing_outputs(output_files: List[str], final_output: str) -> Set[str]:
    """Merge existing output files into one and return processed UUIDs."""
    processed_uuids = set()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    
    # Merge all existing files
    with open(final_output, "w", encoding="utf-8") as outfile:
        for file_path in output_files:
            if file_path == final_output:
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        line = line.strip()
                        if line:
                            try:
                                obj = json.loads(line)
                                uuid = obj.get("uuid")
                                if uuid and uuid not in processed_uuids:
                                    outfile.write(line + "\n")
                                    processed_uuids.add(uuid)
                            except Exception as e:
                                print(f"Warning: Failed to parse line in {file_path}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}", file=sys.stderr)
    
    return processed_uuids


# --------------
# Core processor
# --------------

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
    delay: float


def process_single(record: Dict[str, object], args: Args) -> Optional[Dict[str, object]]:
    problem = record.get("problem", "").strip()
    solution = record.get("solution", "").strip()
    answer = record.get("answer", "").strip()
    uuid = record.get("uuid", "")
    
    if not problem or not answer:
        print(f"Skip record with empty problem or answer: {uuid}", file=sys.stderr)
        return None
    
    print(f"Processing UUID {uuid}: {problem[:80].replace(chr(10), ' ')}...")
    prompt = create_decomposition_prompt(problem, solution, answer)
    resp = call_siliconcloud_api(
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
        print(f"API returned empty/failed for UUID {uuid}, skip.", file=sys.stderr)
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
    """Write a single record to JSONL file (append mode)."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Subproblem decomposition via SiliconFlow API with resume capability")
    parser.add_argument("--input_path", type=str, required=True, 
                       help="Source Parquet directory/file containing full dataset")
    parser.add_argument("--target_file", type=str, required=True,
                       help="Target JSONL file containing problems to process (subset of source)")
    parser.add_argument("--output_file", type=str, required=True, 
                       help="Output JSONL file (fixed name for resume functionality)")
    parser.add_argument("--api_key", type=str, required=True, help="SiliconFlow API key")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="Model name")
    parser.add_argument("--max_tokens", type=int, default=1024, 
                       help="Max generation tokens (recommended: 1024 for 3–8 subproblems)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size (logical loop only)")
    parser.add_argument("--max_retries", type=int, default=3, help="Max API retries")
    parser.add_argument("--delay", type=float, default=0.6, help="Delay seconds between calls")
    parser.add_argument("--merge_existing", action="store_true", 
                       help="Merge existing output files before starting")
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
        delay=args_ns.delay,
    )

    # Step 0: Merge existing output files if requested
    if args_ns.merge_existing:
        output_dir = os.path.dirname(args.output_file)
        base_name = os.path.basename(args.output_file).replace(".jsonl", "")
        existing_files = get_existing_output_files(output_dir, base_name)
        
        if existing_files:
            print(f"Found {len(existing_files)} existing output files, merging...")
            processed_uuids = merge_existing_outputs([str(f) for f in existing_files], args.output_file)
            print(f"Merged {len(processed_uuids)} records into {args.output_file}")
        else:
            print("No existing output files found to merge")

    # Step 1: Read target problems
    print("Reading target problems...")
    target_problems = read_target_problems(args.target_file)
    print(f"Found {len(target_problems)} unique problems in target file")

    # Step 2: Get already processed UUIDs from the fixed output file
    processed_uuids = get_processed_uuids(args.output_file)
    print(f"Found {len(processed_uuids)} already processed UUIDs in {args.output_file}")

    # Step 3: Read source data and filter for target problems
    print("Reading source data and filtering for target problems...")
    source_path = args.input_path
    is_jsonl = source_path.lower().endswith(".jsonl")

    if is_jsonl:
        reader = iter_jsonl_records(source_path)
    else:
        reader = iter_parquet_records(source_path)

    records_to_process = []
    source_count = 0
    matched_count = 0

    for rec in reader:
        source_count += 1
        problem = rec.get("problem", "").strip()
        uuid = rec.get("uuid", "")
        
        if problem in target_problems and uuid not in processed_uuids:
            records_to_process.append(rec)
            matched_count += 1

    print(f"Source data: {source_count} total records, {matched_count} matched target problems (not processed)")

    if not records_to_process:
        print("No new records to process. Exiting.")
        return

    # Step 4: Process records with resume capability
    processed = 0
    success = 0

    print(f"Starting to process {len(records_to_process)} new records...")
    
    for i, rec in enumerate(records_to_process):
        result = process_single(rec, args)
        processed += 1
        
        if result is not None:
            write_single_jsonl(args.output_file, result)
            success += 1
            print(f"Progress: {i+1}/{len(records_to_process)} - Success: {success}")

        if args.delay > 0:
            time.sleep(args.delay)

        # Periodic status update
        if (i + 1) % 10 == 0:
            print(f"Checkpoint: processed {i+1}/{len(records_to_process)}, success: {success}")

    print("Done.")
    print(f"Total processed: {processed}")
    print(f"Total success:   {success}")
    print(f"Total failed:    {processed - success}")


if __name__ == "__main__":
    main()