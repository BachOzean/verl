#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert stepwise data (steps-enum.json) to VERL GRPO Parquet.

Input format (JSON array):
  {
    "uuid": str,
    "turn": int,                 # 1-based step index
    "instruction": str,          # English instruction
    "context": {
      "problem": str,
      "previous_steps": [str, ...],
      "subquestion_hint": str
    },
    "target_step": str           # ground-truth step text
  }

Output Parquet columns (aligned with VERL examples):
  - prompt: str
  - response: str (ground truth for reward)
  - reward_model: {"style": "openr1_steps_enum", "ground_truth": step_or_final_answer}
  - group_id: int (grouped by uuid)
  - turn_index: int (0-based)
  - stage: str (simple)
  - ability: str (math)
  - meta_json: JSON string with extra metadata (uuid, orig_turn, is_final_step, final_answer, instruction_text)

Usage:
  python make_verl_grpo_from_steps.py \
    --input /data/home/scyb494/data/OpenR1-Math-220k/out/steps-enum.json \
    --output /data/home/scyb494/data/OpenR1-Math-220k/grpo_steps.parquet \
    --max_prev 8 \
    --min_response_chars 10 \
    --decomp /data/home/scyb494/data/OpenR1-Math-220k/out/decomp-enum.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd


def build_prompt(problem: str, previous_steps: List[str], instruction: str,  max_prev: int) -> str:
    prior = previous_steps if not max_prev or max_prev < 1 else previous_steps[-max_prev:]
    parts: List[str] = []
    parts.append("You are a helpful math assistant. Solve the problem step by step.")
    if problem:
        parts.append("\nProblem:\n" + problem.strip())
    if prior:
        lines = [f"{i+1}. {s.strip()}" for i, s in enumerate(prior)]
        parts.append("\nPrevious steps:\n" + "\n".join(lines))
    tail = "Now complete the next step."

    parts.append("\n" + tail)

    return "\n".join(parts).strip()


def convert_steps_json_to_rows(
    data: List[Dict[str, Any]],
    max_prev: int = 4,
    min_response_chars: int = 8,
    uuid_to_final_answer: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    # group by uuid
    uuid_to_items: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        uuid = str(item.get("uuid", ""))
        uuid_to_items.setdefault(uuid, []).append(item)

    # deterministic order: sort uuid groups and turns
    rows: List[Dict[str, Any]] = []
    for gid, uuid in enumerate(sorted(uuid_to_items.keys())):
        items = uuid_to_items[uuid]
        items.sort(key=lambda x: int(x.get("turn", 0)))
        last_turn = max(int(x.get("turn", 0)) for x in items) if items else 0
        final_ans = ""
        if uuid_to_final_answer is not None:
            final_ans = uuid_to_final_answer.get(uuid, "") or ""
        for item in items:
            ctx = item.get("context", {}) or {}
            problem = ctx.get("problem", "") or ""
            prev = ctx.get("previous_steps", []) or []
            if not isinstance(prev, list):
                prev = []
            instr = item.get("instruction", "") or ""
            resp = item.get("target_step", "") or ""
            if not resp or len(resp.strip()) < min_response_chars:
                continue
            prompt = build_prompt(problem, prev, instr, max_prev=max_prev)
            turn = int(item.get("turn", 1)) - 1
            is_final_step = int(item.get("turn", 0)) >= last_turn and last_turn > 0
            meta = {
                "uuid": uuid,
                "orig_turn": item.get("turn", 1),
                "instruction_text": instr,
                "is_final_step": is_final_step,
                "final_answer": final_ans,
            }
            reward_target = (final_ans.strip() if is_final_step and final_ans else resp.strip())
            rows.append({
                "data_source": "openr1-math-220k-steps-enum",
                "prompt": prompt,
                "response": resp.strip(),
                "group_id": gid,
                "turn_index": max(turn, 0),
                "stage": "simple",
                "ability": "math",
                "reward_model": {"style": "openr1_steps_enum", "ground_truth": reward_target},
                "meta_json": json.dumps(meta, ensure_ascii=False),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Convert steps-enum.json to VERL GRPO Parquet")
    ap.add_argument("--input", required=True, help="Input steps JSON (array)")
    ap.add_argument("--output", required=True, help="Output Parquet path")
    ap.add_argument("--max_prev", type=int, default=-1, help="Max previous steps included in prompt (<=0 for all)")
    ap.add_argument("--min_response_chars", type=int, default=10, help="Filter out too-short responses")
    ap.add_argument("--decomp", default=None, help="Optional: decomp JSON with uuid->answer mapping to inject final answers")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array of objects")

    uuid_to_final_answer: Dict[str, str] | None = None
    if args.decomp:
        if not os.path.exists(args.decomp):
            raise FileNotFoundError(f"Decomp file not found: {args.decomp}")
        with open(args.decomp, "r", encoding="utf-8") as f:
            decomp_data = json.load(f)
        if isinstance(decomp_data, list):
            uuid_to_final_answer = {}
            for obj in decomp_data:
                if isinstance(obj, dict):
                    u = str(obj.get("uuid", ""))
                    a = obj.get("answer", "") or ""
                    if u:
                        uuid_to_final_answer[u] = str(a)

    rows = convert_steps_json_to_rows(
        data,
        max_prev=args.max_prev,
        min_response_chars=args.min_response_chars,
        uuid_to_final_answer=uuid_to_final_answer,
    )
    if not rows:
        raise RuntimeError("No rows produced from input data")

    df = pd.DataFrame(rows)
    # Ensure column order
    cols = [
        "data_source", "prompt", "response", "group_id", "turn_index",
        "stage", "ability", "reward_model", "meta_json"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()