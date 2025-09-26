#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split math solution texts into stepwise subproblems for GRPO-style training.

Features:
- Read from Parquet (.parquet) or JSON Lines (.jsonl)
- Heuristic segmentation of `solution` into steps using structure and cue words
- Generate two outputs:
  1) Decomposition JSONL: per problem steps with cues/equations/extracted values
  2) Stepwise examples JSONL: each step as a training example conditioned on previous steps

Usage:
  python split_solution_steps.py \
    --input /data/home/scyb494/data/data/train-00000-of-00010.parquet \
    --out-decomp /data/home/scyb494/data/OpenR1-Math-220k/out/decomp.jsonl \
    --out-steps /data/home/scyb494/data/OpenR1-Math-220k/out/steps.jsonl \
    --sample 200

Notes:
- The heuristics are conservative and language-agnostic-ish, but tuned for English math prose.
- You can extend CUE_WORDS and templates to better fit your corpus.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import pyarrow.parquet as pq  # type: ignore
    import pyarrow as pa  # type: ignore
except Exception:
    pq = None
    pa = None


# ------------------------
# Text segmentation config
# ------------------------

CUE_WORDS = [
    # Definition / setup
    "Let", "Define", "Set", "Introduce", "Introducing", "Consider", "Assume",
    # Using / deriving
    "Since", "Because", "By", "Using", "From", "Given",
    # Substitution / algebra
    "Substituting", "Substitute", "Plugging",
    # Results / consequence
    "Hence", "Thus", "Therefore", "Then", "So", "It follows that",
    # Solve
    "Solve", "Solving", "The solution",
    # Generic statement
    "We", "Now",
    # Chinese cues (common in CN math solutions)
    "设", "令", "记", "由", "因为", "由于", "根据", "可得", "代入", "带入", "解得", "故", "于是", "因此", "则", "从而", "求得", "由此",
]

# Compile a regex that splits a paragraph before cue words at natural boundaries
_CUE_SPLIT_REGEX = re.compile(
    # English sentence boundary + cue OR start of string + cue
    r"(?:(?<=^)|(?<=[\.;:])\s+)(?=(?:" + "|".join([re.escape(w) for w in CUE_WORDS]) + r")\b)" \
    # Chinese boundary: after 。；： or start, followed by a CN cue
    r"|(?:(?<=^)|(?<=[。；：])\s*)(?=(?:" + "|".join([re.escape(w) for w in CUE_WORDS]) + r"))"
)

_HEADING_REGEX = re.compile(r"^\s*#{1,6}\s+.+$")
_ENUM_ITEM_REGEX = re.compile(r"^\s*(?:\d+[\)\.]|[-•])\s+")
_NUM_ENUM_ITEM_REGEX = re.compile(r"^\s*(?:\d+[\)\.、]|[（(]\d+[)）])\s*")
_FENCE_REGEX = re.compile(r"^\s*```")
_MATH_FENCE_REGEX = re.compile(r"^\s*\$\$|\$\$$")

# Equation extraction: try to capture simple A = B or A = number, and keep latex-ish symbols
_EQUATION_REGEX = re.compile(r"([A-Za-z0-9_\\{}+\-^()\[\] ]+?)\s*=\s*([^\n$]+)")
_NUMERIC_ONLY_REGEX = re.compile(r"^[+-]?(?:\d+(?:/\d+)?|\d*\.\d+)$")


def normalize_var_name(var: str) -> str:
    # Convert v_{R} -> v_R, x_{1} -> x_1, remove braces for simplicity
    var = var.strip()
    var = var.replace("\\", "")
    var = var.replace("{", "").replace("}", "")
    var = re.sub(r"\s+", "", var)
    return var


def segment_solution_text(text: str, min_segment_chars: int = 12) -> List[str]:
    if not text:
        return []
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    # First pass: split into blocks using headings, blank lines, code fences, math fences
    blocks: List[str] = []
    buf: List[str] = []
    in_code_fence = False
    in_math_fence = False

    for line in lines:
        if _FENCE_REGEX.match(line):
            in_code_fence = not in_code_fence
            buf.append(line)
            continue
        if _MATH_FENCE_REGEX.match(line):
            in_math_fence = not in_math_fence
            buf.append(line)
            continue

        if not in_code_fence and not in_math_fence:
            if _HEADING_REGEX.match(line):
                if buf:
                    blocks.append("\n".join(buf).strip())
                    buf = []
                blocks.append(line.strip())
                continue
            if line.strip() == "":
                if buf:
                    blocks.append("\n".join(buf).strip())
                    buf = []
                continue

        buf.append(line)

    if buf:
        blocks.append("\n".join(buf).strip())

    # Remove trivial headings like '## Solution' '## Note'
    def is_trivial_heading(b: str) -> bool:
        lb = b.lower()
        return lb.startswith("## solution") or lb.startswith("solution:") or lb.startswith("【solution】".lower()) or lb.startswith("## note")

    blocks = [b for b in blocks if b and not is_trivial_heading(b)]

    # Second pass: split paragraphs with special handling for enumerated steps
    segments: List[str] = []
    for b in blocks:
        # We will group numbered items (e.g., "1.", "2)", "- ") with their following lines
        inner_parts: List[Tuple[str, bool]] = []  # (text, is_enumerated_group)
        non_enum_buf: List[str] = []
        enum_buf: List[str] = []
        in_enum_group = False

        b_lines = b.split("\n")
        for ln in b_lines:
            if _ENUM_ITEM_REGEX.match(ln):
                # starting a new enumerated item
                if in_enum_group and enum_buf:
                    inner_parts.append((" ".join([x.strip() for x in enum_buf if x.strip()]), True))
                    enum_buf = []
                if non_enum_buf:
                    inner_parts.append((" ".join([x.strip() for x in non_enum_buf if x.strip()]), False))
                    non_enum_buf = []
                in_enum_group = True
                enum_buf.append(ln.strip())
            else:
                if in_enum_group:
                    enum_buf.append(ln)
                else:
                    non_enum_buf.append(ln)

        # flush buffers
        if in_enum_group and enum_buf:
            inner_parts.append((" ".join([x.strip() for x in enum_buf if x.strip()]), True))
        if non_enum_buf:
            inner_parts.append((" ".join([x.strip() for x in non_enum_buf if x.strip()]), False))

        # Now, for enumerated groups, keep as single segments; for others, split by cues
        for p, is_enum in inner_parts:
            if not p:
                continue
            if is_enum:
                if len(p) >= min_segment_chars:
                    segments.append(p.strip())
            else:
                pieces = _CUE_SPLIT_REGEX.split(p)
                for s in pieces:
                    s = s.strip()
                    if len(s) >= min_segment_chars:
                        segments.append(s)

    # Final cleanup: merge extremely short math-only segments with previous
    merged: List[str] = []
    for seg in segments:
        if merged and (len(seg) < 16 or seg.strip().startswith("$")):
            merged[-1] = (merged[-1] + " " + seg).strip()
        else:
            merged.append(seg)

    return merged


def has_numbered_enumeration(text: str, min_items: int = 2) -> bool:
    if not text:
        return False
    cnt = 0
    for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _NUM_ENUM_ITEM_REGEX.match(ln):
            cnt += 1
            if cnt >= min_items:
                return True
    return False


def classify_cue(segment: str) -> str:
    s = segment.lstrip()
    for w in CUE_WORDS:
        if s.startswith(w + " ") or s.startswith(w + ":") or s == w:
            return w
    # Fallback by keywords
    low = s.lower()
    if low.startswith("we "):
        return "We"
    if low.startswith("now "):
        return "Now"
    return "Generic"


def generate_subquestion_zh(segment: str, cue: str) -> str:
    # Crude templates for CN sub-questions
    if cue in {"Let", "Define", "Set", "Introduce", "Introducing", "Assume", "Consider"}:
        return "定义/设定本题所需变量，并阐明其物理意义。"
    if cue in {"Since", "Because", "Given", "From", "By", "Using"}:
        return "基于已知条件或公式，写出该步可得的等式/关系。"
    if cue in {"Substituting", "Substitute", "Plugging"}:
        return "将前面得到的表达式代入，化简并给出结果。"
    if cue in {"Solve", "Solving", "The solution"}:
        return "解该（组）方程，给出关键变量的取值。"
    if cue in {"Hence", "Thus", "Therefore", "Then", "So", "It follows that"}:
        return "由上一步推导，给出本步的结论或数值。"
    if cue in {"We", "Now", "Generic"}:
        return "完成该步推导，并清晰写出结论或中间结果。"
    return "完成该步推导，并清晰写出结论或中间结果。"


def extract_equations_and_assignments(segment: str) -> Tuple[List[str], Dict[str, str]]:
    equations: List[str] = []
    assigns: Dict[str, str] = {}

    for m in _EQUATION_REGEX.finditer(segment):
        left_raw, right_raw = m.group(1).strip(), m.group(2).strip()
        # Skip if left looks like a LaTeX command rather than a variable/expression
        if left_raw.startswith("\\"):
            continue
        eq = f"{left_raw} = {right_raw}"
        equations.append(eq)

        # Only record assignments when LHS is a single variable token (no operators)
        # and RHS is purely numeric (integer/decimal/fraction)
        lhs_token_match = re.fullmatch(r"\s*([A-Za-z](?:[A-Za-z0-9_]|\\\\|\{|\})*)\s*", left_raw)
        if lhs_token_match and _NUMERIC_ONLY_REGEX.match(right_raw):
            var_name = normalize_var_name(lhs_token_match.group(1))
            # Guard: avoid spurious tokens originating from environment words
            if var_name.lower() not in {"left", "right", "begin", "end", "array", "cases"}:
                assigns[var_name] = right_raw

    return equations, assigns


@dataclasses.dataclass
class Step:
    idx: int
    cue: str
    text: str
    subquestion: str
    equations: List[str]
    conclusions: Dict[str, str]


def decompose_record(record: Dict[str, object], min_segment_chars: int = 12) -> Dict[str, object]:
    problem = (record.get("problem") or "").strip()
    solution = (record.get("solution") or "").strip()
    answer = (record.get("answer") or "").strip()
    uuid = (record.get("uuid") or "").strip()

    segments = segment_solution_text(solution, min_segment_chars=min_segment_chars)
    steps: List[Step] = []
    for i, seg in enumerate(segments, start=1):
        cue = classify_cue(seg)
        subq = generate_subquestion_zh(seg, cue)
        eqs, assigns = extract_equations_and_assignments(seg)
        steps.append(Step(idx=i, cue=cue, text=seg, subquestion=subq, equations=eqs, conclusions=assigns))

    return {
        "uuid": uuid,
        "problem": problem,
        "solution": solution,
        "answer": answer,
        "num_steps": len(steps),
        "steps": [dataclasses.asdict(s) for s in steps],
    }


def stepwise_examples_from_decomp(
    decomp: Dict[str, object],
    instruction_text: str = "Given the problem and previous steps, complete the next step and provide the result.",
) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    steps = decomp.get("steps") or []
    if not isinstance(steps, list):
        return examples
    problem = decomp.get("problem", "")
    uuid = decomp.get("uuid", "")

    prior: List[str] = []
    for s in steps:
        text = s.get("text", "") if isinstance(s, dict) else ""
        subq = s.get("subquestion", "") if isinstance(s, dict) else ""
        idx = s.get("idx", 0) if isinstance(s, dict) else 0
        examples.append({
            "uuid": uuid,
            "turn": idx,
            "instruction": instruction_text,
            "context": {
                "problem": problem,
                "previous_steps": prior.copy(),
            },
            "target_step": text,
        })
        prior.append(text)
    return examples


def read_parquet_records(path: str, sample: Optional[int] = None) -> Iterator[Dict[str, object]]:
    assert pq is not None, "pyarrow is required to read parquet files"
    pf = pq.ParquetFile(path)
    cols = [c for c in ["problem", "solution", "answer", "uuid"] if c in pf.schema.names]
    yielded = 0
    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=cols)
        df = table.to_pandas()
        for _, row in df.iterrows():
            rec = {c: row.get(c, None) for c in cols}
            yield rec
            yielded += 1
            if sample is not None and yielded >= sample:
                return


def read_jsonl_records(path: str, sample: Optional[int] = None) -> Iterator[Dict[str, object]]:
    yielded = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield obj
            yielded += 1
            if sample is not None and yielded >= sample:
                return


def write_jsonl(path: str, records: Iterable[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split solution into steps and export JSONL")
    parser.add_argument("--input", default=None, help="Input Parquet (.parquet) or JSONL (.jsonl)")
    parser.add_argument("--input-glob", default=None, help="Optional: glob of multiple inputs to stream over (e.g., /path/train-*.parquet)")
    parser.add_argument("--out-decomp", required=True, help="Output path: decomp (JSONL or JSON)")
    parser.add_argument("--out-steps", required=True, help="Output path: steps (JSONL or JSON)")
    parser.add_argument("--out-format", choices=["jsonl", "json"], default="jsonl", help="Output file format")
    parser.add_argument("--sample", type=int, default=None, help="Optional: sample first N records")
    parser.add_argument("--min-seg-chars", type=int, default=12, help="Minimum characters per step segment")
    parser.add_argument("--filter-enumerated-only", action="store_true", help="Keep only problems whose solution contains numbered steps")
    args = parser.parse_args()

    readers: List[Iterator[Dict[str, object]]] = []
    if args.input_glob:
        import glob
        paths = sorted(glob.glob(args.input_glob))
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext == ".parquet":
                readers.append(read_parquet_records(p))
            elif ext == ".jsonl":
                readers.append(read_jsonl_records(p))
            else:
                print(f"Skip unsupported: {p}", file=sys.stderr)
        # When using glob we ignore --sample (stream all)
        def chain_all() -> Iterator[Dict[str, object]]:
            for r in readers:
                for rec in r:
                    yield rec
        reader = chain_all()
    else:
        ipath = args.input
        ext = os.path.splitext(ipath)[1].lower()
        if ext == ".parquet":
            reader = read_parquet_records(ipath, sample=args.sample)
        elif ext == ".jsonl":
            reader = read_jsonl_records(ipath, sample=args.sample)
        else:
            print(f"Unsupported input extension: {ext}", file=sys.stderr)
            sys.exit(2)

    if args.out_format == "jsonl":
        decomp_records: List[Dict[str, object]] = []
        step_examples: List[Dict[str, object]] = []
        for rec in reader:
            if args.filter_enumerated_only:
                sol_text = (rec.get("solution") or "") if isinstance(rec, dict) else ""
                if not has_numbered_enumeration(sol_text):
                    continue
            decomp = decompose_record(rec, min_segment_chars=args.min_seg_chars)
            decomp_records.append(decomp)
            step_examples.extend(stepwise_examples_from_decomp(decomp))
        write_jsonl(args.out_decomp, decomp_records)
        write_jsonl(args.out_steps, step_examples)
        print(f"Wrote {len(decomp_records)} problems to {args.out_decomp}")
        print(f"Wrote {len(step_examples)} stepwise examples to {args.out_steps}")
    else:
        # Stream JSON arrays to avoid holding all records in memory
        os.makedirs(os.path.dirname(args.out_decomp), exist_ok=True)
        os.makedirs(os.path.dirname(args.out_steps), exist_ok=True)
        decomp_count = 0
        step_count = 0
        with open(args.out_decomp, "w", encoding="utf-8") as f_decomp, \
             open(args.out_steps, "w", encoding="utf-8") as f_steps:
            f_decomp.write("[")
            f_steps.write("[")
            first_decomp = True
            first_step = True
            for rec in reader:
                if args.filter_enumerated_only:
                    sol_text = (rec.get("solution") or "") if isinstance(rec, dict) else ""
                    if not has_numbered_enumeration(sol_text):
                        continue
                decomp = decompose_record(rec, min_segment_chars=args.min_seg_chars)
                if not first_decomp:
                    f_decomp.write(",")
                json.dump(decomp, f_decomp, ensure_ascii=False)
                first_decomp = False
                decomp_count += 1

                for step_obj in stepwise_examples_from_decomp(decomp):
                    if not first_step:
                        f_steps.write(",")
                    json.dump(step_obj, f_steps, ensure_ascii=False)
                    first_step = False
                    step_count += 1
            f_decomp.write("]")
            f_steps.write("]")
        print(f"Wrote {decomp_count} problems to {args.out_decomp}")
        print(f"Wrote {step_count} stepwise examples to {args.out_steps}")


if __name__ == "__main__":
    main()


