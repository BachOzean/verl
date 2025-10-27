#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_json_or_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Yield records from a JSON file (object or array) or JSONL file.
    - If the file contains a single JSON object, yield it.
    - If the file contains a JSON array, yield each element.
    - If the file is JSONL, yield each parsed line.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Try JSONL first by attempting to parse each non-empty line as an object
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        all_lines_json = True
        for line in lines:
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    all_lines_json = False
                    break
            except json.JSONDecodeError:
                all_lines_json = False
                break
        if all_lines_json:
            for line in lines:
                yield json.loads(line)
            return

    # Fall back to standard JSON
    parsed = json.loads(text)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                yield item
            else:
                raise ValueError("JSON array elements must be objects")
    elif isinstance(parsed, dict):
        yield parsed
    else:
        raise ValueError("Top-level JSON must be an object or array of objects")


def build_grpo_samples(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    From one original record, produce one GRPO sample per subproblem.
    Skips if 'uuid' is missing or empty (after stripping).
    """
    uuid = str(record.get("uuid") or "").strip()
    if not uuid:
        return []

    subproblems = record.get("subproblems") or []
    if not isinstance(subproblems, list):
        return []

    samples: List[Dict[str, Any]] = []
    for sub in subproblems:
        if not isinstance(sub, dict):
            continue
        sp_id = sub.get("subproblem_id")
        q = sub.get("question")
        if q is None:
            # Require at minimum a question
            continue

        # Construct a stable sample id for downstream dedup/debug
        sample_id = f"{uuid}::{sp_id}" if sp_id is not None else f"{uuid}::unknown"

        sample = {
            "sample_uuid": sample_id,
            "parent_uuid": uuid,
            "subproblem_id": sp_id,
            "question": q,
            "answer": sub.get("answer"),
            "reasoning": sub.get("reasoning"),
            "solution": sub.get("solution"),
            # Useful context fields retained for traceability/debugging
            "original_problem": record.get("original_problem"),
            "original_answer": record.get("original_answer"),
            "num_subproblems": record.get("num_subproblems"),
        }
        samples.append(sample)

    return samples


def write_jsonl(records: Iterable[Dict[str, Any]], out_path: Optional[Path]) -> None:
    out_fp = sys.stdout if out_path is None else out_path.open("w", encoding="utf-8")
    close_after = out_path is not None
    try:
        for rec in records:
            out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if close_after:
            out_fp.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GRPO samples from subproblems, filtering empty uuid.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input JSON/JSONL file")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="-",
        help="Path to output JSONL file (default: stdout)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = None if args.output == "-" else Path(args.output)

    seen_in = 0
    skipped_empty_uuid = 0
    produced = 0

    def iter_samples() -> Iterable[Dict[str, Any]]:
        nonlocal seen_in, skipped_empty_uuid, produced
        for rec in read_json_or_jsonl(in_path):
            seen_in += 1
            samples = build_grpo_samples(rec)
            if not samples:
                # Distinguish between truly empty due to uuid and other causes
                uuid_local = str(rec.get("uuid") or "").strip()
                if not uuid_local:
                    skipped_empty_uuid += 1
                continue
            for s in samples:
                produced += 1
                yield s

    write_jsonl(iter_samples(), out_path)

    # Report brief stats to stderr
    print(
        f"processed_records={seen_in} skipped_empty_uuid={skipped_empty_uuid} produced_samples={produced}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
