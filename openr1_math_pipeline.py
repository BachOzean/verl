import argparse
import os
import sys
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


def _try_import(module: str, install_hint: str) -> Any:
    try:
        return __import__(module)
    except Exception as exc:  # pragma: no cover
        print(f"Missing dependency '{module}'. {install_hint}", file=sys.stderr)
        raise


datasets = _try_import(
    "datasets",
    "Install with: pip install -U datasets",
)
np = _try_import(
    "numpy",
    "Install with: pip install -U numpy",
)
regex = _try_import(
    "regex",
    "Install with: pip install -U regex",
)
sympy = _try_import(
    "sympy",
    "Install with: pip install -U sympy",
)


def _import_vllm() -> Tuple[Any, Any]:
    vllm = _try_import("vllm", "Install with: pip install -U vllm")
    from vllm import LLM, SamplingParams  # noqa: WPS433

    return LLM, SamplingParams


# ---------------------------
# Data loading utilities
# ---------------------------

DEFAULT_QUESTION_FIELDS = [
    "question",
    "problem",
    "prompt",
    "input",
    "instruction",
]
DEFAULT_ANSWER_FIELDS = [
    "answer",
    "final_answer",
    "solution",
    "target",
    "output",
]


@dataclass
class Record:
    uid: str
    question: str
    answer: str
    meta: Dict[str, Any]


def _is_path(source: str) -> bool:
    return os.path.exists(source)


def _resolve_field(example: Dict[str, Any], preferred: Optional[str], candidates: List[str]) -> Optional[str]:
    if preferred and preferred in example:
        return preferred
    for name in candidates:
        if name in example:
            return name
    return None


def load_records(
    source: str,
    split: str,
    question_field: Optional[str],
    answer_field: Optional[str],
    limit: Optional[int] = None,
) -> List[Record]:
    if _is_path(source):
        if os.path.isdir(source):
            raise ValueError("For local data, provide a JSON or JSONL file, not a directory.")
        # Auto-detect json/jsonl
        if source.endswith(".jsonl") or source.endswith(".jsonl.gz"):
            dataset = datasets.load_dataset("json", data_files=source, split="train")
        elif source.endswith(".json"):
            dataset = datasets.load_dataset("json", data_files=source, split="train")
        else:
            raise ValueError(f"Unsupported local file format for {source}. Use .json or .jsonl")
    else:
        dataset = datasets.load_dataset(source, split=split)

    records: List[Record] = []
    for idx, ex in enumerate(dataset):
        if question_field is None:
            qf = _resolve_field(ex, None, DEFAULT_QUESTION_FIELDS)
        else:
            qf = question_field if question_field in ex else None
        if answer_field is None:
            af = _resolve_field(ex, None, DEFAULT_ANSWER_FIELDS)
        else:
            af = answer_field if answer_field in ex else None
        if qf is None or af is None:
            # Skip examples lacking required fields
            continue
        uid = str(ex.get("id", idx))
        records.append(Record(uid=uid, question=str(ex[qf]), answer=str(ex[af]), meta={k: v for k, v in ex.items()}))
        if limit is not None and len(records) >= limit:
            break
    return records


# ---------------------------
# Prompt building
# ---------------------------

DEFAULT_PROMPT_TEMPLATE = (
    "You are a helpful math assistant. Solve the following problem. "
    "Provide the final answer clearly inside \\boxed{...}.\n\n"
    "Problem: {question}\n\nAnswer:"
)


def build_prompts(records: List[Record], template: str) -> List[str]:
    return [template.format(question=rec.question) for rec in records]


# ---------------------------
# Answer extraction and comparison
# ---------------------------

_BOXED_PATTERN = regex.compile(r"\\boxed\{(?P<val>.+?)\}")
_HASH_ANSWER_PATTERN = regex.compile(r"^\s*####\s*(?P<val>.+?)\s*$", flags=regex.MULTILINE)
_FINAL_ANSWER_PATTERN = regex.compile(
    r"(?i)(final\s*answer\s*[:\-]?|answer\s*[:\-]?|答案[：:])\s*(?P<val>[^\n]+)"
)
_NUM_FINDER = regex.compile(r"(?<![\d.\-/])([\-]?\d+(?:[\./]\d+)?)(?![\d.\-/])")


def extract_final_answer(text: str) -> str:
    # Prefer the last \boxed{...}
    boxed = list(_BOXED_PATTERN.finditer(text))
    if boxed:
        return boxed[-1].group("val").strip()

    # Try '#### 123' convention
    hash_matches = list(_HASH_ANSWER_PATTERN.finditer(text))
    if hash_matches:
        return hash_matches[-1].group("val").strip()

    # Try Final Answer / Answer:
    last = None
    for m in _FINAL_ANSWER_PATTERN.finditer(text):
        last = m
    if last:
        return last.group("val").strip()

    # Fallback: last number in text
    nums = list(_NUM_FINDER.finditer(text))
    if nums:
        return nums[-1].group(1).strip()

    # Fallback to last non-empty line
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return text.strip()


def _normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "")
    s = s.replace("\\,", "")
    s = re.sub(r"\s+", " ", s)
    # Remove trailing '.' if numeric-like
    if re.fullmatch(r"[-+]?\d+(?:[./]\d+)?\.?", s):
        s = s.rstrip(".")
    return s


def _to_sympy(expr: str):
    try:
        return sympy.sympify(expr, evaluate=True)
    except Exception:
        return None


def answers_equal(a_pred_raw: str, a_true_raw: str, mode: str = "simple", rtol: float = 1e-6, atol: float = 1e-8) -> bool:
    a_pred = _normalize_text(extract_final_answer(a_pred_raw))
    a_true = _normalize_text(extract_final_answer(a_true_raw))

    if mode == "simple":
        return a_pred == a_true

    # sympy mode: try symbolic equality or numeric closeness
    ap = _to_sympy(a_pred)
    at = _to_sympy(a_true)
    if ap is not None and at is not None:
        try:
            diff = sympy.simplify(ap - at)
            if diff == 0:
                return True
        except Exception:
            pass
        # numeric fallback
        try:
            apf = float(ap.evalf())
            atf = float(at.evalf())
            return math.isclose(apf, atf, rel_tol=rtol, abs_tol=atol)
        except Exception:
            return False
    return a_pred == a_true


# ---------------------------
# Inference with vLLM
# ---------------------------

def run_vllm_inference(
    model: str,
    prompts: List[str],
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: Optional[str] = None,
    max_model_len: Optional[int] = None,
    trust_remote_code: bool = False,
) -> List[List[str]]:
    LLM, SamplingParams = _import_vllm()
    llm_kwargs: Dict[str, Any] = {
        "model": model,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
    }
    if dtype:
        llm_kwargs["dtype"] = dtype
    if max_model_len:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    generations: List[List[str]] = []
    for out in outputs:
        texts = [o.text for o in out.outputs]
        generations.append(texts)
    return generations


# ---------------------------
# Orchestration
# ---------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_infer_cli(args: argparse.Namespace) -> None:
    records = load_records(
        source=args.dataset,
        split=args.split,
        question_field=args.question_field,
        answer_field=args.answer_field,
        limit=args.limit,
    )
    if not records:
        raise RuntimeError("No records loaded. Check dataset and field names.")

    ensure_dir(args.output_dir)
    raw_path = os.path.join(args.output_dir, "raw_outputs.jsonl")
    existing_ids: set = set()
    if os.path.exists(raw_path):
        if args.resume:
            # Resume: collect existing ids and skip them
            for rec in _iter_jsonl(raw_path):
                if "id" in rec:
                    existing_ids.add(str(rec["id"]))
        elif not args.overwrite:
            print(f"Refusing to overwrite existing file: {raw_path}. Use --overwrite or --resume.")
            return

    batch_size = args.batch_size
    template = args.prompt_template or DEFAULT_PROMPT_TEMPLATE

    # Filter out already processed when resuming
    if existing_ids:
        records = [r for r in records if r.uid not in existing_ids]
        print(f"Resume enabled: skipping {len(existing_ids)} already-processed items. Remaining: {len(records)}")

    mode = "a" if (args.resume and os.path.exists(raw_path)) else "w"
    with open(raw_path, mode, encoding="utf-8") as f_out:
        for start in tqdm(range(0, len(records), batch_size), desc="Inferring", ncols=100):
            batch = records[start : start + batch_size]
            prompts = build_prompts(batch, template)
            gens = run_vllm_inference(
                model=args.model,
                prompts=prompts,
                n=args.num_generations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                tensor_parallel_size=args.tp_size,
                gpu_memory_utilization=args.gpu_mem_util,
                dtype=args.dtype,
                max_model_len=args.max_model_len,
                trust_remote_code=args.trust_remote_code,
            )
            assert len(gens) == len(batch)
            for rec, texts in zip(batch, gens):
                out = {
                    "id": rec.uid,
                    "question": rec.question,
                    "answer": rec.answer,
                    "generations": texts,
                }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote raw outputs to {raw_path}")


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def run_aggregate_cli(args: argparse.Namespace) -> None:
    raw_path = args.raw
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    ensure_dir(args.output_dir)
    valid_path = os.path.join(args.output_dir, "subset_valid.jsonl")
    all_correct_path = os.path.join(args.output_dir, "subset_all_correct.jsonl")
    all_wrong_path = os.path.join(args.output_dir, "subset_all_wrong.jsonl")

    n_total = 0
    n_valid = 0
    n_all_correct = 0
    n_all_wrong = 0

    with open(valid_path, "w", encoding="utf-8") as f_valid, \
        open(all_correct_path, "w", encoding="utf-8") as f_all_correct, \
        open(all_wrong_path, "w", encoding="utf-8") as f_all_wrong:

        for rec in tqdm(_iter_jsonl(raw_path), desc="Aggregating", ncols=100):
            n_total += 1
            generations: List[str] = rec.get("generations", [])
            gt = rec.get("answer", "")
            correct_mask = [answers_equal(g, gt, mode=args.eval_mode) for g in generations]
            num_true = sum(1 for c in correct_mask if c)

            out = {
                "id": rec.get("id"),
                "question": rec.get("question"),
                "answer": gt,
                "generations": generations,
                "correct_mask": correct_mask,
                "num_correct": num_true,
                "num_total": len(generations),
            }

            if len(correct_mask) == 0:
                # Treat as invalid; do not include
                continue
            if all(correct_mask):
                n_all_correct += 1
                f_all_correct.write(json.dumps(out, ensure_ascii=False) + "\n")
            elif not any(correct_mask):
                n_all_wrong += 1
                f_all_wrong.write(json.dumps(out, ensure_ascii=False) + "\n")
            else:
                n_valid += 1
                f_valid.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "total": n_total,
                "valid": n_valid,
                "all_correct": n_all_correct,
                "all_wrong": n_all_wrong,
            },
            indent=2,
        )
    )
    print(f"Wrote: {valid_path}\nWrote: {all_correct_path}\nWrote: {all_wrong_path}")


def run_all_cli(args: argparse.Namespace) -> None:
    # Run infer then aggregate
    infer_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k in {
        "dataset", "split", "question_field", "answer_field", "limit", "output_dir", "overwrite",
        "batch_size", "prompt_template", "model", "num_generations", "max_new_tokens", "temperature",
        "top_p", "tp_size", "gpu_mem_util", "dtype", "max_model_len", "trust_remote_code"
    }})
    run_infer_cli(infer_args)
    raw_path = os.path.join(args.output_dir, "raw_outputs.jsonl")
    aggregate_args = argparse.Namespace(
        raw=raw_path,
        output_dir=args.output_dir,
        eval_mode=args.eval_mode,
    )
    run_aggregate_cli(aggregate_args)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenR1-Math-220k filtering pipeline with vLLM")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", type=str, required=True, help="HF repo id or path to JSON/JSONL file")
    common.add_argument("--split", type=str, default="train")
    common.add_argument("--question-field", type=str, default=None)
    common.add_argument("--answer-field", type=str, default=None)
    common.add_argument("--limit", type=int, default=None, help="For testing; process only N records")
    common.add_argument("--output-dir", type=str, default="outputs")

    infer = sub.add_parser("infer", parents=[common])
    infer.add_argument("--model", type=str, required=True)
    infer.add_argument("--num-generations", type=int, default=64)
    infer.add_argument("--batch-size", type=int, default=32)
    infer.add_argument("--max-new-tokens", type=int, default=512)
    infer.add_argument("--temperature", type=float, default=1.0)
    infer.add_argument("--top-p", type=float, default=0.95)
    infer.add_argument("--tp-size", type=int, default=8)
    infer.add_argument("--gpu-mem-util", type=float, default=0.90)
    infer.add_argument("--dtype", type=str, default=None, choices=["auto", "float16", "bfloat16", "float32"])
    infer.add_argument("--max-model-len", type=int, default=None)
    infer.add_argument("--trust-remote-code", action="store_true")
    infer.add_argument("--prompt-template", type=str, default=None)
    infer.add_argument("--overwrite", action="store_true")
    infer.add_argument("--resume", action="store_true", help="Resume from existing raw_outputs.jsonl")

    aggregate = sub.add_parser("aggregate", parents=[common])
    aggregate.add_argument("--raw", type=str, required=True)
    aggregate.add_argument("--eval-mode", type=str, choices=["simple", "sympy"], default="sympy")

    both = sub.add_parser("all", parents=[common])
    both.add_argument("--model", type=str, required=True)
    both.add_argument("--num-generations", type=int, default=64)
    both.add_argument("--batch-size", type=int, default=32)
    both.add_argument("--max-new-tokens", type=int, default=512)
    both.add_argument("--temperature", type=float, default=1.0)
    both.add_argument("--top-p", type=float, default=0.95)
    both.add_argument("--tp-size", type=int, default=8)
    both.add_argument("--gpu-mem-util", type=float, default=0.90)
    both.add_argument("--dtype", type=str, default=None, choices=["auto", "float16", "bfloat16", "float32"])
    both.add_argument("--max-model-len", type=int, default=None)
    both.add_argument("--trust-remote-code", action="store_true")
    both.add_argument("--prompt-template", type=str, default=None)
    both.add_argument("--eval-mode", type=str, choices=["simple", "sympy"], default="sympy")
    both.add_argument("--overwrite", action="store_true")
    both.add_argument("--resume", action="store_true")

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    t0 = time.time()
    if args.cmd == "infer":
        run_infer_cli(args)
    elif args.cmd == "aggregate":
        run_aggregate_cli(args)
    elif args.cmd == "all":
        run_all_cli(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s")


if __name__ == "__main__":
    main()

