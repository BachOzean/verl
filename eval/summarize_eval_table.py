#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
汇总 /eval/results 下 {aime24,gsm8k,math500} 三个任务中，各模型的：
- avg_steps_trimmed（来自 details 统计：我们重用 analyze_step_counts 的逻辑计算）
- extractive_match（来自 lighteval 输出 results/*.json）

输出：纯文本表格（制表符分隔），便于直接复制到 Google Docs（粘贴后转表格）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

# 复用已实现的计步逻辑
sys.path.append(str(Path(__file__).parent))
from analyze_step_counts import (
    iter_detail_files,
    analyze_paths,
    summarize,
)


TASK_DIRS = ["aime24", "gsm8k", "math500"]


def find_latest_results_json(task_dir: Path) -> List[Path]:
    out: List[Path] = []
    results_dir = task_dir / "results"
    if not results_dir.exists():
        return out
    # 递归收集所有 results_*.json
    for p in results_dir.rglob("results_*.json"):
        out.append(p)
    return sorted(out)


def parse_acc_from_results_json(p: Path) -> Optional[Tuple[str, float]]:
    """返回 (model_key, acc)；model_key 取 results 文件中的上层目录名（模型名）。"""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    # 解析 results 中的第一个任务名，取 extractive_match
    acc: Optional[float] = None
    results_obj = data.get("results", {})
    for task_name, metrics in results_obj.items():
        if isinstance(metrics, dict) and "extractive_match" in metrics:
            try:
                acc = float(metrics["extractive_match"]) * 100.0  # 转百分比
                break
            except Exception:
                pass

    if acc is None:
        return None

    # 模型 key：results/<anchor>/results_*.json 的 <anchor>
    try:
        model_key = p.parent.name
    except Exception:
        model_key = "unknown"
    return model_key, acc


def compute_steps_trimmed(task_dir: Path, mode: str, exclude_connectives: List[str], trim_frac: float) -> Dict[str, float]:
    """遍历 details 下各模型目录，针对每个模型统计 avg_steps_trimmed。"""
    # collect details files by model anchor
    details_root = task_dir / "details"
    by_model: Dict[str, List[Path]] = {}
    if details_root.exists():
        for f in iter_detail_files(details_root):
            # anchor = details/<anchor>/... 取 <anchor>
            parts = f.parts
            try:
                idx = parts.index("details")
                anchor = parts[idx + 1]
            except Exception:
                anchor = "unknown"
            by_model.setdefault(anchor, []).append(f)

    res: Dict[str, float] = {}
    for model_key, files in by_model.items():
        stats = analyze_paths(files, mode=mode, exclude_connectives=exclude_connectives)
        summ = summarize(stats, trim_frac=trim_frac)
        val = float(summ.get("avg_steps_trimmed", 0.0))
        res[model_key] = val
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总三个任务的步数与准确率")
    parser.add_argument("root", type=str, default="/data/home/scyb494/verl/eval/results", nargs="?")
    parser.add_argument("--mode", type=str, default="connectives-only")
    parser.add_argument("--exclude-connectives", type=str, default="so")
    parser.add_argument("--trim-frac", type=float, default=0.05)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    exclude_connectives = [t.strip().lower() for t in args.exclude_connectives.split(",") if t.strip()]

    rows: List[Tuple[str, str, float, float]] = []  # (task, model, avg_steps_trimmed, acc)

    for task in TASK_DIRS:
        task_dir = root / task
        if not task_dir.exists():
            continue

        # 1) 步数（avg_steps_trimmed）
        steps_map = compute_steps_trimmed(task_dir, mode=args.mode, exclude_connectives=exclude_connectives, trim_frac=args.trim_frac)

        # 2) 准确率（extractive_match）
        acc_map: Dict[str, float] = {}
        for p in find_latest_results_json(task_dir):
            parsed = parse_acc_from_results_json(p)
            if not parsed:
                continue
            model_key, acc = parsed
            acc_map[model_key] = acc

        # 3) 合并（以 steps_map 的模型为基准，如需要也可并集）
        model_keys = sorted(set(steps_map.keys()) | set(acc_map.keys()))
        for mk in model_keys:
            rows.append((task, mk, steps_map.get(mk, 0.0), acc_map.get(mk, float("nan"))))

    # 输出：制表符分隔，便于复制到 Google Docs
    print("Task\tModel\tavg_steps_trimmed\textractive_match(%)")
    for task, mk, steps_v, acc_v in rows:
        print(f"{task}\t{mk}\t{steps_v:.3f}\t{acc_v:.2f}")


if __name__ == "__main__":
    main()


