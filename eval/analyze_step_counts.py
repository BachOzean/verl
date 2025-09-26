#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据 lighteval --save-details 生成的 details 文件，统计模型解题过程里“步骤数”的粗略估计。

支持输入：
- 根目录（默认：eval/results），脚本会递归查找 details/*.csv 与 *.json
- 也可直接传入一个目录或单个文件路径

输出：
- 终端打印总体统计（样本数、平均步骤数、分位数等）
- 可选导出 CSV：每条样本的 id（文件+行号）、估计步骤数、是否包含答案行、原文长度等

说明：
- lighteval 的 details CSV 通常包含列：predictions（或 prediction）、full_prompt、gold 等。
- 这里优先从 predictions 列取模型输出；若为空，会回退到 prediction(s)_logits 之外的可能字段。
- JSON 详情通常是一个由对象组成的数组，字段名类似 CSV。我们做容错解析。

启发式步骤计数：
1) 数字编号：匹配序号模式，如 "1.", "2)", "(3)", "Step 1", "步骤1" 等
2) 中文连接词："首先"、"接着"、"其次"、"然后"、"最后"、"综上"、"因此"
3) 英文连接词："First", "Next", "Then", "Finally", "In conclusion", "Therefore"
4) 连续行/段落前缀中的项目符号："- ", "• ", "* " 等

计数规则（保守）：
- 将上述触发点按出现顺序合并去重、避免同一位置重复计数；
- 若无任何触发点，但文本中包含明显的思维链（例如多段落），则按段落数的上限与最小阈值（2）取较小值；
- 最终至少为 1。

注意：该统计是启发式近似值，目的是比较不同 checkpoint 之间“平均步骤长度”的相对变化。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -------------------------
# 文本解析与启发式
# -------------------------

STEP_NUMBER_PATTERNS: List[re.Pattern] = [
    # 典型编号：1. 2. 3.
    re.compile(r"(?<!\d)(\d{1,2})\.(?!\d)"),
    # 1) 2)
    re.compile(r"(?<!\d)(\d{1,2})\)(?!\d)"),
    # (1) (2)
    re.compile(r"\((\d{1,2})\)"),
    # Step 1, Step 2
    re.compile(r"\bStep\s*(\d{1,2})\b", re.IGNORECASE),
    # 步骤1 / 第1步
    re.compile(r"(步骤|第)\s*(\d{1,2})\s*(步)?"),
]

CONNECTIVES_CN: List[str] = [
    "首先", "先", "第一个", "第一", "其次", "接着", "然后", "再", "最后", "综上", "因此", "所以",
]

CONNECTIVES_EN: List[str] = [
    "first", "firstly", "to begin", "begin with", "next", "then", "after that", "finally",
    "in conclusion", "therefore", "thus", "hence",
]

BULLET_PREFIXES: List[str] = ["- ", "* ", "• ", "– ", "— "]


def _find_indices(pattern: re.Pattern, text: str) -> List[int]:
    return [m.start() for m in pattern.finditer(text)]


def _find_connectives(text: str, exclude: Optional[List[str]] = None) -> List[int]:
    indices: List[int] = []
    lowered = text.lower()
    for token in CONNECTIVES_CN:
        if exclude and token in exclude:
            continue
        for m in re.finditer(re.escape(token), text):
            indices.append(m.start())
    for token in CONNECTIVES_EN:
        if exclude and token in exclude:
            continue
        for m in re.finditer(re.escape(token), lowered):
            indices.append(m.start())
    return indices


def _find_bullets(text: str) -> List[int]:
    indices: List[int] = []
    for line in text.splitlines():
        for prefix in BULLET_PREFIXES:
            if line.strip().startswith(prefix):
                # 使用文本中该行的起始位置作为索引
                idx = text.find(line)
                if idx >= 0:
                    indices.append(idx)
                break
    return indices


def estimate_step_count(text: str, mode: str = "connectives-only", exclude_connectives: Optional[List[str]] = None) -> int:
    """根据启发式估算一步数。

    mode 取值：
    - connectives-only：仅统计显式连接词（首先/其次/然后/最后/Therefore/Then/...）。更保守。
    - number-only：仅统计数字编号（1.、(2)、Step 3 等）。
    - hybrid：连接词 + 数字编号 + 项目符号，并在缺失时使用段落/句子兜底（更敏感）。
    """
    if not text:
        return 0

    hit_positions: List[int] = []

    mode = (mode or "connectives-only").strip().lower()

    if mode == "connectives-only":
        # 仅连接词
        hit_positions = _find_connectives(text, exclude=exclude_connectives)
        if hit_positions:
            return max(1, len(set(hit_positions)))
        return 1

    if mode == "number-only":
        # 仅数字编号
        for pat in STEP_NUMBER_PATTERNS:
            hit_positions.extend(_find_indices(pat, text))
        if hit_positions:
            return max(1, len(set(hit_positions)))
        return 1

    # hybrid：连接词 + 数字编号 + 项目符号，并兜底
    for pat in STEP_NUMBER_PATTERNS:
        hit_positions.extend(_find_indices(pat, text))
    hit_positions.extend(_find_connectives(text, exclude=exclude_connectives))
    hit_positions.extend(_find_bullets(text))

    if hit_positions:
        hit_positions = sorted(set(hit_positions))
        steps = max(1, len(hit_positions))
        return steps

    paragraphs = [p for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(paragraphs) >= 2:
        return min(len(paragraphs), 6)

    sentences = re.split(r"[。！？!.?]\s+", text)
    if len([s for s in sentences if s.strip()]) >= 2:
        return min(len(sentences), 6)

    return 1


# -------------------------
# details 文件读取
# -------------------------

def _read_details_csv(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """逐行读取 CSV，返回 (row_id, row_dict)。row_id 用于导出唯一标识。"""
    # 放大单字段最大字节限制，避免长思维链导致的异常
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        try:
            csv.field_size_limit(2**31 - 1)
        except Exception:
            pass

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):  # 从第2行是首数据行
            rid = f"{path.name}#L{i}"
            yield rid, row


def _read_details_json(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for i, obj in enumerate(data, start=1):
            rid = f"{path.name}#[{i}]"
            if isinstance(obj, dict):
                yield rid, obj
    elif isinstance(data, dict):
        # 也支持单对象或包装结构
        rid = f"{path.name}#[]"
        yield rid, data


def extract_prediction_text(row: Dict[str, Any]) -> str:
    """尽可能从一行 details 中抽取模型输出文本。"""
    candidates = [
        "predictions",  # CSV 中常见：字符串或 JSON 字符串
        "prediction",   # 变体
        "output", "outputs", "response", "responses", "generation", "generated_text", "text", "completion",
    ]

    # 从 dict-like 中提取文本
    def _from_dict_like(d: Any) -> Optional[str]:
        if isinstance(d, dict):
            for k in ("text", "output", "response", "generated_text"):
                v = d.get(k)
                if isinstance(v, str) and v.strip():
                    return v
        return None

    # 从 array-like（list/tuple/numpy.ndarray）中提取合适的字符串
    def _from_array_like(v: Any) -> Optional[str]:
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore

        if isinstance(v, (list, tuple)):
            seq = v
        elif np is not None and isinstance(v, np.ndarray):  # type: ignore
            try:
                seq = v.tolist()
            except Exception:
                try:
                    seq = list(v)
                except Exception:
                    return None
        else:
            return None

        if not seq:
            return None

        first = seq[0]
        if isinstance(first, (bytes, bytearray)):
            try:
                first = first.decode("utf-8", errors="ignore")
            except Exception:
                first = str(first)
        if isinstance(first, str) and first.strip():
            return first
        if isinstance(first, dict):
            got = _from_dict_like(first)
            if got:
                return got

        # 遍历寻找最长可用字符串
        longest = ""
        for item in seq:
            if isinstance(item, (bytes, bytearray)):
                try:
                    item = item.decode("utf-8", errors="ignore")
                except Exception:
                    item = str(item)
            if isinstance(item, str) and len(item) > len(longest):
                longest = item
            if isinstance(item, dict):
                tmp = _from_dict_like(item)
                if isinstance(tmp, str) and len(tmp) > len(longest):
                    longest = tmp
        return longest or None

    for key in candidates:
        if key in row and row[key]:
            value = row[key]
            # 可能是 JSON 字符串，如 "["...text..."]"
            if isinstance(value, str) and value.strip().startswith("["):
                try:
                    arr = json.loads(value)
                    if isinstance(arr, list) and arr:
                        # 有时是 list[str] 或 list[object]
                        first = arr[0]
                        if isinstance(first, str):
                            return first
                        if isinstance(first, dict):
                            for k in ("text", "output", "response"):
                                if k in first and isinstance(first[k], str):
                                    return first[k]
                except Exception:
                    pass
            # bytes -> str
            if isinstance(value, (bytes, bytearray)):
                try:
                    value = value.decode("utf-8", errors="ignore")
                except Exception:
                    value = str(value)
            # 直接字符串
            if isinstance(value, str):
                return value
            # list/tuple/numpy.ndarray
            from_array = _from_array_like(value)
            if isinstance(from_array, str) and from_array.strip():
                return from_array
            if isinstance(value, dict):
                for k in ("text", "output", "response", "generated_text"):
                    if k in value and isinstance(value[k], str):
                        return value[k]

    # 特例：CSV 中 predictions 字段名存在但为空时，可能另有 specifics/prediction_logits 存放 token
    # 我们尽量避免使用 token 序列，返回空让上层跳过。
    # 兜底：尝试在名字包含 pred/output/response/generation/text 的字段中找最长字符串
    try:
        keys_like = [
            k for k in row.keys()
            if any(sub in str(k).lower() for sub in ["pred", "output", "response", "generation", "text"]) and "prompt" not in str(k).lower()
        ]
        best_text = ""
        for k in keys_like:
            v = row.get(k)
            if isinstance(v, (bytes, bytearray)):
                try:
                    v = v.decode("utf-8", errors="ignore")
                except Exception:
                    v = str(v)
            # 统一处理 array-like
            from_array = _from_array_like(v)
            if isinstance(from_array, str) and from_array.strip():
                v = from_array
            if isinstance(v, dict):
                for kk in ("text", "output", "response", "generated_text"):
                    if kk in v and isinstance(v[kk], str):
                        v = v[kk]
                        break
            if isinstance(v, str) and len(v) > len(best_text):
                best_text = v
        if best_text:
            return best_text
    except Exception:
        pass

    return ""


@dataclass
class SampleStat:
    file_id: str
    steps: int
    text_len: int
    has_final_line: bool


FINAL_LINE_PATTERNS: List[re.Pattern] = [
    re.compile(r"Therefore,\s*the\s*final\s*answer\s*is:\s*\\boxed\{", re.IGNORECASE),
    re.compile(r"因此[，,，\s]*最终答案[是为]\s*\\boxed\{"),
]


def contains_final_line(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    for p in FINAL_LINE_PATTERNS:
        if p.search(t):
            return True
    # 兜底：包含 boxed 或标准答案语式
    if "\\boxed{" in t or "boxed{" in t:
        return True
    return False


def iter_detail_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    # 递归匹配 details 下的 csv/json
    for ext in ("*.csv", "*.json", "*.parquet"):
        for p in root.rglob(ext):
            if "details" in p.as_posix().split("/"):
                yield p


def _find_anchor_dir_for_file(file_path: Path) -> Optional[Path]:
    """给定 details 下的文件，返回其 anchor 目录（details/后的一级目录）。"""
    parts = file_path.parts
    try:
        idx = parts.index("details")
    except ValueError:
        return None
    if idx + 1 < len(parts):
        anchor = Path(*parts[: idx + 2])
        return anchor
    return None


def _infer_model_name_from_anchor(anchor_dir: Path) -> str:
    """从 details/<anchor>/ 推断模型名。

    规则：
    - 若目录名包含 "models_"，取其后缀。
    - 若包含 "checkpoints_"，取其后缀（常见为 global_step_* 等）。
    - 否则返回目录名本身。
    """
    name = anchor_dir.name
    if "models_" in name:
        return name.split("models_", 1)[1]
    if "checkpoints_" in name:
        return name.split("checkpoints_", 1)[1]
    return name


def group_files_by_model(files: List[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for f in files:
        anchor_dir = _find_anchor_dir_for_file(f)
        if anchor_dir is None:
            key = "unknown"
        else:
            key = _infer_model_name_from_anchor(anchor_dir)
        groups.setdefault(key, []).append(f)
    return groups


def analyze_paths(paths: List[Path], mode: str = "connectives-only", exclude_connectives: Optional[List[str]] = None) -> List[SampleStat]:
    stats: List[SampleStat] = []
    for p in paths:
        try:
            if p.suffix.lower() == ".csv":
                rows = _read_details_csv(p)
            elif p.suffix.lower() == ".json":
                rows = _read_details_json(p)
            elif p.suffix.lower() == ".parquet":
                rows = _read_details_parquet(p)
            else:
                continue

            for rid, row in rows:
                text = extract_prediction_text(row)
                if not text:
                    continue
                steps = estimate_step_count(text, mode=mode, exclude_connectives=exclude_connectives)
                has_final = contains_final_line(text)
                stats.append(
                    SampleStat(
                        file_id=f"{p.parent.name}/{rid}",
                        steps=steps,
                        text_len=len(text),
                        has_final_line=has_final,
                    )
                )
        except Exception as e:
            # 解析失败不中断，继续
            print(f"[WARN] Failed to parse {p}: {e}")
            continue
    return stats


def _read_details_parquet(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """读取 Parquet 详情。

    优先使用 pandas.read_parquet。若环境缺失依赖，请安装 pyarrow 或 fastparquet。
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pandas 未安装，无法读取 Parquet。请安装 pandas 及 pyarrow 或 fastparquet。"
        ) from e

    df = pd.read_parquet(path)
    # iterrows 对大型数据不是最优，但 details 文件行数通常较少，可接受
    for idx, row in df.iterrows():
        rid = f"{path.name}#R{idx}"
        # 将可能的 pandas NA 转为 None
        row_dict: Dict[str, Any] = {
            k: (None if (isinstance(v, float) and math.isnan(v)) else v)
            for k, v in row.to_dict().items()
        }
        yield rid, row_dict


def _to_list_str(x: Any) -> List[str]:
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        return [str(i) for i in x]
    if np is not None and isinstance(x, np.ndarray):  # type: ignore
        try:
            return [str(i) for i in x.tolist()]
        except Exception:
            try:
                return [str(i) for i in list(x)]
            except Exception:
                return []
    return []


def _normalize_ans(s: str) -> str:
    s2 = s.strip().lower()
    # 简单清洗 latex 与空白
    s2 = s2.replace("\\boxed{", "").replace("}", "")
    s2 = s2.replace("$", "")
    s2 = re.sub(r"\s+", "", s2)
    return s2


def row_is_correct(row: Dict[str, Any]) -> bool:
    # 1) metrics.extractive_match
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        em = metrics.get("extractive_match")
        if isinstance(em, (int, float)) and em >= 0.5:
            return True

    # 2) specifics.extracted_{golds,predictions}
    specifics = row.get("specifics")
    if isinstance(specifics, dict):
        golds = _to_list_str(specifics.get("extracted_golds"))
        preds = _to_list_str(specifics.get("extracted_predictions"))
        if golds and preds:
            gold_set = { _normalize_ans(g) for g in golds }
            pred_set = { _normalize_ans(p) for p in preds }
            if gold_set & pred_set:
                return True

    return False


def split_steps_by_connectives(text: str) -> List[str]:
    if not text:
        return []
    # 使用已定义的连接词位置划分
    idxs = sorted(set(_find_connectives(text)))
    if not idxs:
        return [text]
    segments: List[str] = []
    last = 0
    for i in idxs:
        if i > last:
            segments.append(text[last:i].strip())
        last = i
    segments.append(text[last:].strip())
    return [seg for seg in segments if seg]


def _is_base_group(files: List[Path]) -> bool:
    if not files:
        return False
    anchor = _find_anchor_dir_for_file(files[0])
    if not anchor:
        return False
    # 约定：基础模型目录名包含 "models_"
    return "models_" in anchor.name


def _build_row_key(row: Dict[str, Any]) -> Optional[str]:
    import hashlib
    for k in ("example", "full_prompt", "instruction"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            base = v.strip()
            return hashlib.md5(base.encode("utf-8", errors="ignore")).hexdigest()
    # 回退到 gold 作为 key（不完美，但可用于部分数据集）
    v = row.get("gold")
    if isinstance(v, str) and v.strip():
        import hashlib
        return hashlib.md5(v.strip().encode("utf-8", errors="ignore")).hexdigest()
    return None


def _load_rows_info(
    files: List[Path],
    mode: str,
    exclude_connectives: Optional[List[str]],
) -> Dict[str, Dict[str, Any]]:
    info: Dict[str, Dict[str, Any]] = {}
    for f in files:
        if f.suffix.lower() == ".csv":
            rows_iter = _read_details_csv(f)
        elif f.suffix.lower() == ".json":
            rows_iter = _read_details_json(f)
        else:
            rows_iter = _read_details_parquet(f)
        for rid, row in rows_iter:
            key = _build_row_key(row)
            if not key:
                continue
            text = extract_prediction_text(row)
            if not text:
                continue
            steps = estimate_step_count(text, mode=mode, exclude_connectives=exclude_connectives)
            info[key] = {
                "rid": rid,
                "row": row,
                "text": text,
                "steps": steps,
                "correct": row_is_correct(row),
            }
    return info


def _extract_question(row: Dict[str, Any]) -> str:
    """从行中提取题目文本，优先 example，其次 full_prompt，再次 instruction。"""
    keys = ["example", "full_prompt", "instruction"]
    for k in keys:
        if k in row and row[k]:
            v = row[k]
            # 允许 list/ndarray
            arr = _to_list_str(v)
            if arr:
                return arr[0]
            if isinstance(v, (bytes, bytearray)):
                try:
                    return v.decode("utf-8", errors="ignore")
                except Exception:
                    return str(v)
            if isinstance(v, str):
                return v
    return ""


def summarize(stats: List[SampleStat], trim_frac: float = 0.0) -> Dict[str, Any]:
    if not stats:
        return {
            "num_samples": 0,
        }
    values = [s.steps for s in stats]
    values_sorted = sorted(values)
    n = len(values_sorted)

    def pct(p: float) -> float:
        if n == 0:
            return math.nan
        k = max(0, min(n - 1, int(round(p * (n - 1)))))
        return float(values_sorted[k])

    # 截尾平均（双侧），剔除极端值
    if 0.0 < trim_frac < 0.5 and n > 0:
        k = int(n * trim_frac)
        trimmed = values_sorted[k: n - k] if n - 2 * k > 0 else values_sorted
        avg_trimmed = float(sum(trimmed) / len(trimmed)) if trimmed else float(sum(values_sorted) / n)
    else:
        avg_trimmed = float(sum(values_sorted) / n) if n > 0 else float('nan')

    summary = {
        "num_samples": n,
        "avg_steps": float(sum(values) / n),
        "avg_steps_trimmed": avg_trimmed,
        "min_steps": int(values_sorted[0]),
        "p25_steps": pct(0.25),
        "p50_steps": pct(0.50),
        "p75_steps": pct(0.75),
        "max_steps": int(values_sorted[-1]),
        "ratio_has_final_line": float(sum(1 for s in stats if s.has_final_line) / n),
    }
    return summary


def export_csv(stats: List[SampleStat], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_id", "steps", "text_len", "has_final_line"],
        )
        writer.writeheader()
        for s in stats:
            writer.writerow({
                "file_id": s.file_id,
                "steps": s.steps,
                "text_len": s.text_len,
                "has_final_line": int(s.has_final_line),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 lighteval details 的步骤数启发式")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["/data/home/scyb494/verl/eval/results"],
        help="要分析的根目录/文件（可多个）。默认 eval/results",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default="",
        help="可选：导出每条样本的统计到 CSV 路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="connectives-only",
        choices=["connectives-only", "number-only", "hybrid"],
        help="步骤统计模式：connectives-only（默认，保守）、number-only、hybrid（更敏感）",
    )
    parser.add_argument(
        "--by-model",
        action="store_true",
        help="按模型名（从 details 路径推断）分组统计并分别打印",
    )
    parser.add_argument(
        "--show-example",
        action="store_true",
        help="从每个模型里挑 1 条“正确且多步”的样本，打印 prediction，并按连接词分段预览",
    )
    parser.add_argument(
        "--exclude-connectives",
        type=str,
        default="so",
        help="逗号分隔的需排除的连接词（大小写不敏感，默认: so）",
    )
    parser.add_argument(
        "--trim-frac",
        type=float,
        default=0.05,
        help="计算截尾平均时的单侧截断比例（如 0.05 表示两侧各去 5%）",
    )
    parser.add_argument(
        "--max-preview-steps",
        type=int,
        default=10,
        help="样例展示的最大步骤数（超过将截断，仅显示前 N 步）",
    )
    parser.add_argument(
        "--example-min-steps",
        type=int,
        default=3,
        help="示例筛选的最小步骤数（用于选择样例，而非显示截断）",
    )
    parser.add_argument(
        "--example-max-steps",
        type=int,
        default=10,
        help="示例筛选的最大步骤数（用于选择样例，而非显示截断）",
    )

    args = parser.parse_args()
    input_paths = [Path(p).resolve() for p in args.inputs]
    files: List[Path] = []
    for ip in input_paths:
        files.extend(list(iter_detail_files(ip)))

    # 去重
    files = sorted(set(files))
    print(f"[INFO] Found {len(files)} detail files")

    exclude_connectives = [t.strip().lower() for t in args.exclude_connectives.split(",") if t.strip()]

    if args.by_model:
        groups = group_files_by_model(files)
        # 预先装载 base 与 step 组的明细，便于交叉挑选样例
        base_groups: Dict[str, Dict[str, Any]] = {}
        step_groups: Dict[str, Dict[str, Any]] = {}

        for model_name, group_files in sorted(groups.items()):
            stats = analyze_paths(group_files, mode=args.mode, exclude_connectives=exclude_connectives)
            summary = summarize(stats, trim_frac=args.trim_frac)
            print(f"[MODEL] {model_name}")
            for k, v in summary.items():
                print(f"- {k}: {v}")

            if args.show_example and stats:
                # 先缓存明细
                details = _load_rows_info(group_files, args.mode, exclude_connectives)
                if _is_base_group(group_files):
                    base_groups[model_name] = details
                else:
                    step_groups[model_name] = details

            if args.export_csv and stats:
                out_path = Path(args.export_csv).with_name(
                    Path(args.export_csv).stem + f"_{model_name}.csv"
                ).with_suffix(".csv").resolve()
                export_csv(stats, out_path)
                print(f"[INFO] Exported per-sample stats to: {out_path}")

        # 展示“base 错误但 step 正确且多步”的样例
        if args.show_example and base_groups and step_groups:
            print("[CROSS-EXAMPLE] base 错误、step 正确（多步）的样例：")
            shown = 0
            # 以 base 组为基准，寻找同 key 在任一步组正确的样本
            for base_name, base_map in base_groups.items():
                for key, binfo in base_map.items():
                    if binfo.get("correct"):
                        continue
                    # 在任一步组找正确且多步
                    for step_name, step_map in step_groups.items():
                        sinfo = step_map.get(key)
                        if not sinfo or not sinfo.get("correct"):
                            continue
                        steps_val = int(sinfo.get("steps", 0))
                        if steps_val < int(args.example_min_steps) or steps_val > int(args.example_max_steps):
                            continue
                        text = sinfo.get("text", "")
                        base_text = binfo.get("text", "")
                        # 问题文本
                        question = _extract_question(sinfo.get("row", {})) or _extract_question(binfo.get("row", {}))
                        if question:
                            print("[QUESTION]")
                            print(question[:1200])
                        print(f"[BASE] {base_name} -> wrong")
                        if base_text:
                            print(f"[BASE PRED] {base_text[:800]}")
                        print(f"[STEP] {step_name} -> correct, steps={sinfo['steps']}")
                        segments = split_steps_by_connectives(text)
                        max_n = max(1, int(args.max_preview_steps))
                        total = len(segments)
                        for i, seg in enumerate(segments[:max_n], start=1):
                            print(f"[Step {i}] {seg[:800]}")
                        if total > max_n:
                            print(f"[TRUNCATED] 仅展示前 {max_n} 步，共 {total} 步")
                        shown += 1
                        break
                    if shown:
                        break
                if shown:
                    break
    else:
        stats = analyze_paths(files, mode=args.mode, exclude_connectives=exclude_connectives)
        summary = summarize(stats, trim_frac=args.trim_frac)
        print("[SUMMARY]")
        for k, v in summary.items():
            print(f"- {k}: {v}")

        if args.export_csv:
            out_path = Path(args.export_csv).resolve()
            export_csv(stats, out_path)
            print(f"[INFO] Exported per-sample stats to: {out_path}")


if __name__ == "__main__":
    main()


