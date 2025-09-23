# Copyright 2024
# Reward for OpenR1 stepwise enum data (intermediate steps and final steps)
#
# Design:
# - For all steps: use a lightweight hybrid reward combining numeric match and substring match (reuse gsm8k utils)
# - If ground_truth contains LaTeX math segments, add a LaTeX coverage term (matched segments / total)
# - Final-step special case is handled upstream by setting ground_truth to the final answer in dataset
#
# Output: float in [0, 1]

from __future__ import annotations

import re
from typing import List

from . import gsm8k


_LATEX_INLINE = [
    (r"\$(.+?)\$", re.DOTALL),           # $ ... $
    (r"\\\((.+?)\\\)", re.DOTALL),     # \( ... \)
]
_LATEX_BLOCK = [
    (r"\$\$(.+?)\$\$", re.DOTALL),       # $$ ... $$
    (r"\\\[(.+?)\\\]", re.DOTALL),       # \[ ... \]
]


def _extract_latex_segments(text: str) -> List[str]:
    if not text:
        return []
    segments: List[str] = []
    for pattern, flags in _LATEX_BLOCK + _LATEX_INLINE:
        for m in re.finditer(pattern, text, flags):
            seg = (m.group(1) or "").strip()
            if seg:
                segments.append(seg)
    return segments


def _normalize_inline(s: str) -> str:
    # minimal normalization: collapse spaces, lower-case certain macros, remove redundant braces pairs
    t = re.sub(r"\s+", " ", s).strip()
    t = t.replace("\\,", " ")
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("{ ", "{").replace(" }", "}")
    return t


def _latex_coverage(solution: str, ground_truth: str) -> float:
    gt_segments = _extract_latex_segments(ground_truth)
    if not gt_segments:
        return 0.0
    sol = solution or ""
    sol_norm = _normalize_inline(sol)
    hit = 0
    for seg in gt_segments:
        seg_norm = _normalize_inline(seg)
        if seg_norm and seg_norm in sol_norm:
            hit += 1
    return hit / max(1, len(gt_segments))


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Compute reward for OpenR1 stepwise data.

    - Always compute hybrid score (numbers + substring) using gsm8k utilities
    - If LaTeX exists in ground_truth, add latex coverage component
    - Final-step case is already reflected in ground_truth by upstream converter
    """
    if not ground_truth:
        return 0.0
    sol = solution_str or ""

    # Base hybrid (reuses existing tested logic)
    hybrid = gsm8k.compute_hybrid_score(sol, ground_truth, number_weight=0.7, substring_weight=0.3)

    # LaTeX coverage if present
    latex_cov = _latex_coverage(sol, ground_truth)

    # Dynamic weighting: if gt has latex, give it weight; else 0
    has_latex = bool(_extract_latex_segments(ground_truth))
    w_latex = 0.3 if has_latex else 0.0
    w_hybrid = 1.0 - w_latex

    score = w_hybrid * hybrid + w_latex * latex_cov

    # Clamp to [0,1]
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return float(score)


