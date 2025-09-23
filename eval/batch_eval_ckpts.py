"""
批量评估脚本：

功能概述：
- 遍历指定 checkpoints 根目录下的每个 `global_step_*` 目录
- 若未存在 HuggingFace safetensors 权重，则调用 `scripts/legacy_model_merger.py` 进行转换（FSDP -> HF）
- 为每个 step 生成独立的 vLLM 配置 YAML，调用 lighteval 分别评估以下任务：
  - lighteval|gsm8k|3|1 （内置任务）
  - custom|math_500|0|0 （自定义任务：eval/custom_tasks/eval_math500.py）
  - custom|aime24|0|0 （自定义任务：eval/custom_tasks/eval_aime.py）
  - custom|gpqa:diamond|0|0 （自定义任务：eval/custom_tasks/eval_gpqa.py）

说明：
- 用户输入中写到的 "gpqa|0|0" 这里采用 "gpqa:diamond|0|0"（自定义 GPQA 评测）作为等价替代，结果目录与图例使用 gpqa 命名。
- 评测结果自动解析 `results/*.json` 的 `results["all"]` 字段，按 step 聚合并绘图。

使用方式：
python -u /data/home/scyb494/verl/eval/batch_eval_ckpts.py \
  --ckpt-root "/data/home/scyb494/verl/checkpoints/grpo_steps_enum_0923_02:35" \
  --output-root "/data/home/scyb494/verl/eval" \
  --skip-existing

依赖：
- lighteval CLI 可用
- matplotlib、pandas 可用（用于绘图与导出）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ------------------------------- 日志配置 ----------------------------------
def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ------------------------------- 常量与路径 --------------------------------
EVAL_DIR = Path("/data/home/scyb494/verl/eval").resolve()
PROJECT_ROOT = Path("/data/home/scyb494/verl").resolve()
DEFAULT_CKPT_ROOT = PROJECT_ROOT / "checkpoints/grpo_steps_enum_0923_02:35"
CUSTOM_TASKS_DIR = EVAL_DIR / "custom_tasks"
MODEL_CONFIG_DIR = EVAL_DIR / "model_config"
TMP_MODEL_CONFIG_DIR = EVAL_DIR / "tmp_model_configs"
PLOTS_DIR = EVAL_DIR / "plots"
AGG_DIR = EVAL_DIR / "aggregate"
SBATCH_DIR = EVAL_DIR / "sbatch"
BASELINE_NAME = "DeepSeek-R1-Distill-Qwen-1.5B"



@dataclass(frozen=True)
class TaskSpec:
    name: str  # 任务内部名称（用于标识与绘图）
    spec: str  # lighteval 的任务规范字符串
    custom_module_rel: Optional[str]  # 若为自定义任务，给出相对 eval/ 的 py 路径
    output_subdir: str  # 在 results 下的子目录名称


TASK_SPECS: List[TaskSpec] = [
    TaskSpec(name="gsm8k", spec="lighteval|gsm8k|3|1", custom_module_rel=None, output_subdir="gsm8k"),
    TaskSpec(name="math_500", spec="custom|math_500|0|0", custom_module_rel="custom_tasks/eval_math500.py", output_subdir="math500"),
    TaskSpec(name="aime24", spec="custom|aime24|0|0", custom_module_rel="custom_tasks/eval_aime.py", output_subdir="aime24"),
    TaskSpec(name="gpqa:diamond", spec="custom|gpqa:diamond|0|0", custom_module_rel="custom_tasks/eval_gpqa.py", output_subdir="gpqa"),
]


# ------------------------------- 工具函数 -----------------------------------
def run_shell(cmd: str, cwd: Optional[Path] = None, extra_env: Optional[Dict[str, str]] = None) -> int:
    """在 bash -lc 中执行命令，便于使用 module 与环境变量。

    返回值为进程退出码。
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    shell_cmd = f"bash -lc '{cmd}'"
    logging.debug("Running shell: %s (cwd=%s)", shell_cmd, str(cwd) if cwd else None)
    proc = subprocess.Popen(shell_cmd, shell=True, cwd=str(cwd) if cwd else None, env=env)
    proc.communicate()
    return int(proc.returncode or 0)


def is_hf_safetensors_ready(hf_dir: Path) -> bool:
    """判断 HuggingFace 目录下是否存在 safetensors（或权重文件）。"""
    if not hf_dir.exists():
        return False
    # 常见命名：model.safetensors 或 shard：model-00001-of-000xx.safetensors；亦或回退到 bin
    safes = list(hf_dir.glob("*.safetensors"))
    bins = list(hf_dir.glob("pytorch_model*.bin"))
    config_ok = (hf_dir / "config.json").exists()
    tok_ok = (hf_dir / "tokenizer.json").exists() or (hf_dir / "tokenizer.model").exists()
    return config_ok and tok_ok and (len(safes) > 0 or len(bins) > 0)


def extract_step_number(step_dir: Path) -> Optional[int]:
    """从目录名中提取 step 数字，例如 global_step_300 -> 300。"""
    m = re.search(r"global_step_(\d+)", step_dir.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def generate_vllm_yaml_for_step(model_dir: Path, target_yaml: Path) -> None:
    """为指定 HF 模型目录生成 vLLM 配置 YAML 文件。"""
    yaml_text = f"""
model_parameters:
    model_name: "{str(model_dir)}"
    dtype: "bfloat16"
    tensor_parallel_size: 1
    data_parallel_size: 1
    pipeline_parallel_size: 1
    gpu_memory_utilization: 0.9
    max_model_length: 32768
    swap_space: 4
    seed: 1
    trust_remote_code: True
    use_chat_template: False
    max_num_batched_tokens: 32768
    generation_parameters:
      presence_penalty: 0.0
      repetition_penalty: 1.0
      frequency_penalty: 0.0
      temperature: 0.6
      top_k: 10
      min_p: 0.0
      top_p: 0.95
      seed: 42
      stop_tokens: null
      max_new_tokens: 32768
      min_new_tokens: 0
""".strip()
    target_yaml.parent.mkdir(parents=True, exist_ok=True)
    target_yaml.write_text(yaml_text, encoding="utf-8")


def convert_fsdp_to_hf_if_needed(actor_dir: Path, hf_dir: Path, verbose: bool) -> None:
    """如果 HF safetensors 不存在，则调用 legacy_model_merger.py 进行合并导出。"""
    if is_hf_safetensors_ready(hf_dir):
        logging.info("[convert] 已存在 HF 模型：%s", hf_dir)
        return

    logging.info("[convert] 开始转换 FSDP -> HF: %s -> %s", actor_dir, hf_dir)
    hf_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"module load cuda/12.4; "
        f"python -u {PROJECT_ROOT / 'scripts/legacy_model_merger.py'} merge "
        f" --backend fsdp --local_dir {actor_dir} --target_dir {hf_dir}"
    )
    code = run_shell(cmd, cwd=PROJECT_ROOT)
    if code != 0:
        raise RuntimeError(f"转换失败（退出码 {code}）：{actor_dir}")
    if not is_hf_safetensors_ready(hf_dir):
        raise RuntimeError(f"转换完成但未找到 safetensors/bin：{hf_dir}")
    logging.info("[convert] 转换完成：%s", hf_dir)


def run_lighteval_once(
    yaml_path: Path,
    task: TaskSpec,
    output_root: Path,
    step_num: int,
    verbose: bool,
) -> Optional[Path]:
    """调用 lighteval 评测一个任务，返回最新结果 JSON 路径。

    输出目录按 task/step 归档，避免不同 step 结果相互覆盖：
      {output_root}/results/{task.output_subdir}/global_step_{step}/
    """
    output_dir = output_root / "results" / task.output_subdir / f"global_step_{step_num}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_env = {
        "HYDRA_FULL_ERROR": "1",
        "HF_ENDPOINT": "https://hf-mirror.com",
    }

    # lighteval CLI 调用
    parts: List[str] = [
        "module load cuda/12.4 && lighteval vllm",
        f'"{str(yaml_path)}"',
    ]
    if task.custom_module_rel:
        parts.append(f"--custom-tasks {task.custom_module_rel}")
    parts.append(f'"{task.spec}"')
    parts.append(f"--output-dir={output_dir}")
    parts.append("--save-details")

    cmd = " ".join(parts)
    code = run_shell(cmd, cwd=EVAL_DIR, extra_env=base_env)
    if code != 0:
        logging.error("[eval] 任务 %s 评测失败，退出码=%s", task.name, code)
        return None

    # 在输出目录下递归查找最新的 results_*.json
    json_paths = sorted(output_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_paths:
        logging.error("[eval] 未找到结果 JSON：%s", output_dir)
        return None
    return json_paths[0]


def find_existing_result_json_for_step(
    output_root: Path,
    task: TaskSpec,
    step_num: int,
    hf_dir: Path,
) -> Optional[Path]:
    """查找指定 task 与 step 的已存在结果 JSON。

    优先使用新的按 step 归档的目录；若不存在则回退到旧结构，在文件内容中搜索
    'global_step_{step}' 或模型目录路径片段以匹配该 step 的结果。
    """
    # 新结构：task/global_step_{step}/
    step_dir = output_root / "results" / task.output_subdir / f"global_step_{step_num}"
    if step_dir.exists():
        json_paths = sorted(step_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if json_paths:
            return json_paths[0]

    # 旧结构：task/ 下直接堆叠的若干 runs
    legacy_dir = output_root / "results" / task.output_subdir
    if not legacy_dir.exists():
        return None

    pattern = f"global_step_{step_num}"
    hf_dir_str = str(hf_dir)
    hf_dir_norm = hf_dir_str.replace(os.sep, "/")

    candidates = []
    for p in sorted(legacy_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        if (pattern in text) or (hf_dir_str in text) or (hf_dir_norm in text):
            candidates.append(p)
    return candidates[0] if candidates else None


def parse_results_json(json_path: Path) -> Dict[str, Tuple[float, Optional[float]]]:
    """解析 results json 中 `results["all"]` 的指标。

    返回：{ metric_name: (value, stderr or None) }
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results_all = data.get("results", {}).get("all", {})

    metrics: Dict[str, Tuple[float, Optional[float]]] = {}
    for key, value in results_all.items():
        if key.endswith("_stderr"):
            continue
        stderr = results_all.get(f"{key}_stderr")
        try:
            v = float(value)
            s = float(stderr) if stderr is not None else None
        except Exception:
            continue
        metrics[key] = (v, s)
    return metrics


def collect_global_steps(ckpt_root: Path) -> List[Path]:
    """枚举 ckpt_root 下的所有 global_step_* 目录，按数字升序排序。"""
    step_dirs = [p for p in ckpt_root.glob("global_step_*/actor") if p.is_dir()]
    # actor 目录排序依据其上级的 step 数字
    step_dirs.sort(key=lambda p: extract_step_number(p.parent) or -1)
    return step_dirs


def ensure_dirs() -> None:
    TMP_MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    SBATCH_DIR.mkdir(parents=True, exist_ok=True)


def build_sbatch_script(
    step_num: int,
    actor_dir: Path,
    hf_dir: Path,
    yaml_path: Path,
    output_root: Path,
    partition: Optional[str],
    time_limit: Optional[str],
    gpus: int,
    cpus_per_task: int,
) -> str:
    """构建单个 step 的 sbatch 脚本文本。包含：转换 + 四个任务评测。"""
    lines: List[str] = []
    lines.append("#!/bin/bash")
    lines.append("#sbatch -J eval_step_{}".format(step_num))
    lines.append("#sbatch -o {}/slurm-step_{}-%j.out".format(str(EVAL_DIR), step_num))
    if partition:
        lines.append(f"#sbatch -p {partition}")
    if time_limit:
        lines.append(f"#sbatch -t {time_limit}")
    lines.append(f"#sbatch --gpus={gpus}")
    lines.append("")
    lines.append("set -euo pipefail")
    lines.append("cd {}".format(str(EVAL_DIR)))
    lines.append("export HYDRA_FULL_ERROR=1")
    lines.append("export HF_ENDPOINT=https://hf-mirror.com")
    lines.append("module load cuda/12.4")
    lines.append("")
    # 转换（若无）
    lines.append(f"if [ ! -f '{hf_dir}/config.json' ]; then")
    lines.append(
        f"  python -u {PROJECT_ROOT / 'scripts/legacy_model_merger.py'} merge --backend fsdp --local_dir {actor_dir} --target_dir {hf_dir}"
    )
    lines.append("fi")
    lines.append("")
    # 逐任务评测（按 step 归档输出）
    for task in TASK_SPECS:
        out_dir = output_root / "results" / task.output_subdir / f"global_step_{step_num}"
        if task.custom_module_rel:
            custom_flag = f"--custom-tasks {task.custom_module_rel}"
        else:
            custom_flag = ""
        lines.append(
            "lighteval vllm \"{}\" {} \"{}\" --output-dir={} --save-details".format(
                str(yaml_path), custom_flag, task.spec, str(out_dir)
            ).replace("  ", " ").strip()
        )
    lines.append("")
    return "\n".join(lines)


def write_submit_all_script(script_paths: List[Path], partition: Optional[str], time_limit: Optional[str], gpus: int) -> Path:
    """生成一键提交脚本 submit_all.sh"""
    submit_path = SBATCH_DIR / "submit_all.sh"
    lines: List[str] = []
    lines.append("#!/bin/bash")
    lines.append("set -euo pipefail")
    for sp in script_paths:
        args = []
        if partition:
            args.append(f"-p {partition}")
        if time_limit:
            args.append(f"-t {time_limit}")
        args.append(f"--gpus={gpus}")
        lines.append(f"sbatch {' '.join(args)} {str(sp)}")
    submit_path.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(submit_path, 0o755)
    return submit_path


def aggregate_and_save(agg_rows: List[Dict[str, object]]) -> Path:
    """将聚合结果保存为 CSV 与 JSON。返回 CSV 路径。"""
    import pandas as pd

    df = pd.DataFrame(agg_rows)
    csv_path = AGG_DIR / "metrics.csv"
    json_path = AGG_DIR / "metrics.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(agg_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("[aggregate] 已保存：%s, %s", csv_path, json_path)
    return csv_path


def find_baseline_result_json(output_root: Path, task: TaskSpec) -> Optional[Path]:
    """查找 baseline（非训练 step）的最近一次结果 JSON。

    约定：baseline 结果位于
      {output_root}/results/{task.output_subdir}/results/<model_dir_sanitized>/results_*.json
    若存在多个 runs，取最近修改时间的一个。
    """
    # 典型目录：.../results/<task_subdir>/results/_data_home_..._models_.../
    base_parent = output_root / "results" / task.output_subdir / "results"
    if not base_parent.exists():
        return None
    # 寻找有 BASELINE_NAME 的目录，逐个聚合其下的 results_*.json
    baseline_dirs = list(base_parent.glob(f"*{BASELINE_NAME}*"))
    json_paths: List[Path] = []
    for d in baseline_dirs:
        if d.is_dir():
            json_paths.extend(d.rglob("results_*.json"))
    json_paths = sorted(json_paths, key=lambda p: p.stat().st_mtime, reverse=True)

    return json_paths[0] if json_paths else None


def append_baseline_to_agg(agg_rows: List[Dict[str, object]], output_root: Path) -> int:
    """将 baseline 结果以 step=0 追加到 agg_rows。

    返回新增的行数；若某 task 已存在 step=0 则跳过，避免重复。
    """
    before = len(agg_rows)
    for task in TASK_SPECS:
        # gpqa:diamond 对外显示为 gpqa
        public_task_name = task.name if task.name != "gpqa:diamond" else "gpqa"
        # 若已存在该任务的 step=0，跳过
        if any((row.get("step") == 0 and row.get("task") == public_task_name) for row in agg_rows):
            continue

        json_path = find_baseline_result_json(output_root=output_root, task=task)
        if not json_path:
            continue

        try:
            metrics = parse_results_json(json_path)
        except Exception:
            continue

        for metric, (val, stderr) in metrics.items():
            agg_rows.append(
                {
                    "step": 0,
                    "task": public_task_name,
                    "metric": metric,
                    "value": val,
                    "stderr": stderr,
                    "json_path": str(json_path),
                }
            )
    return len(agg_rows) - before


def plot_metrics(agg_rows: List[Dict[str, object]]) -> None:
    """每个 task 每个 metric 一张图：避免将 score 与 mean_length 画在同一张图里。"""
    import pandas as pd
    from matplotlib import pyplot as plt

    if not agg_rows:
        logging.warning("[plot] 无聚合数据，跳过绘图")
        return

    df = pd.DataFrame(agg_rows)
    # 确保类型正确
    df["step"] = df["step"].astype(int)
    df = df.sort_values(["task", "metric", "step"])  # 保证绘制顺序

    tasks = sorted(df["task"].unique())
    for task_name in tasks:
        sub = df[df["task"] == task_name]
        for metric_name, g in sub.groupby("metric"):
            plt.figure(figsize=(8, 5))
            plt.plot(g["step"], g["value"], marker="o")
            plt.title(f"{task_name} - {metric_name}")
            plt.xlabel("step")
            plt.ylabel(str(metric_name))
            plt.grid(True, linestyle=":", alpha=0.5)
            metric_safe = str(metric_name).replace(os.sep, "_").replace(" ", "_").replace(":", "_")
            task_safe = str(task_name).replace(os.sep, "_").replace(" ", "_").replace(":", "_")
            out_path = PLOTS_DIR / f"task_{task_safe}_{metric_safe}.png"
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            logging.info("[plot] 保存图像：%s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate checkpoints with lighteval")
    parser.add_argument("--ckpt-root", type=str, default=str(DEFAULT_CKPT_ROOT), help="checkpoints 根目录")
    parser.add_argument("--output-root", type=str, default=str(EVAL_DIR), help="评测输出根目录（默认 eval/）")
    parser.add_argument("--skip-existing", action="store_true", help="若已存在结果 JSON 则跳过该任务")
    parser.add_argument("--verbose", action="store_true", help="打印更多日志")
    parser.add_argument("--generate-sbatch", action="store_true", help="仅生成每个 step 的 sbatch 脚本与 submit_all.sh，不直接运行评测")
    parser.add_argument("--partition", type=str, default=None, help="SLURM 分区名，如 gpu")
    parser.add_argument("--time", dest="time_limit", type=str, default="04:00:00", help="作业时间限制，如 04:00:00")
    parser.add_argument("--gpus", type=int, default=1, help="每个作业申请的 GPU 数量")
    parser.add_argument("--cpus-per-task", type=int, default=8, help="每个作业的 CPU 数量")
    parser.add_argument("--just-plot", action="store_true", help="仅绘图，不进行评测")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    ckpt_root = Path(args.ckpt_root).resolve()
    output_root = Path(args.output_root).resolve()

    logging.info("ckpt_root=%s", ckpt_root)
    logging.info("output_root=%s", output_root)
    ensure_dirs()

    step_actor_dirs = collect_global_steps(ckpt_root)
    if not step_actor_dirs:
        logging.error("未找到任何 step 目录：%s", ckpt_root)
        sys.exit(2)

    agg_rows: List[Dict[str, object]] = []

    generated_scripts: List[Path] = []

    for actor_dir in step_actor_dirs:
        step_dir = actor_dir.parent
        step_num = extract_step_number(step_dir)
        if step_num is None:
            logging.warning("跳过无效 step 目录：%s", step_dir)
            continue

        hf_dir = step_dir / "actor/huggingface"
        # 生成 YAML（如不存在则生成/覆盖）
        yaml_path = TMP_MODEL_CONFIG_DIR / f"global_step_{step_num}.yaml"
        generate_vllm_yaml_for_step(model_dir=hf_dir, target_yaml=yaml_path)

        if args.generate_sbatch:
            script_text = build_sbatch_script(
                step_num=step_num,
                actor_dir=actor_dir,
                hf_dir=hf_dir,
                yaml_path=yaml_path,
                output_root=output_root,
                partition=args.partition,
                time_limit=args.time_limit,
                gpus=args.gpus,
                cpus_per_task=args.cpus_per_task,
            )
            script_path = SBATCH_DIR / f"batch_eval_step_{step_num}.sh"
            script_path.write_text(script_text, encoding="utf-8")
            os.chmod(script_path, 0o755)
            generated_scripts.append(script_path)
            logging.info("[sbatch] 已生成：%s", script_path)
            # 跳过直接评测
            continue

        # 非生成模式：直接在当前进程内转换与评测
        if not args.just_plot:
            convert_fsdp_to_hf_if_needed(actor_dir=actor_dir, hf_dir=hf_dir, verbose=args.verbose)

        # 逐任务评测
        for task in TASK_SPECS:
            # 先尝试查找该 step 的已存在结果
            existing_json = find_existing_result_json_for_step(
                output_root=output_root,
                task=task,
                step_num=step_num,
                hf_dir=hf_dir,
            )

            if args.just_plot:
                # 仅绘图：如果有现成结果就解析汇总；没有就跳过
                if existing_json:
                    metrics = parse_results_json(existing_json)
                    for metric, (val, stderr) in metrics.items():
                        agg_rows.append({
                            "step": step_num,
                            "task": task.name if task.name != "gpqa:diamond" else "gpqa",
                            "metric": metric,
                            "value": val,
                            "stderr": stderr,
                            "json_path": str(existing_json),
                        })
                else:
                    logging.info("[plot-only] 未找到现有结果，跳过 task=%s step=%s", task.name, step_num)
                continue

            # 需要评测：若 skip-existing 且已有结果则直接用
            if args.skip_existing and existing_json:
                logging.info("[skip] 已存在结果，使用现有 JSON：task=%s step=%s", task.name, step_num)
                metrics = parse_results_json(existing_json)
                for metric, (val, stderr) in metrics.items():
                    agg_rows.append({
                        "step": step_num,
                        "task": task.name if task.name != "gpqa:diamond" else "gpqa",
                        "metric": metric,
                        "value": val,
                        "stderr": stderr,
                        "json_path": str(existing_json),
                    })
                continue

            # 运行评测并写入 step 归档目录
            json_path = run_lighteval_once(
                yaml_path=yaml_path,
                task=task,
                output_root=output_root,
                step_num=step_num,
                verbose=args.verbose,
            )
            if not json_path:
                continue
            metrics = parse_results_json(json_path)
            for metric, (val, stderr) in metrics.items():
                agg_rows.append({
                    "step": step_num,
                    "task": task.name if task.name != "gpqa:diamond" else "gpqa",
                    "metric": metric,
                    "value": val,
                    "stderr": stderr,
                    "json_path": str(json_path),
                })

    if args.generate_sbatch:
        if generated_scripts:
            submit_all = write_submit_all_script(
                script_paths=generated_scripts,
                partition=args.partition,
                time_limit=args.time_limit,
                gpus=args.gpus,
            )
            logging.info("[sbatch] 提交脚本已生成：%s", submit_all)
            logging.info("使用示例：bash %s", submit_all)
        else:
            logging.warning("[sbatch] 未生成任何 sbatch 脚本，可能未发现 step 目录")
        return

    # 在保存与绘图前，附加 baseline 作为 step=0
    added = append_baseline_to_agg(agg_rows, output_root)
    if added > 0:
        logging.info("[aggregate] 已追加 baseline 行数：%s", added)

    # 直接模式：聚合与绘图
    aggregate_and_save(agg_rows)
    plot_metrics(agg_rows)


if __name__ == "__main__":
    main()


