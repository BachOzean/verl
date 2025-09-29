import argparse
import os
import json
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # fallback later

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--openai_model_name", type=str, default="local")

    parser.add_argument("--dataset", type=str, default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--samples_per_call", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=-1, help="<0 表示不设定，提升多样性")

    parser.add_argument("--max_concurrency", type=int, default=8)
    parser.add_argument("--timeout_s", type=int, default=600)
    parser.add_argument("--max_retries", type=int, default=5)

    return parser.parse_args()


def is_correct(gen: str, answer: Optional[str]) -> bool:
    if answer is None:
        return False
    ans = str(answer).strip()
    if ans == "":
        return False
    return ans in str(gen).strip()


def pick_value(d, keys: List[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def http_post_json(url: str, payload: Dict[str, Any], timeout_s: int) -> Tuple[int, str, Dict[str, Any]]:
    """POST JSON，优先用 requests，不可用时退回 urllib。返回 (status_code, text, json_or_empty)."""
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload)
    if requests is not None:
        resp = requests.post(url, data=data, headers=headers, timeout=timeout_s)
        try:
            j = resp.json()
        except Exception:
            j = {}
        return resp.status_code, resp.text, j
    # fallback
    import urllib.request  # type: ignore
    import urllib.error  # type: ignore
    req = urllib.request.Request(url, data=data.encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8")
            try:
                j = json.loads(text)
            except Exception:
                j = {}
            status = resp.getcode() or 0
            return status, text, j
    except urllib.error.HTTPError as e:  # pragma: no cover
        text = e.read().decode("utf-8", errors="ignore")
        try:
            j = json.loads(text)
        except Exception:
            j = {}
        return e.code, text, j


def looks_like_oom(text: str) -> bool:
    t = text.lower()
    return ("out of memory" in t) or ("cuda oom" in t) or ("allocation" in t and "failed" in t)


def parse_context_limit_error(text: str) -> Optional[Tuple[int, int, int]]:
    """从错误文本解析 (context_max, messages_tokens, completion_tokens)。
    例: "maximum context length is 4096 tokens. However, you requested 4178 tokens (82 in the messages, 4096 in the completion)"
    匹配失败返回 None。
    """
    try:
        m1 = re.search(r"maximum context length is\s+(\d+)\s+tokens", text, re.IGNORECASE)
        m2 = re.search(r"\((\d+)\s+in the messages,\s*(\d+)\s+in the completion\)", text, re.IGNORECASE)
        if m1 and m2:
            ctx = int(m1.group(1))
            msg_tok = int(m2.group(1))
            comp_tok = int(m2.group(2))
            return ctx, msg_tok, comp_tok
    except Exception:
        pass
    return None


def http_get_json(url: str, timeout_s: int) -> Tuple[int, str, Dict[str, Any]]:
    """GET JSON，优先用 requests，不可用时退回 urllib。返回 (status_code, text, json_or_empty)."""
    if requests is not None:
        resp = requests.get(url, timeout=timeout_s)
        text = resp.text
        try:
            j = resp.json()
        except Exception:
            j = {}
        return resp.status_code, text, j
    # fallback
    import urllib.request  # type: ignore
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8")
            try:
                j = json.loads(text)
            except Exception:
                j = {}
            status = resp.getcode() or 0
            return status, text, j
    except Exception as e:  # pragma: no cover
        return 0, str(e), {}


def resolve_model_name(server_url: str, provided: Optional[str], timeout_s: int) -> str:
    """解析要使用的模型名。如果 provided 为空/auto/local，则从 /v1/models 读取第一个 id。"""
    if provided and provided not in ("", "auto", "local"):
        return provided
    url = server_url.rstrip("/") + "/v1/models"
    status, text, j = http_get_json(url, timeout_s)
    if status == 200 and isinstance(j, dict):
        data = j.get("data", [])
        if isinstance(data, list) and len(data) > 0:
            mid = data[0].get("id")
            if isinstance(mid, str) and mid:
                return mid
    # 兜底：返回 provided 或报错
    if provided:
        return provided
    raise RuntimeError(f"Cannot resolve model name from server: status={status}, text={text[:200]}...")

def send_chat_request(
    server_url: str,
    model_name: str,
    prompt: str,
    n: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int],
    timeout_s: int,
    max_retries: int,
) -> List[str]:
    url = server_url.rstrip("/") + "/v1/chat/completions"
    cur_n = max(1, int(n))
    cur_max_tokens = max(1, int(max_tokens))
    attempt = 0
    while True:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": str(prompt)}],
            "n": cur_n,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(cur_max_tokens),
        }
        if seed is not None and seed >= 0:
            payload["seed"] = int(seed)

        status, text, j = http_post_json(url, payload, timeout_s)
        if status == 200 and "choices" in j:
            return [c.get("message", {}).get("content", "") for c in j["choices"]]

        # 错误处理与退让
        attempt += 1
        # 动态处理上下文长度报错
        ctx_info = parse_context_limit_error(text)
        if ctx_info is not None:
            ctx_max, used_msg, used_comp = ctx_info
            # 预留余量，避免边界再次失败
            new_max_tokens = max(1, min(cur_max_tokens, ctx_max - used_msg - 16))
            if new_max_tokens < cur_max_tokens:
                cur_max_tokens = new_max_tokens
                time.sleep(min(1.0 * attempt, 5.0))
                continue

        if attempt > max_retries:
            raise RuntimeError(f"Request failed after retries. status={status}, text={text[:200]}...")

        if looks_like_oom(text) and cur_n > 1:
            cur_n = max(1, cur_n // 2)
        # 简单指数退避
        time.sleep(min(2.0 * attempt, 10.0))


def process_one_example(
    ex: Any,
    prompt_keys: List[str],
    answer_keys: List[str],
    id_keys: List[str],
    server_url: str,
    model_name: str,
    target_samples: int,
    samples_per_call: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int],
    timeout_s: int,
    max_retries: int,
) -> Dict[str, Any]:
    if isinstance(ex, str):
        prompt = ex
        answer = None
        qid = None
    elif isinstance(ex, dict):
        prompt = pick_value(ex, prompt_keys, default=str(ex))
        answer = pick_value(ex, answer_keys, default=None)
        qid = pick_value(ex, id_keys, default=None)
    else:
        prompt = str(ex)
        answer = None
        qid = None

    aggregated: List[str] = []
    remaining = max(1, int(target_samples))
    spc = max(1, int(samples_per_call))

    while remaining > 0:
        cur_n = min(spc, remaining)
        gens = send_chat_request(
            server_url=server_url,
            model_name=model_name,
            prompt=prompt,
            n=cur_n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        aggregated.extend(gens)
        remaining -= cur_n

    if answer is None or str(answer).strip() == "":
        correct_count = 0
        status = "valid"
    else:
        correct_count = sum(is_correct(g, answer) for g in aggregated)
        if correct_count == target_samples:
            status = "all_correct"
        elif correct_count == 0:
            status = "all_wrong"
        else:
            status = "valid"

    return {
        "id": qid,
        "problem": prompt,
        "answer": answer,
        "generations": aggregated,
        "correct_count": correct_count,
        "status": status,
    }


def main():
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 解析服务端模型名
    resolved_model_name = resolve_model_name(args.server_url, args.openai_model_name, args.timeout_s)
    print(f"Using OpenAI model name: {resolved_model_name}")

    dataset = load_dataset(args.dataset, split=args.split)
    total = len(dataset)
    print(f"Dataset loaded: {args.dataset}/{args.split}, size={total}")

    prompt_keys = [
        "problem", "question", "query", "prompt", "instruction", "input", "text"
    ]
    answer_keys = [
        "answer", "label", "target", "output", "solution", "final_answer"
    ]
    id_keys = [
        "id", "_id", "question_id", "idx", "index"
    ]

    valid_path = os.path.join(args.output_dir, "valid.jsonl")
    all_correct_path = os.path.join(args.output_dir, "all_correct.jsonl")
    all_wrong_path = os.path.join(args.output_dir, "all_wrong.jsonl")

    lock = None
    import threading
    lock = threading.Lock()

    counts = {"valid": 0, "all_correct": 0, "all_wrong": 0}
    processed = 0

    with open(valid_path, "w", encoding="utf-8") as f_valid, \
         open(all_correct_path, "w", encoding="utf-8") as f_all_correct, \
         open(all_wrong_path, "w", encoding="utf-8") as f_all_wrong, \
         ThreadPoolExecutor(max_workers=max(1, int(args.max_concurrency))) as executor:

        pending = set()

        def submit_one(example):
            return executor.submit(
                process_one_example,
                example,
                prompt_keys,
                answer_keys,
                id_keys,
                args.server_url,
                resolved_model_name,
                args.num_samples,
                args.samples_per_call,
                args.max_tokens,
                args.temperature,
                args.top_p,
                (None if args.seed < 0 else int(args.seed)),
                args.timeout_s,
                args.max_retries,
            )

        # 提交任务并控制并发
        for ex in dataset:
            pending.add(submit_one(ex))
            if len(pending) >= args.max_concurrency:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    rec = fut.result()
                    status = rec.get("status", "valid")
                    with lock:
                        if status == "all_correct":
                            f_all_correct.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            counts["all_correct"] += 1
                        elif status == "all_wrong":
                            f_all_wrong.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            counts["all_wrong"] += 1
                        else:
                            f_valid.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            counts["valid"] += 1
                        processed += 1
                    if processed % 50 == 0:
                        print(f"Processed {processed}/{total}")

        # 收尾，等待剩余任务
        if len(pending) > 0:
            done, _ = wait(pending)
            for fut in done:
                rec = fut.result()
                status = rec.get("status", "valid")
                with lock:
                    if status == "all_correct":
                        f_all_correct.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        counts["all_correct"] += 1
                    elif status == "all_wrong":
                        f_all_wrong.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        counts["all_wrong"] += 1
                    else:
                        f_valid.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        counts["valid"] += 1
                    processed += 1

    print(
        f"✅ 完成! valid={counts['valid']}, all_correct={counts['all_correct']}, all_wrong={counts['all_wrong']}"
    )


if __name__ == "__main__":
    main()


