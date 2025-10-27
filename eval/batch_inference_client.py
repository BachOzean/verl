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
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000", help="单个服务器URL（当不使用多服务器时）")
    parser.add_argument("--server_urls", type=str, default=None, help="多个服务器URL，用逗号分隔，如：http://127.0.0.1:8000,http://127.0.0.1:8001")
    parser.add_argument("--openai_model_name", type=str, default="local")

    parser.add_argument("--dataset", type=str, default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--samples_per_call", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0)
    parser.add_argument("--seed", type=int, default=-1, help="<0 表示不设定，提升多样性")

    parser.add_argument("--max_concurrency", type=int, default=8)
    parser.add_argument("--timeout_s", type=int, default=3000)
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


def send_batch_requests(server_url, requests_list, timeout_s, max_retries):
    """批量发送请求并返回结果"""
    results = []

    # 限制并发请求数量，避免服务器过载
    max_concurrent = min(1024, len(requests_list))  # 限制为16个并发请求

    # 分批处理请求，每批最多max_concurrent个
    for i in range(0, len(requests_list), max_concurrent):
        batch = requests_list[i:i + max_concurrent]
        print(f"Sending batch {i//max_concurrent + 1}/{(len(requests_list) + max_concurrent - 1)//max_concurrent} ({len(batch)} requests)")

        # 使用ThreadPoolExecutor并发发送本批请求
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_request = {
                executor.submit(send_single_request, server_url, req, timeout_s, max_retries): req
                for req in batch
            }

            for future in future_to_request:
                try:
                    result = future.result(timeout=timeout_s + 10)
                    results.append(result)
                except Exception as e:
                    print(f"Request failed: {e}")
                    # 返回一个错误的结果
                    results.append([])

    return results


def send_single_request(server_url, request_data, timeout_s, max_retries):
    """发送单个请求"""
    model_name, prompt, n, max_tokens, temperature, top_p, top_k, min_p, seed = request_data

    return send_chat_request(
        server_url=server_url,
        model_name=model_name,
        prompt=prompt,
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        seed=seed,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )


def process_chunk(chunk_args):
    """独立处理一个数据分片的函数（全局函数，可被多进程序列化）"""
    (chunk_idx, chunk_data, server_url, chunk_output_dir,
     model_name, num_samples, samples_per_call, max_tokens,
     temperature, top_p, top_k, min_p, seed, timeout_s, max_retries) = chunk_args

    print(f"Processing chunk {chunk_idx} with {len(chunk_data)} samples on {server_url}")

    # 为这个chunk创建独立的结果文件
    valid_path = os.path.join(chunk_output_dir, "valid.jsonl")
    all_correct_path = os.path.join(chunk_output_dir, "all_correct.jsonl")
    all_wrong_path = os.path.join(chunk_output_dir, "all_wrong.jsonl")

    counts = {"valid": 0, "all_correct": 0, "all_wrong": 0}

    prompt_keys = [
        "problem", "question", "query", "prompt", "instruction", "input", "text"
    ]
    answer_keys = [
        "answer", "label", "target", "output", "solution", "final_answer"
    ]
    id_keys = [
        "id", "_id", "question_id", "idx", "index"
    ]

    # 预先生成所有请求
    print("Generating all requests...")
    all_requests = []
    sample_to_requests = {}  # 记录每个样本需要的所有请求

    for ex in chunk_data:
        # 解析样本信息
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

        # 为这个样本生成所有需要的请求（num_samples个请求）
        sample_requests = []
        for _ in range(num_samples):
            sample_requests.append((
                model_name, prompt, 1, max_tokens, temperature, top_p, top_k, min_p, seed
            ))
        all_requests.extend(sample_requests)
        sample_to_requests[prompt] = {
            'answer': answer,
            'qid': qid,
            'requests': sample_requests
        }

    print(f"Total requests to send: {len(all_requests)}")

    # 批量发送所有请求
    print("Sending all requests in batch...")
    batch_results = send_batch_requests(server_url, all_requests, timeout_s, max_retries)

    # 处理结果
    print("Processing batch results...")
    with open(valid_path, "w", encoding="utf-8") as f_valid, \
         open(all_correct_path, "w", encoding="utf-8") as f_all_correct, \
         open(all_wrong_path, "w", encoding="utf-8") as f_all_wrong:

        # 将结果重新组织回每个样本
        result_idx = 0
        for prompt, sample_info in sample_to_requests.items():
            answer = sample_info['answer']
            qid = sample_info['qid']
            num_requests_for_sample = len(sample_info['requests'])

            # 收集这个样本的所有生成结果
            sample_generations = []
            for _ in range(num_requests_for_sample):
                if result_idx < len(batch_results):
                    generation = batch_results[result_idx]
                    if generation:  # 确保不是空结果
                        sample_generations.extend(generation)
                    result_idx += 1
                else:
                    break

            # 计算正确率
            if not sample_generations:
                correct_count = 0
                status = "error"
            elif answer is None or str(answer).strip() == "":
                correct_count = 0
                status = "valid"
            else:
                correct_count = sum(is_correct(g, answer) for g in sample_generations)
                if correct_count == num_samples:
                    status = "all_correct"
                elif correct_count == 0:
                    status = "all_wrong"
                else:
                    status = "valid"

            # 写入结果
            rec = {
                "id": qid,
                "problem": prompt,
                "answer": answer,
                "generations": sample_generations,
                "correct_count": correct_count,
                "status": status,
            }

            if status == "all_correct":
                f_all_correct.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["all_correct"] += 1
            elif status == "all_wrong":
                f_all_wrong.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["all_wrong"] += 1
            elif status == "error":
                # 错误的结果也写入valid文件，但标记为error
                f_valid.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["valid"] += 1
            else:
                f_valid.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["valid"] += 1

    print(f"Chunk {chunk_idx} completed: valid={counts['valid']}, all_correct={counts['all_correct']}, all_wrong={counts['all_wrong']}")
    return counts


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


def get_server_urls(args) -> List[str]:
    """从参数中获取服务器URL列表"""
    if args.server_urls:
        return [url.strip() for url in args.server_urls.split(",") if url.strip()]
    else:
        return [args.server_url]


def resolve_model_name(server_urls: List[str], provided: Optional[str], timeout_s: int) -> str:
    """解析要使用的模型名。如果 provided 为空/auto/local，则从第一个服务器读取。"""
    if provided and provided not in ("", "auto", "local"):
        return provided

    # 从第一个服务器获取模型名称
    server_url = server_urls[0]
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
    top_k: int,
    min_p: float,
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
            "top_k": int(top_k),
            "min_p": float(min_p),
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
    server_urls: List[str],
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
    spc = max(1, int(samples_per_call))
    server_index = 0  # 简单的轮询负载均衡

    # 轮询选择服务器
    server_url = server_urls[server_index % len(server_urls)]
    server_index += 1

    gens = send_chat_request(
        server_url=server_url,
        model_name=model_name,
        prompt=prompt,
        n=spc,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
    aggregated.extend(gens)

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

    # 获取服务器URL列表
    server_urls = get_server_urls(args)
    num_servers = len(server_urls)
    print(f"Using {num_servers} server URLs: {server_urls}")

    # 解析服务端模型名
    resolved_model_name = resolve_model_name(server_urls, args.openai_model_name, args.timeout_s)
    print(f"Using OpenAI model name: {resolved_model_name}")

    dataset = load_dataset(args.dataset, split=args.split)
    total = len(dataset)
    print(f"Dataset loaded: {args.dataset}/{args.split}, size={total}")

    # 数据预切分：将数据集分成num_servers个分片
    print(f"Pre-sharding dataset into {num_servers} chunks...")
    chunk_size = total // num_servers
    data_chunks = []

    for i in range(num_servers):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_servers - 1 else total
        chunk = dataset.select(range(start_idx, end_idx))
        data_chunks.append(chunk)
        print(f"  Chunk {i}: indices [{start_idx}:{end_idx}] ({len(chunk)} samples)")

    # 为每个数据分片创建独立的结果目录
    chunk_output_dirs = []
    for i in range(num_servers):
        chunk_dir = os.path.join(args.output_dir, f"chunk_{i}")
        Path(chunk_dir).mkdir(exist_ok=True)
        chunk_output_dirs.append(chunk_dir)

    # 并行处理每个数据分片
    print("Starting parallel processing of data chunks...")
    start_time = time.time()

    # 准备并行处理参数
    from concurrent.futures import ProcessPoolExecutor

    chunk_args = [
        (i, data_chunks[i], server_urls[i], chunk_output_dirs[i],
         resolved_model_name, args.num_samples, args.samples_per_call,
         args.max_tokens, args.temperature, args.top_p, args.top_k, args.min_p,
         (None if args.seed < 0 else int(args.seed)), args.timeout_s, args.max_retries)
        for i in range(num_servers)
    ]

    # 并行处理所有chunks
    with ProcessPoolExecutor(max_workers=num_servers) as executor:
        results = list(executor.map(process_chunk, chunk_args))

    # 合并结果
    print("Merging results from all chunks...")
    total_counts = {"valid": 0, "all_correct": 0, "all_wrong": 0}

    for i, counts in enumerate(results):
        for key in total_counts:
            total_counts[key] += counts[key]
        print(f"  Chunk {i}: {counts}")

        # 合并这个chunk的结果文件到主目录
        chunk_dir = chunk_output_dirs[i]
        for result_type in ["valid", "all_correct", "all_wrong"]:
            chunk_file = os.path.join(chunk_dir, f"{result_type}.jsonl")
            main_file = os.path.join(args.output_dir, f"{result_type}.jsonl")

            if os.path.exists(chunk_file):
                with open(chunk_file, "r", encoding="utf-8") as src, \
                     open(main_file, "a", encoding="utf-8") as dst:
                    dst.write(src.read())

    end_time = time.time()
    total_time = end_time - start_time

    print(
        f"✅ 完成! valid={total_counts['valid']}, all_correct={total_counts['all_correct']}, all_wrong={total_counts['all_wrong']}"
    )
    print(f"⏱️  总处理时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")


if __name__ == "__main__":
    main()


