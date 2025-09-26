import argparse
import os
import json
import torch
from pathlib import Path
from datasets import load_dataset
from vllm import LLM, SamplingParams
from torch.distributed import init_process_group, get_rank, get_world_size, barrier
import torch.multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--split", type=str, default="default")
    parser.add_argument("--output_dir", type=str, default="/home/ningmiao/ningyuan/verl/eval/results/OpenR1-Math-220k")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    # 可控的模型/生成长度与精度，便于控制显存占用
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "auto"],
    )
    parser.add_argument("--max_num_seqs", type=int, default=1)
    return parser.parse_args()

def is_correct(gen: str, answer: str) -> bool:
    """简单的正确性判定"""
    if answer is None:
        return False
    ans = str(answer).strip()
    if ans == "":
        return False
    return ans in str(gen).strip()

def setup_distributed(rank, world_size):
    """设置分布式环境"""
    # 设置必要的环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # 设置CUDA设备
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        print(f"[Rank {rank}] Using CUDA device: {torch.cuda.current_device()}")

    # 初始化分布式进程组
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"[Rank {rank}] Distributed group initialized")

def main_worker(rank, world_size, args):
    """每个进程的工作函数"""
    try:
        # 关键：将每个进程限制只看到一个独立的 GPU，避免 vLLM 误占同一张卡
        # 必须在任何 CUDA 操作与 vLLM 初始化之前设置
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

        # 设置分布式环境
        setup_distributed(rank, world_size)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # 等待所有进程准备就绪
        barrier()

        print(f"[Rank {rank}] Starting inference...")

        # 加载数据并分 shard
        dataset = load_dataset(args.dataset, split=args.split)
        print(f"[Rank {rank}] Dataset size: {len(dataset)}")
        shard = dataset.shard(num_shards=world_size, index=rank)

        # 加载模型 - 每个进程使用1个GPU
        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            max_num_seqs=args.max_num_seqs,
        )
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            n=args.num_samples,
            max_tokens=args.max_tokens,
        )

        # 流式写出每个 shard 的结果，避免在内存中堆积
        shard_file = os.path.join(args.output_dir, f"shard_{rank}.jsonl")
        written_count = 0
        with open(shard_file, "w", encoding="utf-8") as f:
            for i in range(0, len(shard), args.batch_size):
                batch = shard[i : i + args.batch_size]
                # 兼容多种样本格式：dict、字符串、dict-of-lists 等
                prompt_keys = [
                    "problem", "question", "query", "prompt", "instruction", "input", "text"
                ]
                answer_keys = [
                    "answer", "label", "target", "output", "solution", "final_answer"
                ]
                id_keys = [
                    "id", "_id", "question_id", "idx", "index"
                ]

                def pick_value(d, keys, default=None):
                    for k in keys:
                        if isinstance(d, dict) and k in d and d[k] is not None:
                            return d[k]
                    return default

                prompts, answers, ids = [], [], []
                try:
                    # 优先按“列表/可迭代样本”的形式读取
                    for local_idx, ex in enumerate(batch):
                        if isinstance(ex, str):
                            prompts.append(ex)
                            answers.append(None)
                            ids.append(None)
                        elif isinstance(ex, dict):
                            p = pick_value(ex, prompt_keys, default=str(ex))
                            a = pick_value(ex, answer_keys, default=None)
                            qid = pick_value(ex, id_keys, default=None)
                            prompts.append(p)
                            answers.append(a)
                            ids.append(qid)
                        else:
                            prompts.append(str(ex))
                            answers.append(None)
                            ids.append(None)
                except TypeError:
                    # 某些切片可能返回 dict-of-lists
                    if isinstance(batch, dict):
                        # 在列里选择第一个可用字段
                        col_prompt = next((c for c in prompt_keys if c in batch), None)
                        col_answer = next((c for c in answer_keys if c in batch), None)
                        col_id = next((c for c in id_keys if c in batch), None)
                        size = len(next(iter(batch.values()))) if batch else 0
                        for t in range(size):
                            prompts.append(batch[col_prompt][t] if col_prompt else str({k: batch[k][t] for k in batch}))
                            answers.append(batch[col_answer][t] if col_answer else None)
                            ids.append(batch[col_id][t] if col_id else None)
                    else:
                        raise

                try:
                    outputs = llm.generate(prompts, sampling_params)

                    for j, out in enumerate(outputs):
                        generations = [o.text for o in out.outputs]
                        gt_answer = answers[j]

                        # 当无标准答案时，不做正确性判断，标记为 valid
                        if gt_answer is None or str(gt_answer).strip() == "":
                            correct_count = 0
                            status = "valid"
                        else:
                            correct_count = sum(is_correct(gen, gt_answer) for gen in generations)
                            if correct_count == args.num_samples:
                                status = "all_correct"
                            elif correct_count == 0:
                                status = "all_wrong"
                            else:
                                status = "valid"

                        record = {
                            "id": ids[j],
                            "problem": prompts[j],
                            "answer": gt_answer,
                            "generations": generations,
                            "correct_count": correct_count,
                            "status": status,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written_count += 1

                    print(f"[Rank {rank}] Processed {i+len(batch)}/{len(shard)} samples")

                except Exception as e:
                    print(f"[Rank {rank}] Error in batch {i}: {e}")
                    continue

        print(f"[Rank {rank}] Saved {written_count} results to {shard_file}")

        # 同步所有进程
        barrier()

        # rank=0 合并所有结果（流式处理，避免一次性读入内存）
        if rank == 0:
            valid_path = os.path.join(args.output_dir, "valid.jsonl")
            all_correct_path = os.path.join(args.output_dir, "all_correct.jsonl")
            all_wrong_path = os.path.join(args.output_dir, "all_wrong.jsonl")

            counts = {"valid": 0, "all_correct": 0, "all_wrong": 0}
            with open(valid_path, "w", encoding="utf-8") as f_valid, \
                 open(all_correct_path, "w", encoding="utf-8") as f_all_correct, \
                 open(all_wrong_path, "w", encoding="utf-8") as f_all_wrong:

                for i in range(world_size):
                    shard_file_i = os.path.join(args.output_dir, f"shard_{i}.jsonl")
                    if not os.path.exists(shard_file_i):
                        continue
                    with open(shard_file_i, encoding="utf-8") as f_in:
                        for line in f_in:
                            try:
                                r = json.loads(line)
                            except Exception:
                                continue
                            status = r.get("status", "valid")
                            if status == "all_correct":
                                f_all_correct.write(json.dumps(r, ensure_ascii=False) + "\n")
                                counts["all_correct"] += 1
                            elif status == "all_wrong":
                                f_all_wrong.write(json.dumps(r, ensure_ascii=False) + "\n")
                                counts["all_wrong"] += 1
                            else:
                                f_valid.write(json.dumps(r, ensure_ascii=False) + "\n")
                                counts["valid"] += 1

            print(
                f"✅ 完成! valid={counts['valid']}, "
                f"all_correct={counts['all_correct']}, "
                f"all_wrong={counts['all_wrong']}"
            )

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        raise

def main():
    args = parse_args()

    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用!")
        return

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    # 设置分布式环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # 设置多进程
    mp.set_start_method("spawn", force=True)

    # 获取世界大小（GPU数量或指定数量）
    world_size = min(args.num_gpus, torch.cuda.device_count())
    print(f"Using {world_size} GPUs")

    # 启动多进程
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
