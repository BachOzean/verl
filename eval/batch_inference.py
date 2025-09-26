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
    return parser.parse_args()

def is_correct(gen: str, answer: str) -> bool:
    """简单的正确性判定"""
    return answer.strip() in gen.strip()

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
            dtype="bfloat16",
            max_model_len=4096,
            trust_remote_code=True,
            max_num_seqs=1,
        )
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            n=args.num_samples,
            max_tokens=1024,
        )

        results = []
        for i in range(0, len(shard), args.batch_size):
            batch = shard[i : i + args.batch_size]
            prompts = [ex["problem"] for ex in batch]
            answers = [ex["answer"] for ex in batch]
            ids = [ex["id"] for ex in batch]

            try:
                outputs = llm.generate(prompts, sampling_params)

                for j, out in enumerate(outputs):
                    generations = [o.text for o in out.outputs]
                    gt_answer = answers[j]

                    correct_count = sum(is_correct(gen, gt_answer) for gen in generations)
                    if correct_count == args.num_samples:
                        status = "all_correct"
                    elif correct_count == 0:
                        status = "all_wrong"
                    else:
                        status = "valid"

                    results.append({
                        "id": ids[j],
                        "problem": prompts[j],
                        "answer": gt_answer,
                        "generations": generations,
                        "correct_count": correct_count,
                        "status": status,
                    })

                print(f"[Rank {rank}] Processed {i+len(batch)}/{len(shard)} samples")

            except Exception as e:
                print(f"[Rank {rank}] Error in batch {i}: {e}")
                continue

        # 保存每个 shard 的结果
        shard_file = os.path.join(args.output_dir, f"shard_{rank}.jsonl")
        with open(shard_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[Rank {rank}] Saved {len(results)} results to {shard_file}")

        # 同步所有进程
        barrier()

        # rank=0 合并所有结果
        if rank == 0:
            merged = []
            for i in range(world_size):
                shard_file = os.path.join(args.output_dir, f"shard_{i}.jsonl")
                if os.path.exists(shard_file):
                    with open(shard_file, encoding="utf-8") as f:
                        merged.extend([json.loads(line) for line in f])

            subsets = {"valid": [], "all_correct": [], "all_wrong": []}
            for r in merged:
                subsets[r["status"]].append(r)

            for k, v in subsets.items():
                output_file = os.path.join(args.output_dir, f"{k}.jsonl")
                with open(output_file, "w", encoding="utf-8") as f:
                    for r in v:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            print(f"✅ 完成! valid={len(subsets['valid'])}, "
                  f"all_correct={len(subsets['all_correct'])}, "
                  f"all_wrong={len(subsets['all_wrong'])}")

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
