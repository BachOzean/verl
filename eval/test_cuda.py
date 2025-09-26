#!/usr/bin/env python3
"""
CUDA 和 torch.distributed 可用性测试脚本
"""
import torch
import torch.multiprocessing as mp
import os

def setup_distributed_test(rank, world_size):
    """测试分布式环境设置"""
    try:
        # 设置环境变量
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # 设置CUDA设备
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
            print(f"[Rank {rank}] Using CUDA device: {torch.cuda.current_device()}")

        # 初始化分布式进程组
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        print(f"[Rank {rank}] Distributed group initialized successfully")

        # 测试通信
        tensor = torch.tensor([rank], dtype=torch.float32)
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        print(f"[Rank {rank}] All reduce successful, result: {tensor.item()}")

        torch.distributed.destroy_process_group()
        return True

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        return False

def test_cuda():
    print("🔍 CUDA 可用性测试")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {device_props.name} - {device_props.total_memory / 1024**3:.1f} GB")
            print(f"  - 计算能力: {device_props.major}.{device_props.minor}")

        # 测试 CUDA 初始化
        try:
            torch.cuda.init()
            print("✅ CUDA 初始化成功")
        except Exception as e:
            print(f"❌ CUDA 初始化失败: {e}")
            return False

        # 测试设备设置
        try:
            for i in range(min(2, torch.cuda.device_count())):
                torch.cuda.set_device(i)
                print(f"✅ 设备 {i} 设置成功")
        except Exception as e:
            print(f"❌ 设备设置失败: {e}")
            return False

        return True
    else:
        print("❌ CUDA 不可用")
        return False

def test_vllm_import():
    """测试 vLLM 是否可以正常导入"""
    try:
        from vllm import LLM, SamplingParams
        print("✅ vLLM 导入成功")
        return True
    except Exception as e:
        print(f"❌ vLLM 导入失败: {e}")
        return False

def test_distributed():
    """测试 torch.distributed 是否正常工作"""
    print("\n🔍 torch.distributed 测试")

    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过分布式测试")
        return True

    try:
        # 设置环境变量
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        world_size = min(2, torch.cuda.device_count())  # 测试使用2个GPU
        print(f"使用 {world_size} 个GPU进行分布式测试")

        mp.spawn(setup_distributed_test, args=(world_size,), nprocs=world_size, join=True)
        print("✅ torch.distributed 测试通过")
        return True

    except Exception as e:
        print(f"❌ torch.distributed 测试失败: {e}")
        return False

if __name__ == "__main__":
    cuda_ok = test_cuda()
    vllm_ok = test_vllm_import()
    dist_ok = test_distributed()

    print("\n" + "="*50)
    if cuda_ok and vllm_ok and dist_ok:
        print("🎉 所有测试通过，可以运行批推理脚本了!")
    else:
        print("⚠️  存在问题，需要解决后再运行批推理脚本")
        print(f"  - CUDA: {'✅' if cuda_ok else '❌'}")
        print(f"  - vLLM: {'✅' if vllm_ok else '❌'}")
        print(f"  - torch.distributed: {'✅' if dist_ok else '❌'}")
