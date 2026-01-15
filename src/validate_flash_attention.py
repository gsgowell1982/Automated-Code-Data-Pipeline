"""
Flash Attention 功能验证脚本

此脚本用于验证 Flash Attention 的功能正确性和性能优势。
主要包含：
1. 标准注意力机制实现（作为基准）
2. Flash Attention 调用
3. 数值正确性验证
4. 性能对比测试
"""

import torch
import torch.nn.functional as F
import time
import math
from typing import Optional, Tuple


def standard_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    标准的缩放点积注意力实现（用作基准对比）
    
    Args:
        query: (batch, num_heads, seq_len, head_dim)
        key: (batch, num_heads, seq_len, head_dim)
        value: (batch, num_heads, seq_len, head_dim)
        attn_mask: 可选的注意力掩码
        dropout_p: dropout概率
        is_causal: 是否使用因果掩码（用于自回归模型）
        scale: 缩放因子，默认为 1/sqrt(head_dim)
    
    Returns:
        注意力输出 (batch, num_heads, seq_len, head_dim)
    """
    L, S = query.size(-2), key.size(-2)
    head_dim = query.size(-1)
    
    # 计算缩放因子
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # 计算注意力分数: Q @ K^T / sqrt(d_k)
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # 应用因果掩码
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(L, S, dtype=torch.bool, device=query.device), 
            diagonal=1
        )
        attn_weight = attn_weight.masked_fill(causal_mask, float('-inf'))
    
    # 应用注意力掩码
    if attn_mask is not None:
        attn_weight = attn_weight + attn_mask
    
    # Softmax归一化
    attn_weight = F.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query.dtype)
    
    # 应用dropout
    if dropout_p > 0.0:
        attn_weight = F.dropout(attn_weight, p=dropout_p, training=True)
    
    # 计算输出: attn_weight @ V
    output = torch.matmul(attn_weight, value)
    
    return output


def check_flash_attention_available() -> Tuple[bool, str]:
    """
    检查 Flash Attention 是否可用
    
    Returns:
        (是否可用, 详细信息)
    """
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        return False, "CUDA 不可用。Flash Attention 需要 NVIDIA GPU。"
    
    # 检查PyTorch版本
    torch_version = torch.__version__
    major, minor = map(int, torch_version.split('.')[:2])
    
    if major < 2:
        return False, f"PyTorch 版本 {torch_version} 过低。Flash Attention 需要 PyTorch >= 2.0。"
    
    # 检查GPU计算能力
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    gpu_name = torch.cuda.get_device_name(device)
    
    if capability[0] < 8:
        # Flash Attention 2 需要 SM80+（Ampere及以上）
        # 但PyTorch的SDPA在较老的GPU上会回退到其他实现
        return True, (
            f"GPU: {gpu_name} (SM{capability[0]}{capability[1]})\n"
            f"注意: Flash Attention 2 需要 Ampere (SM80+) 或更新架构。\n"
            f"当前GPU将使用 PyTorch SDPA 的回退实现。"
        )
    
    return True, f"GPU: {gpu_name} (SM{capability[0]}{capability[1]}) - 支持 Flash Attention"


def validate_correctness(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
    is_causal: bool = False,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> Tuple[bool, dict]:
    """
    验证 Flash Attention 的数值正确性
    
    通过对比标准注意力实现和 PyTorch 的 scaled_dot_product_attention
    
    Args:
        batch_size: 批次大小
        num_heads: 注意力头数
        seq_len: 序列长度
        head_dim: 每个头的维度
        dtype: 数据类型
        is_causal: 是否使用因果掩码
        rtol: 相对误差容忍度
        atol: 绝对误差容忍度
    
    Returns:
        (是否通过, 详细结果)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成随机输入
    torch.manual_seed(42)
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    # 标准注意力输出
    with torch.no_grad():
        standard_output = standard_attention(
            query, key, value, 
            is_causal=is_causal
        )
    
    # Flash Attention (通过PyTorch SDPA)
    with torch.no_grad():
        flash_output = F.scaled_dot_product_attention(
            query, key, value,
            is_causal=is_causal,
            dropout_p=0.0,
        )
    
    # 计算误差
    abs_diff = torch.abs(standard_output - flash_output)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    
    rel_diff = abs_diff / (torch.abs(standard_output) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    # 判断是否通过
    passed = torch.allclose(standard_output, flash_output, rtol=rtol, atol=atol)
    
    results = {
        "passed": passed,
        "config": {
            "batch_size": batch_size,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "dtype": str(dtype),
            "is_causal": is_causal,
        },
        "errors": {
            "max_absolute": max_abs_diff,
            "mean_absolute": mean_abs_diff,
            "max_relative": max_rel_diff,
            "mean_relative": mean_rel_diff,
        },
        "tolerances": {
            "rtol": rtol,
            "atol": atol,
        }
    }
    
    return passed, results


def benchmark_performance(
    batch_size: int = 4,
    num_heads: int = 32,
    seq_len: int = 2048,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    is_causal: bool = True,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> dict:
    """
    性能基准测试
    
    对比标准注意力和 Flash Attention 的性能
    
    Args:
        batch_size: 批次大小
        num_heads: 注意力头数
        seq_len: 序列长度
        head_dim: 每个头的维度
        dtype: 数据类型
        is_causal: 是否使用因果掩码
        num_warmup: 预热迭代次数
        num_iterations: 测试迭代次数
    
    Returns:
        性能测试结果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成随机输入
    torch.manual_seed(42)
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    results = {
        "config": {
            "batch_size": batch_size,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "dtype": str(dtype),
            "is_causal": is_causal,
            "device": str(device),
        }
    }
    
    # 测试标准注意力性能
    if device.type == "cuda":
        # 预热
        for _ in range(num_warmup):
            _ = standard_attention(query, key, value, is_causal=is_causal)
        torch.cuda.synchronize()
        
        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_iterations):
            _ = standard_attention(query, key, value, is_causal=is_causal)
        end_event.record()
        torch.cuda.synchronize()
        
        standard_time = start_event.elapsed_time(end_event) / num_iterations
        results["standard_attention_ms"] = standard_time
    else:
        # CPU计时
        for _ in range(num_warmup):
            _ = standard_attention(query, key, value, is_causal=is_causal)
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = standard_attention(query, key, value, is_causal=is_causal)
        end = time.perf_counter()
        
        standard_time = (end - start) * 1000 / num_iterations
        results["standard_attention_ms"] = standard_time
    
    # 测试 Flash Attention (SDPA) 性能
    if device.type == "cuda":
        # 预热
        for _ in range(num_warmup):
            _ = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        torch.cuda.synchronize()
        
        # 计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_iterations):
            _ = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        end_event.record()
        torch.cuda.synchronize()
        
        flash_time = start_event.elapsed_time(end_event) / num_iterations
        results["flash_attention_ms"] = flash_time
    else:
        # CPU计时
        for _ in range(num_warmup):
            _ = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
        end = time.perf_counter()
        
        flash_time = (end - start) * 1000 / num_iterations
        results["flash_attention_ms"] = flash_time
    
    # 计算加速比
    results["speedup"] = standard_time / flash_time if flash_time > 0 else 0
    
    # 估算内存使用
    # 标准注意力需要存储 attention weights: O(batch * heads * seq^2)
    # Flash Attention 不需要存储完整的 attention matrix
    attn_matrix_size = batch_size * num_heads * seq_len * seq_len * 2  # fp16 = 2 bytes
    results["memory_estimate"] = {
        "standard_attention_matrix_mb": attn_matrix_size / (1024 * 1024),
        "flash_attention_note": "Flash Attention 使用分块计算，不存储完整注意力矩阵"
    }
    
    return results


def test_different_configurations():
    """
    测试不同配置下的 Flash Attention
    """
    configurations = [
        # (batch, heads, seq_len, head_dim, dtype, is_causal)
        (1, 8, 128, 64, torch.float16, False),
        (2, 8, 256, 64, torch.float16, False),
        (2, 8, 512, 64, torch.float16, True),
        (4, 16, 1024, 64, torch.float16, True),
        (1, 32, 2048, 128, torch.float16, True),
    ]
    
    if torch.cuda.is_available():
        # 添加 bfloat16 测试（如果支持）
        if torch.cuda.is_bf16_supported():
            configurations.append((2, 8, 512, 64, torch.bfloat16, True))
    
    results = []
    
    for batch, heads, seq_len, head_dim, dtype, is_causal in configurations:
        try:
            passed, result = validate_correctness(
                batch_size=batch,
                num_heads=heads,
                seq_len=seq_len,
                head_dim=head_dim,
                dtype=dtype,
                is_causal=is_causal,
            )
            results.append(result)
        except Exception as e:
            results.append({
                "passed": False,
                "config": {
                    "batch_size": batch,
                    "num_heads": heads,
                    "seq_len": seq_len,
                    "head_dim": head_dim,
                    "dtype": str(dtype),
                    "is_causal": is_causal,
                },
                "error": str(e)
            })
    
    return results


def main():
    """主函数"""
    print("=" * 70)
    print("Flash Attention 功能验证脚本")
    print("=" * 70)
    
    # 1. 检查环境
    print("\n[1] 环境检查")
    print("-" * 50)
    available, info = check_flash_attention_available()
    print(f"Flash Attention 可用: {available}")
    print(f"详细信息: {info}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    
    # 2. 数值正确性验证
    print("\n[2] 数值正确性验证")
    print("-" * 50)
    
    test_results = test_different_configurations()
    
    all_passed = True
    for i, result in enumerate(test_results):
        config = result["config"]
        status = "✓ 通过" if result.get("passed", False) else "✗ 失败"
        if not result.get("passed", False):
            all_passed = False
        
        print(f"\n测试 {i+1}: {status}")
        print(f"  配置: batch={config['batch_size']}, heads={config['num_heads']}, "
              f"seq_len={config['seq_len']}, head_dim={config['head_dim']}, "
              f"dtype={config['dtype']}, causal={config['is_causal']}")
        
        if "errors" in result:
            errors = result["errors"]
            print(f"  最大绝对误差: {errors['max_absolute']:.6e}")
            print(f"  平均绝对误差: {errors['mean_absolute']:.6e}")
        elif "error" in result:
            print(f"  错误: {result['error']}")
    
    print(f"\n数值正确性验证: {'全部通过 ✓' if all_passed else '存在失败 ✗'}")
    
    # 3. 性能基准测试
    print("\n[3] 性能基准测试")
    print("-" * 50)
    
    if torch.cuda.is_available():
        # 小规模测试
        print("\n小规模测试 (batch=2, heads=8, seq=512, dim=64):")
        perf_small = benchmark_performance(
            batch_size=2, num_heads=8, seq_len=512, head_dim=64,
            num_warmup=5, num_iterations=50
        )
        print(f"  标准注意力: {perf_small['standard_attention_ms']:.3f} ms")
        print(f"  Flash Attention: {perf_small['flash_attention_ms']:.3f} ms")
        print(f"  加速比: {perf_small['speedup']:.2f}x")
        
        # 中等规模测试
        print("\n中等规模测试 (batch=4, heads=16, seq=1024, dim=64):")
        perf_medium = benchmark_performance(
            batch_size=4, num_heads=16, seq_len=1024, head_dim=64,
            num_warmup=5, num_iterations=50
        )
        print(f"  标准注意力: {perf_medium['standard_attention_ms']:.3f} ms")
        print(f"  Flash Attention: {perf_medium['flash_attention_ms']:.3f} ms")
        print(f"  加速比: {perf_medium['speedup']:.2f}x")
        
        # 大规模测试（如果内存允许）
        try:
            print("\n大规模测试 (batch=2, heads=32, seq=2048, dim=128):")
            perf_large = benchmark_performance(
                batch_size=2, num_heads=32, seq_len=2048, head_dim=128,
                num_warmup=3, num_iterations=20
            )
            print(f"  标准注意力: {perf_large['standard_attention_ms']:.3f} ms")
            print(f"  Flash Attention: {perf_large['flash_attention_ms']:.3f} ms")
            print(f"  加速比: {perf_large['speedup']:.2f}x")
            print(f"  注意力矩阵内存(标准): {perf_large['memory_estimate']['standard_attention_matrix_mb']:.1f} MB")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("  跳过大规模测试（GPU内存不足）")
            else:
                raise
    else:
        print("GPU 不可用，跳过性能测试。")
        print("Flash Attention 的性能优势主要体现在 GPU 上。")
    
    # 4. 验证 SDPA 后端
    print("\n[4] PyTorch SDPA 后端信息")
    print("-" * 50)
    
    if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
        print(f"Flash SDP 启用: {torch.backends.cuda.flash_sdp_enabled()}")
    if hasattr(torch.backends.cuda, 'mem_efficient_sdp_enabled'):
        print(f"Memory Efficient SDP 启用: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    if hasattr(torch.backends.cuda, 'math_sdp_enabled'):
        print(f"Math SDP 启用: {torch.backends.cuda.math_sdp_enabled()}")
    
    print("\n" + "=" * 70)
    print("验证完成!")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
