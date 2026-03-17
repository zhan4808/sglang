"""
INT4 BMM Benchmark for MLA Reconstruction
==========================================
Compares FP16 vs INT4 BMM performance for MLA reconstruction operations.
Uses torchao's int4 weight-only quantization for actual INT4 GEMM kernels.

Measures:
  - BMM1: Q absorption (H, BS, d_nope) @ (H, d_nope, kv_lora_rank)
  - BMM2: V reconstruction (H, BS, kv_lora_rank) @ (H, kv_lora_rank, d_v)
"""

import torch
import time
import json
import csv

# DeepSeek-V3 MLA dimensions
H = 128          # num_heads
D_NOPE = 128     # qk_nope_head_dim
KV_LORA = 512    # kv_lora_rank
D_V = 128        # v_head_dim

BATCH_SIZES = [1, 4, 16, 64, 128, 256, 512]
WARMUP = 50
ITERS = 200


def bench_fp16_bmm(H, BS, M, K, warmup=WARMUP, iters=ITERS):
    """Benchmark FP16 BMM: (H, BS, M) @ (H, M, K)"""
    x = torch.randn(H, BS, M, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(H, M, K, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(warmup):
        torch.bmm(x, w)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.bmm(x, w)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median ms


def bench_int4_simulated_bmm(H, BS, M, K, warmup=WARMUP, iters=ITERS):
    """
    Benchmark INT4 weight BMM using per-head linear layers with torchao int4.
    Since torch.bmm doesn't support mixed precision, we use a loop over heads
    with int4 weight-only quantized linear layers.
    """
    try:
        from torchao.quantization import int4_weight_only, quantize_
    except ImportError:
        print("  torchao int4_weight_only not available, using manual simulation")
        return bench_int4_manual_bmm(H, BS, M, K, warmup, iters)

    # Create per-head linear layers and quantize them
    layers = []
    for h in range(H):
        lin = torch.nn.Linear(M, K, bias=False, dtype=torch.bfloat16, device="cuda")
        quantize_(lin, int4_weight_only())
        layers.append(lin)

    x = torch.randn(H, BS, M, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(warmup):
        for h in range(H):
            layers[h](x[h])
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for h in range(H):
            layers[h](x[h])
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def bench_int4_manual_bmm(H, BS, M, K, warmup=WARMUP, iters=ITERS):
    """
    Fallback: simulate INT4 by dequantizing from packed int8 storage.
    This tests the memory bandwidth benefit of smaller weight representation.
    """
    x = torch.randn(H, BS, M, dtype=torch.bfloat16, device="cuda")
    # Pack weights as int8 (two int4 per byte) with scales
    w_packed = torch.randint(0, 255, (H, M, K // 2), dtype=torch.uint8, device="cuda")
    scales = torch.randn(H, 1, K, dtype=torch.bfloat16, device="cuda") * 0.01
    zeros = torch.randn(H, 1, K, dtype=torch.bfloat16, device="cuda") * 0.01

    def dequant_and_bmm():
        # Unpack int4 -> fp16
        lo = (w_packed & 0x0F).to(torch.float16)
        hi = ((w_packed >> 4) & 0x0F).to(torch.float16)
        # Interleave to get full K dimension
        w_fp16 = torch.stack([lo, hi], dim=-1).reshape(H, M, K)
        w_fp16 = w_fp16 * scales + zeros
        return torch.bmm(x, w_fp16)

    # Warmup
    for _ in range(warmup):
        dequant_and_bmm()
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dequant_and_bmm()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def bench_int4_linear_fused(BS_total, M, K, warmup=WARMUP, iters=ITERS):
    """
    Benchmark INT4 as a single large matmul (batch dimension folded into M dim).
    This is more realistic since in practice we'd reshape (H*BS, M) @ (M, K).
    """
    try:
        from torchao.quantization import int4_weight_only, quantize_
    except ImportError:
        return None

    lin = torch.nn.Linear(M, K, bias=False, dtype=torch.bfloat16, device="cuda")
    quantize_(lin, int4_weight_only())

    x = torch.randn(BS_total, M, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(warmup):
        lin(x)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        lin(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def main():
    print("=" * 60)
    print("INT4 BMM Benchmark — MLA Reconstruction")
    print("=" * 60)
    print(f"H={H}, d_nope={D_NOPE}, kv_lora_rank={KV_LORA}, d_v={D_V}")
    print(f"Warmup={WARMUP}, Iters={ITERS}")

    results = []

    for bs in BATCH_SIZES:
        print(f"\n── Batch size = {bs} ──")

        # BMM1: (H, BS, d_nope) @ (H, d_nope, kv_lora_rank)
        fp16_bmm1 = bench_fp16_bmm(H, bs, D_NOPE, KV_LORA)

        # BMM2: (H, BS, kv_lora_rank) @ (H, kv_lora_rank, d_v)
        fp16_bmm2 = bench_fp16_bmm(H, bs, KV_LORA, D_V)

        fp16_total = fp16_bmm1 + fp16_bmm2

        # INT4 as fused linear: reshape to (H*BS, dim) @ (dim, out_dim)
        int4_bmm1 = bench_int4_linear_fused(H * bs, D_NOPE, KV_LORA)
        int4_bmm2 = bench_int4_linear_fused(H * bs, KV_LORA, D_V)

        if int4_bmm1 is not None and int4_bmm2 is not None:
            int4_total = int4_bmm1 + int4_bmm2
            speedup = fp16_total / int4_total
        else:
            int4_total = None
            speedup = None

        print(f"  FP16: BMM1={fp16_bmm1:.4f}ms  BMM2={fp16_bmm2:.4f}ms  Total={fp16_total:.4f}ms")
        if int4_total is not None:
            print(f"  INT4: BMM1={int4_bmm1:.4f}ms  BMM2={int4_bmm2:.4f}ms  Total={int4_total:.4f}ms")
            print(f"  Speedup: {speedup:.2f}x")
        else:
            print(f"  INT4: not available (torchao int4_weight_only missing)")

        results.append({
            "batch_size": bs,
            "fp16_bmm1_ms": fp16_bmm1,
            "fp16_bmm2_ms": fp16_bmm2,
            "fp16_total_ms": fp16_total,
            "int4_bmm1_ms": int4_bmm1,
            "int4_bmm2_ms": int4_bmm2,
            "int4_total_ms": int4_total,
            "speedup": speedup,
        })

    # Save CSV
    csv_path = "/root/sglang/profiling/results_int4_bmm.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'BS':>4}  {'FP16 (ms)':>10}  {'INT4 (ms)':>10}  {'Speedup':>8}")
    print("-" * 40)
    for r in results:
        fp = f"{r['fp16_total_ms']:.4f}"
        i4 = f"{r['int4_total_ms']:.4f}" if r['int4_total_ms'] else "N/A"
        sp = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        print(f"{r['batch_size']:>4}  {fp:>10}  {i4:>10}  {sp:>8}")


if __name__ == "__main__":
    main()
