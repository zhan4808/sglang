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
import json
import csv
import os
import sys

sys.path.append(os.path.dirname(__file__))
from bench_l2_barrier import quantize_weights_int4, batched_int4_gemm

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
    x = torch.randn(H, BS, M, dtype=torch.float16, device="cuda")
    w = torch.randn(H, M, K, dtype=torch.float16, device="cuda")

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


def bench_int4_triton_bmm(H, BS, M, K, warmup=WARMUP, iters=ITERS):
    """Benchmark Triton INT4 weight-only BMM with on-the-fly dequant."""
    x = torch.randn(H, BS, M, dtype=torch.float16, device="cuda")
    w = torch.randn(H, M, K, dtype=torch.float16, device="cuda")
    w_packed, scales = quantize_weights_int4(w)

    # Keep BLOCK_M small; scaling it with BS causes massive tile/register blowup.
    block_m = 16
    for _ in range(warmup):
        batched_int4_gemm(
            x,
            w_packed,
            scales,
            M,
            BLOCK_M=block_m,
            BLOCK_N=64,
            BLOCK_K=128,
        )
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        batched_int4_gemm(
            x,
            w_packed,
            scales,
            M,
            BLOCK_M=block_m,
            BLOCK_N=64,
            BLOCK_K=128,
        )
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

        int4_bmm1 = bench_int4_triton_bmm(H, bs, D_NOPE, KV_LORA)
        int4_bmm2 = bench_int4_triton_bmm(H, bs, KV_LORA, D_V)
        int4_total = int4_bmm1 + int4_bmm2
        speedup = fp16_total / int4_total

        print(f"  FP16: BMM1={fp16_bmm1:.4f}ms  BMM2={fp16_bmm2:.4f}ms  Total={fp16_total:.4f}ms")
        print(f"  INT4: BMM1={int4_bmm1:.4f}ms  BMM2={int4_bmm2:.4f}ms  Total={int4_total:.4f}ms")
        print(f"  Speedup: {speedup:.2f}x")

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
        i4 = f"{r['int4_total_ms']:.4f}"
        sp = f"{r['speedup']:.2f}x"
        print(f"{r['batch_size']:>4}  {fp:>10}  {i4:>10}  {sp:>8}")


if __name__ == "__main__":
    main()
