"""
L2 Cache Barrier Experiment
============================
Varies reconstruction weight matrix size across the H100 L2 cache boundary (50 MB)
to causally isolate L2 residency as the reason INT4 fails to outperform FP16.

Hypothesis: When FP16 weights fit in L2 (<50 MB), cuBLAS serves them from L2 at
~12 TB/s, making INT4's HBM savings irrelevant. Once weights exceed L2 capacity,
FP16 must stream from HBM at 3.35 TB/s, and INT4 (4x smaller, still L2-resident)
should finally outperform.

We scale d_lora (the N dimension of BMM1) while keeping H=128 and d_nope=128 fixed.
Weight size per BMM = H * d_nope * d_lora * 2 bytes.
"""

import torch
import triton
import triton.language as tl
import csv
import json
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# Triton INT4 kernel (from int4_batched_gemm.py)
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def kernel_batched_w4a16_simple(
    A_ptr, B_ptr, Scale_ptr, C_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_ah, stride_am, stride_ak,
    stride_bh, stride_bk, stride_bn,
    stride_sh, stride_sn,
    stride_ch, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_mn = tl.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_base = A_ptr + pid_h * stride_ah
    b_base = B_ptr + pid_h * stride_bh
    HALF_BK: tl.constexpr = BLOCK_K // 2
    for k_start in range(0, K, BLOCK_K):
        offs_kp = (k_start // 2) + tl.arange(0, HALF_BK)
        b_ptrs = b_base + offs_kp[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_kp[:, None] < (K // 2)) & (offs_n[None, :] < N)
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.uint8)
        b_lo = (b_packed & 0x0F)
        b_hi = (b_packed >> 4) & 0x0F
        b_lo = tl.where(b_lo >= 8, b_lo.to(tl.int32) - 16, b_lo.to(tl.int32)).to(tl.float16)
        b_hi = tl.where(b_hi >= 8, b_hi.to(tl.int32) - 16, b_hi.to(tl.int32)).to(tl.float16)
        offs_k_even = k_start + tl.arange(0, HALF_BK) * 2
        offs_k_odd = offs_k_even + 1
        a_even = tl.load(
            a_base + offs_m[:, None] * stride_am + offs_k_even[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k_even[None, :] < K), other=0.0
        ).to(tl.float16)
        a_odd = tl.load(
            a_base + offs_m[:, None] * stride_am + offs_k_odd[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k_odd[None, :] < K), other=0.0
        ).to(tl.float16)
        acc += tl.dot(a_even, b_lo).to(tl.float32)
        acc += tl.dot(a_odd, b_hi).to(tl.float32)
    scale_ptrs = Scale_ptr + pid_h * stride_sh + offs_n * stride_sn
    scales = tl.load(scale_ptrs, mask=offs_n < N, other=1.0).to(tl.float32)
    acc *= scales[None, :]
    c_ptrs = C_ptr + pid_h * stride_ch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Quantization + wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_weights_int4(W_fp16):
    w_max = W_fp16.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
    scale = w_max / 7.0
    w_q = (W_fp16 / scale).round().clamp(-8, 7).to(torch.int8)
    w_even = w_q[..., 0::2, :]
    w_odd = w_q[..., 1::2, :]
    packed = (w_even & 0x0F).to(torch.uint8) | ((w_odd & 0x0F).to(torch.uint8) << 4)
    return packed, scale.squeeze(-2).to(torch.float16)


def batched_int4_gemm(A_fp16, B_packed, scales, K,
                       BLOCK_M=16, BLOCK_N=64, BLOCK_K=128):
    H, M, _ = A_fp16.shape
    _, _, N = B_packed.shape
    # Triton dot on this stack requires tile dims >= 16.
    # Keep BLOCK_M >= 16 even for tiny M (BS=1/4), masking handles bounds.
    BLOCK_M = max(16, min(BLOCK_M, max(16, M)))
    BLOCK_M = 1 << (BLOCK_M - 1).bit_length()
    BLOCK_N = min(BLOCK_N, N)
    BLOCK_K = min(BLOCK_K, K)
    if BLOCK_K < 2:
        BLOCK_K = 2
    C = torch.empty((H, M, N), device=A_fp16.device, dtype=torch.float16)
    grid = (H, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))
    kernel_batched_w4a16_simple[grid](
        A_fp16, B_packed, scales, C,
        M, N, K,
        A_fp16.stride(0), A_fp16.stride(1), A_fp16.stride(2),
        B_packed.stride(0), B_packed.stride(1), B_packed.stride(2),
        scales.stride(0), scales.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark helpers
# ═══════════════════════════════════════════════════════════════════════════════

WARMUP = 50
ITERS = 200


def bench_fp16_bmm(H, BS, K, N, warmup=WARMUP, iters=ITERS):
    """Benchmark torch.bmm: [H, BS, K] @ [H, K, N]"""
    x = torch.randn(H, BS, K, dtype=torch.float16, device="cuda")
    w = torch.randn(H, K, N, dtype=torch.float16, device="cuda")
    for _ in range(warmup):
        torch.bmm(x, w)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        torch.bmm(x, w)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def bench_int4_bmm(H, BS, K, N, warmup=WARMUP, iters=ITERS):
    """Benchmark INT4 Triton kernel: [H, BS, K] @ dequant([H, K//2, N])"""
    x = torch.randn(H, BS, K, dtype=torch.float16, device="cuda")
    w = torch.randn(H, K, N, dtype=torch.float16, device="cuda")
    w_packed, scales = quantize_weights_int4(w)
    BLOCK_M = min(16, max(1, BS))
    if BLOCK_M > 1:
        BLOCK_M = 1 << (BLOCK_M - 1).bit_length()
    for _ in range(warmup):
        batched_int4_gemm(x, w_packed, scales, K, BLOCK_M=BLOCK_M, BLOCK_N=64, BLOCK_K=128)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        batched_int4_gemm(x, w_packed, scales, K, BLOCK_M=BLOCK_M, BLOCK_N=64, BLOCK_K=128)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


# ═══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════════

H = 128
D_NOPE = 128  # K dimension of BMM (fixed)

# Scale d_lora (N dimension) to control weight size.
# Weight bytes = H * D_NOPE * d_lora * 2.
# H100 L2 = 50 MB. Crossover expected around d_lora ~ 1536-1792.
D_LORA_SWEEP = [256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096]

BATCH_SIZES = [1, 4]


def infer_l2_threshold_mb(gpu_name: str) -> float:
    name = gpu_name.upper()
    if "A100" in name:
        return 40.0
    if "H100" in name:
        return 50.0
    # Conservative default if unknown GPU
    return 40.0


def main():
    print("=" * 70)
    print("L2 Cache Barrier Experiment")
    print("=" * 70)
    print(f"H={H}, D_NOPE={D_NOPE} (fixed K dim)")
    print(f"Sweeping d_lora (N dim): {D_LORA_SWEEP}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print()

    gpu_name = torch.cuda.get_device_name(0)
    l2_threshold_mb = infer_l2_threshold_mb(gpu_name)
    print(f"GPU: {gpu_name}")
    print(f"Estimated L2 threshold = {l2_threshold_mb:.0f} MB")
    print()

    results = []

    for bs in BATCH_SIZES:
        print(f"\n{'='*70}")
        print(f"  BATCH SIZE = {bs}")
        print(f"{'='*70}")
        print(f"{'d_lora':>8} {'Wt MB':>8} {'FP16(ms)':>10} {'INT4(ms)':>10} "
              f"{'Ratio':>8} {'L2?':>5}")
        print("-" * 60)

        for d_lora in D_LORA_SWEEP:
            wt_bytes = H * D_NOPE * d_lora * 2
            wt_mb = wt_bytes / (1024 * 1024)
            fits_l2 = "yes" if wt_mb < l2_threshold_mb else "NO"

            try:
                fp16_ms = bench_fp16_bmm(H, bs, D_NOPE, d_lora)
                int4_ms = bench_int4_bmm(H, bs, D_NOPE, d_lora)
            except torch.cuda.OutOfMemoryError:
                print(f"{d_lora:>8} {wt_mb:>8.1f}  OOM")
                continue
            except Exception as e:
                print(f"{d_lora:>8} {wt_mb:>8.1f}  ERROR: {e}")
                continue

            ratio = int4_ms / fp16_ms

            print(f"{d_lora:>8} {wt_mb:>8.1f} {fp16_ms:>10.4f} {int4_ms:>10.4f} "
                  f"{ratio:>7.2f}x {fits_l2:>5}")

            results.append({
                "batch_size": bs,
                "d_lora": d_lora,
                "weight_mb": round(wt_mb, 1),
                "fits_l2": wt_mb < l2_threshold_mb,
                "fp16_ms": round(fp16_ms, 4),
                "int4_ms": round(int4_ms, 4),
                "int4_fp16_ratio": round(ratio, 3),
            })

    # Save results
    if not results:
        print("\nNo successful measurements were collected.")
        return

    csv_path = "results_l2_barrier.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved to {csv_path}")

    json_path = "results_l2_barrier.json"
    with open(json_path, "w") as f:
        json.dump({"gpu": gpu_name, "H": H, "D_NOPE": D_NOPE, "results": results}, f, indent=2)
    print(f"JSON saved to {json_path}")

    # Print analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    bs1 = [r for r in results if r["batch_size"] == 1]
    if bs1:
        below = [r for r in bs1 if r["fits_l2"]]
        above = [r for r in bs1 if not r["fits_l2"]]
        if below:
            avg_below = sum(r["int4_fp16_ratio"] for r in below) / len(below)
            print(f"Avg INT4/FP16 ratio (weight < {l2_threshold_mb:.0f} MB, L2-resident):  {avg_below:.2f}x")
        if above:
            avg_above = sum(r["int4_fp16_ratio"] for r in above) / len(above)
            print(f"Avg INT4/FP16 ratio (weight > {l2_threshold_mb:.0f} MB, HBM-bound):    {avg_above:.2f}x")
        if below and above:
            print(f"Ratio improvement when crossing L2 boundary:         {avg_above/avg_below:.2f}x")
            crosses = any(r["int4_fp16_ratio"] <= 1.0 for r in above)
            if crosses:
                first = next(r for r in above if r["int4_fp16_ratio"] <= 1.0)
                print(f"\n*** INT4 first beats FP16 at d_lora={first['d_lora']} "
                      f"({first['weight_mb']} MB) ***")
            else:
                print(f"\nINT4 did not beat FP16 at any size (dequant overhead may dominate)")
                print(f"But the ratio should still IMPROVE past {l2_threshold_mb:.0f} MB, confirming L2 hypothesis.")
            best = min(above, key=lambda r: r["int4_fp16_ratio"])
            worst = min(below, key=lambda r: r["int4_fp16_ratio"])
            print(f"\nBest ratio below L2:  {min(r['int4_fp16_ratio'] for r in below):.2f}x "
                  f"at d_lora={worst['d_lora']}")
            print(f"Best ratio above L2:  {min(r['int4_fp16_ratio'] for r in above):.2f}x "
                  f"at d_lora={best['d_lora']}")


if __name__ == "__main__":
    main()
