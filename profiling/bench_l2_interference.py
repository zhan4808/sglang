"""
L2 Cache Interference & Residency Isolation Tests
====================================================
Two experiments that causally isolate L2 cache residency as the driver of
INT4 vs FP16 performance:

Experiment A — L2 Interference:
  Run the original MLA-sized BMM (d_lora=512, 16 MB weights) in two modes:
    1. Isolated: no other GPU work (weights stay L2-resident)
    2. Under L2 pressure: a concurrent stream continuously reads a large
       buffer, evicting reconstruction weights from L2
  If the L2 hypothesis is correct, FP16 should degrade more than INT4 under
  pressure (since FP16 weights are 4x larger and harder to keep L2-resident),
  and the INT4/FP16 ratio should shrink.

Experiment B — Cache Residency A/B:
  Same kernel, same shape (d_lora=512), but vary effective cache footprint:
    Mode A: Reuse a single weight matrix (16 MB, fits L2 after warmup)
    Mode B: Cycle through 4 distinct weight matrices per iteration
            (64 MB total > 50 MB L2, forcing HBM traffic)
  If performance changes with residency and not arithmetic shape, the effect
  is cache-driven, not kernel-quality-driven.
"""

import torch
import triton
import triton.language as tl
import json
import sys
import time


# ═══════════════════════════════════════════════════════════════════════════════
# Triton INT4 kernel (same as bench_l2_barrier.py)
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
# L2 pollution kernel — streams through a large buffer on a separate CUDA stream
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def kernel_l2_pollute(
    buf_ptr, N: tl.constexpr, BLOCK: tl.constexpr,
):
    """Read-modify-write a large buffer to thrash L2 cache."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(buf_ptr + offs, mask=mask, other=0.0)
    # trivial modify to prevent compiler from optimizing away
    tl.store(buf_ptr + offs, x + 0.001, mask=mask)


def launch_l2_pollution(pollute_buf, pollute_stream, n_launches=8):
    """Launch L2-thrashing kernels on a separate stream."""
    N = pollute_buf.numel()
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    with torch.cuda.stream(pollute_stream):
        for _ in range(n_launches):
            kernel_l2_pollute[grid](pollute_buf, N, BLOCK=BLOCK)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark helpers
# ═══════════════════════════════════════════════════════════════════════════════

WARMUP = 50
ITERS = 200
H = 128
D_NOPE = 128


def bench_fp16(x, w, warmup=WARMUP, iters=ITERS,
               pollute_buf=None, pollute_stream=None):
    for _ in range(warmup):
        if pollute_buf is not None:
            launch_l2_pollution(pollute_buf, pollute_stream)
        torch.bmm(x, w)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if pollute_buf is not None:
            launch_l2_pollution(pollute_buf, pollute_stream)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        torch.bmm(x, w)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def bench_int4(x, w_packed, scales, K, warmup=WARMUP, iters=ITERS,
               pollute_buf=None, pollute_stream=None):
    BS = x.shape[1]
    BLOCK_M = max(16, BS)
    BLOCK_M = 1 << (BLOCK_M - 1).bit_length()
    for _ in range(warmup):
        if pollute_buf is not None:
            launch_l2_pollution(pollute_buf, pollute_stream)
        batched_int4_gemm(x, w_packed, scales, K, BLOCK_M=BLOCK_M)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if pollute_buf is not None:
            launch_l2_pollution(pollute_buf, pollute_stream)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        batched_int4_gemm(x, w_packed, scales, K, BLOCK_M=BLOCK_M)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment A: L2 Interference
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_a():
    print("=" * 70)
    print("EXPERIMENT A: L2 Cache Interference Test")
    print("=" * 70)
    print(f"Shape: [{H}, BS, {D_NOPE}] @ [{H}, {D_NOPE}, 512]")
    print(f"FP16 weight size: 16 MB (fits L2)")
    print(f"L2 pollution buffer: 128 MB (2.5x L2 capacity)")
    print()

    d_lora = 512
    results = []

    # Pollution buffer: 128 MB = 64M float16 elements
    pollute_buf = torch.randn(64 * 1024 * 1024, dtype=torch.float16, device="cuda")
    pollute_stream = torch.cuda.Stream()

    # Warmup the pollution kernel (Triton compile)
    launch_l2_pollution(pollute_buf, pollute_stream)
    torch.cuda.synchronize()

    for bs in [1, 4]:
        x = torch.randn(H, bs, D_NOPE, dtype=torch.float16, device="cuda")
        w = torch.randn(H, D_NOPE, d_lora, dtype=torch.float16, device="cuda")
        w_packed, scales = quantize_weights_int4(w)

        # Isolated run
        fp16_iso = bench_fp16(x, w)
        int4_iso = bench_int4(x, w_packed, scales, D_NOPE)
        ratio_iso = int4_iso / fp16_iso

        # Under L2 pressure
        fp16_press = bench_fp16(x, w, pollute_buf=pollute_buf,
                                pollute_stream=pollute_stream)
        int4_press = bench_int4(x, w_packed, scales, D_NOPE,
                                pollute_buf=pollute_buf,
                                pollute_stream=pollute_stream)
        ratio_press = int4_press / fp16_press

        fp16_degrad = fp16_press / fp16_iso
        int4_degrad = int4_press / int4_iso

        print(f"  BS={bs}:")
        print(f"    Isolated:     FP16={fp16_iso:.4f}ms  INT4={int4_iso:.4f}ms  "
              f"ratio={ratio_iso:.2f}x")
        print(f"    L2 pressure:  FP16={fp16_press:.4f}ms  INT4={int4_press:.4f}ms  "
              f"ratio={ratio_press:.2f}x")
        print(f"    FP16 degradation: {fp16_degrad:.2f}x  "
              f"INT4 degradation: {int4_degrad:.2f}x")
        print(f"    Ratio shift: {ratio_iso:.2f}x → {ratio_press:.2f}x "
              f"({'improved' if ratio_press < ratio_iso else 'worsened'})")
        print()

        results.append({
            "experiment": "A_interference",
            "batch_size": bs,
            "d_lora": d_lora,
            "fp16_isolated_ms": round(fp16_iso, 4),
            "int4_isolated_ms": round(int4_iso, 4),
            "ratio_isolated": round(ratio_iso, 3),
            "fp16_pressure_ms": round(fp16_press, 4),
            "int4_pressure_ms": round(int4_press, 4),
            "ratio_pressure": round(ratio_press, 3),
            "fp16_degradation": round(fp16_degrad, 3),
            "int4_degradation": round(int4_degrad, 3),
        })

    del pollute_buf
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment B: Cache Residency A/B Test
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_b():
    print("=" * 70)
    print("EXPERIMENT B: Cache Residency A/B Test")
    print("=" * 70)
    print(f"Shape: [{H}, BS, {D_NOPE}] @ [{H}, {D_NOPE}, 512]")
    print(f"Mode A (single matrix): 16 MB weight, reused every iteration (L2-resident)")
    print(f"Mode B (cycling):       4 distinct matrices = 64 MB total (exceeds L2)")
    print()

    d_lora = 512
    N_MATRICES = 4  # 4 × 16 MB = 64 MB > 50 MB L2
    results = []

    for bs in [1, 4]:
        x = torch.randn(H, bs, D_NOPE, dtype=torch.float16, device="cuda")

        # Create multiple weight matrices
        ws_fp16 = [torch.randn(H, D_NOPE, d_lora, dtype=torch.float16, device="cuda")
                   for _ in range(N_MATRICES)]
        ws_int4 = [quantize_weights_int4(w) for w in ws_fp16]

        BLOCK_M = min(16, max(1, bs))
        BLOCK_M = max(16, BLOCK_M)
        BLOCK_M = 1 << (BLOCK_M - 1).bit_length()

        # ── Mode A: single matrix (L2-resident) ──
        w_single = ws_fp16[0]
        w_packed_single, scales_single = ws_int4[0]

        # Warmup with single matrix
        for _ in range(WARMUP):
            torch.bmm(x, w_single)
        torch.cuda.synchronize()

        times_fp16_single = []
        for _ in range(ITERS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            torch.bmm(x, w_single)
            e.record()
            torch.cuda.synchronize()
            times_fp16_single.append(s.elapsed_time(e))
        times_fp16_single.sort()
        fp16_single = times_fp16_single[len(times_fp16_single) // 2]

        for _ in range(WARMUP):
            batched_int4_gemm(x, w_packed_single, scales_single, D_NOPE,
                              BLOCK_M=BLOCK_M)
        torch.cuda.synchronize()

        times_int4_single = []
        for _ in range(ITERS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            batched_int4_gemm(x, w_packed_single, scales_single, D_NOPE,
                              BLOCK_M=BLOCK_M)
            e.record()
            torch.cuda.synchronize()
            times_int4_single.append(s.elapsed_time(e))
        times_int4_single.sort()
        int4_single = times_int4_single[len(times_int4_single) // 2]

        ratio_single = int4_single / fp16_single

        # ── Mode B: cycling through matrices (L2-busting) ──
        # Warmup cycling
        for i in range(WARMUP):
            idx = i % N_MATRICES
            torch.bmm(x, ws_fp16[idx])
        torch.cuda.synchronize()

        times_fp16_cycle = []
        for i in range(ITERS):
            idx = i % N_MATRICES
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            torch.bmm(x, ws_fp16[idx])
            e.record()
            torch.cuda.synchronize()
            times_fp16_cycle.append(s.elapsed_time(e))
        times_fp16_cycle.sort()
        fp16_cycle = times_fp16_cycle[len(times_fp16_cycle) // 2]

        for i in range(WARMUP):
            idx = i % N_MATRICES
            w_p, sc = ws_int4[idx]
            batched_int4_gemm(x, w_p, sc, D_NOPE, BLOCK_M=BLOCK_M)
        torch.cuda.synchronize()

        times_int4_cycle = []
        for i in range(ITERS):
            idx = i % N_MATRICES
            w_p, sc = ws_int4[idx]
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            batched_int4_gemm(x, w_p, sc, D_NOPE, BLOCK_M=BLOCK_M)
            e.record()
            torch.cuda.synchronize()
            times_int4_cycle.append(s.elapsed_time(e))
        times_int4_cycle.sort()
        int4_cycle = times_int4_cycle[len(times_int4_cycle) // 2]

        ratio_cycle = int4_cycle / fp16_cycle
        fp16_degrad = fp16_cycle / fp16_single
        int4_degrad = int4_cycle / int4_single

        print(f"  BS={bs}:")
        print(f"    Single matrix (L2):  FP16={fp16_single:.4f}ms  INT4={int4_single:.4f}ms  "
              f"ratio={ratio_single:.2f}x")
        print(f"    Cycling (no L2):     FP16={fp16_cycle:.4f}ms  INT4={int4_cycle:.4f}ms  "
              f"ratio={ratio_cycle:.2f}x")
        print(f"    FP16 degradation: {fp16_degrad:.2f}x  "
              f"INT4 degradation: {int4_degrad:.2f}x")
        print(f"    Ratio shift: {ratio_single:.2f}x → {ratio_cycle:.2f}x "
              f"({'improved' if ratio_cycle < ratio_single else 'worsened'})")
        print()

        results.append({
            "experiment": "B_residency",
            "batch_size": bs,
            "d_lora": d_lora,
            "n_matrices": N_MATRICES,
            "total_weight_mb": N_MATRICES * 16,
            "fp16_single_ms": round(fp16_single, 4),
            "int4_single_ms": round(int4_single, 4),
            "ratio_single": round(ratio_single, 3),
            "fp16_cycle_ms": round(fp16_cycle, 4),
            "int4_cycle_ms": round(int4_cycle, 4),
            "ratio_cycle": round(ratio_cycle, 3),
            "fp16_degradation": round(fp16_degrad, 3),
            "int4_degradation": round(int4_degrad, 3),
        })

        # Clean up
        del ws_fp16, ws_int4
        torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print()

    results_a = experiment_a()
    results_b = experiment_b()

    all_results = results_a + results_b

    # Save
    json_path = "results_l2_interference.json"
    with open(json_path, "w") as f:
        json.dump({"gpu": gpu_name, "results": all_results}, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nExperiment A (Interference):")
    for r in results_a:
        print(f"  BS={r['batch_size']}: ratio {r['ratio_isolated']:.2f}x (isolated) → "
              f"{r['ratio_pressure']:.2f}x (L2 pressure)  "
              f"| FP16 degraded {r['fp16_degradation']:.2f}x, "
              f"INT4 degraded {r['int4_degradation']:.2f}x")

    print("\nExperiment B (Residency A/B):")
    for r in results_b:
        print(f"  BS={r['batch_size']}: ratio {r['ratio_single']:.2f}x (single/L2) → "
              f"{r['ratio_cycle']:.2f}x (cycling/no-L2)  "
              f"| FP16 degraded {r['fp16_degradation']:.2f}x, "
              f"INT4 degraded {r['int4_degradation']:.2f}x")

    print("\nInterpretation:")
    print("  If L2 hypothesis holds: FP16 degrades MORE than INT4 in both experiments,")
    print("  and INT4/FP16 ratio improves (drops toward 1.0) under L2 pressure.")


if __name__ == "__main__":
    main()
