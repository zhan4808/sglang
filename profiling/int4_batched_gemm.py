"""
Batched Weight-Only INT4 GEMM for MLA Reconstruction
=====================================================
Custom Triton kernel: FP16 activations × INT4 weights → FP16 output
Dequantizes INT4 weights to FP16 on-the-fly, then uses FP16 tensor cores.

Key insight: For memory-bound operations, the speedup comes from reading
4x fewer weight bytes from HBM, NOT from cheaper INT4 compute.
The dequant is done in registers/shared memory (free bandwidth-wise).
"""

import torch
import triton
import triton.language as tl
import time
import csv

# ═══════════════════════════════════════════════════════════════════════════════
# Triton Kernel: Batched Weight-Only INT4 GEMM (FP16 tensor core path)
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def kernel_batched_w4a16_gemm(
    # Pointers
    A_ptr,       # [H, M, K] float16 activations
    B_ptr,       # [H, K//2, N] uint8 packed INT4 weights
    Scale_ptr,   # [H, 1, N] float16 per-channel scales
    C_ptr,       # [H, M, N] float16 output
    # Dimensions
    M, N: tl.constexpr, K: tl.constexpr,
    # Strides for A [H, M, K]
    stride_ah, stride_am, stride_ak,
    # Strides for B [H, K//2, N]
    stride_bh, stride_bk, stride_bn,
    # Strides for Scale [H, N]
    stride_sh, stride_sn,
    # Strides for C [H, M, N]
    stride_ch, stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    C[h] = A[h] @ dequant(B_packed[h])
    Dequantizes INT4→FP16 on the fly, uses FP16 tensor cores for matmul.
    """
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

    # BLOCK_K must be even for INT4 packing
    HALF_BK: tl.constexpr = BLOCK_K // 2

    for k_start in range(0, K, BLOCK_K):
        # ── Load A tile [BLOCK_M, BLOCK_K] as FP16 ──
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)

        # ── Load B tile [BLOCK_K//2, BLOCK_N] as packed uint8 ──
        offs_kp = (k_start // 2) + tl.arange(0, HALF_BK)
        b_ptrs = b_base + offs_kp[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_kp[:, None] < (K // 2)) & (offs_n[None, :] < N)
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.uint8)

        # ── Dequant INT4 → FP16 ──
        # Low nibble = even k, high nibble = odd k
        b_lo = (b_packed & 0x0F).to(tl.int8)
        b_hi = ((b_packed >> 4) & 0x0F).to(tl.int8)
        # Signed: 8..15 → -8..-1
        b_lo = tl.where(b_lo >= 8, b_lo - 16, b_lo)
        b_hi = tl.where(b_hi >= 8, b_hi - 16, b_hi)

        # Interleave back to [BLOCK_K, BLOCK_N]
        # Create full weight tile by stacking even/odd rows
        # b_lo is [HALF_BK, BLOCK_N] at positions 0,2,4,...
        # b_hi is [HALF_BK, BLOCK_N] at positions 1,3,5,...
        b_lo_fp = b_lo.to(tl.float16)
        b_hi_fp = b_hi.to(tl.float16)

        # Split A into even and odd columns, do two half-sized matmuls
        a_even = tl.load(
            a_base + offs_m[:, None] * stride_am +
            (k_start + tl.arange(0, HALF_BK) * 2)[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) &
                 ((k_start + tl.arange(0, HALF_BK) * 2)[None, :] < K),
            other=0.0
        ).to(tl.float16)

        a_odd = tl.load(
            a_base + offs_m[:, None] * stride_am +
            (k_start + tl.arange(0, HALF_BK) * 2 + 1)[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) &
                 ((k_start + tl.arange(0, HALF_BK) * 2 + 1)[None, :] < K),
            other=0.0
        ).to(tl.float16)

        # FP16 tensor core matmul
        acc += tl.dot(a_even, b_lo_fp).to(tl.float32)
        acc += tl.dot(a_odd, b_hi_fp).to(tl.float32)

    # Apply per-channel scale
    scale_ptrs = Scale_ptr + pid_h * stride_sh + offs_n * stride_sn
    scales = tl.load(scale_ptrs, mask=offs_n < N, other=1.0).to(tl.float32)
    acc = acc * scales[None, :]

    # Store
    c_ptrs = C_ptr + pid_h * stride_ch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Simpler approach: just load fewer bytes, dequant, standard matmul
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
    """
    Simplified kernel: load INT4 packed weights, dequant to FP16 in shared mem,
    then standard FP16 matmul via tl.dot.

    Weight packing: Along K axis, pairs of consecutive elements packed into uint8.
    B_packed shape: [H, K//2, N], where B_packed[h, k, n] = (W[h, 2k, n] & 0xF) | (W[h, 2k+1, n] << 4)
    """
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
        # Load packed B: [HALF_BK, BLOCK_N]
        offs_kp = (k_start // 2) + tl.arange(0, HALF_BK)
        b_ptrs = b_base + offs_kp[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_kp[:, None] < (K // 2)) & (offs_n[None, :] < N)
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.uint8)

        # Unpack to two FP16 tiles
        b_lo = (b_packed & 0x0F)
        b_hi = (b_packed >> 4) & 0x0F
        b_lo = tl.where(b_lo >= 8, b_lo.to(tl.int32) - 16, b_lo.to(tl.int32)).to(tl.float16)
        b_hi = tl.where(b_hi >= 8, b_hi.to(tl.int32) - 16, b_hi.to(tl.int32)).to(tl.float16)

        # Load A even/odd columns
        offs_k_even = k_start + tl.arange(0, HALF_BK) * 2
        offs_k_odd = offs_k_even + 1

        a_even = tl.load(
            a_base + offs_m[:, None] * stride_am + offs_k_even[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k_even[None, :] < K),
            other=0.0
        ).to(tl.float16)

        a_odd = tl.load(
            a_base + offs_m[:, None] * stride_am + offs_k_odd[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k_odd[None, :] < K),
            other=0.0
        ).to(tl.float16)

        acc += tl.dot(a_even, b_lo).to(tl.float32)
        acc += tl.dot(a_odd, b_hi).to(tl.float32)

    # Scale
    scale_ptrs = Scale_ptr + pid_h * stride_sh + offs_n * stride_sn
    scales = tl.load(scale_ptrs, mask=offs_n < N, other=1.0).to(tl.float32)
    acc *= scales[None, :]

    c_ptrs = C_ptr + pid_h * stride_ch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Python Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_weights_int4(W_fp16):
    """
    Per-channel symmetric INT4 quantization along K axis.
    W_fp16: [H, K, N]
    Returns: packed [H, K//2, N] uint8, scales [H, N] fp16
    """
    w_max = W_fp16.abs().amax(dim=-2, keepdim=True)
    scale = w_max / 7.0
    scale = scale.clamp(min=1e-8)

    w_q = (W_fp16 / scale).round().clamp(-8, 7).to(torch.int8)

    # Pack along K: even→lo nibble, odd→hi nibble
    w_even = w_q[..., 0::2, :]
    w_odd = w_q[..., 1::2, :]
    packed = (w_even & 0x0F).to(torch.uint8) | ((w_odd & 0x0F).to(torch.uint8) << 4)

    return packed, scale.squeeze(-2).to(torch.float16)


def batched_int4_gemm(A_fp16, B_packed, scales, K,
                       BLOCK_M=16, BLOCK_N=64, BLOCK_K=128):
    """
    A_fp16: [H, M, K] float16
    B_packed: [H, K//2, N] uint8
    scales: [H, N] float16
    """
    H, M, _ = A_fp16.shape
    _, _, N = B_packed.shape

    # Clamp block sizes
    BLOCK_M = min(BLOCK_M, max(1, M))
    # Round BLOCK_M to power of 2 for tl.dot compatibility
    if BLOCK_M > 1:
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
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

H = 128
D_NOPE = 128
KV_LORA = 512
D_V = 128
NUM_LAYERS = 61

BATCH_SIZES = [1, 4, 16, 64, 128, 256, 512]
WARMUP = 50
ITERS = 200


def bench_fp16_bmm(H, BS, M, K, warmup=WARMUP, iters=ITERS):
    x = torch.randn(H, BS, M, dtype=torch.float16, device="cuda")
    w = torch.randn(H, M, K, dtype=torch.float16, device="cuda")

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


def bench_int4(H, BS, M, K, warmup=WARMUP, iters=ITERS,
               BLOCK_M=16, BLOCK_N=64, BLOCK_K=128):
    x = torch.randn(H, BS, M, dtype=torch.float16, device="cuda")
    w = torch.randn(H, M, K, dtype=torch.float16, device="cuda")
    w_packed, scales = quantize_weights_int4(w)

    for _ in range(warmup):
        batched_int4_gemm(x, w_packed, scales, M, BLOCK_M, BLOCK_N, BLOCK_K)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        batched_int4_gemm(x, w_packed, scales, M, BLOCK_M, BLOCK_N, BLOCK_K)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def verify():
    print("Verifying correctness...")
    H_t, BS_t, M_t, K_t = 4, 2, 128, 512
    A = torch.randn(H_t, BS_t, M_t, dtype=torch.float16, device="cuda")
    W = torch.randn(H_t, M_t, K_t, dtype=torch.float16, device="cuda")

    ref = torch.bmm(A, W)
    w_packed, scales = quantize_weights_int4(W)
    out = batched_int4_gemm(A, w_packed, scales, M_t)

    # Expected INT4 output
    w_max = W.abs().amax(dim=-2, keepdim=True)
    w_sc = (w_max / 7.0).clamp(min=1e-8)
    w_q = (W / w_sc).round().clamp(-8, 7)
    w_deq = w_q * w_sc
    ref_int4 = torch.bmm(A, w_deq)

    d1 = (out.float() - ref_int4.float()).abs().max().item()
    d2 = (out.float() - ref.float()).abs().max().item()
    print(f"  vs INT4 reference: {d1:.4f}")
    print(f"  vs FP16 reference: {d2:.4f}")
    ok = d1 < 5.0
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def autotune_blocks(H, BS, M, K):
    """Quick autotune over block sizes"""
    best_t = float('inf')
    best_cfg = None
    configs = [
        (1, 64, 128), (1, 128, 128), (1, 64, 64),
        (2, 64, 128), (4, 64, 128), (4, 128, 128),
        (8, 64, 128), (8, 128, 128), (8, 64, 64),
        (16, 64, 128), (16, 128, 128), (16, 64, 64),
        (16, 128, 64),
    ]
    for bm, bn, bk in configs:
        if bm > BS:
            continue
        if bn > K and bn > N:
            continue
        try:
            t = bench_int4(H, BS, M, K, warmup=10, iters=30,
                          BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
            if t < best_t:
                best_t = t
                best_cfg = (bm, bn, bk)
        except Exception:
            continue
    return best_cfg, best_t


def main():
    print("=" * 60)
    print("Batched W4A16 GEMM Benchmark — MLA Reconstruction")
    print("=" * 60)
    print(f"H={H}, d_nope={D_NOPE}, kv_lora_rank={KV_LORA}, d_v={D_V}")
    print(f"Warmup={WARMUP}, Iters={ITERS}")
    print()

    if not verify():
        return

    # Quick autotune for bs=1
    print("\nAutotuning block sizes for bs=1 BMM1 (128→512)...")
    cfg1, _ = autotune_blocks(H, 1, D_NOPE, KV_LORA)
    print(f"  Best config BMM1: BLOCK_M={cfg1[0]}, BLOCK_N={cfg1[1]}, BLOCK_K={cfg1[2]}")

    print("Autotuning block sizes for bs=1 BMM2 (512→128)...")
    cfg2, _ = autotune_blocks(H, 1, KV_LORA, D_V)
    print(f"  Best config BMM2: BLOCK_M={cfg2[0]}, BLOCK_N={cfg2[1]}, BLOCK_K={cfg2[2]}")

    print()
    results = []

    for bs in BATCH_SIZES:
        print(f"── Batch size = {bs} ──")

        fp16_bmm1 = bench_fp16_bmm(H, bs, D_NOPE, KV_LORA)
        fp16_bmm2 = bench_fp16_bmm(H, bs, KV_LORA, D_V)

        # Use autotuned config or reasonable defaults
        bm = min(16, max(1, bs))
        if bm > 1:
            bm = 1 << (bm - 1).bit_length()
        int4_bmm1 = bench_int4(H, bs, D_NOPE, KV_LORA, BLOCK_M=bm, BLOCK_N=64, BLOCK_K=128)
        int4_bmm2 = bench_int4(H, bs, KV_LORA, D_V, BLOCK_M=bm, BLOCK_N=64, BLOCK_K=128)

        fp16_total = fp16_bmm1 + fp16_bmm2
        int4_total = int4_bmm1 + int4_bmm2
        speedup = fp16_total / int4_total

        # Full model cost (× NUM_LAYERS)
        fp16_model = fp16_total * NUM_LAYERS
        int4_model = int4_total * NUM_LAYERS

        print(f"  FP16: BMM1={fp16_bmm1:.4f}ms  BMM2={fp16_bmm2:.4f}ms  Total={fp16_total:.4f}ms")
        print(f"  INT4: BMM1={int4_bmm1:.4f}ms  BMM2={int4_bmm2:.4f}ms  Total={int4_total:.4f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Full model: FP16={fp16_model:.2f}ms  INT4={int4_model:.2f}ms")
        print()

        results.append({
            "batch_size": bs,
            "fp16_bmm1_ms": fp16_bmm1,
            "fp16_bmm2_ms": fp16_bmm2,
            "fp16_total_ms": fp16_total,
            "int4_bmm1_ms": int4_bmm1,
            "int4_bmm2_ms": int4_bmm2,
            "int4_total_ms": int4_total,
            "speedup": speedup,
            "fp16_model_ms": fp16_model,
            "int4_model_ms": int4_model,
        })

    csv_path = "/root/sglang/profiling/results_int4_batched_gemm.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'BS':>4}  {'FP16 (ms)':>10}  {'INT4 (ms)':>10}  {'Speedup':>8}  {'FP16 model':>12}  {'INT4 model':>12}")
    print("-" * 64)
    for r in results:
        print(f"{r['batch_size']:>4}  {r['fp16_total_ms']:>10.4f}  {r['int4_total_ms']:>10.4f}  "
              f"{r['speedup']:>7.2f}x  {r['fp16_model_ms']:>11.2f}  {r['int4_model_ms']:>11.2f}")


if __name__ == "__main__":
    main()
