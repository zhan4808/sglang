"""
Batched W4A16 GEMM v2 — Contiguous K dequant approach
======================================================
Key change: Pack weights along N dimension (2 adjacent N values per byte)
instead of along K. This keeps A loads contiguous and avoids the even/odd
split that doubled activation bandwidth.

Alternative: N-packed INT4 weights
B_packed[h, k, n//2] = W[h, k, 2n] | (W[h, k, 2n+1] << 4)
Then dequant produces contiguous [K, N] without splitting A.
"""

import torch
import triton
import triton.language as tl
import csv

# ═══════════════════════════════════════════════════════════════════════════════
# Pack along N dimension: each byte holds 2 adjacent N values
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_weights_int4_npacked(W_fp16):
    """
    Per-channel symmetric INT4, packed along N dimension.
    W_fp16: [H, K, N]  (N must be even)
    Returns: packed [H, K, N//2] uint8, scales [H, 1, N] fp16
    """
    assert W_fp16.shape[-1] % 2 == 0, "N must be even"
    w_max = W_fp16.abs().amax(dim=-2, keepdim=True)  # [H, 1, N]
    scale = (w_max / 7.0).clamp(min=1e-8)
    w_q = (W_fp16 / scale).round().clamp(-8, 7).to(torch.int8)

    # Pack pairs along N: even N → lo nibble, odd N → hi nibble
    w_even = w_q[..., 0::2]   # [H, K, N//2]
    w_odd = w_q[..., 1::2]    # [H, K, N//2]
    packed = (w_even & 0x0F).to(torch.uint8) | ((w_odd & 0x0F).to(torch.uint8) << 4)

    return packed, scale.to(torch.float16)  # scale is [H, 1, N]


@triton.jit
def kernel_batched_w4a16_npacked(
    A_ptr, B_ptr, Scale_ptr, C_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_ah, stride_am, stride_ak,
    stride_bh, stride_bk, stride_bn,  # B is [H, K, N//2]
    stride_sh, stride_sn,              # Scale is [H, N]
    stride_ch, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # in output N space (actual N, not packed)
    BLOCK_K: tl.constexpr,
):
    """
    C[h] = A[h] @ dequant(B_packed[h])
    B packed along N: B_packed[h, k, n//2] holds W[h,k,2n] and W[h,k,2n+1]
    Dequant produces [BLOCK_K, BLOCK_N] contiguous tile.
    A is loaded once with contiguous K access.
    """
    pid_h = tl.program_id(0)
    pid_mn = tl.program_id(1)

    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output N indices

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_base = A_ptr + pid_h * stride_ah
    b_base = B_ptr + pid_h * stride_bh

    HALF_BN: tl.constexpr = BLOCK_N // 2

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A [BLOCK_M, BLOCK_K] — contiguous, single load!
        a_ptrs = a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)

        # Load packed B [BLOCK_K, HALF_BN]
        n_packed_start = pid_n * HALF_BN
        offs_np = tl.arange(0, HALF_BN)
        b_ptrs = b_base + offs_k[:, None] * stride_bk + (n_packed_start + offs_np)[None, :] * stride_bn
        b_mask = (offs_k[:, None] < K) & ((n_packed_start + offs_np)[None, :] < (N // 2))
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.uint8)

        # Unpack: lo = even N, hi = odd N
        b_lo = (b_packed & 0x0F)
        b_hi = (b_packed >> 4) & 0x0F
        b_lo = tl.where(b_lo >= 8, b_lo.to(tl.int32) - 16, b_lo.to(tl.int32)).to(tl.float16)
        b_hi = tl.where(b_hi >= 8, b_hi.to(tl.int32) - 16, b_hi.to(tl.int32)).to(tl.float16)

        # Interleave along N: [BLOCK_K, BLOCK_N]
        # We need to produce [k, n] where even n = b_lo, odd n = b_hi
        # But tl.dot requires [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        # We can compute as two matmuls and interleave the result:
        #   acc_even = A @ b_lo  -> [BLOCK_M, HALF_BN] at even positions
        #   acc_odd  = A @ b_hi  -> [BLOCK_M, HALF_BN] at odd positions

        acc_lo = tl.dot(a_tile, b_lo)  # [BLOCK_M, HALF_BN]
        acc_hi = tl.dot(a_tile, b_hi)  # [BLOCK_M, HALF_BN]

        # Accumulate into even/odd positions of acc
        # This is tricky with the interleaved output...
        # Actually, we need to store to C with stride-2 in N
        # OR: just compute two separate outputs

    # Hmm, the interleaving issue moves to output instead of input.
    # This doesn't fundamentally solve the problem.
    # Let me just store even/odd separately.

    # Actually this approach is wrong - we'd need to interleave the output.
    # Let me try yet another approach.
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Approach 3: K-packed but dequant to full tile via register shuffle
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_weights_int4_kpacked(W_fp16):
    """
    Standard K-axis packing.
    W_fp16: [H, K, N]
    Returns: packed [H, K//2, N] uint8, scales [H, N] fp16
    """
    w_max = W_fp16.abs().amax(dim=-2, keepdim=True)
    scale = (w_max / 7.0).clamp(min=1e-8)
    w_q = (W_fp16 / scale).round().clamp(-8, 7).to(torch.int8)
    w_even = w_q[..., 0::2, :]
    w_odd = w_q[..., 1::2, :]
    packed = (w_even & 0x0F).to(torch.uint8) | ((w_odd & 0x0F).to(torch.uint8) << 4)
    return packed, scale.squeeze(-2).to(torch.float16)


@triton.jit
def kernel_batched_w4a16_v3(
    A_ptr, B_ptr, Scale_ptr, C_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_ah, stride_am, stride_ak,
    stride_bh, stride_bk, stride_bn,
    stride_sh, stride_sn,
    stride_ch, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # Must be even. Processes BLOCK_K weight elements per iter.
):
    """
    V3: Load A contiguously for BLOCK_K elements.
    Load B_packed for BLOCK_K//2 packed rows, dequant both halves.
    Compute two matmuls: A_lo @ B_lo + A_hi @ B_hi
    where A_lo, A_hi are the first and second halves of A along K.

    Key insight: instead of even/odd interleaving, we pack CONSECUTIVE
    K values. Byte i holds W[2i] and W[2i+1]. So unpacking gives us
    the first K/2 values (lo) and the second K/2 values (hi) of each block.

    Wait, that's not right either — the standard packing IS consecutive:
    packed[i] = W[2i] | (W[2i+1] << 4)

    So b_lo[j] = W[2j] for j=0..BLOCK_K//2-1 (first halves)
    and b_hi[j] = W[2j+1] for j=0..BLOCK_K//2-1 (second halves)

    These correspond to A columns 0,2,4,... and 1,3,5,...

    But we can reorganize: A[:, 0:BLOCK_K:2] @ B_lo + A[:, 1:BLOCK_K:2] @ B_hi
    The A loads are stride-2, which is suboptimal. BUT we only load A once
    as a contiguous block and then split in registers.
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
        # Load A as contiguous [BLOCK_M, BLOCK_K] block
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_full = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)

        # Split A in registers: even and odd columns
        # a_full[:, 0], a_full[:, 2], a_full[:, 4], ... → a_even
        # a_full[:, 1], a_full[:, 3], a_full[:, 5], ... → a_odd
        # Use tl.reshape or indexing
        # Actually, we can just load the two halves separately since stride_ak=1
        offs_k_even = k_start + tl.arange(0, HALF_BK) * 2
        offs_k_odd = k_start + tl.arange(0, HALF_BK) * 2 + 1

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

        # Load packed B [HALF_BK, BLOCK_N]
        offs_kp = (k_start // 2) + tl.arange(0, HALF_BK)
        b_ptrs = b_base + offs_kp[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_kp[:, None] < (K // 2)) & (offs_n[None, :] < N)
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.uint8)

        b_lo = (b_packed & 0x0F)
        b_hi = (b_packed >> 4) & 0x0F
        b_lo = tl.where(b_lo >= 8, b_lo.to(tl.int32) - 16, b_lo.to(tl.int32)).to(tl.float16)
        b_hi = tl.where(b_hi >= 8, b_hi.to(tl.int32) - 16, b_hi.to(tl.int32)).to(tl.float16)

        acc += tl.dot(a_even, b_lo).to(tl.float32)
        acc += tl.dot(a_odd, b_hi).to(tl.float32)

    # Scale
    scale_ptrs = Scale_ptr + pid_h * stride_sh + offs_n * stride_sn
    scales = tl.load(scale_ptrs, mask=offs_n < N, other=1.0).to(tl.float32)
    acc *= scales[None, :]

    c_ptrs = C_ptr + pid_h * stride_ch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def batched_int4_gemm(A, B_packed, scales, K, BLOCK_M=16, BLOCK_N=64, BLOCK_K=128):
    H, M, _ = A.shape
    _, _, N = B_packed.shape
    BLOCK_M = min(BLOCK_M, max(1, M))
    if BLOCK_M > 1:
        BLOCK_M = 1 << (BLOCK_M - 1).bit_length()
    BLOCK_N = min(BLOCK_N, N)
    BLOCK_K = min(BLOCK_K, K)
    BLOCK_K = max(2, BLOCK_K)

    C = torch.empty((H, M, N), device=A.device, dtype=torch.float16)
    grid = (H, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))
    kernel_batched_w4a16_v3[grid](
        A, B_packed, scales, C,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B_packed.stride(0), B_packed.stride(1), B_packed.stride(2),
        scales.stride(0), scales.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark comparing: FP16 BMM vs INT4 Triton vs FP16 torch.matmul (reshaped)
# ═══════════════════════════════════════════════════════════════════════════════

H, D_NOPE, KV_LORA, D_V = 128, 128, 512, 128
NUM_LAYERS = 61
BATCH_SIZES = [1, 4, 16, 64, 128, 256, 512]
WARMUP, ITERS = 50, 200


def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def bench_fp16_bmm(H, BS, M, K):
    x = torch.randn(H, BS, M, dtype=torch.float16, device="cuda")
    w = torch.randn(H, M, K, dtype=torch.float16, device="cuda")
    return bench(lambda: torch.bmm(x, w))


def bench_int4_triton(H, BS, M, K, BLOCK_M=16, BLOCK_N=64, BLOCK_K=128):
    x = torch.randn(H, BS, M, dtype=torch.float16, device="cuda")
    w = torch.randn(H, M, K, dtype=torch.float16, device="cuda")
    wp, sc = quantize_weights_int4_kpacked(w)
    return bench(lambda: batched_int4_gemm(x, wp, sc, M, BLOCK_M, BLOCK_N, BLOCK_K))


def bench_fp16_loop(H, BS, M, K):
    """Benchmark per-head FP16 matmul loop (what torchao would do)"""
    xs = [torch.randn(BS, M, dtype=torch.float16, device="cuda") for _ in range(H)]
    ws = [torch.randn(M, K, dtype=torch.float16, device="cuda") for _ in range(H)]
    def run():
        for h in range(H):
            torch.mm(xs[h], ws[h])
    return bench(run)


def main():
    print("=" * 70)
    print("Batched W4A16 GEMM v2 — MLA Reconstruction Benchmark")
    print("=" * 70)
    print(f"H={H}, d_nope={D_NOPE}, kv_lora_rank={KV_LORA}, d_v={D_V}")
    print()

    # Verify
    print("Correctness check...")
    x = torch.randn(4, 2, 128, dtype=torch.float16, device="cuda")
    w = torch.randn(4, 128, 512, dtype=torch.float16, device="cuda")
    wp, sc = quantize_weights_int4_kpacked(w)
    out = batched_int4_gemm(x, wp, sc, 128)
    w_max = w.abs().amax(dim=-2, keepdim=True)
    w_sc = (w_max / 7.0).clamp(min=1e-8)
    w_deq = (w / w_sc).round().clamp(-8, 7) * w_sc
    ref = torch.bmm(x, w_deq)
    print(f"  Max diff vs INT4 ref: {(out.float() - ref.float()).abs().max():.4f}")

    print()
    results = []

    for bs in BATCH_SIZES:
        print(f"── BS = {bs} ──")

        # BMM1 and BMM2 for FP16
        fp16_1 = bench_fp16_bmm(H, bs, D_NOPE, KV_LORA)
        fp16_2 = bench_fp16_bmm(H, bs, KV_LORA, D_V)
        fp16_tot = fp16_1 + fp16_2

        # BMM1 and BMM2 for INT4
        bm = min(16, max(1, bs))
        if bm > 1:
            bm = 1 << (bm - 1).bit_length()
        int4_1 = bench_int4_triton(H, bs, D_NOPE, KV_LORA, bm, 64, 128)
        int4_2 = bench_int4_triton(H, bs, KV_LORA, D_V, bm, 64, 128)
        int4_tot = int4_1 + int4_2

        # Per-head loop (to show batched kernel overhead)
        loop_1 = bench_fp16_loop(H, bs, D_NOPE, KV_LORA)
        loop_2 = bench_fp16_loop(H, bs, KV_LORA, D_V)
        loop_tot = loop_1 + loop_2

        speedup = fp16_tot / int4_tot
        print(f"  FP16 bmm:   {fp16_tot:.4f} ms")
        print(f"  INT4 batch: {int4_tot:.4f} ms  ({speedup:.2f}x)")
        print(f"  FP16 loop:  {loop_tot:.4f} ms  ({fp16_tot/loop_tot:.2f}x vs bmm)")
        print()

        results.append({
            "bs": bs, "fp16_ms": fp16_tot, "int4_ms": int4_tot,
            "loop_ms": loop_tot, "speedup": speedup,
            "fp16_model_ms": fp16_tot * NUM_LAYERS,
            "int4_model_ms": int4_tot * NUM_LAYERS,
        })

    csv_path = "/root/sglang/profiling/results_int4_batched_v2.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'BS':>4}  {'FP16 bmm':>10}  {'INT4 batch':>10}  {'FP16 loop':>10}  {'INT4/FP16':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['bs']:>4}  {r['fp16_ms']:>10.4f}  {r['int4_ms']:>10.4f}  "
              f"{r['loop_ms']:>10.4f}  {r['speedup']:>9.2f}x")


if __name__ == "__main__":
    main()
