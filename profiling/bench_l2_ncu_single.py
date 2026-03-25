"""
Single-point benchmark for NCU profiling.
Usage: python3 bench_l2_ncu_single.py <kernel_type> <d_lora>
  kernel_type: fp16 or int4
  d_lora: latent dimension (controls weight size)

Runs a few warmup iterations then a small number of measured iterations
for NCU to capture.
"""

import sys
import torch
import triton
import triton.language as tl

H = 128
D_NOPE = 128
BS = 1


# ── Triton INT4 kernel ──

@triton.jit
def kernel_batched_w4a16_simple(
    A_ptr, B_ptr, Scale_ptr, C_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_ah, stride_am, stride_ak,
    stride_bh, stride_bk, stride_bn,
    stride_sh, stride_sn,
    stride_ch, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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


def quantize_weights_int4(W_fp16):
    w_max = W_fp16.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
    scale = w_max / 7.0
    w_q = (W_fp16 / scale).round().clamp(-8, 7).to(torch.int8)
    w_even = w_q[..., 0::2, :]
    w_odd = w_q[..., 1::2, :]
    packed = (w_even & 0x0F).to(torch.uint8) | ((w_odd & 0x0F).to(torch.uint8) << 4)
    return packed, scale.squeeze(-2).to(torch.float16)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <fp16|int4> <d_lora>")
        sys.exit(1)

    kernel_type = sys.argv[1]
    d_lora = int(sys.argv[2])

    x = torch.randn(H, BS, D_NOPE, dtype=torch.float16, device="cuda")
    w = torch.randn(H, D_NOPE, d_lora, dtype=torch.float16, device="cuda")

    if kernel_type == "fp16":
        # Warmup (outside NCU profiling window)
        for _ in range(5):
            torch.bmm(x, w)
        torch.cuda.synchronize()

        # Profiled region — use profiler start/stop range
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(3):
            torch.bmm(x, w)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    elif kernel_type == "int4":
        w_packed, scales = quantize_weights_int4(w)
        # Triton dot requires tile dims >= 16 on this stack.
        BLOCK_M = 16
        BLOCK_N = min(64, d_lora)
        BLOCK_K = min(128, D_NOPE)
        if BLOCK_K < 2:
            BLOCK_K = 2

        C = torch.empty((H, BS, d_lora), device="cuda", dtype=torch.float16)
        grid = (H, triton.cdiv(BS, BLOCK_M) * triton.cdiv(d_lora, BLOCK_N))

        # Warmup
        for _ in range(5):
            kernel_batched_w4a16_simple[grid](
                x, w_packed, scales, C,
                BS, d_lora, D_NOPE,
                x.stride(0), x.stride(1), x.stride(2),
                w_packed.stride(0), w_packed.stride(1), w_packed.stride(2),
                scales.stride(0), scales.stride(1),
                C.stride(0), C.stride(1), C.stride(2),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
        torch.cuda.synchronize()

        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(3):
            kernel_batched_w4a16_simple[grid](
                x, w_packed, scales, C,
                BS, d_lora, D_NOPE,
                x.stride(0), x.stride(1), x.stride(2),
                w_packed.stride(0), w_packed.stride(1), w_packed.stride(2),
                scales.stride(0), scales.stride(1),
                C.stride(0), C.stride(1), C.stride(2),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    else:
        print(f"Unknown kernel type: {kernel_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
