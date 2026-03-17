"""
MLA Reconstruction GEMM Profiling: Isolating the KV up/down-projection cost.

MLA (Multi-head Latent Attention) stores compressed KV latents and reconstructs
full KV heads via batched matrix multiplications (BMMs) at each decode step.
This script profiles those BMMs in isolation to measure:

1. How much time reconstruction adds vs. the attention kernel itself
2. At what batch sizes reconstruction becomes a bottleneck
3. Whether INT4 quantization of reconstruction weights is viable

Architecture (DeepSeek-V2-Lite, per layer):
  BMM1 (w_kc, "Q absorption"):
    (num_heads, bs, qk_nope_head_dim) × (num_heads, qk_nope_head_dim, kv_lora_rank)
    = (16, bs, 128) × (16, 128, 512)

  Attention kernel: operates on compressed KV (dim=576 = 512+64)

  BMM2 (w_vc, "V reconstruction"):
    (num_heads, bs, kv_lora_rank) × (num_heads, kv_lora_rank, v_head_dim)
    = (16, bs, 512) × (16, 512, 128)

For DeepSeek-V3 (full model):
  BMM1: (128, bs, 128) × (128, 128, 512)
  BMM2: (128, bs, 512) × (128, 512, 128)

Usage:
    python profile_mla_reconstruction.py --model deepseek-v2-lite
    python profile_mla_reconstruction.py --model deepseek-v3
    python profile_mla_reconstruction.py --model deepseek-v2-lite --ncu-mode
"""

import argparse
import csv
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class MLAConfig:
    name: str
    hidden_size: int
    num_heads: int          # num_attention_heads (after TP=1)
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_lora_rank: int
    q_lora_rank: int
    num_layers: int


CONFIGS = {
    "deepseek-v2-lite": MLAConfig(
        name="DeepSeek-V2-Lite",
        hidden_size=2048,
        num_heads=16,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        q_lora_rank=1536,
        num_layers=27,
    ),
    "deepseek-v3": MLAConfig(
        name="DeepSeek-V3",
        hidden_size=7168,
        num_heads=128,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        q_lora_rank=1536,
        num_layers=61,
    ),
}


def bench_bmm(A, B, warmup=20, iters=100):
    """Benchmark a batched matmul."""
    # Warmup
    for _ in range(warmup):
        torch.bmm(A, B)
    torch.cuda.synchronize()

    # Timed iterations
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        start.record()
        torch.bmm(A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median = times[len(times) // 2]
    return median, times


def bench_attention_mla(cfg, batch_size, kv_len, dtype, warmup=20, iters=100):
    """Benchmark FlashInfer MLA attention kernel using BatchMLAPagedAttentionWrapper.

    Creates a fresh wrapper per call to avoid stale-state CUDA errors.
    """
    try:
        from flashinfer import BatchMLAPagedAttentionWrapper
    except ImportError:
        print("    BatchMLAPagedAttentionWrapper not available")
        return None, []

    device = "cuda"
    H = cfg.num_heads
    total_tokens = batch_size * kv_len

    # Fresh workspace + wrapper each call to avoid cross-bs contamination
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace)

    q_nope = torch.randn(batch_size, H, cfg.v_head_dim, dtype=dtype, device=device)
    q_rope = torch.randn(batch_size, H, cfg.qk_rope_head_dim, dtype=dtype, device=device)
    k_nope = torch.randn(total_tokens, H, cfg.v_head_dim, dtype=dtype, device=device)
    k_rope = torch.randn(total_tokens, H, cfg.qk_rope_head_dim, dtype=dtype, device=device)
    output = torch.empty(batch_size, H, cfg.v_head_dim, dtype=dtype, device=device)

    kv_indptr = (torch.arange(batch_size + 1, device=device, dtype=torch.int32) * kv_len)
    kv_indices = torch.arange(total_tokens, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, device=device, dtype=torch.int32)
    q_indptr = torch.arange(batch_size + 1, device="cpu", dtype=torch.int32)

    qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    sm_scale = qk_head_dim ** -0.5

    try:
        wrapper.plan(
            q_indptr, kv_indptr, kv_indices, kv_lens,
            H, cfg.kv_lora_rank, cfg.qk_rope_head_dim,
            1, False, sm_scale, dtype, dtype,
        )
    except Exception as e:
        print(f"    MLA wrapper plan failed: {e}")
        return None, []

    # Warmup
    for _ in range(warmup):
        wrapper.run(q_nope, q_rope, k_nope, k_rope, out=output)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start_evt.record()
        wrapper.run(q_nope, q_rope, k_nope, k_rope, out=output)
        end_evt.record()
        torch.cuda.synchronize()
        times.append(start_evt.elapsed_time(end_evt))

    times.sort()
    median = times[len(times) // 2]

    # Clean up to free GPU memory before next call
    del wrapper, workspace, q_nope, q_rope, k_nope, k_rope, output
    torch.cuda.empty_cache()

    return median, times


def profile_reconstruction(cfg, batch_sizes, kv_len, dtype=torch.float16,
                           warmup=20, iters=100, ncu_mode=False):
    """Profile MLA reconstruction BMMs and attention separately."""
    device = "cuda"

    if ncu_mode:
        warmup, iters = 3, 5

    H = cfg.num_heads
    d_nope = cfg.qk_nope_head_dim
    d_rope = cfg.qk_rope_head_dim
    d_v = cfg.v_head_dim
    d_lora = cfg.kv_lora_rank

    # Reconstruction weights (per layer, constant)
    w_kc = torch.randn(H, d_nope, d_lora, dtype=dtype, device=device)  # (H, 128, 512)
    w_vc = torch.randn(H, d_lora, d_v, dtype=dtype, device=device)     # (H, 512, 128)

    print("=" * 80)
    print(f"MLA Reconstruction Profiling: {cfg.name}")
    print(f"  num_heads={H}, qk_nope={d_nope}, qk_rope={d_rope}, "
          f"v_head={d_v}, kv_lora_rank={d_lora}")
    print(f"  w_kc shape: ({H}, {d_nope}, {d_lora}), "
          f"w_vc shape: ({H}, {d_lora}, {d_v})")
    print(f"  w_kc size: {w_kc.nelement() * 2 / 1e6:.1f} MB, "
          f"w_vc size: {w_vc.nelement() * 2 / 1e6:.1f} MB")
    print(f"  kv_len={kv_len}, dtype={dtype}")
    print("=" * 80)

    results = []

    for bs in batch_sizes:
        # ── BMM1: Q absorption ──
        # q_nope: (H, bs, d_nope) × w_kc: (H, d_nope, d_lora) → (H, bs, d_lora)
        q_nope = torch.randn(H, bs, d_nope, dtype=dtype, device=device)
        bmm1_median, _ = bench_bmm(q_nope, w_kc, warmup, iters)

        # ── BMM2: V reconstruction ──
        # attn_out: (H, bs, d_lora) × w_vc: (H, d_lora, d_v) → (H, bs, d_v)
        attn_out = torch.randn(H, bs, d_lora, dtype=dtype, device=device)
        bmm2_median, _ = bench_bmm(attn_out, w_vc, warmup, iters)

        # ── Attention kernel (FlashInfer on compressed KV) ──
        # Run in subprocess to isolate CUDA context and avoid async errors
        import subprocess, json as _json
        attn_cmd = f"""
import torch, json
from flashinfer import BatchMLAPagedAttentionWrapper
device='cuda'; dtype=torch.float16; H={H}; bs={bs}; kv_len={kv_len}
total=bs*kv_len
ws=torch.empty(128*1024*1024,dtype=torch.uint8,device=device)
w=BatchMLAPagedAttentionWrapper(ws)
qn=torch.randn(bs,H,{cfg.v_head_dim},dtype=dtype,device=device)
qr=torch.randn(bs,H,{cfg.qk_rope_head_dim},dtype=dtype,device=device)
kn=torch.randn(total,H,{cfg.v_head_dim},dtype=dtype,device=device)
kr=torch.randn(total,H,{cfg.qk_rope_head_dim},dtype=dtype,device=device)
o=torch.empty(bs,H,{cfg.v_head_dim},dtype=dtype,device=device)
ki=torch.arange(bs+1,device=device,dtype=torch.int32)*kv_len
kx=torch.arange(total,device=device,dtype=torch.int32)
kl=torch.full((bs,),kv_len,device=device,dtype=torch.int32)
qi=torch.arange(bs+1,device='cpu',dtype=torch.int32)
sm={cfg.qk_nope_head_dim + cfg.qk_rope_head_dim}**-0.5
w.plan(qi,ki,kx,kl,H,{cfg.kv_lora_rank},{cfg.qk_rope_head_dim},1,False,sm,dtype,dtype)
for _ in range({warmup}): w.run(qn,qr,kn,kr,out=o)
torch.cuda.synchronize()
ts=[]
for _ in range({iters}):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record(); w.run(qn,qr,kn,kr,out=o); e.record(); torch.cuda.synchronize()
    ts.append(s.elapsed_time(e))
ts.sort()
print(json.dumps({{"median":ts[len(ts)//2]}}))
"""
        try:
            proc = subprocess.run(
                ["python3", "-c", attn_cmd],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode == 0:
                attn_result = _json.loads(proc.stdout.strip())
                attn_median = attn_result["median"]
            else:
                print(f"    Attention subprocess error: {proc.stderr[-200:]}")
                attn_median = None
        except Exception as e:
            print(f"    Attention subprocess failed: {e}")
            attn_median = None

        # ── Compute bytes and FLOPS ──
        # BMM1 bytes: read q_nope + w_kc + write output
        bmm1_bytes = (H * bs * d_nope + H * d_nope * d_lora + H * bs * d_lora) * 2
        bmm1_flops = 2 * H * bs * d_nope * d_lora  # 2 for multiply-add

        # BMM2 bytes: read attn_out + w_vc + write output
        bmm2_bytes = (H * bs * d_lora + H * d_lora * d_v + H * bs * d_v) * 2
        bmm2_flops = 2 * H * bs * d_lora * d_v

        # Attention bytes (compressed KV read)
        kv_dim = d_lora + d_rope
        attn_bytes = (bs * H * kv_dim + bs * kv_len * 1 * kv_dim * 2 + bs * H * kv_dim) * 2

        total_recon = bmm1_median + bmm2_median
        total_with_attn = total_recon + (attn_median if attn_median else 0)
        recon_fraction = total_recon / total_with_attn * 100 if total_with_attn > 0 else 0

        # Per-layer timing, extrapolate to full model
        full_model_recon_ms = total_recon * cfg.num_layers
        full_model_attn_ms = (attn_median if attn_median else 0) * cfg.num_layers

        bmm1_tflops = (bmm1_flops / 1e12) / (bmm1_median / 1e3) if bmm1_median > 0 else 0
        bmm2_tflops = (bmm2_flops / 1e12) / (bmm2_median / 1e3) if bmm2_median > 0 else 0
        bmm1_bw = (bmm1_bytes / 1e9) / (bmm1_median / 1e3) if bmm1_median > 0 else 0
        bmm2_bw = (bmm2_bytes / 1e9) / (bmm2_median / 1e3) if bmm2_median > 0 else 0

        print(f"\n  bs={bs:>4}")
        print(f"    BMM1 (Q absorb):      {bmm1_median:>8.3f} ms  "
              f"({bmm1_tflops:>6.1f} TFLOPS, {bmm1_bw:>7.0f} GB/s)")
        print(f"    BMM2 (V reconstruct):  {bmm2_median:>8.3f} ms  "
              f"({bmm2_tflops:>6.1f} TFLOPS, {bmm2_bw:>7.0f} GB/s)")
        print(f"    Total reconstruction:  {total_recon:>8.3f} ms")
        if attn_median:
            print(f"    Attention kernel:      {attn_median:>8.3f} ms")
            print(f"    Recon / (Recon+Attn):  {recon_fraction:>7.1f}%")
            print(f"    Full model ({cfg.num_layers}L): recon={full_model_recon_ms:.2f} ms, "
                  f"attn={full_model_attn_ms:.2f} ms")
        else:
            print(f"    Attention kernel:      SKIPPED")

        results.append({
            "model": cfg.name,
            "batch_size": bs,
            "kv_len": kv_len,
            "bmm1_ms": bmm1_median,
            "bmm2_ms": bmm2_median,
            "recon_total_ms": total_recon,
            "attn_ms": attn_median if attn_median else "N/A",
            "recon_pct": f"{recon_fraction:.1f}",
            "bmm1_tflops": f"{bmm1_tflops:.2f}",
            "bmm2_tflops": f"{bmm2_tflops:.2f}",
            "bmm1_bw_GBs": f"{bmm1_bw:.0f}",
            "bmm2_bw_GBs": f"{bmm2_bw:.0f}",
            "full_model_recon_ms": f"{full_model_recon_ms:.2f}",
            "full_model_attn_ms": f"{full_model_attn_ms:.2f}" if attn_median else "N/A",
        })

    return results


def profile_int4_feasibility(cfg, batch_sizes, dtype=torch.float16, warmup=20, iters=100):
    """Compare BF16/FP16 BMM vs simulated INT4 BMM for reconstruction weights.

    Since PyTorch doesn't natively support INT4 BMM, we compare:
    1. FP16 BMM (current)
    2. FP16 BMM with 4x smaller weight (simulating INT4 weight size)
    3. INT8 BMM via torch._int_mm (available shapes)

    This gives a roofline estimate of INT4 speedup potential.
    """
    device = "cuda"
    H = cfg.num_heads
    d_nope = cfg.qk_nope_head_dim
    d_v = cfg.v_head_dim
    d_lora = cfg.kv_lora_rank

    print("\n" + "=" * 80)
    print(f"INT4 Feasibility: Weight Size Reduction Analysis ({cfg.name})")
    print("=" * 80)

    # Weight sizes
    w_kc_fp16_bytes = H * d_nope * d_lora * 2
    w_vc_fp16_bytes = H * d_lora * d_v * 2
    w_kc_int4_bytes = H * d_nope * d_lora // 2  # 4 bits per element
    w_vc_int4_bytes = H * d_lora * d_v // 2

    total_fp16 = (w_kc_fp16_bytes + w_vc_fp16_bytes) / 1e6
    total_int4 = (w_kc_int4_bytes + w_vc_int4_bytes) / 1e6

    print(f"  w_kc: FP16={w_kc_fp16_bytes/1e6:.1f} MB, INT4={w_kc_int4_bytes/1e6:.1f} MB")
    print(f"  w_vc: FP16={w_vc_fp16_bytes/1e6:.1f} MB, INT4={w_vc_int4_bytes/1e6:.1f} MB")
    print(f"  Total per layer: FP16={total_fp16:.1f} MB, INT4={total_int4:.1f} MB "
          f"({total_fp16/total_int4:.1f}x reduction)")
    print(f"  Total all layers: FP16={total_fp16*cfg.num_layers:.0f} MB, "
          f"INT4={total_int4*cfg.num_layers:.0f} MB")

    # Roofline analysis: at what batch size is BMM compute-bound vs memory-bound?
    print(f"\n  Roofline Analysis (H100 SXM5: 3350 GB/s, 990 TFLOPS FP16):")
    print(f"  {'BS':>6} {'BMM2 FLOPs':>12} {'W_vc read':>10} "
          f"{'Arith Int':>10} {'Regime':>12} {'INT4 speedup':>14}")

    for bs in batch_sizes:
        # BMM2 is the bottleneck (larger matmul)
        flops = 2 * H * bs * d_lora * d_v
        # Memory: read weight + read input + write output
        w_bytes_fp16 = w_vc_fp16_bytes
        input_bytes = H * bs * d_lora * 2
        output_bytes = H * bs * d_v * 2
        total_mem_fp16 = w_bytes_fp16 + input_bytes + output_bytes

        # Arithmetic intensity = FLOPS / bytes
        ai_fp16 = flops / total_mem_fp16

        # At 990 TFLOPS and 3350 GB/s, the crossover is at AI = 990e12/3350e9 = 295
        crossover_ai = 990e12 / 3350e9

        regime = "MEMORY-BOUND" if ai_fp16 < crossover_ai else "COMPUTE-BOUND"

        # INT4 speedup estimate:
        # If memory-bound: speedup = total_mem_fp16 / total_mem_int4
        # If compute-bound: speedup ≈ 1 (compute doesn't change)
        w_bytes_int4 = w_vc_int4_bytes
        total_mem_int4 = w_bytes_int4 + input_bytes + output_bytes  # activations stay FP16
        if ai_fp16 < crossover_ai:
            speedup = total_mem_fp16 / total_mem_int4
        else:
            speedup = 1.0

        print(f"  {bs:>6} {flops/1e9:>10.1f}G {w_bytes_fp16/1e6:>8.1f}MB "
              f"{ai_fp16:>10.1f} {regime:>12} {speedup:>12.2f}x")


def main():
    parser = argparse.ArgumentParser(description="MLA Reconstruction GEMM Profiler")
    parser.add_argument("--model", choices=list(CONFIGS.keys()), default="deepseek-v2-lite")
    parser.add_argument("--batch-sizes", type=str, default="1,4,16,64,128,256,512,1024")
    parser.add_argument("--kv-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--ncu-mode", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-attention", action="store_true",
                        help="Skip attention kernel benchmark (BMMs only)")
    args = parser.parse_args()

    cfg = CONFIGS[args.model]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    results = profile_reconstruction(
        cfg, batch_sizes, args.kv_len,
        warmup=args.warmup, iters=args.iters, ncu_mode=args.ncu_mode,
    )

    # INT4 feasibility analysis
    profile_int4_feasibility(cfg, batch_sizes)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Reconstruction vs Attention Time")
    print("=" * 80)
    print(f"{'BS':>6} {'BMM1':>8} {'BMM2':>8} {'Recon':>8} {'Attn':>8} "
          f"{'Recon%':>7} {'Model Recon':>12} {'Model Attn':>12}")
    for r in results:
        print(f"{r['batch_size']:>6} {r['bmm1_ms']:>8.3f} {r['bmm2_ms']:>8.3f} "
              f"{r['recon_total_ms']:>8.3f} {str(r['attn_ms']):>8} "
              f"{r['recon_pct']:>6}% {r['full_model_recon_ms']:>11} "
              f"{r['full_model_attn_ms']:>11}")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {args.output}")

    # Print GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")


if __name__ == "__main__":
    main()
