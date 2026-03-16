"""
FlashInfer vs Triton attention kernel microbenchmark for SGLang.

Profiles both FlashInfer and SGLang's Triton attention kernels at the CUDA
kernel level with controlled tensor shapes matching real model configurations.
Produces CSV data for direct comparison.

Usage:
    # Compare FlashInfer vs Triton decode on Llama-3-8B shapes
    python profile_attention_kernels.py --model llama-8b --mode decode --batch-sizes 1,16,64,256

    # FlashInfer-only MLA decode (Triton MLA not yet comparable)
    python profile_attention_kernels.py --model deepseek-v2-lite --mode decode --backend flashinfer

    # All combinations, full sweep, CSV output
    python profile_attention_kernels.py --model llama-8b --mode all --output results.csv

    # Minimal run for NCU profiling (use with ncu_profile.sh)
    python profile_attention_kernels.py --model llama-8b --mode decode --batch-sizes 64 --kv-lens 2048 --ncu-mode

    # With torch profiler trace (viewable in perfetto)
    python profile_attention_kernels.py --model llama-8b --mode decode --torch-profile --trace-dir ./traces
"""

import argparse
import csv
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

# ── Model shape configurations ──────────────────────────────────────────────

@dataclass
class AttentionConfig:
    name: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    # MLA-specific (None for standard attention)
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    @property
    def is_mla(self):
        return self.kv_lora_rank is not None


MODEL_CONFIGS = {
    "llama-8b": AttentionConfig(
        name="Llama-3-8B",
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
    ),
    "llama-70b": AttentionConfig(
        name="Llama-3-70B",
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=128,
    ),
    "deepseek-v2-lite": AttentionConfig(
        name="DeepSeek-V2-Lite",
        num_q_heads=16,
        num_kv_heads=16,  # MLA: all heads process compressed KV
        head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
    ),
    "deepseek-v3": AttentionConfig(
        name="DeepSeek-V3",
        num_q_heads=128,
        num_kv_heads=128,
        head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
    ),
}


# ── Shared helpers ──────────────────────────────────────────────────────────

def compute_decode_bandwidth(cfg, batch_size, kv_len, dtype, median_ms):
    """Compute achieved memory bandwidth for decode attention."""
    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 1
    kv_bytes = 2 * batch_size * kv_len * cfg.num_kv_heads * cfg.head_dim * bytes_per_elem
    q_bytes = batch_size * cfg.num_q_heads * cfg.head_dim * bytes_per_elem
    out_bytes = batch_size * cfg.num_q_heads * cfg.head_dim * bytes_per_elem
    total_bytes = kv_bytes + q_bytes + out_bytes
    bandwidth_gb_s = (total_bytes / 1e9) / (median_ms / 1e3) if median_ms > 0 else 0
    return total_bytes, bandwidth_gb_s


def compute_prefill_tflops(cfg, batch_size, seq_len, median_ms):
    """Compute achieved TFLOPS for prefill attention (causal)."""
    flops = 2 * batch_size * (seq_len * seq_len / 2) * cfg.num_q_heads * cfg.head_dim
    tflops = (flops / 1e12) / (median_ms / 1e3) if median_ms > 0 else 0
    return flops, tflops


def timed_run(fn, warmup, iters, ncu_mode):
    """Generic timed execution with CUDA events and optional NCU markers."""
    if ncu_mode:
        warmup, iters = 2, 5

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if ncu_mode:
        torch.cuda.cudart().cudaProfilerStart()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    if ncu_mode:
        torch.cuda.cudart().cudaProfilerStop()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return times_ms


def summarize_times(times_ms):
    median = sorted(times_ms)[len(times_ms) // 2]
    mean = sum(times_ms) / len(times_ms)
    return median, mean, min(times_ms), max(times_ms)


# ── FlashInfer GQA decode ───────────────────────────────────────────────────

def bench_flashinfer_decode(cfg, batch_size, kv_len, dtype, warmup=10, iters=100, ncu_mode=False):
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    device = "cuda"
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=True)

    q = torch.randn(batch_size, cfg.num_q_heads, cfg.head_dim, dtype=dtype, device=device)
    total_tokens = batch_size * kv_len
    k_buf = torch.randn(total_tokens, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)
    v_buf = torch.randn(total_tokens, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)

    kv_indptr = (torch.arange(batch_size + 1, device=device, dtype=torch.int32) * kv_len)
    kv_indices = torch.arange(total_tokens, device=device, dtype=torch.int32)
    kv_last_page_len = torch.ones(batch_size, device=device, dtype=torch.int32)
    sm_scale = cfg.head_dim ** -0.5

    wrapper.begin_forward(
        kv_indptr, kv_indices, kv_last_page_len,
        cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
        page_size=1, pos_encoding_mode="NONE", data_type=dtype,
    )

    def run():
        wrapper.forward(q, (k_buf, v_buf), sm_scale=sm_scale)

    times_ms = timed_run(run, warmup, iters, ncu_mode)
    wrapper.end_forward()

    median, mean, tmin, tmax = summarize_times(times_ms)
    total_bytes, bw = compute_decode_bandwidth(cfg, batch_size, kv_len, dtype, median)

    return {
        "backend": "flashinfer",
        "model": cfg.name, "mode": "decode", "attn_type": "GQA",
        "batch_size": batch_size, "kv_len": kv_len, "dtype": str(dtype),
        "median_ms": round(median, 4), "mean_ms": round(mean, 4),
        "min_ms": round(tmin, 4), "max_ms": round(tmax, 4),
        "total_bytes_GB": round(total_bytes / 1e9, 4),
        "bandwidth_GB_s": round(bw, 2),
    }


# ── Triton GQA decode ──────────────────────────────────────────────────────

def bench_triton_decode(cfg, batch_size, kv_len, dtype, warmup=10, iters=100, ncu_mode=False):
    from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd

    device = "cuda"
    q = torch.randn(batch_size, cfg.num_q_heads, cfg.head_dim, dtype=dtype, device=device)
    total_tokens = batch_size * kv_len
    k_buf = torch.randn(total_tokens, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)
    v_buf = torch.randn(total_tokens, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)
    o = torch.empty_like(q)

    kv_indptr = (torch.arange(batch_size + 1, device=device, dtype=torch.int64) * kv_len)
    kv_indices = torch.arange(total_tokens, device=device, dtype=torch.int64)
    sm_scale = cfg.head_dim ** -0.5

    # Triton decode uses split-kv: compute num_kv_splits (per-batch tensor)
    num_kv_splits_val = max(1, min(32, (kv_len + 255) // 256))
    max_kv_splits = num_kv_splits_val
    num_kv_splits = torch.full(
        (batch_size,), num_kv_splits_val, dtype=torch.int64, device=device,
    )

    attn_logits = torch.empty(
        (batch_size, cfg.num_q_heads, max_kv_splits, cfg.head_dim + 1),
        dtype=torch.float32, device=device,
    )
    attn_lse = torch.empty(
        (batch_size, cfg.num_q_heads, max_kv_splits),
        dtype=torch.float32, device=device,
    )

    def run():
        decode_attention_fwd(
            q, k_buf, v_buf, o,
            kv_indptr, kv_indices,
            attn_logits, attn_lse,
            num_kv_splits, max_kv_splits,
            sm_scale,
            1.0,   # k_scale
            1.0,   # v_scale
        )

    times_ms = timed_run(run, warmup, iters, ncu_mode)

    median, mean, tmin, tmax = summarize_times(times_ms)
    total_bytes, bw = compute_decode_bandwidth(cfg, batch_size, kv_len, dtype, median)

    return {
        "backend": "triton",
        "model": cfg.name, "mode": "decode", "attn_type": "GQA",
        "batch_size": batch_size, "kv_len": kv_len, "dtype": str(dtype),
        "median_ms": round(median, 4), "mean_ms": round(mean, 4),
        "min_ms": round(tmin, 4), "max_ms": round(tmax, 4),
        "total_bytes_GB": round(total_bytes / 1e9, 4),
        "bandwidth_GB_s": round(bw, 2),
    }


# ── FlashInfer GQA prefill ─────────────────────────────────────────────────

def bench_flashinfer_prefill(cfg, batch_size, seq_len, dtype, warmup=5, iters=50, ncu_mode=False):
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

    device = "cuda"
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")

    total_tokens = batch_size * seq_len
    q = torch.randn(total_tokens, cfg.num_q_heads, cfg.head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)

    qo_indptr = (torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seq_len)
    kv_indptr = qo_indptr.clone()
    sm_scale = cfg.head_dim ** -0.5

    wrapper.begin_forward(
        qo_indptr, kv_indptr,
        cfg.num_q_heads, cfg.num_kv_heads,
        cfg.head_dim, cfg.head_dim,
        q_data_type=dtype,
    )

    def run():
        wrapper.forward(q, k, v, causal=True, sm_scale=sm_scale)

    times_ms = timed_run(run, warmup, iters, ncu_mode)
    wrapper.end_forward()

    median, mean, tmin, tmax = summarize_times(times_ms)
    _, tflops = compute_prefill_tflops(cfg, batch_size, seq_len, median)

    return {
        "backend": "flashinfer",
        "model": cfg.name, "mode": "prefill", "attn_type": "GQA",
        "batch_size": batch_size, "seq_len": seq_len, "dtype": str(dtype),
        "median_ms": round(median, 4), "mean_ms": round(mean, 4),
        "min_ms": round(tmin, 4), "max_ms": round(tmax, 4),
        "tflops": round(tflops, 2),
    }


# ── Triton GQA prefill ─────────────────────────────────────────────────────

def bench_triton_prefill(cfg, batch_size, seq_len, dtype, warmup=5, iters=50, ncu_mode=False):
    from sglang.srt.layers.attention.triton_ops.prefill_attention import context_attention_fwd

    device = "cuda"
    total_tokens = batch_size * seq_len
    q = torch.randn(total_tokens, cfg.num_q_heads, cfg.head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)
    o = torch.empty_like(q)

    b_start_loc = (torch.arange(batch_size, device=device, dtype=torch.int32) * seq_len)
    b_seq_len = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    def run():
        context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, seq_len, is_causal=True)

    times_ms = timed_run(run, warmup, iters, ncu_mode)

    median, mean, tmin, tmax = summarize_times(times_ms)
    _, tflops = compute_prefill_tflops(cfg, batch_size, seq_len, median)

    return {
        "backend": "triton",
        "model": cfg.name, "mode": "prefill", "attn_type": "GQA",
        "batch_size": batch_size, "seq_len": seq_len, "dtype": str(dtype),
        "median_ms": round(median, 4), "mean_ms": round(mean, 4),
        "min_ms": round(tmin, 4), "max_ms": round(tmax, 4),
        "tflops": round(tflops, 2),
    }


# ── FlashInfer MLA decode ──────────────────────────────────────────────────

def bench_flashinfer_mla_decode(cfg, batch_size, kv_len, dtype, warmup=10, iters=100, ncu_mode=False):
    try:
        from flashinfer import BatchMLAPagedAttentionWrapper
    except ImportError:
        print("WARNING: BatchMLAPagedAttentionWrapper not available, skipping MLA")
        return None

    device = "cuda"
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace)

    num_heads = cfg.num_q_heads
    total_tokens = batch_size * kv_len

    q_nope = torch.randn(batch_size, num_heads, cfg.v_head_dim, dtype=dtype, device=device)
    q_rope = torch.randn(batch_size, num_heads, cfg.qk_rope_head_dim, dtype=dtype, device=device)
    k_nope = torch.randn(total_tokens, num_heads, cfg.v_head_dim, dtype=dtype, device=device)
    k_rope = torch.randn(total_tokens, num_heads, cfg.qk_rope_head_dim, dtype=dtype, device=device)
    output = torch.empty(batch_size, num_heads, cfg.v_head_dim, dtype=dtype, device=device)

    kv_indptr = (torch.arange(batch_size + 1, device=device, dtype=torch.int32) * kv_len)
    kv_indices = torch.arange(total_tokens, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), kv_len, device=device, dtype=torch.int32)
    q_indptr = torch.arange(batch_size + 1, device="cpu", dtype=torch.int32)

    qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    sm_scale = qk_head_dim ** -0.5

    wrapper.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        num_heads, cfg.kv_lora_rank, cfg.qk_rope_head_dim,
        1, False, sm_scale, dtype, dtype,
    )

    def run():
        wrapper.run(q_nope, q_rope, k_nope, k_rope, out=output)

    times_ms = timed_run(run, warmup, iters, ncu_mode)

    median, mean, tmin, tmax = summarize_times(times_ms)

    # MLA memory: KV reads (nope + rope for K, and V via nope)
    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 1
    kv_bytes = batch_size * kv_len * num_heads * (cfg.v_head_dim + cfg.qk_rope_head_dim) * 2 * bytes_per_elem
    q_bytes = batch_size * num_heads * (cfg.v_head_dim + cfg.qk_rope_head_dim) * bytes_per_elem
    out_bytes = batch_size * num_heads * cfg.v_head_dim * bytes_per_elem
    total_bytes = kv_bytes + q_bytes + out_bytes
    bw = (total_bytes / 1e9) / (median / 1e3) if median > 0 else 0

    return {
        "backend": "flashinfer",
        "model": cfg.name, "mode": "decode", "attn_type": "MLA",
        "batch_size": batch_size, "kv_len": kv_len, "dtype": str(dtype),
        "median_ms": round(median, 4), "mean_ms": round(mean, 4),
        "min_ms": round(tmin, 4), "max_ms": round(tmax, 4),
        "total_bytes_GB": round(total_bytes / 1e9, 4),
        "bandwidth_GB_s": round(bw, 2),
    }


# ── FlashInfer MLA prefill ─────────────────────────────────────────────────

def bench_flashinfer_mla_prefill(cfg, batch_size, seq_len, dtype, warmup=5, iters=50, ncu_mode=False):
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

    device = "cuda"
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")

    num_heads = cfg.num_q_heads
    total_tokens = batch_size * seq_len
    qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

    q = torch.randn(total_tokens, num_heads, qk_head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, num_heads, qk_head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, num_heads, cfg.v_head_dim, dtype=dtype, device=device)

    qo_indptr = (torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seq_len)
    kv_indptr = qo_indptr.clone()
    sm_scale = qk_head_dim ** -0.5

    wrapper.begin_forward(
        qo_indptr, kv_indptr,
        num_heads, num_heads,
        qk_head_dim, cfg.v_head_dim,
        q_data_type=dtype,
    )

    def run():
        wrapper.forward(q, k, v, causal=True, sm_scale=sm_scale)

    times_ms = timed_run(run, warmup, iters, ncu_mode)
    wrapper.end_forward()

    median, mean, tmin, tmax = summarize_times(times_ms)
    flops = 2 * batch_size * (seq_len * seq_len / 2) * num_heads * qk_head_dim
    tflops = (flops / 1e12) / (median / 1e3) if median > 0 else 0

    return {
        "backend": "flashinfer",
        "model": cfg.name, "mode": "prefill", "attn_type": "MLA",
        "batch_size": batch_size, "seq_len": seq_len, "dtype": str(dtype),
        "median_ms": round(median, 4), "mean_ms": round(mean, 4),
        "min_ms": round(tmin, 4), "max_ms": round(tmax, 4),
        "tflops": round(tflops, 2),
    }


# ── Torch profiler integration ──────────────────────────────────────────────

def run_with_torch_profiler(fn, trace_dir, label, **kwargs):
    import os
    os.makedirs(trace_dir, exist_ok=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        result = fn(**kwargs)
    trace_path = os.path.join(trace_dir, f"{label}.json")
    prof.export_chrome_trace(trace_path)
    print(f"  trace -> {trace_path}")
    return result


# ── Main ────────────────────────────────────────────────────────────────────

def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="FlashInfer vs Triton attention kernel profiler for SGLang"
    )
    parser.add_argument("--model", type=str, default="llama-8b",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"])
    parser.add_argument("--mode", type=str, default="decode",
                        choices=["decode", "prefill", "all"])
    parser.add_argument("--backend", type=str, default="both",
                        choices=["flashinfer", "triton", "both"],
                        help="Which backend(s) to benchmark")
    parser.add_argument("--batch-sizes", type=str, default="1,4,16,64,256")
    parser.add_argument("--kv-lens", type=str, default="512,1024,2048,4096,8192")
    parser.add_argument("--seq-lens", type=str, default="512,1024,2048,4096")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--torch-profile", action="store_true")
    parser.add_argument("--trace-dir", type=str, default="./traces")
    parser.add_argument("--ncu-mode", action="store_true",
                        help="Minimal iters with cudaProfiler markers for NCU")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    batch_sizes = parse_int_list(args.batch_sizes)
    kv_lens = parse_int_list(args.kv_lens)
    seq_lens = parse_int_list(args.seq_lens)

    models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    modes = ["decode", "prefill"] if args.mode == "all" else [args.mode]
    backends = ["flashinfer", "triton"] if args.backend == "both" else [args.backend]

    results = []

    for model_name in models:
        cfg = MODEL_CONFIGS[model_name]
        is_mla = cfg.is_mla

        print(f"\n{'='*70}")
        print(f"Model: {cfg.name}  |  Type: {'MLA' if is_mla else 'GQA'}  |  Backends: {backends}")
        print(f"{'='*70}")

        for mode in modes:
            if mode == "decode":
                for bs, kvl in itertools.product(batch_sizes, kv_lens):
                    for backend in backends:
                        # Triton doesn't have an MLA-specific decode kernel (yet)
                        if is_mla and backend == "triton":
                            continue

                        label = f"{backend}_{model_name}_decode_bs{bs}_kv{kvl}"
                        print(f"\n  [{backend:>10}] decode  bs={bs:>4}  kv_len={kvl:>5} ... ", end="", flush=True)

                        if is_mla:
                            bench_fn = bench_flashinfer_mla_decode
                        elif backend == "flashinfer":
                            bench_fn = bench_flashinfer_decode
                        else:
                            bench_fn = bench_triton_decode

                        kwargs = dict(
                            cfg=cfg, batch_size=bs, kv_len=kvl, dtype=dtype,
                            warmup=args.warmup, iters=args.iters, ncu_mode=args.ncu_mode,
                        )

                        try:
                            if args.torch_profile:
                                r = run_with_torch_profiler(bench_fn, args.trace_dir, label, **kwargs)
                            else:
                                r = bench_fn(**kwargs)

                            if r is not None:
                                results.append(r)
                                bw = r.get("bandwidth_GB_s", "N/A")
                                print(f"median={r['median_ms']:.3f}ms  BW={bw} GB/s")
                        except Exception as e:
                            print(f"ERROR: {e}")

            elif mode == "prefill":
                for bs, sl in itertools.product(batch_sizes, seq_lens):
                    if bs * sl > 500_000:
                        continue

                    for backend in backends:
                        if is_mla and backend == "triton":
                            continue

                        label = f"{backend}_{model_name}_prefill_bs{bs}_seq{sl}"
                        print(f"\n  [{backend:>10}] prefill bs={bs:>4}  seq_len={sl:>5} ... ", end="", flush=True)

                        if is_mla:
                            bench_fn = bench_flashinfer_mla_prefill
                        elif backend == "flashinfer":
                            bench_fn = bench_flashinfer_prefill
                        else:
                            bench_fn = bench_triton_prefill

                        kwargs = dict(
                            cfg=cfg, batch_size=bs, seq_len=sl, dtype=dtype,
                            warmup=args.warmup, iters=args.iters, ncu_mode=args.ncu_mode,
                        )

                        try:
                            if args.torch_profile:
                                r = run_with_torch_profiler(bench_fn, args.trace_dir, label, **kwargs)
                            else:
                                r = bench_fn(**kwargs)

                            if r is not None:
                                results.append(r)
                                tf = r.get("tflops", "N/A")
                                print(f"median={r['median_ms']:.3f}ms  TFLOPS={tf}")
                        except Exception as e:
                            print(f"ERROR: {e}")

    # ── Summary ──

    if results:
        print(f"\n\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")

        # Group by config and compare backends
        from collections import defaultdict
        grouped = defaultdict(dict)
        for r in results:
            key = (r["model"], r["mode"], r.get("batch_size"), r.get("kv_len", r.get("seq_len")))
            grouped[key][r["backend"]] = r

        for key, by_backend in grouped.items():
            model, mode, bs, length = key
            parts = [f"{model} {mode} bs={bs} len={length}"]
            for backend, r in sorted(by_backend.items()):
                metric = r.get("bandwidth_GB_s", r.get("tflops", "?"))
                metric_name = "GB/s" if "bandwidth_GB_s" in r else "TFLOPS"
                parts.append(f"  {backend}: {r['median_ms']:.3f}ms ({metric} {metric_name})")
            if len(by_backend) == 2:
                vals = list(by_backend.values())
                speedup = vals[0]["median_ms"] / vals[1]["median_ms"] if vals[1]["median_ms"] > 0 else 0
                faster = list(by_backend.keys())[0] if speedup < 1 else list(by_backend.keys())[1]
                ratio = min(speedup, 1/speedup) if speedup > 0 else 0
                parts.append(f"  -> {faster} is {1/ratio:.2f}x faster" if ratio > 0 else "")
            print("\n".join(parts))
            print()

    # Write CSV
    if args.output and results:
        all_keys = []
        for r in results:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {args.output}")

    # GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import flashinfer
        print(f"FlashInfer: {flashinfer.__version__}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
