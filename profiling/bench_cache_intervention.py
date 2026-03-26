"""
Controlled L2 residency intervention for MLA reconstruction GEMMs.

Core protocol (fixed shape):
  H=128, D_NOPE=128, D_LORA=512, BS in {1,4}
  Kernels: FP16 torch.bmm vs INT4 Triton W4A16
  Conditions: warm, evict1x, evict4x

Benchmark mode output:
  profiling/results_cache_intervention_<gpu>.json

NCU mode usage example:
  python3 profiling/bench_cache_intervention.py --mode ncu --condition warm --kernel both --bs 1
  python3 profiling/bench_cache_intervention.py --mode ncu --condition evict4x --kernel both --bs 1
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, List

import torch

sys.path.append(os.path.dirname(__file__))
from bench_l2_barrier import batched_int4_gemm, quantize_weights_int4


H = 128
D_NOPE = 128
D_LORA = 512
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class EvictionConfig:
    l2_mb: int
    evict1x_mb: int
    evict4x_mb: int


def get_eviction_config(gpu_name: str) -> EvictionConfig:
    u = gpu_name.upper()
    if "H100" in u:
        return EvictionConfig(l2_mb=50, evict1x_mb=64, evict4x_mb=256)
    if "A100" in u:
        return EvictionConfig(l2_mb=40, evict1x_mb=48, evict4x_mb=192)
    # Conservative fallback.
    return EvictionConfig(l2_mb=40, evict1x_mb=64, evict4x_mb=256)


def alloc_evict_buffer(size_mb: int, device: str) -> torch.Tensor:
    elems = (size_mb * 1024 * 1024) // 2  # float16 elements
    return torch.randn(elems, dtype=torch.float16, device=device)


def evict_l2(buf: torch.Tensor) -> None:
    # Read+write full buffer; unrelated to reconstruction weights.
    buf.add_(0.001)


def run_fp16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.bmm(x, w)


def run_int4(x: torch.Tensor, w_packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return batched_int4_gemm(x, w_packed, scales, D_NOPE, BLOCK_M=16, BLOCK_N=64, BLOCK_K=128)


def benchmark_condition(
    kernel: str,
    condition: str,
    bs: int,
    warmup: int,
    iters: int,
    evict1x_buf: torch.Tensor,
    evict4x_buf: torch.Tensor,
) -> float:
    x = torch.randn(H, bs, D_NOPE, dtype=torch.float16, device="cuda")
    w = torch.randn(H, D_NOPE, D_LORA, dtype=torch.float16, device="cuda")
    w_packed, scales = quantize_weights_int4(w)

    def run_once() -> None:
        if kernel == "fp16":
            run_fp16(x, w)
        else:
            run_int4(x, w_packed, scales)

    # Warm condition explicitly primes L2 with same kernel/weights.
    for _ in range(warmup):
        if condition == "evict1x":
            evict_l2(evict1x_buf)
        elif condition == "evict4x":
            evict_l2(evict4x_buf)
        run_once()
    torch.cuda.synchronize()

    samples: List[float] = []
    for _ in range(iters):
        if condition == "evict1x":
            evict_l2(evict1x_buf)
            torch.cuda.synchronize()
        elif condition == "evict4x":
            evict_l2(evict4x_buf)
            torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run_once()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))

    return statistics.median(samples)


def run_benchmark(args: argparse.Namespace) -> None:
    gpu_name = torch.cuda.get_device_name(0)
    cfg = get_eviction_config(gpu_name)
    evict1x_buf = alloc_evict_buffer(cfg.evict1x_mb, "cuda")
    evict4x_buf = alloc_evict_buffer(cfg.evict4x_mb, "cuda")

    print(f"GPU: {gpu_name}")
    print(f"L2={cfg.l2_mb}MB evict1x={cfg.evict1x_mb}MB evict4x={cfg.evict4x_mb}MB")
    print(f"Fixed shape: H={H}, D_NOPE={D_NOPE}, D_LORA={D_LORA}")

    results: List[Dict] = []
    for bs in [1, 4]:
        for condition in ["warm", "evict1x", "evict4x"]:
            fp16_ms = benchmark_condition(
                "fp16", condition, bs, args.warmup, args.iters, evict1x_buf, evict4x_buf
            )
            int4_ms = benchmark_condition(
                "int4", condition, bs, args.warmup, args.iters, evict1x_buf, evict4x_buf
            )
            ratio = int4_ms / fp16_ms
            row = {
                "batch_size": bs,
                "condition": condition,
                "fp16_ms": round(fp16_ms, 6),
                "int4_ms": round(int4_ms, 6),
                "int4_fp16_ratio": round(ratio, 6),
            }
            results.append(row)
            print(
                f"BS={bs:>2} {condition:>7}: "
                f"FP16={fp16_ms:.4f}ms INT4={int4_ms:.4f}ms ratio={ratio:.3f}x"
            )

    # Normalize to FP16 warm baseline by BS.
    for bs in [1, 4]:
        base = next(r for r in results if r["batch_size"] == bs and r["condition"] == "warm")["fp16_ms"]
        for r in results:
            if r["batch_size"] == bs:
                r["fp16_norm_to_fp16_warm"] = round(r["fp16_ms"] / base, 6)
                r["int4_norm_to_fp16_warm"] = round(r["int4_ms"] / base, 6)

    out_name = f"results_cache_intervention_{'h100' if 'H100' in gpu_name.upper() else 'a100'}.json"
    out_path = os.path.join(BASE_DIR, out_name)
    with open(out_path, "w") as f:
        json.dump(
            {
                "gpu": gpu_name,
                "l2_mb": cfg.l2_mb,
                "evict1x_mb": cfg.evict1x_mb,
                "evict4x_mb": cfg.evict4x_mb,
                "fixed_shape": {"H": H, "D_NOPE": D_NOPE, "D_LORA": D_LORA},
                "warmup": args.warmup,
                "iters": args.iters,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved {out_path}")


def run_ncu_target(args: argparse.Namespace) -> None:
    gpu_name = torch.cuda.get_device_name(0)
    cfg = get_eviction_config(gpu_name)
    evict1x_buf = alloc_evict_buffer(cfg.evict1x_mb, "cuda")
    evict4x_buf = alloc_evict_buffer(cfg.evict4x_mb, "cuda")

    bs = args.bs
    x = torch.randn(H, bs, D_NOPE, dtype=torch.float16, device="cuda")
    w = torch.randn(H, D_NOPE, D_LORA, dtype=torch.float16, device="cuda")
    w_packed, scales = quantize_weights_int4(w)

    def maybe_evict() -> None:
        if args.condition == "evict1x":
            evict_l2(evict1x_buf)
            torch.cuda.synchronize()
        elif args.condition == "evict4x":
            evict_l2(evict4x_buf)
            torch.cuda.synchronize()

    def run_fp16_only() -> None:
        maybe_evict()
        run_fp16(x, w)

    def run_int4_only() -> None:
        maybe_evict()
        run_int4(x, w_packed, scales)

    # Prime warm state without profiling.
    for _ in range(20):
        if args.kernel in ("fp16", "both"):
            run_fp16(x, w)
        if args.kernel in ("int4", "both"):
            run_int4(x, w_packed, scales)
    torch.cuda.synchronize()

    # For cold condition, evict right before the single profiled kernel.
    if args.condition in ("evict1x", "evict4x"):
        maybe_evict()

    # Single profiled launch to preserve cache intervention semantics.
    torch.cuda.cudart().cudaProfilerStart()
    if args.kernel in ("fp16", "both"):
        run_fp16(x, w)
    if args.kernel in ("int4", "both"):
        run_int4(x, w_packed, scales)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["benchmark", "ncu"], default="benchmark")
    p.add_argument("--kernel", choices=["fp16", "int4", "both"], default="both")
    p.add_argument("--condition", choices=["warm", "evict1x", "evict4x"], default="warm")
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "benchmark":
        run_benchmark(args)
    else:
        run_ncu_target(args)
