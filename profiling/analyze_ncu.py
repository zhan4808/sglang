"""
Parse Nsight Compute CSV output and generate a bottleneck analysis report.

Reads NCU --csv output, identifies:
  - Whether each kernel is compute-bound or memory-bound
  - SM utilization and occupancy
  - Memory bandwidth efficiency (% of peak)
  - L1/L2 cache hit rates
  - Warp scheduling efficiency

Usage:
    python analyze_ncu.py --csv ncu_output.csv --label "llama-8b_decode_bs64_kv2048"
    python analyze_ncu.py --csv ncu_output.csv --output analysis.md
"""

import argparse
import csv
import io
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional


def parse_ncu_csv(csv_path: str) -> List[Dict[str, str]]:
    """Parse NCU CSV output, handling the multi-header format."""
    rows = []
    headers = None

    with open(csv_path, "r") as f:
        content = f.read()

    # NCU CSV may have preamble lines starting with ==
    lines = []
    for line in content.split("\n"):
        if line.startswith("==") or line.strip() == "":
            continue
        lines.append(line)

    if not lines:
        print(f"WARNING: No data rows found in {csv_path}")
        return []

    reader = csv.reader(io.StringIO("\n".join(lines)))
    for row in reader:
        if headers is None:
            headers = [h.strip().strip('"') for h in row]
            continue
        if len(row) == len(headers):
            rows.append(dict(zip(headers, [v.strip().strip('"') for v in row])))

    return rows


def safe_float(val: str, default: float = 0.0) -> float:
    """Convert string to float, handling commas and units."""
    if not val or val == "n/a" or val == "N/A":
        return default
    val = val.replace(",", "").replace("%", "")
    # Handle units like "1.234 Kbyte"
    multipliers = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
    for suffix, mult in multipliers.items():
        if val.endswith(suffix):
            return float(val[:-1]) * mult
    try:
        return float(val)
    except ValueError:
        return default


def classify_bottleneck(sm_pct: float, mem_pct: float) -> str:
    """Classify kernel as compute-bound, memory-bound, or latency-bound.

    Uses the standard roofline heuristic:
    - If both SM and memory throughput are low (<10%), the kernel is launch-
      or sync-limited (latency-bound).
    - Otherwise, whichever subsystem is closer to its peak is the bottleneck.
    """
    if sm_pct < 10 and mem_pct < 10:
        return "LATENCY-BOUND"
    elif sm_pct > mem_pct * 1.3:
        return "COMPUTE-BOUND"
    elif mem_pct > sm_pct * 1.3:
        return "MEMORY-BOUND"
    else:
        return "BALANCED"


def analyze_kernels(rows: List[Dict[str, str]], label: str) -> str:
    """Generate markdown analysis from NCU data."""
    lines = []
    lines.append(f"# FlashInfer Attention Kernel Analysis: {label}")
    lines.append("")

    if not rows:
        lines.append("No kernel data found in NCU output.")
        return "\n".join(lines)

    # Group by kernel name
    kernel_data = defaultdict(list)
    for row in rows:
        name = row.get("Kernel Name", row.get("kernel_name", "unknown"))
        kernel_data[name].append(row)

    # Find the metric column names (NCU CSV columns vary by version)
    sample_row = rows[0]
    col_names = list(sample_row.keys())

    def find_col(patterns):
        for pat in patterns:
            for c in col_names:
                if pat.lower() in c.lower():
                    return c
        return None

    sm_col = find_col(["sm__throughput.avg.pct_of_peak", "Compute (SM) Throughput"])
    mem_col = find_col(["dram__throughput.avg.pct_of_peak", "Memory Throughput", "gpu__compute_memory_throughput"])
    occ_col = find_col(["launch__occupancy", "Achieved Occupancy"])
    warp_col = find_col(["sm__warps_active.avg.pct_of_peak", "Warp Occupancy"])
    reg_col = find_col(["launch__registers_per_thread", "Registers Per Thread"])
    dur_col = find_col(["gpu__time_duration.sum", "Duration"])
    block_col = find_col(["launch__block_size", "Block Size"])
    grid_col = find_col(["launch__grid_size", "Grid Size"])

    # L1 cache columns
    l1_hit_col = find_col(["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit"])
    l1_miss_col = find_col(["l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss"])

    # L2 cache columns
    l2_hit_col = find_col(["lts__t_sectors_op_read_lookup_hit"])
    l2_miss_col = find_col(["lts__t_sectors_op_read_lookup_miss"])

    # FP ops columns
    fadd_col = find_col(["sm__sass_thread_inst_executed_op_fadd"])
    fmul_col = find_col(["sm__sass_thread_inst_executed_op_fmul"])
    ffma_col = find_col(["sm__sass_thread_inst_executed_op_ffma"])

    # Tensor core columns
    tensor_pipe_col = find_col(["smsp__inst_executed_pipe_tensor"])
    tensor_op_col = find_col(["smsp__sass_thread_inst_executed_op_tensor"])

    lines.append("## Kernel Summary")
    lines.append("")
    lines.append(f"Total unique kernels: {len(kernel_data)}")
    lines.append("")

    # Sort kernels by total time (if duration available)
    kernel_times = []
    for name, invocations in kernel_data.items():
        total_dur = sum(safe_float(inv.get(dur_col, "0")) for inv in invocations) if dur_col else 0
        kernel_times.append((name, invocations, total_dur))
    kernel_times.sort(key=lambda x: -x[2])

    # ── Per-kernel analysis ──

    for rank, (name, invocations, total_dur) in enumerate(kernel_times, 1):
        # Use first invocation for representative metrics
        rep = invocations[0]

        sm_pct = safe_float(rep.get(sm_col, "0")) if sm_col else 0
        mem_pct = safe_float(rep.get(mem_col, "0")) if mem_col else 0
        occupancy = safe_float(rep.get(occ_col, "0")) if occ_col else 0
        warp_pct = safe_float(rep.get(warp_col, "0")) if warp_col else 0
        regs = safe_float(rep.get(reg_col, "0")) if reg_col else 0
        block_size = safe_float(rep.get(block_col, "0")) if block_col else 0
        grid_size = safe_float(rep.get(grid_col, "0")) if grid_col else 0

        bottleneck = classify_bottleneck(sm_pct, mem_pct)

        # Shorten kernel name for display
        short_name = name
        if len(name) > 80:
            short_name = name[:40] + "..." + name[-37:]

        lines.append(f"### #{rank}: `{short_name}`")
        lines.append("")
        lines.append(f"- **Invocations**: {len(invocations)}")
        if dur_col:
            dur_us = total_dur / 1000 if total_dur > 1e6 else total_dur
            lines.append(f"- **Total time**: {dur_us:.1f} us")
        lines.append(f"- **Classification**: **{bottleneck}**")
        lines.append(f"- SM throughput: {sm_pct:.1f}% of peak")
        lines.append(f"- Memory throughput: {mem_pct:.1f}% of peak")
        if occupancy > 0:
            lines.append(f"- Achieved occupancy: {occupancy:.1f}%")
        if warp_pct > 0:
            lines.append(f"- Active warps: {warp_pct:.1f}% of peak")
        if regs > 0:
            lines.append(f"- Registers/thread: {int(regs)}")
        if block_size > 0:
            lines.append(f"- Block size: {int(block_size)}")
        if grid_size > 0:
            lines.append(f"- Grid size: {int(grid_size)}")

        # L1 cache hit rate
        if l1_hit_col and l1_miss_col:
            l1_hits = safe_float(rep.get(l1_hit_col, "0"))
            l1_misses = safe_float(rep.get(l1_miss_col, "0"))
            if l1_hits + l1_misses > 0:
                l1_rate = 100 * l1_hits / (l1_hits + l1_misses)
                lines.append(f"- L1 cache hit rate: {l1_rate:.1f}%")

        # L2 cache hit rate
        if l2_hit_col and l2_miss_col:
            l2_hits = safe_float(rep.get(l2_hit_col, "0"))
            l2_misses = safe_float(rep.get(l2_miss_col, "0"))
            if l2_hits + l2_misses > 0:
                l2_rate = 100 * l2_hits / (l2_hits + l2_misses)
                lines.append(f"- L2 cache hit rate: {l2_rate:.1f}%")

        # Tensor core utilization
        tensor_insts = 0
        if tensor_pipe_col:
            tensor_insts = safe_float(rep.get(tensor_pipe_col, "0"))
        elif tensor_op_col:
            tensor_insts = safe_float(rep.get(tensor_op_col, "0"))
        if tensor_insts > 0:
            lines.append(f"- Tensor core instructions: {tensor_insts:,.0f}")

        # FP operation mix
        if fadd_col and fmul_col and ffma_col:
            fadd = safe_float(rep.get(fadd_col, "0"))
            fmul = safe_float(rep.get(fmul_col, "0"))
            ffma = safe_float(rep.get(ffma_col, "0"))
            total_fp = fadd + fmul + ffma
            if total_fp > 0:
                lines.append(f"- FP op mix: FMA {100*ffma/total_fp:.0f}%, ADD {100*fadd/total_fp:.0f}%, MUL {100*fmul/total_fp:.0f}%")
            # Ratio of tensor core vs CUDA core work
            if tensor_insts > 0 and total_fp > 0:
                tc_ratio = tensor_insts / (tensor_insts + total_fp)
                lines.append(f"- Tensor core vs CUDA core ratio: {100*tc_ratio:.1f}% tensor core")

        # Bottleneck-specific observations
        lines.append("")
        if bottleneck == "MEMORY-BOUND":
            lines.append("> **Observation**: This kernel is memory-bandwidth limited. "
                        "Performance scales with memory bandwidth, not compute. "
                        "Optimization strategies: reduce memory traffic (quantization, "
                        "compression), improve cache locality, or use tensor cores more.")
        elif bottleneck == "COMPUTE-BOUND":
            lines.append("> **Observation**: This kernel is compute-limited. "
                        "Consider: reducing FLOPs (e.g., smaller head dims, approximate attention), "
                        "increasing tensor core utilization, or using FP8/INT8 compute.")
        elif bottleneck == "LATENCY-BOUND":
            lines.append("> **Observation**: This kernel has low utilization of both compute and memory. "
                        "Likely launch overhead, synchronization, or small problem size. "
                        "Consider: kernel fusion, increasing batch size, or reducing launch overhead.")
        lines.append("")

    # ── Summary section ──

    lines.append("## Overall Assessment")
    lines.append("")

    # Classify all kernels
    bound_counts = defaultdict(int)
    for name, invocations, total_dur in kernel_times:
        rep = invocations[0]
        sm_pct = safe_float(rep.get(sm_col, "0")) if sm_col else 0
        mem_pct = safe_float(rep.get(mem_col, "0")) if mem_col else 0
        bound_counts[classify_bottleneck(sm_pct, mem_pct)] += 1

    for bt, count in sorted(bound_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {bt}: {count} kernels")
    lines.append("")

    # Key takeaways
    lines.append("## Key Metrics to Report")
    lines.append("")
    lines.append("When writing up the analysis, focus on:")
    lines.append("1. **Decode attention**: Is it memory-bound? What % of peak DRAM bandwidth?")
    lines.append("2. **Prefill attention**: Is it compute-bound? What % of peak TFLOPS?")
    lines.append("3. **Occupancy**: Low occupancy often means register pressure or shared memory limits")
    lines.append("4. **L2 cache**: KV cache access patterns - is L2 helping or thrashing?")
    lines.append("5. **MLA vs GQA**: Does MLA's compressed KV actually reduce memory traffic?")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze NCU profiling output")
    parser.add_argument("--csv", required=True, help="NCU CSV output file")
    parser.add_argument("--label", default="analysis", help="Label for the report")
    parser.add_argument("--output", default=None, help="Output markdown file (default: stdout)")
    args = parser.parse_args()

    rows = parse_ncu_csv(args.csv)
    report = analyze_kernels(rows, args.label)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Analysis written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
