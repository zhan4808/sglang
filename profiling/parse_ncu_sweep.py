"""Parse NCU sweep CSV files (long format) into a summary table."""

import os
import csv
import json
import sys

outdir = sys.argv[1] if len(sys.argv) > 1 else "ncu_results/l2_sweep"

results = []

for fname in sorted(os.listdir(outdir)):
    if not fname.endswith(".csv"):
        continue
    parts = fname.replace(".csv", "").split("_")
    kernel = parts[0]
    d_lora = int(parts[1].replace("d", ""))

    filepath = os.path.join(outdir, fname)
    with open(filepath) as f:
        lines = f.readlines()

    # Find CSV header
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip().strip('"')
        if stripped.startswith("ID"):
            header_idx = i
            break
    if header_idx is None:
        print(f"  Warning: no CSV header in {fname}, skipping")
        continue

    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)
    if not rows:
        continue

    # Group by kernel invocation ID — each ID has multiple metric rows
    invocations = {}
    for row in rows:
        kid = row.get("ID", "").strip('"')
        metric = row.get("Metric Name", "").strip('"')
        value = row.get("Metric Value", "").strip('"').replace(",", "")
        kname = row.get("Kernel Name", "").strip('"')
        if kid and metric:
            if kid not in invocations:
                invocations[kid] = {"kernel_name": kname}
            try:
                invocations[kid][metric] = float(value)
            except ValueError:
                invocations[kid][metric] = value

    if not invocations:
        continue

    # Take the last invocation (most stable)
    last_id = max(invocations.keys(), key=int)
    m = invocations[last_id]

    wt_mb = 128 * 128 * d_lora * 2 / (1024 * 1024)

    l2_hit = m.get("lts__t_sectors_op_read_lookup_hit.sum")
    l2_miss = m.get("lts__t_sectors_op_read_lookup_miss.sum")
    l2_hit_rate = None
    if isinstance(l2_hit, (int, float)) and isinstance(l2_miss, (int, float)):
        total = l2_hit + l2_miss
        l2_hit_rate = round(l2_hit / total * 100, 1) if total > 0 else 0

    dram_read = m.get("dram__bytes_read.sum")
    dram_read_mb = round(dram_read / (1024*1024), 2) if isinstance(dram_read, (int, float)) else None

    entry = {
        "kernel": kernel,
        "kernel_name": m.get("kernel_name", ""),
        "d_lora": d_lora,
        "weight_mb": round(wt_mb, 1),
        "fits_l2": wt_mb < 50,
        "dram_pct": m.get("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        "sm_pct": m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "occupancy_pct": m.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "l2_hit_rate_pct": l2_hit_rate,
        "l2_hit_sectors": l2_hit,
        "l2_miss_sectors": l2_miss,
        "tensor_insts": m.get("smsp__inst_executed_pipe_tensor.sum"),
        "registers": m.get("launch__registers_per_thread"),
        "duration_ns": m.get("gpu__time_duration.sum"),
        "dram_read_mb": dram_read_mb,
    }
    results.append(entry)

# Sort by kernel type then d_lora
results.sort(key=lambda r: (r["kernel"], r["d_lora"]))

# Print summary
print("=" * 110)
print("NCU L2 SWEEP SUMMARY")
print("=" * 110)

for kernel in ["fp16", "int4"]:
    kr = [r for r in results if r["kernel"] == kernel]
    if not kr:
        continue

    kname = kr[0].get("kernel_name", "unknown")
    print(f"\n  {kernel.upper()} kernel ({kname}):")
    print(f"  {'d_lora':>6} {'WtMB':>6} {'L2?':>4} {'DRAM%':>6} {'SM%':>6} "
          f"{'Occ%':>6} {'L2hit%':>7} {'L2hit':>9} {'L2miss':>9} {'Regs':>5} "
          f"{'Dur(us)':>8} {'DRAMrd':>8}")
    print("  " + "-" * 100)
    for r in kr:
        def fmt(v, f=".1f"):
            return f"{v:{f}}" if isinstance(v, (int, float)) else "n/a"

        dur_us = f"{r['duration_ns']/1000:.1f}" if isinstance(r['duration_ns'], (int, float)) else "n/a"
        dram_rd = f"{r['dram_read_mb']:.1f}" if r['dram_read_mb'] is not None else "n/a"
        fits = "yes" if r['fits_l2'] else "NO"

        print(f"  {r['d_lora']:>6} {r['weight_mb']:>6.1f} {fits:>4} "
              f"{fmt(r['dram_pct']):>6} {fmt(r['sm_pct']):>6} "
              f"{fmt(r['occupancy_pct']):>6} {fmt(r['l2_hit_rate_pct']):>7} "
              f"{fmt(r['l2_hit_sectors'], '.0f'):>9} {fmt(r['l2_miss_sectors'], '.0f'):>9} "
              f"{fmt(r['registers'], '.0f'):>5} {dur_us:>8} {dram_rd:>8}")

# Cross-kernel comparison at key d_lora points
print(f"\n{'='*110}")
print("CROSS-KERNEL COMPARISON")
print(f"{'='*110}")
print(f"  {'d_lora':>6} {'WtMB':>6} {'L2?':>4} | "
      f"{'FP16 DRAM%':>10} {'INT4 DRAM%':>10} | "
      f"{'FP16 L2hit%':>11} {'INT4 L2hit%':>11} | "
      f"{'FP16 dur(us)':>12} {'INT4 dur(us)':>12} {'ratio':>6}")
print("  " + "-" * 105)

fp16_map = {r["d_lora"]: r for r in results if r["kernel"] == "fp16"}
int4_map = {r["d_lora"]: r for r in results if r["kernel"] == "int4"}

for d_lora in sorted(set(fp16_map.keys()) & set(int4_map.keys())):
    f = fp16_map[d_lora]
    i = int4_map[d_lora]
    wt_mb = f["weight_mb"]
    fits = "yes" if f["fits_l2"] else "NO"
    def fmt(v, f=".1f"):
        return f"{v:{f}}" if isinstance(v, (int, float)) else "n/a"

    f_dur = f["duration_ns"]/1000 if isinstance(f["duration_ns"], (int, float)) else None
    i_dur = i["duration_ns"]/1000 if isinstance(i["duration_ns"], (int, float)) else None
    ratio = f"{i_dur/f_dur:.2f}x" if f_dur and i_dur else "n/a"

    print(f"  {d_lora:>6} {wt_mb:>6.1f} {fits:>4} | "
          f"{fmt(f['dram_pct']):>10} {fmt(i['dram_pct']):>10} | "
          f"{fmt(f['l2_hit_rate_pct']):>11} {fmt(i['l2_hit_rate_pct']):>11} | "
          f"{fmt(f_dur):>12} {fmt(i_dur):>12} {ratio:>6}")

# Save
json_path = os.path.join(outdir, "ncu_sweep_summary.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nJSON saved to {json_path}")
