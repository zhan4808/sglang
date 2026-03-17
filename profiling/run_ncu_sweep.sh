#!/bin/bash
# NCU counter sweep across d_lora sizes for both FP16 and INT4 kernels.
# Collects L2 hit rate, DRAM throughput, SM throughput, occupancy, and duration.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NCU=/usr/local/cuda/bin/ncu
PYTHON=python3
OUTDIR="$SCRIPT_DIR/ncu_results/l2_sweep"
mkdir -p "$OUTDIR"

METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
lts__t_sectors_op_read_lookup_hit.sum,\
lts__t_sectors_op_read_lookup_miss.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,\
smsp__inst_executed_pipe_tensor.sum,\
launch__registers_per_thread,\
launch__block_size,\
launch__grid_size,\
gpu__time_duration.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum"

D_LORAS="256 512 1024 1536 1792 2048 3072 4096"

echo "NCU L2 Sweep — $(date)"
echo "Metrics: $METRICS"
echo "d_lora values: $D_LORAS"
echo "Output: $OUTDIR"
echo ""

for kernel in fp16 int4; do
    for d_lora in $D_LORAS; do
        outfile="$OUTDIR/${kernel}_d${d_lora}.csv"
        echo ">>> Profiling $kernel d_lora=$d_lora ..."
        $NCU --metrics "$METRICS" \
            --profile-from-start no \
            --target-processes all \
            --csv \
            --log-file "$outfile" \
            $PYTHON "$SCRIPT_DIR/bench_l2_ncu_single.py" "$kernel" "$d_lora" \
            2>/dev/null
        echo "    Saved to $outfile"
    done
done

echo ""
echo "All NCU runs complete. Parsing results..."

# Parse all CSV files into a summary
$PYTHON - "$OUTDIR" <<'PYEOF'
import sys, os, csv, json

outdir = sys.argv[1]
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

    # Find the CSV header line (starts with "ID")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith('ID'):
            header_idx = i
            break
    if header_idx is None:
        print(f"  Warning: no CSV header in {fname}, skipping")
        continue

    reader = csv.DictReader(lines[header_idx:])
    rows = list(reader)
    if not rows:
        print(f"  Warning: no data rows in {fname}, skipping")
        continue

    # Take the last kernel invocation (most stable after warmup)
    row = rows[-1]

    def get_val(key):
        for k, v in row.items():
            if key in k:
                try:
                    return float(v.replace(",", "").replace("%", "").strip('"'))
                except (ValueError, AttributeError):
                    return v
        return None

    wt_mb = 128 * 128 * d_lora * 2 / (1024 * 1024)

    l2_hit = get_val("lts__t_sectors_op_read_lookup_hit.sum")
    l2_miss = get_val("lts__t_sectors_op_read_lookup_miss.sum")
    l2_hit_rate = None
    if l2_hit is not None and l2_miss is not None and isinstance(l2_hit, (int, float)) and isinstance(l2_miss, (int, float)):
        total = l2_hit + l2_miss
        l2_hit_rate = round(l2_hit / total * 100, 1) if total > 0 else 0

    entry = {
        "kernel": kernel,
        "d_lora": d_lora,
        "weight_mb": round(wt_mb, 1),
        "fits_l2": wt_mb < 50,
        "dram_pct": get_val("dram__throughput"),
        "sm_pct": get_val("sm__throughput"),
        "occupancy_pct": get_val("sm__warps_active"),
        "l2_hit_rate_pct": l2_hit_rate,
        "l2_hit_sectors": l2_hit,
        "l2_miss_sectors": l2_miss,
        "tensor_insts": get_val("smsp__inst_executed_pipe_tensor"),
        "registers": get_val("launch__registers_per_thread"),
        "duration_ns": get_val("gpu__time_duration"),
        "dram_read_bytes": get_val("dram__bytes_read"),
        "dram_write_bytes": get_val("dram__bytes_write"),
    }
    results.append(entry)

# Print summary table
print("\n" + "=" * 100)
print("NCU L2 SWEEP SUMMARY")
print("=" * 100)

for kernel in ["fp16", "int4"]:
    kr = [r for r in results if r["kernel"] == kernel]
    if not kr:
        continue
    print(f"\n  {kernel.upper()} kernel:")
    print(f"  {'d_lora':>6} {'WtMB':>6} {'L2?':>4} {'DRAM%':>6} {'SM%':>5} "
          f"{'Occ%':>5} {'L2hit%':>7} {'L2hit':>10} {'L2miss':>10} {'Regs':>5} "
          f"{'Dur(ns)':>10}")
    print("  " + "-" * 90)
    for r in kr:
        l2hr = f"{r['l2_hit_rate_pct']:.1f}" if r['l2_hit_rate_pct'] is not None else "n/a"
        dram = f"{r['dram_pct']:.1f}" if isinstance(r['dram_pct'], (int, float)) else str(r['dram_pct'])
        sm = f"{r['sm_pct']:.1f}" if isinstance(r['sm_pct'], (int, float)) else str(r['sm_pct'])
        occ = f"{r['occupancy_pct']:.1f}" if isinstance(r['occupancy_pct'], (int, float)) else str(r['occupancy_pct'])
        l2h = f"{r['l2_hit_sectors']:.0f}" if isinstance(r['l2_hit_sectors'], (int, float)) else str(r['l2_hit_sectors'])
        l2m = f"{r['l2_miss_sectors']:.0f}" if isinstance(r['l2_miss_sectors'], (int, float)) else str(r['l2_miss_sectors'])
        regs = f"{r['registers']:.0f}" if isinstance(r['registers'], (int, float)) else str(r['registers'])
        dur = f"{r['duration_ns']:.0f}" if isinstance(r['duration_ns'], (int, float)) else str(r['duration_ns'])
        fits = "yes" if r['fits_l2'] else "NO"
        print(f"  {r['d_lora']:>6} {r['weight_mb']:>6.1f} {fits:>4} {dram:>6} {sm:>5} "
              f"{occ:>5} {l2hr:>7} {l2h:>10} {l2m:>10} {regs:>5} {dur:>10}")

# Save JSON
json_path = os.path.join(outdir, "ncu_sweep_summary.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nJSON saved to {json_path}")
PYEOF

echo "Done."
