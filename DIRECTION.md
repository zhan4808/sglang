# A100 Profiling Direction

## Goal

Use the existing H100 methodology to produce the A100 comparison first, then move on to optimizer/NcuRunner work.

## Why this order

- H100 characterization is already done.
- A100 is a direct extension (same scripts, different hardware).
- It validates the causal L2 claim with a second architecture.
- It de-risks NCU/NcuRunner differences on `sm_80` before optimizer automation.

## Current machine

- GPU: `NVIDIA A100-SXM4-80GB`
- Relevant hardware fact: A100 L2 is 40 MB (both 40GB and 80GB variants).
- Expected crossover: near 40 MB effective weight size (knee should shift left vs H100 ~50 MB).

## Repo scope to keep

- `profiling/bench_l2_barrier.py`
- `profiling/bench_int4_bmm.py`
- `profiling/bench_l2_ncu_single.py`
- `profiling/analyze_ncu.py`
- `profiling/plot_hierarchical_roofline.py`
- Any tiny-gemm-derived kernel code currently used by these scripts.

## Execution sequence

1. Run A100 L2 barrier sweep (`8MB -> 128MB`) and collect `results_l2_barrier.{csv,json}`.
2. Run A100 INT4 vs FP16 comparison table via `bench_int4_bmm.py`.
3. Generate H100+A100 overlay figure (Figure 7 equivalent).
4. Validate NCU metric availability on A100 (`sm_80`) before full NcuRunner automation.
5. Start NcuRunner integration after the above is stable.

## Immediate commands

```bash
# from repo root
python profiling/bench_l2_barrier.py
python profiling/bench_int4_bmm.py
```

## A100 notes

- Some NCU metrics used on H100 (`sm_90`/TMA-specific) will be unavailable on A100 (`sm_80`).
- Keep common counters for cross-hardware claims: DRAM throughput, L2 hit rate, SM utilization.
- If needed, verify supported names before scripting:

```bash
ncu --query-metrics | rg -i "dram|l2|sm"
```

## Deliverables for this phase

- A100 `results_l2_barrier.csv/json`
- A100 `results_int4_bmm.csv`
- Overlay plot with both H100 and A100 curves
- Short write-up of knee shift and any absolute-latency differences (A100-80GB HBM bandwidth caveat)
