## A100 vs H100 NCU Comparison (L2 Sweep)

- Figure: `profiling/ncu_results/figure_ncu_h100_a100_side_by_side.png`
- Merged data: `profiling/ncu_results/h100_a100_l2_sweep_merged.csv`

### Key observations

- **FP16 DRAM trend (knee-consistent):**
  - H100 FP16 DRAM: below-50MB avg `55.5%` -> above-50MB avg `78.6%`
  - A100 FP16 DRAM: below-40MB avg `43.0%` -> above-40MB avg `66.2%`
- **INT4 remains SM-heavy on both GPUs:**
  - H100 INT4 SM: below-50MB avg `51.7%` -> above-50MB avg `73.8%`
  - A100 INT4 SM: below-40MB avg `51.2%` -> above-40MB avg `74.5%`
- **Mechanism consistency:** FP16 becomes increasingly DRAM-driven beyond each GPU's L2 boundary (50MB H100, 40MB A100), while INT4 remains dominated by SM-side dequant work.

### Suggested paper usage

Use the side-by-side figure as the primary cross-hardware counter evidence, and keep tables as appendix/detail.
