"""
Hierarchical Roofline Figure for MLA Reconstruction
=====================================================
Plots FP16 and INT4 kernel performance against both HBM and L2 roofline
ceilings, showing why INT4 fails to outperform FP16 at MLA's weight sizes.

Uses:
  - bench_l2_barrier.py timing results for achieved throughput
  - NCU data for DRAM/SM utilization context
  - H100 hardware specs for roofline ceilings

Outputs: roofline_hierarchical.png
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# H100 SXM5 specs
# ═══════════════════════════════════════════════════════════════════════════════

HBM_BW = 3.35e12      # 3.35 TB/s
L2_BW = 12e12          # ~12 TB/s effective L2 bandwidth
FP16_PEAK = 990e12     # 990 TFLOPS
L2_SIZE_MB = 50        # 50 MB L2 cache

# ═══════════════════════════════════════════════════════════════════════════════
# Load benchmark data
# ═══════════════════════════════════════════════════════════════════════════════

with open("results_l2_barrier.json") as f:
    barrier_data = json.load(f)

# Use BS=1 results (most relevant for decode latency)
bs1_results = [r for r in barrier_data["results"] if r["batch_size"] == 1]

H = 128
D_NOPE = 128
BS = 1

# ═══════════════════════════════════════════════════════════════════════════════
# Compute operational intensity and achieved FLOPS for each d_lora
# ═══════════════════════════════════════════════════════════════════════════════

# For batched BMM: [H, BS, K] @ [H, K, N]
# FLOPs per BMM = H * 2 * BS * K * N (2 for multiply-add)
# Bytes (FP16): weight = H * K * N * 2, activation = H * BS * K * 2, output = H * BS * N * 2
# Bytes (INT4): weight = H * K * N / 2 + H * N * 2 (packed + scales), activation same, output same

data_points = []
for r in bs1_results:
    d_lora = r["d_lora"]
    N = d_lora
    K = D_NOPE

    flops = H * 2 * BS * K * N

    # FP16 bytes
    fp16_weight_bytes = H * K * N * 2
    fp16_act_bytes = H * BS * K * 2
    fp16_out_bytes = H * BS * N * 2
    fp16_total_bytes = fp16_weight_bytes + fp16_act_bytes + fp16_out_bytes

    # INT4 bytes
    int4_weight_bytes = H * K * N // 2  # packed
    int4_scale_bytes = H * N * 2  # per-column scales
    int4_act_bytes = fp16_act_bytes
    int4_out_bytes = fp16_out_bytes
    int4_total_bytes = int4_weight_bytes + int4_scale_bytes + int4_act_bytes + int4_out_bytes

    # Operational intensity (FLOP/byte)
    fp16_oi = flops / fp16_total_bytes
    int4_oi = flops / int4_total_bytes

    # Achieved GFLOPS from timing
    fp16_gflops = flops / (r["fp16_ms"] * 1e-3) / 1e9
    int4_gflops = flops / (r["int4_ms"] * 1e-3) / 1e9

    weight_mb = r["weight_mb"]
    fits_l2 = r["fits_l2"]

    data_points.append({
        "d_lora": d_lora,
        "weight_mb": weight_mb,
        "fits_l2": fits_l2,
        "fp16_oi": fp16_oi,
        "int4_oi": int4_oi,
        "fp16_gflops": fp16_gflops,
        "int4_gflops": int4_gflops,
        "fp16_ms": r["fp16_ms"],
        "int4_ms": r["int4_ms"],
        "ratio": r["int4_fp16_ratio"],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Hierarchical Roofline
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Roofline ceilings
oi_range = np.logspace(-1, 3, 500)

# HBM roofline
hbm_roof = np.minimum(HBM_BW * oi_range, FP16_PEAK) / 1e9
ax.plot(oi_range, hbm_roof, 'k-', linewidth=2, label='HBM roofline (3.35 TB/s)', zorder=1)

# L2 roofline
l2_roof = np.minimum(L2_BW * oi_range, FP16_PEAK) / 1e9
ax.plot(oi_range, l2_roof, 'k--', linewidth=2, alpha=0.6, label='L2 roofline (~12 TB/s)', zorder=1)

# FP16 compute ceiling line
ax.axhline(y=FP16_PEAK/1e9, color='gray', linestyle=':', alpha=0.3, linewidth=1)

# Plot data points
fp16_l2 = [d for d in data_points if d["fits_l2"]]
fp16_hbm = [d for d in data_points if not d["fits_l2"]]

# FP16 points — L2 resident (hollow blue circles)
if fp16_l2:
    ax.scatter([d["fp16_oi"] for d in fp16_l2],
               [d["fp16_gflops"] for d in fp16_l2],
               s=80, facecolors='none', edgecolors='#2196F3', linewidths=2,
               zorder=5, label='FP16 cuBLAS (L2-resident)')

# FP16 points — HBM bound (filled blue circles)
if fp16_hbm:
    ax.scatter([d["fp16_oi"] for d in fp16_hbm],
               [d["fp16_gflops"] for d in fp16_hbm],
               s=80, c='#2196F3', marker='o', zorder=5,
               label='FP16 cuBLAS (HBM-bound)')

# INT4 points — L2 resident (hollow red triangles)
int4_l2 = [d for d in data_points if d["fits_l2"]]
int4_hbm = [d for d in data_points if not d["fits_l2"]]

if int4_l2:
    ax.scatter([d["int4_oi"] for d in int4_l2],
               [d["int4_gflops"] for d in int4_l2],
               s=80, facecolors='none', edgecolors='#F44336', linewidths=2,
               marker='^', zorder=5, label='INT4 Triton (L2-resident)')

if int4_hbm:
    ax.scatter([d["int4_oi"] for d in int4_hbm],
               [d["int4_gflops"] for d in int4_hbm],
               s=80, c='#F44336', marker='^', zorder=5,
               label='INT4 Triton (HBM-bound)')

# Annotate the MLA operating point (d_lora=512)
mla_point = next(d for d in data_points if d["d_lora"] == 512)
ax.annotate(f'MLA config\n(d_lora=512, {mla_point["weight_mb"]:.0f} MB)',
            xy=(mla_point["fp16_oi"], mla_point["fp16_gflops"]),
            xytext=(mla_point["fp16_oi"] * 3, mla_point["fp16_gflops"] * 2.5),
            arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5),
            fontsize=9, color='#2196F3', fontweight='bold')

# Annotate L2 boundary
boundary_fp16 = next(d for d in data_points if d["d_lora"] == 1792)
ax.axvline(x=boundary_fp16["fp16_oi"], color='gray', linestyle=':', alpha=0.4,
           ymin=0, ymax=0.4)
ax.text(boundary_fp16["fp16_oi"], 30, '← L2 boundary\n(50 MB)',
        fontsize=8, color='gray', ha='center', va='bottom')

# Annotate key d_lora values
for d in data_points:
    if d["d_lora"] in [256, 1536, 4096]:
        ax.annotate(f'{d["d_lora"]}', xy=(d["fp16_oi"], d["fp16_gflops"]),
                    xytext=(0, -15), textcoords='offset points',
                    fontsize=7, color='#2196F3', ha='center')
        ax.annotate(f'{d["d_lora"]}', xy=(d["int4_oi"], d["int4_gflops"]),
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=7, color='#F44336', ha='center')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Operational Intensity (FLOP/byte)', fontsize=12)
ax.set_ylabel('Achieved GFLOPS', fontsize=12)
ax.set_title('Hierarchical Roofline: MLA Reconstruction BMM on H100\n'
             'FP16 cuBLAS vs INT4 Triton across weight sizes', fontsize=13)
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.set_xlim(0.3, 200)
ax.set_ylim(10, 2e6)
ax.grid(True, alpha=0.2, which='both')

plt.tight_layout()
plt.savefig('roofline_hierarchical.png', dpi=150, bbox_inches='tight')
print("Saved: roofline_hierarchical.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: INT4/FP16 Ratio vs Weight Size (the money plot)
# ═══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(1, 1, figsize=(9, 5))

weight_mbs = [d["weight_mb"] for d in data_points]
ratios = [d["ratio"] for d in data_points]

# Color by L2 residency
colors = ['#2196F3' if d["fits_l2"] else '#F44336' for d in data_points]

ax2.scatter(weight_mbs, ratios, c=colors, s=100, zorder=5, edgecolors='black', linewidths=0.5)
ax2.plot(weight_mbs, ratios, 'k-', alpha=0.3, linewidth=1, zorder=3)

# L2 boundary
ax2.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.text(50, max(ratios) * 1.02, 'L2 capacity\n(50 MB)', ha='center', va='bottom',
         fontsize=9, color='gray')

# Parity line
ax2.axhline(y=1.0, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.text(max(weight_mbs) * 0.95, 1.02, 'INT4 = FP16', ha='right', va='bottom',
         fontsize=8, color='green', alpha=0.7)

# Regions
ax2.axvspan(0, 50, alpha=0.05, color='blue')
ax2.axvspan(50, max(weight_mbs) * 1.1, alpha=0.05, color='red')
ax2.text(25, min(ratios) * 0.97, 'L2-resident\n(FP16 served from L2)', ha='center',
         fontsize=8, color='#2196F3', style='italic')
ax2.text(89, min(ratios) * 0.97, 'HBM-bound\n(FP16 falls back to HBM)', ha='center',
         fontsize=8, color='#F44336', style='italic')

# Annotate MLA operating point
mla_idx = next(i for i, d in enumerate(data_points) if d["d_lora"] == 512)
ax2.annotate(f'MLA config\n(ratio={ratios[mla_idx]:.2f}x)',
             xy=(weight_mbs[mla_idx], ratios[mla_idx]),
             xytext=(weight_mbs[mla_idx] + 15, ratios[mla_idx] + 0.1),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             fontsize=9, fontweight='bold')

ax2.set_xlabel('FP16 Weight Size (MB)', fontsize=12)
ax2.set_ylabel('INT4/FP16 Time Ratio', fontsize=12)
ax2.set_title('L2 Cache Barrier: INT4/FP16 Performance Ratio vs Weight Size\n'
              'H100 SXM5, BS=1, H=128, K=128', fontsize=13)
ax2.set_xlim(0, max(weight_mbs) * 1.1)
ax2.set_ylim(min(ratios) * 0.9, max(ratios) * 1.1)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('ratio_vs_weight_size.png', dpi=150, bbox_inches='tight')
print("Saved: ratio_vs_weight_size.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: NCU DRAM utilization comparison
# ═══════════════════════════════════════════════════════════════════════════════

try:
    with open("ncu_results/l2_sweep/ncu_sweep_summary.json") as f:
        ncu_data = json.load(f)

    fp16_ncu = sorted([r for r in ncu_data if r["kernel"] == "fp16"], key=lambda r: r["d_lora"])
    int4_ncu = sorted([r for r in ncu_data if r["kernel"] == "int4"], key=lambda r: r["d_lora"])

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

    # DRAM utilization
    d_loras = [r["d_lora"] for r in fp16_ncu]
    fp16_dram = [r["dram_pct"] for r in fp16_ncu]
    int4_dram = [r["dram_pct"] for r in int4_ncu]
    fp16_sm = [r["sm_pct"] for r in fp16_ncu]
    int4_sm = [r["sm_pct"] for r in int4_ncu]
    wt_mbs_ncu = [r["weight_mb"] for r in fp16_ncu]

    ax3a.plot(wt_mbs_ncu, fp16_dram, 'o-', color='#2196F3', linewidth=2, markersize=8,
              label='FP16 DRAM%')
    ax3a.plot(wt_mbs_ncu, int4_dram, '^-', color='#F44336', linewidth=2, markersize=8,
              label='INT4 DRAM%')
    ax3a.plot(wt_mbs_ncu, fp16_sm, 's--', color='#2196F3', linewidth=1.5, markersize=6,
              alpha=0.6, label='FP16 SM%')
    ax3a.plot(wt_mbs_ncu, int4_sm, 'v--', color='#F44336', linewidth=1.5, markersize=6,
              alpha=0.6, label='INT4 SM%')
    ax3a.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    ax3a.text(50, 5, 'L2\nboundary', ha='center', fontsize=8, color='gray')
    ax3a.set_xlabel('FP16 Weight Size (MB)', fontsize=11)
    ax3a.set_ylabel('% of Peak', fontsize=11)
    ax3a.set_title('NCU: DRAM vs SM Utilization', fontsize=12)
    ax3a.legend(fontsize=9)
    ax3a.grid(True, alpha=0.2)
    ax3a.set_ylim(0, 100)

    # DRAM bytes read
    fp16_dram_mb = [r.get("dram_read_mb") for r in fp16_ncu]
    int4_dram_mb = [r.get("dram_read_mb") for r in int4_ncu]
    theoretical_fp16 = [128 * 128 * d * 2 / (1024*1024) for d in d_loras]
    theoretical_int4 = [128 * 128 * d / 2 / (1024*1024) for d in d_loras]  # packed only

    ax3b.plot(wt_mbs_ncu, fp16_dram_mb, 'o-', color='#2196F3', linewidth=2, markersize=8,
              label='FP16 actual DRAM read')
    ax3b.plot(wt_mbs_ncu, int4_dram_mb, '^-', color='#F44336', linewidth=2, markersize=8,
              label='INT4 actual DRAM read')
    ax3b.plot(wt_mbs_ncu, theoretical_fp16, '--', color='#2196F3', alpha=0.4,
              label='FP16 weight size')
    ax3b.plot(wt_mbs_ncu, theoretical_int4, '--', color='#F44336', alpha=0.4,
              label='INT4 packed weight size')
    ax3b.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    ax3b.set_xlabel('FP16 Weight Size (MB)', fontsize=11)
    ax3b.set_ylabel('DRAM Read (MB)', fontsize=11)
    ax3b.set_title('NCU: Actual DRAM Reads', fontsize=12)
    ax3b.legend(fontsize=9)
    ax3b.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('ncu_utilization.png', dpi=150, bbox_inches='tight')
    print("Saved: ncu_utilization.png")

except Exception as e:
    print(f"Skipping NCU figure: {e}")

print("\nAll figures generated.")
