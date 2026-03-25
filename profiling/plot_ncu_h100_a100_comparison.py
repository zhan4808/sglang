"""
Compare H100 vs A100 NCU L2-sweep counters with figure + merged CSV + note.

Inputs:
  - H100: profiling/ncu_results/l2_sweep/ncu_sweep_summary.json
  - A100: profiling/ncu_results/a100_l2_sweep_ncu.csv
Outputs:
  - profiling/ncu_results/figure_ncu_h100_a100_side_by_side.png
  - profiling/ncu_results/h100_a100_l2_sweep_merged.csv
  - profiling/ncu_results/h100_a100_comparison_note.md
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


H100_PATH = Path("/root/sglang/profiling/ncu_results/l2_sweep/ncu_sweep_summary.json")
A100_PATH = Path("/root/sglang/profiling/ncu_results/a100_l2_sweep_ncu.csv")
OUT_FIG = Path("/root/sglang/profiling/ncu_results/figure_ncu_h100_a100_side_by_side.png")
OUT_CSV = Path("/root/sglang/profiling/ncu_results/h100_a100_l2_sweep_merged.csv")
OUT_NOTE = Path("/root/sglang/profiling/ncu_results/h100_a100_comparison_note.md")


def load_h100() -> pd.DataFrame:
    rows = json.loads(H100_PATH.read_text())
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "l2_hit_rate_pct": "l2_hit_rate",
            "occupancy_pct": "warps_active_pct",
            "registers": "registers_per_thread",
        }
    )
    df["duration_ms"] = df["duration_ns"] / 1e6
    df["gpu"] = "H100"
    return df[
        [
            "gpu",
            "kernel",
            "weight_mb",
            "dram_pct",
            "sm_pct",
            "l2_hit_rate",
            "warps_active_pct",
            "registers_per_thread",
            "duration_ms",
        ]
    ].copy()


def load_a100() -> pd.DataFrame:
    df = pd.read_csv(A100_PATH)
    df = df[df["status"] == "ok"].copy()
    df["gpu"] = "A100"
    return df[
        [
            "gpu",
            "kernel",
            "weight_mb",
            "dram_pct",
            "sm_pct",
            "l2_hit_rate",
            "warps_active_pct",
            "registers_per_thread",
            "duration_ms",
        ]
    ].copy()


def make_plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex=True)
    axes = axes.ravel()

    styles = {
        ("H100", "fp16"): dict(color="#1f77b4", marker="o", linestyle="-"),
        ("A100", "fp16"): dict(color="#ff7f0e", marker="o", linestyle="-"),
        ("H100", "int4"): dict(color="#1f77b4", marker="s", linestyle="--"),
        ("A100", "int4"): dict(color="#ff7f0e", marker="s", linestyle="--"),
    }

    metrics = [
        ("dram_pct", "DRAM Throughput (% peak)"),
        ("sm_pct", "SM Throughput (% peak)"),
        ("l2_hit_rate", "L2 Hit Rate (%)"),
        ("duration_ms", "Kernel Duration (ms)"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for gpu in ["H100", "A100"]:
            for kernel in ["fp16", "int4"]:
                p = df[(df["gpu"] == gpu) & (df["kernel"] == kernel)].sort_values("weight_mb")
                st = styles[(gpu, kernel)]
                ax.plot(
                    p["weight_mb"],
                    p[metric],
                    label=f"{gpu} {kernel.upper()}",
                    linewidth=1.8,
                    markersize=4,
                    **st,
                )
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.axvline(40, linestyle=":", color="#ff7f0e", linewidth=1)
        ax.axvline(50, linestyle=":", color="#1f77b4", linewidth=1)

    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    axes[2].set_xlabel("Weight size (MB)")
    axes[3].set_xlabel("Weight size (MB)")
    fig.suptitle("A100 vs H100 NCU Counter Trends Across L2 Boundary", y=1.01)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")


def write_note(df: pd.DataFrame) -> None:
    def mean_for(gpu: str, kernel: str, metric: str, below: float, above: float) -> tuple[float, float]:
        p = df[(df["gpu"] == gpu) & (df["kernel"] == kernel)]
        lo = p[p["weight_mb"] < below][metric].mean()
        hi = p[p["weight_mb"] > above][metric].mean()
        return float(lo), float(hi)

    h_fp16_d = mean_for("H100", "fp16", "dram_pct", 50, 50)
    a_fp16_d = mean_for("A100", "fp16", "dram_pct", 40, 40)
    h_int4_s = mean_for("H100", "int4", "sm_pct", 50, 50)
    a_int4_s = mean_for("A100", "int4", "sm_pct", 40, 40)

    txt = f"""## A100 vs H100 NCU Comparison (L2 Sweep)

- Figure: `profiling/ncu_results/figure_ncu_h100_a100_side_by_side.png`
- Merged data: `profiling/ncu_results/h100_a100_l2_sweep_merged.csv`

### Key observations

- **FP16 DRAM trend (knee-consistent):**
  - H100 FP16 DRAM: below-50MB avg `{h_fp16_d[0]:.1f}%` -> above-50MB avg `{h_fp16_d[1]:.1f}%`
  - A100 FP16 DRAM: below-40MB avg `{a_fp16_d[0]:.1f}%` -> above-40MB avg `{a_fp16_d[1]:.1f}%`
- **INT4 remains SM-heavy on both GPUs:**
  - H100 INT4 SM: below-50MB avg `{h_int4_s[0]:.1f}%` -> above-50MB avg `{h_int4_s[1]:.1f}%`
  - A100 INT4 SM: below-40MB avg `{a_int4_s[0]:.1f}%` -> above-40MB avg `{a_int4_s[1]:.1f}%`
- **Mechanism consistency:** FP16 becomes increasingly DRAM-driven beyond each GPU's L2 boundary (50MB H100, 40MB A100), while INT4 remains dominated by SM-side dequant work.

### Suggested paper usage

Use the side-by-side figure as the primary cross-hardware counter evidence, and keep tables as appendix/detail.
"""
    OUT_NOTE.write_text(txt)


def main() -> None:
    h100 = load_h100()
    a100 = load_a100()
    merged = pd.concat([h100, a100], ignore_index=True).sort_values(["gpu", "kernel", "weight_mb"])
    merged.to_csv(OUT_CSV, index=False)
    make_plot(merged)
    write_note(merged)
    print(f"Saved {OUT_FIG}")
    print(f"Saved {OUT_CSV}")
    print(f"Saved {OUT_NOTE}")


if __name__ == "__main__":
    main()
