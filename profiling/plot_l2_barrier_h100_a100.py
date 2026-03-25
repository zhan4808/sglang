"""
Create combined H100 + A100 L2-barrier comparison plot.

Inputs:
  - H100 JSON: /root/sglang/profiling/results_l2_barrier.json
  - A100 JSON: /root/sglang/results_l2_barrier.json

Output:
  - /root/sglang/profiling/figure7_h100_a100_l2_barrier.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


H100_PATH = Path("/root/sglang/profiling/results_l2_barrier.json")
A100_PATH = Path("/root/sglang/results_l2_barrier.json")
OUT_PATH = Path("/root/sglang/profiling/figure7_h100_a100_l2_barrier.png")


def load_rows(path: Path, label: str) -> pd.DataFrame:
    with path.open() as f:
        obj = json.load(f)
    df = pd.DataFrame(obj["results"])
    df["gpu_label"] = label
    return df


def main() -> None:
    h100 = load_rows(H100_PATH, "H100 SXM5 (50MB L2)")
    a100 = load_rows(A100_PATH, "A100 SXM4 (40MB L2)")
    df = pd.concat([h100, a100], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = {
        "H100 SXM5 (50MB L2)": "#1f77b4",
        "A100 SXM4 (40MB L2)": "#ff7f0e",
    }

    for i, bs in enumerate([1, 4]):
        ax = axes[i]
        sdf = df[df["batch_size"] == bs].copy()

        for label in ["H100 SXM5 (50MB L2)", "A100 SXM4 (40MB L2)"]:
            part = sdf[sdf["gpu_label"] == label].sort_values("weight_mb")
            ax.plot(
                part["weight_mb"],
                part["int4_fp16_ratio"],
                marker="o",
                linewidth=2,
                markersize=4,
                label=label,
                color=colors[label],
            )

        ax.axhline(1.0, linestyle="--", linewidth=1, color="gray")
        ax.axvline(40, linestyle=":", linewidth=1, color=colors["A100 SXM4 (40MB L2)"])
        ax.axvline(50, linestyle=":", linewidth=1, color=colors["H100 SXM5 (50MB L2)"])
        ax.set_title(f"BS={bs}")
        ax.set_xlabel("Weight size (MB)")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("INT4 / FP16 latency ratio (<1 is INT4 faster)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Figure 7 Equivalent: L2 Cache Barrier Shift (H100 vs A100)", y=1.02)
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")

    # quick sanity summary
    for label, l2_mb in [("H100 SXM5 (50MB L2)", 50), ("A100 SXM4 (40MB L2)", 40)]:
        part = df[(df["gpu_label"] == label) & (df["batch_size"] == 1)]
        below = part[part["weight_mb"] < l2_mb]["int4_fp16_ratio"].mean()
        above = part[part["weight_mb"] > l2_mb]["int4_fp16_ratio"].mean()
        print(f"{label}: below={below:.3f} above={above:.3f}")


if __name__ == "__main__":
    main()
