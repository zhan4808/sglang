"""
Generate publication-style H100 vs A100 L2-barrier figures and tables.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


H100_JSON = Path("/root/sglang/profiling/results_l2_barrier.json")
A100_JSON = Path("/root/sglang/profiling/results_l2_barrier_a100_extended.json")
OUT_FIG = Path("/root/sglang/profiling/figure7_h100_a100_l2_barrier_extended.png")
OUT_TABLE = Path("/root/sglang/profiling/table_bs1_absolute_latency_h100_a100.csv")


def load(path: Path, label: str) -> pd.DataFrame:
    with path.open() as f:
        obj = json.load(f)
    df = pd.DataFrame(obj["results"])
    df["gpu_label"] = label
    return df


def main() -> None:
    h100 = load(H100_JSON, "H100 SXM5")
    a100 = load(A100_JSON, "A100 SXM4")
    df = pd.concat([h100, a100], ignore_index=True)

    # Figure: 2x2 panels for BS 1/4/16/64
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), sharex=True, sharey=True)
    axes = axes.ravel()
    colors = {"H100 SXM5": "#1f77b4", "A100 SXM4": "#ff7f0e"}

    for ax, bs in zip(axes, [1, 4, 16, 64]):
        sdf = df[df["batch_size"] == bs].sort_values("weight_mb")
        for label in ["H100 SXM5", "A100 SXM4"]:
            p = sdf[sdf["gpu_label"] == label]
            if not p.empty:
                ax.plot(
                    p["weight_mb"],
                    p["int4_fp16_ratio"],
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    color=colors[label],
                    label=label,
                )
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        ax.axvline(40, linestyle=":", color=colors["A100 SXM4"], linewidth=1)
        ax.axvline(50, linestyle=":", color=colors["H100 SXM5"], linewidth=1)
        ax.set_title(f"BS={bs}")
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="upper right", fontsize=9)
    for ax in axes[2:]:
        ax.set_xlabel("Weight size (MB)")
    axes[0].set_ylabel("INT4 / FP16 latency ratio")
    axes[2].set_ylabel("INT4 / FP16 latency ratio")
    fig.suptitle("H100 vs A100 L2-Barrier Shift Across Batch Sizes", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    print(f"Saved {OUT_FIG}")

    # Table: BS=1 absolute latencies side-by-side
    t = (
        df[df["batch_size"] == 1][
            ["gpu_label", "d_lora", "weight_mb", "fp16_ms", "int4_ms", "int4_fp16_ratio"]
        ]
        .sort_values(["d_lora", "gpu_label"])
        .reset_index(drop=True)
    )
    pivot = t.pivot(
        index=["d_lora", "weight_mb"],
        columns="gpu_label",
        values=["fp16_ms", "int4_ms", "int4_fp16_ratio"],
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    pivot.to_csv(OUT_TABLE, index=False)
    print(f"Saved {OUT_TABLE}")


if __name__ == "__main__":
    main()
