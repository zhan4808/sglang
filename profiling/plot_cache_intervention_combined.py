"""
Generate combined H100+A100 cache-intervention figure (normalized bars).

Outputs:
  - profiling/results_cache_intervention_combined_bs1.json
  - profiling/paper/figures/cache_intervention_h100_a100_bs1.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
H100_PATH = BASE / "results_cache_intervention_h100.json"
A100_PATH = BASE / "results_cache_intervention_a100.json"

OUT_JSON = BASE / "results_cache_intervention_combined_bs1.json"
OUT_PNG = BASE / "paper" / "figures" / "cache_intervention_h100_a100_bs1.png"


def load_norm_values(path: Path, batch_size: int = 1) -> dict[str, float]:
    data = json.loads(path.read_text())
    rows = [r for r in data["results"] if r["batch_size"] == batch_size]
    row_by_cond = {r["condition"]: r for r in rows}
    return {
        "fp16_warm": row_by_cond["warm"]["fp16_norm_to_fp16_warm"],
        "fp16_cold": row_by_cond["evict4x"]["fp16_norm_to_fp16_warm"],
        "int4_warm": row_by_cond["warm"]["int4_norm_to_fp16_warm"],
        "int4_cold": row_by_cond["evict4x"]["int4_norm_to_fp16_warm"],
    }


def main() -> None:
    h100 = load_norm_values(H100_PATH, batch_size=1)
    a100 = load_norm_values(A100_PATH, batch_size=1)

    combined = {"h100": h100, "a100": a100}
    OUT_JSON.write_text(json.dumps(combined, indent=2))

    labels = ["FP16 warm", "FP16 cold", "INT4 warm", "INT4 cold"]
    h_vals = [h100["fp16_warm"], h100["fp16_cold"], h100["int4_warm"], h100["int4_cold"]]
    a_vals = [a100["fp16_warm"], a100["fp16_cold"], a100["int4_warm"], a100["int4_cold"]]
    colors = ["#1f77b4", "#7fb6e6", "#ff7f0e", "#f4b37a"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, vals, title in zip(axes, [h_vals, a_vals], ["H100", "A100"]):
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_ylim(0.9, 2.1)
        ax.grid(True, axis="y", alpha=0.25)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.tick_params(axis="x", labelrotation=20)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("Latency normalized to FP16 warm")
    fig.suptitle("Controlled cache-residency intervention (fixed MLA shape, BS=1)")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")

    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
