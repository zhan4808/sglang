"""
Parse cache-intervention results and generate paper-ready figures.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("/root/sglang/profiling")
RESULT_JSON = BASE / "results_cache_intervention_h100.json"
NCU_WARM = BASE / "ncu_results/cache_intervention_h100_warm.csv"
NCU_COLD = BASE / "ncu_results/cache_intervention_h100_cold.csv"

OUT_SUMMARY = BASE / "results_cache_intervention_h100_ncu_summary.json"
OUT_FIG_LAT = BASE / "figure_cache_intervention_h100_latency.png"
OUT_FIG_NCU = BASE / "figure_cache_intervention_h100_ncu.png"


def parse_ncu(path: Path, condition: str) -> list[dict]:
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.startswith('"')]
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = list(reader)

    kernel_map = {
        "fp16": ["nvjet_", "Kernel2<cutlass_", "enable_if<T7, void>::type kernel"],
        "int4": ["kernel_batched_w4a16_simple"],
    }
    out = []
    for ktype, pats in kernel_map.items():
        subset = [
            r
            for r in rows
            if any(p in r["Kernel Name"] for p in pats)
            and r["Metric Name"]
            in {
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "lts__t_sector_hit_rate.pct",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
            }
        ]
        by_metric: dict[str, list[float]] = {}
        for r in subset:
            by_metric.setdefault(r["Metric Name"], []).append(float(r["Metric Value"]))
        out.append(
            {
                "condition": condition,
                "kernel": ktype,
                "dram_pct": round(mean(by_metric.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", [0.0])), 4),
                "sm_pct": round(mean(by_metric.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", [0.0])), 4),
                "l2_hit_rate_pct": round(mean(by_metric.get("lts__t_sector_hit_rate.pct", [0.0])), 4),
                "warps_active_pct": round(mean(by_metric.get("sm__warps_active.avg.pct_of_peak_sustained_active", [0.0])), 4),
            }
        )
    return out


def make_latency_fig(results: list[dict]) -> None:
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=True)
    for ax, bs in zip(axes, [1, 4]):
        p = df[df["batch_size"] == bs].set_index("condition")
        labels = ["warm", "evict1x", "evict4x"]
        x = range(len(labels))
        fp = [p.loc[c, "fp16_norm_to_fp16_warm"] for c in labels]
        i4 = [p.loc[c, "int4_norm_to_fp16_warm"] for c in labels]
        ax.bar([i - 0.18 for i in x], fp, width=0.34, label="FP16", color="#1f77b4")
        ax.bar([i + 0.18 for i in x], i4, width=0.34, label="INT4", color="#ff7f0e")
        ax.set_xticks(list(x), labels)
        ax.set_title(f"BS={bs}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Latency normalized to FP16-warm")
    axes[0].legend()
    fig.suptitle("Controlled Cache Intervention (Fixed MLA Reconstruction Shape)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG_LAT, dpi=220, bbox_inches="tight")


def make_ncu_fig(summary_rows: list[dict]) -> None:
    df = pd.DataFrame(summary_rows)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    metrics = [("dram_pct", "DRAM %"), ("sm_pct", "SM %"), ("l2_hit_rate_pct", "L2 hit %")]
    for ax, (m, title) in zip(axes, metrics):
        p = df.pivot(index="condition", columns="kernel", values=m).loc[["warm", "evict4x"]]
        x = range(len(p.index))
        ax.bar([i - 0.18 for i in x], p["fp16"], width=0.34, color="#1f77b4", label="FP16")
        ax.bar([i + 0.18 for i in x], p["int4"], width=0.34, color="#ff7f0e", label="INT4")
        ax.set_xticks(list(x), list(p.index))
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("NCU Mechanism Check: Warm vs Forced Eviction", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_FIG_NCU, dpi=220, bbox_inches="tight")


def main() -> None:
    data = json.loads(RESULT_JSON.read_text())
    results = data["results"]
    ncu_summary = parse_ncu(NCU_WARM, "warm") + parse_ncu(NCU_COLD, "evict4x")

    OUT_SUMMARY.write_text(
        json.dumps(
            {
                "gpu": data["gpu"],
                "benchmark_results": results,
                "ncu_summary": ncu_summary,
            },
            indent=2,
        )
    )
    make_latency_fig(results)
    make_ncu_fig(ncu_summary)
    print(f"Saved {OUT_SUMMARY}")
    print(f"Saved {OUT_FIG_LAT}")
    print(f"Saved {OUT_FIG_NCU}")


if __name__ == "__main__":
    main()
