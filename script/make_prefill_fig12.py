#!/usr/bin/env python3
"""Figure 12 regenerated — prefill TTFT, H100, 4 models (same as decode fig).

Faithful to the paper's Fig 12 structure: per-model panels, x-axis =
workload, three bars = batch {256, 512, 1024}, y = prefill TTFT (seconds).
CLoRA only (the paper's Fig 12 is CLoRA prefill under different settings).

Data: script/results_prefill.json (H100). Out: fig/fig_prefill_fig12.{png,pdf}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "fig")
os.makedirs(FIG, exist_ok=True)

MODELS = ["Llama2-7B", "Llama2-13B", "Llama3-8B", "Qwen3-30B"]
TAG = {"Llama2-7B": "Llama2-7B (MHA)", "Llama2-13B": "Llama2-13B (MHA)",
       "Llama3-8B": "Llama3-8B (GQA)", "Qwen3-30B": "Qwen3-30B (GQA+MoE)"}
WL = ["Uniform", "Uniform-long", "Skewed", "Skewed-long"]
WLAB = {"Uniform": "Unif", "Uniform-long": "Unif-L",
        "Skewed": "Skew", "Skewed-long": "Skew-L"}
BATCHES = [("256", "#9ecae1"), ("512", "#4292c6"), ("1024", "#08519c")]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def main():
    d = json.load(open(os.path.join(HERE, "results_prefill.json")))
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.6), sharey=True)
    x = np.arange(len(WL))
    width = 0.26
    for ax, m in zip(axes, MODELS):
        for i, (b, color) in enumerate(BATCHES):
            vals = [d[m][w][b]["CLoRA"] for w in WL]
            bars = ax.bar(x + (i - 1) * width, vals, width, label=f"batch {b}",
                          color=color, edgecolor="black", linewidth=0.4)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v * 1.03,
                        f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center",
                        va="bottom", fontsize=6, rotation=90)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([WLAB[w] for w in WL], fontsize=8)
        ax.set_title(TAG[m])
        ax.grid(axis="y", which="both", alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_ylim(top=ax.get_ylim()[1] * 2.0)
    axes[0].set_ylabel("Prefill TTFT (s)  ↓ lower is better")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04),
               frameon=False, fontsize=9)
    fig.suptitle("Prefill TTFT (time-to-first-token) on H100, CLoRA, "
                 "1000 adapters", y=1.10, fontsize=9.5, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_prefill_fig12.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote fig/fig_prefill_fig12.png/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
