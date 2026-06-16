#!/usr/bin/env python3
"""Prefill TTFT (paper Fig 12), regenerated on H100 — CLoRA across configs.

NOT a system comparison. Shows CLoRA prefill TTFT as it varies across
models x workloads x batch sizes:
  * 4 model panels (Llama2-7B/13B, Llama3-8B, Qwen3-30B)
  * x = 4 workloads (Uniform / Uniform-long / Skewed / Skewed-long)
  * grouped bars = batch {256, 512, 1024}
Y = TTFT (s), log, lower is better. H100, GPU forward at 75% MFU.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (MODEL_TAGS, TTFT_LABEL, panel_caption,  # noqa: E402
                    clean_axis, save)
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "script")
MODELS = ["Llama2-7B", "Llama2-13B", "Llama3-8B", "Qwen3-30B"]
WL = ["Uniform", "Uniform-long", "Skewed", "Skewed-long"]
WLAB = {"Uniform": "Unif", "Uniform-long": "Unif-L",
        "Skewed": "Skew", "Skewed-long": "Skew-L"}
# batch -> (label, color)
BATCHES = [("256", "B=256", "#9ecae1"),
           ("512", "B=512", "#4292c6"),
           ("1024", "B=1024", "#084594")]


def main():
    p = json.load(open(os.path.join(SCRIPT, "results_prefill.json")))

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0), sharey=True)
    x = np.arange(len(WL))
    n = len(BATCHES)
    width = 0.26
    for idx, (ax, m) in enumerate(zip(axes, MODELS)):
        for i, (b, label, color) in enumerate(BATCHES):
            vals = [p[m][w][b]["CLoRA"] for w in WL]
            xs = [x[j] + (i - (n - 1) / 2) * width for j in range(len(WL))]
            ax.bar(xs, vals, width, label=label, color=color,
                   edgecolor="black", linewidth=0.4)
        clean_axis(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([WLAB[w] for w in WL], rotation=30, ha="right")
        ax.set_ylim(top=ax.get_ylim()[1] * 2.0)
        panel_caption(ax, idx, MODEL_TAGS[m])
    axes[0].set_ylabel(TTFT_LABEL)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07),
               frameon=False, columnspacing=1.4, handlelength=1.4,
               title="Prefill batch size (CLoRA)")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    save(fig, "fig_prefill_ttft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
