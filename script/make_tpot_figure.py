#!/usr/bin/env python3
"""Combined decode TPOT figure (rebuttal: merges old Fig 11 + Fig 14 + MoE).

Y-axis = TPOT (Time Per Output Token) in ms/token = batch / throughput;
lower is better. Reads script/results_decode_all.json.

Produces fig/fig_tpot_combined.{png,pdf}: rows = platforms (A100, H100),
cols = models (Llama2-7B, Llama2-13B, Llama3-8B, Qwen3-30B), grouped bars
over the 4 workloads with 4 systems each. Also emits per-platform 1x4
figures fig/fig_tpot_{a100,h100}.{png,pdf}.
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

BATCH = 32
SYSTEMS = ["CLoRA", "NoCXL", "GraceHopper", "CPULoRA"]
SYS_LABEL = {"CLoRA": "CLoRA (Ours)", "NoCXL": "CLoRA-NoCXL",
             "GraceHopper": "Grace-Hopper", "CPULoRA": "CPU-LoRA-Offload"}
SYS_COLOR = {"CLoRA": "#d62728", "NoCXL": "#1f77b4",
             "GraceHopper": "#2ca02c", "CPULoRA": "#7f7f7f"}
MODELS = ["Llama2-7B", "Llama2-13B", "Llama3-8B", "Qwen3-30B"]
MODEL_TAG = {"Llama2-7B": "Llama2-7B (MHA)", "Llama2-13B": "Llama2-13B (MHA)",
             "Llama3-8B": "Llama3-8B (GQA)", "Qwen3-30B": "Qwen3-30B (GQA+MoE)"}
WORKLOADS = ["Uniform", "Uniform-long", "Skewed", "Skewed-long"]
W_LABEL = {"Uniform": "Unif", "Uniform-long": "Unif-L",
           "Skewed": "Skew", "Skewed-long": "Skew-L"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8.5,
                     "figure.dpi": 120})


def tpot(throughput):
    return BATCH / throughput * 1000.0  # ms/token


def panel(ax, model_grid, title, show_ylabel):
    x = np.arange(len(WORKLOADS))
    width = 0.2
    for i, s in enumerate(SYSTEMS):
        vals = [tpot(model_grid[w][s]) for w in WORKLOADS]
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      label=SYS_LABEL[s], color=SYS_COLOR[s],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.05,
                    f"{v:.1f}" if v < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=5.0, rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([W_LABEL[w] for w in WORKLOADS], fontsize=8)
    ax.set_title(title)
    if show_ylabel:
        ax.set_ylabel("TPOT (ms/token)  ↓ lower is better")
    ax.grid(axis="y", which="both", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(top=ax.get_ylim()[1] * 3)


NOCXL_NOTE = {
    "unfused": "CLoRA-NoCXL = per-adapter / per-request kernel launches",
    "fused":   "CLoRA-NoCXL = BGMV-fused (one launch per op per layer)",
}


def combined(data, suffix=""):
    plats = list(data.keys())
    fig, axes = plt.subplots(len(plats), len(MODELS),
                             figsize=(3.2 * len(MODELS), 2.7 * len(plats)),
                             squeeze=False)
    for r, plat in enumerate(plats):
        for c, m in enumerate(MODELS):
            panel(axes[r][c], data[plat][m],
                  f"{plat}: {MODEL_TAG[m]}", show_ylabel=(c == 0))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.015), frameon=False)
    note = NOCXL_NOTE.get(suffix.strip("_"), "")
    fig.suptitle("Decode TPOT (ms/token, log scale; lower is better), "
                 f"batch 32, 1000 adapters\n{note}", y=1.07, fontsize=9.5,
                 style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_tpot_combined{suffix}.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote fig/fig_tpot_combined{suffix}.png/.pdf")


def per_platform(data):
    for plat in data:
        fig, axes = plt.subplots(1, len(MODELS),
                                 figsize=(3.2 * len(MODELS), 3.0),
                                 squeeze=False)
        for c, m in enumerate(MODELS):
            panel(axes[0][c], data[plat][m], MODEL_TAG[m], show_ylabel=(c == 0))
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4,
                   bbox_to_anchor=(0.5, 1.04), frameon=False)
        fig.suptitle(f"Decode TPOT on {plat} (ms/token, lower is better), "
                     "batch 32, 1000 adapters", y=1.10, fontsize=9,
                     style="italic")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        tag = plat.lower()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"fig_tpot_{tag}.{ext}"),
                        bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"wrote fig/fig_tpot_{tag}.png/.pdf")


def main():
    for mode in ("unfused", "fused"):
        path = os.path.join(HERE, f"results_decode_{mode}.json")
        if not os.path.exists(path):
            print(f"skip {mode}: {path} missing")
            continue
        with open(path) as f:
            data = json.load(f)
        combined(data, suffix=f"_{mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
