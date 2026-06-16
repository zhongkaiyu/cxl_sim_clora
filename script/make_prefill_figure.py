#!/usr/bin/env python3
"""Fig 12 (prefill / TTFT) regenerated on H100, 4 models, 4 systems.

Reads script/results_prefill.json. For each model panel, x-axis = batch
size {256, 512, 1024}, grouped bars = 4 systems, Y = TTFT (seconds, log).
One figure per workload regime: fig/fig_prefill_<workload>.{png,pdf}, plus a
combined 4x4 (rows=workload, cols=model) overview.
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

# NoCXL key chosen per figure (unfused/fused); set in main().
SYSTEMS = ["CLoRA", "NoCXL_unfused", "GraceHopper", "CPULoRA"]
SYS_LABEL = {"CLoRA": "CLoRA (Ours)",
             "NoCXL_unfused": "CLoRA-NoCXL", "NoCXL_fused": "CLoRA-NoCXL",
             "GraceHopper": "Grace-Hopper", "CPULoRA": "CPU-LoRA-Offload"}
SYS_COLOR = {"CLoRA": "#d62728",
             "NoCXL_unfused": "#1f77b4", "NoCXL_fused": "#1f77b4",
             "GraceHopper": "#2ca02c", "CPULoRA": "#7f7f7f"}
MODELS = ["Llama2-7B", "Llama2-13B", "Llama3-8B", "Qwen3-30B"]
MODEL_TAG = {"Llama2-7B": "Llama2-7B (MHA)", "Llama2-13B": "Llama2-13B (MHA)",
             "Llama3-8B": "Llama3-8B (GQA)", "Qwen3-30B": "Qwen3-30B (GQA+MoE)"}
WORKLOADS = ["Uniform", "Uniform-long", "Skewed", "Skewed-long"]
BATCHES = ["256", "512", "1024"]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8.5})


def panel(ax, model_wl, title, ylabel):
    x = np.arange(len(BATCHES))
    width = 0.2
    for i, s in enumerate(SYSTEMS):
        vals = [model_wl[b][s] for b in BATCHES]
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      label=SYS_LABEL[s], color=SYS_COLOR[s],
                      edgecolor="black", linewidth=0.4)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v * 1.06,
                    f"{v:.1f}" if v < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=5.2, rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"B={b}" for b in BATCHES], fontsize=8)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel("TTFT (s)  ↓ lower is better")
    ax.grid(axis="y", which="both", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(top=ax.get_ylim()[1] * 3)


def per_workload(data, wl):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.3 * len(MODELS), 3.0),
                             squeeze=False)
    for c, m in enumerate(MODELS):
        panel(axes[0][c], data[m][wl], MODEL_TAG[m], ylabel=(c == 0))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle(f"Prefill TTFT on H100, {wl} prompts (seconds, log; "
                 "lower is better)", y=1.10, fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    tag = wl.lower().replace("-", "_")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_prefill_{tag}.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote fig/fig_prefill_{tag}.png/.pdf")


def combined(data, suffix):
    fig, axes = plt.subplots(len(WORKLOADS), len(MODELS),
                             figsize=(3.1 * len(MODELS), 2.5 * len(WORKLOADS)),
                             squeeze=False)
    for r, wl in enumerate(WORKLOADS):
        for c, m in enumerate(MODELS):
            panel(axes[r][c], data[m][wl],
                  f"{wl}: {MODEL_TAG[m]}", ylabel=(c == 0))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.01), frameon=False)
    note = ("NoCXL = per-adapter/per-request launches" if "unfused" in suffix
            else "NoCXL = BGMV-fused (one launch per op per layer)")
    fig.suptitle("Prefill TTFT on H100 (seconds, log scale; lower is better)\n"
                 + note, y=1.04, fontsize=10, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_prefill_combined{suffix}.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote fig/fig_prefill_combined{suffix}.png/.pdf")


def main():
    global SYSTEMS
    with open(os.path.join(HERE, "results_prefill.json")) as f:
        data = json.load(f)
    for mode, key in (("unfused", "NoCXL_unfused"), ("fused", "NoCXL_fused")):
        SYSTEMS = ["CLoRA", key, "GraceHopper", "CPULoRA"]
        combined(data, f"_{mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
