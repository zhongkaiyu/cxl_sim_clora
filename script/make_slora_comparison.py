#!/usr/bin/env python3
"""Single-figure decode comparison: our 4 simulated systems vs real-measured
S-LoRA, Llama2-7B, H100, across the 4 synthetic workloads.

Y = TPOT (ms/token) = batch / throughput (ours) or measured tpot_mean_s
(S-LoRA); lower is better, log scale.

Ours: script/results_decode_unfused.json  (H100, Llama2-7B; NoCXL = unfused)
S-LoRA: script/results_slora_llama7b_decode.json  (real H100 measurement)
Out: fig/fig_slora_decode_llama7b.{png,pdf}
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
WORKLOADS = ["Uniform", "Uniform-long", "Skewed", "Skewed-long"]
W_LABEL = {"Uniform": "Uniform", "Uniform-long": "Uniform-L",
           "Skewed": "Skewed", "Skewed-long": "Skewed-L"}

# system -> (label, color). S-LoRA is the real-measured baseline.
SYSTEMS = [
    ("CLoRA",       "CLoRA (Ours)",        "#d62728"),
    ("NoCXL",       "CLoRA-NoCXL",         "#1f77b4"),
    ("GraceHopper", "Grace-Hopper",        "#2ca02c"),
    ("CPULoRA",     "CPU-LoRA-Offload",    "#7f7f7f"),
    ("SLoRA",       "S-LoRA (real H100)",  "#ff7f0e"),
]


def main():
    ours = json.load(open(os.path.join(HERE, "results_decode_unfused.json")))
    g = ours["H100"]["Llama2-7B"]
    slora = json.load(open(os.path.join(HERE, "results_slora_llama7b_decode.json")))

    def tpot(w, key):
        if key == "SLoRA":
            return slora[w]["tpot_mean_s"] * 1000.0
        return BATCH / g[w][key] * 1000.0  # ms/token

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(WORKLOADS))
    n = len(SYSTEMS)
    width = 0.16
    for i, (key, label, color) in enumerate(SYSTEMS):
        vals = [tpot(w, key) for w in WORKLOADS]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width,
                      label=label, color=color, edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.04,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([W_LABEL[w] for w in WORKLOADS])
    ax.set_ylabel("TPOT (ms/token)  ↓ lower is better")
    sps = [tpot(w, "SLoRA") / tpot(w, "CLoRA") for w in WORKLOADS]
    ax.set_title("Llama2-7B decode, H100 — CLoRA & baselines (simulated) vs "
                 "S-LoRA (real)\n"
                 f"CLoRA is {min(sps):.0f}–{max(sps):.0f}× lower TPOT than "
                 "measured S-LoRA", fontsize=10)
    ax.grid(axis="y", which="both", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    ax.legend(ncol=5, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, 0.97), frameon=False)

    # small "N×" tag above each CLoRA bar
    clora_off = (0 - (n - 1) / 2) * width
    for j, w in enumerate(WORKLOADS):
        ax.text(j + clora_off, tpot(w, "CLoRA") * 1.10, f"{sps[j]:.0f}×",
                ha="center", va="bottom", fontsize=7.5, color="#d62728",
                fontweight="bold")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_slora_decode_llama7b.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote fig/fig_slora_decode_llama7b.png/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
