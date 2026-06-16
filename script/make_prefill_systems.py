#!/usr/bin/env python3
"""Prefill systems comparison (prefill analog of the decode Fig 11) on H100.

Rows = workload {Uniform (short prompts), Uniform-long (long prompts)},
cols = 4 models. Per panel: x = batch {256,512,1024}, grouped bars =
CLoRA, CLoRA-NoCXL, Grace-Hopper, CPU-LoRA-Offload (simulated) + S-LoRA
(real H100, Llama2-7B only). Y = prefill TTFT (s, log, lower=better).

S-LoRA is present ONLY on the Llama2-7B panels (only S-LoRA data we have);
no fabricated data. Data: results_prefill.json + results_slora_llama7b_prefill.json.
Out: fig/fig_prefill_systems.{png,pdf}
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

# (json key, label, color). "SLoRA" is real H100, 7B only.
SYS = [("CLoRA", "CLoRA (Ours)", "#d62728"),
       ("NoCXL_unfused", "CLoRA-NoCXL", "#1f77b4"),
       ("GraceHopper", "Grace-Hopper", "#2ca02c"),
       ("CPULoRA", "CPU-LoRA-Offload", "#7f7f7f"),
       ("SLoRA", "S-LoRA (real H100)", "#ff7f0e")]
MODELS = ["Llama2-7B", "Llama2-13B", "Llama3-8B", "Qwen3-30B"]
TAG = {"Llama2-7B": "Llama2-7B (MHA)", "Llama2-13B": "Llama2-13B (MHA)",
       "Llama3-8B": "Llama3-8B (GQA)", "Qwen3-30B": "Qwen3-30B (GQA+MoE)"}
WORKLOADS = [("Uniform", "Uniform (short prompts)"),
             ("Uniform-long", "Uniform-long (long prompts)")]
BATCHES = ["256", "512", "1024"]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})


def main():
    p = json.load(open(os.path.join(HERE, "results_prefill.json")))
    # real S-LoRA prefill, per model; only zero-error cells are present
    sl = {
        "Llama2-7B": json.load(open(os.path.join(HERE,
                      "results_slora_llama7b_prefill.json"))),
        "Llama2-13B": json.load(open(os.path.join(HERE,
                      "results_slora_llama13b_prefill.json"))),
    }

    def ttft(model, wl, b, key):
        if key == "SLoRA":
            sd = sl.get(model)
            if sd is None or wl not in sd or b not in sd[wl]:
                return None
            return sd[wl][b]
        return p[model][wl][b][key]

    fig, axes = plt.subplots(len(WORKLOADS), len(MODELS),
                             figsize=(15.0, 6.6), sharey="row")
    x = np.arange(len(BATCHES))
    n = len(SYS)
    width = 0.16
    for r, (wkey, wlab) in enumerate(WORKLOADS):
        for c, m in enumerate(MODELS):
            ax = axes[r][c]
            for i, (key, label, color) in enumerate(SYS):
                vals = [ttft(m, wkey, b, key) for b in BATCHES]
                xs = [x[j] + (i - (n - 1) / 2) * width for j in range(len(BATCHES))]
                xs = [xx for xx, v in zip(xs, vals) if v is not None]
                vv = [v for v in vals if v is not None]
                if not vv:
                    continue
                bars = ax.bar(xs, vv, width, label=label, color=color,
                              edgecolor="black", linewidth=0.4)
                for b_, v in zip(bars, vv):
                    ax.text(b_.get_x() + b_.get_width() / 2, v * 1.04,
                            f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                            ha="center", va="bottom", fontsize=5, rotation=90)
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([f"B={b}" for b in BATCHES], fontsize=8)
            ax.grid(axis="y", which="both", alpha=0.3, lw=0.4)
            ax.set_axisbelow(True)
            ax.set_ylim(top=ax.get_ylim()[1] * 3)
            if r == 0:
                ax.set_title(TAG[m])
            if c == 0:
                ax.set_ylabel(f"{wlab}\nTTFT (s) ↓")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04),
               frameon=False, fontsize=9)
    fig.suptitle("Prefill TTFT on H100, systems comparison (log scale, lower "
                 "is better), 1000 adapters — S-LoRA is real-measured "
                 "(Llama2-7B/13B, zero-error cells only)", y=1.005,
                 fontsize=9.5, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_prefill_systems.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote fig/fig_prefill_systems.png/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
