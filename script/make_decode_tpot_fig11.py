#!/usr/bin/env python3
"""Combined decode figure (paper Fig 11 + Fig 14 merged), Y-axis = TPOT.

H100, 4 models (Llama2-7B/13B MHA, Llama3-8B GQA, Qwen3-30B GQA+MoE), 4
workloads. Systems: CLoRA, CLoRA-NoCXL, Grace-Hopper, CPU-LoRA-Offload
(all simulated) + S-LoRA (real H100 measured) — S-LoRA is only available
for Llama2-7B, so it appears only in that panel (no fabricated data).

Y = TPOT (ms/token) = batch/throughput (ours) or measured tpot_mean_s
(S-LoRA); log scale, lower is better.

Out: fig/fig_decode_tpot_fig11.{png,pdf}
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

# fixed 5-slot order so bars align across panels; S-LoRA slot empty where absent
SYS = [("CLoRA", "CLoRA (Ours)", "#d62728"),
       ("NoCXL", "CLoRA-NoCXL", "#1f77b4"),
       ("GraceHopper", "Grace-Hopper", "#2ca02c"),
       ("CPULoRA", "CPU-LoRA-Offload", "#7f7f7f"),
       ("SLoRA", "S-LoRA (real H100)", "#ff7f0e")]
MODELS = ["Llama2-7B", "Llama2-13B", "Llama3-8B", "Qwen3-30B"]
TAG = {"Llama2-7B": "Llama2-7B (MHA)", "Llama2-13B": "Llama2-13B (MHA)",
       "Llama3-8B": "Llama3-8B (GQA)", "Qwen3-30B": "Qwen3-30B (GQA+MoE)"}
WL = ["Uniform", "Uniform-long", "Skewed", "Skewed-long", "LMSYS"]
WLAB = {"Uniform": "Unif", "Uniform-long": "Unif-L",
        "Skewed": "Skew", "Skewed-long": "Skew-L", "LMSYS": "LMSYS"}
# LMSYS decode batch is emergent (per-adapter B_ij<=256); the run used this
# operating batch, so TPOT = LMSYS_BATCH / throughput (not the batch-32 BATCH).
LMSYS_BATCH = 64

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def main():
    g = json.load(open(os.path.join(HERE, "results_decode_unfused.json")))["H100"]
    # LMSYS (real Chatbot Arena trace) decode, H100; emergent batch
    lmsys = json.load(open(os.path.join(HERE, "results_lmsys.json")))["H100"]
    # real S-LoRA, per model; only zero-error cells are present in each file
    slora = {
        "Llama2-7B": json.load(open(os.path.join(HERE,
                      "results_slora_llama7b_decode.json"))),
        "Llama2-13B": json.load(open(os.path.join(HERE,
                      "results_slora_llama13b_decode.json"))),
    }

    def tpot(model, wl, key):
        if key == "SLoRA":
            # no LMSYS S-LoRA measurements; only synthetic 7B/13B exist
            if wl == "LMSYS":
                return None
            sd = slora.get(model)
            if sd is None or wl not in sd or "tpot_mean_s" not in sd[wl]:
                return None
            return sd[wl]["tpot_mean_s"] * 1000.0
        if wl == "LMSYS":
            v = lmsys.get(model, {}).get(key)
            return LMSYS_BATCH / v * 1000.0 if v else None
        return BATCH / g[model][wl][key] * 1000.0

    fig, axes = plt.subplots(1, 4, figsize=(17.0, 3.6), sharey=True)
    x = np.arange(len(WL))
    n = len(SYS)
    width = 0.16
    for ax, m in zip(axes, MODELS):
        for i, (key, label, color) in enumerate(SYS):
            vals = [tpot(m, w, key) for w in WL]
            xs = [x[j] + (i - (n - 1) / 2) * width for j in range(len(WL))]
            xs = [xx for xx, v in zip(xs, vals) if v is not None]
            vv = [v for v in vals if v is not None]
            if not vv:
                continue
            bars = ax.bar(xs, vv, width, label=label, color=color,
                          edgecolor="black", linewidth=0.4)
            for b, v in zip(bars, vv):
                ax.text(b.get_x() + b.get_width() / 2, v * 1.04,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=5,
                        rotation=90)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([WLAB[w] for w in WL], fontsize=8)
        ax.set_title(TAG[m])
        ax.grid(axis="y", which="both", alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    axes[0].set_ylabel("TPOT (ms/token)  ↓ lower is better")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06),
               frameon=False, fontsize=9)
    fig.suptitle("Decode TPOT on H100 (log scale, lower is better) — synthetic "
                 "workloads: batch 32, 1000 adapters; LMSYS: real Chatbot Arena "
                 "trace, 25 adapters, emergent batch. S-LoRA real-measured "
                 "(Llama2-7B/13B synthetic cells only)",
                 y=1.10, fontsize=9.0, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_decode_tpot_fig11.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote fig/fig_decode_tpot_fig11.png/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
