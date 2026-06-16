#!/usr/bin/env python3
"""Combined decode figure (paper Fig 11 + 13 merged), Y = TPOT.

4 model panels x 5 workloads (Uniform / Uniform-long / Skewed / Skewed-long /
LMSYS). Systems: CLoRA, CLoRA-NoCXL, Grace-Hopper, CPU-LoRA-Offload (simulated)
+ S-LoRA (real H100). H100. Lower is better.

Data we actually have only -- no fabrication:
  * S-LoRA real-measured on Llama2-7B/13B synthetic cells only (absent on
    Llama3-8B, Qwen3-30B, and LMSYS until measured).
  * Attacc and "Other LLM 10-30B" are NOT plotted (no implementation/config).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import (SYSTEMS, MODEL_TAGS, TPOT_LABEL, panel_caption,  # noqa: E402
                    clean_axis, save)
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "script")
BATCH = 32          # synthetic workloads: fixed batch 32
LMSYS_BATCH = 64    # LMSYS: emergent batch (per-adapter B_ij <= 256)

MODELS = ["Llama2-7B", "Llama2-13B", "Llama3-8B", "Qwen3-30B"]
WL = ["Uniform", "Uniform-long", "Skewed", "Skewed-long", "LMSYS"]
WLAB = {"Uniform": "Unif", "Uniform-long": "Unif-L", "Skewed": "Skew",
        "Skewed-long": "Skew-L", "LMSYS": "LMSYS"}


def main():
    g = json.load(open(os.path.join(SCRIPT, "results_decode_unfused.json")))["H100"]
    lmsys = json.load(open(os.path.join(SCRIPT, "results_lmsys.json")))["H100"]
    slora = {
        "Llama2-7B":  json.load(open(os.path.join(SCRIPT, "results_slora_llama7b_decode.json"))),
        "Llama2-13B": json.load(open(os.path.join(SCRIPT, "results_slora_llama13b_decode.json"))),
    }

    def tpot(model, wl, key):
        if key == "SLoRA":
            sd = slora.get(model)
            if not sd or wl not in sd or "tpot_mean_s" not in sd[wl]:
                return None
            return sd[wl]["tpot_mean_s"] * 1000.0
        if wl == "LMSYS":
            v = lmsys.get(model, {}).get(key)
            return LMSYS_BATCH / v * 1000.0 if v else None
        return BATCH / g[model][wl][key] * 1000.0

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0), sharey=True)
    x = np.arange(len(WL))
    n = len(SYSTEMS)
    width = 0.16
    for idx, (ax, m) in enumerate(zip(axes, MODELS)):
        for i, (key, label, color) in enumerate(SYSTEMS):
            vals = [tpot(m, w, key) for w in WL]
            xs = [x[j] + (i - (n - 1) / 2) * width for j in range(len(WL))]
            xs = [xx for xx, v in zip(xs, vals) if v is not None]
            vv = [v for v in vals if v is not None]
            if vv:
                ax.bar(xs, vv, width, label=label, color=color,
                       edgecolor="black", linewidth=0.4)
        clean_axis(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([WLAB[w] for w in WL], rotation=30, ha="right")
        ax.set_ylim(top=ax.get_ylim()[1] * 2.0)
        panel_caption(ax, idx, MODEL_TAGS[m])
    axes[0].set_ylabel(TPOT_LABEL)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.07),
               frameon=False, columnspacing=1.2, handlelength=1.4)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save(fig, "fig_decode_tpot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
