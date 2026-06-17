#!/usr/bin/env python3
"""Scalability vs CXL device count (paper Fig 17), H100, larger models.

Llama2-13B (MHA) and Qwen3-30B (GQA+MoE), N_CXL in {1,2,4,8,16,32}, 4
workloads. Finds the new sweet point = smallest N_CXL reaching >=99% of the
saturated throughput. Higher is better.

Modeling note: each device gets its own CXL channel, so throughput saturates
at the HBM ceiling rather than declining past the knee (the paper's beyond-knee
drop comes from a single shared GPU<->switch aggregation link, not modeled).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import TPUT_LABEL, MODEL_TAGS, panel_caption, clean_axis, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "script")
# Fig 17 plots Llama2-13B only: its curve is a clean monotone saturation.
# Qwen3-30B is intentionally NOT plotted -- its MoE active footprint (6 GB)
# saturates the GPU HBM ceiling at N_CXL<=2, so adding devices can't help and
# the cost model's strategy re-selection makes the curve non-monotone (noise,
# not a real scaling trend). That finding is reported in the README text.
MODELS = ["Llama2-13B"]
WORKLOADS = [("Uniform", "#1f77b4", "o"), ("Uniform-long", "#d62728", "s"),
             ("Skewed", "#2ca02c", "^"), ("Skewed-long", "#9467bd", "D")]
PAPER_DEFAULT = 4   # Table 4 default N_CXL


def sweet_point(series):
    ceiling = max(series.values())
    for n in sorted(series, key=int):
        if series[n] >= 0.99 * ceiling:
            return int(n)
    return None


def main():
    d = json.load(open(os.path.join(SCRIPT, "results_scalability.json")))
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.5, 5.0), squeeze=False)
    axes = axes[0]
    print("sweet points (smallest N_CXL >= 99% of ceiling):")
    for idx, (ax, m) in enumerate(zip(axes, MODELS)):
        for wkey, color, mk in WORKLOADS:
            series = d[m][wkey]
            ns = sorted(series, key=int)
            xs = [int(n) for n in ns]
            ys = [series[n] for n in ns]
            ax.plot(xs, ys, marker=mk, color=color, label=wkey, lw=1.2, ms=3.5)
            sp = sweet_point(series)
            print(f"  {m:11s} {wkey:13s} -> N_CXL={sp}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
        ax.axvline(PAPER_DEFAULT, color="gray", ls=":", lw=0.8)
        clean_axis(ax, log=False)
        ax.set_xlabel(r"Number of CXL devices $N_{\mathrm{CXL}}$")
        panel_caption(ax, idx, MODEL_TAGS[m])
        if idx == 0:
            ax.set_ylabel(TPUT_LABEL, y=0.45)   # nudge down: clear the legend
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.14),
               frameon=False, columnspacing=1.4, handlelength=1.8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.88])
    save(fig, "fig_scalability")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
