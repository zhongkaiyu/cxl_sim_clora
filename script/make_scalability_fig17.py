#!/usr/bin/env python3
"""Optimal CXL device count on H100 (regenerated Fig 17 + analysis).

Sweeps N_CXL in {1,2,4,8,16,32} for Llama2-7B, Llama2-13B (MHA) and
Llama3-8B (GQA) on H100, 4 workloads. Throughput (tok/s) vs device count,
log-x. The HBM-bound ceiling is marked, and the optimum (smallest N
reaching >=99% of the ceiling on the demanding long-KV workload) is
highlighted. All hardware/workload parameters are printed on the figure.

Data: script/results_scalability.json. Out: fig/fig_scalability_fig17.{png,pdf}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "fig")
os.makedirs(FIG, exist_ok=True)
BATCH = 32
DEVS = [1, 2, 4, 8, 16, 32]

MODELS = ["Llama2-7B", "Llama2-13B", "Llama3-8B"]
# resident base GB on H100 -> HBM ceiling = batch / (base/3350 GB/s)
RESIDENT = {"Llama2-7B": 14, "Llama2-13B": 26, "Llama3-8B": 16}
TAG = {"Llama2-7B": "Llama2-7B (MHA)", "Llama2-13B": "Llama2-13B (MHA)",
       "Llama3-8B": "Llama3-8B (GQA)"}
WL = [("Uniform", "Uniform", "#1f77b4", "o"),
      ("Uniform-long", "Uniform-long", "#d62728", "s"),
      ("Skewed", "Skewed", "#2ca02c", "^"),
      ("Skewed-long", "Skewed-long", "#9467bd", "D")]

PARAMS = ("Parameters:  GPU = NVIDIA H100 SXM (989 TFLOPS FP16, 3350 GB/s "
          "HBM, 80 GB)  |  batch = 32  |  1000 LoRA adapters (ranks "
          "{8,16,32,64,128})  |  per device: 128 GB/s CXL link, 1.1 TB/s DRAM, "
          "2 TFLOPS NDP  |  KV short = [100,1024] (GQA /4), long = [2048,4096] "
          "(GQA /4) tokens  |  seed 42, median of 3")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def optimum(ys):
    mx = max(ys)
    return next(n for n, y in zip(DEVS, ys) if y >= 0.99 * mx), mx


def main():
    s = json.load(open(os.path.join(HERE, "results_scalability.json")))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), sharey=False)
    for ax, m in zip(axes, MODELS):
        ceil = BATCH / (RESIDENT[m] * 1e9 / 3350e9)
        for wkey, wlab, color, marker in WL:
            ys = [s[m][wkey][str(n)] for n in DEVS]
            ax.plot(DEVS, ys, marker=marker, color=color, lw=1.7, ms=6,
                    label=wlab)
        ax.axhline(ceil, color="black", ls="--", lw=0.9, alpha=0.6)
        ax.text(DEVS[0], ceil, f" HBM ceiling {ceil:,.0f}", fontsize=7,
                va="bottom", ha="left", alpha=0.7)
        # optimum on the long-KV (demanding) workload
        n_opt, _ = optimum([s[m]["Uniform-long"][str(n)] for n in DEVS])
        ax.axvline(n_opt, color="orange", ls="-", lw=2.0, alpha=0.5)
        ax.text(n_opt, ax.get_ylim()[0], f" opt N={n_opt}", color="darkorange",
                fontsize=9, fontweight="bold", va="bottom", ha="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(DEVS)
        ax.set_xticklabels([str(n) for n in DEVS])
        ax.set_xlabel("Number of CLoRA (CXL) devices  N_CXL")
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_title(TAG[m])
        ax.grid(alpha=0.3, lw=0.4, which="both")
        ax.set_axisbelow(True)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Optimal CXL device count on H100 — throughput saturates at "
                 "the HBM ceiling; optimum (orange) is N_CXL=16 for MHA "
                 "long-context, N_CXL=4 for GQA",
                 fontsize=10)
    fig.text(0.5, -0.02, PARAMS, ha="center", va="top", fontsize=7.2,
             wrap=True, color="#222",
             bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#bbb"))
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig_scalability_fig17.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote fig/fig_scalability_fig17.png/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
