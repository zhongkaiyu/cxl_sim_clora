#!/usr/bin/env python3
"""Render the rebuttal figures from the saved result JSONs.

  fig/fig_decode_a100.{png,pdf}   -- new Fig 11: 4 systems x 4 workloads,
                                     panels (a) Llama2-7B (b) Llama2-13B
                                     (c) Qwen3-30B-A3B MoE, A100 40GB
  fig/fig_decode_h100.{png,pdf}   -- same layout on H100 80GB
  fig/fig_gqa.{png,pdf}           -- new Fig 14: Llama3-8B (GQA), A100
  fig/fig_sensitivity.{png,pdf}   -- new sensitivity figure: 2x2 panels
                                     (CXL latency, CXL bandwidth,
                                     NDP throughput, NDP buffer)

Inputs: script/results_grid.json (A100), script/results_h100.json,
script/results_gqa.json, script/sensitivity_results.json.
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

SYSTEMS = ["CLoRA", "NoCXL", "GraceHopper", "CPULoRA"]
SYS_LABEL = {"CLoRA": "CLoRA (Ours)", "NoCXL": "CLoRA-NoCXL",
             "GraceHopper": "Grace-Hopper", "CPULoRA": "CPU-LoRA-Offload"}
SYS_COLOR = {"CLoRA": "#d62728", "NoCXL": "#1f77b4",
             "GraceHopper": "#2ca02c", "CPULoRA": "#7f7f7f"}
WORKLOADS = ["Uniform", "Uniform-long", "Skewed", "Skewed-long"]
W_LABEL = {"Uniform": "Uniform", "Uniform-long": "Uniform-L",
           "Skewed": "Skewed", "Skewed-long": "Skewed-L"}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 120,
})


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def grouped_bars(ax, grid_for_model, title):
    workloads = [w for w in WORKLOADS if w in grid_for_model]
    x = np.arange(len(workloads))
    width = 0.2
    for i, sysname in enumerate(SYSTEMS):
        vals = [grid_for_model[w][sysname] for w in workloads]
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      label=SYS_LABEL[sysname], color=SYS_COLOR[sysname],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.08,
                    f"{v:,.0f}", ha="center", va="bottom",
                    fontsize=5.4, rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([W_LABEL[w] for w in workloads])
    ax.set_title(title)
    ax.grid(axis="y", which="both", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)


def decode_figure(grids, hw_note, outname):
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 2.9), sharey=False)
    if n == 1:
        axes = [axes]
    letters = "abc"
    for ax, (mname, mgrid), letter in zip(axes, grids.items(), letters):
        grouped_bars(ax, mgrid, f"({letter}) {mname}")
        ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    axes[0].set_ylabel("Throughput (tokens/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(hw_note, y=1.10, fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{outname}.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote fig/{outname}.png/.pdf")


def sensitivity_figure(sens, outname):
    panels = [
        ("cxl_latency", "(a) CXL switch latency", "added one-way latency (ns)",
         lambda lbl: int(lbl.lstrip("+")), "linear"),
        ("cxl_bandwidth", "(b) CXL link bandwidth", "GB/s per device",
         lambda lbl: int(lbl), "log"),
        ("ndp_throughput", "(c) NDP core throughput", "TFLOPS per device",
         lambda lbl: float(lbl.split()[0]), "log"),
        ("ndp_buffer", "(d) NDP controller buffer", "in-flight requests",
         lambda lbl: int(lbl.split("(")[1].split()[0]), "log"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    for ax, (key, title, xlabel, parse_x, xscale) in zip(axes.flat, panels):
        rows = sens[key]["rows"]
        # drop the legacy-PE continuity reference from the NDP plot
        rows = [r for r in rows if "legacy" not in r["label"]]
        pts = sorted((parse_x(r["label"]), r["throughput"]) for r in rows)
        xs = [p[0] for p in pts]
        for wname, color, marker in [("Uniform", "#1f77b4", "o"),
                                     ("Uniform-long", "#d62728", "s")]:
            base = next(t[wname] for x, t in pts
                        if x == (0 if key == "cxl_latency" else
                                 128 if key == "cxl_bandwidth" else
                                 2.0 if key == "ndp_throughput" else 8))
            ys = [t[wname] / base for _, t in pts]
            ax.plot(xs, ys, marker=marker, markersize=4, color=color,
                    linewidth=1.4, label=wname)
        ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--")
        if xscale == "log":
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("normalized throughput")
        ax.set_ylim(0.4, 1.15)
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle("CLoRA sensitivity (Llama2-7B, A100; default marked by dashed line = 1.0)",
                 y=1.05, fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{outname}.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote fig/{outname}.png/.pdf")


def main():
    a100 = load("results_grid.json")
    decode_figure(a100,
                  "Decode throughput, NVIDIA A100 40GB, batch 32, 1000 adapters",
                  "fig_decode_a100")

    h100 = load("results_h100.json")
    decode_figure(h100["grid"],
                  "Decode throughput, NVIDIA H100 SXM 80GB, batch 32, 1000 adapters",
                  "fig_decode_h100")

    gqa = load("results_gqa.json")
    decode_figure(gqa["grid"],
                  "Llama3-8B with GQA (4x smaller KV), A100 40GB, batch 32",
                  "fig_gqa")

    sens = load("sensitivity_results.json")
    sensitivity_figure(sens, "fig_sensitivity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
