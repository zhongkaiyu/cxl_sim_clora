#!/usr/bin/env python3
"""Hardware sensitivity figures: ABSOLUTE metric vs parameter (not normalized).

Reads script/sensitivity_abs.json (HW-only one-factor-at-a-time sweeps,
Llama2-7B, A100, batch 32). Each knob has rows {x, Uniform, Uniform-long}
of absolute CLoRA throughput.

Produces two 2x2 figures:
  fig/fig_sensitivity_tpot.{png,pdf}        — TPOT ms/token (lower better)
  fig/fig_sensitivity_throughput.{png,pdf}  — throughput tok/s (higher better)
Each panel: parameter on x, absolute metric on y, one line per workload;
the paper default (vertical dashed) and the HBM-bound limit (horizontal
dashed) are marked.
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
HBM_FLOOR_MS = 14e9 / 1935e9 * 1e3            # Llama2-7B A100 base HBM read
HBM_CEIL_TOKS = BATCH / (HBM_FLOOR_MS / 1e3)
WORKLOADS = [("Uniform", "Uniform (short-KV)", "#1f77b4", "o"),
             ("Uniform-long", "Uniform-long (long-KV)", "#d62728", "s")]
ORDER = ["cxl_latency", "cxl_bandwidth", "ndp_throughput", "dram_bandwidth",
         "ndp_buffer"]


def _make(sens, metric):
    # log y so the knees/cliffs are crisp; shared across panels.
    ylim = (6.0, 200.0) if metric == "tpot" else (150.0, 5000.0)
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.8), sharey=True)
    for ax, knob in zip(axes.flat, ORDER):
        s = sens[knob]
        rows = sorted(s["rows"], key=lambda r: r["x"])
        xs = [r["x"] for r in rows]
        # panels with their own "series" list (e.g. ndp_buffer: two latency
        # regimes) are plotted with those keys; others use the two workloads.
        if s.get("series"):
            palette = [("#2ca02c", "o"), ("#9467bd", "^")]
            plot_series = [(name, name, palette[k % 2][0], palette[k % 2][1])
                           for k, name in enumerate(s["series"])]
        else:
            plot_series = WORKLOADS
        for wkey, wlabel, color, marker in plot_series:
            ys = [(BATCH / r[wkey] * 1000 if metric == "tpot" else r[wkey])
                  for r in rows]
            ax.plot(xs, ys, marker=marker, color=color, linewidth=1.7,
                    markersize=5.5, label=wlabel)
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.axvline(s["default_x"], color="gray", ls=":", lw=1.4)
        lim = HBM_FLOOR_MS if metric == "tpot" else HBM_CEIL_TOKS
        ax.axhline(lim, color="black", ls="--", lw=0.9, alpha=0.55)
        if s["log"]:
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=7)
        ax.set_title(s["title"])
        ax.set_xlabel(s["xlabel"])
        ax.set_ylabel("TPOT (ms/token) ↓" if metric == "tpot"
                      else "throughput (tok/s) ↑")
        ax.grid(alpha=0.3, lw=0.4, which="both")
        ax.set_axisbelow(True)
    # 6th cell: legend + reference-line key
    leg_ax = axes.flat[5]
    leg_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    leg_ax.legend(handles, labels, loc="upper center", fontsize=10,
                  frameon=False, title="workload")
    leg_ax.text(0.5, 0.45,
                "gray ⋮  = paper default value\n"
                "black --  = HBM-bound limit\n"
                "(GPU floor CLoRA cannot beat)\n\n"
                "Bottleneck knobs: CXL bandwidth,\n"
                "NDP throughput, device DRAM BW.\n"
                "Flat (non-bottleneck): CXL latency,\n"
                "NDP buffer.",
                transform=leg_ax.transAxes, ha="center", va="top",
                fontsize=8.5, color="#333")
    better = "lower is better" if metric == "tpot" else "higher is better"
    fig.suptitle("CLoRA hardware sensitivity — Llama2-7B, A100, batch 32 "
                 f"(absolute {'TPOT' if metric=='tpot' else 'throughput'}, "
                 f"{better}, log–log; one knob varied, all else fixed)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    name = f"fig_sensitivity_{metric}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight",
                    dpi=300)
    plt.close(fig)
    print(f"wrote fig/{name}.png/.pdf")


def main():
    sens = json.load(open(os.path.join(HERE, "sensitivity_abs.json")))
    _make(sens, "tpot")
    _make(sens, "throughput")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
