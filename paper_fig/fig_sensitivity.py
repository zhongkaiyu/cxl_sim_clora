#!/usr/bin/env python3
"""Hardware sensitivity study (paper section 7.8).

5 knobs (CXL link latency, CXL link bandwidth, NDP core throughput, device
DRAM bandwidth, NDP controller buffer), absolute decode throughput vs the
parameter, ranges chosen to cross each knob's knee. Two workloads per panel:
Uniform (short-KV, HBM-bound) and Uniform-long (long-KV, CXL/NDP-bound).
HW-only sweep (Llama2-7B, A100, fixed cost model). Higher is better.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import TPUT_LABEL, panel_caption, clean_axis, save, SIZE  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "script")
ORDER = ["cxl_latency", "cxl_bandwidth", "ndp_throughput",
         "dram_bandwidth", "ndp_buffer"]
SERIES = [("Uniform", "Uniform (short-KV)", "#1f77b4", "o"),
          ("Uniform-long", "Uniform-long (long-KV)", "#d62728", "s")]


def main():
    d = json.load(open(os.path.join(SCRIPT, "sensitivity_abs.json")))
    fig, axes = plt.subplots(1, 5, figsize=(15.0, 2.9))
    for idx, (ax, knob) in enumerate(zip(axes, ORDER)):
        spec = d[knob]
        xs = [r["x"] for r in spec["rows"]]
        for wkey, wlab, color, mk in SERIES:
            ys = [r[wkey] for r in spec["rows"]]
            ax.plot(xs, ys, marker=mk, color=color, label=wlab,
                    lw=1.2, ms=3.5)
        if spec.get("log"):
            ax.set_xscale("log")
        ax.axvline(spec["default_x"], color="gray", ls=":", lw=0.8)
        clean_axis(ax, log=False)
        ax.set_xlabel(spec["xlabel"])
        # title field is like "(a) CXL link latency"; reuse the text only
        txt = spec["title"].split(") ", 1)[-1]
        panel_caption(ax, idx, txt)
        if idx == 0:
            ax.set_ylabel(TPUT_LABEL)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10),
               frameon=False, columnspacing=1.5, handlelength=1.8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    save(fig, "fig_sensitivity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
