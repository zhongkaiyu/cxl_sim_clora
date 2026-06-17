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
from _style import TPUT_LABEL, panel_xlabel, clean_axis, save, SIZE  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "script")
ORDER = ["cxl_latency", "cxl_bandwidth", "ndp_throughput",
         "dram_bandwidth", "ndp_buffer"]
# sub-caption carries the name + unit (no separate x-label, to avoid collisions
# across the 5 panels at body font); the x-ticks already show the values.
CAP = {"cxl_latency": "CXL latency (ns)", "cxl_bandwidth": "CXL link BW (GB/s)",
       "ndp_throughput": "NDP compute (TFLOPS)",
       "dram_bandwidth": "Device DRAM BW (GB/s)",
       "ndp_buffer": "NDP buffer (reqs)"}
# default two-workload series (panels a-d)
SERIES = [("Uniform", "Uniform (short-KV)", "#1f77b4", "o"),
          ("Uniform-long", "Uniform-long (long-KV)", "#d62728", "s")]
# palette for panels that carry their own "series" list (panel e)
PANEL_COLORS = ["#2ca02c", "#9467bd", "#e377c2", "#8c564b"]
PANEL_MARKERS = ["o", "^", "s", "D"]


def main():
    src = os.environ.get("SENS_OUT", "sensitivity_abs.json")
    out_name = os.environ.get("SENS_FIG", "fig_sensitivity")
    d = json.load(open(os.path.join(SCRIPT, src)))
    fig, axes = plt.subplots(1, 5, figsize=(20.0, 4.8))
    for idx, (ax, knob) in enumerate(zip(axes, ORDER)):
        spec = d[knob]
        if spec.get("series"):                       # panel-specific series
            # rise-then-saturate: 1-command (under-buffered) -> saturated.
            # Show the relevant range (<=16, i.e. up to 2x the Table-4 default
            # of 8 in-flight commands); deeper buffers only over-provision.
            pts = [r for r in spec["rows"] if r["x"] <= 16]
            xs = [r["x"] for r in pts]
            for k, name in enumerate(spec["series"]):
                ys = [r[name] for r in pts]
                ax.plot(xs, ys, marker=PANEL_MARKERS[k % 4], label=name,
                        color=PANEL_COLORS[k % 4], lw=1.2, ms=3.5)
            ax.legend(loc="lower center", fontsize=7, frameon=False,
                      handlelength=1.4, labelspacing=0.2)
        else:
            xs = [r["x"] for r in spec["rows"]]
            for wkey, wlab, color, mk in SERIES:
                ys = [r[wkey] for r in spec["rows"]]
                ax.plot(xs, ys, marker=mk, color=color, label=wlab,
                        lw=1.2, ms=3.5)
        if spec.get("log"):
            ax.set_xscale("log")
        # explicit tick mark + label at EVERY swept value (no sparse log ticks)
        ax.xaxis.set_major_locator(FixedLocator(xs))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in xs]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", rotation=45)
        ax.axvline(spec["default_x"], color="gray", ls=":", lw=1.0)
        clean_axis(ax, log=False)
        panel_xlabel(ax, idx, CAP[knob], pad=26)
        if idx == 0:
            ax.set_ylabel(TPUT_LABEL)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13),
               frameon=False, columnspacing=1.5, handlelength=1.8)
    fig.tight_layout(rect=[0, 0.0, 1, 0.91], w_pad=2.0)
    save(fig, out_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
