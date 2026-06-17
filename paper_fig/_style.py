"""Shared conference-figure style for the paper_fig/ deliverable.

House rules (applied uniformly to every figure here):
  * ONE font size everywhere (no per-element sizes).
  * NO axes/figure titles -- panels are identified by a sub-caption at the
    BOTTOM, e.g. "(a) Llama2-7B (MHA)".
  * Axis-title units use superscript exponents, not slashes
    (e.g. "TPOT (ms token^-1)", "Throughput (tokens s^-1)").
  * No value labels on top of bars.
  * Vector PDF (for Overleaf) + PNG preview.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.dirname(os.path.abspath(__file__))

# ---- single uniform font size ------------------------------------------------
# These wide multi-panel figures are included at the paper's full text width, so
# the source figure is ~2x wider than its printed size and gets scaled down.
# SIZE is therefore set large (~2x body) so the printed font reads at body size.
SIZE = 18
plt.rcParams.update({
    "font.size": SIZE, "axes.titlesize": SIZE, "axes.labelsize": SIZE,
    "xtick.labelsize": SIZE, "ytick.labelsize": SIZE, "legend.fontsize": SIZE,
    "figure.titlesize": SIZE, "font.family": "serif",
    "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "pdf.fonttype": 42,
    "ps.fonttype": 42, "lines.linewidth": 1.8, "lines.markersize": 5,
})

# ---- superscript-unit axis labels --------------------------------------------
TPOT_LABEL = r"TPOT (ms$\cdot$token$^{-1}$) $\downarrow$"
TPUT_LABEL = r"Throughput (tokens$\cdot$s$^{-1}$) $\uparrow$"
TTFT_LABEL = r"TTFT (s) $\downarrow$"

# ---- consistent system styling (order = legend AND bar order, left->right) ---
# (json key, legend label, color)
SYSTEMS = [
    ("SLoRA",       "S-LoRA",           "#ff7f0e"),
    ("CPULoRA",     "CPU-LoRA-Offload", "#7f7f7f"),
    ("GraceHopper", "Grace-Hopper",     "#2ca02c"),
    ("CLoRA",       "CLoRA",            "#d62728"),
    ("NoCXL",       "CLoRA-NoCXL",      "#1f77b4"),
]

MODEL_TAGS = {
    "Llama2-7B":  "Llama2-7B (MHA)",
    "Llama2-13B": "Llama2-13B (MHA)",
    "Llama3-8B":  "Llama3-8B (GQA)",
    "Qwen3-30B":  "Qwen3-30B (GQA+MoE)",
}

_LETTERS = "abcdefghijklmnop"


def panel_caption(ax, idx, text, dy=-0.34):
    """Sub-caption centred under the panel (replaces a title)."""
    ax.text(0.5, dy, f"({_LETTERS[idx]}) {text}", transform=ax.transAxes,
            ha="center", va="top", fontsize=SIZE)


def panel_xlabel(ax, idx, text, pad=22):
    """Sub-caption as the x-axis label (robustly placed BELOW the tick labels
    by matplotlib; ``pad`` is the gap from the tick labels)."""
    ax.set_xlabel(f"({_LETTERS[idx]}) {text}", labelpad=pad)


def clean_axis(ax, log=True):
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", which="both", alpha=0.3, lw=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote paper_fig/{name}.pdf + .png")
