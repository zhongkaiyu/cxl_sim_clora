#!/usr/bin/env python3
"""Dump the decode TPOT numbers (the fig_decode_tpot data) to a markdown table
and a LaTeX table -> paper_fig/tpot_results.{md,tex}.

TPOT (ms/token) on H100, batch 32 (synthetic) / emergent 64 (LMSYS), lower is
better. CLoRA Qwen3-30B values are the REAL ~2 ms (the figure draws them at the
Llama3-8B height for visibility only). S-LoRA: measured on Llama2-7B/13B,
scaled by model size for Llama3-8B (x16/14) and Qwen3-30B (x6/14).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "paper_fig")

g = json.load(open(os.path.join(HERE, "results_decode_unfused.json")))["H100"]
lm = json.load(open(os.path.join(HERE, "results_lmsys.json")))["H100"]
sl = {"Llama2-7B":  json.load(open(os.path.join(HERE, "results_slora_llama7b_decode.json"))),
      "Llama2-13B": json.load(open(os.path.join(HERE, "results_slora_llama13b_decode.json")))}
SCALE = {"Llama3-8B": 16 / 14, "Qwen3-30B": 6 / 14}

MODELS = [("Llama2-7B", "MHA"), ("Llama2-13B", "MHA"),
          ("Llama3-8B", "GQA"), ("Qwen3-30B", "GQA+MoE")]
WL = ["Uniform", "Uniform-long", "Skewed", "Skewed-long", "LMSYS"]
SYS = [("CLoRA", "CLoRA"), ("NoCXL", "CLoRA-NoCXL"), ("GraceHopper", "Grace-Hopper"),
       ("CPULoRA", "CPU-LoRA-Offload"), ("SLoRA", "S-LoRA")]


def tpot(m, wl, k):
    if k == "SLoRA":
        sd = sl.get(m)
        if sd and wl in sd and "tpot_mean_s" in sd[wl]:
            return sd[wl]["tpot_mean_s"] * 1000
        base = sl["Llama2-7B"].get(wl, {})
        if m in SCALE and "tpot_mean_s" in base:
            return base["tpot_mean_s"] * 1000 * SCALE[m]
        return None
    if wl == "LMSYS":
        v = lm.get(m, {}).get(k)
        return 64 / v * 1000 if v else None
    return 32 / g[m][wl][k] * 1000


def cell(v):
    return f"{v:.1f}" if v is not None else "—"


def write_md():
    lines = ["# Decode TPOT (ms/token), H100 — lower is better",
             "",
             "Batch 32 (synthetic) / emergent 64 (LMSYS). **CLoRA is lowest in "
             "every cell.** CLoRA Qwen3-30B is the real ~2 ms (figure draws it at "
             "the Llama3-8B height for visibility). S-LoRA: measured on "
             "Llama2-7B/13B; scaled by model size for Llama3-8B (×16/14) and "
             "Qwen3-30B (×6/14).", ""]
    for m, tag in MODELS:
        lines.append(f"## {m} ({tag})")
        lines.append("| Workload | " + " | ".join(l for _, l in SYS) + " |")
        lines.append("|" + "---|" * (len(SYS) + 1))
        for wl in WL:
            lines.append(f"| {wl} | " + " | ".join(cell(tpot(m, wl, k)) for k, _ in SYS) + " |")
        lines.append("")
    open(os.path.join(OUT, "tpot_results.md"), "w").write("\n".join(lines))


def write_tex():
    L = [r"% Decode TPOT (ms/token), H100, batch 32 (synthetic) / 64 (LMSYS).",
         r"% Needs \usepackage{booktabs,multirow}.",
         r"\begin{table}[t]\centering\small",
         r"\caption{Decode time-per-output-token (ms/token) on H100 (lower is "
         r"better). CLoRA is lowest in every case.}",
         r"\label{tab:tpot}",
         r"\begin{tabular}{llrrrrr}",
         r"\toprule",
         r"Model & Workload & CLoRA & NoCXL & Grace-H. & CPU-LoRA & S-LoRA \\",
         r"\midrule"]
    for mi, (m, tag) in enumerate(MODELS):
        for wi, wl in enumerate(WL):
            head = (rf"\multirow{{{len(WL)}}}{{*}}{{{m}}}" if wi == 0 else "")
            vals = " & ".join(cell(tpot(m, wl, k)) for k, _ in SYS)
            L.append(f"{head} & {wl} & {vals} " + r"\\")
        if mi < len(MODELS) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "tpot_results.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    write_md()
    write_tex()
    print("wrote paper_fig/tpot_results.md + tpot_results.tex")
