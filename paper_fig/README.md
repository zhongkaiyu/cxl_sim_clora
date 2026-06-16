# `paper_fig/` — conference-ready figures (Overleaf drop-in)

Vector **PDF** (for LaTeX `\includegraphics`) + **PNG** preview for every figure.

**House style** (uniform across all figures, `_style.py`):
- One font size everywhere (9 pt, serif, `pdf.fonttype=42` so text stays
  selectable/embeddable in Overleaf).
- **No titles.** Each panel is identified by a **sub-caption at the bottom**
  — `(a) Llama2-7B (MHA)`, etc.
- Axis-title units use **superscript exponents**, not slashes:
  `TPOT (ms·token⁻¹)`, `Throughput (tokens·s⁻¹)`.
- **No value labels on top of bars.**
- Parallel multi-panel layout; legend as a single row above the panels.

**Honesty policy (no fabricated data).** Only systems/models we actually
measured are drawn. The following are intentionally **absent** (not faked):
- **Attacc** — not implemented (HBM-PIM attention accelerator, ASPLOS'24).
- **S-LoRA** on Llama3-8B / Qwen3-30B / LMSYS — not measured.
- **"Other LLM 10–30B"** — no validated config.

---

## Fig. `fig_decode_tpot` — Decode TPOT (paper Fig 11 + 13 merged)

4 model panels × 5 workloads (Uniform, Uniform-long, Skewed, Skewed-long,
**LMSYS**), grouped bars = **CLoRA, CLoRA-NoCXL, Grace-Hopper,
CPU-LoRA-Offload, S-LoRA (real H100)**. Y = TPOT (ms·token⁻¹), log scale, ↓
lower is better. H100. Synthetic workloads use batch 32 / 1000 adapters; LMSYS
is the real Chatbot Arena trace (25 adapters, Vicuna ≈49% popularity, emergent
per-adapter batch ≤256) at operating batch 64.

S-LoRA bars appear only on the Llama2-7B/13B synthetic cells (the only
real-measured S-LoRA we have).

**LaTeX caption:**
> Decode time-per-output-token (TPOT) on H100 across four models and five
> workloads (lower is better). CLoRA achieves the lowest TPOT in every cell,
> sitting at the HBM-bandwidth floor; the CXL-less, host-offload, and CPU
> baselines are 1–2 orders of magnitude slower, and real-measured S-LoRA
> lands in the CLoRA-NoCXL band, validating the strawman.

**Sub-captions:** (a) Llama2-7B (MHA) · (b) Llama2-13B (MHA) ·
(c) Llama3-8B (GQA) · (d) Qwen3-30B (GQA+MoE).

---

## Fig. `fig_prefill_ttft` — Prefill TTFT (paper Fig 12, H100)

**CLoRA only** — how prefill TTFT varies across **models × workloads × batch
sizes** (not a system comparison). 4 model panels; x = 4 workloads (Uniform,
Uniform-long, Skewed, Skewed-long); grouped bars = batch {256, 512, 1024}.
Y = TTFT (s), log, ↓ lower is better. H100, GPU forward pass at **75% MFU**
(realistic model-FLOP utilization). Same model set as the decode figure.

**LaTeX caption:**
> CLoRA prefill time-to-first-token (TTFT) on H100 across models, workloads,
> and batch sizes (lower is better). Prefill is compute-bound, so TTFT scales
> roughly linearly with batch and with prompt length (long-context workloads
> Unif-L / Skew-L are ~3–6× the short-context ones); Qwen3-30B is fastest per
> its small active-parameter (MoE) footprint.

**Sub-captions:** (a) Llama2-7B (MHA) · (b) Llama2-13B (MHA) ·
(c) Llama3-8B (GQA) · (d) Qwen3-30B (GQA+MoE).

---

## Fig. `fig_scalability` — CXL device-count scalability (paper Fig 17, H100)

**Llama2-13B (MHA)**, N_CXL ∈ {1,2,4,8,16,32}, 4 workloads. Y = decode
throughput (tokens·s⁻¹), ↑. Dotted line = paper's default N_CXL = 4.

**New sweet point (H100, smallest N_CXL reaching ≥99% of the HBM ceiling
4,123 tok/s):**

| workload | sweet point |
|---|---|
| Uniform (short-KV)      | **N_CXL = 4** |
| Skewed (short-KV)       | **N_CXL = 4** |
| Uniform-long (long-KV)  | **N_CXL = 16** |
| Skewed-long (long-KV)   | **N_CXL = 16** |

So on H100 the demanding long-context workloads need **16 devices** (vs the
paper's A100 answer of 4–16): H100's faster HBM (3350 vs 1935 GB/s) lowers the
base-model floor, so the CXL side must keep up with a shorter step → more
devices. Short-context is saturated by the paper default of 4.

**Qwen3-30B excluded from the figure (reported in text):** its MoE active
footprint is only 6 GB, so the GPU HBM ceiling (≈17.9k tok/s) is already
saturated at **N_CXL = 1–2**. Beyond that, more devices cannot help and the
cost model's strategy re-selection adds un-hidden CXL overhead, making the
curve non-monotone (noise, not a scaling trend). **Finding:** MoE
active-parameter models need only **1–2 CXL devices**; the data is in
`results_scalability.json` for reference.

**Modeling note (state in text):** each device gets its own CXL channel, so
throughput **saturates** at the HBM ceiling past the knee rather than
declining; the paper's beyond-knee drop comes from a single shared
GPU↔switch aggregation link, which we do not model. We report the saturation
honestly.

**LaTeX caption:**
> Decode throughput vs number of CXL memory devices on H100, Llama2-13B
> (higher is better). Long-context (Uniform-long / Skewed-long) workloads
> benefit from more devices up to N_CXL = 16, after which throughput saturates
> at the GPU HBM ceiling; short-context workloads saturate by N_CXL = 4.

**Sub-caption:** (a) Llama2-13B (MHA).

---

## Fig. `fig_sensitivity` — Hardware sensitivity (paper §7.8)

5 knobs in a row: (a) CXL link latency, (b) CXL link bandwidth, (c) NDP core
throughput, (d) device DRAM bandwidth, (e) NDP controller buffer. Absolute
decode throughput (tokens·s⁻¹, ↑) vs the parameter; two workload lines
(Uniform short-KV, Uniform-long long-KV). Dotted line = the value used in
Table 4. HW-only sweep (Llama2-7B, A100, fixed cost model) so curves are clean
and monotonic.

**LaTeX caption:**
> Sensitivity of decode throughput to each CLoRA hardware parameter (ranges
> chosen to cross every knee). Device DRAM bandwidth is the dominant knob and
> NDP compute shows a sharp cliff below ~2 TFLOPS; CXL link bandwidth saturates
> by ~16 GB/s. CXL link latency and NDP controller buffer are flat across the
> swept range — neither is the bottleneck at the Table 4 operating point.

**Sub-captions:** (a)–(e) as above.

---

## Table 4 — `table4.tex`

The paper's *CLoRA System Configuration Details* with the simulator's
fine-grained **latency parameters added in blue** (`\textcolor{blue}{…}`):
L_CXL 200 ns, t_CMD,CXL 30, t_ANALYZE 10, t_CMD,DRAM 10, t_DRAM,RD 30,
t_DRAM,WR 40, switch/NDP-cmd 0. Requires `\usepackage[table]{xcolor}` and
`\usepackage{multirow}`.

---

## Reproduce

```
# data (run from repo root)
python script/run_decode_all.py     # synthetic decode grid -> results_decode_unfused.json
python script/run_lmsys.py          # LMSYS trace          -> results_lmsys.json
python script/run_prefill.py        # prefill TTFT (75% MFU)-> results_prefill.json
python script/run_scalability.py    # 13B/30B device sweep -> results_scalability.json
python script/run_sensitivity.py    # HW knobs             -> sensitivity_abs.json

# figures
python paper_fig/fig_decode_tpot.py
python paper_fig/fig_prefill_ttft.py
python paper_fig/fig_scalability.py
python paper_fig/fig_sensitivity.py
```
