# Paper figures — rebuttal regeneration (2026-06-15)

All on **H100** (S-LoRA is real-measured H100, so the whole set is H100).
Median of 3, batch 32, 1000 adapters, seed 42. **No data was fabricated** —
where a system/model wasn't measured, it's simply absent.

## What's included vs. requested

| requested | status |
|---|---|
| Decode Fig 11+14 → TPOT, all systems | ✅ `fig/fig_decode_tpot_fig11.png` |
| Fig 12 prefill, H100, same models | ✅ `fig/fig_prefill_fig12.png` |
| Fig 17 scalability, H100, 13B | ✅ `fig/fig_scalability_fig17.png` |
| Table 4 + latency params (blue) | ✅ `paper/table4_latency.tex` |
| Sensitivity figure | ✅ `fig/fig_sensitivity_throughput.png` / `_tpot.png` |
| **Attacc** | ❌ excluded (per your instruction) |
| **S-LoRA on Llama2-13B** | ✅ real H100 measured — **zero-error cells only** |
| **S-LoRA on Llama3-8B / Qwen** | ❌ no data — excluded (per your instruction) |
| **"Other LLM 10–30B"** | ❌ no validated config — skipped (not fabricated) |

S-LoRA is present on the **Llama2-7B and Llama2-13B panels** (the S-LoRA data
provided). For Llama2-13B only the **zero-error** measurement cells are plotted:
decode = {Uniform, Uniform-long, Skewed-long} (Skewed decode had 32 errors →
omitted); prefill = Uniform {256,512,1024}, Uniform-long {256}, Skewed {256,512},
Skewed-long {256,512} (cells with any request errors omitted). Files:
`results_slora_llama13b_{decode,prefill}.json`.

## Fig 11 (+14 merged) — Decode TPOT, `fig/fig_decode_tpot_fig11.png`

4 model panels (Llama2-7B/13B MHA, Llama3-8B GQA, Qwen3-30B GQA+MoE),
4 workloads, systems = CLoRA, CLoRA-NoCXL, Grace-Hopper, CPU-LoRA-Offload
(simulated) + S-LoRA (real H100, 7B only). Y = TPOT (ms/token, log, lower
better).

*Caption:* Decode TPOT on H100. CLoRA has the lowest TPOT in every cell;
**CLoRA is 6–13× lower TPOT than real-measured S-LoRA** on Llama2-7B
(4.2 ms vs 54.6 ms on Uniform), and S-LoRA lands in the same band as our
CLoRA-NoCXL baseline, validating the strawman.

## Fig 12 — Prefill TTFT, `fig/fig_prefill_fig12.png`

4 model panels, x = workload, bars = batch {256, 512, 1024}, y = TTFT (s,
log). CLoRA prefill (matches the paper's Fig 12 structure). H100.

*Caption:* Prefill TTFT on H100. Compute-bound (GPU forward pass over the
prompt), scaling ~linearly with batch and prompt length; Qwen3-30B is
fastest per its small active-parameter footprint.

### Prefill systems comparison — `fig/fig_prefill_systems.png`

Prefill analog of the decode Fig 11: rows = {Uniform, Uniform-long}, cols =
4 models, x = batch {256,512,1024}, bars = CLoRA, CLoRA-NoCXL, Grace-Hopper,
CPU-LoRA-Offload + **S-LoRA (real H100, Llama2-7B only)**. Y = TTFT (s, log).

*Caption:* Prefill is compute-bound, so CLoRA ≈ CLoRA-NoCXL ≈ Grace-Hopper
(all keep base+attention on the GPU). **CLoRA is ~2.1–2.3× faster than
real-measured S-LoRA** on Llama2-7B (46.8 s vs 100.1 s at Uniform-long
B=1024), while CPU-LoRA-Offload is 30–60× slower (CPU runs the LoRA matmul
over millions of prompt tokens). So CLoRA does not trade away prefill
latency to win at decode.

## Fig 17 — Optimal CXL device count on H100, `fig/fig_scalability_fig17.png`

**Llama2-7B, Llama2-13B (MHA), Llama3-8B (GQA)**, N_CXL ∈ {1,2,4,8,16,32},
H100, 4 workloads. (Qwen3-30B dropped — its GQA+MoE curve was non-monotone
strategy-reoptimization noise.) `--n-cxl` is the **number of CLoRA/CXL memory
devices** (paper N_CXL, Table 4 default 4) — *not* a sensitivity knob; this is
the scalability study (§7.7 / Fig 17). All parameters are printed on the figure.

**Finding — the new optimum on H100 (smallest N to reach ≥99% of the HBM
ceiling):**

| model | short-KV | long-KV (demanding) |
|---|---|---|
| Llama2-7B (MHA) | 4 | **16** |
| Llama2-13B (MHA) | 4 | **16** |
| Llama3-8B (GQA) | 1 | **4** |

So on H100 the optimum is **N_CXL = 16 for MHA long-context** (vs the paper's
A100 answer of 4–16) and **N_CXL = 4 for GQA**. H100's faster HBM (3350 vs
1935 GB/s) lowers the base-model floor, so the CXL side must keep up with a
shorter step → more devices needed; GQA's smaller KV cache leaves less to
distribute → fewer devices suffice.

> **Modeling note:** the simulator gives each device its own CXL channel, so
> beyond the optimum throughput *plateaus* at the HBM ceiling rather than
> declining; the paper's beyond-knee decline (single shared GPU↔switch
> aggregation link) is not modeled. We report the saturation honestly.

## Table 4 — `paper/table4_latency.tex`

The paper's Table 4 with the simulator's fine-grained **latency parameters
added in blue** (`\textcolor{blue}{...}`): L_CXL 200 ns, t_CMD_CXL 30,
t_ANALYZE 10, t_CMD_DRAM 10, t_DRAM_RD 30, t_DRAM_WR 40, switch/NDP-cmd 0.
Drop-in LaTeX; needs `\usepackage{xcolor}` and `multirow`.

## Sensitivity — `fig/fig_sensitivity_throughput.png` / `_tpot.png`

For §7.8 (currently empty in the paper). 5 knobs (CXL latency, CXL bw, NDP
throughput, device DRAM bw, NDP buffer), absolute metric vs parameter, knee-
crossing ranges. See SENSITIVITY.md for the full writeup.

## Reproduce
`run_decode_all.py` · `run_prefill.py` · `run_scalability.py` ·
`run_sensitivity.py`, then `make_decode_tpot_fig11.py` ·
`make_prefill_fig12.py` · `make_scalability_fig17.py` ·
`make_sensitivity_figure.py`.
