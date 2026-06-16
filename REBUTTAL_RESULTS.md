# MICRO 2026 #785 — Rebuttal Experiments & Figures

> **⚠️ SUPERSEDED (2026-06-14).** This round-1 doc uses throughput-axis
> figures (`fig_decode_a100/h100`, `fig_gqa` — now removed) and the old
> flat NoCXL launch model. The current results live in
> [`REBUTTAL_R2.md`](./REBUTTAL_R2.md) (decode TPOT + prefill TTFT, both
> NoCXL settings) and [`RESULTS.md`](./RESULTS.md) (unfused NoCXL numbers).
> Kept for history; numbers below are stale.

All results below were produced with the existing simulator and driver
(no code changes); every number is the median of 3 trials, 10 warmup +
10 measured decode steps, batch 32, seed 42, on the four synthetic
workloads (Uniform / Uniform-long / Skewed / Skewed-long, 1000
adapters). Raw data: `script/results_grid.json` (A100),
`script/results_h100.json`, `script/results_gqa.json`,
`script/sensitivity_results.json`. Regenerate figures with
`python3 script/make_rebuttal_figures.py` → `fig/*.{png,pdf}`.

## SE-item status

| SE | item | status | artifact |
|---|---|---|---|
| 1 | latency parameters (CXL switch, CPU-GPU access) | done | revised Table 4 below; `L_CXL_switch`, `L_read_compute_cmd` (event sim); `L_pcie_setup`, `L_kernel_launch`, `L_cpu_gpu_offload` (CPU baseline) |
| 2 | CPU compute offloading baseline | done | `script/baseline_cpu_offload.py`; Figs R1/R2/R3 |
| 3 | without-CXL baseline (kernel overhead) | done | `script/baseline_no_cxl.py`; Figs R1/R2/R3 |
| 4 | Grace-Hopper superchip (450 GB/s) | done | `script/baseline_grace_hopper.py`; Figs R1/R2/R3 |
| 5 | MoE model (Qwen3-30B-A3B) | done | panel (c) of Figs R1/R2 |
| 6 | sensitivity: CXL latency / CXL bw / NDP throughput / NDP buffer | done | Fig R4; SENSITIVITY.md |

---

## Revised Table 4 — system configuration with all latency parameters

### Markdown version

**GPU (two evaluated variants)**

| parameter | A100 SXM | H100 SXM | unit |
|---|---:|---:|---|
| FP16 dense compute | 312 | 989 | TFLOPS |
| HBM bandwidth | 1,935 | 3,350 | GB/s |
| HBM capacity | 40 | 80 | GB |

**CLoRA memory device (×4, CXL 3.1)**

| parameter | value | unit |
|---|---:|---|
| capacity / device | 512 | GB |
| CXL link bandwidth / device | 128 | GB/s |
| device DRAM bandwidth (8 pkg × 136 GB/s) | 1,088 | GB/s |
| NDP compute / device (8 TFLOPS aggregate) | 2 | TFLOPS FP16 |
| NDP on-chip buffer | 3 | MB |
| controller request buffer (8 × 128 B in-flight) | 1,024 | B |

**CXL link & switch timing (event simulator)**

| parameter | value | unit | note |
|---|---:|---|---|
| `t_CMD_CXL` — command transfer on CXL link | 30 | ns | per request |
| `t_ANALYZE` — controller request decode | 10 | ns | per request |
| `t_CMD_DRAM` — controller→DRAM command | 10 | ns | per request |
| `t_DRAM_READ_LATENCY` | 30 | ns | + bytes / W_DRAM |
| `t_DRAM_WRITE_LATENCY` | 40 | ns | + bytes / W_CXL |
| `L_CXL_switch` — switch traversal (one-way) | 0 (default); swept 0–1,600 | ns | applied per CA transfer and per data return (2× per round trip) |
| `L_read_compute_cmd` — NDP command setup | 0 (default); sensitivity knob | ns | added to `t_ANALYZE` for read-compute |
| `L_CXL` — link latency in the cost model (Eq 8: 2·L_CXL per attention round trip) | 200 | ns | strategy selection |

**PCIe + CPU-offload baseline (SE-2)**

| parameter | value | unit | note |
|---|---:|---|---|
| PCIe effective bandwidth | 128 | GB/s | matched to CXL link for fairness (native PCIe 5.0 ×16 ≈ 58) |
| `L_pcie_setup` — per-DMA setup | 1,000 | ns | CPU-GPU data access latency |
| `L_kernel_launch` — CUDA kernel launch | 5,000 | ns | per launch; 2 per offloaded op |
| `L_cpu_gpu_offload` — sync + dependent-kernel round trip | 10,000 | ns | per GPU↔CPU round |
| CPU FP16 matmul throughput | 200 | GFLOPS | single socket, no AMX (conservative) |
| CPU DRAM bandwidth | 100 | GB/s | 8-ch DDR4-3200 sustained |
| GPU LRU adapter cache | = CLoRA GPU budget | GB | memory parity (26 / 14 / 34 / 66 / 54 / 74 / 24 per config) |

**Without-CXL baseline (PCIe-NDP, SE-3)** — identical hardware to CLoRA;
each GPU↔NDP boundary costs `L_kernel_launch` (5,000) +
`L_device_command` (1,000) + `L_sync` (1,000) = **7,000 ns**, ×2
boundaries (batched LoRA + attention) × n_layers per step.

**Grace-Hopper baseline (SE-4)**

| parameter | value | unit |
|---|---:|---|
| NVLink-C2C bandwidth (per direction) | 450 | GB/s |
| `L_c2c_setup` — per-transfer setup (coherent) | 500 | ns |
| `L_kernel_launch` | 5,000 | ns |
| LoRA matmul on | GPU (312 / 989 TFLOPS) | — |
| GPU LRU adapter cache | = CLoRA GPU budget | GB |

### LaTeX version (drop-in for Table 4)

```latex
\begin{table}[t]\centering\small
\caption{CLoRA system configuration, including all latency parameters.}
\begin{tabular}{llr}
\toprule
\textbf{GPU} & A100 SXM (40\,GB) & 312 TFLOPS, 1935 GB/s \\
             & H100 SXM (80\,GB) & 989 TFLOPS, 3350 GB/s \\
\midrule
\textbf{CLoRA device} & capacity / CXL link & 512 GB, 128 GB/s \\
(\(\times\)4)         & device DRAM BW      & 1.1 TB/s \\
                      & NDP compute / buffer & 2 TFLOPS FP16, 3 MB \\
\midrule
\textbf{CXL timing}   & $t_\mathrm{CMD,CXL}$ / $t_\mathrm{ANALYZE}$ / $t_\mathrm{CMD,DRAM}$ & 30 / 10 / 10 ns \\
                      & $t_\mathrm{DRAM,RD}$ / $t_\mathrm{DRAM,WR}$ & 30 / 40 ns \\
                      & $L_\mathrm{CXL switch}$ (one-way; swept) & 0--1600 ns \\
                      & $L_\mathrm{CXL}$ (cost model, Eq.~8) & 200 ns \\
\midrule
\textbf{PCIe+CPU}     & PCIe BW (matched) & 128 GB/s \\
 baseline             & $L_\mathrm{PCIe setup}$ / $L_\mathrm{kernel}$ / $L_\mathrm{offload}$ & 1 / 5 / 10 $\mu$s \\
                      & CPU compute / DRAM BW & 200 GFLOPS, 100 GB/s \\
\midrule
\textbf{No-CXL}       & per GPU$\leftrightarrow$NDP boundary & 7 $\mu$s ($\times$2/layer) \\
\midrule
\textbf{Grace-Hopper} & NVLink-C2C / $L_\mathrm{C2C}$ / $L_\mathrm{kernel}$ & 450 GB/s, 0.5 / 5 $\mu$s \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Figures

### Fig R1 (replaces Fig 11) — `fig/fig_decode_a100.{png,pdf}`

Decode throughput on A100 40GB: 4 systems × 4 workloads × 3 models
(Llama2-7B, Llama2-13B, **Qwen3-30B-A3B MoE** — SE-5), 1000 adapters.

Suggested caption: *Decode throughput of CLoRA and the three new
baselines (CLoRA-NoCXL, Grace-Hopper, CPU-LoRA-Offload) on A100 across
four workloads and three base models. All baselines receive a GPU
adapter cache equal to CLoRA's GPU memory budget. CLoRA leads every
cell: over Grace-Hopper by 3.8–11.8× (7B) and 13–21× (MoE); over
CPU-LoRA-Offload by 17.7–54× (7B); the CXL interface itself (vs NoCXL)
contributes 4–18%.*

Key numbers: 7B Uniform 4,423 vs 4,165 / 819 / 173 tok/s; MoE Skewed
8,555 vs 7,252 / 506 / 97.

### Fig R2 (new: H100 80GB) — `fig/fig_decode_h100.{png,pdf}`

Same layout on H100 SXM 80GB (989 TFLOPS, 3,350 GB/s; GPU budget =
80 GB − base; parity caches). Devices/links unchanged, isolating the
GPU generation.

Suggested caption: *CLoRA scales with the GPU: moving A100→H100 lifts
average 7B throughput 1.54× (3,634 → 5,582 tok/s); short-KV throughput
reaches the H100 HBM floor (7,657 tok/s for 7B — consistent with the
7,928 tokens/s cited in the abstract). The four-way ordering is
unchanged; the CXL-interface contribution grows to ~9–11% because the
faster GPU shortens the step that NoCXL's fixed 448 µs kernel-launch
overhead is amortized over.*

Key numbers: 7B Uniform 7,657 / 6,916 / 887 / 187; MoE Skewed 10,662 /
8,712 / 647 / 124. KV-duplication rises to 26–35% on long-KV (more HBM
headroom), still policy-chosen, not greedy.

### Fig R3 (replaces Fig 14) — `fig/fig_gqa.{png,pdf}`

Llama3-8B with GQA on A100 (same D=4096 / 32 layers as Llama2-7B; base
16 GB; GQA's 4× smaller KV emulated by 4× fewer KV tokens — exact in
bytes, second-order in compute since attention is bandwidth-dominated
here).

Suggested caption: *With GQA, KV traffic shrinks 4× and every system
becomes less memory-bound. CLoRA pins the HBM floor (3,870 tok/s in all
four workloads, base-model-bound). The honest observation: GQA narrows
CLoRA's advantage (1.6–5.7× over Grace-Hopper, 7.4–27× over CPU
offload) precisely because it shrinks the traffic CLoRA optimizes —
CLoRA's value concentrates where memory pressure lives (MHA, long
contexts, MoE).*

**Presentation note vs the paper's Fig 14:** the paper says Llama3-8B
delivers *higher* throughput than Llama2-7B; at our batch-32 operating
point CLoRA's absolute number is slightly *lower* (3,870 vs 4,423)
because the 16 GB base raises the HBM floor while the KV savings were
already hidden under it. The GQA benefit shows up in the *baselines*
(Grace-Hopper roughly doubles). Worth one sentence in the rebuttal
rather than glossing over.

### Fig R4 (new: sensitivity, SE-6) — `fig/fig_sensitivity.{png,pdf}`

2×2 normalized-throughput panels: CXL switch latency (0→+1,600 ns),
CXL link bandwidth (32→512 GB/s), NDP throughput (0.25→8 TFLOPS),
NDP controller buffer (1→32 in-flight requests); Uniform and
Uniform-long, Llama2-7B A100.

Suggested caption: *CLoRA is insensitive to CXL switch latency (≤0.2%
hardware-only effect at +1,600 ns) and link bandwidth (<2% over a 16×
range) — by design, since NDP offload and distributed attention remove
the link from the critical path. NDP core throughput is the one real
cliff (−22% at 0.5 TFLOPS, −44% at 0.25); the provisioned 2
TFLOPS/device sits at the knee. Controller buffer depth is flat within
~1% from 1 to 32 outstanding requests.*

Full decomposition (hardware effect vs strategy-selector response) and
methodology notes are in `SENSITIVITY.md`.

---

## Items needing your input

1. **Fig 12 (prefill / TTFT) cannot be redrawn without code changes** —
   the driver and JSON bridge currently model decode steps only
   (`input_tokens_per_request=1`; no prefill path is exercised). Options:
   (a) keep the paper's original Fig 12, (b) green-light a prefill mode
   in the driver (roughly a day of work), or (c) drop Fig 12 in the
   revision and address TTFT in text.
2. **LMSYS is intentionally omitted** (per your instruction): all
   figures use only the four synthetic 1000-adapter workloads. The
   driver could approximate LMSYS synthetically (25 adapters, skewed,
   KV 2–430) if you change your mind, but it would not be a trace
   replay.
3. **GQA emulation + the Fig 14 divergence** described above — confirm
   you're comfortable presenting Llama3-8B as "HBM-floor-bound, smaller
   CLoRA-vs-baseline gap" rather than the original "higher throughput
   than Llama2-7B" framing.
4. **H100 figure uses 80 GB budgets** (66/54/74 GB after base) and
   keeps CXL/PCIe/C2C identical to the A100 runs so the GPU is the only
   variable — confirm that matches what you want for "new GPUs".
