# MICRO 2026 #785 — Rebuttal Round 2 (combined decode TPOT + prefill TTFT)

All numbers median of 3, batch/seed as noted, produced with the existing
methodology (CLoRA → C event sim, baselines → Python analytical). S-LoRA
and Attacc are excluded per request, leaving four systems: **CLoRA,
CLoRA-NoCXL, Grace-Hopper, CPU-LoRA-Offload**.

## Deliverables

| Figure | File | Replaces / adds |
|---|---|---|
| Combined decode **TPOT** | `fig/fig_tpot_combined_unfused.{png,pdf}` (primary) + `_fused` | merges old Fig 11 + Fig 14 + MoE |
| Prefill **TTFT** (Fig 12) | `fig/fig_prefill_combined_unfused.{png,pdf}` (primary) + `_fused` | regenerated on H100, 4 models |

**Primary = unfused NoCXL** (per-adapter/per-request launches) in all
figures; `_fused` (BGMV) kept as the alternative. See the "Revised
CLoRA-NoCXL launch model" section below and RESULTS.md §4–§5.

Raw data: `script/results_decode_all.json`, `script/results_prefill.json`.
Regenerate: `run_decode_all.py` / `run_prefill.py`, then
`make_tpot_figure.py` / `make_prefill_figure.py`.

## Models & systems (both figures)

- **Llama2-7B (MHA)**, **Llama2-13B (MHA)**, **Llama3-8B (GQA, 4×)**,
  **Qwen3-30B-A3B (GQA 8× + MoE)**. GQA emulated as a KV-token divisor
  (exact in KV bytes); MoE via the active-parameter (6 GB) HBM read.
- 4 systems above. Decode: batch 32, 1000 adapters, both A100 40GB and
  H100 80GB. Prefill: batch {256, 512, 1024}, prompt lengths from the
  workload input ranges, H100.

## Decode figure — Y-axis = TPOT (ms/token, = batch/throughput, lower better)

CLoRA has the lowest TPOT in all 32 cells. Representative H100 numbers:

| model | workload | CLoRA | NoCXL | Grace-Hopper | CPU-LoRA |
|---|---|--:|--:|--:|--:|
| Llama2-7B | Uniform | **4.2** | 4.6 | 39 | 171 |
| Qwen3-30B | Skewed | **2.0** | 2.7 | 37 | 200 |
| Llama3-8B | Uniform | **4.8** | 5.2 | 20 | 99 |

Notes worth putting in the rebuttal:
- **Qwen3-30B now GQA+MoE**: with 8× GQA its long-context TPOT is nearly
  flat vs short (A100 3.7 vs 3.3 ms), unlike the earlier MoE-only-with-MHA
  version that degraded badly on long KV.
- **Llama3-8B (GQA) is HBM-floor-bound**: identical TPOT across all four
  workloads (the KV is small enough to hide under the base-model floor);
  the systems separate on the baseline side, not on CLoRA.

## Prefill figure (Fig 12) — Y-axis = TTFT (seconds, lower better)

Modeled per paper §5.2: prefill attention runs on the **GPU**
(FlashAttention); the freshly built KV cache is **written** to CXL. So
prefill is compute-bound and the GPU forward pass dominates TTFT.

Result (honest, paper-consistent): **CLoRA ≈ CLoRA-NoCXL ≈ Grace-Hopper**
on prefill — all three keep base + attention on the GPU, so they share the
forward pass and differ only marginally (CLoRA hides its LoRA offload + KV
write under the forward pass; Grace-Hopper pays a bit more because LoRA
runs on the same GPU and KV crosses C2C). **CPU-LoRA-Offload is 30–400×
worse** because the CPU runs the LoRA matmul over millions of prompt
tokens. Example (Llama2-7B, Uniform-long, B=1024): CLoRA 46.8 s, NoCXL
46.8 s, Grace-Hopper 51.1 s, CPU-LoRA 2,873 s.

The takeaway for the prefill story: CLoRA does **not** sacrifice prefill
TTFT to win at decode — it stays on par with the strong GPU-resident
baselines while crushing CPU offload.

## Revised CLoRA-NoCXL launch model (2026-06-14)

NoCXL overhead now scales with **serving adapters and requests**, not a flat
2 offloads/layer. Per decoder layer the GPU↔NDP offload boundaries are:

    Proj QKV : 1 launch / serving adapter   (fused Q,K,V LoRA)
    Proj O   : 1 launch / serving adapter
    FFN      : ffn_gemms launches / adapter  (SwiGLU=3; Qwen MoE = 3×8 = 24)
    Attention: 1 launch / request

    launches/layer = n_adapters·(2 + ffn_gemms) + n_requests
    overhead = launches/layer · n_layers · 7 µs   (kernel+cmd+sync per launch)

Rationale: without CXL.mem's fine-grained load/store the GPU cannot fuse
different adapters' remote NDP ops into one kernel, so each adapter (and each
request's attention) is a separate launch. This is the realistic penalty of
losing the CXL interface.

**Consequence — the 4-way ordering changes.** The launch overhead is large
enough (e.g. 7B decode: 32 adapters × 5 + 32 = 192 launches/layer × 32 layers
× 7 µs ≈ 43 ms, vs a 4 ms CLoRA step) that NoCXL falls **below Grace-Hopper**
in 12/16 H100 decode cells, and below CPU-LoRA in the 2 Qwen-Skewed cells.
Grace-Hopper keeps its lead because it runs LoRA on the GPU with S-LoRA-style
BGMV fusion (one launch per rank-group, not per adapter). So the new story is:
**CLoRA ≫ {NoCXL, Grace-Hopper} ≫ CPU-LoRA**, with NoCXL vs GH depending on
workload — NoCXL wins on long-MHA (where GH's KV-over-C2C dominates), GH wins
on short/GQA/MoE (where NoCXL's per-adapter launches dominate). This makes the
CXL interface a *first-order* contribution (CLoRA/NoCXL now 6–130×), not the
~5% it was under the old flat model.

Two modeling choices worth confirming (see end):
- per-adapter (unfused) launches for NoCXL — vs allowing NoCXL to BGMV-fuse;
- Qwen MoE `ffn_gemms = 24` (3 gemms × 8 active experts) — vs a logical 3.

## C-simulator changes for prefill (authorized)

1. `main.c` — `emit_attention_for_adapter` gains a `kind==prefill` branch:
   KV is **written** to CXL (distributed across devices) instead of the
   decode-time read-compute; LoRA path uses the effective token count.
2. **64-bit widening** (`addr_info`, `request`/`sub_request` byte fields,
   flash.c accumulators): prefill moves tens of GB per step, which
   overflowed the old 32-bit byte fields. Decode values are small, so
   **decode is byte-identical** (verified: 7B Uniform A100 still 4,422.9).
3. `npu_request.{begin,end}_time` widened `unsigned int → int64_t`: a
   32-bit ns field wraps at ~2.15 s, which silently truncated any prefill
   step longer than that. (Decode steps are sub-ms, so this never showed.)

Regression: all **110** Python tests pass; decode grid regenerated on the
clean binary is identical to before.

> **Build note:** the Makefile does not track header dependencies — after
> editing any `include/*.h`, run `make clean && make all`, or stale object
> files silently produce wrong results.

## Still open

- **Workload coverage in the figures**: decode uses the 4 synthetic
  workloads (LMSYS intentionally omitted, per earlier instruction); say
  the word to add an LMSYS-approximation column.
- **Prefill attention placement** was set to GPU-FlashAttention (your
  choice). If you ever want the stronger "distributed-NDP prefill" claim,
  that's a different model.
