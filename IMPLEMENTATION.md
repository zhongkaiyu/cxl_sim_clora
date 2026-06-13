# CLoRA Simulator — Implementation & Design Notes

This document is the **deep technical reference** for the simulator and
the four systems it models. It complements [`DESIGN.md`](./DESIGN.md):

- **`DESIGN.md`** answers *"how do I drive this?"* — architectures,
  knobs, recipes
- **`IMPLEMENTATION.md`** (this file) answers *"why is it built this way?
  what's exactly happening under the hood?"* — formulas, derivations,
  state-machine semantics, design decisions, edge cases, approximations

If you only have time for one, read `DESIGN.md`. If you need to **modify**,
**defend**, or **extend** the simulator, read both.

---

## Table of Contents

1. [Cost Model — formula-by-formula derivations](#1-cost-model--formula-by-formula-derivations)
2. [Memory accounting — bytes that matter](#2-memory-accounting--bytes-that-matter)
3. [Algorithm 1 — step-by-step implementation](#3-algorithm-1--step-by-step-implementation)
4. [Temperature dynamics](#4-temperature-dynamics)
5. [Hot-adapter pre-caching](#5-hot-adapter-pre-caching)
6. [KV-cache duplication policy](#6-kv-cache-duplication-policy)
7. [C event simulator — state-machine specifics](#7-c-event-simulator--state-machine-specifics)
8. [JSON contract between Python and C](#8-json-contract-between-python-and-c)
9. [CPU-LoRA-Offload baseline](#9-cpu-lora-offload-baseline)
10. [CLoRA-NoCXL baseline](#10-clora-nocxl-baseline)
11. [Grace-Hopper Offload baseline](#11-grace-hopper-offload-baseline)
12. [Driver orchestration & steady state](#12-driver-orchestration--steady-state)
13. [Design decisions — the "why"s](#13-design-decisions--the-whys)
14. [Modeling approximations — what's exact, what's approximate](#14-modeling-approximations--whats-exact-whats-approximate)
15. [Edge cases handled](#15-edge-cases-handled)
16. [Verification approach](#16-verification-approach)

---

## 1. Cost Model — formula-by-formula derivations

The cost model lives in `script/clora_strategy.py::cost_lora`. It implements
paper §6.3 Equations (1)–(6) for ranking per-adapter strategies.

### Strategy E1 (cached in GPU HBM)

The LoRA path `y = x · A · B` runs entirely on the GPU. No CXL traffic.

```
FLOPs:    4 · B · D · R   (per matrix per layer; x·A: 2·B·D·R, then ·B: 2·B·R·D)
GPU mem:  2 · R · D · S   (read A and B from HBM, once per batch)

t_compute = 4·B·D·R / C_GPU
t_hbm     = 2·R·D·S / W_GPU
cost(E1)  = max(t_compute, t_hbm)
```

The **`max`** reflects that compute and HBM-read pipeline within the SM — the
slower of the two limits throughput. Paper Eq (1) writes this as
`max{4·Σ Bi1·D·Ri / C_GPU, 2·D·Ri·S / W_GPU}`.

### Strategy E2 (A,B in CXL, GPU loads via load/store)

GPU computes the matmul but A and B must arrive over the CXL link first.

```
t_compute = 4·B·D·R / C_GPU
t_link    = L_CXL + 2·R·D·S / W_CXL    (load A and B once per batch)
cost(E2)  = max(t_compute, t_link)
```

Paper Eq (1) for E2 uses `W_CXL` instead of `W_GPU` in the bandwidth term —
GPU is link-bound, not HBM-bound, in this configuration.

### Strategy E3 (NDP runs entire LoRA)

Only `x` goes out over the link; `y = xAB` returns.

```
link bytes = 2 · B · D · S   (x out: B·D·S, y back: B·D·S)
NDP FLOPs  = 4 · B · D · R   (full LoRA on NDP)

t_link    = L_CXL + 2·B·D·S / W_CXL
t_ndp     = 4·B·D·R / (C_CXL · N_CXL)
cost(E3)  = t_link + t_ndp     (sequential within CXL)
```

The **sum** (not max) reflects paper Eq (5): `T_LoRA_CXL = T_cpt + T_trs`.
NDP can't compute until input arrives, and result can't return until NDP
finishes.

### Strategy E4 (NDP computes `m = xA`, GPU finishes `y = mB`)

Split work between NDP and GPU.

```
link bytes = B · (D + R) · S       (x out: B·D·S; m back: B·R·S)
NDP FLOPs  = 2 · B · D · R          (only x·A)
GPU FLOPs  = 2 · B · D · R          (only m·B)
GPU mem    = R · D · S              (read B from HBM)

t_link  = L_CXL + B·(D+R)·S / W_CXL
t_ndp   = 2·B·D·R / (C_CXL · N_CXL)
t_gpu   = 2·B·D·R / C_GPU
t_hbm   = R·D·S / W_GPU
cost(E4) = max(t_link + t_ndp, max(t_gpu, t_hbm))
```

The outer **`max`** reflects that GPU work runs in parallel with CXL work
(separate pipelines synchronized at the layer boundary). Paper Eq (6) is
more precise — `max(T_Base, T_LoRA_CXL) + T_LoRA_GPU` — sequencing GPU work
*after* CXL. My ranking simplification slightly underestimates absolute E4
cost when GPU compute is the bottleneck, but **ranking ordering is
preserved** because E1 dominates whenever cacheable and E2/E3/E4 differ
mainly in link traffic which is captured exactly.

### When each strategy wins (closed-form analysis)

For B=1 (typical per-adapter at small batches): E1 = 135 ns < E4 = 280 ns
< E3 = 360 ns < E2 = 2,250 ns. E1 dominates whenever it fits.

For B → ∞: E1 ~ B (compute-bound), E2 ~ const (link-bound by 2RDS),
E3 ~ B (link-bound by 2BDS), E4 ~ B. E2 becomes most attractive among
{E2, E3, E4} when B exceeds ≈ 2R (so the constant E2 link cost beats E3's
per-request link).

**Verified by `t13_e2_vs_e3_crossover` in `test_clora_strategy.py`.**

---

## 2. Memory accounting — bytes that matter

The `gpu_mem_for` function tracks how many bytes each strategy consumes
in GPU HBM. The critical multiplier is `n_layers × n_matrices`.

```python
def gpu_mem_for(strategy, adapter, model, *, n_layers=1, n_matrices=1):
    one = adapter.matrix_bytes(model) * n_layers * n_matrices
    return {Strategy.E1: 2 * one,    # both A and B for every layer × matrix
            Strategy.E2: 0,           # A,B in CXL
            Strategy.E3: 0,           # A,B in CXL
            Strategy.E4: one}[strategy]  # B only
```

### Why `n_layers × n_matrices`

A LoRA adapter has a *separate* A and B matrix for each transformer
weight (Q, K, V, output, FFN-gate, FFN-up, FFN-down = 7 matrices) at each
of the 32 decoder layers. A fully cached rank-16 adapter on Llama2-7B
holds:

```
2 · 16 · 4096 · 2 · 32 · 7  =  58.7 MB
```

vs. one matrix's worth (`2 · 16 · 4096 · 2 = 256 KB`). The 224× multiplier
is *not* optional — it's the difference between "1000 adapters fit in
GPU" (wrong: 256 MB) and "only ~30 fit" (correct: ~1.8 GB at avg rank).

**Verified by paper §7.5**: "ND(E4) would require over 500 GB of GPU
memory" — consistent with our 1000 × 235 MB (R=128, n_l=32, n_m=7) = 235 GB
at extreme rank.

### Default kwargs

`n_layers=1`, `n_matrices=1` defaults exist so the early per-matrix unit
tests (which predate the multiplier fix) still pass without modification.
The driver passes `n_layers=args.n_layers, n_matrices=args.n_matrices`
to every consumer (`choose_strategy`, `pre_cache_hot_adapters`,
`compute_kv_in_gpu_fraction`, `emit_step_json`, `baseline_step_ns`,
`grace_hopper_step_ns`, plus both LRU caches). `--n-matrices` defaults
to **7** (Llama2: Q/K/V/O + FFN gate/up/down) and is overridden for
other architectures:

| architecture | typical `--n-matrices` | rationale |
|---|---:|---|
| Llama2-7B / 13B (dense) | 7 | Q,K,V,O attention + gate,up,down FFN |
| Qwen3-30B-A3B (MoE, attention-only LoRA) | 4 | Q,K,V,O attention only |
| Qwen3-30B-A3B (MoE, attention + active experts) | **28** | 4 attention + 3 × 8 active-expert FFNs |
| MoE with K active experts, attention + experts | `4 + 3 * K` | top-K routing |

Dormant experts in MoE do NOT contribute to `n_matrices` — only the
experts actually activated per request (top-K) have LoRA applied. The
non-routed 120 experts/layer in Qwen3-30B-A3B's 128-expert pool are
ignored for LoRA accounting.

---

## 3. Algorithm 1 — step-by-step implementation

Paper §6.2 Algorithm 1, in code (`choose_strategy`):

```python
def choose_strategy(adapter_id, state, hw, model, *, n_layers, n_matrices, top_k_hot):
    # Sort all four strategies by per-strategy cost (Step 1, 3)
    ranked = sorted(Strategy, key=lambda E: cost_lora(E, target, hw, model))
    other_bytes = state.total_other_bytes(adapter_id, model, ...)
    budget = hw.gpu_mem_bytes - other_bytes
    best, mem_best = ranked[0], gpu_mem_for(ranked[0], target, ...)

    # Step 2: cheapest fits → return immediately
    if mem_best <= budget:
        return Decision(best, cost_lora(best, target, hw, model))

    # Step 3: cold-only eviction for the cheapest strategy
    _, _, cold = classify_adapters(state, top_k=top_k_hot)
    cold_ids, cold_freed = _evict_set(cold, mem_best - budget, ...)
    if cold_freed >= mem_best - budget:
        return Decision(best, ..., evict_cold_ids=cold_ids)

    # Step 4: enumerate (strategy × eviction-kind) variants
    candidates = []
    for E in Strategy:
        mem = gpu_mem_for(E, target, ...)
        if mem <= budget:
            candidates.append(Decision(E, ...))         # (a) fits as-is
        else:
            deficit = mem - budget
            if state.kv_cache_bytes >= deficit:
                candidates.append(Decision(E, ..., evict_kv_bytes=deficit))   # (b)
            # (c) cold (d) hot (e) other-serving — each with smallest-sufficient-set
            for pool_name, pool in [("cold", cold), ("hot", hot), ("serving", serving)]:
                ids, freed = _evict_set(pool, deficit, ...)
                if freed >= deficit:
                    candidates.append(Decision(E, ..., evict_<pool>_ids=ids))

    candidates.sort(key=lambda d: d.cost_ns)
    return candidates[0]
```

### `_evict_set` — smallest-sufficient-set greedy

Paper says "Replace Cold" without specifying the size. We implement the
**least-disruptive** version: sort the candidate pool by descending size,
take adapters off in that order until enough bytes are freed.

```python
ranked = sorted(pool, key=lambda a: -state.adapter_gpu_bytes(a.id, model, ...))
freed = 0; chosen = []
for a in ranked:
    if freed >= need: break
    bytes_here = state.adapter_gpu_bytes(a.id, model, ...)
    if bytes_here == 0: continue  # nothing to evict
    chosen.append(a.id); freed += bytes_here
```

Largest-first means we typically evict 1–2 big adapters instead of many
small ones — fewer cache disturbances per Algorithm 1 invocation.

### Why include cold-eviction in Step 4

Paper Step 4 only mentions KV / hot / serving eviction variants. We also
include cold there because **Step 3's cold-only check was strategy-specific
to the *best* strategy**. A non-best strategy might fit with cold eviction
when the best one doesn't — our Step 4 correctly enumerates that
possibility. This is strictly more flexible than the paper's pseudocode
without changing correctness.

### Per-adapter independence (and what it doesn't track)

Each call to `choose_strategy` reads the *current* `state.adapters[*]`. We
do **not** sequentially mutate state between per-adapter calls in the
driver — all serving decisions in a step see the same pre-decision state.
The hot pre-cache (run before strategy selection) mutates state once, so
serving decisions correctly see those bytes consumed.

This means: if 32 serving adapters all independently say "I want E1", and
they collectively exceed `gpu_mem_bytes`, we'll over-allocate on paper. In
practice, with our defaults (26 GB free, ~50 MB per E1 adapter), 32 × 50 MB
= 1.6 GB ≪ 26 GB, so this never bites. For very tight budgets it would
need a sequential pass; flagged as a known limitation.

---

## 4. Temperature dynamics

Paper §6.2 specifies:
- "New request arrivals increase the corresponding adapter's temperature by T_1"
- "All adapter temperatures decay over time, with decay rate proportional
  to the temperature gap from the system's average temperature"

Code (`TemperatureModel`):

```python
def on_request(self, adapter_id):
    self.temps[adapter_id] = self.temps.get(adapter_id, 0.0) + self.T_1

def step(self):
    if not self.temps: return
    mean_t = sum(self.temps.values()) / len(self.temps)
    for aid in list(self.temps.keys()):
        self.temps[aid] -= self.alpha * (self.temps[aid] - mean_t)
```

### The decay equation

`temp[a] -= α · (temp[a] − mean)`. This is **mean-reverting** (paper's
"proportional to gap from mean" interpretation):
- If `temp[a] > mean`: drifts down toward mean (decay)
- If `temp[a] < mean`: drifts up toward mean (anti-decay)

After many steps without new requests, all adapters converge to the same
temperature. Steady-state behavior: with continuous request arrivals,
frequently-requested adapters maintain temperature *above* mean (because
they keep getting bumped); rarely-requested adapters settle *below*. This
gives the hot/cold separation the algorithm needs.

### Why decay at end-of-step

The driver orders operations as:
1. Bump temperatures for arriving requests
2. Build SystemState with current temps
3. Run strategy selection
4. Run baselines
5. **Decay at end** for next step's setup

If decay ran first, this step's classification would use already-decayed
temps; the just-arrived requests would dominate. Putting decay at the end
keeps a request's "heat" intact for at least one step of classification
benefit.

### Edge case: single non-serving adapter

If `len(others) == 1`, `mean == temps[a]`, and `temps[a] >= mean` (true
trivially), so that adapter lands in `hot` under mean-based
classification. With `top_k` mode, top-1 still puts it in hot. Both are
defensible.

### Default knobs

- `T_1 = 1.0`: each request adds 1 unit of heat
- `α = 0.1`: 10% of the gap to mean is shed per step

Steady-state hot-set size with these defaults: in our 1000-adapter
Skewed workload (50 nominal hot adapters), after 10 warmup steps the
top-50 by temperature reliably contain ≥40 of the nominal hot set
(empirical observation; not a tested invariant).

---

## 5. Hot-adapter pre-caching

Paper §6.2 Figure 10 shows GPU memory containing **both serving and hot
adapter** A,B matrices. Hot adapters are speculatively cached as E1 so
that when a request arrives, no load is needed.

### Algorithm (`pre_cache_hot_adapters`)

```python
def pre_cache_hot_adapters(state, hw, model, *, top_k, n_layers, n_matrices):
    serving, hot, _ = classify_adapters(state, top_k=top_k)

    # Reserve memory for serving adapters' upper-bound E1 footprint first
    serving_reserved = sum(2 * a.rank * D * S * n_layers * n_matrices
                          for a in serving)
    budget = hw.gpu_mem_bytes - serving_reserved
    if budget <= 0: return []

    cached = []
    for a in sorted(hot, key=lambda a: -a.temperature):
        needed = 2 * a.rank * D * S * n_layers * n_matrices
        if needed <= budget:
            a.a_in_gpu = True; a.b_in_gpu = True
            budget -= needed; cached.append(a.adapter_id)
    return cached
```

### Two key design decisions

**(1) Reserve serving E1 footprint first.** We pessimistically subtract
the full E1 footprint of every serving adapter from the budget *before*
deciding hot pre-cache, even though the eventual strategy might be E4
(half the bytes) or E2/E3 (zero). This guarantees serving adapters
*always* fit, at the cost of slightly under-using the hot cache.

Alternative: estimate the per-adapter expected strategy first. Chicken-
and-egg: strategy depends on what's cached, which depends on the budget,
which depends on serving footprint. Iterative fixed-point is overkill.

**(2) Cache as E1 (full A+B), not E4 (B only).** Paper Figure 10 shows a
mix — some entries labeled `A_i, B_j` (E1-like) and others just `B_k`
(E4-like). We chose E1-only because it maximizes throughput-per-cached-
adapter at runtime. A future refinement: also cache as E4 when memory is
extremely tight to fit more hot adapters at half the cost per slot.

### Greedy hottest-first

The hot pool is sorted by *descending temperature*. The hottest adapters
get the first crack at GPU memory; if memory runs out before all `top_k`
are cached, the coolest ones miss out. This converges quickly because the
temperature signal already prioritizes likely-needed-soon adapters.

---

## 6. KV-cache duplication policy

Paper §5.2: "a small portion of the KV cache is duplicated in GPU memory
(HBM), enabling parallel computation between GPU and NDP cores."
Paper Figure 16 reports 2–14% duplicated in practice.

### Computation (`compute_kv_in_gpu_fraction`)

```python
def compute_kv_in_gpu_fraction(decisions, state, hw, model, kv_tokens_per_adapter,
                               *, n_layers, n_matrices,
                               total_batch, base_model_bytes,
                               moe_active_base_bytes=None):
    # memory cap: whatever GPU memory the LoRA decisions left over
    free  = max(0, hw.gpu_mem_bytes - lora_bytes)          # as before
    p_max = min(free, kv_total_bytes) / kv_total_bytes

    # cost-aware choice (Eqs 7-9): duplicated KV is NOT free -- it adds
    # P-proportional HBM bytes + FLOPs to the GPU's per-layer roofline,
    # while the devices' share shrinks with (1 - P). Both sides run in
    # parallel, so pick P minimizing
    #     max( T_GPU_layer(P), T_ATT_CXL_layer(P) )   for P in [0, p_max]
    # via a 200-point grid scan (ties -> smaller P).
```

The chosen fraction lands at **0% on short-KV** (the device-side
attention hides under the base-model HBM floor, so duplication only
adds GPU time) and **12–16% on long-KV** workloads — consistent with
paper Fig 16's 2–14%. The duplicated share's GPU cost is then charged
through `estimate_base_model_ns_per_layer(kv_tokens_in_gpu=...)`, which
folds Eq (7)'s `4·D·K` FLOPs and `2·D·S·K` HBM bytes into the per-layer
roofline that becomes the C-sim's `NPU_COMPUTE` duration.

> Earlier drafts used a greedy fill (`min(free, kv_total) / kv_total`),
> which reached 100% at batch 32 *and never billed the GPU for the
> duplicated share's attention* — inflating long-KV throughput by
> 6–19%. The greedy path survives only as a fallback when
> `base_model_bytes` is not supplied (used by some unit tests).

### Why this ordering matters

The fraction is computed **after** strategy selection (so it sees real
serving E1/E4 footprints) **and after** hot pre-caching (so it knows what
the hot pool reserved). Memory accounting is single-source-of-truth: the
remaining `free` is what's actually available for KV duplication.

### Why "fraction" (single number) is enough

We don't need per-adapter KV-in-GPU fractions because attention treats
the whole KV cache as one tensor — duplicating a fraction means "for each
layer's KV slice on each device, only `1-fraction` actually has to be
read from device DRAM; `fraction` is already in HBM". The C simulator
scales `per_dev_dram` accordingly:

```c
unsigned int per_dev_dram = 2.0 * D * S * kv_per_device * n_layers * (1 - kv_in_gpu_fraction);
```

### What the C simulator does *not* model

- We don't reduce `per_dev_input` or `per_dev_output` based on
  `kv_in_gpu_fraction` because Q still needs to be sent to all devices
  even if only a fraction of KV lives in CXL.
- We don't reduce NDP compute proportionally because the C-sim's NDP
  compute term collapses to ~1 ns anyway (pre-existing `(addr_num-1)`
  quirk in `flash.c`).

Net result: KV duplication primarily reduces the *DRAM read time* on each
CXL device, which is the long-sequence bottleneck. This is the dominant
effect per paper §5.2.

---

## 7. C event simulator — state-machine specifics

Most of the C event sim is original SSDsim code we left untouched. The
relevant CLoRA additions are in `main.c::npu_process` (request emission)
and `flash.c` (state machine timing, already pre-existing).

### CXL READ request lifecycle

For a `READ` sub-request (used by E2):

```
Time    State                              Duration / Source
─────────────────────────────────────────────────────────────────────
0       SR_WAIT                            queue
+0      → SR_CHANNEL_R_CA_TRANSFER         when scheduled
+30     → SR_CXLCTRL_R_CA_ANALYZE          tCMDCXL = 30 ns (parameters.conf)
                                           [+ L_CXL_switch if non-zero]
+10     → SR_CXLCTRL_R_CA_TRANSFER         tANALYZE = 10 ns
+10     → SR_CXLCTRL_R_READ                tCMDDRAM = 10 ns
+T_dram → SR_CHANNEL_R_DATA_TRANSFER       T_dram = tDRAMRL + bytes/W_DRAM
+T_link → SR_COMPLETE                      T_link = bytes/W_CXL [+ L_CXL_switch]
```

Total per-request time = 30 + 10 + 10 + T_dram + T_link ≈ 50 ns + DRAM
read + link transfer. For a 256 KB E2 load: ≈ 2,368 ns at default params.
**Verified to the nanosecond by `verify_read.json`.**

### CXL READ_COMPUTE request lifecycle

For a `READ_COMPUTE` (used by E3, E4, attention):

```
Time    State                              Duration / Source
─────────────────────────────────────────────────────────────────────
+0      SR_CHANNEL_RC_CA_TRANSFER          tCMDCXL + input_size/W_CXL
+10     SR_CXLCTRL_RC_CA_ANALYZE           tANALYZE [+ L_read_compute_cmd]
+10     SR_CXLCTRL_RC_CA_TRANSFER          tCMDDRAM
+T_dram SR_CXLCTRL_RC_READ                 T_dram = tDRAMRL + size/W_DRAM
+T_pe   SR_CXLCTRL_RC_COMPUTE              (addr_num-1)*size/(datatype/8)/C_NDP  ← pre-existing quirk
+1      SR_CXLCTRL_RC_DATA_TRANSFER        hardcoded 1 ns
+T_out  SR_CHANNEL_RC_DATA_TRANSFER        output_size/W_CXL [+ L_CXL_switch]
```

For a typical attention sub-request: ≈ 450 ns at default params (1 layer,
1 matrix, kv=400 tokens distributed). **Verified by `verify_rc.json`.**

### The `(addr_num - 1)` quirk in NDP compute time

`flash.c:1733` computes NDP PE time as
`(addr_num − 1) × size_of_addr[0] / (datatype/8) / chip_computing_power`.
The `(addr_num − 1)` factor pre-dates this work. `slice_request` always
groups sub-requests by `cxl_id` (each cxl_id gets its own sub-request),
so each sub-request has `addr_num == 1` and the term collapses to 0
(clamped to 1 ns).

**Net effect:** NDP PE compute is essentially free in our sim. Bytes
transferred and DRAM-read time are exactly modeled; only the PE compute
is undercounted. For ranking purposes this doesn't matter — NDP compute
time would be ~5% of step time at most. For absolute throughput it
slightly inflates CLoRA's numbers (we estimate by <5%).

A fix would be `flash.c:1733 (addr_num - 1) → addr_num`. We left this
untouched per the original repo state.

### Adapter → CXL device mapping (`adapter_home_device`)

```c
static unsigned int adapter_home_device(unsigned int adapter_id,
                                        unsigned int n_cxl) {
    return adapter_id % n_cxl;
}
```

Round-robin by adapter id. With 32 active adapters and 4 CXL devices,
each device hosts ~8 adapters. The simplification: a single adapter's
LoRA traffic always uses one device — devices serialize per-adapter LoRA
operations on a single channel. **Cross-device parallelism comes from
having different adapters on different devices**, not from striping one
adapter's traffic.

### Distributed attention

Per paper §5.2, KV cache is sharded by token across all `N_CXL` devices.
Each device computes its slice of `Q·K^T → softmax → S·V` and returns a
full B×D partial output. The GPU combines partial outputs.

```c
unsigned int per_dev_dram   = 2.0 * D * S * kv_per_device * n_layers * (1 - kv_in_gpu_fraction);
unsigned int per_dev_input  = ad->batch * D * S * n_layers;   // full Q to each device
unsigned int per_dev_output = ad->batch * D * S * n_layers;   // full partial O back
```

`addr_num = n_cxl` with distinct `cxl_id` per addr → `slice_request`
generates `n_cxl` parallel sub-requests, one per device. Devices process
concurrently on their CXL channels.

### Event-loop convergence

`npu_process` issues all requests at step 0, sets `npu_finish = 1`. The
event loop in `ssd.c::find_nearest_event_sys` advances time to the
nearest scheduled state transition until all sub-request queues are
empty AND no NPU requests are pending. `simulation_end_time` is the
final timestamp; that's what gets reported as `step_ns`.

---

## 8. JSON contract between Python and C

The schema in `include/readJson.h` (parsed by `readJson.c`):

```jsonc
{
  "kind":      "decode" | "prefill",        // currently only "decode" exercised
  "model":     { "d":4096, "n_layers":32, "n_matrices":7, "s_dtype":2 },
  "hw":        { "n_cxl":4 },               // # of CXL memory devices
  "input_tokens_per_request": 1,            // 1 for decode; >1 for prefill
  "base_model_compute_ns":    50000.0,      // per-layer GPU base time
  "kv": {
      "kv_tokens_total":    18000,          // sum across adapters
      "kv_in_gpu_fraction": 0.14            // 0..1
  },
  "adapters": [
      { "id":7, "rank":16, "batch":4, "strategy":3, "kv_tokens":562 },
      ...
  ]
}
```

### Field semantics

| field | type | C-side use |
|---|---|---|
| `kind` | string | `"prefill"` → C side sets `d->kind = 1`; currently no different behavior |
| `model.d` | uint | D in size formulas |
| `model.n_layers` | uint | multiplier in per-adapter byte calculations |
| `model.n_matrices` | uint | multiplier (LoRA-touched per-layer matrices, default 7; settable via `--n-matrices`; MoE: 4 for attention-only or 4+3·K for attention + K active experts) |
| `model.s_dtype` | uint | bytes per FP element (FP16 = 2) |
| `hw.n_cxl` | uint | # of CXL devices used for attention distribution; clamped to `cxl_channel_number` from conf |
| `input_tokens_per_request` | uint | reserved for prefill; currently not used in size formulas |
| `base_model_compute_ns` | double | per-layer; C sim multiplies by `n_layers` and creates one `NPU_COMPUTE` event |
| `kv.kv_tokens_total` | uint | reserved; not used (per-adapter kv_tokens used instead) |
| `kv.kv_in_gpu_fraction` | double | scales `per_dev_dram` in attention; 0 = full CXL read, 1 = no CXL DRAM access |
| `adapters[].id` | uint | used by `adapter_home_device` for round-robin |
| `adapters[].rank` | uint | R in size formulas |
| `adapters[].batch` | uint | B in size formulas; 0 = no emission |
| `adapters[].strategy` | uint | 1..4; clamped to E3 with warning if invalid |
| `adapters[].kv_tokens` | uint | drives `per_dev_dram` and attention emission |

### Why this schema (vs the rejected `[5][10]` approach)

The original repo had `q_rc_req[5][10]` fixed arrays — at most 5 per
phase, 10 fields each, with parsing disabled. We replaced it with a
dynamic `AdapterDecision[]` array for unlimited per-step adapters.
**Trade-off:** the C parser allocates per call (small) and the JSON file
gets re-written per step (~1 KB at our scale; ~0.5 ms total cost
including atomic rename), so this scales cleanly to 1000s of adapters.

### Atomic write from Python (`write_step_json`)

```python
def write_step_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(payload, f, indent=2)
    os.replace(tmp, path)
```

`os.replace` is atomic on POSIX — the C side never reads a half-written
file. Indent=2 is purely for human debugging; doesn't affect parse time.

### C-side timestamp collision

The C sim creates `raw/<timestamp>/` for logs and aborts on `mkdir`
conflict. The driver passes `--timestamp <pid>_<counter>` so consecutive
sub-second invocations don't collide. See `_run_counter` in
`clora_driver.py`.

---

## 9. CPU-LoRA-Offload baseline

Code: `script/baseline_cpu_offload.py`. The naive strawman where the CPU
runs the LoRA matmul itself.

### Per-(adapter, layer, matrix) cost on miss

```
t_pcie     = 2 * L_pcie_setup + (x_bytes + y_bytes) / pcie_bw
t_cpu_cmp  = 4 * B * D * R / C_CPU
t_cpu_dram = 2 * R * D * S / W_DRAM_CPU         (load A,B from DRAM)
t_kernels  = 2 * (L_kernel_launch + L_cpu_gpu_offload)
```

### Why `2 ×` kernel launches per op

Each LoRA op fires **two** kernels:
1. `cudaMemcpyAsync(x → CPU)` — launch + offload overhead
2. `cudaMemcpyAsync(y → GPU)` — same

The compound `L_cpu_gpu_offload` (default 10 µs) covers everything in
between — host-side DMA setup, CPU compute kick-off, sync. We don't
model CPU compute time as a separate kernel because the CPU isn't
launching CUDA kernels — but it *is* a sync point that the GPU has to
wait for.

### Fused vs unfused kernel modes

```python
if fusion == "fused":
    # Group cache misses by rank → 1 op per (layer, matrix, unique-rank)
    n_kernel_pairs += n_layers * n_matrices    # per rank-group
else:
    # 1 op per (layer, matrix, adapter)
    for aid in miss_ids:
        n_kernel_pairs += n_layers * n_matrices
```

S-LoRA/PUNICA use **BGMV** (batched general matrix-vector) where all
adapters of the same rank in a batch are processed by one fused kernel.
Unfused is a strawman where every adapter gets its own kernel — useful
for showing kernel-launch overhead dominates without fusion. The actual
multipliers in our default workloads: ~5 unique ranks → fused has
5 × 32 × 7 = 1,120 pairs; unfused has 32 × 32 × 7 = 7,168 pairs.

### Per-layer parallelism (`parallel=True`)

```
gpu_path_per_layer  = base_per_layer + cached_per_layer
missed_per_layer    = (pcie + cpu + kernels) / n_layers
attn_per_layer      = attn_total / n_layers

t_per_layer = max(gpu_path, missed_path) + attn_per_layer       # parallel
            | gpu_path + missed_path + attn_per_layer           # serial

total_ns = t_per_layer * n_layers
```

The `max(...)` reflects CUDA stream concurrency: while CPU is computing
LoRA on one stream, GPU computes base model on the compute stream.
`--baseline-serial` falls back to sum (worst case naive).

### Attention separate from LoRA path

```python
attn_pcie_bytes = 2 * total_batch * D * S * n_layers   # Q out + result back
attn_pcie_setup = 2 * L_pcie_setup * n_layers          # one setup per direction per layer
t_attn_pcie     = attn_pcie_setup + attn_pcie_bytes / pcie_bw

attn_cpu_flops = 4 * D * total_kv * n_layers           # Q·K^T + S·V batched
attn_dram      = 2 * D * S * total_kv * n_layers       # K + V

t_attn = t_attn_pcie + max(cpu_compute, dram) + kernel_overhead
```

Attention is its own offload path because Q (computed by GPU base model)
must arrive before CPU can compute attention, and result must arrive
before GPU can do post-attention layers (O, FFN).

### LRU cache semantics

```python
class LRUAdapterCache:
    def touch(self, adapter_id, rank) -> bool:
        if adapter_id in self._stored:
            self._order.remove(adapter_id); self._order.append(adapter_id)
            self.n_hits += 1; return True
        # miss: evict until room, then insert
        self.n_misses += 1
        needed = adapter_full_bytes(rank, model, n_layers, n_matrices)
        while self.used_bytes + needed > self.capacity_bytes and self._order:
            ev = self._order.pop(0); del self._stored[ev]
        if needed <= self.capacity_bytes:
            self._stored[adapter_id] = needed; self._order.append(adapter_id)
        return False
```

Standard LRU; one entry per adapter (holds all 32 × 7 layer-matrices of
A,B). Eviction frees one entry at a time until the new entry fits, or the
order list is empty (cache too small for even one adapter → silently
fail to cache).

### Why dormant adapters are filtered

In the driver:
```python
adapter_specs = {a.adapter_id: (a.rank, a.batch)
                 for a in state.adapters.values() if a.batch > 0}
```

Without `if a.batch > 0`, every dormant adapter from the temperature
model would `touch` the cache, evicting useful warm entries for no
reason. Filtering to serving keeps hit rates realistic.

---

## 10. CLoRA-NoCXL baseline

Code: `script/baseline_no_cxl.py`. **60 lines.** Intentionally tiny.

### The model in one equation

```
T_NoCxl = T_CLoRA + per_offload × offloads_per_layer × n_layers

where per_offload = L_kernel_launch + L_device_command + L_sync
```

### Why a thin model

NoCxl differs from CLoRA only in *what happens between layers*:
- Same NDP throughput, same DRAM bandwidth, same link bandwidth
- Same strategy selection, hot pre-cache, KV duplication
- The only addition: every GPU↔NDP dependency boundary needs an explicit
  DMA setup + kernel relaunch + sync, instead of one fused
  read-compute request

So the C event simulator already produces the right `T_CLoRA` for
everything *except* the per-boundary host overhead. We layer that on top.

### `offloads_per_layer = 2` decomposition

| boundary | reason |
|---|---|
| LoRA op for non-cached adapters | GPU launches matmul, NDP completes, GPU consumes result |
| Attention | Depends on Q/K/V; GPU launches separately, NDP completes, GPU consumes partial O |

We batch all 7 LoRA-touched matrices' offloads into one ("the LoRA
offload"), per the "be-reviewer-friendly" framing — even though reality
might use 1 per matrix. Set `--nocxl-offloads-per-layer 8` to model
per-matrix accounting.

### Edge case: all serving adapters are E1 (no LoRA offload)

We still charge `offloads_per_layer = 2` per layer. Slightly pessimistic
in this case — we could skip the LoRA offload when no non-E1 adapter
exists. We didn't because:
1. Even with E1, attention still offloads
2. The conservative count favors NoCxl (more pessimistic for CLoRA's
   relative speedup claim → harder to defend, easier to publish)

---

## 11. Grace-Hopper Offload baseline

Code: `script/baseline_grace_hopper.py`. ~170 lines.

### The structural difference from CPU-LoRA-Offload

| | CPU-LoRA-Offload | Grace-Hopper |
|---|---|---|
| LoRA matmul on | CPU | **GPU** (much faster) |
| Link | PCIe | **NVLink-C2C 450 GB/s** (coherent) |
| Per-op pattern | round-trip with explicit DMA + sync | streaming load + compute, GPU-side only |
| Kernel count per op | 2 (send + receive) | 1 (just the GPU matmul) |
| `L_cpu_gpu_offload` overhead | 10 µs | 0 (no sync between CPU and GPU compute) |

### Per-(layer, matrix, rank-group) on miss

```
bytes_per_op   = 2 * R * D * S * n_adapters_in_group       (A and B for each adapter)
t_c2c_transfer = L_c2c_setup + bytes_per_op / c2c_bw
t_gpu_compute  = 4 * agg_batch * D * R / C_GPU
t_kernel       = L_kernel_launch    (one launch per fused op)

t_missed_lora  = max(t_c2c_transfer, t_gpu_compute) + n_kernels * L_kernel_launch
```

### Why `max(transfer, compute)` for Grace-Hopper

NVLink-C2C is *coherent*. The GPU can issue loads that miss in HBM and
hit Grace CPU memory transparently. As the matmul executes, weights
stream in over the link in the background. **Transfer and compute
pipeline on the same kernel** — total wall time is the slower of the two,
not their sum.

This is the key qualitative difference from PCIe + explicit DMA, where
transfer and compute are strictly sequential because of the host sync.

### Attention path

```
kv_bytes_per_layer = 2 * D * S * total_kv
t_attn_transfer    = (L_c2c_setup + kv_bytes_per_layer / c2c_bw) * n_layers
t_attn_compute     = 4 * D * total_kv * n_layers / C_GPU
t_attn_kernels     = n_layers * L_kernel_launch
t_attn             = max(transfer, compute) + kernel_overhead
```

Same overlap semantics — coherent C2C streams KV during GPU attention
kernel execution.

### Why GPU compute time is essentially free at moderate batch

At default H100 (989 TFLOPS) on Uniform 7B (B=32, avg R=49):
```
t_gpu_compute(LoRA, all 32 layers, 7 matrices, all 32 adapters)
= 4 · 32 · 4096 · 49 · 32 · 7 / 989e12
= 56.6 G FLOPs / 989 TFLOPS
= 57 µs
```

Transfer time (4.9 GB of A,B at 450 GB/s) ≈ 10.9 ms. **Transfer
dominates** by 190×. Conclusion: Grace-Hopper is link-bound. Whatever
GPU compute throughput we plug in, throughput scales linearly with
`c2c_bw`. Verified by the sensitivity sweep in `DESIGN.md` recipes.

### Per-layer parallelism (same shape as CPU-LoRA)

```
t_per_layer = max(base + cached_lora, missed_lora_path) + attn_per_layer
```

`missed_lora_path` here is the C2C+GPU path (still synchronous within
that path due to `max(transfer, compute)`).

---

## 12. Driver orchestration & steady state

Code: `script/clora_driver.py::run_smoke`. Per-step flow:

```python
for step in range(warmup + measure):
    is_warmup = step < warmup

    reqs = gen_step(...)                          # generate batch
    for aid, _ in reqs:
        temp_model.on_request(aid)                # 1. bump temperatures

    state, kv_per_adapter = make_state(reqs, ranks, temp_model)
                                                  # 2. include dormant adapters
    pre_cached = pre_cache_hot_adapters(state, ...)   # 3. mutate state

    serving_ids = [aid for aid in state.adapters if state.adapters[aid].batch > 0]
    decisions = {aid: choose_strategy(aid, state, ...) for aid in serving_ids}
                                                  # 4. per-serving-adapter strategies
    kv_in_gpu_fraction = compute_kv_in_gpu_fraction(decisions, state, ...)
                                                  # 5. residual GPU mem → KV

    payload = emit_step_json(...)                 # 6. write JSON
    write_step_json(TRACE, payload)
    clora_step_ns = run_c_sim(TRACE)              # 7. invoke C simulator

    if cpu_off_on:                                # 8. CPU-LoRA-Offload baseline
        baseline_step_info = baseline_step_ns(...)
    if no_cxl_on:                                 # 9. NoCxl overhead
        no_cxl_step = no_cxl_step_ns(clora_step_ns, ...)
    if gh_on:                                     # 10. Grace-Hopper baseline
        gh_step_info = grace_hopper_step_ns(...)

    temp_model.step()                             # 11. decay for next step

    if not is_warmup:                             # 12. accumulate
        total_tokens += batch_total
        total_ns += clora_step_ns
        # baseline_total_ns += ...
```

### Persistent state across steps

| object | reset per | what it tracks |
|---|---|---|
| `temp_model` | per driver invocation | adapter temperatures with cross-step decay |
| `ranks` | per driver invocation | each adapter's rank, assigned on first arrival |
| `baseline_cache` | per driver invocation | LRU state for CPU-LoRA-Offload |
| `gh_cache` | per driver invocation | LRU state for Grace-Hopper |
| `state.adapters` | **per step** | rebuilt each step from temp_model + reqs |

The per-step rebuilding of `state.adapters` is intentional: it gives
hot pre-cache and strategy selection a clean view that includes both
serving adapters (this step's requests) and dormant adapters (from the
persistent temp model) without carrying stale `a_in_gpu`/`b_in_gpu`
flags from prior steps.

### Steady state via warmup

`--warmup-steps N` runs N steps before measurement. Reasons we need it:

1. **Temperature model** needs ~10 steps to establish hot/cold separation
   (the 50 most-frequently-requested adapters need to accumulate higher
   temperatures than the rest)
2. **CPU-LoRA-Offload LRU cache** needs ~5–10 steps to fill (initial
   misses inflate baseline cost otherwise)
3. **Grace-Hopper LRU cache** same as above

Measurement-only throughput is `total_tokens / sum(measure_step_ns)`.
Default `--warmup-steps 10 --steps 10` is enough for our test workloads;
longer skew distributions might benefit from `--warmup-steps 30`.

### RNG determinism

`random.Random(args.seed)` is used everywhere. Same seed → identical
workload across runs. Useful for A/B testing knob changes.

---

## 13. Design decisions — the "why"s

### Why C event sim + Python control plane (not all-Python or all-C)

- **Event sim is performance-critical** at sub-µs granularity (per-state
  transitions on a CXL link). C is appropriate.
- **Strategy logic is dev-velocity-critical** (algorithm changes are
  research questions). Python is appropriate.
- The JSON contract is a clean separation: Python decides, C times.

### Why we left the original CXL state machines untouched

The CXL/DRAM/PE state machines in `flash.c` predate this work and are
non-trivial (~1000 lines of state transitions). Modifying them risks
introducing subtle timing bugs that the existing microbenches wouldn't
catch. The `addr_num-1` quirk in NDP compute is a known imperfection,
flagged but not fixed.

### Why `n_matrices = 7` for dense Llama2

Llama2/3 attention has Q, K, V, O = 4 matrices per layer; FFN has
gate, up, down = 3 matrices per layer. LoRA is applied to all 7 per
paper §7.1: "We add LoRA adapters to all weight matrices for
generality, including Q/K/V, projection, and FFN."

### Why `n_matrices = 28` for Qwen3-30B-A3B MoE

For sparse MoE models, we apply LoRA to attention matrices (Q, K, V, O
= 4) plus the FFN matrices (gate, up, down = 3) of the **actively
routed** experts only. Qwen3-30B-A3B routes top-8 of 128 experts per
token (`num_experts_per_tok = 8`), giving `4 + 3·8 = 28` matrices/layer.

Key modeling decisions:
- **Active experts only.** The dormant 120 experts/layer never see
  LoRA traffic per request. This matches what multi-LoRA serving
  systems actually do (you wouldn't apply LoRA to experts you didn't
  route to).
- **Same 8 active experts per step assumed.** In reality the union of
  active experts across a batch of 32 tokens can be all 128. We
  conservatively assume the *typical* per-token activation pattern
  applies, not the batch-union. This is consistent with how the active
  base bytes are modeled (`--moe-active-base-gb 6` = per-token, not
  per-batch).
- **Selectable via `--n-matrices` flag.** A user can override to
  `--n-matrices 4` for attention-only LoRA placement (the "S-LoRA on
  MoE" style), or any other custom configuration without code changes.

### Why E1-only proactive hot caching (not mixed E1+E4)

Paper Figure 10 shows a mix, but the figure is illustrative. Going
E1-only:
- Simpler (one decision: cache full or not)
- Better-per-adapter throughput when re-touched
- Slightly less cache-fitting capacity than mixed E1+E4

A future ablation could compare; doesn't affect any current paper claim.

### Why per-adapter strategy selection runs in parallel (not sequentially)

Sequential per-adapter would model "as adapter A consumes memory, adapter
B sees less budget". Parallel models "all serving adapters see the same
pre-decision budget". With our defaults (26 GB free, ~50 MB per E1
adapter, 32 active), the over-counting is ≤6% of total memory and never
exceeds budget. Sequential would be more accurate at very tight memory
budgets; flagged as a refinement opportunity.

### Why top-K = 50 default for hot classification

Paper §7.1 Skewed workload uses 50 hot adapters. Defaulting top-K to 50
makes the algorithm's hot pool size match the workload's natural skew on
the Skewed evaluation point. For Uniform workloads with no inherent skew,
50 still represents the "most recently active" — temperature naturally
selects them.

### Why decay rate `α = 0.1` default

A new request adds `T_1 = 1.0`. After one decay step, an adapter at
temp=1 with mean ≈ 0.5 (mid-distribution) drops by 0.1·(1 − 0.5) = 0.05.
So an adapter needs another request within ~10 steps to stay in the top
half. This roughly matches the LRU "stale after 10 steps" intuition.

### Why we report tokens/s instead of latency

- Tokens/s is what the paper reports
- Latency per request requires modeling request arrival times, which we
  don't have a workload trace for
- Steady-state throughput is unambiguous: tokens / wall time

### Why we don't have a multi-GPU mode

Paper Figure 18 covers multi-GPU. We don't because:
- The event sim has no multi-GPU state machine
- Multi-GPU strategy selection (which GPU caches what) is a different
  algorithm than Algorithm 1
- The reviewers' primary asks were single-GPU baselines

---

## 14. Modeling approximations — what's exact, what's approximate

### Exact (to the nanosecond)

- CXL link transfer time (`bytes / cxl_bandwidth`)
- DRAM read time inside CXL device (`bytes / dram_bandwidth`)
- State machine sequence per request (CA transfer → analyze → DRAM read → ...)
- All hand-derived microbench cases (verified by `verify_*.json`)

### Approximate (within stated tolerance)

| approximation | bound | impact |
|---|---|---|
| `24·D²` base-model FLOP coefficient | 3% high for FFN_DIM=3.5D (Llama3) | per-step base time low by 3% |
| GPU LoRA compute (E1, E2, E4) not in C sim | LoRA FLOPs / base FLOPs ≈ 7R/(6D) ≈ 0.4% at R=16, D=4096 | total step time low by ≤1% |
| NDP PE compute = 0 in C sim (`addr_num-1` quirk) | NDP throughput vs link bandwidth at typical R: ~5% of step | step time low by ≤5% |
| Per-adapter sequential memory accounting | ≤6% over-allocation at typical free=26 GB | rare; only matters at tight budgets |
| Round-robin adapter→device (no work-stealing) | suboptimal load balance with N_adapters ≤ N_CXL | irrelevant at typical 1000 adapters |
| One JSON = one fused decoder pass | no inter-layer pipelining | step time correct; per-layer latency breakdown not extractable |
| Static KV per step (no growth) | true KV grows by 1 token per request per step | steady-state averages are correct |
| Static base-model HBM-load (no L2 reuse) | assumes full 14 GB reload per step | slightly pessimistic for short workloads |

### Intentionally crude (for reviewer-friendliness, not accuracy)

| choice | why |
|---|---|
| NoCxl `offloads_per_layer = 2` (batched LoRA) | per-matrix would penalize NoCxl 4× — strawman territory |
| CPU-LoRA `cpu_compute = 200 GFLOPS` default | AVX-512 + oneDNN can hit 500–2000 GFLOPS; 200 is conservative-pessimistic |
| Hot pre-cache reserves serving E1 footprint | even E4 serving adapters reserve E1 bytes; under-fills hot cache | 

### Conservatively pessimistic (vs reality)

These bias *away from CLoRA's winning claim*, making the reviewer story
harder to defend (and therefore stronger when verified):

- NoCxl per-layer offload count = 2 (not 1)
- CPU-LoRA assumes no L3 caching of A,B between layers
- CPU-LoRA assumes per-layer offload overhead at every layer
- Grace-Hopper L_c2c_setup = 500 ns (real systems may achieve ≈ 0)

---

## 15. Edge cases handled

### In `clora_strategy.py`

| case | behavior |
|---|---|
| `B = 0` (dormant adapter) | `cost_lora` returns 0 for all strategies; HBM term clamped to 0 |
| `gpu_mem_bytes = 0` | budget becomes negative; only E2/E3 (0-byte) strategies feasible |
| `state.adapters` empty | algorithm raises `KeyError` if asked about an unknown id |
| `kv_tokens_per_adapter` empty | `compute_kv_in_gpu_fraction` returns 0.0 |
| no non-serving adapters (all in batch) | `classify_adapters` returns `(serving, [], [])` |
| single non-serving adapter | mean = its temp → lands in "hot" by `temp >= mean` predicate |
| no candidates feasible in Algorithm 1 | raises `RuntimeError` with budget details |

### In `baseline_cpu_offload.py`

| case | behavior |
|---|---|
| empty miss set | all costs = 0; total = base + attn |
| empty hit set | `t_cached_gpu` = 0 |
| `total_kv = 0` | `t_attn` = 0 |
| cache capacity = 0 | every touch is a miss; never inserts |
| cache too small for any single adapter | silently fails to cache, every touch misses |
| `total_batch = 0` | step time = 0 (degenerate) |

### In `baseline_no_cxl.py`

| case | behavior |
|---|---|
| `offloads_per_layer = 0` | overhead = 0; behaves identically to CLoRA (sanity ablation) |
| `n_layers ≤ 0` | overhead = 0 |
| `clora_step_ns = 0` | returns just the overhead |

### In `baseline_grace_hopper.py`

| case | behavior |
|---|---|
| infinite `c2c_bw` | transfer → 0; compute-bound (see test `t6`) |
| zero `c2c_bw` | division-by-zero protection in `_ns_transfer` returns 0 |
| empty miss set | LoRA path collapses to cached + kernels |

### In `clora_driver.py`

| case | behavior |
|---|---|
| sub-second consecutive C-sim invocations | `--timestamp <pid>_<counter>` avoids `raw/<timestamp>/` collisions |
| C sim returns non-zero exit | driver raises `RuntimeError` with stderr |
| no `CLORA_RESULT` line in C-sim output | driver raises with last 2000 chars of stdout |
| binary not built | clear error message: `"binary not built: ./main — run 'make all' first"` |

---

## 16. Verification approach

### Three layers of verification

**(a) Microbench verification of C event sim** — `verify_read.json` and
`verify_rc.json` exercise the CXL state machine with single sub-requests
whose timing can be computed by hand:

```
READ:
  30 (tCMDCXL) + 10 (tANALYZE) + 10 (tCMDDRAM) + 271 (DRAM read for 256KB)
                + 2048 (link transfer for 256KB at 128 GB/s)
  = 2369 ns        (sim reports 2368, off by 1 from integer truncation)

READ_COMPUTE (attn, kv=400, 4 devices):
  94 + 10 + 10 + 270 + 1 + 1 + 64  =  450 ns        (sim reports 450)
```

These run as part of the regression suite. Any C-sim drift would
immediately fail them.

**(b) Hand-derived Python tests** — every cost-model assertion in the
test suites cites the expected number with its derivation. Example from
`test_baseline_grace_hopper.py::t1`:

```
bytes_per_op       = 2 · 16 · 4096 · 2  = 262,144 B
t_c2c_transfer     = 500 + 262144 / 450e9 * 1e9 = 1,082.54 ns
t_gpu_lora_compute = 4 · 1 · 4096 · 16 / 100e12 * 1e9 = 2.62 ns
t_kernels_lora     = 1 · 5000 = 5,000 ns
t_missed_lora      = max(1082.54, 2.62) + 5000 = 6,082.54 ns
total              = base (4,026.53) + missed (6,082.54) ...
                   = max(4,026.53, 6,082.54)   (parallel) = 6,082.54 ns
```

100 such assertions across 4 modules, all green.

**(c) Coverage tests for strategy selection** — `t9_force_e1` through
`t13_e2_vs_e3_crossover` in `test_clora_strategy.py` construct
hand-picked `(B, R, gpu_budget)` triples and assert that each of
E1/E2/E3/E4 wins under the predicted conditions. This is a *permanent
regression check* that the cost model covers all four regions of the
design space (paper Figure 2).

### Regression policy

Every code change must pass:
```
python3 script/test_clora_strategy.py        # 35 assertions
python3 script/test_baseline.py              # 31 assertions
python3 script/test_baseline_no_cxl.py       #  9 assertions
python3 script/test_baseline_grace_hopper.py # 25 assertions

make all
./main --file script/verify_read.json --timestamp t1    # duration_ns=2368
./main --file script/verify_rc.json   --timestamp t2    # duration_ns=450
```

Total: **100 Python assertions + 2 C microbenches**, all expected to
pass.

### What we do *not* verify automatically

- Paper-table-faithfulness (compared manually to Figure 11 bars; off by
  ~3% on 7B avg, ~10% on 13B avg — see `out.md`)
- Multi-step temperature convergence (visual inspection only)
- Per-workload optimal batch (not modeled — paper varies it, we fix at 32)

---

## Document end

For everyday driving, use [`DESIGN.md`](./DESIGN.md). For modifying
or extending the simulator, return here for the formulas and the
"why we did it this way" notes.

If you find a defect, the test suite is the right place to encode the
fix as a regression. If you find a paper-formula inconsistency, both
this file and `DESIGN.md` should be updated alongside the code change.
