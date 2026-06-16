# CLoRA Hardware Sensitivity Study

How CLoRA's decode performance moves as each hardware knob is varied, in
**absolute** terms (no normalization), across ranges chosen to **cross
each knob's knee** so bottlenecks are actually visible.

## Why the first cut looked "flat" (and what fixed it)

An earlier version swept each knob over a narrow, realistic range and
found almost no change. That was misleading: at the paper's operating
point the swept knobs were all **well above their knees** (over-provisioned),
and the *dominant* bottleneck — device DRAM bandwidth — was not even in
the set. Stress tests confirmed the knobs do bottleneck at extreme values
(CXL link at 1 GB/s → 357 tok/s; device DRAM at 68 GB/s → 175 tok/s). This
study therefore (1) widens each range to span the knee and (2) adds
**device DRAM bandwidth** as a fifth knob.

## Experiment design

One-factor-at-a-time around the paper's operating point (Table 4: A100,
4 CLoRA devices, **Llama2-7B**, batch 32, 1000 adapters, seed 42, median
of 3 trials). **HW-only**: for each knob we fix the workload, model,
platform, and the strategy/policy (the Python cost model stays at default)
and vary only the C-simulator hardware parameter, re-timing the identical
request stream. This isolates the raw hardware effect (a coupled sweep
that also moves the cost model adds non-monotone strategy-reopt noise).

**Two workload regimes per knob:**
- **Uniform (short-KV, 100–1024 tok)** — HBM-bound: the step is pinned at
  the base-model HBM floor (14 GB / 1935 GB/s = **7.235 ms**, 4,423 tok/s)
  and is insensitive until a knob is starved so hard the CXL work can no
  longer hide under it.
- **Uniform-long (long-KV, 2048–4096 tok)** — the CXL/NDP attention path is
  on the critical path; this is where the knobs bite.

Figures (absolute, log–log, shared y-axis): `fig/fig_sensitivity_throughput.{png,pdf}`
(tok/s, higher better) and `fig/fig_sensitivity_tpot.{png,pdf}` (ms/token,
lower better). Gray dotted = paper default; black dashed = HBM-bound limit.
Run `python3 script/run_sensitivity.py` then `python3 script/make_sensitivity_figure.py`.
Data: `script/sensitivity_abs.json`.

## Which knobs are bottlenecks (long-KV throughput span)

| knob | long-KV range (tok/s) | span | bottleneck? |
|---|---|---:|---|
| Device DRAM bandwidth | 165 → 3,011 | **18×** | **yes — dominant** |
| NDP core throughput | 640 → 2,786 | **4.4×** | **yes (knee at 2 TFLOPS)** |
| CXL link bandwidth | 801 → 2,804 | **3.5×** | **yes (below ~16 GB/s)** |
| CXL link latency | 2,769 → 2,796 | 1.0× | no (flat) |
| NDP controller buffer | 2,756 → 2,802 | 1.0× | no (flat) |

(NDP/DRAM/CXL-BW numbers with the FP16-corrected conf that now matches
Table 4 exactly: NDP 2 TFLOPS = 2000 GOPS, DRAM 8 × 136 GB/s = 1088 GB/s,
`data type = 16`.)

---

## Results

### (a) CXL link latency — 200 → 6400 ns one-way  → flat

| latency (ns) | short-KV | long-KV |
|---:|---:|---:|
| 200 (default) | 4,423 | 2,796 |
| 1600 | 4,423 | 2,790 |
| 6400 | 4,423 | 2,769 |

A 32× latency increase costs **1%** — per-request latency is amortized
over MB-scale KV transfers. Latency is genuinely not a bottleneck.

### (b) CXL link bandwidth — 2 → 512 GB/s  → knee at ~16–32 GB/s

| bandwidth (GB/s) | short-KV | long-KV |
|---:|---:|---:|
| 2 | 702 | 793 |
| 4 | 1,404 | 1,596 |
| 8 | 2,986 | 2,542 |
| 16 | 4,423 | 2,706 |
| 32 | 4,423 | 2,756 |
| 128 (default) | 4,423 | 2,796 |
| 512 | 4,423 | 2,804 |

Below ~16 GB/s the link bottlenecks **both** workloads hard (4× drop at
2 GB/s). Above ~32 GB/s it saturates — so the paper's CXL 3.1 link
(128 GB/s) is comfortably past the knee, with ~4× headroom. This is the
core CXL+NDP claim: because NDP keeps the big data (KV) on-device and only
small Q/O cross the link, modest link bandwidth suffices.

### (c) NDP core throughput — 0.25 → 8 TFLOPS  → cliff, knee at 2 TFLOPS

| NDP TFLOPS | short-KV | long-KV |
|---:|---:|---:|
| 0.25 | 2,374 | 640 |
| 0.5 | 4,369 | 1,268 |
| 1.0 | 4,423 | 2,482 |
| 2.0 (default) | 4,423 | 2,756 |
| 4.0 | 4,423 | 2,775 |
| 8.0 | 4,423 | 2,786 |

Below ~2 TFLOPS the PE is slower than its own DRAM scan and attention
compute stops hiding — long-KV throughput halves each time NDP halves
(2,756 → 1,268 → 640), and even short-KV dips at 0.25. Saturates just
above 2 TFLOPS, so the paper provisions **exactly at the knee**. (With the
FP16-corrected element count — `data type = 16` — the knee is at 2 TFLOPS;
the earlier draft's `data type = 32` undercounted NDP ops 2× and put the
knee at ~1.)

### (d) Device DRAM bandwidth — 68 → 4352 GB/s  → the dominant bottleneck

| DRAM bw (GB/s) | short-KV | long-KV |
|---:|---:|---:|
| 64 | 613 | 165 |
| 136 | 1,300 | 351 |
| 272 | 2,596 | 701 |
| 544 | 4,419 | 1,401 |
| 1088 (default) | 4,423 | 2,796 |
| 2176 | 4,423 | 3,011 |
| 4352 | 4,423 | 3,011 |

**Nearly linear** below 1 TB/s — long-KV throughput tracks DRAM bandwidth
because reading the distributed KV cache is the largest term on the
critical path. The paper's 1.1 TB/s sits on the **rising** part (2,796,
vs 3,011 saturated), so device DRAM bandwidth is the binding resource at
the default operating point, alongside NDP throughput at its knee. (This
is the knob the reviewer list omitted; it's the real story.)

### (e) NDP controller buffer — 1 → 32 in-flight  → flat

| in-flight | short-KV | long-KV |
|---:|---:|---:|
| 1 | 4,423 | 2,756 |
| 8 (default) | 4,423 | 2,796 |
| 32 | 4,423 | 2,762 |

Within ±1.5%: at decode batch sizes the admission window never gates — a
single in-flight request nearly saturates the device.

---

## Takeaways

1. **CLoRA is balanced at the default point**, co-limited by **device DRAM
   bandwidth** (on its rising region) and **NDP throughput** (at its knee).
   Neither is grossly over- or under-provisioned.
2. **CXL link bandwidth and latency are not the bottleneck** at the
   paper's 128 GB/s — the link has ~4× headroom and latency is amortized.
   This is the strongest evidence for the CXL+NDP design: it deliberately
   moves the bottleneck off the bandwidth-limited link and onto the
   high-bandwidth on-device DRAM/NDP.
3. **NDP controller buffer is not a bottleneck** at decode batch sizes.
4. Short-KV decode is HBM-bound and only perturbs when a knob is starved
   enough to break the hiding of CXL work under the GPU floor.

## Methodology notes

- **NDP PE timing.** The NDP-throughput sweep uses the opt-in ops-based PE
  model (`ndp_compute_model = 1`); the legacy formula collapses to ~1 ns
  for `addr_num==1` sub-requests and would flat-line the panel. Off by
  default elsewhere; <1% effect at the default 2 TFLOPS.
- **Device DRAM bandwidth** is swept via `dram_channel_num` × 17 GB/s/channel
  (default 64 → 1088 GB/s).
- **Conf gotcha:** `config/parameters.conf` has no trailing newline; appended
  keys must be newline-prefixed (`run_sensitivity.py::make_conf` handles it).
- **Jitter:** C-sim channel order uses `srand(time)`, ~±1% on long-KV; medians of 3.
