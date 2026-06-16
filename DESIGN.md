# CLoRA Simulator — Design & Controls

This document explains the **architecture** of every system we modeled,
how each one is **implemented**, and the **levers** (CLI flags + config
parameters) available to control them. For background on the CLoRA paper
itself, see the MICRO 2026 submission referenced in `out.md`.

---

## Table of Contents

1. [Overview](#overview)
2. [System 1 — CLoRA (Ours)](#system-1--clora-ours)
3. [System 2 — CLoRA-NoCXL baseline](#system-2--clora-nocxl-baseline)
4. [System 3 — CPU-LoRA-Offload baseline](#system-3--cpu-lora-offload-baseline)
5. [System 4 — Grace-Hopper Offload baseline](#system-4--grace-hopper-offload-baseline)
6. [Repository layout](#repository-layout)
7. [Per-step driver pipeline](#per-step-driver-pipeline)
8. [Complete knob reference](#complete-knob-reference)
9. [Recipes](#recipes)
10. [Tests](#tests)

---

## Overview

We simulate **four multi-LoRA serving systems** on a shared workload
generator with identical hardware (where applicable), so any side-by-side
throughput number is apples-to-apples. The driver can run any one system
in isolation or all four in parallel and report comparative speedups.

| system | LoRA matrices live in | LoRA matmul runs on | GPU↔remote interface | what it isolates |
|---|---|---|---|---|
| **CLoRA (Ours)** | CXL devices | NDP cores (CLoRA devices) | CXL.mem + read-compute (integrated) | the full proposed design |
| **CLoRA-NoCXL** | PCIe-NDP devices | NDP cores (same as CLoRA) | PCIe + explicit DMA + kernel relaunches | value of the CXL **interface** |
| **CPU-LoRA-Offload** | Host DRAM | **CPU cores** (~500 GFLOPS) | PCIe + `cudaMemcpyAsync` | weak strawman: move compute *and* data to CPU |
| **Grace-Hopper Offload** | Grace CPU memory | **GPU** (S-LoRA/PUNICA style) | NVLink-C2C 450 GB/s coherent | strong baseline: fast coherent CPU-GPU link |

A single driver invocation:

```bash
python3 script/clora_driver.py [knobs...] \
        [--baseline {none, cpu_offload, no_cxl, grace_hopper, all}]
```

generates the same Uniform/Skewed workload, runs the appropriate
system(s), and prints per-step + aggregate throughput.

### Expected ordering

With the **primary (unfused) NoCXL** launch model, the CXL interface is a
first-order effect and CLoRA-NoCXL drops well below the GPU-resident
baselines on short/GQA/MoE workloads:

```
CPU-LoRA-Offload  <  CLoRA-NoCXL(unfused)  ≲  Grace-Hopper  ≪  CLoRA
   (CPU matmul)        (per-adapter launches)   (GPU+C2C)        (CXL+NDP)
```

Under the **fused** NoCXL model (S-LoRA/BGMV batching) the original
ordering is restored: `CPU < Grace-Hopper < CLoRA-NoCXL < CLoRA`, with
CLoRA/NoCXL ≈ 1.3–1.6×. See RESULTS.md §4 for both columns.

The four-way comparison answers two reviewer questions cleanly:
- **CLoRA vs CLoRA-NoCXL (6–12× unfused / 1.3–1.6× fused)**: the CXL interface adds
- **CLoRA vs Grace-Hopper (~5.9×)**: NDP placement matters even with a coherent 450 GB/s CPU-GPU link

---

## System 1 — CLoRA (Ours)

### What it is

The paper's full CXL+NDP architecture. GPU base model runs on HBM; LoRA
adapters and KV cache live in CXL memory expanders with NDP cores;
GPU↔CXL communication uses both load/store memory semantics and a new
"read-compute request" that lets the GPU offload a matmul to an NDP core
without any host CPU involvement.

### Architecture

```
┌──────────────────┐
│       GPU        │   HBM 1935 GB/s
│  - base weights  │   GPU FP16: 312 TFLOPS (A100)
│  - small LoRA    │
│    cache         │
│  - small KV      │
│    duplicate     │
└────────┬─────────┘
         │
         │  CXL 3.1 (128 GB/s) — load/store + read-compute requests
         │
┌────────┴───────────────────────────┐
│         CXL Switch                 │
└──┬───────┬───────┬───────┬─────────┘
   │       │       │       │
 ┌─┴──┐  ┌─┴──┐  ┌─┴──┐  ┌─┴──┐
 │NDP │  │NDP │  │NDP │  │NDP │  4 CLoRA devices
 │DRAM│  │DRAM│  │DRAM│  │DRAM│  - 1.1 TB/s device DRAM
 │ 2  │  │ 2  │  │ 2  │  │ 2  │  - 2 TFLOPS NDP each (8 TFLOPS agg)
 │TFLP│  │TFLP│  │TFLP│  │TFLP│  - holds LoRA adapters + KV cache slice
 └────┘  └────┘  └────┘  └────┘
```

### What's implemented

**(a) Event simulator core (C)** — `src/flash.c`, `src/ssd.c`, `src/initialize.c`

Implements per-device state machines for:
- Channel CA transfer (`tCMDCXL`)
- CXL controller analyze (`tANALYZE`)
- Controller→DRAM CA transfer (`tCMDDRAM`)
- DRAM read at `dram_bandwidth`
- NDP PE compute (with the pre-existing `addr_num-1` quirk)
- Channel data transfer (link at `cxl_bandwidth`)
- Per-device sub-request queues with arbitration

Two request types:
- `READ` — fetch bytes from a CXL device to GPU (used by E2)
- `READ_COMPUTE` — send input to a device, NDP computes, result returns (used by E3, E4, attention)

**(b) Per-step JSON bridge** — `include/readJson.h`, `src/readJson.c`

Schema:
```jsonc
{
  "kind":      "decode" | "prefill",
  "model":     { "d": 4096, "n_layers": 32, "n_matrices": 7, "s_dtype": 2 },
  "hw":        { "n_cxl": 4 },
  "input_tokens_per_request": 1,
  "base_model_compute_ns":    50000.0,
  "kv":        { "kv_tokens_total": 18000, "kv_in_gpu_fraction": 0.14 },
  "adapters": [
    { "id": 7, "rank": 16, "batch": 4, "strategy": 3, "kv_tokens": 562 },
    ...
  ]
}
```

**(c) Strategy plugin (Python)** — `script/clora_strategy.py`

Five components, all paper-section-marked:

| component | paper § | implements |
|---|---|---|
| `cost_lora(...)` | §6.3 Eq (1)–(6) | per-strategy time for one adapter |
| `gpu_mem_for(...)` | §5.1 Fig 7(a) | bytes consumed in GPU by each strategy |
| `choose_strategy(...)` | §6.2 Algorithm 1 | sort → fits → cold-evict → variants |
| `classify_adapters(...)` | §6.2 | three-way split (serving / hot / cold) |
| `pre_cache_hot_adapters(...)` | §6.2 + Fig 10 | top-K hottest non-serving cached as E1 |
| `TemperatureModel` | §6.2 | `+T_1` on request, mean-reverting decay |
| `compute_kv_in_gpu_fraction(...)` | §5.2 + Eqs (7)–(9) | cost-aware P_KV: balances GPU-side vs CXL-side attention under the memory cap (0–16% in practice, cf. paper Fig 16) |
| `estimate_base_model_ns_per_layer(...)` | not in paper, complement | `max(compute, HBM)` per layer, including the duplicated-KV attention share (Eq 7) |
| `emit_step_json(...)`, `write_step_json(...)` | — | serialize decisions to disk |

The four LoRA execution strategies (per-adapter):

| strategy | placement | GPU mem | link bytes per call | who computes |
|---|---|---:|---|---|
| **E1** | A,B both in GPU HBM | 2·R·D·S | 0 | GPU |
| **E2** | A,B in CXL | 0 | 2·R·D·S | GPU (loads via CXL.mem) |
| **E3** | A,B in CXL | 0 | 2·B·D·S | NDP (full LoRA: `xAB`) |
| **E4** | A in CXL, B in GPU | R·D·S | B·(D+R)·S | NDP (`m=xA`) + GPU (`y=mB`) |

(Sizes scale by `n_layers × n_matrices` for the actual cache footprint.)

### Per-step pipeline (CLoRA)

```
1. Generate batch of requests for this step (uniform or skewed)
2. For each request: temp_model.on_request(adapter_id) → += T_1
3. Build SystemState (serving + dormant adapters with their temps)
4. classify_adapters(state, top_k=50) → (serving, hot, cold)
5. pre_cache_hot_adapters(state) → mutate state.a_in_gpu/b_in_gpu for top-K
6. For each serving adapter:
       choose_strategy(adapter_id, state) →
            sort 4 strategies by cost_lora
            if cheapest fits in budget → return it
            elif cold can free enough bytes → return best with cold eviction
            else enumerate (strategy × {fits/KV/cold/hot/serving eviction})
                 → return lowest-cost feasible
7. compute_kv_in_gpu_fraction(decisions, state) → fraction
8. emit_step_json(decisions, kv_in_gpu_fraction, ...) → write to disk
9. Invoke C simulator → step_ns
10. temp_model.step() → mean-reverting decay
```

### Levers

**Algorithm-side knobs** (Python):

| flag | default | controls |
|---|---:|---|
| `--temp-increment` | 1.0 | T_1 — temperature bump per request |
| `--temp-decay` | 0.1 | α — fraction of (temp − mean) shed per step |
| `--top-k-hot` | 50 | size of the proactive hot-adapter pool; `0` disables proactive caching |

**Hardware knobs** (Python side):

| flag | default | controls |
|---|---:|---|
| `--gpu-compute-tflops` | 989 | GPU FP16 peak. A100 = 312, H100 = 989 |
| `--gpu-mem-bw-gb` | 3350 | HBM bandwidth. A100 = 1935, H100 = 3350 |
| `--gpu-mem-gb` | 8.0 | bytes free for LoRA cache + KV duplication after base model |
| `--base-model-gb` | 14.0 | base-model weight footprint (7B = 14, 8B = 16, 13B = 26) |
| `--model-d` | 4096 | model dimension D |
| `--n-layers` | 32 | decoder layers |

**CXL/NDP hardware knobs** (live in `config/parameters.conf`, read by the C sim):

| key | typical | controls |
|---|---:|---|
| `cxl_bandwidth` | 128 (B/ns ≡ GB/s) | CXL link bandwidth per device |
| `dram_channel_num` | 64 | DRAM channels per CXL device |
| `dram_channel_bandwidth` | 17 (B/ns) | per-channel DRAM BW → aggregate ≈ 1088 B/ns (1.1 TB/s) |
| `chip_computing_power` | 2000 (GOPS) | NDP PE throughput |
| `t_CMD_CXL` | 30 ns | command transfer on CXL link |
| `t_ANALYZE` | 10 ns | controller decode |
| `t_CMD_DRAM` | 10 ns | controller→DRAM command |
| `t_DRAM_READ_LATENCY` | 30 ns | DRAM access latency |
| `cxl_channel_number` | 16 | physical channels allocated |
| `L_CXL_switch` | 0 | reviewer-sensitivity knob: switch traversal latency |
| `L_read_compute_cmd` | 0 | reviewer-sensitivity knob: RC command overhead |

---

## System 2 — CLoRA-NoCXL baseline

### What it is

**Same NDP placement as CLoRA, no CXL interface.** Isolates the cost of
the CXL fabric+protocol from the cost of putting compute near data. NDP
devices are connected over PCIe instead of CXL; the GPU cannot integrate
remote ops into a single execution path. Every GPU↔NDP dependency
boundary requires a kernel relaunch + DMA + sync (one launch = 5 µs
kernel + 1 µs device command + 1 µs sync = **7 µs**).

**Two launch-granularity settings** (both kept; `--nocxl-fused` selects):

- **unfused (primary, default)** — without CXL.mem's fine-grained
  load/store, different adapters' remote NDP ops cannot be fused into one
  kernel, so the launch count scales with serving adapters and requests:

  ```
  launches/layer = n_adapters·(QKV[1]+O[1]+FFN[ffn_gemms]) + n_requests·attn[1]
  ```

  with `ffn_gemms = 3` (SwiGLU; used for all models including Qwen MoE).
  At batch 32 this is ~192 launches/layer → ~43 ms/step overhead, so the
  CXL interface is a **first-order** effect (CLoRA/NoCXL ≈ 6–12×) and
  NoCXL can fall *below* Grace-Hopper on short/GQA/MoE workloads.

- **fused (`--nocxl-fused`)** — S-LoRA/BGMV-style batching: one launch per
  *op* per layer (QKV+O+3 FFN + attention = 6), independent of adapter and
  request count → ~1.3 ms/step overhead, CLoRA/NoCXL ≈ 1.3–1.6×, preserving
  the `CLoRA > NoCXL > Grace-Hopper > CPU` ordering.

The truth lies between: a real PCIe-NDP system can BGMV-batch the
projection LoRA but probably cannot fuse attention across requests with
distinct KV as cheaply. Plots and headline numbers use **unfused**.

### Architecture

```
┌──────────────────┐
│       GPU        │
│  - base weights  │
│  - LoRA cache    │
└────────┬─────────┘
         │
         │  PCIe (128 GB/s, same as CXL for fairness)
         │  cudaMemcpyAsync + explicit driver commands
         │
┌────────┴───────────────────────────┐
│       PCIe-NDP devices             │  Same NDP throughput,
│  (DRAM + NDP, same as CLoRA)       │  same DRAM BW,
└────────────────────────────────────┘  same capacity.

Per GPU↔NDP offload boundary, NoCxl pays:

   L_kernel_launch    (relaunch dependent kernel)
 + L_device_command   (CPU/driver issues NDP op)
 + L_sync             (stream/event sync)
```

### What's implemented

`script/baseline_no_cxl.py` — a thin overhead layer that augments a CLoRA
simulator step time with the per-launch GPU↔NDP offload cost:

```
launches/layer = (unfused) n_adapters·(2+ffn_gemms) + n_requests
                 (fused)   (2+ffn_gemms) + 1
T_NoCxl        = T_CLoRA + launches/layer × n_layers × (L_kernel+L_device+L_sync)
```

Why a thin model: everything else (DRAM read time, NDP compute time, link
transfer time, strategy choices, memory accounting) is identical to
CLoRA. We literally take the C simulator's per-step time and add the
host-side launch overhead the CXL interface would have avoided.

### Per-step pipeline (NoCxl)

```
1-8. Identical to CLoRA (same strategy decisions, same workload, same
     C simulator run → step_ns)
9.   no_cxl_step_ns = step_ns + launches/layer × n_layers
                                 × (L_kernel + L_device + L_sync)
     where launches/layer counts per-adapter QKV/O/FFN launches and
     per-request attention launches (unfused), or one launch per op (fused).
```

### Levers

| flag | default | controls |
|---|---:|---|
| `--nocxl-L-kernel-launch-ns` | 5000 | CUDA kernel relaunch (per offload boundary) |
| `--nocxl-L-device-command-ns` | 1000 | CPU/driver issues NDP op |
| `--nocxl-L-sync-ns` | 1000 | stream/event sync |
| `--nocxl-offloads-per-layer` | 2 | how many offload boundaries per decoder layer |

**Special values for `--nocxl-offloads-per-layer`:**
- `0` — disables NoCxl penalty entirely (sanity ablation; matches CLoRA exactly)
- `1` — merges LoRA + attention into one offload (most generous)
- **`2`** — default: 1 batched LoRA op + 1 attention op
- `8` — per-matrix accounting (1 per Q/K/V/O/G/U/output + 1 attn); pessimistic

NoCxl shares all CLoRA hardware knobs (`--gpu-*`, `--n-layers`, etc.)
because everything except the offload boundary cost is identical.

---

## System 3 — CPU-LoRA-Offload baseline

### What it is

**No CXL, no NDP.** Naive offloading: LoRA adapters live in host DRAM and
the **CPU runs the LoRA matmul** itself. GPU only does base model + any
LoRA adapters cached in its LRU cache. For non-cached adapters, each
LoRA op fires `cudaMemcpyAsync` to ship `x` to a CPU staging buffer, the
CPU computes `y = xAB` reading A,B from host DRAM, then `cudaMemcpyAsync`
ships `y` back.

### Architecture

```
┌──────────────────┐
│       GPU        │           ┌──────────────────────┐
│  - base weights  │           │     Host CPU         │
│  - LoRA LRU      │           │  ~500 GFLOPS         │
│    cache (8 GB)  │           │  ~200 GB/s DRAM      │
└────────┬─────────┘           │  Holds all LoRA A,B  │
         │                     │  Holds KV cache      │
         │  PCIe (default 58)  │                      │
         │  cudaMemcpyAsync    │                      │
         └────────────────────►│                      │
                               └──────────────────────┘

Per (cache miss adapter, layer, matrix):
   1. kernel: cudaMemcpyAsync(x, CPU)         [L_kernel + L_offload]
   2. CPU loads A,B from DRAM, runs y = xAB
   3. kernel: cudaMemcpyAsync(y, GPU)         [L_kernel + L_offload]
   4. (next GPU kernel consumes y)

Fused mode: one round-trip per (layer, matrix, unique-rank)
Unfused: one per (layer, matrix, adapter)
```

### What's implemented

`script/baseline_cpu_offload.py`:

**(a) `LRUAdapterCache`** — per-adapter cache; each entry holds the
adapter's full A,B for all layers × matrices. Hits skip the offload
path; misses incur the full round-trip.

**(b) `baseline_step_ns(...)`** — per-step cost breakdown:

| component | formula | notes |
|---|---|---|
| `t_cached_gpu_lora` | `4·B·D·R / C_GPU` summed over cache hits, all layers × matrices | E1 analog |
| `t_pcie_lora` | per fused rank-group: `2·L_pcie_setup + (x+y)/pcie_bw`, × `n_layers × n_matrices` | per-call PCIe |
| `t_cpu_compute` | `4·B·D·R / C_CPU` per matrix per layer, summed | CPU matmul |
| `t_cpu_dram` | `2·R·D·S / W_DRAM_CPU` per matrix per layer per adapter | CPU loads A,B |
| `t_kernels_lora` | `2 · n_kernel_pairs · (L_kernel + L_offload)` | host-side launches |
| `t_base` | `max(24·D²·B / C_GPU, base_bytes/n_layers / W_HBM) × n_layers` | GPU base, HBM-bound at small batch |
| `t_attn` | `2·L_pcie_setup·n_layers + 2·B·D·S·n_layers/pcie_bw + max(CPU_compute_attn, CPU_DRAM_attn) + kernel_overhead` | KV in CPU |

**(c) Per-layer parallelism**: GPU base + cached-LoRA compute runs on one
stream, missed-LoRA PCIe+CPU path runs on another. Per-layer
`max(gpu_path, missed_path) + attn`. Set `--baseline-serial` for
worst-case serial accounting.

**(d) Fusion modes**:
- `fused` (default): one batched op per (layer, matrix, unique-rank) — S-LoRA/PUNICA BGMV style
- `unfused`: one op per (layer, matrix, adapter) — strawman with huge kernel count

### Per-step pipeline (CPU-offload)

```
1-7. Same workload + strategy decisions as CLoRA (decisions are computed
     but the CPU-offload baseline ignores them; it uses its own LRU cache
     hit/miss for the LoRA path)
8.   For each serving adapter: LRU touch → hit / miss bucket
9.   Compute t_pcie_lora + t_cpu_compute + t_cpu_dram + t_kernels_lora
10.  Compute t_base, t_attn
11.  Combine per-layer with chosen parallelism mode
12.  Return total_ns + breakdown
```

### Levers

| flag | default | controls |
|---|---:|---|
| `--baseline-cache-gb` | 8.0 | LRU cache capacity for adapter A,B |
| `--baseline-fusion {fused, unfused}` | fused | fused = paper-friendly BGMV; unfused = pessimistic strawman |
| `--baseline-serial` | (off) | force GPU base + CPU LoRA to serialize within a layer |
| `--cpu-compute-gflops` | 200 | CPU FP16 matmul throughput |
| `--cpu-dram-bw-gb` | 100 | CPU DRAM bandwidth |
| `--pcie-bw-gb` | 58 | PCIe effective BW. 4.0 ×16 = 28, 5.0 ×16 = 58, paper Table 4 = 128 |
| `--L-kernel-launch-ns` | 5000 | CUDA kernel launch per LoRA op |
| `--L-cpu-gpu-offload-ns` | 10000 | compound GPU↔CPU round-trip overhead (sync + dep kernel launch) |
| `--L-pcie-setup-ns` | 1000 | per-DMA PCIe setup |

---

## System 4 — Grace-Hopper Offload baseline

### What it is

A **strong CPU-memory baseline** that addresses reviewer A's concern:
"would a tightly-coupled coherent CPU-GPU interconnect like NVIDIA
NVLink-C2C obviate the need for CXL+NDP?" This is the same architecture
class as S-LoRA / PUNICA (GPU does the LoRA matmul, adapters live in
host memory), but with the PCIe link replaced by NVLink-C2C between
Grace CPU and Hopper GPU at 450 GB/s per direction (900 GB/s
bidirectional, what NVIDIA reports for GH200).

It is **not a new CXL or NDP baseline.** No CXL, no NDP. Just a much
faster coherent CPU-GPU link than PCIe.

### Architecture

```
┌──────────────────┐
│       GPU        │           ┌──────────────────────┐
│  (H100 SXM)      │           │   Grace CPU memory   │
│  - base weights  │           │  - all LoRA A,B      │
│  - LoRA LRU      │           │  - KV cache          │
│    cache (E1)    │           │                      │
│  - LoRA matmul   │           │  CPU does NOT        │
│    (this is the  │           │  compute LoRA        │
│     key diff vs  │           │                      │
│     CPU-LoRA)    │           └──────────┬───────────┘
└────────┬─────────┘                      │
         │                                │
         │   NVLink-C2C 450 GB/s          │
         │   coherent — GPU streams       │
         │   A,B from CPU during matmul   │
         └────────────────────────────────┘

Per cache miss, per (layer, matrix, rank-group):
   stream A,B from CPU memory   [transfer time = bytes / 450 GB/s]
   GPU matmul                   [compute = 4·B·D·R / C_GPU]
   transfer and compute overlap via coherent access
   1 kernel launch per fused op
```

### What changes vs CPU-LoRA-Offload

| feature | CPU-LoRA-Offload | Grace-Hopper |
|---|---|---|
| LoRA matmul runs on | CPU (~500 GFLOPS) | **GPU (~989 TFLOPS on H100)** |
| Link | PCIe (28–128 GB/s) | **NVLink-C2C 450 GB/s coherent** |
| Per-op host overhead | `L_kernel_launch + L_cpu_gpu_offload` (15 µs) | only `L_kernel_launch` (5 µs); no explicit DMA sync |
| Setup overhead per DMA | `L_pcie_setup` (1 µs) | `L_c2c_setup` (0.5 µs) |
| Adapters / KV live in | host DRAM | Grace CPU memory |

### What's implemented

`script/baseline_grace_hopper.py` — ~170 LOC, reuses `LRUAdapterCache`
from `baseline_cpu_offload`:

**(a) `GraceHopperHWConfig`** — link, GPU, and overhead constants

**(b) `grace_hopper_step_ns(...)`** — per-step cost breakdown:

| component | formula | notes |
|---|---|---|
| `t_cached_gpu_lora` | `4·B·D·R / C_GPU` summed over cache hits, all layers × matrices | E1 analog: pure GPU |
| `t_c2c_transfer` | `L_c2c_setup + bytes(A,B) / c2c_bw` per fused op × layers × matrices | adapter loads |
| `t_gpu_lora_compute` | `4·B·D·R / C_GPU` per matrix per layer (GPU side) | the matmul |
| `t_missed_lora` | `max(c2c_transfer, gpu_lora_compute) + kernel_overhead` | transfer/compute overlap (coherent) |
| `t_base` | `max(24·D²·B / C_GPU, base_bytes/n_layers / W_HBM) × n_layers` | base model |
| `t_attn` | `max(kv_bytes/c2c_bw, attn_flops/C_GPU) + kernel_overhead` per layer × n_layers | KV streamed over C2C |

**(c) Per-layer parallelism**: GPU base + cached-LoRA compute on one
stream, missed-LoRA C2C-load + GPU compute on another. Per-layer
`max(gpu_path, missed_path) + attn`.

**(d) Fusion modes** (shared with CPU-LoRA baseline via `--baseline-fusion`):
- `fused` (default): 1 batched op per `(layer, matrix, unique-rank)` — S-LoRA/PUNICA BGMV style
- `unfused`: 1 op per `(layer, matrix, adapter)` — strawman

### Per-step pipeline (Grace-Hopper)

```
1-7. Same workload, same strategy decisions and KV fraction as CLoRA
     (Grace-Hopper has its own independent LRU cache; it sees only the
      serving-adapter list, ignores the strategy decisions, ignores
      hot pre-cache)
8.   For each serving adapter: LRU touch → hit / miss bucket
9.   Compute t_c2c_transfer + t_gpu_lora_compute + t_kernels_lora
10.  Compute t_base, t_attn
11.  Combine per-layer with chosen parallelism mode
12.  Return total_ns + breakdown
```

### Levers

| flag | default | controls |
|---|---:|---|
| `--gh-c2c-bw-gb` | 450 | NVLink-C2C effective bandwidth (GB/s) |
| `--gh-L-c2c-setup-ns` | 500 | per-transfer C2C setup (lower than PCIe's 1000) |
| `--gh-L-kernel-launch-ns` | 5000 | CUDA kernel launch overhead |
| `--gh-cache-gb` | 8.0 | GPU LRU cache budget |

Shares with CPU-LoRA-Offload (via shared CLI flags):
- `--baseline-fusion {fused, unfused}`
- `--baseline-serial`
- `--gpu-compute-tflops`, `--gpu-mem-bw-gb`
- `--base-model-gb`, `--model-d`, `--n-layers`

### Sensitivity story

Bandwidth-sweep result that reviewers can verify:

```bash
for BW in 200 450 900 1800; do
    python3 script/clora_driver.py [...flags...] \
        --baseline grace_hopper --gh-c2c-bw-gb $BW
done
```

Even at **1800 GB/s** (4× the actual NVLink-C2C), CLoRA still wins on
long-sequence workloads by **~5×** because the gap is dominated by
*data volume crossing the link*, not raw link bandwidth. NDP at the
device eliminates that crossing.

---

## Repository layout

```
cxl_sim_clora/
├── config/
│   └── parameters.conf          # CXL / DRAM / NDP timing constants (C-sim)
├── include/
│   ├── readJson.h               # JSON schema struct (CLoRA-specific)
│   ├── initialize.h             # event-sim core types
│   └── flash.h, ssd.h, ...
├── src/
│   ├── main.c                   # npu_process — per-step issuer
│   ├── readJson.c               # cJSON parser
│   ├── flash.c                  # CXL state machines (untouched)
│   ├── ssd.c                    # event loop (untouched)
│   ├── initialize.c             # parameter loading (untouched)
│   └── pagemap.c
├── lib/
│   ├── cJSON/                   # JSON library
│   └── avlTree/
├── script/
│   ├── clora_strategy.py            # Algorithm 1, cost model, hot/cold, KV
│   ├── clora_driver.py              # workload + driver orchestrator
│   ├── baseline_cpu_offload.py      # CPU-LoRA-offload model
│   ├── baseline_no_cxl.py           # NoCxl overhead model
│   ├── baseline_grace_hopper.py     # Grace-Hopper offload model
│   ├── test_clora_strategy.py       # 35 strategy / Algorithm 1 tests
│   ├── test_baseline.py             # 31 CPU-offload tests
│   ├── test_baseline_no_cxl.py      # 9 NoCxl tests
│   ├── test_baseline_grace_hopper.py # 25 Grace-Hopper tests
│   ├── verify_read.json             # CXL state-machine microbench (READ)
│   ├── verify_rc.json               # CXL state-machine microbench (READ_COMPUTE)
│   └── clora_step.json              # last emitted step (transient)
├── main                          # built C binary
├── readme.md                     # original SSDsim notes (preserved)
└── DESIGN.md                     # this file
```

**What we built:**
- `script/clora_strategy.py`, `clora_driver.py`
- `script/baseline_cpu_offload.py`, `baseline_no_cxl.py`, `baseline_grace_hopper.py`
- All `test_*.py` files
- `include/readJson.h`, `src/readJson.c`, `src/main.c` (rewrote from upstream)
- `config/parameters.conf` additions: `L_CXL_switch`, `L_read_compute_cmd`

**What we left untouched (original SSDsim event-sim core):**
- `src/flash.c`, `src/ssd.c`, `src/initialize.c`, `src/pagemap.c`
- `include/initialize.h`, `include/flash.h`, `include/ssd.h`, `include/pagemap.h`
- Both libraries (`lib/cJSON/`, `lib/avlTree/`)

---

## Per-step driver pipeline

A single decode step in `clora_driver.py::run_smoke`:

```
┌────────────────────────────────────────────────────────────┐
│ For each step in (warmup + measurement):                   │
└────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. Generate `batch` requests                                 │
│    Each request → (adapter_id, kv_tokens)                    │
│    Distribution: uniform | skewed (80/20 over hot pool)      │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Temperature bump: for each request, temp[a] += T_1        │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. Build SystemState                                         │
│    - serving adapters (batch > 0): rank, batch, temp         │
│    - dormant adapters known to temp model: batch = 0         │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. Pre-cache hot adapters (paper §6.2 + Fig 10)              │
│    - classify_adapters(state, top_k)                         │
│    - reserve memory for serving (pessimistic E1 sizing)      │
│    - sort hot by temp desc, greedy-fill remaining budget     │
│    - mutate state.adapter[a].a_in_gpu = b_in_gpu = True      │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. For each serving adapter:                                 │
│    choose_strategy(adapter_id, state, hw, model)             │
│    → returns Decision(strategy, cost_ns, evictions)          │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. compute_kv_in_gpu_fraction(decisions, state, kv_total)    │
│    = min(free_gpu_after_lora, kv_bytes) / kv_bytes           │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 7. emit_step_json(decisions, kv_in_gpu_fraction, ...)        │
│    write JSON to script/clora_step.json                      │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 8. Invoke C simulator: ./main --file clora_step.json         │
│    parse CLORA_RESULT line → clora_step_ns                   │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 9. If --baseline cpu_offload or all:                         │
│    baseline_step_ns(...) on serving adapters → cpu_step_ns   │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 10. If --baseline no_cxl or all:                             │
│     no_cxl_step_ns(clora_step_ns, hw, n_layers)              │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 11. If --baseline grace_hopper or all:                       │
│     grace_hopper_step_ns(...) → gh_step_ns                   │
│     (independent LRU cache, GPU computes, C2C link)          │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 12. temp_model.step() — mean-reverting decay for next step   │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ 13. If not in warmup: accumulate tokens + per-system ns      │
│     Print per-step line                                      │
└──────────────────────────────────────────────────────────────┘
```

---

## Complete knob reference

### Workload knobs

| flag | default | description |
|---|---:|---|
| `--steps N` | 5 | measurement steps (counted toward throughput) |
| `--warmup-steps N` | 0 | steps run before measurement; temperature + caches reach steady state |
| `--batch B` | 16 | requests per decode step |
| `--n-adapters N` | 20 | adapter pool size |
| `--dist {uniform, skewed}` | uniform | request distribution across adapters |
| `--n-hot K` | 50 | skewed: number of "hot" adapters receiving 80% of requests |
| `--kv-min` / `--kv-max` | 64 / 512 | uniform range for per-request KV tokens |
| `--seed S` | 42 | RNG seed |

### Model knobs

| flag | default | description |
|---|---:|---|
| `--model-d` | 4096 | model dimension D |
| `--n-layers` | 32 | decoder layers |
| `--n-matrices` | 7 | LoRA-touched weight matrices per layer |
| `--base-model-gb` | 14.0 | base-model weight footprint (FP16) — *total* size |
| `--moe-active-base-gb` | 0.0 | MoE: HBM bytes read per step (active params only). `0` = dense behavior, falls back to `--base-model-gb` |

Quick reference:
| model | D | layers | n-matrices | base_gb | moe-active-base-gb |
|---|---:|---:|---:|---:|---:|
| Llama2-7B / Llama3-8B | 4096 | 32 | 7 | 14 / 16 | 0 (dense) |
| Llama2-13B | 5120 | 40 | 7 | 26 | 0 (dense) |
| Qwen3-30B-A3B MoE (attention only) | 2048 | 48 | 4 | 60 | 6 |
| Qwen3-30B-A3B MoE (attention + active experts) | 2048 | 48 | **28** | 60 | 6 |

`n_matrices` decomposition:
- **Dense Llama2** (7) = 4 attention (Q,K,V,O) + 3 FFN (gate, up, down)
- **MoE attention-only** (4) = Q, K, V, O only
- **MoE attention + active experts** (28) = 4 attention + 3 × 8 routed experts

### GPU + remote hardware knobs (shared by all systems)

| flag | default | description |
|---|---:|---|
| `--gpu-compute-tflops` | 989 | GPU FP16 peak. A100 = 312, H100 = 989 |
| `--gpu-mem-bw-gb` | 3350 | HBM bandwidth (GB/s). A100 = 1935, H100 = 3350 |
| `--gpu-mem-gb` | 8.0 | bytes available for LoRA cache + KV duplication |
| `--pcie-bw-gb` | 58 | PCIe effective BW. 4.0 ×16 = 28, 5.0 ×16 = 58, paper Table 4 = 128 |

CXL-specific parameters (from `config/parameters.conf`):
- `cxl_bandwidth = 128` (B/ns) — link
- `dram_channel_num = 64`, `dram_channel_bandwidth = 17` — internal DRAM
- `chip_computing_power = 2000` — NDP PE (2 TFLOPS)
- `L_CXL_switch` — switch traversal sensitivity (default 0)
- `L_read_compute_cmd` — RC overhead sensitivity (default 0)

### CLoRA strategy plugin knobs (paper §6.2)

| flag | default | description |
|---|---:|---|
| `--temp-increment T1` | 1.0 | temperature bump per arriving request |
| `--temp-decay α` | 0.1 | mean-reverting decay fraction per step |
| `--top-k-hot K` | 50 | proactive hot-adapter pool size (0 disables) |

### Baseline selection

| flag | default | description |
|---|---:|---|
| `--baseline {none, cpu_offload, no_cxl, grace_hopper, all}` | none | which baseline(s) to run alongside CLoRA |

### CLoRA-NoCXL knobs

| flag | default | description |
|---|---:|---|
| `--nocxl-L-kernel-launch-ns` | 5000 | CUDA kernel relaunch per offload boundary |
| `--nocxl-L-device-command-ns` | 1000 | CPU/driver command issue per offload |
| `--nocxl-L-sync-ns` | 1000 | stream/event sync per offload |
| `--nocxl-offloads-per-layer` | 2 | offload boundaries per layer (1 batched LoRA + 1 attention) |

### CPU-LoRA-Offload knobs

| flag | default | description |
|---|---:|---|
| `--baseline-cache-gb` | 8.0 | LRU cache capacity |
| `--baseline-fusion {fused, unfused}` | fused | kernel fusion mode |
| `--baseline-serial` | (off) | force serial GPU+CPU within a layer |
| `--cpu-compute-gflops` | 200 | CPU matmul throughput |
| `--cpu-dram-bw-gb` | 100 | CPU DRAM bandwidth |
| `--L-kernel-launch-ns` | 5000 | CUDA kernel launch per LoRA op |
| `--L-cpu-gpu-offload-ns` | 10000 | compound GPU↔CPU round-trip overhead |
| `--L-pcie-setup-ns` | 1000 | per-DMA PCIe setup |

### Grace-Hopper Offload knobs

| flag | default | description |
|---|---:|---|
| `--gh-c2c-bw-gb` | 450 | NVLink-C2C effective bandwidth (GB/s) |
| `--gh-L-c2c-setup-ns` | 500 | per-transfer C2C setup (vs 1000 for PCIe) |
| `--gh-L-kernel-launch-ns` | 5000 | CUDA kernel launch overhead |
| `--gh-cache-gb` | 8.0 | GPU LRU cache budget for LoRA adapters |
| `--baseline-fusion {fused, unfused}` | fused | shared with CPU-LoRA-offload |
| `--baseline-serial` | (off) | shared with CPU-LoRA-offload |

---

## Recipes

### Single CLoRA run (Llama2-7B, Uniform, A100)

```bash
python3 script/clora_driver.py \
    --steps 10 --warmup-steps 10 --batch 32 \
    --n-adapters 1000 --dist uniform --kv-min 100 --kv-max 1024 \
    --gpu-mem-gb 26 --base-model-gb 14 \
    --gpu-compute-tflops 312 --gpu-mem-bw-gb 1935
```

### 4-way comparison (bandwidth-matched per paper Table 4)

```bash
python3 script/clora_driver.py \
    --steps 10 --warmup-steps 10 --batch 32 \
    --n-adapters 1000 --dist uniform --kv-min 100 --kv-max 1024 \
    --gpu-mem-gb 26 --base-model-gb 14 \
    --gpu-compute-tflops 312 --gpu-mem-bw-gb 1935 \
    --pcie-bw-gb 128 \
    --baseline all
# Outputs: CLoRA ~4,423 | NoCXL ~4,165 | Grace-Hopper ~796 | CPU-LoRA ~168
```

### Only the Grace-Hopper baseline

```bash
python3 script/clora_driver.py \
    --steps 10 --warmup-steps 10 --batch 32 \
    --n-adapters 1000 --dist uniform --kv-min 100 --kv-max 1024 \
    --gpu-mem-gb 26 --base-model-gb 14 \
    --gpu-compute-tflops 312 --gpu-mem-bw-gb 1935 \
    --baseline grace_hopper
```

### Grace-Hopper bandwidth sensitivity

```bash
for BW in 200 450 900 1800; do
    echo "--- C2C = $BW GB/s ---"
    python3 script/clora_driver.py [...standard flags...] \
        --baseline grace_hopper --gh-c2c-bw-gb $BW 2>&1 | grep -E "(Grace|SPEEDUP)"
done
```

### Ablation — disable hot pre-caching

```bash
python3 script/clora_driver.py [...standard flags...] --top-k-hot 0
```

### Ablation — disable NoCxl penalty (sanity check, should match CLoRA exactly)

```bash
python3 script/clora_driver.py [...flags...] --baseline no_cxl \
    --nocxl-offloads-per-layer 0
```

### Sensitivity — vary NoCxl kernel launch overhead

```bash
for L in 500 2000 5000 10000; do
    python3 script/clora_driver.py [...flags...] \
        --baseline no_cxl --nocxl-L-kernel-launch-ns $L
done
```

### Sensitivity — vary PCIe bandwidth for CPU-offload baseline

```bash
for BW in 28 58 128; do
    python3 script/clora_driver.py [...flags...] \
        --baseline cpu_offload --pcie-bw-gb $BW
done
```

### Sensitivity — vary CLoRA CXL link bandwidth (edit `config/parameters.conf`)

```bash
sed -i.bak 's/cxl_bandwidth = .*/cxl_bandwidth = 64;/' config/parameters.conf
make all
python3 script/clora_driver.py [...flags...]
```

### Llama2-13B (only knobs that change)

```bash
python3 script/clora_driver.py [...rest...] \
    --gpu-mem-gb 14 --base-model-gb 26 --model-d 5120 --n-layers 40
```

### Qwen3-30B-A3B MoE with LoRA on attention + active experts

```bash
python3 script/clora_driver.py \
    --steps 10 --warmup-steps 10 --batch 32 \
    --n-adapters 1000 --dist uniform --kv-min 100 --kv-max 1024 \
    --gpu-mem-gb 34 --base-model-gb 60 --moe-active-base-gb 6 \
    --model-d 2048 --n-layers 48 --n-matrices 28 \
    --gpu-compute-tflops 312 --gpu-mem-bw-gb 1935 --pcie-bw-gb 128 \
    --top-k-hot 50 --seed 42 --baseline all
```

MoE-specific flags explained:
- `--base-model-gb 60` — total model footprint (30B × FP16)
- `--moe-active-base-gb 6` — active footprint per step (3B × FP16); only this many bytes are HBM-read
- `--gpu-mem-gb 34` — leftover GPU budget for LoRA + KV (assumes dormant experts paged out, modeled abstractly)
- `--n-matrices 28` — 4 attention (Q,K,V,O) + 3 × 8 active-expert FFNs

For attention-only LoRA placement (S-LoRA / PUNICA style on MoE), use `--n-matrices 4`.

### Push to steady state on long-running workload

```bash
python3 script/clora_driver.py \
    --steps 50 --warmup-steps 30 --batch 64 \
    --n-adapters 1000 --dist skewed --n-hot 50 \
    --kv-min 2048 --kv-max 4096 \
    --gpu-mem-gb 26 --base-model-gb 14 \
    --gpu-compute-tflops 312 --gpu-mem-bw-gb 1935 \
    --top-k-hot 50 --baseline all
```

---

## Tests

```bash
# Python: 100 assertions across 4 modules
python3 script/test_clora_strategy.py        # 35 — cost model, Algorithm 1, hot/cold, top-K, pre-cache
python3 script/test_baseline.py              # 31 — CPU-LoRA-offload model
python3 script/test_baseline_no_cxl.py       #  9 — NoCxl overhead formula
python3 script/test_baseline_grace_hopper.py # 25 — Grace-Hopper C2C transfer, GPU compute, attention

# C state-machine microbenches (require `make all`)
./main --file script/verify_read.json --timestamp t1   # expect: duration_ns=2368
./main --file script/verify_rc.json   --timestamp t2   # expect:  duration_ns=450
```

Strategy-coverage tests (`t9`–`t13` in `test_clora_strategy.py`) construct
hand-picked (B, R, GPU-budget) triples that force each of E1/E2/E3/E4 to
win, and verify the E2/E3 crossover at B ≈ R. Permanent regression check
that the cost model covers all four regions of the design space described
in paper Figure 2.

---

## Quick mental model

When you're driving the simulator, remember this:

```
CLoRA          = C event sim (CXL traffic + NDP timing)
                 + Python strategy plugin (per-step decisions)

CLoRA-NoCXL    = CLoRA + a constant per-layer host overhead
                 (kernel launches + DMA + sync)
                 — no other changes, all decisions identical

CPU-LoRA-Off   = Pure Python analytical model
                 (LRU cache + per-(adapter, layer, matrix) PCIe + CPU
                  + kernel launches, with per-layer GPU/CPU overlap)
                 — CPU does the LoRA matmul; weak strawman

Grace-Hopper   = Pure Python analytical model
                 (LRU cache + per-(layer, matrix, rank) C2C transfer
                  + GPU LoRA matmul with overlap, lower per-op overhead)
                 — GPU does the LoRA matmul, NVLink-C2C 450 GB/s;
                   strong baseline addressing "is CXL+NDP necessary
                   if we have coherent CPU-GPU?"
```

**Bandwidth knobs affect only the relevant system:**
- `--pcie-bw-gb` → CPU-LoRA-Offload only
- `--gh-c2c-bw-gb` → Grace-Hopper only
- CXL link bandwidth → CLoRA + CLoRA-NoCXL (via `config/parameters.conf`)

**Compute knobs:**
- `--gpu-compute-tflops` → all four (shared)
- `--cpu-compute-gflops` → CPU-LoRA-Offload only
- NDP throughput → CLoRA + CLoRA-NoCXL (via `config/parameters.conf`)

This keeps any pairwise comparison clean — fix everything except the one
axis you want to study, and run the appropriate baseline(s).

### Four-way decomposition story

```
   CLoRA  vs  NoCXL     →   1.06×   ←  cost of removing CXL interface
   CLoRA  vs  GraceH    →   5-7×    ←  cost of removing NDP (even with 450 GB/s C2C)
   GraceH vs  CPU-LoRA  →   4-5×    ←  cost of moving compute from GPU to CPU
   CLoRA  vs  CPU-LoRA  →   27-33×  ←  product of all three
```

Each pairwise ratio cleanly isolates a single design axis. Together they
tell the full reviewer story: the CXL interface, NDP placement, and
keeping LoRA matmul on GPU each contribute, and the dominant gain is
NDP placement — even at 4× the bandwidth of the CXL link.
