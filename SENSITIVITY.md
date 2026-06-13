# CLoRA Hardware Sensitivity Study

Reviewer-requested one-factor-at-a-time sweeps around the paper's
operating point (Table 4: A100, 4 CLoRA devices, Llama2-7B,
batch 32, 1000 adapters, seed 42, 10 warmup + 10 measured steps,
median of 3 trials). The first row of each table is the
default configuration; 'rel' columns are normalized to it.

All numbers use the corrected attention accounting (GPU-side attention
charged per paper Eq 7; cost-aware KV-duplication fraction — see
RESULTS.md §4 methodology note), which is why the Uniform-long default
reads 2,796 tok/s rather than the earlier draft's 3,324.

Knob plumbing: every point sets the C event simulator's parameter
file *and* the matching cost-model value, so both the measured
timing and the strategy-selection algorithm (Algorithm 1, Eqs 1-9)
see the same hardware.

## CXL link latency (added one-way switch latency, ns)

| point | Uniform (tok/s) | rel | Uniform-long (tok/s) | rel |
|---|---:|---:|---:|---:|
| +0 | 4,423 | 1.00x | 2,796 | 1.00x |
| +100 | 4,423 | 1.00x | 2,797 | 1.00x |
| +200 | 4,423 | 1.00x | 2,797 | 1.00x |
| +400 | 4,423 | 1.00x | 2,845 | 1.02x |
| +800 | 4,423 | 1.00x | 2,999 | 1.07x |
| +1600 | 4,423 | 1.00x | 2,991 | 1.07x |

## CXL link bandwidth per device (GB/s)

| point | Uniform (tok/s) | rel | Uniform-long (tok/s) | rel |
|---|---:|---:|---:|---:|
| 128 | 4,423 | 1.00x | 2,796 | 1.00x |
| 32 | 4,423 | 1.00x | 2,787 | 1.00x |
| 64 | 4,423 | 1.00x | 2,794 | 1.00x |
| 256 | 4,423 | 1.00x | 2,632 | 0.94x |
| 512 | 4,423 | 1.00x | 2,632 | 0.94x |

## NDP core throughput per device (FP16 TFLOPS, ops-based PE model)

| point | Uniform (tok/s) | rel | Uniform-long (tok/s) | rel |
|---|---:|---:|---:|---:|
| 2.0 (legacy PE model) | 4,423 | 1.00x | 2,796 | 1.00x |
| 2.0 | 4,423 | 1.00x | 2,775 | 0.99x |
| 0.25 | 3,909 | 0.88x | 1,558 | 0.56x |
| 0.5 | 4,423 | 1.00x | 2,178 | 0.78x |
| 1.0 | 4,423 | 1.00x | 2,800 | 1.00x |
| 4.0 | 4,423 | 1.00x | 2,651 | 0.95x |
| 8.0 | 4,423 | 1.00x | 2,656 | 0.95x |

## NDP controller buffer size (bytes; 128 B per in-flight request)

| point | Uniform (tok/s) | rel | Uniform-long (tok/s) | rel |
|---|---:|---:|---:|---:|
| 1024 (8 reqs) | 4,423 | 1.00x | 2,796 | 1.00x |
| 128 (1 reqs) | 4,423 | 1.00x | 2,756 | 0.99x |
| 256 (2 reqs) | 4,423 | 1.00x | 2,802 | 1.00x |
| 512 (4 reqs) | 4,423 | 1.00x | 2,800 | 1.00x |
| 2048 (16 reqs) | 4,423 | 1.00x | 2,785 | 1.00x |
| 4096 (32 reqs) | 4,423 | 1.00x | 2,762 | 0.99x |

---

## Analysis

**Short-KV workloads are insensitive to every knob except an extreme
NDP setting.** The Uniform column is flat at 4,423 tok/s everywhere but
NDP = 0.25 TFLOPS. At this operating point the decode step is
GPU-HBM-bound: step time (7.235 ms) equals the base-model HBM read
(14 GB / 1935 GB/s), the cost-aware policy keeps all KV on the devices
(P_KV = 0), and the device-side attention (~2.2 ms across 4 devices)
hides underneath. At 0.25 TFLOPS/device the devices become so slow that
the policy starts duplicating KV into GPU memory; the GPU-side
attention charge then breaks the HBM floor (-12%, 3,909 tok/s).

**CXL latency: negligible direct effect (-0.2% at +1600 ns,
hardware-only).** Per-request latency is amortized over MB-scale
transfers.

**CXL bandwidth: a 16x range (32 -> 512 GB/s) moves throughput by less
than 2% (hardware-only).** This directly validates the paper's central
design claim: E3-style NDP offload and distributed attention exist
precisely to take the CXL link off the critical path. Only Q-in /
partial-O-out traffic crosses the link per decode step.

**NDP throughput is the one real cliff: -22% at 0.5, -44% at 0.25
TFLOPS/device (coupled), saturating above ~1-2 TFLOPS.** Below ~1
TFLOPS the PE is slower than its own device-DRAM read, so attention
compute stops hiding behind the KV scan. The paper's 2 TFLOPS/device
provisioning (Table 4) sits at the knee: sufficient, not
over-provisioned.

**NDP controller buffer: flat within ~1% from 1 to 32 in-flight
requests.** Controller-buffer occupancy never gates progress at decode
batch sizes.

### Decomposing hardware effect vs. policy response

Each coupled sweep point updates both the simulated hardware and the
cost model, so Algorithm 1 re-selects strategies and the KV fraction at
every point. Holding the cost model at defaults and changing only the
C-sim hardware isolates the raw effect (Uniform-long, median of 3,
default = 2,796):

| point | coupled (HW + policy) | HW only | raw HW effect |
|---|---:|---:|---:|
| link bw 512 GB/s  | 2,632 (0.94x) | 2,804 (1.00x) | +0.3% |
| link bw 32 GB/s   | 2,787 (1.00x) | 2,756 (0.99x) | -1.4% |
| latency +1600 ns  | 2,991 (1.07x) | 2,789 (1.00x) | -0.2% |
| NDP 0.5 TFLOPS    | 2,178 (0.78x) | 2,482 (0.89x) | -11% |

Two honest observations follow:

1. The raw hardware effects are monotone and small for link knobs; the
   +/-6-7% wiggles in the coupled link sweeps are the strategy selector
   re-optimizing, whose effect at this operating point exceeds the
   hardware deltas being studied. The +7% at +800/+1600 ns latency
   means a latency-pessimistic cost model accidentally lands on a
   better operating point than the default — i.e., the analytic
   policy is within ~7% of the sim-optimal setting, not exactly on it.
2. At weak NDP the coupled result (2,178) is *worse* than
   hardware-only (2,482): the cost model overlaps device DRAM read and
   PE compute (Eq 8's max) while the event simulator serializes them
   per request, so the policy over-duplicates KV when NDP is slow.
   This bounds the cost-model fidelity: within a few percent at the
   paper's operating point, degrading at extreme (4x-derated) NDP
   settings.

## Methodology notes & limitations

- **NDP PE timing model.** The C sim's legacy PE-compute formula
  (`(addr_num-1) * size / elem / GOPS`, flash.c) collapses to ~1 ns for
  the `addr_num == 1` sub-requests CLoRA emits, which would make the NDP
  sweep a flat line. The study enables an opt-in ops-based model
  (`ndp_compute_model = 1` in the conf: 2 ops per element read from
  device DRAM). It is off by default, so RESULTS.md numbers are
  unaffected; at the default 2 TFLOPS the model switch changes
  Uniform-long throughput by only -0.7% (2,796 -> 2,775), confirming
  continuity.
- **Buffer scope.** `cxlctrl_buf_size` models the controller's
  request/instruction buffer (admission window), not the 3 MB on-chip
  SRAM data buffer of paper Table 6. Data-buffer capacity effects
  (tiling of oversized working sets) are not modeled; at decode-time
  request sizes this is second-order.
- **Trial jitter.** The C sim's channel-service order uses
  `srand(time(NULL))`, giving roughly +/-1% run-to-run variance on
  long-KV workloads; medians of 3 trials are reported, same as
  RESULTS.md.
- **Conf-file gotcha.** `config/parameters.conf` has no trailing
  newline; when appending keys (e.g. `ndp_compute_model = 1;`) prepend
  a newline or the key silently concatenates onto the last line and
  never parses. `sensitivity_study.py::make_conf` handles this.

## Reproducing

```bash
# full study (~3 minutes)
python3 script/sensitivity_study.py --trials 3

# one knob only
python3 script/sensitivity_study.py --only ndp_throughput

# a single custom point, by hand
python3 script/clora_driver.py ... --cxl-link-bw-gb 256 \
    --c-parameter-file my.conf      # conf with cxl_bandwidth = 256;
```

Raw per-point data: `script/sensitivity_results.json`.
