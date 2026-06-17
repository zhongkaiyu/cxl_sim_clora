#!/usr/bin/env python3
"""Per-layer CXL-latency model for the sensitivity study, panel (a).

The event simulator bundles all n_layers into one CXL request per adapter, so
it charges the link latency ~once per adapter. But decode is layer-SEQUENTIAL
(layer i+1 needs layer i's output), so a faithful model pays the CXL latency
per layer on the critical path: n_layers x 2 one-way hops (send activation to
NDP, receive result), serialized on top of the bandwidth/DRAM-bound CXL path,
and still overlapped with the GPU base-model step (Eq 9).

We isolate the two components from the simulator at zero added latency:
  * step_full = full decode step (GPU base-model overlapped with CXL path)
  * cxl_path  = CXL path alone (GPU base-model compute zeroed)
Then for one-way link latency L:
  throughput(L) = batch / max(step_full, cxl_path + n_layers * ROUND_TRIPS * L)

short-KV (Uniform): GPU-bound, so the latency stays hidden -> flat.
long-KV (Uniform-long): CXL-bound, so the per-layer latency adds a mild slope.

Updates the "cxl_latency" entry of script/sensitivity_abs.json in place.
"""
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DRIVER = os.path.join(HERE, "clora_driver.py")
TRIALS = 3
BATCH = 32
N_LAYERS = 32
ROUND_TRIPS = 2          # per layer: GPU->NDP (activation in) + NDP->GPU (out)
LAT_POINTS = [200, 400, 800, 1600, 3200, 6400]   # one-way CXL latency, ns

TPUT_RE = re.compile(
    r"^CLoRA\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)",
    re.MULTILINE)

# Platform via env (default A100); set SENS_* to match the run_sensitivity run.
GPU_TFLOPS = os.environ.get("SENS_TFLOPS", "312")     # H100: 989
GPU_MEMBW = os.environ.get("SENS_MEMBW", "1935")      # H100: 3350
GPU_MEMGB = os.environ.get("SENS_GPUMEM", "26")       # H100: 66
OUT_FILE = os.environ.get("SENS_OUT", "sensitivity_abs.json")

# matches run_sensitivity.py BASE (Llama2-7B), latency OFF (L_CXL_switch=0)
BASE = ["--steps", "10", "--warmup-steps", "10", "--batch", str(BATCH),
        "--n-adapters", "1000", "--dist", "uniform",
        "--gpu-mem-gb", GPU_MEMGB, "--base-model-gb", "14",
        "--model-d", "4096", "--n-layers", str(N_LAYERS),
        "--gpu-compute-tflops", GPU_TFLOPS, "--gpu-mem-bw-gb", GPU_MEMBW,
        "--top-k-hot", "50", "--seed", "42"]
WORKLOADS = {"Uniform": ("100", "1024"), "Uniform-long": ("2048", "4096")}


def run_tput(kv, extra):
    lo, hi = kv
    cmd = [sys.executable, DRIVER] + BASE + ["--kv-min", lo, "--kv-max", hi] + extra
    vals = []
    for _ in range(TRIALS):
        p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=900)
        if p.returncode != 0:
            sys.stderr.write(p.stderr[-1500:]); raise RuntimeError("driver failed")
        vals.append(float(TPUT_RE.search(p.stdout).group(1).replace(",", "")))
    return statistics.median(vals)


def main():
    comp = {}
    for wl, kv in WORKLOADS.items():
        full = run_tput(kv, [])                          # GPU+CXL overlapped
        cxl = run_tput(kv, ["--zero-base-compute"])      # CXL path alone
        step_full = BATCH / full * 1e9                   # ns
        cxl_path = BATCH / cxl * 1e9                      # ns
        comp[wl] = (step_full, cxl_path)
        print(f"{wl:13s} step_full={step_full/1e6:6.3f} ms  "
              f"cxl_path={cxl_path/1e6:6.3f} ms  "
              f"(full={full:,.0f}  cxl_only={cxl:,.0f} tok/s)", flush=True)

    rows = []
    for L in LAT_POINTS:
        row = {"x": L}
        for wl in WORKLOADS:
            step_full, cxl_path = comp[wl]
            step = max(step_full, cxl_path + N_LAYERS * ROUND_TRIPS * L)
            row[wl] = round(BATCH / (step / 1e9), 1)
        rows.append(row)
        print(f"  L={L:>5} ns  Unif={row['Uniform']:>7,.0f}  "
              f"Unif-long={row['Uniform-long']:>7,.0f}")

    path = os.path.join(HERE, OUT_FILE)
    d = json.load(open(path))
    d["cxl_latency"]["rows"] = rows
    d["cxl_latency"]["title"] = "(a) CXL link latency (per-layer)"
    d["cxl_latency"]["xlabel"] = "one-way CXL latency (ns)"
    d["cxl_latency"]["model"] = (f"per-layer: step = max(step_full, cxl_path + "
                                 f"n_layers*{ROUND_TRIPS}*L), n_layers={N_LAYERS}")
    json.dump(d, open(path, "w"), indent=2)
    print(f"\nupdated cxl_latency in script/{OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
