#!/usr/bin/env python3
"""Hardware sensitivity sweeps — ABSOLUTE metric vs parameter, HW-only.

Design (one-factor-at-a-time, isolating the hardware effect):
  * Fix the workload, the model (Llama2-7B), the platform (A100), and the
    *strategy/policy* (the Python cost model stays at its default hardware).
  * Vary ONLY the C-simulator hardware parameter (via a per-point parameter
    file) and re-time the identical request stream.
  This isolates the raw effect of each knob — no strategy re-optimization
  confound — so the curves are clean and monotonic, unlike a coupled sweep
  where the selector re-picks E1-E4 / KV-fraction at every point.

Four knobs: CXL link latency, CXL link bandwidth, NDP core throughput, NDP
controller buffer. Two workloads per knob: Uniform (short-KV, HBM-bound) and
Uniform-long (long-KV, CXL/NDP-bound). Median of 3 trials.

Writes script/sensitivity_abs.json with rows {x, Uniform, Uniform-long}.
"""
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DRIVER = os.path.join(HERE, "clora_driver.py")
BASE_CONF = os.path.join(ROOT, "config", "parameters.conf")
TRIALS = 3
BATCH = 32

TPUT_RE = re.compile(
    r"^CLoRA\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)",
    re.MULTILINE)

# Platform via env (default A100); set SENS_* for an H100 run.
GPU_TFLOPS = os.environ.get("SENS_TFLOPS", "312")     # H100: 989
GPU_MEMBW = os.environ.get("SENS_MEMBW", "1935")      # H100: 3350
GPU_MEMGB = os.environ.get("SENS_GPUMEM", "26")       # H100: 66 (80-14)
OUT_FILE = os.environ.get("SENS_OUT", "sensitivity_abs.json")

BASE = ["--steps", "10", "--warmup-steps", "10", "--batch", str(BATCH),
        "--n-adapters", "1000", "--dist", "uniform",
        "--gpu-mem-gb", GPU_MEMGB, "--base-model-gb", "14",
        "--model-d", "4096", "--n-layers", "32",
        "--gpu-compute-tflops", GPU_TFLOPS, "--gpu-mem-bw-gb", GPU_MEMBW,
        "--top-k-hot", "50", "--seed", "42"]
WORKLOADS = {"Uniform": ("100", "1024"), "Uniform-long": ("2048", "4096")}

# knob -> dict(title, xlabel, default_x, log, points=[(x, {conf overrides})]).
# Ranges are chosen to CROSS each knob's knee, so the panel shows the
# well-provisioned (flat) region AND the under-provisioned (cliff) region.
SWEEPS = {
    "cxl_latency": dict(
        title="(a) CXL link latency", xlabel="one-way CXL latency (ns)",
        default_x=200, log=True,
        points=[(200 + d, {"L_CXL_switch": d})
                for d in (0, 200, 600, 1400, 3000, 6200)]),  # total 200..6400
    "cxl_bandwidth": dict(
        title="(b) CXL link bandwidth", xlabel="link bandwidth (GB/s/device)",
        default_x=128, log=True,
        points=[(bw, {"cxl_bandwidth": bw})
                for bw in (2, 4, 8, 16, 32, 64, 128, 256, 512)]),
    "ndp_throughput": dict(
        title="(c) NDP core throughput", xlabel="NDP compute (TFLOPS/device)",
        default_x=2.0, log=True,
        points=[(tf, {"chip_computing_power": int(tf * 1000),
                      "ndp_compute_model": 1})
                for tf in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]),
    "dram_bandwidth": dict(
        title="(d) Device DRAM bandwidth", xlabel="device DRAM bandwidth (GB/s)",
        default_x=1088, log=True,
        # dram_bw = dram_channel_num * dram_channel_bandwidth; set BOTH explicitly
        # so the sweep is independent of the conf default (8 pkg x 136 GB/s).
        points=[(bw, {"dram_channel_num": 8, "dram_channel_bandwidth": bw // 8})
                for bw in (64, 136, 272, 544, 1088, 2176, 4352)]),
    "ndp_buffer": dict(
        title="(e) NDP controller buffer", xlabel="in-flight requests",
        default_x=8, log=True,
        points=[(b // 128, {"cxlctrl_buf_size": b})
                for b in (128, 256, 512, 1024, 2048, 4096)]),
}


def make_conf(overrides, dest, tag):
    with open(BASE_CONF) as f:
        text = f.read()
    for k, v in overrides.items():
        pat = re.compile(rf"^{re.escape(k)}\s*=\s*[^;#\n]*;", re.MULTILINE)
        repl = f"{k} = {v};"
        text = pat.sub(repl, text) if pat.search(text) else text + f"\n{repl}\n"
    path = os.path.join(dest, f"p_{tag}.conf")
    with open(path, "w") as f:
        f.write(text)
    return path


def run(conf_path, wl):
    lo, hi = WORKLOADS[wl]
    cmd = ([sys.executable, DRIVER] + BASE + ["--kv-min", lo, "--kv-max", hi,
            "--c-parameter-file", conf_path])   # HW-only: NO cost-model flags
    vals = []
    for _ in range(TRIALS):
        p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=900)
        if p.returncode != 0:
            sys.stderr.write(p.stderr[-1500:]); raise RuntimeError("driver failed")
        vals.append(float(TPUT_RE.search(p.stdout).group(1).replace(",", "")))
    return statistics.median(vals)


def main():
    confdir = tempfile.mkdtemp(prefix="cs_", dir="/tmp")
    out = {}
    t0 = time.time()
    for knob, spec in SWEEPS.items():
        out[knob] = {"title": spec["title"], "xlabel": spec["xlabel"],
                     "default_x": spec["default_x"], "log": spec["log"], "rows": []}
        for i, (x, conf) in enumerate(spec["points"]):
            cp = make_conf(conf, confdir, f"{knob[:6]}{i}")
            row = {"x": x}
            for wl in WORKLOADS:
                row[wl] = round(run(cp, wl), 1)
            out[knob]["rows"].append(row)
            print(f"{knob:15s} x={x:<8} Uniform={row['Uniform']:>8,.0f}  "
                  f"Uniform-long={row['Uniform-long']:>8,.0f}  "
                  f"({time.time()-t0:4.0f}s)", flush=True)
    with open(os.path.join(HERE, OUT_FILE), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote script/{OUT_FILE} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
