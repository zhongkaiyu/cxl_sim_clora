#!/usr/bin/env python3
"""Regenerate the RESULTS.md benchmark grid (3 models x 4 workloads x 4
systems), median of N trials, with:
  - corrected attention accounting (T_ATT_GPU charged, cost-aware P_KV)
  - GPU-memory-parity baseline caches (cache GB == CLoRA's --gpu-mem-gb)
Writes script/results_grid.json.
"""
import json
import re
import statistics
import subprocess
import sys
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DRIVER = os.path.join(HERE, "clora_driver.py")

TRIALS = 3

LINE_RE = {
    "CLoRA":        re.compile(r"^CLoRA\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M),
    "NoCXL":        re.compile(r"^CLoRA-NoCXL\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M),
    "GraceHopper":  re.compile(r"^Grace-Hopper\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M),
    "CPULoRA":      re.compile(r"^CPU-LoRA-Off\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M),
}
KV_RE = re.compile(r"kv_in_gpu=\s*([\d.]+)%")

MODELS = {
    "Llama2-7B":  ["--model-d", "4096", "--n-layers", "32",
                   "--base-model-gb", "14", "--gpu-mem-gb", "26",
                   "--baseline-cache-gb", "26", "--gh-cache-gb", "26"],
    "Llama2-13B": ["--model-d", "5120", "--n-layers", "40",
                   "--base-model-gb", "26", "--gpu-mem-gb", "14",
                   "--baseline-cache-gb", "14", "--gh-cache-gb", "14"],
    "Qwen3-30B-A3B": ["--model-d", "2048", "--n-layers", "48",
                      "--base-model-gb", "60", "--moe-active-base-gb", "6",
                      "--gpu-mem-gb", "34", "--n-matrices", "28",
                      "--baseline-cache-gb", "34", "--gh-cache-gb", "34"],
}

WORKLOADS = {
    "Uniform":      ["--dist", "uniform", "--kv-min", "100", "--kv-max", "1024"],
    "Uniform-long": ["--dist", "uniform", "--kv-min", "2048", "--kv-max", "4096"],
    "Skewed":       ["--dist", "skewed", "--n-hot", "50",
                     "--kv-min", "100", "--kv-max", "1024"],
    "Skewed-long":  ["--dist", "skewed", "--n-hot", "50",
                     "--kv-min", "2048", "--kv-max", "4096"],
}

BASE = ["--steps", "10", "--warmup-steps", "10", "--batch", "32",
        "--n-adapters", "1000", "--gpu-compute-tflops", "312",
        "--gpu-mem-bw-gb", "1935", "--pcie-bw-gb", "128",
        "--top-k-hot", "50", "--seed", "42", "--baseline", "all"]


def run_once(flags):
    proc = subprocess.run([sys.executable, DRIVER] + flags, cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=1800)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        raise RuntimeError("driver failed")
    out = {}
    for name, rex in LINE_RE.items():
        m = rex.search(proc.stdout)
        out[name] = float(m.group(1).replace(",", "")) if m else None
    kvs = [float(x) for x in KV_RE.findall(proc.stdout)]
    out["kv_in_gpu_pct_mean"] = round(sum(kvs) / len(kvs), 2) if kvs else None
    return out


def main():
    t0 = time.time()
    grid = {}
    for mname, mflags in MODELS.items():
        grid[mname] = {}
        for wname, wflags in WORKLOADS.items():
            samples = [run_once(BASE + mflags + wflags) for _ in range(TRIALS)]
            med = {}
            for k in list(LINE_RE) + ["kv_in_gpu_pct_mean"]:
                vals = [s[k] for s in samples if s[k] is not None]
                med[k] = round(statistics.median(vals), 1) if vals else None
            grid[mname][wname] = med
            print(f"{mname:14s} {wname:13s} " +
                  "  ".join(f"{k}={med[k]:>9,.1f}" for k in LINE_RE) +
                  f"  kv_gpu={med['kv_in_gpu_pct_mean']}%"
                  f"  ({time.time()-t0:4.0f}s)", flush=True)
    with open(os.path.join(HERE, "results_grid.json"), "w") as f:
        json.dump(grid, f, indent=2)
    # averages + ratios for the doc
    print()
    for mname in grid:
        avgs = {k: statistics.mean(grid[mname][w][k] for w in WORKLOADS)
                for k in LINE_RE}
        print(f"{mname} averages: " +
              "  ".join(f"{k}={v:,.0f}" for k, v in avgs.items()) +
              f"  | CLoRA/NoCXL={avgs['CLoRA']/avgs['NoCXL']:.2f}x"
              f"  CLoRA/GH={avgs['CLoRA']/avgs['GraceHopper']:.2f}x"
              f"  CLoRA/CPU={avgs['CLoRA']/avgs['CPULoRA']:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
