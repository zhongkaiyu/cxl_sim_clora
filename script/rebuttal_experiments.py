#!/usr/bin/env python3
"""Rebuttal experiment grids beyond the A100 RESULTS.md grid:

  GRID H100  -- same 3 models x 4 workloads x 4 systems on an H100 80GB
                (989 TFLOPS FP16 dense, 3350 GB/s HBM3; GPU budget =
                80 GB - base; parity caches). Links/devices unchanged from
                the A100 runs so the GPU is the only variable.
  GRID GQA   -- Llama3-8B on A100: same D/n_layers as Llama2-7B, base 16 GB,
                GQA's 4x-smaller KV cache emulated by quartering the KV token
                ranges (KV bytes/token is fixed at 2*D*S in the driver, so
                tokens/4 == bytes/4; attention in this regime is
                bandwidth-dominated, so the compute-side error is
                second-order).

Median of 3 trials, identical knobs to script/regen_results.py otherwise.
Writes script/results_h100.json and script/results_gqa.json.
"""
import json
import os
import re
import statistics
import subprocess
import sys
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

H100 = ["--gpu-compute-tflops", "989", "--gpu-mem-bw-gb", "3350"]

GRIDS = {
    "h100": {
        "out": "results_h100.json",
        "hw_note": "H100 SXM 80GB (989 TFLOPS FP16 dense, 3350 GB/s HBM3)",
        "models": {
            "Llama2-7B":  H100 + ["--model-d", "4096", "--n-layers", "32",
                                  "--base-model-gb", "14", "--gpu-mem-gb", "66",
                                  "--baseline-cache-gb", "66", "--gh-cache-gb", "66"],
            "Llama2-13B": H100 + ["--model-d", "5120", "--n-layers", "40",
                                  "--base-model-gb", "26", "--gpu-mem-gb", "54",
                                  "--baseline-cache-gb", "54", "--gh-cache-gb", "54"],
            "Qwen3-30B-A3B": H100 + ["--model-d", "2048", "--n-layers", "48",
                                     "--base-model-gb", "60", "--moe-active-base-gb", "6",
                                     "--gpu-mem-gb", "74", "--n-matrices", "28",
                                     "--baseline-cache-gb", "74", "--gh-cache-gb", "74"],
        },
        "kv_scale": 1,
    },
    "gqa": {
        "out": "results_gqa.json",
        "hw_note": "Llama3-8B (GQA: 32 Q heads / 8 KV heads -> 4x smaller KV, "
                   "emulated by 4x fewer KV tokens), A100 40GB",
        "models": {
            "Llama3-8B": ["--gpu-compute-tflops", "312", "--gpu-mem-bw-gb", "1935",
                          "--model-d", "4096", "--n-layers", "32",
                          "--base-model-gb", "16", "--gpu-mem-gb", "24",
                          "--baseline-cache-gb", "24", "--gh-cache-gb", "24"],
        },
        "kv_scale": 4,   # divide kv ranges by this (GQA emulation)
    },
}

WORKLOADS = {
    "Uniform":      ("uniform", 100, 1024),
    "Uniform-long": ("uniform", 2048, 4096),
    "Skewed":       ("skewed", 100, 1024),
    "Skewed-long":  ("skewed", 2048, 4096),
}

BASE = ["--steps", "10", "--warmup-steps", "10", "--batch", "32",
        "--n-adapters", "1000", "--pcie-bw-gb", "128",
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
    for gname, grid in GRIDS.items():
        results = {"hw_note": grid["hw_note"], "grid": {}}
        for mname, mflags in grid["models"].items():
            results["grid"][mname] = {}
            for wname, (dist, kmin, kmax) in WORKLOADS.items():
                kmin //= grid["kv_scale"]
                kmax //= grid["kv_scale"]
                wflags = ["--dist", dist, "--kv-min", str(kmin),
                          "--kv-max", str(kmax)]
                if dist == "skewed":
                    wflags += ["--n-hot", "50"]
                samples = [run_once(BASE + mflags + wflags)
                           for _ in range(TRIALS)]
                med = {}
                for k in list(LINE_RE) + ["kv_in_gpu_pct_mean"]:
                    vals = [s[k] for s in samples if s[k] is not None]
                    med[k] = round(statistics.median(vals), 1) if vals else None
                results["grid"][mname][wname] = med
                print(f"[{gname}] {mname:14s} {wname:13s} " +
                      "  ".join(f"{k}={med[k]:>9,.1f}" for k in LINE_RE) +
                      f"  kv_gpu={med['kv_in_gpu_pct_mean']}%"
                      f"  ({time.time()-t0:4.0f}s)", flush=True)
        with open(os.path.join(HERE, grid["out"]), "w") as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
