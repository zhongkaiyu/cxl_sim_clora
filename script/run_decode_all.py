#!/usr/bin/env python3
"""Unified decode grid for the combined TPOT figure (rebuttal).

4 systems (CLoRA, CLoRA-NoCXL, Grace-Hopper, CPU-LoRA-Offload)
x 4 models  (Llama2-7B MHA, Llama2-13B MHA, Llama3-8B GQA, Qwen3-30B GQA+MoE)
x 4 workloads (Uniform / Uniform-long / Skewed / Skewed-long, 1000 adapters)
x 2 platforms (A100 40GB, H100 80GB).

GQA is emulated as a KV-token divisor (model property): Llama3-8B = 4x
(32 Q / 8 KV heads), Qwen3-30B-A3B = 8x (32 Q / 4 KV heads). MHA models = 1x.
This is exact in KV *bytes* (the bandwidth-bound term) and was the same
convention used for the standalone Llama3-8B run.

Median of 3 trials. Writes script/results_decode_all.json with structure
  {platform: {model: {workload: {system: throughput_tok_s}}}}.
TPOT (ms/token) = batch / throughput is computed at plot time.
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
BATCH = 32

LINE_RE = {
    "CLoRA":       re.compile(r"^CLoRA\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M),
    "NoCXL":       re.compile(r"^CLoRA-NoCXL\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M),
    "GraceHopper": re.compile(r"^Grace-Hopper\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M),
    "CPULoRA":     re.compile(r"^CPU-LoRA-Off\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M),
}

PLATFORMS = {
    "A100": {"tflops": "312", "membw": "1935", "total_gb": 40},
    "H100": {"tflops": "989", "membw": "3350", "total_gb": 80},
}

# model -> (extra driver flags, base_gb, gqa_factor, nocxl_ffn_gemms)
# nocxl_ffn_gemms = NoCXL FFN kernel launches per adapter per layer; SwiGLU = 3
# (logical FFN, used for all models including Qwen MoE per author's choice).
MODELS = {
    "Llama2-7B":  (["--model-d", "4096", "--n-layers", "32"], 14, 1, 3),
    "Llama2-13B": (["--model-d", "5120", "--n-layers", "40"], 26, 1, 3),
    "Llama3-8B":  (["--model-d", "4096", "--n-layers", "32"], 16, 4, 3),   # GQA 4x
    "Qwen3-30B":  (["--model-d", "2048", "--n-layers", "48",
                    "--n-matrices", "28", "--moe-active-base-gb", "6"], 60, 8, 3),  # GQA 8x + MoE
}
# NoCXL launch model: "unfused" = per-adapter/per-request; "fused" = BGMV.
NOCXL_MODES = {"unfused": [], "fused": ["--nocxl-fused"]}
# Qwen3 base_gb 60 is the *total* footprint; HBM read uses moe-active 6 GB.
# For the GPU memory budget we treat the resident footprint as the active 6 GB
# (the rest is paged/offloaded, modeled abstractly), matching RESULTS.md.
MODEL_RESIDENT_GB = {"Llama2-7B": 14, "Llama2-13B": 26, "Llama3-8B": 16,
                     "Qwen3-30B": 6}

WORKLOADS = {
    "Uniform":      ("uniform", 100, 1024),
    "Uniform-long": ("uniform", 2048, 4096),
    "Skewed":       ("skewed", 100, 1024),
    "Skewed-long":  ("skewed", 2048, 4096),
}


def run_once(flags):
    p = subprocess.run([sys.executable, DRIVER] + flags, cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, timeout=1800)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:])
        raise RuntimeError("driver failed: " + " ".join(flags))
    return {k: float(r.search(p.stdout).group(1).replace(",", ""))
            for k, r in LINE_RE.items()}


def main():
    t0 = time.time()
    for mode, mode_flags in NOCXL_MODES.items():
        out = {}
        for plat, pcfg in PLATFORMS.items():
            out[plat] = {}
            for mname, (mflags, base_gb, gqa, ffn_gemms) in MODELS.items():
                budget = pcfg["total_gb"] - MODEL_RESIDENT_GB[mname]
                base_flags = [
                    "--steps", "10", "--warmup-steps", "10", "--batch", str(BATCH),
                    "--n-adapters", "1000", "--seed", "42", "--top-k-hot", "50",
                    "--pcie-bw-gb", "128", "--baseline", "all",
                    "--nocxl-ffn-gemms", str(ffn_gemms),
                    "--gpu-compute-tflops", pcfg["tflops"],
                    "--gpu-mem-bw-gb", pcfg["membw"],
                    "--base-model-gb", str(base_gb),
                    "--gpu-mem-gb", str(budget),
                    "--baseline-cache-gb", str(budget), "--gh-cache-gb", str(budget),
                ] + mode_flags + mflags
                out[plat][mname] = {}
                for wname, (dist, kmin, kmax) in WORKLOADS.items():
                    km, kx = kmin // gqa, kmax // gqa
                    wflags = ["--dist", dist, "--kv-min", str(km),
                              "--kv-max", str(kx)]
                    if dist == "skewed":
                        wflags += ["--n-hot", "50"]
                    samples = [run_once(base_flags + wflags) for _ in range(TRIALS)]
                    med = {k: round(statistics.median(s[k] for s in samples), 1)
                           for k in LINE_RE}
                    out[plat][mname][wname] = med
                    tpot = BATCH / med["CLoRA"] * 1000
                    print(f"[{mode:7s} {plat}] {mname:11s} {wname:13s} "
                          f"CLoRA TPOT={tpot:5.2f}ms  "
                          f"(NoCXL={med['NoCXL']:,.0f} GH={med['GraceHopper']:,.0f} "
                          f"CPU={med['CPULoRA']:,.0f})  ({time.time()-t0:4.0f}s)",
                          flush=True)
        fname = f"results_decode_{mode}.json"
        with open(os.path.join(HERE, fname), "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote script/{fname} ({time.time()-t0:.0f}s)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
