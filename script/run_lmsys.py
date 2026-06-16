#!/usr/bin/env python3
"""LMSYS (Chatbot Arena) decode benchmark — A100 replication + H100.

The LMSYS workload is the real Chatbot Arena trace that CLoRA / S-LoRA
downsample (NOT a synthetic Uniform/Skewed config):
  * 25 adapters ("treating each model as a LoRA variant"); popularity is the
    real Arena distribution — Vicuna-13B ~49% + a steep power-law tail
    (LMSYS-Chat-1M Table 4), built by clora_driver.build_lmsys_weights().
  * Right-skewed, truncated request lengths (Llama2 tokenizer means:
    input 69.5, output 214.5; CLoRA Table 5 range [2, 430]). We sample a
    representative decode context length ~ lognormal(mean=177) on [2, 430].
  * Decode batch is EMERGENT, not fixed: per-adapter B_ij in 0-256
    (CLoRA §5.1, Fig 1). The step-batch is the sum of per-adapter batches,
    shaped by the Vicuna skew, capped at 256/adapter.

Calibration target = OUR CLoRA + A100 numbers: 7B ~7.9k tok/s, 13B ~4.8k.
These are above the batch-32 HBM floor (4423 / 2382), so they correspond to a
larger emergent batch; PHASE 1 sweeps the step-batch on A100 to pin it.
PHASE 2 runs the full system comparison at the operating batch on A100 + H100.

Writes script/results_lmsys.json.
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
    "CLoRA":       re.compile(r"^CLoRA\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M),
    "NoCXL":       re.compile(r"^CLoRA-NoCXL\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M),
    "GraceHopper": re.compile(r"^Grace-Hopper\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M),
    "CPULoRA":     re.compile(r"^CPU-LoRA-Off\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)", re.M),
}

PLATFORMS = {
    "A100": {"tflops": "312", "membw": "1935", "total_gb": 40},
    "H100": {"tflops": "989", "membw": "3350", "total_gb": 80},
}

# model -> (driver flags, base_gb, gqa, ffn_gemms, resident_gb)
MODELS = {
    "Llama2-7B":  (["--model-d", "4096", "--n-layers", "32"], 14, 1, 3, 14),
    "Llama2-13B": (["--model-d", "5120", "--n-layers", "40"], 26, 1, 3, 26),
    "Llama3-8B":  (["--model-d", "4096", "--n-layers", "32"], 16, 4, 3, 16),
    "Qwen3-30B":  (["--model-d", "2048", "--n-layers", "48",
                    "--n-matrices", "28", "--moe-active-base-gb", "6"], 60, 8, 3, 6),
}

# LMSYS workload knobs (real Chatbot Arena trace)
N_ADAPTERS = 25
KV_MIN, KV_MAX = 2, 430          # CLoRA Table 5
KV_MEAN, KV_SIGMA = 177, 0.8     # input 69.5 + ~half of output 214.5, right-skewed
SEED = 42

# Phase-1 calibration: emergent step-batch grid (A100, CLoRA only).
BATCH_GRID = [32, 48, 57, 64, 80, 96, 128]
TARGET = {"Llama2-7B": 7900, "Llama2-13B": 4800}   # our CLoRA + A100 targets
OPERATING_BATCH = 64             # phase-2 operating point (see writeup)


def run_once(flags):
    p = subprocess.run([sys.executable, DRIVER] + flags, cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, timeout=1800)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:])
        raise RuntimeError("driver failed: " + " ".join(flags))
    return {k: float(r.search(p.stdout).group(1).replace(",", ""))
            for k, r in LINE_RE.items() if r.search(p.stdout)}


def lmsys_workload_flags(gqa):
    km, kx = max(1, KV_MIN // gqa), max(1, KV_MAX // gqa)
    return ["--dist", "lmsys", "--n-adapters", str(N_ADAPTERS),
            "--top-k-hot", str(N_ADAPTERS),
            "--kv-dist", "lognormal", "--kv-mean", str(KV_MEAN // gqa),
            "--kv-sigma", str(KV_SIGMA), "--kv-min", str(km), "--kv-max", str(kx)]


def base_flags(plat, mname, batch):
    pcfg = PLATFORMS[plat]
    mflags, base_gb, gqa, ffn, resident = MODELS[mname]
    budget = pcfg["total_gb"] - resident
    return [
        "--steps", "10", "--warmup-steps", "10", "--batch", str(batch),
        "--seed", str(SEED), "--baseline", "all",
        "--nocxl-ffn-gemms", str(ffn), "--pcie-bw-gb", "128",
        "--gpu-compute-tflops", pcfg["tflops"], "--gpu-mem-bw-gb", pcfg["membw"],
        "--base-model-gb", str(base_gb), "--gpu-mem-gb", str(budget),
        "--baseline-cache-gb", str(budget), "--gh-cache-gb", str(budget),
    ] + mflags


def median_run(plat, mname, batch):
    _, _, gqa, _, _ = MODELS[mname]
    flags = base_flags(plat, mname, batch) + lmsys_workload_flags(gqa)
    samples = [run_once(flags) for _ in range(TRIALS)]
    keys = set().union(*samples)
    return {k: round(statistics.median(s[k] for s in samples if k in s), 1)
            for k in keys}


def phase1_calibration(t0):
    print("=== PHASE 1: A100 emergent-batch calibration (CLoRA) ===")
    print(f"{'batch':>6} | " + " | ".join(f"{m:>11}" for m in TARGET))
    table = {}
    for b in BATCH_GRID:
        row = {}
        for m in TARGET:
            row[m] = median_run("A100", m, b)["CLoRA"]
        table[b] = row
        print(f"{b:>6} | " + " | ".join(f"{row[m]:>11,.0f}" for m in TARGET)
              + f"   ({time.time()-t0:4.0f}s)", flush=True)
    print(f"targets: " + ", ".join(f"{m}={TARGET[m]:,}" for m in TARGET))
    # report the batch best reproducing each target
    for m in TARGET:
        best = min(BATCH_GRID, key=lambda b: abs(table[b][m] - TARGET[m]))
        print(f"  {m}: closest to {TARGET[m]:,} at batch {best} "
              f"({table[best][m]:,.0f} tok/s)")
    return table


def phase2_full(t0):
    print(f"\n=== PHASE 2: full system comparison @ emergent batch "
          f"{OPERATING_BATCH} (A100 + H100) ===")
    out = {"_config": {
        "workload": "LMSYS (Chatbot Arena real trace)",
        "n_adapters": N_ADAPTERS, "popularity": "Vicuna ~49% + power-law tail",
        "kv_dist": f"lognormal mean={KV_MEAN} sigma={KV_SIGMA} trunc[{KV_MIN},{KV_MAX}]",
        "emergent_step_batch": OPERATING_BATCH,
        "note": "decode batch emergent (per-adapter B_ij<=256, CLoRA 5.1); "
                "operating batch chosen to reproduce CLoRA+A100 targets "
                "(7B~7.9k, 13B~4.8k).",
        "calibration_targets_A100": TARGET}}
    for plat in PLATFORMS:
        out[plat] = {}
        for m in MODELS:
            med = median_run(plat, m, OPERATING_BATCH)
            out[plat][m] = med
            tpot = OPERATING_BATCH / med["CLoRA"] * 1000
            print(f"[{plat}] {m:11s} CLoRA={med['CLoRA']:>8,.0f} tok/s "
                  f"(TPOT={tpot:4.1f}ms)  NoCXL={med.get('NoCXL',0):>7,.0f} "
                  f"GH={med.get('GraceHopper',0):>7,.0f} "
                  f"CPU={med.get('CPULoRA',0):>6,.0f}  ({time.time()-t0:4.0f}s)",
                  flush=True)
    with open(os.path.join(HERE, "results_lmsys.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote script/results_lmsys.json")


def main():
    t0 = time.time()
    phase1_calibration(t0)
    phase2_full(t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
