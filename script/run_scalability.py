#!/usr/bin/env python3
"""Figure 17 — scalability vs number of CXL devices, on H100, larger models.

Sweeps N_CXL in {1,2,4,8,16,32} for Llama2-13B and Qwen3-30B (GQA+MoE) on
H100, across the 4 synthetic workloads, holding GPU and per-device hardware
fixed (paper Fig 17 setup). Reports CLoRA decode throughput; the "sweet
point" is the device count past which adding devices stops helping.

NOTE on modeling: the C simulator gives each device its own CXL channel
(parallel links), so it captures the *benefit* of more devices (KV cache and
NDP work split across devices) and the diminishing-returns saturation, but
NOT the paper's beyond-knee *degradation* (which comes from the GPU
aggregating partial outputs over a single shared GPU<->switch link). We
report exactly what the simulator shows.

Writes script/results_scalability.json.
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
DEVICE_COUNTS = [1, 2, 4, 8, 16, 32]

TPUT_RE = re.compile(
    r"^CLoRA\s+tokens=\d+\s+sim_time=[\d.,]+ ms\s+throughput=([\d,]+\.?\d*)",
    re.MULTILINE)

# H100 80GB; budget = 80 - resident base. Fig 17 = LARGER models (13B / 30B).
MODELS = {
    "Llama2-13B": dict(flags=["--model-d", "5120", "--n-layers", "40"],
                       base=26, budget=54, gqa=1),
    "Qwen3-30B":  dict(flags=["--model-d", "2048", "--n-layers", "48",
                              "--n-matrices", "28", "--moe-active-base-gb", "6"],
                       base=60, budget=74, gqa=8),   # GQA+MoE; resident 6 GB
}
WORKLOADS = {
    "Uniform":      ("uniform", 100, 1024),
    "Uniform-long": ("uniform", 2048, 4096),
    "Skewed":       ("skewed", 100, 1024),
    "Skewed-long":  ("skewed", 2048, 4096),
}


def make_conf(dest):
    """parameters.conf with cxl_channel_number raised to 64 so up to 32+
    devices each get their own channel."""
    with open(BASE_CONF) as f:
        text = f.read()
    text = re.sub(r"^cxl_channel_number\s*=\s*[^;]*;",
                  "cxl_channel_number = 64;", text, flags=re.MULTILINE)
    path = os.path.join(dest, "scal.conf")
    with open(path, "w") as f:
        f.write(text)
    return path


def run(model_cfg, wl, n_cxl, conf):
    dist, lo, hi = WORKLOADS[wl]
    g = model_cfg["gqa"]
    flags = ["--steps", "10", "--warmup-steps", "10", "--batch", str(BATCH),
             "--n-adapters", "1000", "--seed", "42", "--top-k-hot", "50",
             "--gpu-compute-tflops", "989", "--gpu-mem-bw-gb", "3350",
             "--base-model-gb", str(model_cfg["base"]),
             "--gpu-mem-gb", str(model_cfg["budget"]),
             "--n-cxl", str(n_cxl),
             "--dist", dist, "--kv-min", str(lo // g), "--kv-max", str(hi // g),
             "--c-parameter-file", conf] + model_cfg["flags"]
    if dist == "skewed":
        flags += ["--n-hot", "50"]
    vals = []
    for _ in range(TRIALS):
        p = subprocess.run([sys.executable, DRIVER] + flags, cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=900)
        if p.returncode != 0:
            sys.stderr.write(p.stderr[-1500:]); raise RuntimeError("driver failed")
        vals.append(float(TPUT_RE.search(p.stdout).group(1).replace(",", "")))
    return statistics.median(vals)


def main():
    confdir = tempfile.mkdtemp(prefix="cs_", dir="/tmp")
    conf = make_conf(confdir)
    out = {}
    t0 = time.time()
    for mname, mcfg in MODELS.items():
        out[mname] = {}
        for wl in WORKLOADS:
            out[mname][wl] = {}
            for n in DEVICE_COUNTS:
                t = run(mcfg, wl, n, conf)
                out[mname][wl][str(n)] = round(t, 1)
                print(f"{mname:11s} {wl:13s} N_CXL={n:<3d} "
                      f"{t:>9,.0f} tok/s  ({time.time()-t0:4.0f}s)", flush=True)
    with open(os.path.join(HERE, "results_scalability.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote script/results_scalability.json ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
