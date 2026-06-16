#!/usr/bin/env python3
"""Prefill / TTFT experiment for Fig 12 (rebuttal), H100.

Model (paper §5.2, confirmed GPU-FlashAttention prefill):
  Every system runs the SAME GPU forward pass over the B*L prompt tokens
  (base-model matmuls + FlashAttention), which is compute-bound and is the
  dominant TTFT term:

      GPU_fwd/layer = max( (24 D^2 T + 4 D Σ L_i^2) / C_GPU ,
                            resident_bytes/n_layers / W_GPU )
      GPU_fwd       = GPU_fwd/layer * n_layers          (T = Σ L_i)

  Systems differ only in the LoRA path and where the freshly built KV cache
  is written:

    CLoRA       : LoRA on NDP over CXL + KV written to CXL devices.
                  Measured by the C simulator (kind="prefill"): the GPU
                  forward is the NPU_COMPUTE term, overlapped (Eq 9) with the
                  CXL-side LoRA offload + KV write. TTFT = one prefill step.
    CLoRA-NoCXL : CLoRA + per-layer kernel-relaunch overhead (no CXL.mem).
    Grace-Hopper: LoRA runs on the *same* GPU (serializes after the forward
                  pass) + KV written to host over NVLink-C2C.
    CPU-LoRA    : LoRA runs on the CPU (200 GFLOPS) + activation/KV traffic
                  over PCIe + kernel launches. CPU compute dominates at
                  prefill token counts.

Batch B ∈ {256, 512, 1024}; prompt length L per request drawn from the
workload input range (paper Table 5). Reports TTFT in seconds.
Writes script/results_prefill.json.
"""
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
DRIVER_BIN = os.path.join(ROOT, "main")
TRACE = os.path.join(HERE, "prefill_step.json")

from clora_strategy import (  # noqa: E402
    Adapter, SystemState, ModelConfig, HardwareConfig,
    TemperatureModel, choose_strategy, pre_cache_hot_adapters,
    emit_step_json, write_step_json,
)

RESULT_RE = re.compile(r"CLORA_RESULT duration_ns=(\d+)")

# ---- H100 platform + 4 CLoRA devices ----
HW = HardwareConfig(
    gpu_compute=989e12, gpu_mem_bw=3350e9, gpu_mem_bytes=0,  # budget per-model
    cxl_compute=2e12, cxl_dram_bw=1.1e12, cxl_link_bw=128e9,
    cxl_latency=200, n_cxl=4)
MFU = 0.75           # real prefill model-FLOP utilization (peak FLOPS is never hit)
C_GPU = HW.gpu_compute * MFU   # effective (achievable) GPU FP16 throughput
W_GPU = HW.gpu_mem_bw
C_CPU = 200e9        # CPU FP16 GFLOPS
PCIE = 128e9
C2C = 450e9
L_KERNEL = 5000.0
L_OFFLOAD = 10000.0
NOCXL_PER_LAUNCH = 5000.0 + 1000.0 + 1000.0   # kernel + device cmd + sync = 7us

# model -> (D, n_layers, n_matrices, resident_gb, gqa, total_budget_gb, ffn_gemms)
# ffn_gemms = NoCXL FFN kernel launches per adapter per layer: SwiGLU = 3;
# Qwen3-30B MoE = 3 gemms x 8 active experts = 24 (matches n_matrices=4+24=28).
MODELS = {
    "Llama2-7B":  (4096, 32, 7,  14, 1, 80 - 14, 3),
    "Llama2-13B": (5120, 40, 7,  26, 1, 80 - 26, 3),
    "Llama3-8B":  (4096, 32, 7,  16, 4, 80 - 16, 3),
    "Qwen3-30B":  (2048, 48, 28, 6,  8, 80 - 6, 3),
}
WORKLOADS = {  # input length range (paper Table 5)
    "Uniform":      ("uniform", 100, 1024),
    "Uniform-long": ("uniform", 2048, 4096),
    "Skewed":       ("skewed", 100, 1024),
    "Skewed-long":  ("skewed", 2048, 4096),
}
BATCHES = [256, 512, 1024]
N_ADAPTERS = 1000
RANK_CHOICES = [8, 16, 32, 64, 128]
SEED = 42


def gen_prefill(batch, in_range, dist, hot_ids, rng, ranks, gqa):
    """Return list of (adapter_id, prompt_len) and per-adapter token totals."""
    reqs = []
    for _ in range(batch):
        if dist == "skewed" and rng.random() < 0.8:
            aid = rng.choice(hot_ids)
        else:
            aid = rng.randrange(N_ADAPTERS)
        L = rng.randint(*in_range)
        if aid not in ranks:
            ranks[aid] = rng.choice(RANK_CHOICES)
        reqs.append((aid, L))
    # GQA: KV bytes/token shrink by gqa -> emulate as fewer effective KV tokens
    tokens = {}
    kv_tokens = {}
    for aid, L in reqs:
        tokens[aid] = tokens.get(aid, 0) + L
        kv_tokens[aid] = kv_tokens.get(aid, 0) + L // gqa
    return reqs, tokens, kv_tokens


def gpu_forward_ns(D, n_layers, resident_gb, T, sumsq):
    compute = (24.0 * D * D * T + 4.0 * D * sumsq) / C_GPU * 1e9
    hbm = (resident_gb * 1e9 / n_layers) / W_GPU * 1e9
    return max(compute, hbm) * n_layers, max(compute, hbm)


def run_c_sim(json_path, stamp):
    p = subprocess.run([DRIVER_BIN, "--file", json_path, "--timestamp", stamp],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, timeout=600)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-1500:])
        raise RuntimeError("C sim failed")
    return int(RESULT_RE.search(p.stdout).group(1))


_ctr = 0


def one_point(mname, wname, batch, rng):
    global _ctr
    D, NL, NM, resident, gqa, budget_gb, ffn_gemms = MODELS[mname]
    dist, lo, hi = WORKLOADS[wname]
    model = ModelConfig(d=D)
    hw = replace(HW, gpu_mem_bytes=int(budget_gb * 1e9))
    ranks = {}
    hot_ids = rng.sample(range(N_ADAPTERS), 50) if dist == "skewed" else []
    reqs, tokens, kv_tokens = gen_prefill(batch, (lo, hi), dist, hot_ids,
                                          rng, ranks, gqa)
    T = sum(L for _, L in reqs)
    sumsq = sum(L * L for _, L in reqs)

    # shared GPU forward (base + FlashAttention), compute-bound
    fwd_ns, fwd_per_layer = gpu_forward_ns(D, NL, resident, T, sumsq)

    # build state: adapter.batch = token count so LoRA scales with tokens
    temp = TemperatureModel()
    for aid, _ in reqs:
        temp.on_request(aid)
    state = SystemState(adapters={})
    for aid in tokens:
        state.adapters[aid] = Adapter(adapter_id=aid, rank=ranks[aid],
                                      batch=tokens[aid], temperature=temp.get(aid))
    pre_cache_hot_adapters(state, hw, model, top_k=50,
                           n_layers=NL, n_matrices=NM)
    decisions = {aid: choose_strategy(aid, state, hw, model,
                                      n_layers=NL, n_matrices=NM, top_k_hot=50)
                 for aid in tokens}

    # ---- CLoRA: C-sim prefill step ----
    payload = emit_step_json("prefill", model, hw, decisions, state,
                             n_layers=NL, n_matrices=NM,
                             input_tokens_per_request=1,
                             base_model_compute_ns=fwd_per_layer,
                             kv_tokens_per_adapter=kv_tokens,
                             kv_in_gpu_fraction=0.0)
    write_step_json(TRACE, payload)
    _ctr += 1
    clora_ns = run_c_sim(TRACE, f"pf_{os.getpid()%100000}_{_ctr:06d}")

    # ---- shared LoRA FLOP time ----
    lora_flops = sum(4 * tokens[a] * D * ranks[a] for a in tokens) * NL * NM
    lora_gpu_ns = lora_flops / C_GPU * 1e9
    lora_cpu_ns = lora_flops / C_CPU * 1e9
    kv_bytes = 2 * D * model.s_dtype * T * NL          # full KV built
    act_bytes = 2 * D * model.s_dtype * T * NL         # x down + y back

    # ---- NoCXL: CLoRA + per-launch kernel overhead, two fusion models. ----
    n_adapters = len(tokens)
    per_adapter = 1 + 1 + ffn_gemms                       # QKV + O + FFN
    lpl_unfused = n_adapters * per_adapter + batch * 1     # per adapter / req
    lpl_fused = per_adapter + 1                            # BGMV: per op + attn
    nocxl_unfused_ns = clora_ns + lpl_unfused * NL * NOCXL_PER_LAUNCH
    nocxl_fused_ns = clora_ns + lpl_fused * NL * NOCXL_PER_LAUNCH

    # ---- Grace-Hopper: LoRA on GPU (serial after fwd) + KV to host over C2C ----
    gh_ns = fwd_ns + lora_gpu_ns + kv_bytes / C2C * 1e9 + NL * L_KERNEL

    # ---- CPU-LoRA: LoRA on CPU + PCIe activations/KV + kernels ----
    cpu_path = lora_cpu_ns + (act_bytes + kv_bytes) / PCIE * 1e9 \
        + 2 * NL * NM * (L_KERNEL + L_OFFLOAD)
    cpu_ns = max(fwd_ns, cpu_path)   # CPU LoRA overlaps GPU forward

    return {"CLoRA": clora_ns / 1e9,
            "NoCXL_unfused": nocxl_unfused_ns / 1e9,
            "NoCXL_fused": nocxl_fused_ns / 1e9,
            "GraceHopper": gh_ns / 1e9, "CPULoRA": cpu_ns / 1e9,
            "gpu_fwd_s": fwd_ns / 1e9, "tokens": T}


def main():
    t0 = time.time()
    out = {}
    for mname in MODELS:
        out[mname] = {}
        for wname in WORKLOADS:
            out[mname][wname] = {}
            for B in BATCHES:
                samples = []
                for tr in range(3):
                    rng = random.Random(SEED + tr)
                    samples.append(one_point(mname, wname, B, rng))
                med = {k: round(statistics.median(s[k] for s in samples), 3)
                       for k in ("CLoRA", "NoCXL_unfused", "NoCXL_fused",
                                 "GraceHopper", "CPULoRA", "gpu_fwd_s")}
                med["tokens"] = int(statistics.median(s["tokens"] for s in samples))
                out[mname][wname][str(B)] = med
                print(f"{mname:11s} {wname:13s} B={B:<5d} "
                      f"CLoRA={med['CLoRA']:6.2f}s  "
                      f"NoCXL(u={med['NoCXL_unfused']:6.2f} f={med['NoCXL_fused']:6.2f})s  "
                      f"GH={med['GraceHopper']:6.2f}s  CPU={med['CPULoRA']:7.2f}s  "
                      f"(fwd={med['gpu_fwd_s']:.2f}s, {med['tokens']:,}tok)"
                      f"  ({time.time()-t0:4.0f}s)", flush=True)
    with open(os.path.join(HERE, "results_prefill.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote script/results_prefill.json ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
