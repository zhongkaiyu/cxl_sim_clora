#!/usr/bin/env python3
"""
clora_driver.py
===============

End-to-end smoke driver for the CLoRA simulator.

Pipeline per decode step:
    1. Generate a Uniform-style batch (paper Table 5).
    2. Aggregate per adapter, run choose_strategy.
    3. emit_step_json -> write to disk.
    4. Invoke the C binary on that JSON, parse the CLORA_RESULT line.
    5. Sum step times; throughput = tokens / total_time.

Usage:
    python3 script/clora_driver.py                     # default smoke config
    python3 script/clora_driver.py --steps 5 --n-adapters 20
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil as _shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from clora_strategy import (  # noqa: E402
    Adapter, Decision, HardwareConfig, ModelConfig, Strategy, SystemState,
    TemperatureModel, choose_strategy, compute_kv_in_gpu_fraction,
    emit_step_json, estimate_base_model_ns_per_layer,
    pre_cache_hot_adapters, write_step_json,
)
from baseline_cpu_offload import (  # noqa: E402
    BaselineHWConfig, LRUAdapterCache, baseline_step_ns,
    DEFAULT_HW as BASELINE_DEFAULT_HW,
)
from baseline_no_cxl import (  # noqa: E402
    NoCxlHWConfig, no_cxl_step_ns,
    DEFAULT_HW as NO_CXL_DEFAULT_HW,
)
from baseline_grace_hopper import (  # noqa: E402
    GraceHopperHWConfig, grace_hopper_step_ns,
    DEFAULT_HW as GH_DEFAULT_HW,
)

ROOT = os.path.dirname(HERE)
BINARY = os.path.join(ROOT, "main")
TRACE = os.path.join(ROOT, "script", "clora_step.json")
RESULT_RE = re.compile(
    r"CLORA_RESULT duration_ns=(\d+) start_ns=(\d+) end_ns=(\d+)")


# ---- hardware / model fixture (paper Table 4, A100 + 4 CLoRA devices) -----

DEFAULT_HW = HardwareConfig(
    gpu_compute=989e12,         # H100 SXM FP16 dense
    gpu_mem_bw=3350e9,          # H100 SXM HBM3
    gpu_mem_bytes=8 * 10**9,    # for LoRA + KV after base-model weights
    cxl_compute=2e12,           # paper: 2 TFLOPS / CLoRA device
    cxl_dram_bw=1.1e12,         # paper Table 4
    cxl_link_bw=128e9,          # CXL 3.1 ~ 128 GB/s effective
    cxl_latency=200,
    n_cxl=4,
)
DEFAULT_MODEL = ModelConfig(d=4096)


# ---- workload --------------------------------------------------------------

RANK_CHOICES = [8, 16, 32, 64, 128]  # paper §7.1


def _sample_adapter(n_adapters: int, dist: str,
                    hot_ids: List[int], rng: random.Random) -> int:
    """Pick one adapter id per request.

    ``dist`` is "uniform" or "skewed". For skewed, 80% of requests are routed
    to one of ``hot_ids`` (paper §7.1: 50 hot adapters); the rest are uniform
    over the remaining adapters.
    """
    if dist == "uniform":
        return rng.randrange(n_adapters)
    # skewed
    if hot_ids and rng.random() < 0.8:
        return rng.choice(hot_ids)
    return rng.randrange(n_adapters)


def gen_step(n_adapters: int, batch_size: int,
             kv_range: Tuple[int, int],
             dist: str,
             hot_ids: List[int],
             rng: random.Random,
             ranks: Dict[int, int]) -> List[Tuple[int, int]]:
    """Return [(adapter_id, kv_tokens), ...] for one decode step."""
    out = []
    for _ in range(batch_size):
        aid = _sample_adapter(n_adapters, dist, hot_ids, rng)
        kv = rng.randint(kv_range[0], kv_range[1])
        out.append((aid, kv))
        if aid not in ranks:
            ranks[aid] = rng.choice(RANK_CHOICES)
    return out


def make_state(reqs: List[Tuple[int, int]],
               ranks: Dict[int, int],
               temp_model: TemperatureModel,
               ) -> Tuple[SystemState, Dict[int, int]]:
    """Aggregate into per-adapter (batch, total_kv_tokens) and build SystemState.

    Adapters from previous steps that have no requests this step are still
    added to the SystemState (batch=0) so the hot/cold classifier sees them.
    Their temperatures come from the persistent ``temp_model``.
    """
    per_adapter_batch: Dict[int, int] = defaultdict(int)
    per_adapter_kv:    Dict[int, int] = defaultdict(int)
    for aid, kv in reqs:
        per_adapter_batch[aid] += 1
        per_adapter_kv[aid]    += kv

    state = SystemState()
    # serving adapters (have requests this step)
    for aid, batch in per_adapter_batch.items():
        state.adapters[aid] = Adapter(
            adapter_id=aid, rank=ranks[aid], batch=batch,
            temperature=temp_model.get(aid))
    # dormant adapters that the temperature model knows about
    for aid in temp_model.temps:
        if aid not in state.adapters and aid in ranks:
            state.adapters[aid] = Adapter(
                adapter_id=aid, rank=ranks[aid], batch=0,
                temperature=temp_model.get(aid))
    return state, dict(per_adapter_kv)


# ---- C simulator invocation ------------------------------------------------

_run_counter = 0

# Optional override for the C sim's parameter file (config/parameters.conf by
# default).  Set via --c-parameter-file; the sensitivity study uses this to
# hand each sweep point its own conf.
C_PARAMETER_FILE: str | None = None


def run_c_sim(json_path: str) -> int:
    global _run_counter
    if not os.path.isfile(BINARY):
        raise RuntimeError(f"binary not built: {BINARY} -- run 'make all' first")
    # The C sim mkdir's raw/<timestamp>/ for logs and aborts on collision.
    # The timestamp buffer is char[16] (15 chars max), so the tag MUST stay
    # short: pid(5) + counter(6) + '_' = 12 chars. To avoid collisions across
    # re-runs that reuse a pid, we delete the per-run raw/<stamp> dir after
    # parsing (see below) so raw/ never accumulates.
    _run_counter += 1
    stamp = f"{os.getpid() % 100000:05d}_{_run_counter:06d}"
    cmd = [BINARY, "--file", json_path, "--timestamp", stamp]
    if C_PARAMETER_FILE:
        cmd += ["--parameter", C_PARAMETER_FILE]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=300)
    # Remove this run's raw log dir so raw/ never accumulates (keeps the short
    # timestamp tag collision-free across many re-runs).
    _shutil.rmtree(os.path.join(ROOT, "raw", stamp), ignore_errors=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"C sim returned {proc.returncode}")
    m = RESULT_RE.search(proc.stdout)
    if not m:
        sys.stderr.write(proc.stdout[-2000:])
        raise RuntimeError("CLORA_RESULT line not found in C-sim output")
    return int(m.group(1))


# ---- main loop -------------------------------------------------------------

def run_smoke(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    ranks: Dict[int, int] = {}

    hw = replace(DEFAULT_HW,
                 gpu_compute=args.gpu_compute_tflops * 1e12,
                 gpu_mem_bw=args.gpu_mem_bw_gb * 1e9,
                 gpu_mem_bytes=int(args.gpu_mem_gb * 10**9),
                 cxl_compute=args.ndp_tflops * 1e12,
                 cxl_dram_bw=args.cxl_dram_bw_gb * 1e9,
                 cxl_link_bw=args.cxl_link_bw_gb * 1e9,
                 cxl_latency=args.cxl_latency_ns,
                 n_cxl=args.n_cxl)
    model = ModelConfig(d=args.model_d)

    # ---- baselines (independent toggles) ----
    cpu_off_on = args.baseline in ("cpu_offload", "all")
    no_cxl_on  = args.baseline in ("no_cxl", "all")
    gh_on      = args.baseline in ("grace_hopper", "all")

    baseline_hw = replace(BASELINE_DEFAULT_HW,
                          gpu_compute=args.gpu_compute_tflops * 1e12,
                          gpu_mem_bw=args.gpu_mem_bw_gb * 1e9,
                          gpu_mem_bytes=int(args.gpu_mem_gb * 10**9),
                          pcie_bw=args.pcie_bw_gb * 10**9,
                          cpu_compute=args.cpu_compute_gflops * 10**9,
                          cpu_dram_bw=args.cpu_dram_bw_gb * 10**9,
                          L_pcie_setup=args.L_pcie_setup_ns,
                          L_kernel_launch=args.L_kernel_launch_ns,
                          L_cpu_gpu_offload=args.L_cpu_gpu_offload_ns)
    baseline_cache = LRUAdapterCache(
        capacity_bytes=int(args.baseline_cache_gb * 10**9),
        model=model, n_layers=args.n_layers, n_matrices=args.n_matrices,
    ) if cpu_off_on else None
    baseline_total_ns = 0

    no_cxl_hw = NoCxlHWConfig(
        L_kernel_launch_ns=args.nocxl_L_kernel_launch_ns,
        L_device_command_ns=args.nocxl_L_device_command_ns,
        L_sync_ns=args.nocxl_L_sync_ns,
        ffn_gemms_per_adapter=args.nocxl_ffn_gemms,
        fused=args.nocxl_fused,
    )
    no_cxl_total_ns = 0

    gh_hw = GraceHopperHWConfig(
        gpu_compute=args.gpu_compute_tflops * 1e12,
        gpu_mem_bw=args.gpu_mem_bw_gb * 1e9,
        gpu_mem_bytes=int(args.gh_cache_gb * 10**9),
        c2c_bw=args.gh_c2c_bw_gb * 1e9,
        L_kernel_launch_ns=args.gh_L_kernel_launch_ns,
        L_c2c_setup_ns=args.gh_L_c2c_setup_ns,
    )
    gh_cache = LRUAdapterCache(
        capacity_bytes=int(args.gh_cache_gb * 10**9),
        model=model, n_layers=args.n_layers, n_matrices=args.n_matrices,
    ) if gh_on else None
    gh_total_ns = 0

    # legacy name for backwards-compat in the printing section below
    baseline_on = cpu_off_on

    n_hot = min(args.n_hot, args.n_adapters)
    hot_ids: List[int] = rng.sample(range(args.n_adapters), n_hot) \
        if args.dist == "skewed" else []
    total_tokens = 0
    total_ns = 0
    per_step_log: List[dict] = []

    temp_model = TemperatureModel(T_1=args.temp_increment,
                                  alpha=args.temp_decay)
    warmup_steps = max(0, args.warmup_steps)
    measure_steps = args.steps
    total_steps = warmup_steps + measure_steps

    print(f"smoke config: warmup={warmup_steps}, measure={measure_steps}, "
          f"batch={args.batch}, "
          f"n_adapters={args.n_adapters}, dist={args.dist}"
          + (f" (hot={n_hot})" if args.dist == "skewed" else "")
          + f", kv={args.kv_min}-{args.kv_max}, "
          + f"D={args.model_d}, layers={args.n_layers}, "
          + f"base={args.base_model_gb}GB, "
          + f"temp(T1={args.temp_increment}, alpha={args.temp_decay})")
    print("-" * 78)

    wall_start = time.time()
    for step in range(total_steps):
        is_warmup = step < warmup_steps
        reqs = gen_step(args.n_adapters, args.batch,
                        (args.kv_min, args.kv_max),
                        args.dist, hot_ids, rng, ranks)
        # apply temperature bump on each arriving request
        for aid, _ in reqs:
            temp_model.on_request(aid)
        state, kv_per_adapter = make_state(reqs, ranks, temp_model)

        # ---- Paper §6.2: pre-cache top-K hottest non-serving adapters as E1
        # BEFORE strategy selection. Mutates state.adapters[a].a_in_gpu/b_in_gpu.
        pre_cached_ids = pre_cache_hot_adapters(
            state, hw, model,
            top_k=args.top_k_hot,
            n_layers=args.n_layers, n_matrices=args.n_matrices)

        # only call strategy selection for adapters that have requests this step;
        # dormant adapters (batch=0) still live in state.adapters for the hot/cold
        # classifier but don't need a strategy choice
        serving_ids = [aid for aid, a in state.adapters.items() if a.batch > 0]
        decisions: Dict[int, Decision] = {
            aid: choose_strategy(aid, state, hw, model,
                                 n_layers=args.n_layers, n_matrices=args.n_matrices,
                                 top_k_hot=args.top_k_hot)
            for aid in serving_ids
        }
        strat_counts = defaultdict(int)
        for d in decisions.values():
            strat_counts[int(d.strategy)] += 1

        batch_total = sum(a.batch for a in state.adapters.values())
        moe_kwargs = {}
        if args.moe_active_base_gb > 0:
            moe_kwargs["moe_active_base_bytes"] = int(
                args.moe_active_base_gb * 1e9)

        # KV duplication (paper Eqs 7-9): pick the fraction that balances the
        # GPU-side attention cost against the CXL-side cost, capped by the GPU
        # memory left over after the LoRA decisions.
        total_kv_tokens = sum(kv_per_adapter.values())
        kv_in_gpu_fraction = compute_kv_in_gpu_fraction(
            decisions, state, hw, model, kv_per_adapter,
            n_layers=args.n_layers, n_matrices=args.n_matrices,
            total_batch=batch_total,
            base_model_bytes=int(args.base_model_gb * 10**9),
            **moe_kwargs,
        )

        # GPU busy time per layer = base model + its share of attention
        # (the duplicated-KV portion). The C sim runs this as one NPU_COMPUTE
        # in parallel with the CXL-side attention/LoRA requests (Eq 9).
        base_ns_per_layer = estimate_base_model_ns_per_layer(
            batch_total, model, hw,
            base_model_bytes=int(args.base_model_gb * 10**9),
            n_layers=args.n_layers,
            kv_tokens_in_gpu=int(kv_in_gpu_fraction * total_kv_tokens),
            **moe_kwargs)

        payload = emit_step_json(
            "decode", model, hw, decisions, state,
            n_layers=args.n_layers, n_matrices=args.n_matrices,
            input_tokens_per_request=1,
            base_model_compute_ns=base_ns_per_layer,
            kv_tokens_per_adapter=kv_per_adapter,
            kv_in_gpu_fraction=kv_in_gpu_fraction,
        )
        write_step_json(TRACE, payload)

        step_ns = run_c_sim(TRACE)

        # ---- CPU-LoRA-compute baseline (pure Python, no C sim) ----
        baseline_step_info = None
        if cpu_off_on:
            # only feed serving adapters to the LRU cache; dormant ones (from
            # the temperature model) would needlessly evict warm entries.
            adapter_specs = {a.adapter_id: (a.rank, a.batch)
                             for a in state.adapters.values() if a.batch > 0}
            baseline_step_info = baseline_step_ns(
                adapter_specs, kv_per_adapter,
                baseline_hw, model, baseline_cache,
                base_model_bytes=int(args.base_model_gb * 10**9),
                n_layers=args.n_layers, n_matrices=args.n_matrices,
                fusion=args.baseline_fusion,
                parallel=not args.baseline_serial,
            )

        # ---- NoCxl baseline: CLoRA step time + per-launch offload overhead.
        # Launches scale with serving adapters (QKV/O/FFN LoRA) and requests
        # (attention), since CXL.mem's fine-grained access can't be fused away.
        no_cxl_step = None
        if no_cxl_on:
            no_cxl_step = no_cxl_step_ns(
                step_ns, no_cxl_hw,
                n_layers=args.n_layers,
                n_serving_adapters=len(serving_ids),
                n_requests=args.batch)

        # ---- Grace-Hopper baseline: GPU computes, C2C link to CPU memory ----
        gh_step_info = None
        if gh_on:
            adapter_specs = {a.adapter_id: (a.rank, a.batch)
                             for a in state.adapters.values() if a.batch > 0}
            gh_step_info = grace_hopper_step_ns(
                adapter_specs, kv_per_adapter,
                gh_hw, model, gh_cache,
                base_model_bytes=int(args.base_model_gb * 10**9),
                n_layers=args.n_layers, n_matrices=args.n_matrices,
                fusion=args.baseline_fusion,
                parallel=not args.baseline_serial,
            )
        # accumulation gated on warmup phase below

        # decay temperatures for next step
        temp_model.step()

        if not is_warmup:
            total_tokens += batch_total
            total_ns += step_ns
            if cpu_off_on and baseline_step_info is not None:
                baseline_total_ns += int(baseline_step_info["total_ns"])
            if no_cxl_on and no_cxl_step is not None:
                no_cxl_total_ns += no_cxl_step
            if gh_on and gh_step_info is not None:
                gh_total_ns += int(gh_step_info["total_ns"])

        per_step_log.append({
            "step": step,
            "phase": "warmup" if is_warmup else "measure",
            "adapters_in_batch": sum(1 for a in state.adapters.values() if a.batch > 0),
            "tokens": batch_total,
            "kv_in_gpu_frac": round(kv_in_gpu_fraction, 4),
            "base_ns_per_layer": round(base_ns_per_layer, 1),
            "step_ns": step_ns,
            "strategy_counts": dict(strat_counts),
            "baseline_step_ns": (
                int(baseline_step_info["total_ns"]) if baseline_step_info else None),
        })
        tag = "warm" if is_warmup else " run"
        active = sum(1 for a in state.adapters.values() if a.batch > 0)
        n_hot_cached = len(pre_cached_ids)
        line = (f"{tag} {step:>2}: tok={batch_total:>4}  "
                f"serving={active:>3} hot_cached={n_hot_cached:>3}  "
                f"strat={dict(strat_counts)}  "
                f"kv_in_gpu={kv_in_gpu_fraction:5.1%}  "
                f"clora_ns={step_ns:>10,d}")
        if no_cxl_on and no_cxl_step is not None:
            line += f"  no_cxl_ns={no_cxl_step:>10,d}"
        if gh_on and gh_step_info is not None:
            line += f"  gh_ns={int(gh_step_info['total_ns']):>11,d}"
        if cpu_off_on and baseline_step_info is not None:
            line += f"  cpu_off_ns={int(baseline_step_info['total_ns']):>12,d}"
        print(line)
    wall_end = time.time()

    print("-" * 78)
    throughput = 0.0
    if total_ns > 0:
        throughput = total_tokens / (total_ns / 1e9)
        print(f"CLoRA           tokens={total_tokens}  "
              f"sim_time={total_ns/1e6:.3f} ms  "
              f"throughput={throughput:,.1f} tokens/s")
    if no_cxl_on and no_cxl_total_ns > 0:
        n_throughput = total_tokens / (no_cxl_total_ns / 1e9)
        speedup = throughput / n_throughput if n_throughput > 0 else 0
        print(f"CLoRA-NoCXL     tokens={total_tokens}  "
              f"sim_time={no_cxl_total_ns/1e6:.3f} ms  "
              f"throughput={n_throughput:,.1f} tokens/s  "
              f"(launches/adapter/layer={2 + no_cxl_hw.ffn_gemms_per_adapter}, "
              f"+1/req attn)")
        print(f"  SPEEDUP    CLoRA / CLoRA-NoCXL = {speedup:.2f}x")
    if gh_on and gh_total_ns > 0:
        gh_throughput = total_tokens / (gh_total_ns / 1e9)
        speedup = throughput / gh_throughput if gh_throughput > 0 else 0
        snap = gh_cache.snapshot()
        hit_rate = snap["n_hits"] / max(1, snap["n_hits"] + snap["n_misses"])
        print(f"Grace-Hopper    tokens={total_tokens}  "
              f"sim_time={gh_total_ns/1e6:.3f} ms  "
              f"throughput={gh_throughput:,.1f} tokens/s  "
              f"(c2c={args.gh_c2c_bw_gb:.0f} GB/s, "
              f"cache_hit_rate={hit_rate:.1%})")
        print(f"  SPEEDUP    CLoRA / Grace-Hopper = {speedup:.2f}x")
    if cpu_off_on and baseline_total_ns > 0:
        b_throughput = total_tokens / (baseline_total_ns / 1e9)
        speedup = throughput / b_throughput if b_throughput > 0 else 0
        snap = baseline_cache.snapshot()
        hit_rate = snap["n_hits"] / max(1, snap["n_hits"] + snap["n_misses"])
        print(f"CPU-LoRA-Off    tokens={total_tokens}  "
              f"sim_time={baseline_total_ns/1e6:.3f} ms  "
              f"throughput={b_throughput:,.1f} tokens/s  "
              f"(fusion={args.baseline_fusion}, "
              f"cache_hit_rate={hit_rate:.1%})")
        print(f"  SPEEDUP    CLoRA / CPU-LoRA-Off = {speedup:.2f}x")
    print(f"(wall clock for driver: {wall_end - wall_start:.2f}s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16,
                    help="number of requests per decode step")
    ap.add_argument("--n-adapters", type=int, default=20,
                    help="size of the adapter pool")
    ap.add_argument("--n-layers", type=int, default=32)
    ap.add_argument("--n-matrices", type=int, default=7,
                    help="LoRA-touched weight matrices per decoder layer. "
                         "Llama2 default: 7 (Q,K,V,O + FFN gate/up/down). "
                         "MoE attention-only: 4 (Q,K,V,O). "
                         "MoE attention + active experts: 4 + 3*top_k_experts "
                         "(e.g., Qwen3-30B-A3B: 4 + 3*8 = 28).")
    ap.add_argument("--kv-min", type=int, default=64)
    ap.add_argument("--kv-max", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="Steps run but excluded from throughput accounting; "
                         "lets the temperature model and baseline LRU cache "
                         "reach steady state before measurement.")
    ap.add_argument("--temp-increment", type=float, default=1.0,
                    help="Temperature bump per arriving request (paper T_1)")
    ap.add_argument("--temp-decay", type=float, default=0.1,
                    help="Decay rate per step toward system mean (paper alpha)")
    ap.add_argument("--top-k-hot", type=int, default=50,
                    help="Paper §6.2: size of the hot-adapter pool to "
                         "proactively pre-cache as E1 (top-K by temperature). "
                         "Default 50 matches the paper's Skewed configuration.")
    ap.add_argument("--gpu-mem-gb", type=float, default=8.0,
                    help="GPU memory budget for LoRA + KV (after base weights)")
    ap.add_argument("--gpu-compute-tflops", type=float, default=989.0,
                    help="GPU FP16 dense peak TFLOPS. A100=312, H100=989 (default)")
    ap.add_argument("--gpu-mem-bw-gb", type=float, default=3350.0,
                    help="HBM bandwidth in GB/s. A100=1935, H100=3350 (default)")
    ap.add_argument("--base-model-gb", type=float, default=14.0,
                    help="Base model size in GB (Llama2-7B FP16 = 14, "
                         "13B = 26, Llama3-8B = 16)")
    ap.add_argument("--moe-active-base-gb", type=float, default=0.0,
                    help="MoE active-parameter footprint in GB across all "
                         "layers. When > 0, only this many bytes are read "
                         "from HBM per step (the active expert slice) "
                         "instead of --base-model-gb. Default 0 = disabled "
                         "(dense model uses --base-model-gb).")
    ap.add_argument("--model-d", type=int, default=4096,
                    help="model dim D (Llama2-7B=4096, 13B=5120, Llama3-8B=4096)")
    ap.add_argument("--dist", choices=["uniform", "skewed"], default="uniform")
    ap.add_argument("--n-hot", type=int, default=50,
                    help="Skewed: number of hot adapters that receive 80% of requests")
    ap.add_argument("--baseline",
                    choices=["none", "cpu_offload", "no_cxl",
                             "grace_hopper", "all"],
                    default="none",
                    help="Run baselines alongside CLoRA. "
                         "'cpu_offload' = CPU runs LoRA matmul (weak baseline). "
                         "'no_cxl' = NDP keeps LoRA matmul but PCIe + explicit "
                         "DMA + kernel relaunches per boundary. "
                         "'grace_hopper' = GPU runs LoRA matmul, adapters in "
                         "Grace CPU memory accessed via NVLink-C2C (450 GB/s). "
                         "'all' = all three.")
    ap.add_argument("--baseline-cache-gb", type=float, default=8.0,
                    help="LoRA adapter LRU cache size for the baseline (GB)")
    ap.add_argument("--baseline-fusion", choices=["fused", "unfused"],
                    default="fused",
                    help="Baseline: fused BGMV kernels (S-LoRA/PUNICA style) "
                         "or one kernel per adapter (strawman)")
    ap.add_argument("--baseline-serial", action="store_true",
                    help="Baseline: serialize GPU base + CPU LoRA path per "
                         "layer (worst case). Default is CUDA-stream-style "
                         "overlap.")
    # ---- reviewer-requested sensitivity knobs for the baseline ----
    ap.add_argument("--pcie-bw-gb", type=float, default=58.0,
                    help="PCIe effective bandwidth (GB/s). "
                         "PCIe 4.0 x16 = 28, PCIe 5.0 x16 = 58 (default)")
    ap.add_argument("--cpu-compute-gflops", type=float, default=200.0,
                    help="CPU FP16 compute throughput in GFLOPS")
    ap.add_argument("--cpu-dram-bw-gb", type=float, default=100.0,
                    help="CPU DRAM bandwidth (GB/s)")
    ap.add_argument("--L-kernel-launch-ns", type=float, default=5000.0,
                    help="CUDA kernel launch overhead (ns)")
    ap.add_argument("--L-cpu-gpu-offload-ns", type=float, default=10000.0,
                    help="Compound GPU<->CPU offload overhead per round (ns)")
    ap.add_argument("--L-pcie-setup-ns", type=float, default=1000.0,
                    help="Per-DMA PCIe setup overhead (ns)")
    # ---- NoCxl baseline knobs ----
    ap.add_argument("--nocxl-L-kernel-launch-ns", type=float, default=5000.0,
                    help="NoCxl: CUDA kernel relaunch per offload boundary")
    ap.add_argument("--nocxl-L-device-command-ns", type=float, default=1000.0,
                    help="NoCxl: CPU/driver command issue per offload")
    ap.add_argument("--nocxl-L-sync-ns", type=float, default=1000.0,
                    help="NoCxl: stream/event sync per offload")
    ap.add_argument("--nocxl-ffn-gemms", type=int, default=3,
                    help="NoCxl: FFN GEMM kernel launches per adapter per "
                         "layer (SwiGLU gate/up/down = 3, classic up/down = 2)")
    ap.add_argument("--nocxl-fused", action="store_true",
                    help="NoCxl: BGMV/S-LoRA-style fusion — one launch per op "
                         "per layer, independent of adapter/request count "
                         "(default: per-adapter unfused launches)")
    # ---- Grace-Hopper baseline knobs ----
    ap.add_argument("--gh-c2c-bw-gb", type=float, default=450.0,
                    help="Grace-Hopper: NVLink-C2C effective bandwidth (GB/s). "
                         "Default 450 (one direction; 900 bidirectional)")
    ap.add_argument("--gh-L-c2c-setup-ns", type=float, default=500.0,
                    help="Grace-Hopper: per-transfer C2C setup overhead (ns)")
    ap.add_argument("--gh-L-kernel-launch-ns", type=float, default=5000.0,
                    help="Grace-Hopper: CUDA kernel launch overhead (ns)")
    ap.add_argument("--gh-cache-gb", type=float, default=8.0,
                    help="Grace-Hopper: GPU LoRA LRU cache budget (GB)")
    # ---- reviewer-requested CLoRA hardware sensitivity knobs ----
    # These feed the Python cost model (strategy selection, Eqs 1-9).  The C
    # event sim reads its own copies from the parameter file; pass a matching
    # --c-parameter-file so both layers see the same hardware.
    ap.add_argument("--cxl-latency-ns", type=float, default=200.0,
                    help="CXL link latency L_CXL in ns (paper Eq 8 uses "
                         "2*L_CXL per attention round trip). Default 200.")
    ap.add_argument("--cxl-link-bw-gb", type=float, default=128.0,
                    help="CXL link bandwidth per device W_CXL in GB/s. "
                         "Default 128 (CXL 3.1, paper Table 4).")
    ap.add_argument("--ndp-tflops", type=float, default=2.0,
                    help="NDP compute per CLoRA device C_CXL in FP16 TFLOPS. "
                         "Default 2 (paper Table 4).")
    ap.add_argument("--cxl-dram-bw-gb", type=float, default=1100.0,
                    help="Device-internal DRAM bandwidth W_DRAM in GB/s. "
                         "Default 1100 (paper Table 4).")
    ap.add_argument("--n-cxl", type=int, default=4,
                    help="Number of CLoRA (CXL) memory devices N_CXL "
                         "(default 4, paper Table 4). For >16, pass a "
                         "--c-parameter-file with cxl_channel_number >= N_CXL.")
    ap.add_argument("--c-parameter-file", type=str, default=None,
                    help="Override the C sim's parameter file "
                         "(default: config/parameters.conf). The sensitivity "
                         "study generates one conf per sweep point.")
    args = ap.parse_args()
    global C_PARAMETER_FILE
    if args.c_parameter_file:
        C_PARAMETER_FILE = os.path.abspath(args.c_parameter_file)
    run_smoke(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
