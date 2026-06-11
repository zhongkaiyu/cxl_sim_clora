#!/usr/bin/env python3
"""
test_baseline.py
================

Manually verifiable tests for ``baseline_cpu_offload.py``. Each test sets
round numbers for the HW knobs so the expected nanoseconds can be hand-
derived from the formulas in the module's docstring.

Run:  python3 script/test_baseline.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from clora_strategy import ModelConfig  # noqa: E402
from baseline_cpu_offload import (  # noqa: E402
    BaselineHWConfig,
    LRUAdapterCache,
    adapter_full_bytes,
    baseline_step_ns,
)


# --------------------------------------------------------------- fixture

# Round numbers chosen so every test result is a clean hand-derivation.
HW = BaselineHWConfig(
    gpu_compute=100e12,        # 100 TFLOPS
    gpu_mem_bw=1000e9,         # 1 TB/s HBM
    gpu_mem_bytes=8 * 10**9,
    cpu_compute=100e9,         # 100 GFLOPS
    cpu_dram_bw=100e9,         # 100 GB/s
    pcie_bw=32e9,              # 32 GB/s
    L_pcie_setup=1000.0,       # 1 us
    L_kernel_launch=5000.0,    # 5 us
    L_cpu_gpu_offload=10000.0, # 10 us
)
MODEL = ModelConfig(d=4096)


# --------------------------------------------------------------- harness

_PASS = 0
_FAIL = 0


def _close(a: float, b: float, tol: float = 1e-3) -> bool:
    if abs(a) < 1e-9 and abs(b) < 1e-9:
        return True
    return abs(a - b) / max(abs(a), abs(b)) < tol


def _expect_close(actual: float, expected: float, name: str) -> None:
    global _PASS, _FAIL
    if _close(actual, expected):
        _PASS += 1
        print(f"  PASS  {name}  ({actual:,.2f} ns ~ {expected:,.2f})")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  got {actual:,.2f}  expected {expected:,.2f}")


def _expect_eq(actual, expected, name: str) -> None:
    global _PASS, _FAIL
    if actual == expected:
        _PASS += 1
        print(f"  PASS  {name}  ({actual})")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  got {actual}  expected {expected}")


# --------------------------------------------------------------- tests

def t1_cold_single_adapter() -> None:
    """One rank=16 adapter, batch=1, cold. base=0 so only LoRA + small GPU compute.

    Hand derivation (D=4096, S=2, n_layers=1, n_matrices=1, base=0,
                     test HW: cpu=100GF, dram=100GB/s, pcie=32GB/s):
      x_bytes = y_bytes = 1*4096*2 = 8192 B
      per_pcie = 2*1000 + (8192+8192)/32e9*1e9 = 2000 + 512 = 2512 ns
      cpu_flops = 4*1*4096*16 = 262_144 -> 262144/100e9*1e9 = 2621.44 ns
      cpu_dram  = 2*16*4096*2 = 262_144 -> 2621.44 ns
      cpu_path  = max(2621.44, 2621.44) = 2621.44
      kernels   = 2*1*(5000+10000) = 30_000
      base_per_layer = 24*4096^2*1 / 100e12 * 1e9 ~ 4026.53 (n_layers=1, base=0
                     so HBM term = 0, max = compute)
      missed_per_layer = (2512 + 2621.44 + 30000) / 1 = 35133.44
      gpu_path_per_layer = 4026.53 + 0 = 4026.53  (no cached, no attn)
      t_per_layer (parallel) = max(4026.53, 35133.44) + 0 = 35133.44
      total = 35133.44 * 1 = 35133.44
    """
    cache = LRUAdapterCache(capacity_bytes=10**12, model=MODEL,
                            n_layers=1, n_matrices=1)
    r = baseline_step_ns(
        {0: (16, 1)}, {0: 0}, HW, MODEL, cache,
        base_model_bytes=0, n_layers=1, n_matrices=1, fusion="fused")
    print("\n[t1] cold single rank-16 adapter (parallel mode)")
    _expect_close(r["t_pcie_lora"], 2512.0, "t1 pcie_lora")
    _expect_close(r["t_cpu_lora"], 2621.44, "t1 cpu_lora")
    _expect_close(r["t_kernels_lora"], 30000.0, "t1 kernels_lora")
    _expect_close(r["t_base"], 4026.53184, "t1 base_gpu")
    _expect_close(r["total_ns"], 35133.44, "t1 total (max parallel)")
    _expect_eq(r["n_missed"], 1, "t1 missed=1")
    _expect_eq(r["n_cached"], 0, "t1 cached=0")


def t2_warm_after_t1() -> None:
    """Now the cache holds adapter 0. Same call -> hit, only cached LoRA + base.

    cached_gpu = 4*1*4096*16 / 100e12 * 1e9 = 2.62144 ns
    base       = 4026.53184  (unchanged)
    With parallel=True and no LoRA path (missed_per_layer=0):
      t_per_layer = max(4026.53 + 2.62, 0) + 0 = 4029.15
    """
    cache = LRUAdapterCache(capacity_bytes=10**12, model=MODEL,
                            n_layers=1, n_matrices=1)
    # prime
    baseline_step_ns({0: (16, 1)}, {0: 0}, HW, MODEL, cache,
                     base_model_bytes=0, n_layers=1, n_matrices=1)
    # warm call
    r = baseline_step_ns({0: (16, 1)}, {0: 0}, HW, MODEL, cache,
                         base_model_bytes=0, n_layers=1, n_matrices=1)
    print("\n[t2] warm same adapter after prime")
    _expect_close(r["t_cached_gpu_lora"], 2.62144, "t2 cached_gpu")
    _expect_close(r["t_pcie_lora"], 0.0, "t2 pcie=0")
    _expect_close(r["t_kernels_lora"], 0.0, "t2 kernels=0")
    _expect_close(r["t_base"], 4026.53184, "t2 base unchanged")
    _expect_close(r["total_ns"], 4029.15328, "t2 total ~ base+cached")
    _expect_eq(r["n_cached"], 1, "t2 cached=1")
    _expect_eq(r["n_missed"], 0, "t2 missed=0")


def t3_lru_evicts_then_remisses() -> None:
    """Cache capacity = exactly one rank-16 adapter. Inserting 1, then 2,
    evicts 1; touching 1 again misses."""
    one_ad_bytes = adapter_full_bytes(16, MODEL, n_layers=1, n_matrices=1)
    cache = LRUAdapterCache(capacity_bytes=one_ad_bytes, model=MODEL,
                            n_layers=1, n_matrices=1)

    _expect_eq(cache.touch(1, 16), False, "t3 first touch(1) = miss")
    _expect_eq(cache.touch(2, 16), False, "t3 touch(2) = miss + evict 1")
    _expect_eq(cache.touch(1, 16), False, "t3 touch(1) again = miss")
    snap = cache.snapshot()
    _expect_eq(snap["n_hits"], 0, "t3 hits=0")
    _expect_eq(snap["n_misses"], 3, "t3 misses=3")


def t4_fused_vs_unfused_kernel_overhead() -> None:
    """4 cold adapters all rank=16, batch=1.
       Fused  -> 1 rank group -> 1 kernel pair -> 30,000 ns kernel cost.
       Unfused -> 4 kernel pairs -> 120,000 ns kernel cost.
    """
    cap = adapter_full_bytes(16, MODEL, n_layers=1, n_matrices=1) * 10
    adapters = {i: (16, 1) for i in range(4)}

    # fused
    cache_f = LRUAdapterCache(capacity_bytes=cap, model=MODEL,
                              n_layers=1, n_matrices=1)
    r_f = baseline_step_ns(adapters, {i: 0 for i in adapters},
                           HW, MODEL, cache_f,
                           base_model_bytes=0, n_layers=1, n_matrices=1,
                           fusion="fused")
    # unfused -- need a fresh cache
    cache_u = LRUAdapterCache(capacity_bytes=cap, model=MODEL,
                              n_layers=1, n_matrices=1)
    r_u = baseline_step_ns(adapters, {i: 0 for i in adapters},
                           HW, MODEL, cache_u,
                           base_model_bytes=0, n_layers=1, n_matrices=1,
                           fusion="unfused")

    print("\n[t4] fused vs unfused with 4 cold rank=16 adapters")
    _expect_close(r_f["t_kernels_lora"], 30000.0, "t4 fused kernels=30k")
    _expect_close(r_u["t_kernels_lora"], 120000.0, "t4 unfused kernels=120k")
    _expect_eq(r_f["n_kernel_pairs_lora"], 1, "t4 fused pairs=1")
    _expect_eq(r_u["n_kernel_pairs_lora"], 4, "t4 unfused pairs=4")


def t5_attention_kv_in_cpu() -> None:
    """Adapter with kv_tokens=128 (E1 cached, so LoRA path is just t_cached_gpu).

    Attention (D=4096, S=2, n_layers=1, total_batch=1, total_kv=128):
      pcie_bytes = 2 * 1 * 4096 * 2 * 1 = 16384
      pcie_setup = 2 * 1000 * 1 = 2000
      t_attn_pcie = 2000 + 16384/32e9*1e9 = 2000 + 512 = 2512
      cpu_flops  = 4 * 4096 * 128 * 1 = 2_097_152
      cpu_compute = 2_097_152 / 100e9 * 1e9 = 20_971.52
      dram_bytes = 2 * 4096 * 2 * 128 * 1 = 2_097_152
      cpu_dram    = 20_971.52
      kernels     = 2 * 1 * 15_000 = 30_000
      t_attn      = 2512 + 20_971.52 + 30_000 = 53_483.52

    With parallel mode the cached LoRA + base still overlap with missed-LoRA
    path (which is 0 here), then attention adds on top:
      t_per_layer = max(4026.53 + 2.62, 0) + 53483.52 = 57512.67
    """
    cache = LRUAdapterCache(capacity_bytes=10**12, model=MODEL,
                            n_layers=1, n_matrices=1)
    # prime the cache so LoRA path is hit
    baseline_step_ns({0: (16, 1)}, {0: 0}, HW, MODEL, cache,
                     base_model_bytes=0, n_layers=1, n_matrices=1)
    r = baseline_step_ns({0: (16, 1)}, {0: 128}, HW, MODEL, cache,
                         base_model_bytes=0, n_layers=1, n_matrices=1)
    print("\n[t5] attention with kv=128 in CPU DRAM")
    _expect_close(r["t_attn"], 53483.52, "t5 t_attn")


def t6_multiple_ranks_fused_kernels() -> None:
    """2 rank=16 + 2 rank=32 adapters, all cold.  Fused groups by rank ->
    2 kernel pairs per (layer, matrix). For n_layers=n_matrices=1: 2 pairs."""
    cap = 10**12
    adapters = {0: (16, 1), 1: (16, 1), 2: (32, 1), 3: (32, 1)}
    cache = LRUAdapterCache(capacity_bytes=cap, model=MODEL,
                            n_layers=1, n_matrices=1)
    r = baseline_step_ns(adapters, {i: 0 for i in adapters},
                         HW, MODEL, cache,
                         base_model_bytes=0, n_layers=1, n_matrices=1,
                         fusion="fused")
    print("\n[t6] multiple ranks, fused")
    _expect_eq(r["n_kernel_pairs_lora"], 2, "t6 pairs=2 (one per rank group)")
    _expect_close(r["t_kernels_lora"], 60000.0, "t6 kernels=60k")


def t7_parallel_vs_serial() -> None:
    """The headline behavior change: with parallel=True (default), GPU base +
    cached LoRA overlaps with the missed-LoRA path per layer. With parallel=False,
    they sum.

    Setup: 1 cold rank-16 adapter, batch=1, base_model_bytes=0,
           n_layers=1, n_matrices=1 (so all per-layer terms == totals).

    From t1 we already have:
       base_per_layer       = 4026.53
       missed_per_layer     = 35133.44 (= 2512 + 2621.44 + 30000)
    parallel: max(4026.53, 35133.44) = 35133.44
    serial:   4026.53 + 35133.44     = 39159.97
    Ratio    = 39159.97 / 35133.44   = 1.115 -- modest because LoRA dominates.
    """
    cache_p = LRUAdapterCache(capacity_bytes=10**12, model=MODEL,
                              n_layers=1, n_matrices=1)
    cache_s = LRUAdapterCache(capacity_bytes=10**12, model=MODEL,
                              n_layers=1, n_matrices=1)
    r_p = baseline_step_ns({0: (16, 1)}, {0: 0}, HW, MODEL, cache_p,
                           base_model_bytes=0, n_layers=1, n_matrices=1,
                           parallel=True)
    r_s = baseline_step_ns({0: (16, 1)}, {0: 0}, HW, MODEL, cache_s,
                           base_model_bytes=0, n_layers=1, n_matrices=1,
                           parallel=False)
    print("\n[t7] parallel vs serial layer overlap")
    _expect_close(r_p["total_ns"], 35133.44, "t7 parallel total")
    _expect_close(r_s["total_ns"], 39159.97, "t7 serial total")
    assert r_s["total_ns"] >= r_p["total_ns"], "serial must be >= parallel"


def t8_zero_capacity_cache_always_misses() -> None:
    """Cache capacity = 0 -> every touch is a miss, nothing ever caches."""
    cache = LRUAdapterCache(capacity_bytes=0, model=MODEL,
                            n_layers=1, n_matrices=1)
    for _ in range(5):
        cache.touch(0, 16)
    snap = cache.snapshot()
    print("\n[t7] zero-capacity cache")
    _expect_eq(snap["n_hits"], 0, "t7 hits=0")
    _expect_eq(snap["n_misses"], 5, "t7 misses=5")
    _expect_eq(snap["n_cached"], 0, "t7 cached=0")


# --------------------------------------------------------------- main

def main() -> int:
    print("baseline_cpu_offload tests")
    print("HW:", HW)
    print("Model:", MODEL)
    t1_cold_single_adapter()
    t2_warm_after_t1()
    t3_lru_evicts_then_remisses()
    t4_fused_vs_unfused_kernel_overhead()
    t5_attention_kv_in_cpu()
    t6_multiple_ranks_fused_kernels()
    t7_parallel_vs_serial()
    t8_zero_capacity_cache_always_misses()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
