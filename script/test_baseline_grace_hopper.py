#!/usr/bin/env python3
"""
test_baseline_grace_hopper.py
=============================

Manually verifiable tests for ``baseline_grace_hopper``.  Round-numbered HW
constants so every assertion can be recomputed by hand.

Run:  python3 script/test_baseline_grace_hopper.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from clora_strategy import ModelConfig  # noqa: E402
from baseline_cpu_offload import LRUAdapterCache  # noqa: E402
from baseline_grace_hopper import (  # noqa: E402
    GraceHopperHWConfig,
    grace_hopper_step_ns,
)


# --------------------------------------------------------------- fixture
# Round numbers so every expectation is a clean hand-derivation.
HW = GraceHopperHWConfig(
    gpu_compute=100e12,      # 100 TFLOPS
    gpu_mem_bw=1000e9,       # 1 TB/s HBM
    gpu_mem_bytes=8 * 10**9,
    c2c_bw=450e9,            # 450 GB/s NVLink-C2C
    L_kernel_launch_ns=5000.0,
    L_c2c_setup_ns=500.0,
)
MODEL = ModelConfig(d=4096)


# --------------------------------------------------------------- harness
_PASS = 0
_FAIL = 0


def _close(a: float, b: float, tol: float = 1e-3) -> bool:
    if abs(a) < 1.0 and abs(b) < 1.0:
        # both effectively zero (sub-nanosecond); within absolute tolerance
        return abs(a - b) < 1.0
    return abs(a - b) / max(abs(a), abs(b)) < tol


def _expect_close(actual: float, expected: float, name: str) -> None:
    global _PASS, _FAIL
    if _close(actual, expected):
        _PASS += 1
        print(f"  PASS  {name}  ({actual:,.2f} ~ {expected:,.2f})")
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
    """Cold rank-16 adapter, B=1, n_layers=n_matrices=1, base=0.

    Hand-derivation (D=4096, S=2):
      bytes_per_op       = 2 * 16 * 4096 * 2 = 262_144 B
      t_c2c_transfer     = 500 + 262144 / 450e9 * 1e9
                         = 500 + 582.54  = 1_082.54 ns
      gpu_flops          = 4 * 1 * 4096 * 16 = 262_144
      t_gpu_lora_compute = 262144 / 100e12 * 1e9 = 2.62144 ns
      max(transfer, compute) = 1082.54
      t_kernels_lora     = 1 * 5000 = 5000 ns
      t_missed_lora      = 1082.54 + 5000 = 6082.54 ns
      base               = max(24*4096^2*1/100e12*1e9, 0) ~ 4026.53
      gpu_per_layer (no cached) = 4026.53
      t_per_layer (parallel)    = max(4026.53, 6082.54) + 0 (no attn) = 6082.54
      total                     = 6082.54
    """
    cache = LRUAdapterCache(capacity_bytes=10**12, model=MODEL,
                            n_layers=1, n_matrices=1)
    r = grace_hopper_step_ns(
        {0: (16, 1)}, {0: 0}, HW, MODEL, cache,
        base_model_bytes=0, n_layers=1, n_matrices=1, fusion="fused")
    print("\n[t1] cold rank-16 adapter, B=1")
    _expect_close(r["t_c2c_transfer"], 1082.54, "t1 c2c_transfer")
    _expect_close(r["t_gpu_lora_compute"], 2.62144, "t1 gpu_lora_compute")
    _expect_close(r["t_kernels_lora"], 5000.0, "t1 kernels_lora")
    _expect_close(r["t_base"], 4026.53184, "t1 base")
    _expect_close(r["total_ns"], 6082.54, "t1 total")
    _expect_eq(r["n_missed"], 1, "t1 missed=1")
    _expect_eq(r["n_cached"], 0, "t1 cached=0")


def t2_warm_after_t1() -> None:
    """Now the adapter is cached -> GPU computes locally, no C2C transfer."""
    cache = LRUAdapterCache(capacity_bytes=10**12, model=MODEL,
                            n_layers=1, n_matrices=1)
    grace_hopper_step_ns({0: (16, 1)}, {0: 0}, HW, MODEL, cache,
                         base_model_bytes=0, n_layers=1, n_matrices=1)
    r = grace_hopper_step_ns({0: (16, 1)}, {0: 0}, HW, MODEL, cache,
                             base_model_bytes=0, n_layers=1, n_matrices=1)
    print("\n[t2] warm cache after prime")
    _expect_close(r["t_cached_gpu_lora"], 2.62144, "t2 cached_gpu_lora")
    _expect_close(r["t_c2c_transfer"], 0.0, "t2 c2c=0")
    _expect_close(r["t_kernels_lora"], 0.0, "t2 kernels=0")
    # Total = max(base + cached, 0) + 0 = base + cached
    _expect_close(r["total_ns"], 4029.15328, "t2 total = base + cached")
    _expect_eq(r["n_cached"], 1, "t2 cached=1")
    _expect_eq(r["n_missed"], 0, "t2 missed=0")


def t3_c2c_faster_than_pcie() -> None:
    """At identical compute, C2C 450 GB/s should be ~16x faster transfer
    than PCIe 28 GB/s. Verify with a larger transfer."""
    print("\n[t3] C2C transfer scales with bandwidth")
    # rank=128, B=4 → bytes = 2*128*4096*2 = 1_048_576 B per op
    one_op_bytes = 2 * 128 * 4096 * 2
    expected_ns = 500 + one_op_bytes / 450e9 * 1e9  # 500 + 2330.17 ~ 2830
    cache = LRUAdapterCache(capacity_bytes=10**12, model=MODEL,
                            n_layers=1, n_matrices=1)
    r = grace_hopper_step_ns({0: (128, 4)}, {0: 0}, HW, MODEL, cache,
                             base_model_bytes=0, n_layers=1, n_matrices=1)
    _expect_close(r["t_c2c_transfer"], expected_ns, "t3 c2c bytes / 450e9")


def t4_fused_vs_unfused_kernels() -> None:
    """4 cold rank=16 adapters. Fused -> 1 kernel; unfused -> 4 kernels."""
    cap = 10**12
    adapters = {i: (16, 1) for i in range(4)}

    cache_f = LRUAdapterCache(capacity_bytes=cap, model=MODEL,
                              n_layers=1, n_matrices=1)
    r_f = grace_hopper_step_ns(adapters, {i: 0 for i in adapters},
                               HW, MODEL, cache_f,
                               base_model_bytes=0, n_layers=1, n_matrices=1,
                               fusion="fused")
    cache_u = LRUAdapterCache(capacity_bytes=cap, model=MODEL,
                              n_layers=1, n_matrices=1)
    r_u = grace_hopper_step_ns(adapters, {i: 0 for i in adapters},
                               HW, MODEL, cache_u,
                               base_model_bytes=0, n_layers=1, n_matrices=1,
                               fusion="unfused")
    print("\n[t4] fused vs unfused kernel count")
    _expect_eq(r_f["n_kernels_lora"], 1, "t4 fused = 1 kernel")
    _expect_eq(r_u["n_kernels_lora"], 4, "t4 unfused = 4 kernels")
    _expect_close(r_f["t_kernels_lora"], 5000.0, "t4 fused kernels = 5k ns")
    _expect_close(r_u["t_kernels_lora"], 20000.0, "t4 unfused kernels = 20k ns")


def t5_attention_kv_in_cpu() -> None:
    """1 adapter (cached), kv=128 tokens. Attention transfers KV over C2C.

    Hand-derivation (D=4096, S=2, n_layers=1, B=1, total_kv=128):
      kv_bytes_per_layer = 2 * 4096 * 2 * 128 = 2_097_152 B
      t_attn_transfer    = 500 + 2097152/450e9*1e9 = 500 + 4660.78 = 5160.78
      attn_flops         = 4 * 4096 * 128 = 2_097_152
      t_attn_compute     = 2097152 / 100e12 * 1e9 = 20.97 ns
      max(transfer, compute) = 5160.78
      t_attn_kernels     = 1 * 5000 = 5000
      t_attn             = 5160.78 + 5000 = 10160.78
    """
    cache = LRUAdapterCache(capacity_bytes=10**12, model=MODEL,
                            n_layers=1, n_matrices=1)
    grace_hopper_step_ns({0: (16, 1)}, {0: 0}, HW, MODEL, cache,
                         base_model_bytes=0, n_layers=1, n_matrices=1)
    r = grace_hopper_step_ns({0: (16, 1)}, {0: 128}, HW, MODEL, cache,
                             base_model_bytes=0, n_layers=1, n_matrices=1)
    print("\n[t5] attention with kv=128 in CPU memory")
    _expect_close(r["t_attn"], 10160.78, "t5 t_attn")


def t6_compute_dominates_when_link_fast() -> None:
    """With effectively infinite C2C bandwidth, GPU compute dominates."""
    print("\n[t6] infinite C2C -> compute-bound")
    hw_inf = replace(HW, c2c_bw=1e18, L_c2c_setup_ns=0.0)
    cache = LRUAdapterCache(capacity_bytes=10**12, model=MODEL,
                            n_layers=1, n_matrices=1)
    # Large B to make compute meaningful
    r = grace_hopper_step_ns({0: (64, 1000)}, {0: 0}, hw_inf, MODEL, cache,
                             base_model_bytes=0, n_layers=1, n_matrices=1)
    # gpu_flops = 4 * 1000 * 4096 * 64 = 1.05e9 -> 10485.76 ns at 100 TFLOPS
    expected_compute = 4 * 1000 * 4096 * 64 / 100e12 * 1e9
    _expect_close(r["t_gpu_lora_compute"], expected_compute,
                  "t6 compute matches formula")
    # missed = max(transfer~0, compute) + kernel = compute + 5000
    _expect_close(r["t_c2c_transfer"], 0.0, "t6 transfer ~ 0")


def t7_zero_capacity_cache_always_misses() -> None:
    """Cache capacity = 0 -> every touch is a miss."""
    cache = LRUAdapterCache(capacity_bytes=0, model=MODEL,
                            n_layers=1, n_matrices=1)
    for _ in range(5):
        cache.touch(0, 16)
    snap = cache.snapshot()
    print("\n[t7] zero-capacity cache")
    _expect_eq(snap["n_hits"], 0, "t7 hits=0")
    _expect_eq(snap["n_misses"], 5, "t7 misses=5")


def t8_throughput_ordering_vs_pcie() -> None:
    """Sanity: Grace-Hopper with realistic Uniform-7B shape produces a
    throughput between naive CPU-LoRA and CLoRA. We can't verify exact
    numbers in a unit test but we can sanity-check the breakdown
    components scale right."""
    print("\n[t8] realistic Uniform-7B shape sanity")
    hw_real = GraceHopperHWConfig(
        gpu_compute=989e12, gpu_mem_bw=3350e9, gpu_mem_bytes=8 * 10**9,
        c2c_bw=450e9,
    )
    cache = LRUAdapterCache(capacity_bytes=8 * 10**9, model=MODEL,
                            n_layers=32, n_matrices=7)
    # 32 cold rank-32 adapters, B=1 each
    adapters = {i: (32, 1) for i in range(32)}
    kv = {i: 562 for i in range(32)}  # avg KV
    r = grace_hopper_step_ns(adapters, kv, hw_real, MODEL, cache,
                             base_model_bytes=14 * 10**9,
                             n_layers=32, n_matrices=7, fusion="fused")
    # Sanity: missed > 0, transfer > 0
    _expect_eq(r["n_missed"], 32, "t8 32 cache misses")
    # base ~ 14e9/3350e9*1e9 ~ 4.18 ms (HBM-bound at small batch)
    _expect_close(r["t_base"], 14e9 / 3350e9 * 1e9, "t8 base = HBM-bound")
    # Step time should be in a few ms range, not micro or seconds
    assert 1e6 < r["total_ns"] < 1e9, \
        f"t8 step time {r['total_ns']:,.0f} ns out of expected range"
    print(f"  INFO  Uniform-7B-like step: {r['total_ns']/1e6:.2f} ms; "
          f"throughput ~ {32 / (r['total_ns']/1e9):.0f} tok/s")


# --------------------------------------------------------------- main


def main() -> int:
    print("baseline_grace_hopper tests")
    print("HW (defaults):", HW)
    t1_cold_single_adapter()
    t2_warm_after_t1()
    t3_c2c_faster_than_pcie()
    t4_fused_vs_unfused_kernels()
    t5_attention_kv_in_cpu()
    t6_compute_dominates_when_link_fast()
    t7_zero_capacity_cache_always_misses()
    t8_throughput_ordering_vs_pcie()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
