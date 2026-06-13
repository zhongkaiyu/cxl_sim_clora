#!/usr/bin/env python3
"""
test_clora_strategy.py
======================

Manually verifiable tests for ``clora_strategy.py``.

Run with:
    python3 script/test_clora_strategy.py

Each test prints the inputs and the numbers it derived from the cost model so a
reader can re-derive them with a calculator. Pass/fail is asserted at the end.

The HW fixture below roughly matches the paper (A100 + 4 CLoRA mem devices,
Table 4) but is deliberately quotable round numbers, not measurements.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

# allow running this file directly from anywhere
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from clora_strategy import (  # noqa: E402
    Adapter,
    Decision,
    HardwareConfig,
    ModelConfig,
    Strategy,
    SystemState,
    choose_strategy,
    classify_adapters,
    compute_kv_in_gpu_fraction,
    estimate_base_model_ns_per_layer,
    pre_cache_hot_adapters,
    cost_lora,
    gpu_mem_for,
)


# ----------------------------------------------------------------- fixture

HW = HardwareConfig(
    gpu_compute=312e12,      # A100 FP16 peak ~ 312 TFLOPS
    gpu_mem_bw=1935e9,       # 1.935 TB/s HBM
    gpu_mem_bytes=40 * 10**9,  # 40 GB for LoRA + KV (after base model)
    cxl_compute=2e12,        # 2 TFLOPS / device
    cxl_dram_bw=1.1e12,      # 1.1 TB/s internal
    cxl_link_bw=128e9,       # 128 GB/s CXL link
    cxl_latency=200,         # 200 ns
    n_cxl=4,
)
MODEL = ModelConfig(d=4096)


# ----------------------------------------------------------------- helpers


_PASS = 0
_FAIL = 0


def _expect(cond: bool, name: str, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}: {detail}")


# ----------------------------------------------------------------- t1

def t1_costs_grow_with_batch() -> None:
    """Sanity: for any fixed strategy, doubling the batch must not lower cost."""
    a_small = Adapter(0, rank=16, batch=10)
    a_big = Adapter(0, rank=16, batch=100)
    print("\n[t1] cost monotone in B (rank=16, D=4096, S=2)")
    for E in Strategy:
        c_small = cost_lora(E, a_small, HW, MODEL)
        c_big = cost_lora(E, a_big, HW, MODEL)
        print(f"     E{int(E)}:  B=10 -> {c_small:8.1f} ns   B=100 -> {c_big:8.1f} ns")
        _expect(c_big + 1e-9 >= c_small,
                f"E{int(E)} monotone in B",
                f"B=10 cost={c_small}, B=100 cost={c_big}")


# ----------------------------------------------------------------- t2

def t2_small_batch_prefers_e3_over_e2() -> None:
    """B=1: E2 must load 2.R.D.S bytes (~256 kB at R=16). E3 only sends 2.B.D.S
    bytes (~16 kB). E3 should be the cheaper of the two."""
    a = Adapter(0, rank=16, batch=1)
    e2 = cost_lora(Strategy.E2, a, HW, MODEL)
    e3 = cost_lora(Strategy.E3, a, HW, MODEL)
    # By hand:  link_bytes_E2 = 2*16*4096*2 = 262 144 B over 128 GB/s ~ 2050 ns + 200 ns latency
    #           link_bytes_E3 =  2* 1*4096*2 =  16 384 B over 128 GB/s ~  128 ns + 200 ns latency
    print(f"\n[t2] small batch (B=1):  E2={e2:.1f} ns,  E3={e3:.1f} ns")
    print("     by hand:               E2 ~ 200 + 262144/128e9*1e9 ~ 2248 ns")
    print("                            E3 ~ 200 + 16384/128e9*1e9  ~ 328  ns  + tiny NDP")
    _expect(e3 < e2, "t2 small batch: E3 < E2", f"E2={e2}, E3={e3}")


# ----------------------------------------------------------------- t3

def t3_large_batch_prefers_e1_over_e3() -> None:
    """B=512: E3's per-request link traffic scales with B and dwarfs E1's fixed
    HBM read. E1 must win by a wide margin."""
    a = Adapter(0, rank=16, batch=512)
    e1 = cost_lora(Strategy.E1, a, HW, MODEL)
    e3 = cost_lora(Strategy.E3, a, HW, MODEL)
    # By hand: E1 GPU compute = 4*512*4096*16 / 312e12 *1e9 ~ 430 ns
    #          E3 link  = 2*512*4096*2 / 128e9 * 1e9 ~ 65 536 ns
    print(f"\n[t3] large batch (B=512): E1={e1:.1f} ns,  E3={e3:.1f} ns")
    print("     by hand:                 E1 ~ max(430, 135) ns")
    print("                              E3 ~ 65 736 + NDP(~16 800) ns")
    _expect(e1 < e3, "t3 large batch: E1 < E3", f"E1={e1}, E3={e3}")


# ----------------------------------------------------------------- t4

def t4_fits_picks_e1() -> None:
    """One adapter, plenty of GPU memory: cheapest is E1 for a sizable batch."""
    a = Adapter(7, rank=16, batch=512)
    state = SystemState(adapters={7: a})
    d = choose_strategy(7, state, HW, MODEL)
    print(f"\n[t4] one adapter, 40 GB free: chose {d}")
    _expect(d.strategy is Strategy.E1, "t4 picks E1 when memory is abundant",
            f"got {d}")
    _expect(d.evict_cold_ids == [] and d.evict_kv_bytes == 0,
            "t4 no eviction needed", f"got {d}")


# ----------------------------------------------------------------- t5

def t5_evicts_cold_to_use_e1() -> None:
    """Budget fits exactly two adapters' (A+B). One warm + one cold occupy both
    slots; the new adapter needs E1. Algorithm should evict only the cold one."""
    one_ab = 16 * 4096 * 2 * 2  # 256 KiB: both matrices of a rank-16 adapter
    hw = replace(HW, gpu_mem_bytes=2 * one_ab)

    warm = Adapter(adapter_id=1, rank=16, batch=0,
                   a_in_gpu=True, b_in_gpu=True, temperature=5.0)
    cold = Adapter(adapter_id=2, rank=16, batch=0,
                   a_in_gpu=True, b_in_gpu=True, temperature=0.1)
    target = Adapter(adapter_id=3, rank=16, batch=512)
    state = SystemState(adapters={1: warm, 2: cold, 3: target})

    d = choose_strategy(3, state, hw, MODEL)
    print(f"\n[t5] 2-slot budget, warm + cold parked, new big batch: chose {d}")
    print("     mean(temps of non-serving) = (5.0 + 0.1)/2 = 2.55")
    print("     -> warm (5.0) is hot, cold (0.1) is cold")
    _expect(d.strategy is Strategy.E1, "t5 picks E1", f"got {d}")
    _expect(d.evict_cold_ids == [2], "t5 evicts only the cold adapter",
            f"got {d}")


# ----------------------------------------------------------------- t6

def t6_zero_budget_falls_back_to_zero_mem_strategy() -> None:
    """Budget = 0 and no cold adapters. Only E2 and E3 need zero GPU memory;
    the algorithm must pick whichever of those is cheaper."""
    hw = replace(HW, gpu_mem_bytes=0)
    target = Adapter(adapter_id=0, rank=16, batch=1)
    state = SystemState(adapters={0: target})

    d = choose_strategy(0, state, hw, MODEL)
    print(f"\n[t6] budget = 0, no cold: chose {d}")
    _expect(gpu_mem_for(d.strategy, target, MODEL) == 0,
            "t6 picks a zero-GPU-mem strategy", f"got {d}")
    # at B=1, E3 should beat E2 (see t2)
    _expect(d.strategy is Strategy.E3, "t6 picks E3 at B=1",
            f"got {d}")


# ----------------------------------------------------------------- t7

def t7_classify_adapters() -> None:
    """temperature-based hot/cold split (§6.2). Mean over non-serving = (5+1+2)/3 = 2.67."""
    state = SystemState(adapters={
        1: Adapter(1, rank=8, batch=4, temperature=10.0),  # serving
        2: Adapter(2, rank=8, batch=0, temperature=5.0),   # hot (5 >= 2.67)
        3: Adapter(3, rank=8, batch=0, temperature=1.0),   # cold (1 < 2.67)
        4: Adapter(4, rank=8, batch=0, temperature=2.0),   # cold (2 < 2.67)
    })
    serving, hot, cold = classify_adapters(state)
    print("\n[t7] classify_adapters: "
          f"serving={[a.adapter_id for a in serving]}, "
          f"hot={[a.adapter_id for a in hot]}, "
          f"cold={[a.adapter_id for a in cold]}")
    _expect({a.adapter_id for a in serving} == {1}, "t7 serving = {1}")
    _expect({a.adapter_id for a in hot} == {2}, "t7 hot = {2}")
    _expect({a.adapter_id for a in cold} == {3, 4}, "t7 cold = {3, 4}")


# ----------------------------------------------------------------- t8

def t8_evict_kv_when_no_cold() -> None:
    """No cold adapters; only KV cache is takeable. Algorithm should evict KV
    to fit E1, since E1 is much cheaper than E3 for large B."""
    one_ab = 16 * 4096 * 2 * 2
    hw = replace(HW, gpu_mem_bytes=one_ab)   # exactly one adapter (A+B)

    target = Adapter(adapter_id=0, rank=16, batch=512)
    state = SystemState(adapters={0: target}, kv_cache_bytes=one_ab)

    d = choose_strategy(0, state, hw, MODEL)
    print(f"\n[t8] budget tight, only KV evictable: chose {d}")
    _expect(d.strategy is Strategy.E1, "t8 picks E1", f"got {d}")
    _expect(d.evict_kv_bytes == one_ab, "t8 evicts full KV bytes",
            f"evict_kv={d.evict_kv_bytes}")


# ----------------------- strategy coverage tests (E1/E2/E3/E4) ---------

def _force_strategy(name: str, expected: Strategy, *,
                    rank: int, batch: int, gpu_mem_bytes: int) -> None:
    """Construct a single-adapter scenario that should force ``expected``."""
    hw = replace(HW, gpu_mem_bytes=gpu_mem_bytes)
    a = Adapter(adapter_id=0, rank=rank, batch=batch)
    state = SystemState(adapters={0: a})
    d = choose_strategy(0, state, hw, MODEL)
    costs = {E: cost_lora(E, a, HW, MODEL) for E in Strategy}
    sizes = {E: gpu_mem_for(E, a, MODEL) for E in Strategy}
    detail = (f"chose E{int(d.strategy)} | "
              f"costs: " + ", ".join(f"E{int(E)}={costs[E]:.0f}ns" for E in Strategy)
              + f" | sizes: "
              + ", ".join(f"E{int(E)}={sizes[E]}B" for E in Strategy)
              + f" | budget={gpu_mem_bytes}B")
    _expect(d.strategy is expected, name, detail)


def t9_force_e1() -> None:
    """E1 wins whenever it fits in GPU memory — no CXL traffic at all."""
    print("\n[t9] force E1 (abundant GPU memory)")
    # rank=16, gpu_mem big enough for both A and B
    _force_strategy("t9 E1 picked at small B",
                    Strategy.E1, rank=16, batch=1,
                    gpu_mem_bytes=10**9)
    _force_strategy("t9 E1 still picked at large B",
                    Strategy.E1, rank=16, batch=512,
                    gpu_mem_bytes=10**9)


def t10_force_e4() -> None:
    """E4 wins when E1 doesn't fit but E4 does (B matrix only) AND B < 2R
    so per-request link beats E2's constant 2RD link."""
    print("\n[t10] force E4 (mem fits only B, small B)")
    # E1 needs 2*R*D*S = 256 KB; E4 needs R*D*S = 128 KB. Budget = 128 KB exactly.
    one_b = 16 * 4096 * 2
    _force_strategy("t10 E4 at B=1 R=16",
                    Strategy.E4, rank=16, batch=1,
                    gpu_mem_bytes=one_b)
    _force_strategy("t10 E4 at B=8 R=16 (still B < 2R)",
                    Strategy.E4, rank=16, batch=8,
                    gpu_mem_bytes=one_b)


def t11_force_e3() -> None:
    """E3 wins when nothing fits in GPU AND B is small enough that per-request
    link (2BD) beats E2's constant link (2RD), i.e., B < R, AND E2's no-NDP
    advantage is dominated."""
    print("\n[t11] force E3 (no cache, B small)")
    # No GPU memory at all -> E1 and E4 infeasible. B=1 < R=16, so E3 wins.
    _force_strategy("t11 E3 at B=1 R=16 no mem",
                    Strategy.E3, rank=16, batch=1,
                    gpu_mem_bytes=0)
    _force_strategy("t11 E3 at B=1 R=128 no mem",
                    Strategy.E3, rank=128, batch=1,
                    gpu_mem_bytes=0)


def t12_force_e2() -> None:
    """E2 wins when nothing fits in GPU AND B > 2R, where E2's constant link
    (2RD) beats E3's per-request 2BD and E4's per-request B*(D+R)."""
    print("\n[t12] force E2 (no cache, large B)")
    # rank=16 -> 2R=32. Pick B=128 >> 2R.
    _force_strategy("t12 E2 at B=128 R=16 no mem",
                    Strategy.E2, rank=16, batch=128,
                    gpu_mem_bytes=0)
    # Even bigger ratio
    _force_strategy("t12 E2 at B=1024 R=16 no mem",
                    Strategy.E2, rank=16, batch=1024,
                    gpu_mem_bytes=0)


def t13_e2_vs_e3_crossover() -> None:
    """Verify the B ~ R crossover from the cost model: at B << R, E3 wins;
    at B >> R, E2 wins. Right around B = R is messy and we don't pin it."""
    print("\n[t13] E2/E3 crossover at no-cache")
    # R=32, no cache. B=4 << R should pick E3; B=512 >> R should pick E2.
    _force_strategy("t13 small B (<<R) -> E3",
                    Strategy.E3, rank=32, batch=4,
                    gpu_mem_bytes=0)
    _force_strategy("t13 large B (>>R) -> E2",
                    Strategy.E2, rank=32, batch=512,
                    gpu_mem_bytes=0)


# ----------------------- hot pool & pre-caching tests ------------------

def t14_classify_top_k() -> None:
    """top_k mode: top-K by temperature are hot, rest are cold."""
    print("\n[t14] classify_adapters with top_k=2")
    state = SystemState(adapters={
        1: Adapter(1, rank=8, batch=4, temperature=10.0),   # serving
        2: Adapter(2, rank=8, batch=0, temperature=5.0),    # hot (1st by temp)
        3: Adapter(3, rank=8, batch=0, temperature=1.0),    # cold
        4: Adapter(4, rank=8, batch=0, temperature=8.0),    # hot (highest temp)
        5: Adapter(5, rank=8, batch=0, temperature=2.0),    # cold
    })
    serving, hot, cold = classify_adapters(state, top_k=2)
    _expect({a.adapter_id for a in serving} == {1}, "t14 serving={1}")
    # Hot = adapters 4 (temp 8) and 2 (temp 5)
    _expect({a.adapter_id for a in hot} == {2, 4},
            f"t14 hot={{2,4}}, got {[a.adapter_id for a in hot]}")
    # Cold = 3, 5
    _expect({a.adapter_id for a in cold} == {3, 5},
            f"t14 cold={{3,5}}, got {[a.adapter_id for a in cold]}")


def t15_pre_cache_greedy_by_temp() -> None:
    """Hottest-first greedy fill of GPU cache.
    E1 footprint for a rank-8 adapter at n_layers=n_matrices=1:
       2 * R * D * S = 2 * 8 * 4096 * 2 = 131_072 B = 128 KiB.
    With cap = 3 * 128 KiB = 384 KiB, exactly 3 adapters fit."""
    print("\n[t15] pre_cache_hot_adapters (greedy by temp)")
    one_e1 = 2 * 8 * 4096 * 2          # = 131_072
    cap = 3 * one_e1                    # exactly 3 rank-8 adapters
    hw = replace(HW, gpu_mem_bytes=cap)
    state = SystemState(adapters={
        1: Adapter(1, rank=8, batch=0, temperature=5.0),
        2: Adapter(2, rank=8, batch=0, temperature=9.0),
        3: Adapter(3, rank=8, batch=0, temperature=1.0),
        4: Adapter(4, rank=8, batch=0, temperature=7.0),
        5: Adapter(5, rank=8, batch=0, temperature=3.0),
    })
    cached = pre_cache_hot_adapters(state, hw, MODEL, top_k=5,
                                    n_layers=1, n_matrices=1)
    # Hottest 3 (temps 9, 7, 5) fit -> ids 2, 4, 1 in that order
    _expect(cached == [2, 4, 1],
            f"t15 pre_cached order [2,4,1], got {cached}")
    _expect(state.adapters[2].a_in_gpu and state.adapters[2].b_in_gpu,
            "t15 adapter 2 cached (a+b)")
    _expect(not state.adapters[3].a_in_gpu,
            "t15 adapter 3 not cached (temp 1, lowest)")


def t16_pre_cache_reserves_serving() -> None:
    """A serving adapter's E1 footprint is reserved before filling the hot
    pool, so we don't over-allocate. With only enough space for 1 adapter
    AND a serving adapter present, the hot pool can't claim any space."""
    print("\n[t16] pre-cache reserves serving adapter memory")
    one_e1 = 8 * 4096 * 2 * 2
    hw = replace(HW, gpu_mem_bytes=one_e1)  # exactly one rank-8 E1
    state = SystemState(adapters={
        1: Adapter(1, rank=8, batch=4, temperature=5.0),   # serving
        2: Adapter(2, rank=8, batch=0, temperature=9.0),   # hottest non-serving
    })
    cached = pre_cache_hot_adapters(state, hw, MODEL, top_k=5,
                                    n_layers=1, n_matrices=1)
    # All capacity reserved for the serving adapter; hot pool gets nothing.
    _expect(cached == [],
            f"t16 hot pool empty (serving reserved all), got {cached}")
    _expect(not state.adapters[2].a_in_gpu,
            "t16 hot adapter not cached")


# ----------------------------------------------------------------- t17 MoE base

def t17_moe_base_time() -> None:
    """MoE base-model HBM time uses moe_active_base_bytes instead of the full
    base_model_bytes.

    Round-number fixture so the math is easy:
        gpu_compute = 1e15 FLOPS   (1 PFLOPS, makes compute negligible)
        gpu_mem_bw  = 1e12 B/s     (1 TB/s)
        n_layers    = 30
        D           = 4096
        batch       = 1

    Compute term (same in both cases):
        24 * 4096^2 * 1 / 1e15 * 1e9 ~= 402.65 ns   (<<< HBM, so result == hbm_ns)

    Dense (base=60 GB):
        bytes_per_layer = 60e9 / 30 = 2e9
        hbm_ns          = 2e9 / 1e12 * 1e9 = 2_000_000 ns  (2 ms)

    MoE   (active=6 GB):
        bytes_per_layer = 6e9 / 30 = 2e8
        hbm_ns          = 2e8 / 1e12 * 1e9 =   200_000 ns  (200 us)

    Expected ratio: dense / moe = 10x.
    """
    print("\n[t17] MoE active base bytes drive HBM term")
    hw = replace(HW, gpu_compute=1e15, gpu_mem_bw=1e12)
    model = ModelConfig(d=4096)

    dense = estimate_base_model_ns_per_layer(
        total_batch=1, model=model, hw=hw,
        base_model_bytes=60 * 10**9, n_layers=30)
    moe = estimate_base_model_ns_per_layer(
        total_batch=1, model=model, hw=hw,
        base_model_bytes=60 * 10**9,
        moe_active_base_bytes=6 * 10**9,
        n_layers=30)

    print(f"     dense (base=60 GB) -> {dense:>12,.1f} ns  (expect 2,000,000)")
    print(f"     moe   (active=6 GB) -> {moe:>12,.1f} ns  (expect   200,000)")

    _expect(abs(dense - 2_000_000.0) < 1.0,
            "t17 dense hbm_ns ~= 2,000,000",
            f"got {dense}")
    _expect(abs(moe - 200_000.0) < 1.0,
            "t17 moe hbm_ns ~= 200,000 (uses 6 GB not 60 GB)",
            f"got {moe}")
    _expect(abs(dense / moe - 10.0) < 1e-6,
            "t17 dense / moe == 10x",
            f"dense={dense}, moe={moe}, ratio={dense/moe}")


# ----------------------------------------------------------------- t18

def t18_gpu_attention_charged() -> None:
    """estimate_base_model_ns_per_layer charges Eq (7)'s GPU attention share.

    Round-number fixture (compute negligible, HBM-bound):
        gpu_mem_bw = 1e12 B/s, n_layers = 10, base = 10 GB
        -> base bytes/layer = 1e9 -> 1,000,000 ns
        kv_tokens_in_gpu = 61,440 with D=4096, S=2
        -> attention bytes/layer = 2*4096*2*61440 = 1,006,632,960
        -> per_layer = (1e9 + 1.00663e9) / 1e12 * 1e9 ~= 2,006,633 ns
    """
    print("\n[t18] GPU-side attention rides the base roofline")
    hw = replace(HW, gpu_compute=1e15, gpu_mem_bw=1e12)
    model = ModelConfig(d=4096)

    base_only = estimate_base_model_ns_per_layer(
        total_batch=1, model=model, hw=hw,
        base_model_bytes=10 * 10**9, n_layers=10)
    with_kv = estimate_base_model_ns_per_layer(
        total_batch=1, model=model, hw=hw,
        base_model_bytes=10 * 10**9, n_layers=10,
        kv_tokens_in_gpu=61440)

    _expect(abs(base_only - 1_000_000.0) < 1.0,
            "t18 base-only per-layer == 1 ms", f"got {base_only}")
    _expect(abs(with_kv - 2_006_632.96) < 1.0,
            "t18 +61,440 KV tokens adds their HBM bytes", f"got {with_kv}")
    _expect(estimate_base_model_ns_per_layer(
                total_batch=1, model=model, hw=hw,
                base_model_bytes=10 * 10**9, n_layers=10,
                kv_tokens_in_gpu=0) == base_only,
            "t18 kv_tokens_in_gpu=0 preserves old behavior")


# ----------------------------------------------------------------- t19

def t19_cost_aware_kv_fraction() -> None:
    """compute_kv_in_gpu_fraction balances Eq (7) vs Eq (8).

    All cases use the A100 fixture (base 14 GB / 32 layers -> 226 us/layer
    HBM floor) with one serving adapter on E3 (0 GPU bytes for LoRA).

      (a) short KV (18k tokens): CXL attention (~72 us/layer across 4
          devices) hides under the base floor -> duplicating any KV in GPU
          only adds GPU time -> P* == 0.
      (b) long KV (98,304 tokens): CXL side (~374 us/layer) exceeds the
          base floor -> P* > 0, analytically ~0.12, memory unconstrained.
      (c) tight memory (1 GB free): the unconstrained optimum exceeds the
          cap -> P* == cap.
      (d) no base info -> legacy greedy fill (== cap).
    """
    print("\n[t19] cost-aware KV duplication fraction")
    model = ModelConfig(d=4096)
    state = SystemState(adapters={1: Adapter(adapter_id=1, rank=8, batch=32)})
    decisions = {1: Decision(strategy=Strategy.E3, cost_ns=0.0)}
    kw = dict(n_layers=32, n_matrices=7,
              total_batch=32, base_model_bytes=14 * 10**9)

    p_short = compute_kv_in_gpu_fraction(
        decisions, state, HW, model, {1: 18_000}, **kw)
    _expect(p_short == 0.0,
            "t19a short KV -> P == 0 (CXL hides under base)",
            f"got {p_short}")

    p_long = compute_kv_in_gpu_fraction(
        decisions, state, HW, model, {1: 98_304}, **kw)
    _expect(0.05 < p_long < 0.20,
            "t19b long KV -> 0 < P < 0.2 (analytic ~0.12)",
            f"got {p_long}")

    hw_tight = replace(HW, gpu_mem_bytes=1 * 10**9)
    kv_total_bytes = 2 * 4096 * 2 * 98_304 * 32
    cap = (1 * 10**9) / kv_total_bytes
    p_capped = compute_kv_in_gpu_fraction(
        decisions, state, hw_tight, model, {1: 98_304}, **kw)
    _expect(abs(p_capped - cap) < 1e-9,
            "t19c tight memory -> P == cap",
            f"got {p_capped}, cap {cap}")

    p_legacy = compute_kv_in_gpu_fraction(
        decisions, state, hw_tight, model, {1: 98_304},
        n_layers=32, n_matrices=7)
    _expect(abs(p_legacy - cap) < 1e-9,
            "t19d no base info -> legacy greedy fill",
            f"got {p_legacy}, cap {cap}")


# ----------------------------------------------------------------- demo

def demo() -> None:
    """Narrated walkthrough: a batch grows over time for one adapter and the
    algorithm flips strategies. Useful for eyeballing the cost model."""
    print("\n" + "=" * 60)
    print("DEMO — strategy crossover as batch grows (rank=16, D=4096)")
    print("=" * 60)
    print(f"{'B':>5} | {'E1':>10} {'E2':>10} {'E3':>10} {'E4':>10} | best")
    print("-" * 60)
    for B in [1, 4, 16, 64, 128, 256, 512, 1024]:
        a = Adapter(0, rank=16, batch=B)
        costs = {E: cost_lora(E, a, HW, MODEL) for E in Strategy}
        best = min(costs, key=costs.get)
        print(f"{B:>5} | "
              f"{costs[Strategy.E1]:>10.0f} "
              f"{costs[Strategy.E2]:>10.0f} "
              f"{costs[Strategy.E3]:>10.0f} "
              f"{costs[Strategy.E4]:>10.0f} | E{int(best)}")


# ----------------------------------------------------------------- main

def main() -> int:
    print("CLoRA strategy plugin — tests")
    print("HW:", HW)
    print("Model:", MODEL)
    t1_costs_grow_with_batch()
    t2_small_batch_prefers_e3_over_e2()
    t3_large_batch_prefers_e1_over_e3()
    t4_fits_picks_e1()
    t5_evicts_cold_to_use_e1()
    t6_zero_budget_falls_back_to_zero_mem_strategy()
    t7_classify_adapters()
    t8_evict_kv_when_no_cold()
    t9_force_e1()
    t10_force_e4()
    t11_force_e3()
    t12_force_e2()
    t13_e2_vs_e3_crossover()
    t14_classify_top_k()
    t15_pre_cache_greedy_by_temp()
    t16_pre_cache_reserves_serving()
    t17_moe_base_time()
    t18_gpu_attention_charged()
    t19_cost_aware_kv_fraction()
    demo()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
