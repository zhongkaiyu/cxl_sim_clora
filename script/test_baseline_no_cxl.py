#!/usr/bin/env python3
"""
test_baseline_no_cxl.py
=======================

Hand-derived microtests for ``baseline_no_cxl``.  Every expected value
in this file is a small integer computation a reviewer can re-derive
with a calculator.

Run:  python3 script/test_baseline_no_cxl.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from baseline_no_cxl import (  # noqa: E402
    NoCxlHWConfig,
    launches_per_adapter,
    launches_per_layer,
    no_cxl_overhead_ns,
    no_cxl_step_ns,
    per_offload_ns,
)


# --------------------------------------------------------------- harness

_PASS = 0
_FAIL = 0


def _expect_eq(actual, expected, name: str) -> None:
    global _PASS, _FAIL
    if actual == expected:
        _PASS += 1
        print(f"  PASS  {name}  ({actual})")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}: got {actual}, expected {expected}")


# --------------------------------------------------------------- tests

HW = NoCxlHWConfig()  # defaults: 7000 ns/launch; per adapter QKV1+O1+FFN3=5


def t1_per_offload_default() -> None:
    """5 us kernel + 1 us device + 1 us sync = 7,000 ns per launch."""
    print("\n[t1] per_offload_ns at defaults")
    _expect_eq(per_offload_ns(HW), 7000.0, "t1 per_offload = 7000 ns")


def t2_launches_per_adapter() -> None:
    """QKV(1) + O(1) + FFN(3 SwiGLU) = 5 launches per adapter per layer."""
    print("\n[t2] launches_per_adapter (SwiGLU FFN=3)")
    _expect_eq(launches_per_adapter(HW), 5, "t2 per-adapter launches = 5")


def t3_launches_per_layer() -> None:
    """30 adapters * 5 + 32 requests * 1 = 182 launches per layer."""
    print("\n[t3] launches_per_layer with 30 adapters, 32 requests")
    _expect_eq(launches_per_layer(HW, n_serving_adapters=30, n_requests=32),
               30 * 5 + 32, "t3 launches/layer = 182")


def t4_overhead_full() -> None:
    """182 launches/layer * 32 layers * 7000 ns = 40,768,000 ns."""
    print("\n[t4] overhead: 30 adapters, 32 req, 32 layers")
    _expect_eq(no_cxl_overhead_ns(HW, n_layers=32,
                                  n_serving_adapters=30, n_requests=32),
               int((30 * 5 + 32) * 32 * 7000), "t4 overhead = 40,768,000 ns")


def t5_moe_ffn_gemms() -> None:
    """Qwen MoE: FFN = 3 gemms * 8 experts = 24 -> 26 launches per adapter."""
    print("\n[t5] ffn_gemms=24 (MoE) -> 26 launches per adapter")
    hw_moe = replace(HW, ffn_gemms_per_adapter=24)
    _expect_eq(launches_per_adapter(hw_moe), 26, "t5 MoE per-adapter = 26")


def t6_step_total_is_additive() -> None:
    """no_cxl_step_ns == clora_step_ns + overhead."""
    print("\n[t6] no_cxl_step_ns adds overhead on top of CLoRA")
    clora_step = 7_235_142
    expected = clora_step + int((30 * 5 + 32) * 32 * 7000)
    _expect_eq(no_cxl_step_ns(clora_step, HW, n_layers=32,
                              n_serving_adapters=30, n_requests=32),
               expected, "t6 total = clora + overhead")


def t7_zero_layers_zero_overhead() -> None:
    """Edge case."""
    print("\n[t7] n_layers=0 -> zero overhead")
    _expect_eq(no_cxl_overhead_ns(HW, n_layers=0,
                                  n_serving_adapters=30, n_requests=32), 0,
               "t7 zero layers -> zero overhead")


def t8_scales_with_adapters_and_requests() -> None:
    """Overhead is linear in both adapter count and request count."""
    print("\n[t8] linear in adapters and requests")
    base = no_cxl_overhead_ns(HW, n_layers=1, n_serving_adapters=10,
                              n_requests=10)
    dbl_ad = no_cxl_overhead_ns(HW, n_layers=1, n_serving_adapters=20,
                                n_requests=10)
    dbl_req = no_cxl_overhead_ns(HW, n_layers=1, n_serving_adapters=10,
                                 n_requests=20)
    # adapters contribute 5/launch, requests 1/launch: doubling adapters adds
    # 10*5*7000; doubling requests adds 10*1*7000.
    _expect_eq(dbl_ad - base, 10 * 5 * 7000, "t8 +10 adapters = +350k ns")
    _expect_eq(dbl_req - base, 10 * 1 * 7000, "t8 +10 requests = +70k ns")


def t9_aggressive_sync() -> None:
    """What if L_sync is 5 us instead of 1 us?"""
    print("\n[t9] L_sync_ns = 5000 -> per_offload = 11,000 ns")
    hw_slow = replace(HW, L_sync_ns=5000.0)
    _expect_eq(per_offload_ns(hw_slow), 11000.0, "t9 per_offload = 11k ns")


# --------------------------------------------------------------- main

def main() -> int:
    print("baseline_no_cxl tests")
    print("HW (defaults):", HW)
    t1_per_offload_default()
    t2_launches_per_adapter()
    t3_launches_per_layer()
    t4_overhead_full()
    t5_moe_ffn_gemms()
    t6_step_total_is_additive()
    t7_zero_layers_zero_overhead()
    t8_scales_with_adapters_and_requests()
    t9_aggressive_sync()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
