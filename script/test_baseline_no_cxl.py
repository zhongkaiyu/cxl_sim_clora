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

HW = NoCxlHWConfig()  # defaults: 5000 + 1000 + 1000 = 7000 ns per offload


def t1_per_offload_default() -> None:
    """5 us kernel + 1 us device + 1 us sync = 7,000 ns."""
    print("\n[t1] per_offload_ns at defaults")
    _expect_eq(per_offload_ns(HW), 7000.0, "t1 per_offload = 7000 ns")


def t2_overhead_2_offloads_32_layers() -> None:
    """7000 ns * 2 offloads * 32 layers = 448,000 ns total per step."""
    print("\n[t2] overhead with default 2 offloads/layer, n_layers=32")
    _expect_eq(no_cxl_overhead_ns(HW, n_layers=32),
               int(7000 * 2 * 32), "t2 overhead = 448,000 ns")


def t3_overhead_scales_with_n_layers() -> None:
    """Linear in n_layers."""
    print("\n[t3] overhead scales linearly in n_layers")
    _expect_eq(no_cxl_overhead_ns(HW, n_layers=40),
               int(7000 * 2 * 40), "t3 overhead at 40 layers = 560,000 ns")


def t4_per_layer_offloads_knob() -> None:
    """offloads_per_layer = 1 halves the overhead."""
    print("\n[t4] offloads_per_layer = 1 (no separate attention offload)")
    hw1 = replace(HW, offloads_per_layer=1)
    _expect_eq(no_cxl_overhead_ns(hw1, n_layers=32),
               int(7000 * 1 * 32), "t4 overhead with 1 offload/layer = 224,000 ns")


def t5_step_total_is_additive() -> None:
    """no_cxl_step_ns == clora_step_ns + overhead."""
    print("\n[t5] no_cxl_step_ns adds overhead on top of CLoRA")
    clora_step = 7_235_142
    expected = clora_step + 7000 * 2 * 32
    _expect_eq(no_cxl_step_ns(clora_step, HW, n_layers=32),
               expected, "t5 total = clora + overhead")


def t6_zero_layers_zero_overhead() -> None:
    """Edge case."""
    print("\n[t6] n_layers=0 -> zero overhead")
    _expect_eq(no_cxl_overhead_ns(HW, n_layers=0), 0,
               "t6 zero layers -> zero overhead")


def t7_zero_offloads_zero_overhead() -> None:
    """offloads_per_layer=0 disables the baseline (degenerates to CLoRA)."""
    print("\n[t7] offloads_per_layer=0 -> zero overhead")
    hw0 = replace(HW, offloads_per_layer=0)
    _expect_eq(no_cxl_overhead_ns(hw0, n_layers=32), 0,
               "t7 zero offloads -> zero overhead")
    _expect_eq(no_cxl_step_ns(1_234_567, hw0, n_layers=32), 1_234_567,
               "t7 step == clora when no offloads")


def t8_aggressive_sync() -> None:
    """Reviewer's question: what if L_sync is 5 us instead of 1 us?"""
    print("\n[t8] L_sync_ns = 5000 -> per_offload = 11,000 ns")
    hw_slow = replace(HW, L_sync_ns=5000.0)
    _expect_eq(per_offload_ns(hw_slow), 11000.0, "t8 per_offload = 11k ns")


# --------------------------------------------------------------- main

def main() -> int:
    print("baseline_no_cxl tests")
    print("HW (defaults):", HW)
    t1_per_offload_default()
    t2_overhead_2_offloads_32_layers()
    t3_overhead_scales_with_n_layers()
    t4_per_layer_offloads_knob()
    t5_step_total_is_additive()
    t6_zero_layers_zero_overhead()
    t7_zero_offloads_zero_overhead()
    t8_aggressive_sync()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
