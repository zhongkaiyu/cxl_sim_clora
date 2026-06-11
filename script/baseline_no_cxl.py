#!/usr/bin/env python3
"""
baseline_no_cxl.py
==================

CLoRA-NoCXL baseline (a.k.a. PCIe-NDP). Isolates the value of the CXL
interface itself by removing CXL.mem / GPU load-store / read-compute
semantics, while keeping everything else identical to CLoRA:

  - same number of NDP devices, same NDP throughput
  - same remote DRAM capacity and bandwidth
  - same link bandwidth (default kept at CLoRA's 128 GB/s for fairness)
  - same Algorithm 1 strategy selection
  - same hot-adapter pre-cache / KV duplication

The *only* difference: the GPU cannot integrate the remote operation
inside a single kernel path. Every GPU<->NDP boundary requires

    (1) issuing a GPU kernel that produces the activation
    (2) explicit DMA of inputs
    (3) host/driver command to start the NDP op
    (4) synchronization wait
    (5) explicit DMA of outputs
    (6) relaunching the next dependent GPU kernel

So per offload boundary we pay:

    L_GPU_kernel_launch  +  L_device_command  +  L_sync

Following the reviewer's "be reviewer-friendly" framing we batch all
LoRA-touched matrices within a transformer layer into ONE offload call
(so a layer has at most one LoRA offload). Attention is a separate
offload because it depends on Q/K/V results. So default ``offloads_per_layer = 2``.

Step-time model
---------------

    T_NoCxl = T_CLoRA + (L_kernel + L_device + L_sync)
                       * offloads_per_layer * n_layers

This module is intentionally tiny: ~60 LOC, no CLoRA imports, no
state. It takes the CLoRA simulator's per-step time and returns the
NoCxl-augmented per-step time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NoCxlHWConfig:
    """Synchronization-overhead constants for the NoCxl baseline."""
    L_kernel_launch_ns:  float = 5000.0  # CUDA kernel launch (~5 us)
    L_device_command_ns: float = 1000.0  # CPU/driver issues NDP op (~1 us)
    L_sync_ns:           float = 1000.0  # stream/event sync (~1 us)
    offloads_per_layer:  int   = 2       # 1 LoRA (batched) + 1 attention


# Round-numbered default the driver can `replace(...)` from CLI flags.
DEFAULT_HW = NoCxlHWConfig()


def per_offload_ns(hw: NoCxlHWConfig) -> float:
    """Cost of one GPU<->NDP offload boundary."""
    return hw.L_kernel_launch_ns + hw.L_device_command_ns + hw.L_sync_ns


def no_cxl_overhead_ns(hw: NoCxlHWConfig, n_layers: int) -> int:
    """Total per-step host-side overhead introduced by removing CXL.

    This is purely additive on top of whatever the CLoRA C simulator
    already accounted for (link transfer, NDP compute, DRAM read).
    """
    if n_layers <= 0 or hw.offloads_per_layer <= 0:
        return 0
    return int(per_offload_ns(hw) * hw.offloads_per_layer * n_layers)


def no_cxl_step_ns(clora_step_ns: int,
                   hw: NoCxlHWConfig,
                   n_layers: int) -> int:
    """Augment a CLoRA-simulator step time with the NoCxl overhead."""
    return int(clora_step_ns) + no_cxl_overhead_ns(hw, n_layers)
