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
inside a single fine-grained load/store path. Every GPU<->NDP operation
must be its own kernel launch + device command + sync. Crucially, without
CXL.mem you cannot fuse the work of *different adapters* into one kernel,
so the launch count scales with the number of serving adapters (not a
flat per-layer constant).

Per decoder layer the offloaded operations are:

    Proj QKV   : 1 launch per serving adapter   (fused Q,K,V LoRA)
    Proj O     : 1 launch per serving adapter
    FFN        : ffn_gemms launches per adapter  (SwiGLU gate/up/down = 3,
                                                  classic up/down = 2)
    Attention  : 1 launch per request            (each request attends over
                                                  its own KV cache)

So:

    launches_per_layer = n_adapters * (1 + 1 + ffn_gemms) + n_requests
    overhead_ns        = launches_per_layer * n_layers * per_offload_ns

where per_offload_ns = L_kernel_launch + L_device_command + L_sync is the
cost of one GPU<->NDP round trip (the thing CXL.mem's load/store avoids).

The overhead is purely additive on top of the CLoRA C-simulator step
(which already models the link transfer / NDP compute / DRAM access):

    T_NoCxl = T_CLoRA + overhead_ns

This module is intentionally tiny: no CLoRA imports, no state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NoCxlHWConfig:
    """Per-launch latencies and per-operation launch counts."""
    # one GPU<->NDP offload boundary = launch + device command + sync
    L_kernel_launch_ns:  float = 5000.0  # CUDA kernel launch (~5 us)
    L_device_command_ns: float = 1000.0  # CPU/driver issues NDP op (~1 us)
    L_sync_ns:           float = 1000.0  # stream/event sync (~1 us)
    # launch counts per decoder layer
    qkv_launches_per_adapter: int = 1    # fused Q,K,V projection LoRA
    o_launches_per_adapter:   int = 1    # output projection LoRA
    ffn_gemms_per_adapter:    int = 3    # FFN GEMMs (SwiGLU=3, classic=2)
    attn_launches_per_request: int = 1   # attention, one per request
    # fused=False : NDP-over-PCIe can't fuse remote per-adapter offloads, so a
    #               launch per adapter (projections/FFN) and per request (attn).
    # fused=True  : S-LoRA/BGMV-style batching — one launch per *op* per layer,
    #               independent of adapter/request count.
    fused: bool = False


# Round-numbered default the driver can `replace(...)` from CLI flags.
DEFAULT_HW = NoCxlHWConfig()


def per_offload_ns(hw: NoCxlHWConfig) -> float:
    """Cost of one GPU<->NDP offload boundary (one kernel launch)."""
    return hw.L_kernel_launch_ns + hw.L_device_command_ns + hw.L_sync_ns


def launches_per_adapter(hw: NoCxlHWConfig) -> int:
    """Per-layer LoRA kernel launches for one serving adapter."""
    return (hw.qkv_launches_per_adapter
            + hw.o_launches_per_adapter
            + hw.ffn_gemms_per_adapter)


def launches_per_layer(hw: NoCxlHWConfig,
                       n_serving_adapters: int,
                       n_requests: int) -> int:
    """Total GPU<->NDP offload boundaries in one decoder layer.

    fused=False: launch per adapter (LoRA) + per request (attention).
    fused=True : one launch per op + one batched attention launch (BGMV).
    """
    if hw.fused:
        return launches_per_adapter(hw) + hw.attn_launches_per_request
    return (n_serving_adapters * launches_per_adapter(hw)
            + n_requests * hw.attn_launches_per_request)


def no_cxl_overhead_ns(hw: NoCxlHWConfig,
                       n_layers: int,
                       n_serving_adapters: int,
                       n_requests: int) -> int:
    """Total per-step host-side overhead introduced by removing CXL.

    Purely additive on top of whatever the CLoRA C simulator already
    accounted for (link transfer, NDP compute, DRAM access).
    """
    if n_layers <= 0:
        return 0
    lpl = launches_per_layer(hw, n_serving_adapters, n_requests)
    return int(lpl * n_layers * per_offload_ns(hw))


def no_cxl_step_ns(clora_step_ns: int,
                   hw: NoCxlHWConfig,
                   n_layers: int,
                   n_serving_adapters: int,
                   n_requests: int) -> int:
    """Augment a CLoRA-simulator step time with the NoCxl launch overhead."""
    return int(clora_step_ns) + no_cxl_overhead_ns(
        hw, n_layers, n_serving_adapters, n_requests)
