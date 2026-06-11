#!/usr/bin/env python3
"""
baseline_grace_hopper.py
========================

Grace-Hopper offload baseline (NVLink-C2C between Grace CPU and Hopper GPU).

This is a *strong* CPU-memory baseline that addresses reviewer A's question
"would a tightly-coupled CPU-GPU coherent interconnect like NVLink-C2C make
CXL+NDP unnecessary?"

What it models
--------------

  GPU (H100):
    - base model weights and base model computation
    - LoRA adapter LRU cache (E1-style hits)
    - LoRA matmul itself (NOT on CPU -- this is the key difference vs
      our CPU-LoRA-compute baseline)
    - Attention computation

  Grace CPU memory:
    - LoRA adapters that don't fit in the GPU LRU
    - KV cache spillover when the GPU can't hold it

  NVLink-C2C:
    - 450 GB/s per direction (NVIDIA reports 900 GB/s bidirectional)
    - Coherent; no explicit DMA / sync per op as in PCIe; the GPU can
      stream A,B from CPU memory during the matmul

Cost model (per decode step)
----------------------------

For each cache-missed adapter (per layer × matrix × rank-group when fused):

    t_c2c_transfer    = L_c2c_setup + bytes(A,B) / c2c_bw
    t_gpu_lora_compute = 4 * B * D * R / C_GPU
    -- transfer and compute can overlap via coherent access, so take max
    t_kernel          = L_kernel_launch    (per fused op)

Cache hits (E1 analog): pure GPU compute, no transfer.

Attention: KV in CPU memory, transferred over C2C per layer; GPU does
the matmul. Same overlap.

Base model: GPU compute or HBM-bound, max() per layer.

Per-layer parallelism: GPU base + cached LoRA overlaps with missed-LoRA
C2C-transfer + GPU compute on a separate stream; attention follows.

This module reuses ``LRUAdapterCache`` from ``baseline_cpu_offload`` for
the LoRA cache. The only thing structurally different from the CPU-LoRA
baseline is *who runs the matmul*: GPU at ~989 TFLOPS instead of CPU at
~500 GFLOPS, plus the C2C link replaces PCIe.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from clora_strategy import ModelConfig
from baseline_cpu_offload import LRUAdapterCache  # reuse


# --------------------------------------------------------------- HW config


@dataclass
class GraceHopperHWConfig:
    """Hardware constants for the Grace-Hopper offload baseline."""
    gpu_compute:        float                # FLOPS (H100 SXM ~989 TFLOPS)
    gpu_mem_bw:         float                # B/s, HBM3
    gpu_mem_bytes:      int                  # LoRA cache budget
    c2c_bw:             float                # B/s, NVLink-C2C (450e9)
    L_kernel_launch_ns: float = 5000.0       # CUDA kernel launch
    L_c2c_setup_ns:     float = 500.0        # NVLink-C2C setup (vs 1000ns PCIe)


# Default fixture: Grace-Hopper GH200 superchip.
#   - gpu_compute  = 989 TFLOPS: H100 SXM FP16 dense
#   - gpu_mem_bw   = 3350 GB/s : H100 SXM HBM3
#   - c2c_bw       = 450  GB/s : NVLink-C2C per direction (900 bidirectional)
#   - L_c2c_setup  = 0.5 us    : lower than PCIe due to coherent access
DEFAULT_HW = GraceHopperHWConfig(
    gpu_compute=989e12,
    gpu_mem_bw=3350e9,
    gpu_mem_bytes=8 * 10**9,
    c2c_bw=450e9,
)


# --------------------------------------------------------------- step model


def grace_hopper_step_ns(
    adapters: Dict[int, Tuple[int, int]],          # id -> (rank, batch)
    kv_tokens_per_adapter: Dict[int, int],
    hw: GraceHopperHWConfig,
    model: ModelConfig,
    cache: LRUAdapterCache,
    *,
    base_model_bytes: int = 14 * 10**9,            # Llama2-7B FP16
    n_layers: int = 32,
    n_matrices: int = 7,
    fusion: str = "fused",
    parallel: bool = True,
) -> dict:
    """One decode step on the Grace-Hopper offload baseline.

    Mirrors the structure of ``baseline_cpu_offload.baseline_step_ns`` but
    with the LoRA matmul running on the GPU and the link being C2C instead
    of PCIe + CPU.
    """
    if fusion not in ("fused", "unfused"):
        raise ValueError(f"unknown fusion: {fusion}")

    D, S = model.d, model.s_dtype
    total_batch = sum(b for _, b in adapters.values())

    # Touch cache; partition into hits / misses
    hit_ids, miss_ids = [], []
    for aid, (rank, _) in adapters.items():
        (hit_ids if cache.touch(aid, rank) else miss_ids).append(aid)

    # ---- Cached LoRA path (E1 analog: pure GPU compute) ----
    cached_flops = sum(
        4 * adapters[aid][1] * D * adapters[aid][0]
        for aid in hit_ids) * n_layers * n_matrices
    t_cached_gpu = cached_flops / hw.gpu_compute * 1e9 \
        if hw.gpu_compute > 0 else 0.0

    # ---- Missed LoRA: load A,B over C2C; GPU computes (overlap) ----
    t_c2c_transfer = 0.0
    t_gpu_lora_compute = 0.0
    n_kernels_lora = 0

    if fusion == "fused":
        by_rank = defaultdict(lambda: {"batch": 0, "n_adapters": 0})
        for aid in miss_ids:
            r, b = adapters[aid]
            by_rank[r]["batch"]      += b
            by_rank[r]["n_adapters"] += 1

        for rank, info in by_rank.items():
            agg_batch = info["batch"]
            n_ad      = info["n_adapters"]
            # Transfer A,B for every adapter in this rank group, once per
            # layer × matrix. C2C is coherent so the GPU streams them.
            bytes_per_op = 2 * rank * D * S * n_ad
            t_c2c_transfer += (hw.L_c2c_setup_ns
                               + bytes_per_op / hw.c2c_bw * 1e9
                               ) * n_layers * n_matrices
            # GPU LoRA matmul: 4 * agg_batch * D * R per layer per matrix
            gpu_flops = 4 * agg_batch * D * rank * n_layers * n_matrices
            t_gpu_lora_compute += gpu_flops / hw.gpu_compute * 1e9
            # 1 fused kernel per (layer, matrix, rank); just 1 launch each.
            n_kernels_lora += n_layers * n_matrices
    else:  # unfused
        for aid in miss_ids:
            rank, batch = adapters[aid]
            bytes_per_op = 2 * rank * D * S
            t_c2c_transfer += (hw.L_c2c_setup_ns
                               + bytes_per_op / hw.c2c_bw * 1e9
                               ) * n_layers * n_matrices
            gpu_flops = 4 * batch * D * rank * n_layers * n_matrices
            t_gpu_lora_compute += gpu_flops / hw.gpu_compute * 1e9
            n_kernels_lora += n_layers * n_matrices

    t_kernels_lora = n_kernels_lora * hw.L_kernel_launch_ns
    # C2C transfer and GPU compute overlap (coherent access); take max.
    t_missed_lora = max(t_c2c_transfer, t_gpu_lora_compute) + t_kernels_lora

    # ---- Base model on GPU (per layer, max(compute, HBM)) ----
    base_compute_per_layer = 24.0 * D * D * total_batch / hw.gpu_compute * 1e9
    base_hbm_per_layer = (base_model_bytes / max(n_layers, 1)) \
        / hw.gpu_mem_bw * 1e9 if hw.gpu_mem_bw > 0 else 0.0
    base_per_layer = max(base_compute_per_layer, base_hbm_per_layer)
    t_base = base_per_layer * n_layers

    # ---- Attention: KV in CPU memory, transferred per layer ----
    total_kv = sum(kv_tokens_per_adapter.values())
    if total_kv > 0 and total_batch > 0:
        # KV (K + V) read from CPU memory over C2C per layer
        kv_bytes_per_layer = 2 * D * S * total_kv
        t_attn_transfer = (hw.L_c2c_setup_ns
                           + kv_bytes_per_layer / hw.c2c_bw * 1e9
                           ) * n_layers
        # GPU computes attention (Q*K^T + softmax + S*V)
        attn_flops = 4 * D * total_kv * n_layers
        t_attn_compute = attn_flops / hw.gpu_compute * 1e9
        # Kernel launches (1 attention kernel per layer)
        t_attn_kernels = n_layers * hw.L_kernel_launch_ns
        # Transfer and compute overlap
        t_attn = max(t_attn_transfer, t_attn_compute) + t_attn_kernels
    else:
        t_attn = 0.0

    # ---- Per-layer parallelism (CUDA streams) ----
    cached_per_layer  = t_cached_gpu       / n_layers
    missed_per_layer  = t_missed_lora      / n_layers
    attn_per_layer    = t_attn             / n_layers
    gpu_path_per_layer = base_per_layer + cached_per_layer

    if parallel:
        # GPU base + cached LoRA overlaps with C2C-load + GPU LoRA compute
        # on a separate stream; attention follows.
        t_per_layer = max(gpu_path_per_layer, missed_per_layer) + attn_per_layer
    else:
        # Worst-case: everything within a layer serializes.
        t_per_layer = gpu_path_per_layer + missed_per_layer + attn_per_layer
    total_ns = t_per_layer * n_layers

    return {
        "total_ns":            total_ns,
        "t_base":              t_base,
        "t_cached_gpu_lora":   t_cached_gpu,
        "t_c2c_transfer":      t_c2c_transfer,
        "t_gpu_lora_compute":  t_gpu_lora_compute,
        "t_kernels_lora":      t_kernels_lora,
        "t_attn":              t_attn,
        "base_per_layer":      base_per_layer,
        "missed_per_layer":    missed_per_layer,
        "gpu_path_per_layer":  gpu_path_per_layer,
        "t_per_layer":         t_per_layer,
        "n_cached":            len(hit_ids),
        "n_missed":            len(miss_ids),
        "n_kernels_lora":      n_kernels_lora,
    }
