#!/usr/bin/env python3
"""
baseline_cpu_offload.py
=======================

PCIe + CPU-compute offload baseline for multi-LoRA serving.

This is the structural analog of CLoRA with the CXL+NDP path replaced by a
PCIe link + a CPU acting as the off-device compute engine. It has nothing to
do with the CLoRA C simulator; it's a self-contained analytical model that
the driver can run alongside CLoRA to produce side-by-side numbers.

Per decode step we model, for each (active adapter, layer, matrix):

  IF adapter cached in GPU memory:
      GPU runs y = x A B locally (E1 analog).  Cost: 4 B D R FLOPs on GPU.

  ELSE (cache miss):
      kernel launch + cudaMemcpyAsync : send  x  to CPU buffer  (PCIe down)
      CPU:                              load A,B from CPU DRAM, compute y=xAB
      kernel launch + cudaMemcpyAsync : recv  y  from CPU       (PCIe up)

      Costs:
        PCIe time     = (L_pcie_setup + bytes / pcie_bw)
        CPU compute   = FLOPs / cpu_compute   (CPU vectorised matmul)
        CPU DRAM      = (2 R D S) / cpu_dram_bw   per matrix per layer
        kernel cost   = 2 * (L_kernel_launch + L_cpu_gpu_offload)

Plus base-model GPU compute (max of FLOPs/C_GPU and base_bytes/W_HBM).
Plus attention: KV in CPU DRAM, Q/result shuttled over PCIe per layer.

Fusion modes:
  "fused"   -- adapters of the same rank fire one batched kernel pair per
               (layer, matrix). Models S-LoRA / PUNICA style BGMV.
  "unfused" -- one kernel pair per adapter per (layer, matrix). Strawman.

Defaults are chosen so a cold rank-16 adapter has a reasonable cost; see
test_baseline.py for hand-derived spot checks.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# We piggy-back on ModelConfig for D and s_dtype; nothing else from clora.
from clora_strategy import ModelConfig


# --------------------------------------------------------------- HW config

@dataclass
class BaselineHWConfig:
    """Hardware constants for the PCIe + CPU-offload baseline."""
    # GPU side
    gpu_compute:     float     # FLOPS for base-model matmuls
    gpu_mem_bw:      float     # B/s, HBM
    gpu_mem_bytes:   int       # bytes free for LoRA cache (+ optional KV dup)
    # CPU side
    cpu_compute:     float     # FLOPS, CPU matmul throughput
    cpu_dram_bw:     float     # B/s, CPU DRAM
    # Link
    pcie_bw:         float     # B/s, PCIe 4.0 x16 effective ~ 32e9
    L_pcie_setup:    float = 1000.0      # ns per DMA op (1 us)
    L_kernel_launch: float = 5000.0      # ns per CUDA kernel launch (5 us)
    L_cpu_gpu_offload: float = 10000.0   # ns compound per GPU<->CPU round
                                          # (sync + dep kernel launch)


# Default fixture: H100-era node, modern Xeon, PCIe 5.0.
#
# Each constant has a brief justification so reviewers can challenge it.
#  - gpu_compute  = 989 TFLOPS: H100 SXM FP16 dense peak.
#  - gpu_mem_bw   = 3350 GB/s: H100 SXM HBM3.
#  - cpu_compute  = 500 GFLOPS: typical FP16/BF16 matmul throughput on a single
#                   modern server CPU (~32 cores, AVX-512, oneDNN). With Intel
#                   AMX you can reach 1.5-2 TFLOPS; without, ~500 GFLOPS is a
#                   reasonable middle estimate.
#  - cpu_dram_bw  = 200 GB/s: 8-channel DDR4-3200 ~ 205 GB/s sustained.
#                   DDR5 servers hit ~300 GB/s.
#  - pcie_bw      = 58 GB/s: PCIe 5.0 x16 has 64 GB/s raw, ~58 GB/s effective
#                   after protocol overhead. (A100 was PCIe 4 at 28; H100 is
#                   PCIe 5 at 58.) Override with --pcie-bw-gb on the driver.
DEFAULT_HW = BaselineHWConfig(
    gpu_compute=989e12,
    gpu_mem_bw=3350e9,
    gpu_mem_bytes=8 * 10**9,
    cpu_compute=500e9,
    cpu_dram_bw=200e9,
    pcie_bw=58e9,
)


# --------------------------------------------------------------- LRU cache

def adapter_full_bytes(rank: int, model: ModelConfig,
                       n_layers: int, n_matrices: int) -> int:
    """Bytes for one adapter's full set of A,B across all layers and matrices."""
    return 2 * rank * model.d * model.s_dtype * n_layers * n_matrices


class LRUAdapterCache:
    """Per-adapter LRU cache in GPU memory.

    Entry == one adapter, fully populated (all layers / matrices of A,B).
    On miss, we evict the least-recently-used until the new entry fits.
    """

    def __init__(self, capacity_bytes: int, model: ModelConfig,
                 *, n_layers: int = 32, n_matrices: int = 7):
        self.capacity_bytes = max(0, capacity_bytes)
        self.model = model
        self.n_layers = n_layers
        self.n_matrices = n_matrices
        self._order: List[int] = []          # oldest first
        self._stored: Dict[int, int] = {}    # id -> bytes
        self.n_hits = 0
        self.n_misses = 0

    @property
    def used_bytes(self) -> int:
        return sum(self._stored.values())

    def touch(self, adapter_id: int, rank: int) -> bool:
        """True on hit, False on miss (then insert and possibly evict)."""
        if adapter_id in self._stored:
            self._order.remove(adapter_id)
            self._order.append(adapter_id)
            self.n_hits += 1
            return True

        self.n_misses += 1
        needed = adapter_full_bytes(rank, self.model,
                                    self.n_layers, self.n_matrices)

        while (self.used_bytes + needed > self.capacity_bytes
               and self._order):
            ev = self._order.pop(0)
            del self._stored[ev]

        if needed <= self.capacity_bytes:
            self._stored[adapter_id] = needed
            self._order.append(adapter_id)
        return False

    def snapshot(self) -> dict:
        return {
            "n_cached": len(self._stored),
            "used_bytes": self.used_bytes,
            "n_hits": self.n_hits,
            "n_misses": self.n_misses,
        }


# --------------------------------------------------------------- step model

def baseline_step_ns(
    adapters: Dict[int, Tuple[int, int]],          # id -> (rank, batch)
    kv_tokens_per_adapter: Dict[int, int],
    hw: BaselineHWConfig,
    model: ModelConfig,
    cache: LRUAdapterCache,
    *,
    base_model_bytes: int = 14 * 10**9,            # Llama2-7B FP16
    n_layers: int = 32,
    n_matrices: int = 7,
    fusion: str = "fused",
    parallel: bool = True,
) -> dict:
    """Cost of one decode step on the PCIe+CPU baseline. Cache is mutated."""
    if fusion not in ("fused", "unfused"):
        raise ValueError(f"unknown fusion: {fusion}")

    D, S = model.d, model.s_dtype
    total_batch = sum(b for _, b in adapters.values())

    # Touch cache, partition into hit / miss
    hit_ids, miss_ids = [], []
    for aid, (rank, _) in adapters.items():
        (hit_ids if cache.touch(aid, rank) else miss_ids).append(aid)

    # ---- Cached LoRA path (E1 equivalent: pure GPU compute) ----
    cached_flops = sum(
        4 * adapters[aid][1] * D * adapters[aid][0]
        for aid in hit_ids) * n_layers * n_matrices
    t_cached_gpu = cached_flops / hw.gpu_compute * 1e9 \
        if hw.gpu_compute > 0 else 0.0

    # ---- Missed LoRA path: PCIe round-trip + CPU compute + kernels ----
    t_pcie_lora    = 0.0
    t_cpu_compute  = 0.0
    t_cpu_dram     = 0.0
    n_kernel_pairs = 0

    if fusion == "fused":
        by_rank = defaultdict(lambda: {"batch": 0, "n_adapters": 0})
        for aid in miss_ids:
            r, b = adapters[aid]
            by_rank[r]["batch"]      += b
            by_rank[r]["n_adapters"] += 1

        for rank, info in by_rank.items():
            agg_batch = info["batch"]
            n_ad      = info["n_adapters"]
            # PCIe: x out, y back (each batch * D * S bytes)
            x_bytes = agg_batch * D * S
            y_bytes = agg_batch * D * S
            per_pcie = (2 * hw.L_pcie_setup
                        + (x_bytes + y_bytes) / hw.pcie_bw * 1e9)
            t_pcie_lora += per_pcie * n_layers * n_matrices
            # CPU compute: 4 * batch * D * R per matrix per layer
            cpu_flops = 4 * agg_batch * D * rank * n_layers * n_matrices
            t_cpu_compute += cpu_flops / hw.cpu_compute * 1e9
            # CPU DRAM: 2 * R * D * S bytes per matrix per layer per adapter
            dram_bytes = 2 * rank * D * S * n_ad * n_layers * n_matrices
            t_cpu_dram += dram_bytes / hw.cpu_dram_bw * 1e9
            # 1 fused op per (layer, matrix, rank); each op fires 2 launches.
            n_kernel_pairs += n_layers * n_matrices
    else:  # unfused
        for aid in miss_ids:
            rank, batch = adapters[aid]
            x_bytes = batch * D * S
            y_bytes = batch * D * S
            per_pcie = (2 * hw.L_pcie_setup
                        + (x_bytes + y_bytes) / hw.pcie_bw * 1e9)
            t_pcie_lora += per_pcie * n_layers * n_matrices
            cpu_flops = 4 * batch * D * rank * n_layers * n_matrices
            t_cpu_compute += cpu_flops / hw.cpu_compute * 1e9
            dram_bytes = 2 * rank * D * S * n_layers * n_matrices
            t_cpu_dram += dram_bytes / hw.cpu_dram_bw * 1e9
            n_kernel_pairs += n_layers * n_matrices

    t_cpu_path = max(t_cpu_compute, t_cpu_dram)
    t_kernels_lora = (2 * n_kernel_pairs) * \
        (hw.L_kernel_launch + hw.L_cpu_gpu_offload)

    # ---- Base model on GPU (per layer, taken correctly) ----
    # Per layer the GPU does ~24 D^2 batch FLOPs (Q,K,V,O,FFN-G/U/out) and
    # has to read its slice of base weights from HBM. Per layer time is the
    # max of compute and HBM load.  Total t_base = per_layer_time * n_layers.
    base_compute_per_layer_ns = 24.0 * D * D * total_batch \
        / hw.gpu_compute * 1e9
    base_hbm_per_layer_ns = (base_model_bytes / max(n_layers, 1)) \
        / hw.gpu_mem_bw * 1e9 if hw.gpu_mem_bw > 0 else 0.0
    base_per_layer = max(base_compute_per_layer_ns, base_hbm_per_layer_ns)
    t_base = base_per_layer * n_layers

    # ---- Attention: KV in CPU DRAM, transferred over PCIe per layer ----
    total_kv = sum(kv_tokens_per_adapter.values())
    if total_kv > 0 and total_batch > 0:
        # Q out and result back, batched per layer.
        attn_pcie_bytes = 2 * total_batch * D * S * n_layers
        attn_pcie_setup = 2 * hw.L_pcie_setup * n_layers
        t_attn_pcie = attn_pcie_setup + attn_pcie_bytes / hw.pcie_bw * 1e9
        # CPU compute = 4 * D * total_kv  per layer (Q*K^T + S*V batched)
        attn_cpu_flops = 4 * D * total_kv * n_layers
        t_attn_cpu_compute = attn_cpu_flops / hw.cpu_compute * 1e9
        # CPU DRAM: K + V per layer
        attn_dram_bytes = 2 * D * S * total_kv * n_layers
        t_attn_cpu_dram = attn_dram_bytes / hw.cpu_dram_bw * 1e9
        # 2 kernel launches per layer (send Q, recv result)
        t_attn_kernels = 2 * n_layers * \
            (hw.L_kernel_launch + hw.L_cpu_gpu_offload)
        t_attn = (t_attn_pcie
                  + max(t_attn_cpu_compute, t_attn_cpu_dram)
                  + t_attn_kernels)
    else:
        t_attn = 0.0

    # ---- per-layer parallelism (CUDA streams) ----
    # Within each layer, the GPU compute path (base matmul + cached-LoRA matmul)
    # runs on the main compute stream while the missed-LoRA path (PCIe + CPU
    # compute + dependent kernel launches) runs on a copy/exec stream. They
    # synchronize before attention. Attention itself is sequenced after the
    # Q/K/V matmuls of that layer. Layers serialize.
    cached_per_layer = t_cached_gpu / n_layers
    missed_per_layer = (t_pcie_lora + t_cpu_path + t_kernels_lora) / n_layers
    attn_per_layer   = t_attn / n_layers
    gpu_path_per_layer = base_per_layer + cached_per_layer

    if parallel:
        # GPU base+cached overlaps with CPU LoRA path within a layer.
        t_per_layer = max(gpu_path_per_layer, missed_per_layer) + attn_per_layer
    else:
        # Worst-case naive: everything within a layer serializes.
        t_per_layer = gpu_path_per_layer + missed_per_layer + attn_per_layer
    total_ns = t_per_layer * n_layers

    return {
        "total_ns":            total_ns,
        "t_base":              t_base,
        "t_cached_gpu_lora":   t_cached_gpu,
        "t_pcie_lora":         t_pcie_lora,
        "t_cpu_lora":          t_cpu_path,
        "t_kernels_lora":      t_kernels_lora,
        "t_attn":              t_attn,
        "base_per_layer":      base_per_layer,
        "missed_per_layer":    missed_per_layer,
        "gpu_path_per_layer":  gpu_path_per_layer,
        "t_per_layer":         t_per_layer,
        "n_cached":            len(hit_ids),
        "n_missed":            len(miss_ids),
        "n_kernel_pairs_lora": n_kernel_pairs,
    }
