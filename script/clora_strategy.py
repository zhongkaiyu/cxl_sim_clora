#!/usr/bin/env python3
"""
clora_strategy.py
=================

CLoRA strategy-selection plugin.

This module implements the upper-half of the CLoRA paper that the C simulator
currently leaves as a stub:

    * Cost model         -- §6.3, eqs. (1)-(6) for the per-adapter LoRA path
    * Memory accounting  -- §5.1, Fig. 7(a)  (placements P1 / P2 / P3)
    * Algorithm 1        -- §6.2  ChooseExeStrategy(.)

It is deliberately self-contained: no third-party deps, no global state, no
file I/O.  The C simulator (or any harness) can call ``choose_strategy`` per
adapter when a new request arrives, then convert the returned ``Decision``
into the JSON the simulator already expects (one record per adapter telling
it which of the four strategies to execute for the current batch).

Strategy summary (paper Fig. 7):

      placement  GPU mem (S = bytes/elem)        link traffic / call
  E1   P1       2·R·D·S    (A and B cached)     0
  E2   P2       0                                2·R·D·S  (load A, B once)
  E3   P2       0                                2·B·D·S  (send x, recv y)
  E4   P3       R·D·S      (B cached)            B·(D + R)·S
"""

from __future__ import annotations

import json as _json
import os
from dataclasses import dataclass, field
from enum import IntEnum
from statistics import mean
from typing import Dict, List, Tuple


FP16_BYTES = 2  # paper uses FP16 throughout (S in eqs. 1-8)


class Strategy(IntEnum):
    E1 = 1  # A,B in GPU              | GPU only
    E2 = 2  # A,B in CXL, GPU computes | load A,B over CXL
    E3 = 3  # A,B in CXL, NDP computes | send x, recv y
    E4 = 4  # A in CXL, B in GPU       | NDP m=xA, GPU y=mB


# --------------------------------------------------------------- data classes


@dataclass
class HardwareConfig:
    """One per system (paper Tables 3, 4). All BWs in bytes/s, compute in FLOPS."""
    gpu_compute: float       # C_GPU
    gpu_mem_bw: float        # W_GPU  (HBM bandwidth)
    gpu_mem_bytes: int       # bytes available for LoRA + KV (after base model)
    cxl_compute: float       # C_CXL  (per device)
    cxl_dram_bw: float       # W_DRAM (per device, internal)
    cxl_link_bw: float       # W_CXL  (per device, external link)
    cxl_latency: float       # L_CXL  (ns)
    n_cxl: int               # N_CXL


@dataclass
class ModelConfig:
    d: int                   # D (model dim)
    s_dtype: int = FP16_BYTES


@dataclass
class Adapter:
    adapter_id: int
    rank: int                # R_i
    batch: int = 0           # B_i  (requests in current step using this adapter)
    a_in_gpu: bool = False
    b_in_gpu: bool = False
    temperature: float = 0.0 # for §6.2 hot/cold classification

    def matrix_bytes(self, model: ModelConfig) -> int:
        """Size of one of A or B in bytes (both are R x D)."""
        return self.rank * model.d * model.s_dtype


@dataclass
class SystemState:
    adapters: Dict[int, Adapter] = field(default_factory=dict)
    kv_cache_bytes: int = 0     # bytes of KV cache duplicated in GPU memory

    def adapter_gpu_bytes(self, adapter_id: int, model: ModelConfig,
                          *, n_layers: int = 1, n_matrices: int = 1) -> int:
        a = self.adapters[adapter_id]
        per_matrix = a.matrix_bytes(model)
        return ((int(a.a_in_gpu) + int(a.b_in_gpu))
                * per_matrix * n_layers * n_matrices)

    def total_other_bytes(self, exclude_id: int, model: ModelConfig,
                          *, n_layers: int = 1, n_matrices: int = 1) -> int:
        """Bytes consumed by everything except the target adapter."""
        total = self.kv_cache_bytes
        for aid in self.adapters:
            if aid != exclude_id:
                total += self.adapter_gpu_bytes(aid, model,
                                                n_layers=n_layers,
                                                n_matrices=n_matrices)
        return total


@dataclass
class Decision:
    strategy: Strategy
    cost_ns: float
    evict_kv_bytes: int = 0
    evict_cold_ids: List[int] = field(default_factory=list)
    evict_hot_ids: List[int] = field(default_factory=list)
    evict_serving_ids: List[int] = field(default_factory=list)

    def __repr__(self) -> str:
        bits = [f"E{int(self.strategy)}", f"{self.cost_ns:.1f}ns"]
        if self.evict_kv_bytes:
            bits.append(f"evict_kv={self.evict_kv_bytes}B")
        if self.evict_cold_ids:
            bits.append(f"evict_cold={self.evict_cold_ids}")
        if self.evict_hot_ids:
            bits.append(f"evict_hot={self.evict_hot_ids}")
        if self.evict_serving_ids:
            bits.append(f"evict_serving={self.evict_serving_ids}")
        return "Decision(" + ", ".join(bits) + ")"


# --------------------------------------------------------------- §6.3 cost model


def _ns_compute(flops: float, throughput: float) -> float:
    return (flops / throughput) * 1e9 if (flops > 0 and throughput > 0) else 0.0


def _ns_transfer(nbytes: int, bw: float) -> float:
    return (nbytes / bw) * 1e9 if (nbytes > 0 and bw > 0) else 0.0


def cost_lora(strategy: Strategy, adapter: Adapter,
              hw: HardwareConfig, model: ModelConfig) -> float:
    """
    Estimate the LoRA path cost in nanoseconds for one adapter.

    The LoRA path is y = x . A . B where x:1xD, A:DxR, B:RxD.
    Per-sample FLOPs: 2DR (x.A) + 2RD (.B) = 4DR.
    """
    B, R, D, S = adapter.batch, adapter.rank, model.d, model.s_dtype

    if strategy is Strategy.E1:
        # All work on GPU; no CXL traffic.
        t_compute = _ns_compute(4 * B * D * R, hw.gpu_compute)
        t_hbm = _ns_transfer(2 * R * D * S, hw.gpu_mem_bw) if B > 0 else 0
        return max(t_compute, t_hbm)

    if strategy is Strategy.E2:
        # GPU computes, but A and B come over the CXL link once per batch.
        t_compute = _ns_compute(4 * B * D * R, hw.gpu_compute)
        t_link = (hw.cxl_latency + _ns_transfer(2 * R * D * S, hw.cxl_link_bw)
                  ) if B > 0 else 0.0
        return max(t_compute, t_link)

    if strategy is Strategy.E3:
        # NDP computes the full LoRA; only x and y cross the link, scaling with B.
        t_link = (hw.cxl_latency + _ns_transfer(2 * B * D * S, hw.cxl_link_bw)
                  ) if B > 0 else 0.0
        t_ndp = _ns_compute(4 * B * D * R, hw.cxl_compute * hw.n_cxl)
        return t_link + t_ndp

    if strategy is Strategy.E4:
        # NDP computes m=x.A and ships R-wide vectors; GPU finishes y=m.B.
        t_link = (hw.cxl_latency + _ns_transfer(B * (D + R) * S, hw.cxl_link_bw)
                  ) if B > 0 else 0.0
        t_ndp = _ns_compute(2 * B * D * R, hw.cxl_compute * hw.n_cxl)
        t_gpu = _ns_compute(2 * B * D * R, hw.gpu_compute)
        t_hbm = _ns_transfer(R * D * S, hw.gpu_mem_bw) if B > 0 else 0
        return max(t_link + t_ndp, max(t_gpu, t_hbm))

    raise ValueError(f"unknown strategy: {strategy}")


def gpu_mem_for(strategy: Strategy, adapter: Adapter, model: ModelConfig,
                *, n_layers: int = 1, n_matrices: int = 1) -> int:
    """Bytes of GPU memory this strategy requires for the given adapter.

    A cached adapter holds A and/or B for *every* layer × *every* LoRA-touched
    weight matrix, so the per-matrix size scales by ``n_layers * n_matrices``.
    Default 1*1 keeps existing per-matrix tests valid; the driver passes
    n_layers=32, n_matrices=7 for realistic accounting.
    """
    one = adapter.matrix_bytes(model) * n_layers * n_matrices
    return {Strategy.E1: 2 * one,
            Strategy.E2: 0,
            Strategy.E3: 0,
            Strategy.E4: one}[strategy]


# --------------------------------------------------------- §6.2 classification


def classify_adapters(state: SystemState,
                      *, top_k: int = None
                      ) -> Tuple[List[Adapter], List[Adapter], List[Adapter]]:
    """Split adapters into (serving, hot, cold) lists per §6.2.

    serving : currently in the running batch (batch > 0)
    hot     : (top_k mode) top-K non-serving adapters by temperature
              (default) non-serving with temperature >= mean(non-serving temps)
    cold    : the remaining non-serving adapters

    The driver passes ``top_k`` (default 50, matching the paper's Skewed config)
    so the partition is a fixed-size hot pool, independent of pool size.
    Tests that omit ``top_k`` keep mean-based behavior.
    """
    serving = [a for a in state.adapters.values() if a.batch > 0]
    others = [a for a in state.adapters.values() if a.batch == 0]
    if not others:
        return serving, [], []

    if top_k is not None:
        # Top-K hottest non-serving = hot; remainder = cold
        sorted_desc = sorted(others, key=lambda a: -a.temperature)
        k = min(top_k, len(sorted_desc))
        hot  = sorted_desc[:k]
        cold = sorted_desc[k:]
    else:
        avg = mean(a.temperature for a in others)
        hot  = [a for a in others if a.temperature >= avg]
        cold = [a for a in others if a.temperature <  avg]
    return serving, hot, cold


# -------------------------------------- §6.2 hot-adapter proactive caching


def pre_cache_hot_adapters(state: SystemState,
                           hw: HardwareConfig,
                           model: ModelConfig,
                           *,
                           top_k: int = 50,
                           n_layers: int = 1,
                           n_matrices: int = 1) -> List[int]:
    """Paper §6.2 + Figure 10: pre-cache the top-K hottest non-serving adapters
    in GPU memory as E1 (both A and B). Hottest-first greedy fill until we run
    out of memory.

    Mutates state.adapters[id].a_in_gpu / .b_in_gpu = True for cached entries.
    The serving adapters' bytes are reserved up-front so we don't crowd them
    out. Returns the list of pre-cached adapter ids (longest-temp first).
    """
    serving, hot, _ = classify_adapters(state, top_k=top_k)
    # Reserve memory for serving adapters' upper-bound footprint (E1 sizing)
    # so we don't over-allocate to the hot pool.
    serving_reserved = sum(
        2 * a.rank * model.d * model.s_dtype * n_layers * n_matrices
        for a in serving)
    budget = hw.gpu_mem_bytes - serving_reserved
    if budget <= 0:
        return []

    hot_sorted = sorted(hot, key=lambda a: -a.temperature)
    cached: List[int] = []
    for a in hot_sorted:
        needed = (2 * a.rank * model.d * model.s_dtype
                  * n_layers * n_matrices)
        if needed <= budget:
            a.a_in_gpu = True
            a.b_in_gpu = True
            budget -= needed
            cached.append(a.adapter_id)
    return cached


# ------------------------------------------------------- §6.2 Algorithm 1


def _evict_set(pool: List[Adapter], need: int,
               state: SystemState, model: ModelConfig,
               *, n_layers: int = 1, n_matrices: int = 1
               ) -> Tuple[List[int], int]:
    """Greedily pick adapters from ``pool`` (largest first) until ``need`` bytes
    are freed. Returns (chosen ids, total bytes freed)."""
    ranked = sorted(pool,
                    key=lambda a: -state.adapter_gpu_bytes(
                        a.adapter_id, model,
                        n_layers=n_layers, n_matrices=n_matrices))
    chosen: List[int] = []
    freed = 0
    for a in ranked:
        if freed >= need:
            break
        bytes_here = state.adapter_gpu_bytes(
            a.adapter_id, model, n_layers=n_layers, n_matrices=n_matrices)
        if bytes_here == 0:
            continue   # nothing to evict from this one
        chosen.append(a.adapter_id)
        freed += bytes_here
    return chosen, freed


def emit_step_json(step_kind: str,
                   model: ModelConfig,
                   hw: HardwareConfig,
                   decisions: Dict[int, Decision],
                   state: SystemState,
                   *,
                   n_layers: int = 32,
                   n_matrices: int = 7,
                   input_tokens_per_request: int = 1,
                   base_model_compute_ns: float = 0.0,
                   kv_tokens_per_adapter: Dict[int, int] | None = None,
                   kv_in_gpu_fraction: float = 0.0) -> dict:
    """Translate per-adapter decisions into the JSON the C simulator consumes.

    See the contract documented in the driver. ``step_kind`` is "decode" or
    "prefill"; for prefill, set ``input_tokens_per_request`` to the prompt length.
    """
    if step_kind not in ("decode", "prefill"):
        raise ValueError(f"unknown step_kind: {step_kind}")

    kv_tokens_per_adapter = kv_tokens_per_adapter or {}
    kv_total = sum(kv_tokens_per_adapter.values())

    adapters_json = []
    for aid, dec in decisions.items():
        a = state.adapters[aid]
        adapters_json.append({
            "id": aid,
            "rank": a.rank,
            "batch": a.batch,
            "strategy": int(dec.strategy),
            "kv_tokens": kv_tokens_per_adapter.get(aid, 0),
        })

    return {
        "kind": step_kind,
        "model": {
            "d": model.d,
            "n_layers": n_layers,
            "n_matrices": n_matrices,
            "s_dtype": model.s_dtype,
        },
        "hw": {
            "n_cxl": hw.n_cxl,
            "s_dtype": model.s_dtype,
        },
        "input_tokens_per_request": input_tokens_per_request,
        "base_model_compute_ns": base_model_compute_ns,
        "kv": {
            "kv_tokens_total": kv_total,
            "kv_in_gpu_fraction": kv_in_gpu_fraction,
        },
        "adapters": adapters_json,
    }


def compute_kv_in_gpu_fraction(decisions: Dict[int, "Decision"],
                               state: "SystemState",
                               hw: HardwareConfig,
                               model: ModelConfig,
                               kv_tokens_per_adapter: Dict[int, int],
                               *,
                               n_layers: int = 32,
                               n_matrices: int = 7,
                               total_batch: int = 0,
                               base_model_bytes: int = 0,
                               moe_active_base_bytes: int = None) -> float:
    """Paper §5.2 + Eqs (7)-(9): choose the fraction P_KV of the KV cache to
    duplicate in GPU memory.

    The GPU's share of attention is NOT free -- it adds P_KV-proportional HBM
    reads and FLOPs to the GPU's per-layer roofline (Eq 7), while the CXL
    devices' share shrinks with (1 - P_KV) (Eq 8). Since the two sides run in
    parallel (Eq 9), the best P_KV minimizes

        T(P) = max( T_GPU_layer(P), T_ATT_CXL_layer(P) )

    subject to the memory cap  P <= gpu_free / kv_total_bytes.

    With 4 devices the aggregate device DRAM bandwidth usually beats the
    GPU's leftover HBM bandwidth, so P* is small (0 when the CXL side already
    hides under base compute) -- consistent with the 2-14% reported in paper
    Fig 16, and unlike a greedy fill which can reach 100% at small batch.

    Backward compatibility: when ``base_model_bytes`` is not given (<= 0),
    falls back to the legacy greedy fill  min(free, kv_total) / kv_total.
    """
    if not kv_tokens_per_adapter:
        return 0.0
    total_kv_tokens = sum(kv_tokens_per_adapter.values())
    if total_kv_tokens <= 0:
        return 0.0
    kv_total_bytes = 2 * model.d * model.s_dtype * total_kv_tokens * n_layers
    if kv_total_bytes <= 0:
        return 0.0

    lora_bytes = 0
    # Serving adapters: bytes consumed by their chosen strategy
    for aid, dec in decisions.items():
        a = state.adapters[aid]
        lora_bytes += gpu_mem_for(dec.strategy, a, model,
                                  n_layers=n_layers, n_matrices=n_matrices)
    # Pre-cached hot adapters (set by pre_cache_hot_adapters) live in
    # state.adapters with a_in_gpu/b_in_gpu marked.  Count them too.
    for aid, a in state.adapters.items():
        if aid in decisions:
            continue
        lora_bytes += state.adapter_gpu_bytes(aid, model,
                                              n_layers=n_layers,
                                              n_matrices=n_matrices)

    free = max(0, hw.gpu_mem_bytes - lora_bytes)
    p_max = min(free, kv_total_bytes) / kv_total_bytes

    if base_model_bytes <= 0:
        # Legacy greedy fill (kept for callers that don't supply base info).
        return p_max

    D, S, K = model.d, model.s_dtype, total_kv_tokens
    B = max(total_batch, 0)
    n_dev = max(hw.n_cxl, 1)

    # GPU per-layer roofline components (Eq 7 folded into the base roofline)
    hbm_bytes_total = (moe_active_base_bytes
                       if moe_active_base_bytes is not None
                       else base_model_bytes)
    base_c = _ns_compute(24.0 * D * D * B, hw.gpu_compute)
    base_h = _ns_transfer(hbm_bytes_total / max(n_layers, 1), hw.gpu_mem_bw)
    attn_c = _ns_compute(4.0 * D * K, hw.gpu_compute)        # at P = 1
    attn_h = _ns_transfer(2 * D * S * K, hw.gpu_mem_bw)      # at P = 1

    # CXL per-layer attention components (Eq 8), at P = 0
    cxl_unit = max(_ns_compute(4.0 * D * K, hw.cxl_compute * n_dev),
                   _ns_transfer(2 * D * S * K, hw.cxl_dram_bw * n_dev))
    cxl_link = 2 * hw.cxl_latency + _ns_transfer(2 * D * B * S, hw.cxl_link_bw)

    def step_attn_ns(p: float) -> float:
        t_gpu = max(base_c + p * attn_c, base_h + p * attn_h)
        t_cxl = (1.0 - p) * cxl_unit + cxl_link
        return max(t_gpu, t_cxl)

    # 1-D grid search over [0, p_max]; ties prefer smaller P (frees memory).
    best_p, best_t = 0.0, step_attn_ns(0.0)
    steps = 200
    for i in range(1, steps + 1):
        p = p_max * i / steps
        t = step_attn_ns(p)
        if t < best_t - 1e-9:
            best_p, best_t = p, t
    return best_p


# -------------------------------------------------------- §6.2 temperature

class TemperatureModel:
    """Mean-reverting temperature model from paper §6.2.

        on_request(a):   temp[a] += T_1
        step():          temp[a] -= alpha * (temp[a] - mean(temp))

    Hot/cold classification then runs over these temperatures. The model is
    persistent across driver steps; the driver re-uses one instance.

    Tunable knobs:
      T_1     -- temperature bump on each request arrival.
      alpha   -- decay rate; fraction of (temp - mean) shed per step.
    """

    def __init__(self, T_1: float = 1.0, alpha: float = 0.1):
        self.T_1 = float(T_1)
        self.alpha = float(alpha)
        self.temps: Dict[int, float] = {}

    def on_request(self, adapter_id: int) -> None:
        self.temps[adapter_id] = self.temps.get(adapter_id, 0.0) + self.T_1

    def step(self) -> None:
        if not self.temps:
            return
        mean_t = sum(self.temps.values()) / len(self.temps)
        for aid in list(self.temps.keys()):
            self.temps[aid] -= self.alpha * (self.temps[aid] - mean_t)

    def get(self, adapter_id: int) -> float:
        return self.temps.get(adapter_id, 0.0)


def write_step_json(path: str, payload: dict) -> None:
    """Atomically write a step's JSON to disk so the C simulator can read it."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        _json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def estimate_base_model_ns_per_layer(total_batch: int,
                                     model: ModelConfig,
                                     hw: HardwareConfig,
                                     *,
                                     base_model_bytes: int = 14 * 10**9,
                                     n_layers: int = 32,
                                     moe_active_base_bytes: int = None,
                                     kv_tokens_in_gpu: int = 0) -> float:
    """Per-decoder-layer GPU busy time in ns (base model + GPU-side attention).

    Decode is HBM-bound at small batches and compute-bound at large ones:
    every layer must load its full weight slice from HBM regardless of batch,
    while compute scales linearly with batch.

        compute_ns  = (24 D^2 B + 4 D K_gpu) / C_GPU       (FLOPs / FLOPS * 1e9)
        hbm_ns      = (base_bytes/n_layers + 2 D S K_gpu) / W_GPU
        per_layer   = max(compute_ns, hbm_ns)

    The 24 * D^2 coefficient covers Q/K/V/O (8 * D^2) and FFN G/U/output
    (3 * 2 * D * FFN_DIM ~ 16 * D^2 for FFN_DIM ~ 2.7 D). ``base_model_bytes``
    defaults to Llama2-7B in FP16. Override for 13B (~26 GB) or 8B (~16 GB).

    ``kv_tokens_in_gpu`` is the per-layer token count of the KV-cache portion
    duplicated in GPU memory (P_KV * total KV tokens). Its attention work --
    paper Eq (7): 4*D*K FLOPs and 2*D*S*K HBM bytes per layer -- rides the
    same per-layer roofline as the base model because it competes for the
    same HBM bandwidth and SMs. Default 0 preserves the old behavior.

    For Mixture-of-Experts (MoE) base models, only a subset of expert weights
    are activated per token, so the HBM read per layer is the *active*
    parameter slice rather than the full model footprint. Pass
    ``moe_active_base_bytes`` (the total active-parameter bytes across all
    layers) to use that value for ``hbm_ns`` instead of ``base_model_bytes``.
    When ``moe_active_base_bytes`` is None (the default), the dense
    ``base_model_bytes`` is used (existing behavior). The per-layer compute
    formula (24 * D^2 * batch for attention-only LoRA) is unchanged because
    LoRA in this simulator only touches the attention projections.
    """
    if total_batch <= 0 or hw.gpu_compute <= 0 or hw.gpu_mem_bw <= 0:
        return 0.0
    K = max(kv_tokens_in_gpu, 0)
    compute_ns = (24.0 * model.d * model.d * total_batch
                  + 4.0 * model.d * K) / hw.gpu_compute * 1e9
    hbm_bytes_total = (moe_active_base_bytes
                       if moe_active_base_bytes is not None
                       else base_model_bytes)
    bytes_per_layer = (hbm_bytes_total / max(n_layers, 1)
                       + 2 * model.d * model.s_dtype * K)
    hbm_ns = bytes_per_layer / hw.gpu_mem_bw * 1e9
    return max(compute_ns, hbm_ns)


def choose_strategy(adapter_id: int, state: SystemState,
                    hw: HardwareConfig, model: ModelConfig,
                    *, n_layers: int = 1, n_matrices: int = 1,
                    top_k_hot: int = None) -> Decision:
    """Algorithm 1 from §6.2.

    Steps (matching the pseudocode):
        1. Sort strategies by cost.
        2. If the cheapest fits in the GPU memory budget -> return it.
        3. Else, if cold adapters can free enough space -> evict cold, return cheapest.
        4. Else, for each strategy, consider evicting KV cache / hot / serving
           adapters; keep all feasible variants and return the lowest-cost one.
    """
    if adapter_id not in state.adapters:
        raise KeyError(f"adapter {adapter_id} not in state")
    target = state.adapters[adapter_id]
    NL, NM = n_layers, n_matrices

    ranked = sorted(Strategy,
                    key=lambda E: cost_lora(E, target, hw, model))

    # Memory currently occupied by everything else (including target's own
    # cached matrices, which we *don't* count as "other"):
    other_bytes = state.total_other_bytes(adapter_id, model,
                                          n_layers=NL, n_matrices=NM)
    budget = hw.gpu_mem_bytes - other_bytes  # may be negative when over-committed

    best = ranked[0]
    mem_best = gpu_mem_for(best, target, model, n_layers=NL, n_matrices=NM)

    # ---- Step 2: cheapest already fits.
    if mem_best <= budget:
        return Decision(strategy=best,
                        cost_ns=cost_lora(best, target, hw, model))

    # ---- Step 3: cold-only eviction for the cheapest strategy.
    _, _, cold = classify_adapters(state, top_k=top_k_hot)
    cold = [a for a in cold if a.adapter_id != adapter_id]
    deficit_best = mem_best - budget
    cold_ids, cold_freed = _evict_set(cold, deficit_best, state, model,
                                      n_layers=NL, n_matrices=NM)
    if cold_freed >= deficit_best:
        return Decision(strategy=best,
                        cost_ns=cost_lora(best, target, hw, model),
                        evict_cold_ids=cold_ids)

    # ---- Step 4: enumerate (strategy, eviction-kind) variants and pick best.
    serving, hot, _ = classify_adapters(state, top_k=top_k_hot)
    serving = [a for a in serving if a.adapter_id != adapter_id]

    candidates: List[Decision] = []
    for E in Strategy:
        mem = gpu_mem_for(E, target, model, n_layers=NL, n_matrices=NM)
        cost = cost_lora(E, target, hw, model)
        deficit = mem - budget

        # (a) fits as-is
        if deficit <= 0:
            candidates.append(Decision(strategy=E, cost_ns=cost))
            continue

        # (b) evict KV cache
        if state.kv_cache_bytes >= deficit:
            candidates.append(Decision(strategy=E, cost_ns=cost,
                                       evict_kv_bytes=deficit))

        # (c) evict cold adapters
        c_ids, c_freed = _evict_set(cold, deficit, state, model,
                                    n_layers=NL, n_matrices=NM)
        if c_freed >= deficit:
            candidates.append(Decision(strategy=E, cost_ns=cost,
                                       evict_cold_ids=c_ids))

        # (d) evict hot adapters
        h_ids, h_freed = _evict_set(hot, deficit, state, model,
                                    n_layers=NL, n_matrices=NM)
        if h_freed >= deficit:
            candidates.append(Decision(strategy=E, cost_ns=cost,
                                       evict_hot_ids=h_ids))

        # (e) evict other serving adapters (last resort)
        s_ids, s_freed = _evict_set(serving, deficit, state, model,
                                    n_layers=NL, n_matrices=NM)
        if s_freed >= deficit:
            candidates.append(Decision(strategy=E, cost_ns=cost,
                                       evict_serving_ids=s_ids))

    if not candidates:
        raise RuntimeError(
            f"no feasible strategy: budget={budget} bytes, target needs at "
            f"least {min(gpu_mem_for(E, target, model, n_layers=NL, n_matrices=NM) for E in Strategy)} bytes")

    candidates.sort(key=lambda d: d.cost_ns)
    return candidates[0]
