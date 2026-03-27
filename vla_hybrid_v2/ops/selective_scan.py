"""Selective SSM scan — with CUDA fast path when available.

Exports:
    ssm_scan          — Pre-discretized scan (JIT fallback, used by core Mamba).
    selective_scan_fn  — Raw CUDA kernel (None when mamba_ssm not installed).
    HAS_MAMBA_CUDA     — True when mamba_ssm CUDA ops are available.
    causal_conv1d_fn   — Fused causal conv1d (None when causal-conv1d not installed).
    HAS_CAUSAL_CONV1D  — True when causal-conv1d is available.
"""

from typing import Tuple

import torch
from torch import Tensor

# ---- CUDA fast-path: mamba_ssm selective scan kernel ----
try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as _selective_scan_fn,
    )
    HAS_MAMBA_CUDA = True
    selective_scan_fn = _selective_scan_fn
except ImportError:
    HAS_MAMBA_CUDA = False
    selective_scan_fn = None  # type: ignore[assignment]

# ---- CUDA fast-path: fused causal conv1d ----
try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn
    HAS_CAUSAL_CONV1D = True
    causal_conv1d_fn = _causal_conv1d_fn
except ImportError:
    HAS_CAUSAL_CONV1D = False
    causal_conv1d_fn = None  # type: ignore[assignment]


@torch.jit.script
def ssm_scan(
    dA: Tensor, dBx: Tensor, C: Tensor, state: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Pre-discretized SSM scan (JIT-compiled fallback).

    Used by the core Mamba blocks that carry recurrent state across temporal
    steps.  For zero-initial-state scenarios (e.g. ExpertMambaBlock), prefer
    ``selective_scan_fn`` from mamba_ssm when available — it runs the same
    recurrence inside a single fused CUDA kernel.
    """
    B = dA.shape[0]
    L = dA.shape[1]
    D = dA.shape[2]
    y = torch.empty(B, L, D, device=dA.device, dtype=dA.dtype)
    for t in range(L):
        state = dA[:, t] * state + dBx[:, t]
        y[:, t] = (state * C[:, t].unsqueeze(1)).sum(-1)
    return y, state
