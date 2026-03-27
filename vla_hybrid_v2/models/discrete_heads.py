"""Discrete heads for HybridVLA v2.

v2 changes: 512 vocab, 2048d input, label smoothing, affordance head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FASTDiscreteHead(nn.Module):
    """FAST head v2: 512 bins, 2048d input, factorized prediction."""

    def __init__(self, input_dim=2048, action_dim=14, vocab_size=512,
                 chunk_horizon=24) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.vocab_size = vocab_size
        self.chunk_horizon = chunk_horizon
        hidden, step_dim = 768, 192
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
        )
        self.step_proj = nn.Linear(hidden, chunk_horizon * step_dim)
        self.dim_head = nn.Sequential(
            nn.LayerNorm(step_dim), nn.Linear(step_dim, action_dim * vocab_size),
        )

    def forward(self, fused_state: Tensor) -> Tensor:
        B = fused_state.shape[0]
        h = self.encoder(fused_state)
        h = self.step_proj(h).view(B * self.chunk_horizon, -1)
        logits = self.dim_head(h)
        return logits.view(B, self.chunk_horizon, self.action_dim, self.vocab_size)

    @staticmethod
    def discretise_actions(actions, lo=-1.0, hi=1.0, V=512):
        actions = actions.clamp(lo, hi)
        return ((actions - lo) / (hi - lo) * (V - 1)).long()

    @staticmethod
    def undiscretise_actions(indices, lo=-1.0, hi=1.0, V=512):
        return indices.float() / (V - 1) * (hi - lo) + lo


class PhaseHead(nn.Module):
    def __init__(self, input_dim=2048, num_phases=16) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2), nn.GELU(),
            nn.Linear(input_dim // 2, num_phases),
        )

    def forward(self, phase_token: Tensor) -> Tensor:
        return self.head(phase_token)


class AffordanceHead(nn.Module):
    """Predicts categorical affordance type from the affordance token.

    Returns logits over `num_affordance_types` classes (e.g., grasp, push,
    pull, place). Spatial affordance map is not yet implemented.
    """

    def __init__(self, input_dim=2048, num_affordance_types=8) -> None:
        super().__init__()
        self.type_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2), nn.GELU(),
            nn.Linear(input_dim // 2, num_affordance_types),
        )

    def forward(self, affordance_token: Tensor) -> Tensor:
        return self.type_head(affordance_token)
