"""Latent Subgoal Planner — v0.3 §9.

Predicts phase-level subgoals in latent space (no image generation).
Given current z_full + phase_token + language embedding, outputs the
expected z_full when the current phase completes.

Parameters: ~20M
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LatentSubgoalPlanner(nn.Module):
    """Predict the latent world state at the end of the current task phase.

    Uses a residual MLP: z_goal = z_full + net(z_full, phase, language).
    """

    def __init__(self, z_dim: int = 4096, d_model: int = 2048) -> None:
        super().__init__()
        input_dim = z_dim + d_model + d_model  # z_full + phase + language
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, 2048),
            nn.SiLU(),
            nn.Linear(2048, 1536),
            nn.SiLU(),
            nn.Linear(1536, z_dim),
        )

    def forward(
        self, z_full: Tensor, phase_token: Tensor, language_embed: Tensor
    ) -> Tensor:
        combined = torch.cat([z_full, phase_token, language_embed], dim=-1)
        return z_full + self.net(combined)
