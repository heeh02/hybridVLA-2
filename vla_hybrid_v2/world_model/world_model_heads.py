"""Reward / Value / Termination prediction heads — v0.3 §7.

Uses DreamerV3's symlog two-hot regression for reward and value prediction,
and a Bernoulli head for episode termination.

Parameters: ~15M total across three heads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SymlogTwoHot:
    """Utility for symlog two-hot distributed regression (DreamerV3)."""

    @staticmethod
    def symlog(x: Tensor) -> Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def symexp(x: Tensor) -> Tensor:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    @staticmethod
    def twohot_encode(x: Tensor, bins: Tensor) -> Tensor:
        """Soft two-hot encoding of scalar *x* onto *bins*."""
        x_clamped = x.clamp(bins[0], bins[-1])
        # Find the two nearest bins
        below = (bins <= x_clamped.unsqueeze(-1)).sum(-1) - 1
        below = below.clamp(0, len(bins) - 2)
        above = below + 1
        weight_above = (x_clamped - bins[below]) / (bins[above] - bins[below] + 1e-8)
        weight_below = 1.0 - weight_above
        target = torch.zeros(*x.shape, len(bins), device=x.device, dtype=x.dtype)
        target.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
        target.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
        return target

    @staticmethod
    def loss(logits: Tensor, target: Tensor, bins: Tensor) -> Tensor:
        """Cross-entropy on two-hot encoded symlog target."""
        symlog_target = SymlogTwoHot.symlog(target)
        soft_target = SymlogTwoHot.twohot_encode(symlog_target, bins)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_target * log_probs).sum(-1).mean()


class WorldModelHeads(nn.Module):
    """Reward + Value + Done prediction from z_full.

    Parameters
    ----------
    z_dim : int
        Full world state dimension (2 * d_model).
    reward_bins, value_bins : int
        Number of symlog regression bins.
    """

    def __init__(
        self,
        z_dim: int = 4096,
        reward_bins: int = 255,
        value_bins: int = 255,
    ) -> None:
        super().__init__()

        self.reward_head = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, reward_bins),
        )

        self.value_head = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, value_bins),
        )

        self.done_head = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 1),
        )

        self.register_buffer(
            "reward_bins_t", torch.linspace(-20, 20, reward_bins)
        )
        self.register_buffer(
            "value_bins_t", torch.linspace(-20, 20, value_bins)
        )

    def forward(self, z_full: Tensor) -> dict[str, Tensor]:
        return {
            "reward_logits": self.reward_head(z_full),
            "value_logits": self.value_head(z_full),
            "done_logit": self.done_head(z_full),
        }

    def decode_reward(self, logits: Tensor) -> Tensor:
        probs = F.softmax(logits, dim=-1)
        return SymlogTwoHot.symexp((probs * self.reward_bins_t).sum(-1))

    def decode_value(self, logits: Tensor) -> Tensor:
        probs = F.softmax(logits, dim=-1)
        return SymlogTwoHot.symexp((probs * self.value_bins_t).sum(-1))
