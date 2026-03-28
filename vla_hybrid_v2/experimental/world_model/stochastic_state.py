"""Stochastic State Module — DreamerV3-style discrete latent with 48×48 categories.

Implements the dual-distribution latent state from v0.3 §4.3:
- Posterior q(z_sto | z_det, obs): used during training (has real observations)
- Prior     p(z_sto | z_det):     used during imagination (no observations)

Uses straight-through Gumbel-Softmax with 1% uniform mixing to prevent
codebook collapse (DreamerV3 technique).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class StochasticStateModule(nn.Module):
    """48×48 categorical stochastic state (2,304 discrete codes → project → d_model).

    Parameters
    ----------
    d_model : int
        Deterministic state dimension (matches fused_state from Mamba Core).
    n_categories : int
        Number of independent categorical distributions.
    n_classes : int
        Number of classes per categorical.
    unimix : float
        Fraction of uniform probability mixed in to prevent collapse.
    """

    def __init__(
        self,
        d_model: int = 2048,
        n_categories: int = 48,
        n_classes: int = 48,
        unimix: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_categories = n_categories
        self.n_classes = n_classes
        self.unimix = unimix
        z_dim = n_categories * n_classes  # 2304

        # Posterior: q(z_sto | z_det, obs_encoding)
        self.posterior = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, z_dim),
        )

        # Prior: p(z_sto | z_det)
        self.prior = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, z_dim),
        )

        # Project one-hot z_sto back to d_model
        self.z_proj = nn.Linear(z_dim, d_model)

    def _sample(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        """Straight-through sampling with uniform mixing."""
        logits = logits.view(-1, self.n_categories, self.n_classes)
        probs = (1 - self.unimix) * F.softmax(logits, dim=-1) + (
            self.unimix / self.n_classes
        )
        # Straight-through: forward uses argmax, backward uses probs
        indices = probs.argmax(dim=-1)
        sample = F.one_hot(indices, self.n_classes).float()
        sample = sample + probs - probs.detach()  # ST gradient
        z_flat = sample.reshape(logits.shape[0], -1)  # [B, n_cat * n_cls]
        return z_flat, probs

    def encode_posterior(
        self, z_det: Tensor, obs_encoding: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Training path: condition on both z_det and observation."""
        logits = self.posterior(torch.cat([z_det, obs_encoding], dim=-1))
        z_sto, probs = self._sample(logits)
        z_full = torch.cat([z_det, self.z_proj(z_sto)], dim=-1)  # [B, 2*d_model]
        return z_full, probs, logits

    def encode_prior(self, z_det: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Imagination path: condition on z_det only."""
        logits = self.prior(z_det)
        z_sto, probs = self._sample(logits)
        z_full = torch.cat([z_det, self.z_proj(z_sto)], dim=-1)
        return z_full, probs, logits
