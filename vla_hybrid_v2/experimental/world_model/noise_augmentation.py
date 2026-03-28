"""Noise Augmentation — GameNGen-style anti-drift for imagination rollouts.

v0.3 §8.1: During training, add random noise to the previous step's
predicted state, forcing the model to recover from imperfect inputs.
The noise level is fed as an explicit input so the model knows the
reliability of its input.

At inference time no noise is added, but the model has learned to be
robust to small prediction errors — preventing exponential drift
in multi-step imagination.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NoiseAugmentation(nn.Module):
    """GameNGen-style noise augmentation with learned noise-level embedding.

    Parameters
    ----------
    z_dim : int
        Dimension of the state to augment (z_full = 2 * d_model).
    num_buckets : int
        Discrete noise-level buckets.
    max_noise_sigma : float
        Maximum noise standard deviation (GameNGen uses 0.7).
    """

    def __init__(
        self,
        z_dim: int = 4096,
        num_buckets: int = 16,
        max_noise_sigma: float = 0.7,
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.max_noise_sigma = max_noise_sigma
        # d_model-sized embedding (half of z_dim) — matches imagination_mamba input
        embed_dim = z_dim // 2
        self.noise_encoder = nn.Sequential(
            nn.Embedding(num_buckets, 512),
            nn.SiLU(),
            nn.Linear(512, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def augment(
        self,
        z: Tensor,
        step_idx: int,
        total_steps: int,
        training: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Add scheduled noise and return (noisy_z, noise_embedding).

        Noise level increases linearly with *step_idx* — later predictions
        are less certain and get more noise during training.
        """
        B = z.shape[0]
        device = z.device

        if not training:
            bucket = torch.zeros(B, dtype=torch.long, device=device)
            return z, self.noise_encoder(bucket)

        sigma = self.max_noise_sigma * (step_idx / max(total_steps, 1))
        noise = torch.randn_like(z) * sigma

        bucket_idx = min(
            int(sigma / self.max_noise_sigma * self.num_buckets),
            self.num_buckets - 1,
        )
        bucket = torch.full((B,), bucket_idx, dtype=torch.long, device=device)

        return z + noise, self.noise_encoder(bucket)
