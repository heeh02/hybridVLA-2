"""Visual World Decoder — v0.3 §5 three-level decode strategy.

L0: Latent contrastive (no parameters, just L2 in latent space)
L1: CNN decoder → 112×112 RGB (40M params, ~1ms on H100)
L2: Placeholder for frozen pretrained diffusion decoder (not trained here)
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class CNNWorldDecoder(nn.Module):
    """Decode z_full → 112×112 RGB for training signal + quick visualisation.

    Architecture: Linear projection → reshape → 4-stage ConvTranspose2d
    Resolution ladder: 7×7 → 14 → 28 → 56 → 112

    Parameters: ~40M
    """

    def __init__(self, z_dim: int = 4096, image_size: int = 112) -> None:
        super().__init__()
        self.image_size = image_size

        self.proj = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, 1024 * 7 * 7),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (1024, 7, 7)),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
        )

    def forward(self, z_full: Tensor) -> Tensor:
        """z_full: [B, z_dim] → image: [B, 3, 112, 112]"""
        h = self.proj(z_full)
        return self.decoder(h)


class WorldDecoderLoss(nn.Module):
    """Combined L1 + perceptual loss on 112×112 predictions."""

    def __init__(self, perceptual_weight: float = 0.5) -> None:
        super().__init__()
        self.perceptual_weight = perceptual_weight
        # Lazy init: LPIPS loaded only when first called
        self._lpips = None

    def _get_lpips(self, device):
        if self._lpips is None:
            try:
                import lpips

                self._lpips = lpips.LPIPS(net="vgg").to(device).eval()
                for p in self._lpips.parameters():
                    p.requires_grad = False
            except ImportError:
                self._lpips = False  # sentinel: library not available
        return self._lpips

    def forward(self, pred_112: Tensor, target_224: Tensor) -> dict[str, Tensor]:
        target_112 = F.interpolate(
            target_224, size=(112, 112), mode="bilinear", align_corners=False
        )
        losses: dict[str, Tensor] = {}
        losses["recon_l1"] = F.l1_loss(pred_112, target_112)

        lpips_model = self._get_lpips(pred_112.device)
        if lpips_model and lpips_model is not False:
            losses["perceptual"] = (
                lpips_model(pred_112, target_112).mean() * self.perceptual_weight
            )

        return losses
