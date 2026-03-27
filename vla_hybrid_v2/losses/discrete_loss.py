"""Discrete losses for v2 — adds label smoothing."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class DiscreteCELoss(nn.Module):
    def __init__(self, reduction="mean", label_smoothing=0.1) -> None:
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        B, H, A, V = logits.shape
        return F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1),
            reduction=self.reduction, label_smoothing=self.label_smoothing,
        )


class PhaseLoss(nn.Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(logits, targets, reduction=self.reduction)
