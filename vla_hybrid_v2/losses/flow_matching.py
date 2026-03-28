"""Flow matching loss for v2 — same as v1 with logit_normal scheduling."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FlowMatchingLoss(nn.Module):
    def __init__(self, reduction="mean",
                 timestep_schedule="logit_normal") -> None:
        super().__init__()
        self.reduction = reduction
        self.timestep_schedule = timestep_schedule

    def forward(self, velocity_pred, x_0, x_1, t=None, step_weights=None):
        # NOTE: `t` is accepted for API compatibility but unused —
        # under Rectified Flow the target velocity (x_1 - x_0) is t-independent.
        target_velocity = x_1 - x_0
        loss = (velocity_pred - target_velocity).pow(2)
        if step_weights is not None:
            loss = loss * step_weights.unsqueeze(-1)
        return loss.mean() if self.reduction == "mean" else loss

    def sample_timestep(self, batch_size, device):
        if self.timestep_schedule == "logit_normal":
            return torch.sigmoid(torch.randn(batch_size, device=device))
        return torch.rand(batch_size, device=device)

    @staticmethod
    def interpolate(x_0, x_1, t):
        """Linear interpolation for Rectified Flow: x_t = (1-t)*x_0 + t*x_1."""
        t_exp = t[:, None, None]
        return (1.0 - t_exp) * x_0 + t_exp * x_1
