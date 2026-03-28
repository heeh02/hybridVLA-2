"""v2 Consistency losses: contrastive temporal + slow-fast agreement.

v2 changes: replaces simple L2 smoothness with InfoNCE-style contrastive
loss and adds slow-fast agreement term.

v0.10.10 fixes (L-12/13/14):
- ContrastiveTemporalLoss: added VICReg-style variance regularisation to
  prevent representation collapse.
- ActionConsistencyLoss: replaced projection+cosine with direct MSE in
  14D action space (projection layers could trivially collapse to constant).
- SlowFastAgreementLoss: removed .detach() so slow and fast streams train
  bidirectionally.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ContrastiveTemporalLoss(nn.Module):
    """InfoNCE-style: consecutive fused states should be more similar to each
    other than to random states in the batch.

    L-12 fix: added variance regularisation (VICReg-style) to prevent all
    fused_states from collapsing to a constant vector.
    """

    def __init__(self, temperature: float = 0.1,
                 variance_weight: float = 1.0,
                 variance_target: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.variance_weight = variance_weight
        self.variance_target = variance_target

    def forward(self, fused_states: Tensor) -> Tensor:
        """fused_states: [B, T, D]"""
        if fused_states.shape[1] < 2:
            return torch.tensor(0.0, device=fused_states.device)
        anchors = F.normalize(fused_states[:, :-1], dim=-1)  # [B, T-1, D]
        positives = F.normalize(fused_states[:, 1:], dim=-1)  # [B, T-1, D]
        B, T_minus_1, D = anchors.shape
        a = anchors.reshape(B * T_minus_1, D)
        p = positives.reshape(B * T_minus_1, D)
        logits = torch.matmul(a, p.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        info_nce = F.cross_entropy(logits, labels)

        # VICReg-style variance term: penalise if std across batch drops
        # below target — prevents all representations collapsing to constant.
        if self.variance_weight > 0:
            flat = fused_states.reshape(-1, D)  # [B*T, D]
            std = flat.std(dim=0)  # [D]
            var_loss = F.relu(self.variance_target - std).mean()
            return info_nce + self.variance_weight * var_loss

        return info_nce


class SlowFastAgreementLoss(nn.Module):
    """Slow stream output should be a smoothed version of recent fast outputs.
    Prevents slow/fast divergence.

    L-14 fix: removed .detach() so both streams receive gradients
    (bidirectional alignment).
    """

    def forward(self, fast_tokens: Tensor, slow_token: Tensor) -> Tensor:
        """
        fast_tokens: [B, T, D] — fast stream outputs over time
        slow_token: [B, D] — latest slow stream output
        """
        T = fast_tokens.shape[1]
        weights = torch.exp(torch.linspace(-2, 0, T, device=fast_tokens.device))
        weights = weights / weights.sum()
        fast_ema = (fast_tokens * weights[None, :, None]).sum(dim=1)  # [B, D]
        # L-14: bidirectional — both slow and fast get gradients
        return F.mse_loss(slow_token, fast_ema)


class ActionConsistencyLoss(nn.Module):
    """Multi-head action consistency: discrete and continuous predictions
    should agree in the original action space.

    L-13 fix: replaced projection+cosine (which could collapse to constant
    output) with direct MSE in the 14D action space.
    """

    def __init__(self, action_dim: int = 14, embed_dim: int = 256) -> None:
        super().__init__()
        # Keep parameters for backward compatibility but they are unused
        self.action_dim = action_dim

    def forward(self, discrete_actions: Tensor, continuous_actions: Tensor) -> Tensor:
        return F.mse_loss(discrete_actions, continuous_actions)


class V2ConsistencyLoss(nn.Module):
    """Combined v2 consistency loss.

    L-18: sub-weights are now constructor arguments (configurable).
    """

    def __init__(self, action_dim: int = 14,
                 temperature: float = 0.1,
                 slow_fast_weight: float = 0.5,
                 action_weight: float = 0.5) -> None:
        super().__init__()
        self.temporal = ContrastiveTemporalLoss(temperature=temperature)
        self.slow_fast = SlowFastAgreementLoss()
        self.action = ActionConsistencyLoss(action_dim)
        self.slow_fast_weight = slow_fast_weight
        self.action_weight = action_weight

    def forward(self, fused_states: Tensor,
                fast_tokens: Optional[Tensor] = None,
                slow_token: Optional[Tensor] = None,
                discrete_actions: Optional[Tensor] = None,
                continuous_actions: Optional[Tensor] = None) -> Tensor:
        loss = self.temporal(fused_states)
        if fast_tokens is not None and slow_token is not None:
            loss = loss + self.slow_fast_weight * self.slow_fast(fast_tokens, slow_token)
        if discrete_actions is not None and continuous_actions is not None:
            loss = loss + self.action_weight * self.action(discrete_actions, continuous_actions)
        return loss
