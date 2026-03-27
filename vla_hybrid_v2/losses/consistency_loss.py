"""v2 Consistency losses: contrastive temporal + slow-fast agreement.

v2 changes: replaces simple L2 smoothness with InfoNCE-style contrastive
loss and adds slow-fast agreement term.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ContrastiveTemporalLoss(nn.Module):
    """InfoNCE-style: consecutive fused states should be more similar to each
    other than to random states in the batch. Learns meaningful temporal
    structure rather than just penalizing magnitude changes.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, fused_states: Tensor) -> Tensor:
        """fused_states: [B, T, D]"""
        if fused_states.shape[1] < 2:
            return torch.tensor(0.0, device=fused_states.device)
        anchors = F.normalize(fused_states[:, :-1], dim=-1)  # [B, T-1, D]
        positives = F.normalize(fused_states[:, 1:], dim=-1)  # [B, T-1, D]
        B, T_minus_1, D = anchors.shape
        # Flatten: each anchor-positive pair is one sample
        a = anchors.reshape(B * T_minus_1, D)
        p = positives.reshape(B * T_minus_1, D)
        # Similarity matrix: [N, N] where N = B*(T-1)
        logits = torch.matmul(a, p.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)


class SlowFastAgreementLoss(nn.Module):
    """Slow stream output should be a smoothed version of recent fast outputs.
    Prevents slow/fast divergence.
    """

    def forward(self, fast_tokens: Tensor, slow_token: Tensor) -> Tensor:
        """
        fast_tokens: [B, T, D] — fast stream outputs over time
        slow_token: [B, D] — latest slow stream output
        """
        # Exponential moving average of fast tokens (simulate what slow should see)
        T = fast_tokens.shape[1]
        weights = torch.exp(torch.linspace(-2, 0, T, device=fast_tokens.device))
        weights = weights / weights.sum()
        fast_ema = (fast_tokens * weights[None, :, None]).sum(dim=1)  # [B, D]
        return F.mse_loss(slow_token, fast_ema.detach())


class ActionConsistencyLoss(nn.Module):
    """Multi-head action consistency: project discrete and continuous
    predictions into shared embedding space, compute cosine similarity.
    """

    def __init__(self, action_dim: int = 14, embed_dim: int = 256) -> None:
        super().__init__()
        self.discrete_proj = nn.Linear(action_dim, embed_dim)
        self.continuous_proj = nn.Linear(action_dim, embed_dim)

    def forward(self, discrete_actions: Tensor, continuous_actions: Tensor) -> Tensor:
        d = F.normalize(self.discrete_proj(discrete_actions), dim=-1)
        c = F.normalize(self.continuous_proj(continuous_actions), dim=-1)
        return 1.0 - (d * c).sum(dim=-1).mean()


class V2ConsistencyLoss(nn.Module):
    """Combined v2 consistency loss."""

    def __init__(self, action_dim: int = 14) -> None:
        super().__init__()
        self.temporal = ContrastiveTemporalLoss()
        self.slow_fast = SlowFastAgreementLoss()
        self.action = ActionConsistencyLoss(action_dim)

    def forward(self, fused_states: Tensor,
                fast_tokens: Optional[Tensor] = None,
                slow_token: Optional[Tensor] = None,
                discrete_actions: Optional[Tensor] = None,
                continuous_actions: Optional[Tensor] = None) -> Tensor:
        loss = self.temporal(fused_states)
        if fast_tokens is not None and slow_token is not None:
            loss = loss + 0.5 * self.slow_fast(fast_tokens, slow_token)
        if discrete_actions is not None and continuous_actions is not None:
            loss = loss + 0.5 * self.action(discrete_actions, continuous_actions)
        return loss
