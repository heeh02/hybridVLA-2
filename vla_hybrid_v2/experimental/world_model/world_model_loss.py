"""World Model Loss — v0.4 revised.

v0.4 fixes:
- KL free_bits applied per-category (not to total KL) — DreamerV3 correct
- Removed duplicate slot_smoothness (identical to slot_pred)
- Added visual reconstruction loss integration
- All sub-losses return individual terms for logging
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vla_hybrid_v2.experimental.world_model.visual_decoder import WorldDecoderLoss
from vla_hybrid_v2.experimental.world_model.world_model_heads import SymlogTwoHot


class KLLoss(nn.Module):
    """DreamerV3-style KL loss with per-category free bits and KL balancing.

    v0.4 fix: free_bits is applied **per category** before summing,
    not to the total KL.  With 48 categories and free_bits=1.0,
    the effective minimum total KL is 48 nats (not 1 nat as in v0.3).
    """

    def __init__(self, free_bits: float = 1.0, alpha: float = 0.8) -> None:
        super().__init__()
        self.free_bits = free_bits
        self.alpha = alpha

    def forward(
        self,
        posterior_logits: Tensor,
        prior_logits: Tensor,
        n_categories: int,
        n_classes: int,
    ) -> Tensor:
        post = posterior_logits.view(-1, n_categories, n_classes)
        pri = prior_logits.view(-1, n_categories, n_classes)

        post_dist = torch.distributions.Categorical(logits=post)
        pri_dist = torch.distributions.Categorical(logits=pri)

        # Per-category KL: [B, n_categories]
        kl_fwd = torch.distributions.kl_divergence(post_dist, pri_dist)
        kl_rev = torch.distributions.kl_divergence(pri_dist, post_dist)

        # --- v0.4 FIX: per-category free bits ---
        kl_fwd_clamped = torch.clamp(kl_fwd, min=self.free_bits)
        kl_rev_clamped = torch.clamp(kl_rev, min=self.free_bits)

        # KL balancing (α=0.8): encourage prior to match posterior
        kl = (
            self.alpha * kl_fwd_clamped.sum(-1).mean()
            + (1 - self.alpha) * kl_rev_clamped.sum(-1).mean()
        )
        return kl


class PhysicsLoss(nn.Module):
    """Self-supervised physics constraints.

    v0.4 fix: removed duplicate slot_smoothness (was identical to slot_pred).
    """

    def forward(
        self,
        pred_slots: Tensor,
        target_slots: Tensor,
        intrinsic: Optional[Tensor],
        next_intrinsic: Optional[Tensor],
        interaction_weights: Optional[Tensor],
    ) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}

        losses["slot_pred"] = F.mse_loss(pred_slots, target_slots)

        if intrinsic is not None and next_intrinsic is not None:
            losses["intrinsic_invariance"] = F.mse_loss(
                intrinsic, next_intrinsic.detach()
            )

        if interaction_weights is not None:
            losses["interaction_sparsity"] = interaction_weights.mean()

        return losses


class WorldModelLoss(nn.Module):
    """Combined world model training loss.

    v0.4: integrates visual decoder loss and accepts ImaginationTrajectory
    fields directly.
    """

    def __init__(
        self,
        kl_free_bits: float = 1.0,
        kl_alpha: float = 0.8,
        enable_visual_loss: bool = True,
    ) -> None:
        super().__init__()
        self.kl_loss = KLLoss(free_bits=kl_free_bits, alpha=kl_alpha)
        self.physics_loss = PhysicsLoss()
        self.visual_loss: Optional[WorldDecoderLoss] = None
        if enable_visual_loss:
            self.visual_loss = WorldDecoderLoss()

    def forward(
        self,
        # KL
        posterior_logits: Tensor,
        prior_logits: Tensor,
        n_categories: int,
        n_classes: int,
        # Latent prediction
        z_pred: Optional[Tensor] = None,
        z_true: Optional[Tensor] = None,
        # Reward
        reward_logits: Optional[Tensor] = None,
        reward_target: Optional[Tensor] = None,
        reward_bins: Optional[Tensor] = None,
        # Done
        done_logit: Optional[Tensor] = None,
        done_target: Optional[Tensor] = None,
        # Physics
        pred_slots: Optional[Tensor] = None,
        target_slots: Optional[Tensor] = None,
        intrinsic: Optional[Tensor] = None,
        next_intrinsic: Optional[Tensor] = None,
        interaction_weights: Optional[Tensor] = None,
        # Visual
        pred_image: Optional[Tensor] = None,
        target_image: Optional[Tensor] = None,
        # Weights
        w_kl: float = 1.0,
        w_latent: float = 10.0,
        w_reward: float = 1.0,
        w_done: float = 0.5,
        w_slot: float = 5.0,
        w_intrinsic: float = 2.0,
        w_sparsity: float = 0.1,
        w_visual: float = 1.0,
    ) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}

        # 1. KL (per-category free bits — v0.4 fixed)
        losses["kl"] = w_kl * self.kl_loss(
            posterior_logits, prior_logits, n_categories, n_classes,
        )

        # 2. Latent prediction
        if z_pred is not None and z_true is not None:
            losses["latent_pred"] = w_latent * F.mse_loss(z_pred, z_true)

        # 3. Reward (symlog two-hot)
        if reward_logits is not None and reward_target is not None and reward_bins is not None:
            losses["reward"] = w_reward * SymlogTwoHot.loss(
                reward_logits, reward_target, reward_bins,
            )

        # 4. Done (BCE)
        if done_logit is not None and done_target is not None:
            losses["done"] = w_done * F.binary_cross_entropy_with_logits(
                done_logit.view(-1), done_target.float().view(-1),
            )

        # 5. Physics (no duplicate slot_smoothness — v0.4 fixed)
        if pred_slots is not None and target_slots is not None:
            phys = self.physics_loss(
                pred_slots, target_slots, intrinsic, next_intrinsic,
                interaction_weights,
            )
            losses["slot_pred"] = w_slot * phys["slot_pred"]
            if "intrinsic_invariance" in phys:
                losses["intrinsic_inv"] = w_intrinsic * phys["intrinsic_invariance"]
            if "interaction_sparsity" in phys:
                losses["interaction_sp"] = w_sparsity * phys["interaction_sparsity"]

        # 6. Visual reconstruction (v0.4: integrated, was orphaned)
        if (
            self.visual_loss is not None
            and pred_image is not None
            and target_image is not None
        ):
            vis_losses = self.visual_loss(pred_image, target_image)
            for k, v in vis_losses.items():
                losses[f"visual_{k}"] = w_visual * v

        losses["wm_total"] = sum(losses.values())
        return losses
