"""Imagination Engine — v0.4 revised.

v0.4 fixes vs v0.3:
- ImaginationTrajectory now stores raw logits + physics outputs for loss
- _single_step returns prior_logits and full physics outputs
- Physics intrinsic/next_intrinsic/interaction_weights are preserved
- Visual decoder optionally integrated
- SubgoalPlanner optionally integrated
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from vla_hybrid_v2.experimental.world_model.imagination_mamba import ImaginationMamba
from vla_hybrid_v2.experimental.world_model.noise_augmentation import NoiseAugmentation
from vla_hybrid_v2.experimental.world_model.object_physics import ObjectPhysicsEngine
from vla_hybrid_v2.experimental.world_model.stochastic_state import StochasticStateModule
from vla_hybrid_v2.experimental.world_model.subgoal_planner import LatentSubgoalPlanner
from vla_hybrid_v2.experimental.world_model.visual_decoder import CNNWorldDecoder
from vla_hybrid_v2.experimental.world_model.world_model_heads import WorldModelHeads


@dataclass
class ImaginationTrajectory:
    """Container for a multi-step imagination rollout.

    v0.4: includes raw logits and physics outputs required by WorldModelLoss.
    """

    states: Tensor                                  # [B, H, z_dim]
    rewards: Tensor                                 # [B, H]
    values: Tensor                                  # [B, H]
    dones: Tensor                                   # [B, H]
    actions: Tensor                                 # [B, H, action_dim]
    slots: Optional[Tensor] = None                  # [B, H, S, d_model]

    # ---- raw logits for loss computation (v0.4 fix) ----
    reward_logits: Optional[Tensor] = None          # [B, H, reward_bins]
    value_logits: Optional[Tensor] = None           # [B, H, value_bins]
    done_logits: Optional[Tensor] = None            # [B, H, 1]
    prior_logits: Optional[Tensor] = None           # [B, H, n_cat*n_cls]

    # ---- physics outputs for loss computation (v0.4 fix) ----
    intrinsic: Optional[Tensor] = None              # [B, H, S, d_node//2]
    next_intrinsic: Optional[Tensor] = None         # [B, H, S, d_node//2]
    interaction_weights: Optional[Tensor] = None    # [B, H, S, S]

    # ---- visual predictions (optional) ----
    predicted_images: Optional[Tensor] = None       # [B, H, 3, 112, 112]


class ImaginationEngine(nn.Module):
    """32-step latent imagination with full loss-data collection."""

    def __init__(
        self,
        d_model: int = 2048,
        action_dim: int = 14,
        num_slots: int = 24,
        horizon: int = 32,
        checkpoint_every: int = 8,
        n_categories: int = 48,
        n_classes: int = 48,
        enable_visual_decoder: bool = True,
        enable_subgoal_planner: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.z_dim = 2 * d_model
        self.horizon = horizon
        self.checkpoint_every = checkpoint_every

        # Core components
        self.stochastic = StochasticStateModule(
            d_model=d_model, n_categories=n_categories, n_classes=n_classes,
        )
        self.dynamics = ImaginationMamba(
            d_model=d_model, d_state=128, num_layers=8, action_dim=action_dim,
        )
        self.noise_aug = NoiseAugmentation(z_dim=self.z_dim)
        self.heads = WorldModelHeads(z_dim=self.z_dim)
        self.physics = ObjectPhysicsEngine(
            d_model=d_model, num_slots=num_slots,
            action_dim=action_dim, z_dim=self.z_dim,
        )

        # Slot decoder for imagination (no real Grounder available)
        self.slot_decoder = nn.Sequential(
            nn.LayerNorm(self.z_dim),
            nn.Linear(self.z_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_slots * d_model),
        )
        self._num_slots = num_slots

        # Optional components (v0.4: integrated, not orphaned)
        self.visual_decoder: Optional[CNNWorldDecoder] = None
        if enable_visual_decoder:
            self.visual_decoder = CNNWorldDecoder(z_dim=self.z_dim)

        self.subgoal_planner: Optional[LatentSubgoalPlanner] = None
        if enable_subgoal_planner:
            self.subgoal_planner = LatentSubgoalPlanner(
                z_dim=self.z_dim, d_model=d_model,
            )

    def _decode_slots(self, z_full: Tensor) -> Tensor:
        B = z_full.shape[0]
        return self.slot_decoder(z_full).view(B, self._num_slots, self.d_model)

    def _single_step(
        self,
        z_det: Tensor,
        z_full: Tensor,
        action: Tensor,
        step_idx: int,
        ssm_states: Optional[list],
        conv_states: Optional[list],
        training: bool,
    ) -> dict:
        """One imagination step — returns all data needed by loss."""
        # Noise augmentation
        z_noisy, noise_emb = self.noise_aug.augment(
            z_full, step_idx, self.horizon, training,
        )

        # Dynamics transition (with state persistence via .step())
        delta_z, new_ssm, new_conv = self.dynamics(
            z_noisy, action, noise_emb, ssm_states, conv_states,
        )
        z_det_next = z_det + delta_z

        # Stochastic prior
        z_full_next, _, prior_logits = self.stochastic.encode_prior(z_det_next)

        # Prediction heads (keep raw logits!)
        head_out = self.heads(z_full_next)

        # Object physics (keep ALL outputs!)
        slots = self._decode_slots(z_full_next)
        next_slots, interaction_w, intrinsic, next_intrinsic = self.physics(
            slots, action, z_full_next,
        )

        # Visual decode (optional)
        pred_image = None
        if self.visual_decoder is not None:
            pred_image = self.visual_decoder(z_full_next)

        return {
            "z_det": z_det_next,
            "z_full": z_full_next,
            "reward_logits": head_out["reward_logits"],
            "value_logits": head_out["value_logits"],
            "done_logit": head_out["done_logit"],
            "prior_logits": prior_logits,
            "next_slots": next_slots,
            "intrinsic": intrinsic,
            "next_intrinsic": next_intrinsic,
            "interaction_weights": interaction_w,
            "pred_image": pred_image,
            "ssm_states": new_ssm,
            "conv_states": new_conv,
        }

    def rollout(
        self,
        z_det_init: Tensor,
        policy: nn.Module,
        training: bool = True,
    ) -> ImaginationTrajectory:
        """Multi-step imagination rollout with full loss-data collection."""
        z_full, _, _ = self.stochastic.encode_prior(z_det_init)
        z_det = z_det_init

        ssm_states: Optional[list] = None
        conv_states: Optional[list] = None

        # Collectors
        C: dict[str, List[Tensor]] = {
            k: [] for k in [
                "z_full", "action", "reward_logits", "value_logits",
                "done_logit", "prior_logits", "next_slots",
                "intrinsic", "next_intrinsic", "interaction_weights",
            ]
        }
        pred_images: List[Optional[Tensor]] = []

        for t in range(self.horizon):
            with torch.no_grad():
                action = policy(z_full)

            step_out = self._single_step(
                z_det, z_full, action, t, ssm_states, conv_states, training,
            )

            z_det = step_out["z_det"]
            z_full = step_out["z_full"]
            ssm_states = step_out["ssm_states"]
            conv_states = step_out["conv_states"]

            C["z_full"].append(z_full)
            C["action"].append(action)
            C["reward_logits"].append(step_out["reward_logits"])
            C["value_logits"].append(step_out["value_logits"])
            C["done_logit"].append(step_out["done_logit"])
            C["prior_logits"].append(step_out["prior_logits"])
            C["next_slots"].append(step_out["next_slots"])
            C["intrinsic"].append(step_out["intrinsic"])
            C["next_intrinsic"].append(step_out["next_intrinsic"])
            C["interaction_weights"].append(step_out["interaction_weights"])
            pred_images.append(step_out["pred_image"])

        def _stack(lst: list) -> Tensor:
            return torch.stack(lst, dim=1)

        r_logits = _stack(C["reward_logits"])
        v_logits = _stack(C["value_logits"])

        return ImaginationTrajectory(
            states=_stack(C["z_full"]),
            rewards=self.heads.decode_reward(r_logits.view(-1, r_logits.shape[-1])).view(r_logits.shape[:2]),
            values=self.heads.decode_value(v_logits.view(-1, v_logits.shape[-1])).view(v_logits.shape[:2]),
            dones=torch.sigmoid(_stack(C["done_logit"]).squeeze(-1)),
            actions=_stack(C["action"]),
            slots=_stack(C["next_slots"]),
            reward_logits=r_logits,
            value_logits=v_logits,
            done_logits=_stack(C["done_logit"]),
            prior_logits=_stack(C["prior_logits"]),
            intrinsic=_stack(C["intrinsic"]),
            next_intrinsic=_stack(C["next_intrinsic"]),
            interaction_weights=_stack(C["interaction_weights"]),
            predicted_images=(
                _stack(pred_images)
                if pred_images and all(img is not None for img in pred_images)
                else None
            ),
        )
