"""Object-Centric Physics Engine — v0.3 §6.

6-layer GNN with multi-head attention for object interaction modelling.
Physics inductive biases are injected through architecture, not loss functions:
1. Residual connections → object inertia (state unchanged without interaction)
2. Attention-based message passing → soft symmetric interaction graph
3. Intrinsic/extrinsic attribute separation → intrinsic properties don't change fast

Parameters: ~35M (for 48 compressed slots at d_node=512, 6 GNN layers)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PhysicsGNNLayer(nn.Module):
    """Attention-based GNN layer for object interaction.

    Uses ``nn.MultiheadAttention`` which dispatches to FlashAttention
    on supported hardware.  Attention weights serve as the soft
    interaction graph (no extra parameters needed).
    """

    def __init__(self, d_node: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_node)
        self.attn = nn.MultiheadAttention(d_node, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, d_node * 2),
            nn.SiLU(),
            nn.Linear(d_node * 2, d_node),
        )

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        h_normed = self.norm(h)
        attn_out, attn_weights = self.attn(
            h_normed, h_normed, h_normed, need_weights=True
        )
        h = h + attn_out
        h = h + self.ffn(h)
        return h, attn_weights


class ObjectPhysicsEngine(nn.Module):
    """Object-centric dynamics prediction with GNN.

    Parameters
    ----------
    d_model : int
        Slot feature dimension (matches grounder output, e.g. 2048).
    num_slots : int
        Number of object slots to process.
    action_dim : int
        Robot action dimension.
    z_dim : int
        Full world-state dimension (z_det + z_sto).
    d_node : int
        Internal GNN node dimension.
    num_gnn_layers : int
        Number of GNN interaction layers.
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_slots: int = 24,
        action_dim: int = 14,
        z_dim: int = 4096,
        d_node: int = 512,
        num_gnn_layers: int = 6,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots

        # Slot encoder
        self.slot_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_node),
            nn.SiLU(),
            nn.Linear(d_node, d_node),
        )

        # Intrinsic / extrinsic separation
        self.intrinsic_proj = nn.Linear(d_node, d_node // 2)
        self.extrinsic_proj = nn.Linear(d_node, d_node // 2)

        # Action + global context → per-slot modulation
        self.action_context = nn.Sequential(
            nn.Linear(action_dim + z_dim, d_node),
            nn.SiLU(),
            nn.Linear(d_node, d_node),
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList(
            [PhysicsGNNLayer(d_node, num_heads=8) for _ in range(num_gnn_layers)]
        )

        # Output projection (residual)
        self.slot_output = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, d_model),
        )

    def forward(
        self,
        object_slots: Tensor,
        action: Tensor,
        z_full: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        """
        Returns
        -------
        next_slots : [B, S, d_model]
        interaction_weights : [B, S, S]  (from last GNN layer)
        intrinsic : [B, S, d_node//2]
        next_intrinsic : [B, S, d_node//2]
        """
        B, S, D = object_slots.shape

        h = self.slot_encoder(object_slots)  # [B, S, d_node]

        # Attribute separation
        intrinsic = self.intrinsic_proj(h)
        extrinsic = self.extrinsic_proj(h)  # noqa: F841 (kept for symmetry)

        # Action context broadcast
        ctx = self.action_context(
            torch.cat([action, z_full], dim=-1)
        )  # [B, d_node]
        h = h + ctx.unsqueeze(1)

        # GNN message passing
        interaction_weights: Optional[Tensor] = None
        for layer in self.gnn_layers:
            h, attn_w = layer(h)
            interaction_weights = attn_w

        # Residual prediction (inertia bias)
        next_slots = object_slots + self.slot_output(h)

        # Intrinsic consistency check
        next_h = self.slot_encoder(next_slots)
        next_intrinsic = self.intrinsic_proj(next_h)

        return next_slots, interaction_weights, intrinsic, next_intrinsic
