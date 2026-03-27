"""Hierarchical Attention Grounder for HybridVLA v2.

v2 changes from v1:
- 96 latents: global(1) + 48 object slots + phase(1) + uncertainty(1)
  + affordance(1) + 44 auxiliary
- 8 layers (up from 4), 2048d, 16 heads
- Hierarchical compression: after layer 4, compress 48 raw slots → 24 refined
  via learned cross-attention routing
- New affordance_token output
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vla_hybrid_v2.types import GrounderOutput


# ---------------------------------------------------------------------------
# Building blocks (same pattern as v1, scaled to 2048d)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.ffn = FeedForward(dim, mlp_ratio, dropout=dropout)

    def forward(self, queries: Tensor, context: Tensor,
                context_mask: Optional[Tensor] = None) -> Tensor:
        B, Nq, D = queries.shape
        Nkv = context.shape[1]
        q = self.q_proj(self.norm_q(queries)).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.norm_kv(context)
        k = self.k_proj(kv).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        # SDPA: auto-dispatches to Flash Attention / Memory-Efficient backend
        sdpa_mask = None
        if context_mask is not None:
            # Additive float mask: 0 = attend, -inf = ignore
            sdpa_mask = torch.zeros(B, 1, 1, Nkv, device=q.device, dtype=q.dtype)
            sdpa_mask = sdpa_mask.masked_fill(~context_mask[:, None, None, :], float("-inf"))
        drop_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, dropout_p=drop_p,
        ).transpose(1, 2).contiguous().view(B, Nq, D)
        x = queries + self.proj_drop(self.out_proj(out))
        x = x + self.ffn(x)
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.ffn = FeedForward(dim, mlp_ratio, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        drop_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop_p,
        ).transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.proj_drop(self.out_proj(out))
        x = x + self.ffn(x)
        return x


class GrounderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.cross_attn = CrossAttentionLayer(dim, num_heads, mlp_ratio, dropout)
        self.self_attn = SelfAttentionLayer(dim, num_heads, mlp_ratio, dropout)

    def forward(self, latents: Tensor, context: Tensor,
                context_mask: Optional[Tensor] = None) -> Tensor:
        latents = self.cross_attn(latents, context, context_mask)
        latents = self.self_attn(latents)
        return latents


# ---------------------------------------------------------------------------
# Slot Compression (v2 new)
# ---------------------------------------------------------------------------

class SlotCompression(nn.Module):
    """Compress 48 raw object slots → 24 refined slots via learned routing.

    Uses learned routing queries that cross-attend to the raw slots,
    effectively learning which objects to group and prioritize.
    """

    def __init__(self, hidden_size: int = 2048, num_raw: int = 48,
                 num_compressed: int = 24, num_heads: int = 16) -> None:
        super().__init__()
        self.route_queries = nn.Parameter(
            torch.randn(1, num_compressed, hidden_size) * 0.02
        )
        self.cross_attn = CrossAttentionLayer(hidden_size, num_heads)
        self.self_attn = SelfAttentionLayer(hidden_size, num_heads)

    def forward(self, raw_slots: Tensor) -> Tensor:
        """raw_slots: [B, 48, D] → compressed: [B, 24, D]"""
        B = raw_slots.shape[0]
        queries = self.route_queries.expand(B, -1, -1)
        compressed = self.cross_attn(queries, raw_slots)
        compressed = self.self_attn(compressed)
        return compressed


# ---------------------------------------------------------------------------
# Hierarchical Attention Grounder
# ---------------------------------------------------------------------------

class HierarchicalAttentionGrounder(nn.Module):
    """v2 Grounder: 96 latents, 8 layers, hierarchical compression.

    Slot layout:
      [global(1), objects(48), phase(1), uncertainty(1), affordance(1), aux(44)]
      Total = 96 latents

    Processing:
      Layers 0-3: all 96 latents cross-attend to backbone features
      Slot compression: 48 raw objects → 24 compressed objects
      Layers 4-7: continue processing with full latent set
    """

    def __init__(self, hidden_size=2048, num_latents=96,
                 num_object_slots=48, compressed_slots=24,
                 num_layers=8, num_heads=16, mlp_ratio=4.0,
                 dropout=0.0, hierarchical_compression=True,
                 compression_layer=4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_latents = num_latents
        self.num_object_slots = num_object_slots
        self.compressed_slots = compressed_slots
        self.hierarchical_compression = hierarchical_compression
        self.compression_layer = compression_layer

        assert num_latents >= num_object_slots + 4  # global + phase + unc + afford

        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latents, hidden_size) * 0.02
        )

        self.blocks = nn.ModuleList([
            GrounderBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        if hierarchical_compression:
            self.slot_compression = SlotCompression(
                hidden_size, num_object_slots, compressed_slots, num_heads,
            )

        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, backbone_hidden: Tensor,
                attention_mask: Optional[Tensor] = None) -> GrounderOutput:
        B = backbone_hidden.shape[0]
        latents = self.latent_queries.expand(B, -1, -1)

        # Snapshot of raw object slots (saved before mid-layer compression)
        raw_object_slots: Optional[Tensor] = None

        # Process through layers with mid-layer hierarchical compression.
        # Before compression (layers 0 .. compression_layer-1):
        #   Layout: [global(1), objects(48), phase(1), unc(1), aff(1), aux(44)] = 96
        # After compression (layers compression_layer .. end):
        #   Layout: [global(1), compressed(24), phase(1), unc(1), aff(1), aux(44)] = 72
        for i, block in enumerate(self.blocks):
            latents = block(latents, backbone_hidden, attention_mask)

            if (self.hierarchical_compression
                    and i == self.compression_layer - 1):
                obj_start = 1
                obj_end = obj_start + self.num_object_slots
                raw_object_slots = latents[:, obj_start:obj_end, :].clone()
                compressed_slots = self.slot_compression(raw_object_slots)
                latents = torch.cat([
                    latents[:, :obj_start, :],      # global(1)
                    compressed_slots,                 # compressed(24)
                    latents[:, obj_end:, :],         # phase + unc + aff + aux
                ], dim=1)

        latents = self.final_norm(latents)

        # Carve out slots — layout depends on whether compression happened
        global_token = latents[:, 0, :]

        if self.hierarchical_compression:
            # Post-compression layout:
            # [global(1), compressed(24), phase(1), unc(1), aff(1), aux(44)]
            c_end = 1 + self.compressed_slots
            compressed = latents[:, 1:c_end, :]
            phase_token = latents[:, c_end, :]
            uncertainty_token = latents[:, c_end + 1, :]
            affordance_token = latents[:, c_end + 2, :]
        else:
            # Original layout (no compression):
            # [global(1), objects(48), phase(1), unc(1), aff(1), aux(44)]
            obj_end = 1 + self.num_object_slots
            raw_object_slots = latents[:, 1:obj_end, :]
            compressed = raw_object_slots[:, :self.compressed_slots]
            phase_token = latents[:, obj_end, :]
            uncertainty_token = latents[:, obj_end + 1, :]
            affordance_token = latents[:, obj_end + 2, :]

        return GrounderOutput(
            global_token=global_token,
            object_slots=(raw_object_slots if raw_object_slots is not None
                          else compressed),
            compressed_object_slots=compressed,
            phase_token=phase_token,
            uncertainty_token=uncertainty_token,
            affordance_token=affordance_token,
        )
