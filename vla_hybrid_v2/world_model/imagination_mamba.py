"""Imagination Transition Network — 8-layer Mamba-2 for latent world dynamics.

v0.4 fix: Uses ``MambaBlock.step()`` for single-token recurrence with
explicit SSM/conv state management.  This ensures state persists across
the 32 imagination steps (fixing the v0.3 regression where official
Mamba2.forward() returned None states, making the model memoryless).

Spec:
    layers:  8
    d_model: 2048
    d_state: 128
    expand:  2
    params:  ~80M
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from vla_hybrid_v2.models.mamba_core import MambaBlock


class ImaginationMamba(nn.Module):
    """Dedicated transition model for latent-space imagination rollouts.

    Uses ``MambaBlock.step()`` — the single-token recurrent API that
    carries SSM and conv state explicitly across steps — so that the
    model retains memory of prior imagination steps regardless of
    whether the official ``mamba_ssm`` CUDA kernel is used.
    """

    def __init__(
        self,
        d_model: int = 2048,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 8,
        action_dim: int = 14,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        z_full_dim = 2 * d_model

        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.input_proj = nn.Sequential(
            nn.LayerNorm(z_full_dim + d_model + d_model),
            nn.Linear(z_full_dim + d_model + d_model, d_model),
            nn.SiLU(),
        )

        self.layers = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand) for _ in range(num_layers)]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        z_full: Tensor,
        action: Tensor,
        noise_embed: Tensor,
        ssm_states: Optional[List[Optional[Tensor]]] = None,
        conv_states: Optional[List[Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, List[Optional[Tensor]], List[Optional[Tensor]]]:
        """Single imagination step with explicit state propagation.

        Parameters
        ----------
        z_full : [B, 2*d_model]
        action : [B, action_dim]
        noise_embed : [B, d_model]
        ssm_states : list of per-layer SSM states (None on first step)
        conv_states : list of per-layer conv states (None on first step)

        Returns
        -------
        delta_z : [B, d_model]
        new_ssm_states : list of [B, ...] per-layer SSM states
        new_conv_states : list of [B, ...] per-layer conv states
        """
        a_emb = self.action_embed(action)
        h = self.input_proj(
            torch.cat([z_full, a_emb, noise_embed], dim=-1)
        )  # [B, d_model]

        # Initialise state lists on first call
        if ssm_states is None:
            ssm_states = [None] * self.num_layers
        if conv_states is None:
            conv_states = [None] * self.num_layers

        new_ssm: List[Optional[Tensor]] = []
        new_conv: List[Optional[Tensor]] = []

        for i, layer in enumerate(self.layers):
            # --- KEY FIX (v0.4) ---
            # Use .step() instead of .forward() so SSM state persists
            # across the 32 imagination steps.
            h, s_new, c_new = layer.step(h, ssm_states[i], conv_states[i])
            new_ssm.append(s_new)
            new_conv.append(c_new)

        delta_z = self.output_proj(h)
        return delta_z, new_ssm, new_conv
