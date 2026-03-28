"""Flow Action Expert v2: 18-layer hybrid Mamba/Attention with AdaRMSNorm.

v2 changes from v1:
- 18 layers (M-M-A × 6), d_model=1536, num_heads=24, d_state=96
- AdaRMSNorm conditioned on flow timestep (from π₀.5)
- Midpoint ODE solver for inference (2nd-order accuracy)
- chunk_horizon=24, cond_tokens=32
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vla_hybrid_v2.ops.selective_scan import (
    HAS_MAMBA_CUDA,
    selective_scan_fn,
    ssm_scan,
)
from vla_hybrid_v2.types import ActionExpertOutput

# L-16: detect official Mamba2 block for optional use in ExpertMambaBlock
try:
    from mamba_ssm.modules.mamba2 import Mamba2 as _OfficialMamba2
    HAS_MAMBA2_MODULE = True
except ImportError:
    HAS_MAMBA2_MODULE = False


# ---------------------------------------------------------------------------
# AdaRMSNorm (from π₀.5)
# ---------------------------------------------------------------------------

class AdaRMSNorm(nn.Module):
    """Adaptive RMS Normalization conditioned on a context vector.

    Produces (scale, shift, gate) from the condition, enabling the flow
    timestep to multiplicatively modulate features — critical for
    high-quality denoising at each noise level.
    """

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.eps = 1e-6
        self.cond_proj = nn.Linear(cond_dim, 3 * dim)
        # v0.7.1: Initialize gate bias to +2 so sigmoid(gate) ≈ 0.88 at
        # init, preventing activation halving through 18 residual layers.
        with torch.no_grad():
            self.cond_proj.bias.data[2 * dim:].fill_(2.0)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        x_normed = x * rms
        scale, shift, gate = self.cond_proj(cond).chunk(3, dim=-1)
        return gate.sigmoid() * (x_normed * (1 + scale) + shift)


# ---------------------------------------------------------------------------
# Timestep & positional embeddings
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_len, dim)

    def forward(self, seq_len: int) -> Tensor:
        pos = torch.arange(seq_len, device=self.embedding.weight.device)
        return self.embedding(pos).unsqueeze(0)


# ---------------------------------------------------------------------------
# Expert Mamba Block with AdaRMSNorm
# ---------------------------------------------------------------------------

class ExpertMambaBlock(nn.Module):
    """Mamba-2 block for the action expert. Uses AdaRMSNorm."""

    def __init__(self, d_model=1536, d_state=96, d_conv=4, expand=2,
                 cond_dim: int = 1536) -> None:
        super().__init__()
        self.d_inner = d_model * expand
        self.norm = AdaRMSNorm(d_model, cond_dim)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )
        dt_rank = max(d_model // 16, 1)
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank

    def forward(self, x: Tensor, t_cond: Tensor) -> Tensor:
        residual = x
        x = self.norm(x, t_cond)
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_main, z = xz.split(self.d_inner, dim=-1)
        x_conv = x_main.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_main = F.silu(x_conv.transpose(1, 2))
        A = -torch.exp(self.A_log)
        x_dbl = self.x_proj(x_main)
        dt_raw, B_ssm, C_ssm = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1,
        )

        if HAS_MAMBA_CUDA:
            # CUDA fast path — fused SSM + D residual + silu(z) gating.
            # ExpertMambaBlock always starts from zero state, so
            # selective_scan_fn (which does not accept initial state) is safe.
            dt_proj_out = self.dt_proj(dt_raw)  # [B, L, d_inner], before softplus
            y = selective_scan_fn(
                x_main.transpose(1, 2).contiguous(),       # u   [B, D, L]
                dt_proj_out.transpose(1, 2).contiguous(),   # dt  [B, D, L]
                A.contiguous(),                              # A   [D, N]
                B_ssm.transpose(1, 2).contiguous(),         # B   [B, N, L]
                C_ssm.transpose(1, 2).contiguous(),         # C   [B, N, L]
                D=self.D.float(),
                z=z.transpose(1, 2).contiguous(),           # gate [B, D, L]
                delta_softplus=True,
            )
            y = y.transpose(1, 2)  # [B, L, d_inner]
        else:
            # Fallback: JIT-compiled Python scan
            dt = F.softplus(self.dt_proj(dt_raw))
            dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))
            dBx = (dt.unsqueeze(-1) * B_ssm.unsqueeze(2)) * x_main.unsqueeze(-1)
            state = torch.zeros(
                B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype,
            )
            y, _ = ssm_scan(dA, dBx, C_ssm, state)
            y = y + x_main * self.D.unsqueeze(0).unsqueeze(0)
            y = y * F.silu(z)

        return residual + self.out_proj(y)


# ---------------------------------------------------------------------------
# Expert Attention Block with AdaRMSNorm
# ---------------------------------------------------------------------------

class ExpertAttentionBlock(nn.Module):
    """Cross-attention + self-attention + FFN, all with AdaRMSNorm."""

    def __init__(self, d_model=1536, num_heads=24, cond_dim=1536,
                 mlp_ratio=4.0, dropout=0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Cross-attention with AdaRMSNorm
        self.norm_cross_q = AdaRMSNorm(d_model, cond_dim)
        self.norm_cross_kv = nn.LayerNorm(cond_dim)  # cond prefix uses standard norm
        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_k = nn.Linear(cond_dim, d_model)
        self.cross_v = nn.Linear(cond_dim, d_model)
        self.cross_out = nn.Linear(d_model, d_model)

        # Self-attention with AdaRMSNorm
        self.norm_self = AdaRMSNorm(d_model, cond_dim)
        self.self_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_out = nn.Linear(d_model, d_model)

        # FFN with AdaRMSNorm
        hidden = int(d_model * mlp_ratio)
        self.norm_ffn = AdaRMSNorm(d_model, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, d_model),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def _mha(self, q, k, v):
        drop_p = self.attn_drop.p if self.training else 0.0
        return F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)

    def forward(self, x: Tensor, cond: Tensor, t_cond: Tensor) -> Tensor:
        B, L, D = x.shape
        C = cond.shape[1]

        # Cross-attention
        q = self.cross_q(self.norm_cross_q(x, t_cond)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kv_in = self.norm_cross_kv(cond)
        k = self.cross_k(kv_in).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.cross_v(kv_in).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        cross_out = self._mha(q, k, v).transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.proj_drop(self.cross_out(cross_out))

        # Self-attention
        h = self.norm_self(x, t_cond)
        qkv = self.self_qkv(h).view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        sq, sk, sv = qkv.unbind(0)
        self_out = self._mha(sq, sk, sv).transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.proj_drop(self.self_out(self_out))

        # FFN
        x = x + self.ffn(self.norm_ffn(x, t_cond))
        return x


# ---------------------------------------------------------------------------
# Flow Action Expert v2
# ---------------------------------------------------------------------------

class FlowActionExpert(nn.Module):
    """18-layer hybrid expert with AdaRMSNorm and midpoint ODE solver.

    Pattern: M, M, A, M, M, A, M, M, A, M, M, A, M, M, A, M, M, A
    """

    PATTERN = ["mamba", "mamba", "attn"] * 6  # 18 layers

    def __init__(self, d_model=1536, num_layers=18, num_heads=24,
                 chunk_horizon=24, action_dim=14, d_state=96,
                 d_conv=4, expand=2, cond_dim=2048,
                 cond_tokens=32, dropout=0.0) -> None:
        super().__init__()
        assert num_layers == 18
        self.d_model = d_model
        self.chunk_horizon = chunk_horizon
        self.action_dim = action_dim
        self.cond_tokens = cond_tokens

        # Embeddings
        self.action_proj = nn.Linear(action_dim, d_model)
        self.action_pos_emb = LearnedPositionalEmbedding(chunk_horizon, d_model)
        self.timestep_emb = SinusoidalTimestepEmbedding(d_model)

        # Timestep → condition for AdaRMSNorm (separate from additive embedding)
        self.t_cond_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )

        self.cond_proj = nn.Linear(cond_dim, d_model) if cond_dim != d_model else nn.Identity()
        self.proprio_proj = nn.Linear(d_model, d_model)
        self.embodiment_proj = nn.Linear(d_model, d_model)

        # Layers
        # L-16: when official Mamba2 module is available, log it.
        # The ExpertMambaBlock already uses selective_scan_fn CUDA kernel
        # when HAS_MAMBA_CUDA is True. Full Mamba2 block replacement requires
        # adapter work for AdaRMSNorm conditioning — flagged for future.
        if HAS_MAMBA2_MODULE:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "Official mamba_ssm.Mamba2 detected. ExpertMambaBlock uses "
                "selective_scan_fn CUDA kernel for acceleration."
            )
        self.layers = nn.ModuleList()
        for lt in self.PATTERN:
            if lt == "mamba":
                self.layers.append(ExpertMambaBlock(d_model, d_state, d_conv, expand, cond_dim=d_model))
            elif lt == "attn":
                self.layers.append(ExpertAttentionBlock(d_model, num_heads, cond_dim=d_model, dropout=dropout))

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, action_dim)

    def forward(self, noisy_actions, flow_t, cond_prefix,
                proprio_token, embodiment_token) -> ActionExpertOutput:
        B, H, A = noisy_actions.shape
        action_tokens = self.action_proj(noisy_actions) + self.action_pos_emb(H)
        t_emb = self.timestep_emb(flow_t)
        action_tokens = action_tokens + t_emb.unsqueeze(1)

        # AdaRMSNorm condition
        t_cond = self.t_cond_mlp(t_emb)  # [B, d_model]
        # Broadcast to match sequence: [B, d_model] for all tokens
        # Mamba blocks get [B, d_model], Attn blocks get [B, d_model]

        proprio = self.proprio_proj(proprio_token).unsqueeze(1)
        embodiment = self.embodiment_proj(embodiment_token).unsqueeze(1)
        x = torch.cat([proprio, embodiment, action_tokens], dim=1)  # [B, 2+H, d_model]

        cond = self.cond_proj(cond_prefix)

        for layer in self.layers:
            if isinstance(layer, ExpertMambaBlock):
                # Broadcast t_cond to match sequence length
                t_cond_seq = t_cond.unsqueeze(1).expand(-1, x.shape[1], -1)
                x = layer(x, t_cond_seq)
            elif isinstance(layer, ExpertAttentionBlock):
                t_cond_seq = t_cond.unsqueeze(1).expand(-1, x.shape[1], -1)
                x = layer(x, cond, t_cond_seq)

        action_out = x[:, 2:, :]
        velocity = self.out_proj(self.out_norm(action_out))
        return ActionExpertOutput(velocity=velocity, denoised_action=None)

    @torch.no_grad()
    def sample_euler(self, cond_prefix, proprio_token, embodiment_token,
                     num_steps=8) -> Tensor:
        H, B = self.chunk_horizon, cond_prefix.shape[0]
        device, dtype = cond_prefix.device, cond_prefix.dtype
        x = torch.randn(B, H, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device, dtype=dtype)
            out = self.forward(x, t, cond_prefix, proprio_token, embodiment_token)
            x = x + out.velocity * dt
        return x

    @torch.no_grad()
    def sample_midpoint(self, cond_prefix, proprio_token, embodiment_token,
                        num_steps=8) -> Tensor:
        """2nd-order midpoint method — same cost as Euler but ~2× accuracy."""
        H, B = self.chunk_horizon, cond_prefix.shape[0]
        device, dtype = cond_prefix.device, cond_prefix.dtype
        x = torch.randn(B, H, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device, dtype=dtype)
            t_mid = torch.full((B,), (i + 0.5) * dt, device=device, dtype=dtype)
            v1 = self.forward(x, t, cond_prefix, proprio_token, embodiment_token).velocity
            x_mid = x + 0.5 * dt * v1
            v2 = self.forward(x_mid, t_mid, cond_prefix, proprio_token, embodiment_token).velocity
            x = x + dt * v2
        return x

    @torch.no_grad()
    def sample(self, cond_prefix, proprio_token, embodiment_token,
               num_steps=8, solver="midpoint") -> Tensor:
        if solver == "midpoint":
            return self.sample_midpoint(cond_prefix, proprio_token, embodiment_token, num_steps)
        return self.sample_euler(cond_prefix, proprio_token, embodiment_token, num_steps)
