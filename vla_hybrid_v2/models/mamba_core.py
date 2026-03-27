"""Tri-Rate Mamba Core for HybridVLA v2.

v2 changes from v1:
- Three streams: Fast (20L, d_state=128) + Medium (6L, d_state=128)
  + Slow (10L, d_state=256)
- Cross-attention fusion replaces scalar gate
- ActionHistoryEncoder (4L Mamba) encodes last K=8 actions
- StaleTimeEncoding at d_model=2048
- All dimensions at 2048

v0.2 revision: Uses the official ``mamba_ssm.modules.mamba2.Mamba2`` block
from https://github.com/state-spaces/mamba instead of hand-rolled SSM
primitives.  Falls back to a minimal pure-PyTorch implementation only when
the ``mamba_ssm`` package is not installed.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from vla_hybrid_v2.types import TemporalOutput, TriRateTemporalState

# ---------------------------------------------------------------------------
# Try to import the official Mamba-2 block from the mamba_ssm library.
# If unavailable, we define a lightweight fallback that matches the API.
# ---------------------------------------------------------------------------

try:
    from mamba_ssm import Mamba2 as _OfficialMamba2  # noqa: F401

    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False
    _OfficialMamba2 = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Stale-time positional encoding
# ---------------------------------------------------------------------------

class StaleTimeEncoding(nn.Module):
    """Sinusoidal encoding of elapsed steps since last semantic refresh."""

    def __init__(self, d_model: int = 2048, max_staleness: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_staleness = max_staleness
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, steps_since_refresh: Tensor) -> Tensor:
        steps = steps_since_refresh.float().clamp(0, self.max_staleness)
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=steps.device, dtype=steps.dtype)
            / half
        )
        args = steps[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_model % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Mamba-2 block wrapper — delegates to official library when available
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Single Mamba-2 block.

    When the ``mamba_ssm`` package is installed the heavy lifting is
    performed by ``mamba_ssm.Mamba2`` which runs fused CUDA kernels.
    Otherwise a pure-PyTorch fallback is used (functional but slower).

    The block always follows the pre-norm residual pattern::

        out = x + out_proj(SSM(SiLU(conv1d(in_proj(LN(x))))))
    """

    def __init__(
        self,
        d_model: int = 2048,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        # v0.9: Per-block residual scale — initialized by _MambaStack to
        # 1/sqrt(2*N) for deep stacks, preventing activation explosion.
        # Works for BOTH official and fallback paths.
        self.res_scale = nn.Parameter(torch.ones(1))

        if HAS_MAMBA_SSM:
            # ---- Official Mamba-2 block (CUDA-accelerated) ----
            self.mamba = _OfficialMamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.norm = nn.LayerNorm(d_model)  # v0.7: pre-norm for official path
            self._use_official = True
        else:
            # ---- Pure-PyTorch fallback ----
            from vla_hybrid_v2.ops.selective_scan import ssm_scan

            self._ssm_scan = ssm_scan
            self.norm = nn.LayerNorm(d_model)
            self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
            self.conv1d = nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )
            dt_rank = max(d_model // 16, 1)
            self.x_proj = nn.Linear(
                self.d_inner, dt_rank + 2 * d_state, bias=False
            )
            self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
            A = (
                torch.arange(1, d_state + 1, dtype=torch.float32)
                .unsqueeze(0)
                .expand(self.d_inner, -1)
            )
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
            self.dt_rank = dt_rank
            self._use_official = False

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: Tensor,
        ssm_state: Optional[Tensor] = None,
        conv_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Parameters
        ----------
        x : [B, L, D] or [B, D] (single-step)
        ssm_state, conv_state : recurrent caches (used by the fallback path
            for temporal-core blocks that carry state across control steps;
            the official Mamba2 manages its own internal cache when called
            via ``step()``).

        Returns
        -------
        out : same shape as *x*
        new_ssm_state, new_conv_state : updated caches (None when using
            the official block in sequence mode — it fuses the recurrence
            internally).
        """
        if self._use_official:
            return self._forward_official(x)
        return self._forward_fallback(x, ssm_state, conv_state)

    def step(
        self,
        x: Tensor,
        ssm_state: Optional[Tensor] = None,
        conv_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Single-token recurrent step with explicit state management.

        This is the API that ImaginationMamba and any other component that
        needs cross-step SSM state persistence should use.  It works
        identically on both the official and fallback paths.

        Parameters
        ----------
        x : [B, D]  (single token, no sequence dimension)
        ssm_state : optional recurrent SSM state.
            - Official Mamba-2: [B, nheads, headdim, d_state]
            - Fallback Mamba-1: [B, d_inner, d_state]
        conv_state : optional recurrent conv state.
            - Official Mamba-2: [B, d_inner + 2*ngroups*d_state, d_conv]
            - Fallback Mamba-1: [B, d_inner, d_conv-1]

        Returns
        -------
        out : [B, D]
        new_ssm_state, new_conv_state : updated states (shapes match input)
        """
        if self._use_official:
            return self._step_official(x, ssm_state, conv_state)
        return self._forward_fallback(x, ssm_state, conv_state)

    def _step_official(
        self,
        x: Tensor,
        ssm_state: Optional[Tensor],
        conv_state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Single-token step through the official Mamba2, managing state
        explicitly so that it persists across imagination steps."""
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # Allocate initial states when missing
        if ssm_state is None:
            ssm_state = torch.zeros(
                B, self.mamba.nheads, self.mamba.headdim, self.d_state,
                device=device, dtype=dtype,
            )
        if conv_state is None:
            conv_state = torch.zeros(
                B, self.d_inner + 2 * self.mamba.ngroups * self.d_state,
                self.d_conv,
                device=device, dtype=dtype,
            )

        # Mamba2.step() expects x: [B, D], returns (out, conv_state, ssm_state)
        out, new_conv_state, new_ssm_state = self.mamba.step(
            self.norm(x), conv_state, ssm_state,  # v0.7: pre-norm
        )
        out = self.res_scale * out + x  # v0.9: scaled residual
        return out, new_ssm_state, new_conv_state

    # ---- Official mamba_ssm path (sequence mode, no cross-step state) ----

    def _forward_official(self, x: Tensor) -> Tuple[Tensor, None, None]:
        is_single = x.dim() == 2
        if is_single:
            x = x.unsqueeze(1)

        residual = x
        out = self.mamba(self.norm(x))  # v0.7: pre-norm before Mamba2
        out = self.res_scale * out + residual  # v0.9: scaled residual

        if is_single:
            out = out.squeeze(1)
        return out, None, None

    # ---- Pure-PyTorch fallback path ----

    def _forward_fallback(
        self,
        x: Tensor,
        ssm_state: Optional[Tensor],
        conv_state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        is_single = x.dim() == 2
        residual = x
        x = self.norm(x)

        if is_single:
            x = x.unsqueeze(1)
            if residual.dim() == 2:
                residual = residual.unsqueeze(1)

        B, L, D = x.shape

        xz = self.in_proj(x)
        x_main, z = xz.split(self.d_inner, dim=-1)

        # Conv1d with state persistence
        x_conv = x_main.transpose(1, 2)  # [B, d_inner, L]
        if conv_state is not None:
            # v0.7: Prepend historical context and use padding=0 logic.
            # Conv1d has padding=d_conv-1 (symmetric), which adds zeros
            # on both sides. When conv_state is prepended, the left
            # zero-padding overwrites the historical context. Fix: take
            # the correct causal slice after convolution.
            x_conv = torch.cat([conv_state, x_conv], dim=-1)
            x_conv = self.conv1d(x_conv)
            # Take the last L elements: these correspond to causal outputs
            # aligned with the input positions.
            x_conv = x_conv[:, :, (self.d_conv - 1):(self.d_conv - 1) + L]
        else:
            x_conv = self.conv1d(x_conv)
            x_conv = x_conv[:, :, :L] if x_conv.shape[-1] > L else x_conv
        new_conv_state = x_main.transpose(1, 2)[:, :, -(self.d_conv - 1) :]

        x_main = F.silu(x_conv.transpose(1, 2))

        # SSM
        A = -torch.exp(self.A_log)
        x_dbl = self.x_proj(x_main)
        dt, B_ssm, C_ssm = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))
        dBx = (dt.unsqueeze(-1) * B_ssm.unsqueeze(2)) * x_main.unsqueeze(-1)

        if ssm_state is None:
            ssm_state = torch.zeros(
                B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype
            )
        y, new_ssm_state = self._ssm_scan(dA, dBx, C_ssm, ssm_state)
        y = y + x_main * self.D.unsqueeze(0).unsqueeze(0)

        y = y * F.silu(z)
        out = self.res_scale * self.out_proj(y) + residual  # v0.9: scaled residual

        if is_single:
            out = out.squeeze(1)
        return out, new_ssm_state, new_conv_state


# ---------------------------------------------------------------------------
# Mamba stacks
# ---------------------------------------------------------------------------

class _MambaStack(nn.Module):
    """Stack of ``MambaBlock`` layers."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.d_inner = d_model * expand
        self.d_state = d_state

        # v0.9: Initialize per-block residual scale to 1/sqrt(N) for deep
        # stacks.  Works for BOTH official Mamba2 and fallback paths (unlike
        # the v0.7.1 out_proj init which only reached the fallback path).
        # With N residual additions each scaled by 1/sqrt(N), total variance
        # from residual branch ≈ N * (1/sqrt(N))^2 = 1, maintaining unit scale.
        init_scale = 1.0 / math.sqrt(num_layers) if num_layers > 1 else 1.0
        for layer in self.layers:
            layer.res_scale.data.fill_(init_scale)
        self.d_conv = d_conv

    def init_states(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        # v0.7: Create correct state shapes depending on official vs fallback
        uses_official = self.layers[0]._use_official if self.layers else False
        if uses_official:
            layer0 = self.layers[0]
            nheads = layer0.mamba.nheads
            headdim = layer0.mamba.headdim
            ngroups = layer0.mamba.ngroups
            ssm = [
                torch.zeros(
                    batch_size, nheads, headdim, self.d_state,
                    device=device, dtype=dtype,
                )
                for _ in range(self.num_layers)
            ]
            conv = [
                torch.zeros(
                    batch_size,
                    self.d_inner + 2 * ngroups * self.d_state,
                    self.d_conv,
                    device=device, dtype=dtype,
                )
                for _ in range(self.num_layers)
            ]
        else:
            ssm = [
                torch.zeros(
                    batch_size, self.d_inner, self.d_state,
                    device=device, dtype=dtype,
                )
                for _ in range(self.num_layers)
            ]
            conv = [
                torch.zeros(
                    batch_size, self.d_inner, self.d_conv - 1,
                    device=device, dtype=dtype,
                )
                for _ in range(self.num_layers)
            ]
        return ssm, conv

    def forward(
        self,
        x: Tensor,
        ssm_states: Optional[List[Tensor]] = None,
        conv_states: Optional[List[Tensor]] = None,
        use_checkpoint: bool = False,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """Process a token sequence and return output + final recurrent states.

        v0.5 fix: When using the official Mamba2 CUDA path, processes the
        input sequence token-by-token via ``MambaBlock.step()`` so that
        per-layer SSM/conv states are explicitly captured and returned.
        This ensures state persists across temporal steps in the VLA
        Tri-Rate Core (fixing the v0.3-era regression).

        The fallback path already returned states correctly.
        """
        B = x.shape[0] if x.dim() >= 2 else 1
        L = x.shape[1] if x.dim() == 3 else 1
        device, dtype = x.device, x.dtype
        uses_official = self.layers[0]._use_official if self.layers else False

        if ssm_states is None or conv_states is None:
            ssm_states_list: List[Optional[Tensor]] = [None] * self.num_layers
            conv_states_list: List[Optional[Tensor]] = [None] * self.num_layers
        else:
            ssm_states_list = list(ssm_states)
            conv_states_list = list(conv_states)

        if uses_official:
            # --- v0.5 FIX: token-by-token step() to capture final state ---
            # Official Mamba2.forward() is fused but returns (out, None, None).
            # We use .step() per token so states persist across temporal steps.
            # Cost: loses intra-sequence parallelism (L=33 Python loop), but
            # still uses CUDA step kernel.  For L≤33 this is acceptable.
            is_single = x.dim() == 2
            if is_single:
                x = x.unsqueeze(1)

            out = torch.empty_like(x)
            for t in range(x.shape[1]):
                x_t = x[:, t, :]  # [B, D]
                for i, layer in enumerate(self.layers):
                    x_t, ssm_states_list[i], conv_states_list[i] = layer.step(
                        x_t, ssm_states_list[i], conv_states_list[i],
                    )
                out[:, t, :] = x_t

            if is_single:
                out = out.squeeze(1)

            return out, ssm_states_list, conv_states_list  # type: ignore[return-value]

        # --- Fallback path (already correct) ---
        new_ssm: List[Tensor] = []
        new_conv: List[Tensor] = []

        for i, layer in enumerate(self.layers):
            s_i = ssm_states_list[i]
            c_i = conv_states_list[i]

            if use_checkpoint and self.training:
                x, s, c = activation_checkpoint(
                    layer, x, s_i, c_i, use_reentrant=False,
                )
            else:
                x, s, c = layer(x, s_i, c_i)

            new_ssm.append(s)
            new_conv.append(c)

        return x, new_ssm, new_conv


class FastMamba(_MambaStack):
    """Fast stream — 20 layers, d_state=128. Updated every control step."""

    def __init__(
        self, d_model: int = 2048, d_state: int = 128, d_conv: int = 4, expand: int = 2
    ):
        super().__init__(20, d_model, d_state, d_conv, expand)


class MediumMamba(_MambaStack):
    """Medium stream — 6 layers, d_state=128. Updated every 2nd control step."""

    def __init__(
        self, d_model: int = 2048, d_state: int = 128, d_conv: int = 4, expand: int = 2
    ):
        super().__init__(6, d_model, d_state, d_conv, expand)


class SlowMamba(_MambaStack):
    """Slow stream — 10 layers, d_state=256. Updated on semantic refresh only."""

    def __init__(
        self, d_model: int = 2048, d_state: int = 256, d_conv: int = 4, expand: int = 2
    ):
        super().__init__(10, d_model, d_state, d_conv, expand)


# ---------------------------------------------------------------------------
# Action History Encoder
# ---------------------------------------------------------------------------

class ActionHistoryEncoder(nn.Module):
    """Encodes the last K actions into a single summary token.

    Uses a small Mamba stack (4 layers, d_state=64) to process the
    action history sequence and returns the last hidden state.
    """

    def __init__(
        self,
        action_dim: int = 14,
        d_model: int = 2048,
        d_state: int = 64,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.action_proj = nn.Linear(action_dim, d_model)
        self.stack = _MambaStack(num_layers, d_model, d_state, d_conv=4, expand=2)

    def encode(self, action_history: Tensor) -> Tensor:
        """action_history: [B, K, A] → action_history_token: [B, D]"""
        h = self.action_proj(action_history)  # [B, K, D]
        out, _, _ = self.stack(h)
        return out[:, -1, :]  # last token as summary


# ---------------------------------------------------------------------------
# Cross-Attention Fusion
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """Fuses fast / medium / slow stream tokens via learned cross-attention.

    A learned fusion query attends to the 3 stream tokens, conditioned on
    the stale-time encoding.  Replaces v1's scalar sigmoid gate with a
    content-dependent, per-dimension fusion mechanism.
    """

    def __init__(
        self,
        d_model: int = 2048,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.fusion_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm_q": nn.LayerNorm(d_model),
                        "norm_kv": nn.LayerNorm(d_model),
                        "cross_attn": nn.MultiheadAttention(
                            d_model, num_heads, batch_first=True
                        ),
                        "ffn": nn.Sequential(
                            nn.LayerNorm(d_model),
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Linear(d_model * 4, d_model),
                        ),
                    }
                )
            )
        self.stale_proj = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        fast_token: Tensor,
        medium_token: Tensor,
        slow_token: Tensor,
        stale_token: Tensor,
    ) -> Tensor:
        B = fast_token.shape[0]
        kv = torch.stack([fast_token, medium_token, slow_token], dim=1)  # [B, 3, D]
        kv = kv + self.stale_proj(stale_token).unsqueeze(1)

        q = self.fusion_query.expand(B, -1, -1)  # [B, 1, D]
        for layer in self.layers:
            q_norm = layer["norm_q"](q)
            kv_norm = layer["norm_kv"](kv)
            attn_out, _ = layer["cross_attn"](q_norm, kv_norm, kv_norm)
            q = q + attn_out
            q = q + layer["ffn"](q)

        return self.output_norm(q.squeeze(1))  # [B, D]


# ---------------------------------------------------------------------------
# Tri-Rate Mamba Core
# ---------------------------------------------------------------------------

class TriRateMambaCore(nn.Module):
    """Three-stream temporal processing for HybridVLA v2.

    - Fast  (20L, d_state=128): every control step   (~50 Hz)
    - Medium (6L, d_state=128): every 2nd step        (~25 Hz)
    - Slow  (10L, d_state=256): semantic refresh only  (~12.5 Hz)

    Uses the official ``mamba_ssm.Mamba2`` block when available (CUDA
    fast path), otherwise falls back to a pure-PyTorch implementation
    with JIT-compiled selective scan.
    """

    def __init__(
        self,
        fast_layers: int = 20,
        medium_layers: int = 6,
        slow_layers: int = 10,
        d_model: int = 2048,
        fast_d_state: int = 128,
        medium_d_state: int = 128,
        slow_d_state: int = 256,
        d_conv: int = 4,
        expand: int = 2,
        fusion_type: str = "cross_attention",
        fusion_heads: int = 8,
        fusion_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.fast_mamba = FastMamba(d_model, fast_d_state, d_conv, expand)
        self.medium_mamba = MediumMamba(d_model, medium_d_state, d_conv, expand)
        self.slow_mamba = SlowMamba(d_model, slow_d_state, d_conv, expand)

        # v0.9: Removed redundant stack-level LayerNorm. Each MambaBlock
        # already applies pre-norm (self.norm) inside forward/step.  The
        # previous double-LN on layer 0 was harmless but wasted compute.

        self.stale_encoder = StaleTimeEncoding(d_model)

        self.fusion = CrossAttentionFusion(d_model, fusion_heads, fusion_layers)

    # ------------------------------------------------------------------ #
    # Token composition
    # ------------------------------------------------------------------ #

    def _compose_input_sequence(
        self,
        global_token: Tensor,
        object_slots: Tensor,
        phase_token: Tensor,
        uncertainty_token: Tensor,
        affordance_token: Tensor,
        proprio_token: Tensor,
        prev_action_token: Tensor,
        stale_token: Tensor,
        embodiment_token: Tensor,
        action_history_token: Tensor,
    ) -> Tensor:
        singles = torch.stack(
            [
                global_token,
                phase_token,
                uncertainty_token,
                affordance_token,
                proprio_token,
                prev_action_token,
                stale_token,
                embodiment_token,
                action_history_token,
            ],
            dim=1,
        )  # [B, 9, D]
        return torch.cat([singles, object_slots], dim=1)  # [B, 9+S, D]

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        global_token: Tensor,
        object_slots: Tensor,
        phase_token: Tensor,
        uncertainty_token: Tensor,
        affordance_token: Tensor,
        proprio_token: Tensor,
        prev_action_token: Tensor,
        stale_token: Tensor,
        embodiment_token: Tensor,
        action_history_token: Tensor,
        state: TriRateTemporalState,
        semantic_refresh: bool,
        medium_update: bool,
        use_checkpoint: bool = False,
    ) -> TemporalOutput:
        input_seq = self._compose_input_sequence(
            global_token,
            object_slots,
            phase_token,
            uncertainty_token,
            affordance_token,
            proprio_token,
            prev_action_token,
            stale_token,
            embodiment_token,
            action_history_token,
        )
        B = input_seq.shape[0]
        device, dtype = input_seq.device, input_seq.dtype

        # ---- Fast stream (ALWAYS runs) ----
        fast_out, fast_new_ssm, fast_new_conv = self.fast_mamba(
            input_seq,
            state.fast_ssm_states,
            state.fast_conv_states,
            use_checkpoint=use_checkpoint,
        )
        fast_token = fast_out.mean(dim=1)

        # ---- Medium stream (every 2nd step) ----
        if medium_update:
            med_out, med_new_ssm, med_new_conv = self.medium_mamba(
                input_seq,
                state.medium_ssm_states,
                state.medium_conv_states,
                use_checkpoint=use_checkpoint,
            )
            medium_token = med_out.mean(dim=1)
        else:
            med_new_ssm = state.medium_ssm_states
            med_new_conv = state.medium_conv_states
            medium_token = (
                state.last_medium_token
                if state.last_medium_token is not None
                else torch.zeros(B, self.d_model, device=device, dtype=dtype)
            )

        # ---- Slow stream (semantic refresh only) ----
        if semantic_refresh:
            slow_out, slow_new_ssm, slow_new_conv = self.slow_mamba(
                input_seq,
                state.slow_ssm_states,
                state.slow_conv_states,
                use_checkpoint=use_checkpoint,
            )
            slow_token = slow_out.mean(dim=1)
        else:
            slow_new_ssm = state.slow_ssm_states
            slow_new_conv = state.slow_conv_states
            slow_token = (
                state.last_slow_token
                if state.last_slow_token is not None
                else torch.zeros(B, self.d_model, device=device, dtype=dtype)
            )

        # ---- Cross-attention fusion ----
        fused = self.fusion(fast_token, medium_token, slow_token, stale_token)

        # ---- Build next state ----
        next_state = TriRateTemporalState(
            fast_ssm_states=fast_new_ssm,
            fast_conv_states=fast_new_conv,
            medium_ssm_states=med_new_ssm,
            medium_conv_states=med_new_conv,
            last_medium_token=medium_token,
            slow_ssm_states=slow_new_ssm,
            slow_conv_states=slow_new_conv,
            last_slow_token=slow_token,
            step_counter=state.step_counter + 1,
            steps_since_refresh=(
                0 if semantic_refresh else state.steps_since_refresh + 1
            ),
            steps_since_medium=(
                0 if medium_update else state.steps_since_medium + 1
            ),
        )

        return TemporalOutput(
            fused_state=fused,
            fast_token=fast_token,
            medium_token=medium_token,
            slow_token=slow_token,
            next_state=next_state,
        )
