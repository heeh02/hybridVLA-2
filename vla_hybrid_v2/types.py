"""Core type definitions for HybridVLA v2.

v2 changes: TriRateTemporalState, affordance_token, compressed slots,
ActionHistoryBuffer, medium stream state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class GrounderOutput:
    """Output of the HierarchicalAttentionGrounder."""

    global_token: Tensor               # [B, D]
    object_slots: Tensor               # [B, num_object_slots, D] — raw (48)
    compressed_object_slots: Tensor    # [B, compressed_slots, D] — refined (24)
    phase_token: Tensor                # [B, D]
    uncertainty_token: Tensor          # [B, D]
    affordance_token: Tensor           # [B, D] — v2 new
    attn_maps: Optional[Dict[str, Tensor]] = None


@dataclass
class TriRateTemporalState:
    """Recurrent state for the tri-rate Mamba core."""

    # Fast stream
    fast_ssm_states: Optional[List[Tensor]] = None
    fast_conv_states: Optional[List[Tensor]] = None
    # Medium stream
    medium_ssm_states: Optional[List[Tensor]] = None
    medium_conv_states: Optional[List[Tensor]] = None
    last_medium_token: Optional[Tensor] = None
    # Slow stream
    slow_ssm_states: Optional[List[Tensor]] = None
    slow_conv_states: Optional[List[Tensor]] = None
    last_slow_token: Optional[Tensor] = None
    # Counters
    step_counter: int = 0
    steps_since_refresh: int = 0
    steps_since_medium: int = 0


@dataclass
class TemporalOutput:
    """Output of TriRateMambaCore at a single timestep."""

    fused_state: Tensor       # [B, D]
    fast_token: Tensor        # [B, D]
    medium_token: Tensor      # [B, D]
    slow_token: Tensor        # [B, D]
    next_state: TriRateTemporalState


@dataclass
class ActionExpertOutput:
    """Output of the FlowActionExpert."""

    velocity: Tensor                        # [B, H, A]
    denoised_action: Optional[Tensor] = None


@dataclass
class ControlStepOutput:
    """Output of control_step() — the single action to execute NOW.

    v0.9.1: replaces returning full ActionExpertOutput with meaningless
    velocity=zeros. The caller gets the [B, A] action directly.
    """

    action: Tensor                     # [B, A] — the ONE action to execute
    chunk: Optional[Tensor] = None     # [B, H, A] — full chunk (for logging/debug)
    chunk_step: int = 0                # which step within the chunk was returned
    semantic_refresh: bool = False     # whether a new chunk was generated


@dataclass
class RuntimeCache:
    """Per-episode runtime state for online inference."""

    temporal_state: TriRateTemporalState = field(
        default_factory=TriRateTemporalState
    )
    last_semantic: Optional[GrounderOutput] = None
    # v0.9.1: monotonic counter replaces fragile id() comparison (Issue A2).
    # Caller increments refresh_counter after each semantic_step().
    refresh_counter: int = 0
    _last_seen_refresh: int = -1
    current_chunk: Optional[Tensor] = None
    chunk_step: int = 0
    action_history: Optional[Tensor] = None  # [K, A]
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    # v0.11: RTC — tail of previous chunk for overlap inpainting at inference
    prev_chunk_tail: Optional[Tensor] = None  # [B, overlap, A]


@dataclass
class ActionHistoryBuffer:
    """Ring buffer for the last K actions."""

    buffer: Optional[Tensor] = None  # [B, K, A]
    max_len: int = 8
    current_len: int = 0

    def push(self, action: Tensor) -> None:
        """Add an action [B, A] to the buffer."""
        B, A = action.shape
        if self.buffer is None:
            self.buffer = torch.zeros(
                B, self.max_len, A, device=action.device, dtype=action.dtype
            )
        if self.current_len < self.max_len:
            self.buffer[:, self.current_len] = action
            self.current_len += 1
        else:
            self.buffer = torch.roll(self.buffer, -1, dims=1)
            self.buffer[:, -1] = action

    def get(self) -> Tensor:
        """Return [B, K, A] with zero-padding for unfilled slots."""
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized. Call push() first.")
        return self.buffer
