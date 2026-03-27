"""Batch protocol for HybridVLA v2.

Defines the data contract between the data layer and the model.
Dataset adapters return dicts conforming to the WindowSample field spec;
the collate function stacks them into the batch that `forward_train()` expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from torch import Tensor


@dataclass
class WindowSample:
    """One training window from an episode.

    Produced by dataset adapters, consumed by `vla_collate_fn`.

    Required fields must always be present. Vision and refresh fields
    are None when the sample is text-only or single-observation.
    """

    # --- Required ---
    actions: Tensor          # [T, H, A] — normalized to action_range
    proprio: Tensor          # [T, P] — normalized
    prev_actions: Tensor     # [T, A] — normalized
    input_ids: Tensor        # [L] — tokenized text (+ image placeholders)
    attention_mask: Tensor   # [L]

    # --- Vision (None if text-only) ---
    pixel_values: Optional[Tensor] = None       # [N_patches, patch_dim]
    image_grid_thw: Optional[Tensor] = None     # [N_images, 3]

    # --- Refresh frames (None if single-observation) ---
    refresh_input_ids: Optional[Tensor] = None        # [R, L]
    refresh_attention_mask: Optional[Tensor] = None    # [R, L]
    refresh_pixel_values_list: Optional[List[Tensor]] = None   # List[Tensor] len=R
    refresh_image_grid_thw_list: Optional[List[Tensor]] = None  # List[Tensor] len=R

    # --- Optional labels ---
    phase_labels: Optional[Tensor] = None        # [T]
    affordance_labels: Optional[Tensor] = None   # [T]
    embodiment_id: int = 0
    step_weights: Optional[Tensor] = None        # [H]


# Batch protocol documentation — the dict keys forward_train() expects.
# This is not enforced as a type but serves as the authoritative reference.
BATCH_REQUIRED_KEYS = frozenset({
    "actions",          # [B, T, H, A]
    "proprio",          # [B, T, P]
    "prev_actions",     # [B, T, A]
    "input_ids",        # [B, L]
    "attention_mask",   # [B, L]
})

BATCH_VISION_KEYS = frozenset({
    "pixel_values",     # [B, N_patches, patch_dim]
    "image_grid_thw",   # [B, N_images, 3]
})

BATCH_OPTIONAL_KEYS = frozenset({
    "semantic_refresh_steps",   # List[int]
    "embodiment_id",            # [B]
    "phase_labels",             # [B, T]
    "affordance_labels",        # [B, T]
    "step_weights",             # [B, H]
    "refresh_input_ids",        # [B, R, L]
    "refresh_attention_mask",   # [B, R, L]
    "refresh_pixel_values_list",  # List[Tensor] len=R
    "refresh_image_grid_thw_list",  # List[Tensor] len=R
})
