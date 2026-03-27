"""Custom collate function for HybridVLA v2.

Handles the batch assembly for forward_train(), including:
- Standard tensor stacking for fixed-shape fields
- List-of-tensors for variable-length vision features (refresh frames)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Keys whose first tensor dimension (N_patches) may vary across samples.
# These get padded instead of crashing on shape mismatch.
_VISION_KEYS = {"pixel_values", "image_grid_thw"}


def _safe_stack_vision(tensors: List[Tensor], key: str) -> Tensor:
    """Stack tensors, padding dim-0 if shapes differ (defense-in-depth)."""
    shapes = [t.shape[0] for t in tensors]
    if len(set(shapes)) == 1:
        return torch.stack(tensors, dim=0)
    # Should not happen after hdf5_adapter resize fix — warn loudly.
    max_n = max(shapes)
    logger.warning(
        "collate: variable dim-0 in '%s' (shapes %s). "
        "Padding to %d — check image preprocessing.",
        key, shapes, max_n,
    )
    padded = []
    for t in tensors:
        if t.shape[0] < max_n:
            pad = torch.zeros(max_n - t.shape[0], *t.shape[1:],
                              dtype=t.dtype, device=t.device)
            padded.append(torch.cat([t, pad], dim=0))
        else:
            padded.append(t)
    return torch.stack(padded, dim=0)


def vla_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of sample dicts into a training batch.

    Fixed-shape tensors are stacked along a new batch dimension.
    List-valued fields (e.g., refresh_pixel_values_list) are preserved
    as lists since their inner shapes may vary per refresh frame.
    """
    batch: Dict[str, Any] = {}
    keys = samples[0].keys()

    for key in keys:
        values = [s[key] for s in samples]

        # Explicit None handling: collapse list-of-Nones to single None
        if values[0] is None:
            batch[key] = None
            continue

        if isinstance(values[0], Tensor):
            if key in _VISION_KEYS:
                batch[key] = _safe_stack_vision(values, key)
            else:
                batch[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], list):
            # Variable-length lists (e.g., refresh pixel values):
            # transpose from [B][R] to [R][B] for per-refresh-frame access,
            # then stack within each refresh frame
            R = len(values[0])
            is_vision = any(vk in key for vk in _VISION_KEYS)
            transposed = []
            for r in range(R):
                frame_vals = [v[r] for v in values]
                # N2 fix: check ALL samples, not just [0], for None/Tensor mixing.
                # If any sample is None while others are Tensor, skip the None
                # samples and pad with zeros to keep batch dimension consistent.
                non_none = [v for v in frame_vals if v is not None]
                all_none = len(non_none) == 0
                if all_none:
                    transposed.append(None)
                elif isinstance(non_none[0], Tensor):
                    # Replace None entries with zero tensors matching shape
                    if len(non_none) < len(frame_vals):
                        template = non_none[0]
                        frame_vals = [
                            v if v is not None
                            else torch.zeros_like(template)
                            for v in frame_vals
                        ]
                        logger.warning(
                            "collate: mixed None/Tensor in '%s[%d]' — "
                            "padded %d None entries with zeros.",
                            key, r, len(frame_vals) - len(non_none),
                        )
                    if is_vision:
                        transposed.append(
                            _safe_stack_vision(frame_vals, f"{key}[{r}]"))
                    else:
                        transposed.append(torch.stack(frame_vals, dim=0))
                else:
                    transposed.append(frame_vals)
            batch[key] = transposed
        elif isinstance(values[0], (int, float)):
            batch[key] = torch.tensor(values)
        else:
            # Pass through (e.g., strings)
            batch[key] = values

    return batch
