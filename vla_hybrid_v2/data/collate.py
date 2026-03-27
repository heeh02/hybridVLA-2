"""Custom collate function for HybridVLA v2.

Handles the batch assembly for forward_train(), including:
- Standard tensor stacking for fixed-shape fields
- List-of-tensors for variable-length vision features (refresh frames)
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import Tensor


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
            batch[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], list):
            # Variable-length lists (e.g., refresh pixel values):
            # transpose from [B][R] to [R][B] for per-refresh-frame access,
            # then stack within each refresh frame
            R = len(values[0])
            transposed = []
            for r in range(R):
                frame_vals = [v[r] for v in values]
                if isinstance(frame_vals[0], Tensor):
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
