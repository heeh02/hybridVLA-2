"""Action / proprio normalization for HybridVLA v2.

Maps raw sensor values to the model's canonical range (default [-1, 1])
and back. Statistics are computed once via `fit()` and persisted alongside
checkpoints so that inference uses the same normalization as training.

Supports two strategies:
- "min_max": maps [min, max] → [lo, hi]  (robust to outliers with clip)
- "mean_std": maps via z-score then tanh squash → [lo, hi]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class Normalizer:
    """Generic normalizer for action or proprio vectors.

    Usage::

        norm = Normalizer(strategy="min_max", target_range=(-1.0, 1.0))
        norm.fit(raw_actions_np)          # compute stats from dataset
        normed = norm.normalize(raw_t)    # raw → model space
        raw = norm.denormalize(normed_t)  # model space → raw
        norm.save(path)                   # persist stats
        norm.load(path)                   # restore stats
    """

    def __init__(
        self,
        strategy: str = "min_max",
        target_range: Tuple[float, float] = (-1.0, 1.0),
        eps: float = 1e-6,
    ) -> None:
        assert strategy in ("min_max", "mean_std"), f"Unknown strategy: {strategy}"
        self.strategy = strategy
        self.lo, self.hi = target_range
        self.eps = eps

        # Statistics (set by fit() or load())
        self._fitted = False
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: np.ndarray) -> None:
        """Compute normalization statistics from a dataset.

        Parameters
        ----------
        data : np.ndarray of shape [N, D] or [N, T, D]
            Raw values. Flattened to [*, D] internally.
        """
        if data.ndim > 2:
            data = data.reshape(-1, data.shape[-1])

        self._min = data.min(axis=0).astype(np.float32)
        self._max = data.max(axis=0).astype(np.float32)
        self._mean = data.mean(axis=0).astype(np.float32)
        self._std = data.std(axis=0).astype(np.float32)

        # D3: warn about near-zero std dimensions (risk of exploding z-scores)
        low_std = self._std < 1e-4
        if low_std.any():
            dims = list(np.where(low_std)[0])
            logger.warning(
                "Normalizer: %d/%d dims have std < 1e-4 (dims=%s). "
                "These may be constant features.",
                int(low_std.sum()), data.shape[-1], dims,
            )

        self._fitted = True

        D = data.shape[-1]
        logger.info(
            "Normalizer fitted: strategy=%s, dim=%d, "
            "range=[%.3f, %.3f] → [%.1f, %.1f]",
            self.strategy, D,
            float(self._min.min()), float(self._max.max()),
            self.lo, self.hi,
        )

    # ------------------------------------------------------------------
    # Normalize / Denormalize
    # ------------------------------------------------------------------

    def normalize(self, raw: Tensor) -> Tensor:
        """raw space → model space [lo, hi]."""
        assert self._fitted, "Call fit() or load() before normalize()"

        if self.strategy == "min_max":
            mn = torch.as_tensor(self._min, dtype=raw.dtype, device=raw.device)
            mx = torch.as_tensor(self._max, dtype=raw.dtype, device=raw.device)
            scale = mx - mn
            scale = scale.clamp(min=self.eps)
            normed = (raw - mn) / scale  # [0, 1]
            return normed * (self.hi - self.lo) + self.lo  # [lo, hi]
        else:  # mean_std
            mu = torch.as_tensor(self._mean, dtype=raw.dtype, device=raw.device)
            sd = torch.as_tensor(self._std, dtype=raw.dtype, device=raw.device)
            sd = sd.clamp(min=self.eps)
            z = (raw - mu) / sd
            # tanh squash to bound output
            squashed = torch.tanh(z)
            return squashed * (self.hi - self.lo) / 2 + (self.hi + self.lo) / 2

    def denormalize(self, normed: Tensor) -> Tensor:
        """model space [lo, hi] → raw space."""
        assert self._fitted, "Call fit() or load() before denormalize()"

        if self.strategy == "min_max":
            mn = torch.as_tensor(self._min, dtype=normed.dtype, device=normed.device)
            mx = torch.as_tensor(self._max, dtype=normed.dtype, device=normed.device)
            t01 = (normed - self.lo) / (self.hi - self.lo)  # [0, 1]
            return t01 * (mx - mn) + mn
        else:  # mean_std
            mu = torch.as_tensor(self._mean, dtype=normed.dtype, device=normed.device)
            sd = torch.as_tensor(self._std, dtype=normed.dtype, device=normed.device)
            sd = sd.clamp(min=self.eps)
            # inverse tanh (atanh). atanh undefined at +/-1; with eps=1e-6,
            # atanh(1 - eps) = 0.5 * ln(2e6 - 1) ~ 7.25. Smaller eps improves
            # invertibility at tails but risks numerical instability.
            centered = (normed - (self.hi + self.lo) / 2) / ((self.hi - self.lo) / 2)
            centered = centered.clamp(-1 + self.eps, 1 - self.eps)
            z = torch.atanh(centered)
            return z * sd + mu

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        stats = {
            "strategy": self.strategy,
            "target_range": [self.lo, self.hi],
            "min": self._min.tolist() if self._min is not None else None,
            "max": self._max.tolist() if self._max is not None else None,
            "mean": self._mean.tolist() if self._mean is not None else None,
            "std": self._std.tolist() if self._std is not None else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Normalizer saved to %s", path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path) as f:
            stats = json.load(f)
        loaded_range = tuple(stats["target_range"])
        if (self.lo, self.hi) != loaded_range:
            logger.warning(
                "Normalizer.load(): constructor target_range=(%.2f, %.2f) "
                "overridden by saved range=(%.2f, %.2f) from %s",
                self.lo, self.hi, loaded_range[0], loaded_range[1], path,
            )
        self.strategy = stats["strategy"]
        self.lo, self.hi = loaded_range
        self._min = np.array(stats["min"], dtype=np.float32) if stats["min"] else None
        self._max = np.array(stats["max"], dtype=np.float32) if stats["max"] else None
        self._mean = np.array(stats["mean"], dtype=np.float32) if stats["mean"] else None
        self._std = np.array(stats["std"], dtype=np.float32) if stats["std"] else None
        self._fitted = True
        logger.info("Normalizer loaded from %s (strategy=%s)", path, self.strategy)


# Convenience aliases
ActionNormalizer = Normalizer
ProprioNormalizer = Normalizer
