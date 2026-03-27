"""Test normalizer roundtrip: normalize -> denormalize = identity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vla_hybrid_v2.data.normalizer import Normalizer


class TestMinMaxNormalizer:
    def test_roundtrip(self):
        norm = Normalizer(strategy="min_max")
        data = np.random.randn(100, 7).astype(np.float32) * 5 + 3
        norm.fit(data)
        raw = torch.from_numpy(data[:5])
        recovered = norm.denormalize(norm.normalize(raw))
        assert torch.allclose(raw, recovered, atol=1e-5)

    def test_range(self):
        norm = Normalizer(strategy="min_max", target_range=(-1.0, 1.0))
        data = np.random.randn(200, 3).astype(np.float32)
        norm.fit(data)
        normed = norm.normalize(torch.from_numpy(data))
        # Values from the fit range should be in [-1, 1]
        assert normed.min() >= -1.0 - 1e-5
        assert normed.max() <= 1.0 + 1e-5

    def test_save_load(self, tmp_path):
        norm = Normalizer(strategy="min_max")
        data = np.random.randn(50, 7).astype(np.float32)
        norm.fit(data)
        norm.save(tmp_path / "stats.json")

        norm2 = Normalizer(strategy="min_max")
        norm2.load(tmp_path / "stats.json")
        x = torch.randn(5, 7)
        assert torch.allclose(norm.normalize(x), norm2.normalize(x))


class TestMeanStdNormalizer:
    def test_roundtrip(self):
        norm = Normalizer(strategy="mean_std")
        data = np.random.randn(200, 7).astype(np.float32) * 2 + 1
        norm.fit(data)
        raw = torch.from_numpy(data[:5])
        recovered = norm.denormalize(norm.normalize(raw))
        # mean_std uses tanh, so roundtrip is approximate for extreme values
        assert torch.allclose(raw, recovered, atol=1e-3)

    def test_output_bounded(self):
        norm = Normalizer(strategy="mean_std", target_range=(-1.0, 1.0))
        data = np.random.randn(100, 3).astype(np.float32)
        norm.fit(data)
        normed = norm.normalize(torch.from_numpy(data))
        # tanh squash: output is always in (-1, 1)
        assert normed.min() > -1.0
        assert normed.max() < 1.0


class TestEdgeCases:
    def test_constant_dimension(self):
        """Constant dims should not cause NaN."""
        norm = Normalizer(strategy="min_max")
        data = np.zeros((50, 3), dtype=np.float32)
        data[:, 0] = 5.0  # constant
        data[:, 1] = np.random.randn(50)
        data[:, 2] = np.random.randn(50)
        norm.fit(data)
        normed = norm.normalize(torch.from_numpy(data[:5]))
        assert not torch.isnan(normed).any()

    def test_unfitted_raises(self):
        norm = Normalizer()
        with pytest.raises(AssertionError):
            norm.normalize(torch.randn(5, 3))
