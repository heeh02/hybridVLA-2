"""Tests for gradient accumulation no_sync behaviour.

Verifies that:
1. FSDP model.no_sync() is called on non-final micro-steps
2. no_sync is NOT called on the final micro-step (gradient sync happens)
3. Single-GPU / non-FSDP path works without errors (nullcontext)
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch, call

import pytest
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Helper: minimal _maybe_no_sync extracted from train_unified
# ---------------------------------------------------------------------------

def _make_maybe_no_sync(model):
    """Reproduce the _maybe_no_sync logic from train_unified.py."""
    _has_no_sync = hasattr(model, "no_sync")

    def _maybe_no_sync(is_accumulating: bool):
        if is_accumulating and _has_no_sync:
            return model.no_sync()
        return contextlib.nullcontext()

    return _maybe_no_sync


class TestMaybeNoSync:
    """Unit tests for the _maybe_no_sync helper."""

    def test_calls_no_sync_when_accumulating_and_fsdp(self):
        """FSDP model: no_sync should be invoked on non-final micro-steps."""
        model = MagicMock()
        model.no_sync.return_value = contextlib.nullcontext()
        fn = _make_maybe_no_sync(model)

        ctx = fn(is_accumulating=True)
        # Should have called model.no_sync()
        model.no_sync.assert_called_once()

    def test_no_no_sync_on_final_step_fsdp(self):
        """FSDP model: final micro-step should NOT call no_sync."""
        model = MagicMock()
        fn = _make_maybe_no_sync(model)

        ctx = fn(is_accumulating=False)
        model.no_sync.assert_not_called()
        # Should return a nullcontext (usable as context manager)
        with ctx:
            pass

    def test_single_gpu_no_sync_absent(self):
        """Non-FSDP model (no .no_sync attr): always returns nullcontext."""
        model = nn.Linear(4, 4)
        assert not hasattr(model, "no_sync")
        fn = _make_maybe_no_sync(model)

        # Both accumulating and non-accumulating should work fine
        with fn(is_accumulating=True):
            pass
        with fn(is_accumulating=False):
            pass


class TestAccumulationPattern:
    """Verify the micro-step counting and is_accumulating flag logic."""

    @pytest.mark.parametrize("grad_accum", [1, 2, 4, 8])
    def test_sync_only_on_final_micro_step(self, grad_accum):
        """
        Simulates the training loop micro-step counter.
        Only the last micro-step in each accumulation window should sync.
        """
        model = MagicMock()
        model.no_sync.return_value = contextlib.nullcontext()
        fn = _make_maybe_no_sync(model)

        total_micro_steps = grad_accum * 3  # 3 full optimizer steps
        sync_steps = []
        nosync_steps = []

        micro_step = 0
        for _ in range(total_micro_steps):
            is_accumulating = (micro_step + 1) % grad_accum != 0
            fn(is_accumulating)
            if is_accumulating:
                nosync_steps.append(micro_step)
            else:
                sync_steps.append(micro_step)
            micro_step += 1

        # Exactly 3 sync points
        assert len(sync_steps) == 3
        # Sync at micro_step = grad_accum-1, 2*grad_accum-1, 3*grad_accum-1
        assert sync_steps == [grad_accum - 1, 2 * grad_accum - 1, 3 * grad_accum - 1]
        # Remaining are no-sync
        assert len(nosync_steps) == total_micro_steps - 3

    def test_grad_accum_1_never_calls_no_sync(self):
        """When grad_accum=1, every step syncs — no_sync is never invoked."""
        model = MagicMock()
        model.no_sync.return_value = contextlib.nullcontext()
        fn = _make_maybe_no_sync(model)

        for micro_step in range(10):
            is_accumulating = (micro_step + 1) % 1 != 0  # always False
            fn(is_accumulating)

        model.no_sync.assert_not_called()
