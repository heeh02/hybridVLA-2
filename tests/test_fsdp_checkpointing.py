"""Activation checkpointing conflict tests for the FSDP main path.

Validates that:
- _MambaStack detects FSDP CheckpointWrapper on its layers
- Internal activation_checkpoint is skipped when FSDP manages it
- _unwrap_layer correctly reaches the underlying MambaBlock
- Fallback path works correctly with and without FSDP checkpointing
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import _build_model, _mini_cfg
from vla_hybrid_v2.models.mamba_core import MambaBlock, _MambaStack


class TestUnwrapLayer:
    """Tests for _MambaStack._unwrap_layer."""

    def test_unwrap_bare_layer(self):
        stack = _MambaStack(2, 64, 16, 4, 2, force_fallback=True)
        inner = stack._unwrap_layer(stack.layers[0])
        assert isinstance(inner, MambaBlock)
        assert inner is stack.layers[0]

    def test_unwrap_checkpoint_wrapped_layer(self):
        """Simulate FSDP checkpoint wrapping and verify unwrap works."""
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                checkpoint_wrapper,
            )
        except ImportError:
            pytest.skip("checkpoint_wrapper not available in this PyTorch version")

        stack = _MambaStack(2, 64, 16, 4, 2, force_fallback=True)
        original_layer = stack.layers[0]

        # Wrap layer with CheckpointWrapper (simulates FSDP checkpointing)
        wrapped = checkpoint_wrapper(
            original_layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        stack.layers[0] = wrapped

        inner = stack._unwrap_layer(stack.layers[0])
        assert isinstance(inner, MambaBlock)
        assert inner is original_layer


class TestFsdpManagesCheckpointing:
    """Tests for _MambaStack._fsdp_manages_checkpointing."""

    def test_false_for_bare_stack(self):
        stack = _MambaStack(2, 64, 16, 4, 2, force_fallback=True)
        assert not stack._fsdp_manages_checkpointing()

    def test_true_after_checkpoint_wrapping(self):
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                checkpoint_wrapper,
            )
        except ImportError:
            pytest.skip("checkpoint_wrapper not available")

        stack = _MambaStack(2, 64, 16, 4, 2, force_fallback=True)
        # Wrap first layer
        stack.layers[0] = checkpoint_wrapper(
            stack.layers[0], checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        assert stack._fsdp_manages_checkpointing()


class TestNoDoubleCheckpointing:
    """Verify internal checkpointing is skipped when FSDP manages it."""

    def test_fallback_path_with_fsdp_ckpt(self):
        """Forward should succeed when layers are CheckpointWrapper-wrapped."""
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                checkpoint_wrapper,
            )
        except ImportError:
            pytest.skip("checkpoint_wrapper not available")

        B, L, D = 2, 4, 64
        stack = _MambaStack(2, D, 16, 4, 2, force_fallback=True)
        stack.train()

        # Wrap all layers (simulates _apply_activation_checkpointing)
        for i in range(len(stack.layers)):
            stack.layers[i] = checkpoint_wrapper(
                stack.layers[i], checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

        x = torch.randn(B, L, D)
        # use_checkpoint=True but FSDP manages it → should NOT double-checkpoint
        out, ssm, conv = stack(x, use_checkpoint=True)
        assert out.shape == (B, L, D)

    def test_fallback_path_without_fsdp_ckpt(self):
        """Internal checkpointing should work normally without FSDP wrapping."""
        B, L, D = 2, 4, 64
        stack = _MambaStack(2, D, 16, 4, 2, force_fallback=True)
        stack.train()

        x = torch.randn(B, L, D, requires_grad=True)
        out, ssm, conv = stack(x, use_checkpoint=True)
        assert out.shape == (B, L, D)
        # Verify gradient flows
        out.sum().backward()
        assert x.grad is not None

    def test_stateless_path_with_checkpoint_wrapper(self):
        """Stateless path must work when layers are CheckpointWrapper-wrapped."""
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                checkpoint_wrapper,
            )
        except ImportError:
            pytest.skip("checkpoint_wrapper not available")

        B, L, D = 2, 4, 64
        stack = _MambaStack(2, D, 16, 4, 2, force_fallback=True)

        for i in range(len(stack.layers)):
            stack.layers[i] = checkpoint_wrapper(
                stack.layers[i], checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

        x = torch.randn(B, L, D)
        # stateless=True → should use layer(x) instead of layer._forward_official
        out, ssm, conv = stack(x, stateless=True)
        assert out.shape == (B, L, D)


class TestFullModelCheckpointingIntegration:
    """Integration: full model forward succeeds after simulated FSDP checkpointing."""

    def test_forward_train_after_checkpoint_wrapping(self):
        """Simulate FSDP checkpoint wrapping on MambaBlocks, then run forward_train."""
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )
        except ImportError:
            pytest.skip("checkpoint_wrapper not available")

        import functools

        cfg = _mini_cfg("a")
        model = _build_model(cfg)
        model.train()

        # Apply checkpoint wrapping to MambaBlock layers (like FSDP does)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda m: isinstance(m, MambaBlock),
        )

        B, T, H, A, P, L = 2, 2, cfg.model.action_expert.chunk_horizon, 7, 9, 16
        batch = {
            "actions": torch.randn(B, T, H, A),
            "proprio": torch.randn(B, T, P),
            "prev_actions": torch.randn(B, T, A),
            "input_ids": torch.randint(0, 1000, (B, L)),
            "attention_mask": torch.ones(B, L, dtype=torch.long),
        }

        losses = model.forward_train(batch)
        assert "loss_total" in losses
        assert losses["loss_total"].isfinite()
