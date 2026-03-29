"""Dtype consistency tests for the FSDP main path.

Validates that:
- normalize_model_dtypes_for_fsdp converts all float params to target dtype
- verify_model_dtypes detects mismatched dtypes
- Checkpoint save → load preserves dtype consistency
- SSM parameters (A_log, D) are handled correctly
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.conftest import _build_model, _mini_cfg
from vla_hybrid_v2.utils.distributed import (
    normalize_model_dtypes_for_fsdp,
    verify_model_dtypes,
)


class TestDtypeNormalization:
    """Tests for normalize_model_dtypes_for_fsdp."""

    def test_converts_float32_params(self):
        cfg = _mini_cfg("a")
        model = _build_model(cfg)
        # Non-backbone params default to float32
        f32_before = sum(
            1 for _, p in model.named_parameters()
            if p.is_floating_point() and p.dtype == torch.float32
        )
        assert f32_before > 0, "Expected some float32 params before normalization"

        normalize_model_dtypes_for_fsdp(model, target_dtype=torch.bfloat16)

        for name, p in model.named_parameters():
            if p.is_floating_point():
                assert p.dtype == torch.bfloat16, f"{name} is {p.dtype}, expected bfloat16"

    def test_converts_float_buffers(self):
        cfg = _mini_cfg("a")
        model = _build_model(cfg)
        # _fast_bin_centers is a float buffer
        normalize_model_dtypes_for_fsdp(model, target_dtype=torch.bfloat16)

        for name, buf in model.named_buffers():
            if buf.is_floating_point():
                assert buf.dtype == torch.bfloat16, f"buffer {name} is {buf.dtype}"

    def test_preserves_int_buffers(self):
        """Integer/bool buffers must not be converted."""
        cfg = _mini_cfg("a")
        model = _build_model(cfg)
        # Register an integer buffer for testing
        model.register_buffer("_test_int_buf", torch.tensor([1, 2, 3], dtype=torch.long))

        normalize_model_dtypes_for_fsdp(model, target_dtype=torch.bfloat16)
        assert model._test_int_buf.dtype == torch.long

    def test_idempotent(self):
        """Calling normalize twice is a no-op the second time."""
        cfg = _mini_cfg("a")
        model = _build_model(cfg)
        normalize_model_dtypes_for_fsdp(model, target_dtype=torch.bfloat16)

        # Snapshot param values
        before = {n: p.data.clone() for n, p in model.named_parameters()}

        normalize_model_dtypes_for_fsdp(model, target_dtype=torch.bfloat16)

        for name, p in model.named_parameters():
            assert torch.equal(p.data, before[name]), f"{name} changed on second call"

    def test_ssm_params_converted(self):
        """SSM A_log and D parameters (float32 init) must be converted to bf16."""
        cfg = _mini_cfg("a")
        model = _build_model(cfg)

        # Collect A_log and D params before normalization
        ssm_params = {
            n: p.dtype
            for n, p in model.named_parameters()
            if "A_log" in n or (n.endswith(".D") and "d_conv" not in n)
        }
        assert any(
            dt == torch.float32 for dt in ssm_params.values()
        ), f"Expected float32 SSM params, got {ssm_params}"

        normalize_model_dtypes_for_fsdp(model, target_dtype=torch.bfloat16)

        for name, p in model.named_parameters():
            if "A_log" in name or (name.endswith(".D") and "d_conv" not in name):
                assert p.dtype == torch.bfloat16, f"SSM param {name} is {p.dtype}"


class TestDtypeVerification:
    """Tests for verify_model_dtypes."""

    def test_all_bf16_passes(self):
        cfg = _mini_cfg("a")
        model = _build_model(cfg)
        normalize_model_dtypes_for_fsdp(model, target_dtype=torch.bfloat16)
        assert verify_model_dtypes(model, expected_dtype=torch.bfloat16)

    def test_mixed_dtype_fails(self):
        cfg = _mini_cfg("a")
        model = _build_model(cfg)
        # Model has float32 params by default (non-normalized)
        assert not verify_model_dtypes(model, expected_dtype=torch.bfloat16)

    def test_with_label(self):
        cfg = _mini_cfg("a")
        model = _build_model(cfg)
        normalize_model_dtypes_for_fsdp(model, target_dtype=torch.bfloat16)
        assert verify_model_dtypes(model, expected_dtype=torch.bfloat16, label="test-label")


class TestCheckpointDtypeRoundtrip:
    """Verify dtype consistency survives checkpoint save → load."""

    def test_save_load_preserves_bf16(self):
        cfg = _mini_cfg("a")
        model = _build_model(cfg)
        normalize_model_dtypes_for_fsdp(model, target_dtype=torch.bfloat16)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            ckpt_dir.mkdir()
            torch.save(model.state_dict(), ckpt_dir / "model.pt")

            # Load into a fresh model, then normalize (mimics real train flow)
            model2 = _build_model(cfg)
            state = torch.load(ckpt_dir / "model.pt", weights_only=True)
            model2.load_state_dict(state, strict=False)
            normalize_model_dtypes_for_fsdp(model2, target_dtype=torch.bfloat16)

            for name, p in model2.named_parameters():
                if p.is_floating_point():
                    assert p.dtype == torch.bfloat16, (
                        f"{name} is {p.dtype} after load+normalize"
                    )

    def test_load_float32_checkpoint_then_normalize(self):
        """Loading an old f32 checkpoint + normalize must produce bf16."""
        cfg = _mini_cfg("a")
        model = _build_model(cfg)
        # Save WITHOUT normalization (float32)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            ckpt_dir.mkdir()
            torch.save(model.state_dict(), ckpt_dir / "model.pt")

            # Load into fresh model
            model2 = _build_model(cfg)
            state = torch.load(ckpt_dir / "model.pt", weights_only=True)
            model2.load_state_dict(state, strict=False)

            # Params are float32 (from checkpoint)
            assert not verify_model_dtypes(model2, expected_dtype=torch.bfloat16)

            # Normalize fixes it
            normalize_model_dtypes_for_fsdp(model2, target_dtype=torch.bfloat16)
            assert verify_model_dtypes(model2, expected_dtype=torch.bfloat16)
