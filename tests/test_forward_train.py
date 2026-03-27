"""Integration test: forward_train -> backward -> gradient check.

The single most important test. If this passes, the model can train.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import B, T, H, A, P, L, _mini_cfg, _build_model


def _make_batch(stage: str, cfg):
    """Create a dummy batch appropriate for the given stage."""
    batch = {
        "actions": torch.randn(B, T, H, A),
        "proprio": torch.randn(B, T, P),
        "prev_actions": torch.randn(B, T, A),
        "input_ids": torch.randint(0, 1000, (B, L)),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
    }
    return batch


class TestForwardBackward:
    @pytest.mark.parametrize("stage", ["a", "b", "c"])
    def test_loss_computes_all_stages(self, stage):
        """Core sanity: loss computes and is finite for all 3 stages."""
        cfg = _mini_cfg(stage)
        model = _build_model(cfg)
        batch = _make_batch(stage, cfg)

        losses = model.forward_train(batch)

        assert "loss_total" in losses
        assert losses["loss_total"].requires_grad
        assert not torch.isnan(losses["loss_total"]), f"NaN loss in stage {stage}"
        assert not torch.isinf(losses["loss_total"]), f"Inf loss in stage {stage}"

    @pytest.mark.parametrize("stage", ["a", "b", "c"])
    def test_backward_no_crash(self, stage):
        """Backward pass completes without error."""
        cfg = _mini_cfg(stage)
        model = _build_model(cfg)
        batch = _make_batch(stage, cfg)

        losses = model.forward_train(batch)
        losses["loss_total"].backward()

        # At least one parameter should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, f"No gradients after backward in stage {stage}"

    @pytest.mark.parametrize("stage", ["a", "b", "c"])
    def test_no_nan_gradients(self, stage):
        """No parameter should have NaN gradients after backward."""
        cfg = _mini_cfg(stage)
        model = _build_model(cfg)
        batch = _make_batch(stage, cfg)

        losses = model.forward_train(batch)
        losses["loss_total"].backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name} (stage {stage})"

    def test_stage_a_no_flow_loss(self):
        """Stage A must NOT produce flow matching loss."""
        cfg = _mini_cfg("a")
        model = _build_model(cfg)
        batch = _make_batch("a", cfg)
        losses = model.forward_train(batch)
        assert "loss_fm" not in losses
        assert "loss_fast" in losses

    def test_stage_b_has_flow_loss(self):
        """Stage B must produce flow matching loss."""
        cfg = _mini_cfg("b")
        model = _build_model(cfg)
        batch = _make_batch("b", cfg)
        losses = model.forward_train(batch)
        assert "loss_fm" in losses


class TestStageC:
    def _make_stage_c_cfg(self, rtc=False, faster=False):
        cfg = _mini_cfg("c")
        cfg.train.rtc.enable = rtc
        cfg.train.faster.enable = faster
        if rtc:
            cfg.train.rtc.execution_horizon = 3
            cfg.train.rtc.overlap_ratio = 0.333
            cfg.train.rtc.prev_chunk_steps = 2
        if faster:
            cfg.train.faster.near_ratio = 0.3
            cfg.train.faster.near_steps = 2
            cfg.train.faster.far_steps = 4
        return cfg

    def test_rtc_loss_appears(self):
        """Stage C with RTC enabled should produce loss_rtc."""
        cfg = self._make_stage_c_cfg(rtc=True)
        model = _build_model(cfg)
        batch = _make_batch("c", cfg)
        losses = model.forward_train(batch)
        assert "loss_rtc" in losses
        assert not torch.isnan(losses["loss_rtc"])

    def test_faster_weighted_fm(self):
        """Stage C with FASTER enabled should still produce loss_fm."""
        cfg = self._make_stage_c_cfg(faster=True)
        model = _build_model(cfg)
        batch = _make_batch("c", cfg)
        losses = model.forward_train(batch)
        assert "loss_fm" in losses
        assert not torch.isnan(losses["loss_fm"])

    def test_rtc_and_faster_together(self):
        """Stage C with both RTC + FASTER should produce all losses."""
        cfg = self._make_stage_c_cfg(rtc=True, faster=True)
        model = _build_model(cfg)
        batch = _make_batch("c", cfg)
        losses = model.forward_train(batch)
        assert "loss_rtc" in losses
        assert "loss_fm" in losses
        losses["loss_total"].backward()
        # Should not crash
