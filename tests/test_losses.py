"""Test loss functions: flow matching, discrete CE, consistency."""

from __future__ import annotations

import torch
import pytest

from tests.conftest import B, T, H, A, D


class TestFlowMatchingLoss:
    def test_perfect_prediction_zero_loss(self):
        from vla_hybrid_v2.losses.flow_matching import FlowMatchingLoss

        loss_fn = FlowMatchingLoss()
        x_0 = torch.randn(B, H, A)
        x_1 = torch.randn(B, H, A)
        target_v = x_1 - x_0
        t = torch.rand(B)
        loss = loss_fn(target_v, x_0, x_1, t)
        assert loss.item() < 1e-6

    def test_sample_timestep_range(self):
        from vla_hybrid_v2.losses.flow_matching import FlowMatchingLoss

        for schedule in ("logit_normal", "uniform"):
            loss_fn = FlowMatchingLoss(timestep_schedule=schedule)
            t = loss_fn.sample_timestep(1000, "cpu")
            assert (t >= 0).all() and (t <= 1).all()

    def test_interpolate(self):
        from vla_hybrid_v2.losses.flow_matching import FlowMatchingLoss

        x_0 = torch.zeros(B, H, A)
        x_1 = torch.ones(B, H, A)
        t = torch.full((B,), 0.5)
        x_t = FlowMatchingLoss.interpolate(x_0, x_1, t)
        assert torch.allclose(x_t, torch.full_like(x_t, 0.5), atol=1e-6)


class TestConsistencyLoss:
    def test_temporal_loss_runs(self):
        from vla_hybrid_v2.losses.consistency_loss import ContrastiveTemporalLoss

        loss_fn = ContrastiveTemporalLoss()
        fused = torch.randn(B, T, D)
        loss = loss_fn(fused)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_temporal_single_step_zero(self):
        from vla_hybrid_v2.losses.consistency_loss import ContrastiveTemporalLoss

        loss_fn = ContrastiveTemporalLoss()
        fused = torch.randn(B, 1, D)  # only 1 step
        loss = loss_fn(fused)
        assert loss.item() == 0.0

    def test_slow_fast_agreement(self):
        from vla_hybrid_v2.losses.consistency_loss import SlowFastAgreementLoss

        loss_fn = SlowFastAgreementLoss()
        fast = torch.randn(B, T, D)
        slow = torch.randn(B, D)
        loss = loss_fn(fast, slow)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_action_consistency(self):
        from vla_hybrid_v2.losses.consistency_loss import ActionConsistencyLoss

        loss_fn = ActionConsistencyLoss(action_dim=A)
        d = torch.randn(B, H, A)
        c = torch.randn(B, H, A)
        loss = loss_fn(d, c)
        assert loss.item() >= 0.0  # MSE is non-negative

    def test_combined_loss(self):
        from vla_hybrid_v2.losses.consistency_loss import V2ConsistencyLoss

        loss_fn = V2ConsistencyLoss(action_dim=A)
        loss = loss_fn(
            fused_states=torch.randn(B, T, D),
            fast_tokens=torch.randn(B, T, D),
            slow_token=torch.randn(B, D),
            discrete_actions=torch.randn(B, H, A),
            continuous_actions=torch.randn(B, H, A),
        )
        assert not torch.isnan(loss)


class TestDiscreteLoss:
    def test_output_is_scalar(self):
        from vla_hybrid_v2.losses.discrete_loss import DiscreteCELoss

        loss_fn = DiscreteCELoss(label_smoothing=0.1)
        V = 32
        logits = torch.randn(B, H, A, V)
        targets = torch.randint(0, V, (B, H, A))
        loss = loss_fn(logits, targets)
        assert loss.shape == ()
        assert not torch.isnan(loss)
