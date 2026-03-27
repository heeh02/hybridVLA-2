"""Test FlowActionExpert: forward, ODE solvers, AdaRMSNorm."""

from __future__ import annotations

import pytest
import torch

from tests.conftest import D, D_EXP, H, A, B, _mini_cfg


class TestAdaRMSNorm:
    def test_gate_bias_init(self):
        """Gate bias should be initialized to +2 (sigmoid ≈ 0.88)."""
        from vla_hybrid_v2.models.flow_action_expert import AdaRMSNorm

        norm = AdaRMSNorm(dim=64, cond_dim=32)
        gate_bias = norm.cond_proj.bias.data[128:]  # third chunk
        assert torch.allclose(gate_bias, torch.full_like(gate_bias, 2.0))

    def test_output_shape(self):
        from vla_hybrid_v2.models.flow_action_expert import AdaRMSNorm

        norm = AdaRMSNorm(dim=64, cond_dim=32)
        x = torch.randn(B, 10, 64)
        cond = torch.randn(B, 10, 32)
        out = norm(x, cond)
        assert out.shape == x.shape


class TestFlowActionExpert:
    @pytest.fixture
    def expert(self):
        from vla_hybrid_v2.models.flow_action_expert import FlowActionExpert

        # Constructor takes keyword args directly, not a config object
        return FlowActionExpert(
            d_model=D_EXP, num_layers=18, num_heads=2,
            chunk_horizon=H, action_dim=A, d_state=8,
            d_conv=4, expand=2, cond_dim=D, cond_tokens=8,
        )

    def test_forward_shape(self, expert):
        # cond_prefix has cond_dim=D; proprio/emb have d_model=D_EXP (expert internal)
        out = expert(
            noisy_actions=torch.randn(B, H, A),
            flow_t=torch.rand(B),
            cond_prefix=torch.randn(B, 8, D),     # cond_dim
            proprio_token=torch.randn(B, D_EXP),   # d_model
            embodiment_token=torch.randn(B, D_EXP), # d_model
        )
        assert out.velocity.shape == (B, H, A)

    def test_sample_euler(self, expert):
        expert.eval()
        cond = torch.randn(1, 8, D)      # cond_dim
        prop = torch.randn(1, D_EXP)      # d_model
        emb = torch.randn(1, D_EXP)       # d_model
        result = expert.sample_euler(cond, prop, emb, num_steps=4)
        assert result.shape == (1, H, A)
        assert not torch.isnan(result).any()

    def test_sample_midpoint(self, expert):
        expert.eval()
        cond = torch.randn(1, 8, D)
        prop = torch.randn(1, D_EXP)
        emb = torch.randn(1, D_EXP)
        result = expert.sample_midpoint(cond, prop, emb, num_steps=4)
        assert result.shape == (1, H, A)
        assert not torch.isnan(result).any()

    def test_sample_dispatch(self, expert):
        expert.eval()
        cond = torch.randn(1, 8, D)
        prop = torch.randn(1, D_EXP)
        emb = torch.randn(1, D_EXP)
        for solver in ("euler", "midpoint"):
            result = expert.sample(cond, prop, emb, num_steps=2, solver=solver)
            assert result.shape == (1, H, A)
