"""Tests for online control_step() behavior."""

from __future__ import annotations

import pytest
import torch

from tests.conftest import A, L, P, _build_model, _mini_cfg


def _semantic_summary(model):
    return model.semantic_step(
        input_ids=torch.randint(0, 1000, (1, L)),
        attention_mask=torch.ones(1, L, dtype=torch.long),
        num_cameras=1,
    )


class TestControlStep:
    def test_default_rtc_path_runs_with_overlap_ratio(self):
        cfg = _mini_cfg("b")
        cfg.infer.execution_horizon = 2
        cfg.infer.rtc.enable = True
        model = _build_model(cfg)
        runtime = model.init_runtime(batch_size=1, device="cpu")

        out = model.control_step(
            proprio=torch.randn(1, P),
            prev_action=torch.zeros(1, A),
            semantic_summary=_semantic_summary(model),
            runtime_state=runtime,
            num_sample_steps=1,
        )

        assert out.action.shape == (1, A)
        assert runtime.prev_chunk_tail is not None

    def test_faster_infer_fails_fast_until_implemented(self):
        """L-6: FASTER is train-only — inference must fail fast with clear error."""
        cfg = _mini_cfg("b")
        cfg.infer.faster.enable = True
        model = _build_model(cfg)
        runtime = model.init_runtime(batch_size=1, device="cpu")

        with pytest.raises(NotImplementedError, match="FASTER"):
            model.control_step(
                proprio=torch.randn(1, P),
                prev_action=torch.zeros(1, A),
                semantic_summary=_semantic_summary(model),
                runtime_state=runtime,
                num_sample_steps=1,
            )
