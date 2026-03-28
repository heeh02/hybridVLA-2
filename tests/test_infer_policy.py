"""Tests for the unified LIBERO inference policy wrapper."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import A, P, _mini_cfg
from vla_hybrid_v2.data.normalizer import ActionNormalizer, ProprioNormalizer
from vla_hybrid_v2.infer.libero_policy import HybridVLALiberoPolicy
from vla_hybrid_v2.types import ControlStepOutput, GrounderOutput, RuntimeCache


def _fit_normalizers():
    action_norm = ActionNormalizer(target_range=(-1.0, 1.0))
    proprio_norm = ProprioNormalizer(target_range=(-1.0, 1.0))

    action_norm.fit(
        np.stack([
            np.full(A, -2.0, dtype=np.float32),
            np.full(A, 2.0, dtype=np.float32),
        ])
    )
    proprio_norm.fit(
        np.stack([
            np.zeros(P, dtype=np.float32),
            np.full(P, 10.0, dtype=np.float32),
        ])
    )
    return action_norm, proprio_norm


def _fake_grounder(batch_size: int = 1, dim: int = 64) -> GrounderOutput:
    return GrounderOutput(
        global_token=torch.zeros(batch_size, dim),
        object_slots=torch.zeros(batch_size, 4, dim),
        compressed_object_slots=torch.zeros(batch_size, 2, dim),
        phase_token=torch.zeros(batch_size, dim),
        uncertainty_token=torch.zeros(batch_size, dim),
        affordance_token=torch.zeros(batch_size, dim),
    )


class _DummyModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_proprio = None
        self.prev_actions_seen = []

    def init_runtime(self, batch_size=1, device="cpu"):
        return RuntimeCache(device=torch.device(device))

    def control_step(
        self,
        proprio,
        prev_action,
        semantic_summary,
        runtime_state,
        embodiment_id=None,
        num_sample_steps=8,
    ):
        self.last_proprio = proprio.detach().clone()
        self.prev_actions_seen.append(prev_action.detach().clone())
        action = torch.full_like(prev_action, 0.5)
        chunk = torch.full(
            (prev_action.shape[0], self.cfg.model.action_expert.chunk_horizon, prev_action.shape[1]),
            0.5,
            device=prev_action.device,
            dtype=prev_action.dtype,
        )
        return ControlStepOutput(action=action, chunk=chunk, chunk_step=1, semantic_refresh=False)


class TestHybridVLALiberoPolicy:
    def test_control_step_normalizes_proprio_and_denormalizes_action(self):
        cfg = _mini_cfg("b")
        cfg.data.proprio_keys = ["joint_states", "gripper_states"]
        cfg.data.proprio_key = "joint_states"
        action_norm, proprio_norm = _fit_normalizers()
        model = _DummyModel(cfg)
        policy = HybridVLALiberoPolicy(
            model=model,
            cfg=cfg,
            processor=object(),
            action_normalizer=action_norm,
            proprio_normalizer=proprio_norm,
            device="cpu",
        )
        runtime = policy.init_runtime()
        obs = {
            "robot0_joint_pos": np.full(7, 2.0, dtype=np.float32),
            "robot0_gripper_qpos": np.full(2, 2.0, dtype=np.float32),
        }

        step_1 = policy.control_step_from_obs(obs, runtime, _fake_grounder(dim=cfg.model.grounder.hidden_size))
        step_2 = policy.control_step_from_obs(obs, runtime, _fake_grounder(dim=cfg.model.grounder.hidden_size))

        expected_proprio = torch.full((1, P), -0.6)
        assert torch.allclose(model.last_proprio.cpu(), expected_proprio, atol=1e-5)
        assert torch.allclose(step_1.action_env.cpu(), torch.full((1, A), 1.0), atol=1e-5)
        assert torch.allclose(step_1.action_model.cpu(), torch.full((1, A), 0.5), atol=1e-5)
        assert torch.allclose(model.prev_actions_seen[0].cpu(), torch.zeros(1, A), atol=1e-5)
        assert torch.allclose(model.prev_actions_seen[1].cpu(), step_1.action_model.cpu(), atol=1e-5)
        assert torch.allclose(runtime.prev_action_model.cpu(), step_2.action_model.cpu(), atol=1e-5)

    def test_missing_proprio_keys_fail_fast(self):
        cfg = _mini_cfg("b")
        cfg.data.proprio_keys = ["joint_states", "gripper_states"]
        cfg.data.proprio_key = "joint_states"
        action_norm, proprio_norm = _fit_normalizers()
        policy = HybridVLALiberoPolicy(
            model=_DummyModel(cfg),
            cfg=cfg,
            processor=object(),
            action_normalizer=action_norm,
            proprio_normalizer=proprio_norm,
            device="cpu",
        )
        runtime = policy.init_runtime()
        obs = {"robot0_joint_pos": np.full(7, 2.0, dtype=np.float32)}

        with pytest.raises(KeyError, match="missing proprio keys"):
            policy.control_step_from_obs(obs, runtime, _fake_grounder(dim=cfg.model.grounder.hidden_size))

    def test_missing_processor_fails_in_semantic_step(self):
        cfg = _mini_cfg("b")
        cfg.data.proprio_keys = ["joint_states", "gripper_states"]
        cfg.data.proprio_key = "joint_states"
        action_norm, proprio_norm = _fit_normalizers()
        policy = HybridVLALiberoPolicy(
            model=_DummyModel(cfg),
            cfg=cfg,
            processor=None,
            action_normalizer=action_norm,
            proprio_normalizer=proprio_norm,
            device="cpu",
        )

        with pytest.raises(RuntimeError, match="Processor is required"):
            policy.semantic_step_from_obs(
                {
                    "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
                    "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
                },
                "pick up the block",
            )
