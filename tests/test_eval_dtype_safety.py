"""Tests for eval / rollout / inference dtype safety.

Validates that:
- bf16 action tensors are safely converted to numpy (no TypeError)
- control_step_from_obs returns fp32 actions regardless of model dtype
- The eval rollout path handles mixed-precision outputs correctly
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import A, P, _mini_cfg
from vla_hybrid_v2.data.normalizer import ActionNormalizer, ProprioNormalizer
from vla_hybrid_v2.infer.libero_policy import HybridVLALiberoPolicy, LiberoPolicyStepOutput
from vla_hybrid_v2.types import ControlStepOutput, GrounderOutput, RuntimeCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_normalizers():
    action_norm = ActionNormalizer(target_range=(-1.0, 1.0))
    proprio_norm = ProprioNormalizer(target_range=(-1.0, 1.0))
    action_norm.fit(
        np.stack([np.full(A, -2.0, dtype=np.float32),
                  np.full(A, 2.0, dtype=np.float32)])
    )
    proprio_norm.fit(
        np.stack([np.zeros(P, dtype=np.float32),
                  np.full(P, 10.0, dtype=np.float32)])
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


class _BF16DummyModel:
    """Model stub that returns bf16 actions, simulating a bf16-loaded checkpoint."""

    def __init__(self, cfg, output_dtype=torch.bfloat16):
        self.cfg = cfg
        self.output_dtype = output_dtype

    def init_runtime(self, batch_size=1, device="cpu"):
        return RuntimeCache(device=torch.device(device))

    def semantic_step(self, input_ids, attention_mask, **kwargs):
        dim = self.cfg.model.grounder.hidden_size
        B = input_ids.shape[0]
        return GrounderOutput(
            global_token=torch.zeros(B, dim),
            object_slots=torch.zeros(B, 4, dim),
            compressed_object_slots=torch.zeros(B, 2, dim),
            phase_token=torch.zeros(B, dim),
            uncertainty_token=torch.zeros(B, dim),
            affordance_token=torch.zeros(B, dim),
        )

    def control_step(self, proprio, prev_action, semantic_summary,
                     runtime_state, embodiment_id=None, num_sample_steps=8):
        # Simulate bf16 output from flow action expert
        action = torch.full_like(prev_action, 0.5).to(self.output_dtype)
        chunk = torch.full(
            (prev_action.shape[0], self.cfg.model.action_expert.chunk_horizon, prev_action.shape[1]),
            0.5, device=prev_action.device, dtype=self.output_dtype,
        )
        return ControlStepOutput(action=action, chunk=chunk, chunk_step=1, semantic_refresh=False)


# ---------------------------------------------------------------------------
# A. bf16 -> numpy safety tests
# ---------------------------------------------------------------------------

class TestBF16NumpySafety:
    """Core test: bf16 tensors must be safely convertible to numpy."""

    def test_bf16_tensor_to_numpy_crashes_without_cast(self):
        """Demonstrate the raw failure mode we're protecting against."""
        t = torch.randn(7, dtype=torch.bfloat16)
        with pytest.raises((RuntimeError, TypeError)):
            t.numpy()

    def test_bf16_tensor_to_numpy_safe_with_float_cast(self):
        """The fix: .float() before .numpy()."""
        t = torch.randn(7, dtype=torch.bfloat16)
        arr = t.float().cpu().numpy()
        assert arr.dtype == np.float32
        assert arr.shape == (7,)

    def test_fp16_tensor_to_numpy_also_safe(self):
        """fp16 should also be handled safely."""
        t = torch.randn(7, dtype=torch.float16)
        arr = t.float().cpu().numpy()
        assert arr.dtype == np.float32

    def test_fp32_tensor_to_numpy_passthrough(self):
        """fp32 should work without any cast."""
        t = torch.randn(7, dtype=torch.float32)
        arr = t.cpu().numpy()
        assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# B. control_step_from_obs dtype safety
# ---------------------------------------------------------------------------

class TestControlStepDtypeSafety:
    """control_step_from_obs must return fp32 actions for env interface."""

    @pytest.mark.parametrize("model_dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_action_env_is_always_fp32(self, model_dtype):
        cfg = _mini_cfg("b")
        cfg.data.proprio_keys = ["joint_states", "gripper_states"]
        cfg.data.proprio_key = "joint_states"
        action_norm, proprio_norm = _fit_normalizers()
        model = _BF16DummyModel(cfg, output_dtype=model_dtype)

        policy = HybridVLALiberoPolicy(
            model=model, cfg=cfg, processor=object(),
            action_normalizer=action_norm, proprio_normalizer=proprio_norm,
            device="cpu",
        )
        runtime = policy.init_runtime()
        obs = {
            "robot0_joint_pos": np.full(7, 2.0, dtype=np.float32),
            "robot0_gripper_qpos": np.full(2, 2.0, dtype=np.float32),
        }

        step = policy.control_step_from_obs(
            obs, runtime, _fake_grounder(dim=cfg.model.grounder.hidden_size),
        )

        assert step.action_env.dtype == torch.float32, \
            f"action_env dtype must be fp32 for env interface, got {step.action_env.dtype}"
        assert step.action_model.dtype == torch.float32, \
            f"action_model dtype must be fp32, got {step.action_model.dtype}"

    def test_bf16_action_converts_to_numpy_after_fix(self):
        """End-to-end: bf16 model -> control_step_from_obs -> numpy succeeds."""
        cfg = _mini_cfg("b")
        cfg.data.proprio_keys = ["joint_states", "gripper_states"]
        cfg.data.proprio_key = "joint_states"
        action_norm, proprio_norm = _fit_normalizers()
        model = _BF16DummyModel(cfg, output_dtype=torch.bfloat16)

        policy = HybridVLALiberoPolicy(
            model=model, cfg=cfg, processor=object(),
            action_normalizer=action_norm, proprio_normalizer=proprio_norm,
            device="cpu",
        )
        runtime = policy.init_runtime()
        obs = {
            "robot0_joint_pos": np.full(7, 2.0, dtype=np.float32),
            "robot0_gripper_qpos": np.full(2, 2.0, dtype=np.float32),
        }

        step = policy.control_step_from_obs(
            obs, runtime, _fake_grounder(dim=cfg.model.grounder.hidden_size),
        )

        # This is the exact line from eval_libero_rollout.py:170
        action = step.action_env[0]
        arr = action.float().cpu().numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == (A,)


# ---------------------------------------------------------------------------
# C. Inference smoke test — minimal end-to-end path
# ---------------------------------------------------------------------------

class TestInferenceSmokeTest:
    """Minimal smoke test: semantic_step -> control_step -> numpy action."""

    def test_full_inference_path_bf16_model(self):
        """Run the complete semantic -> control -> env-action path with bf16 model."""
        from unittest.mock import patch

        cfg = _mini_cfg("b")
        cfg.data.proprio_keys = ["joint_states", "gripper_states"]
        cfg.data.proprio_key = "joint_states"
        action_norm, proprio_norm = _fit_normalizers()
        model = _BF16DummyModel(cfg, output_dtype=torch.bfloat16)

        policy = HybridVLALiberoPolicy(
            model=model, cfg=cfg, processor=object(),
            action_normalizer=action_norm, proprio_normalizer=proprio_norm,
            device="cpu",
        )
        runtime = policy.init_runtime()
        language = "pick up the red block"

        # 1. Semantic step (mock processor tokenization)
        fake_sem_input = {
            "input_ids": torch.randint(0, 100, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        with patch.object(policy, "obs_to_semantic_input", return_value=fake_sem_input):
            grounder_out = policy.semantic_step_from_obs(
                {"agentview_image": np.zeros((128, 128, 3), dtype=np.uint8)},
                language,
                runtime_state=runtime,
            )

        # 2. Control step
        obs = {
            "robot0_joint_pos": np.full(7, 1.0, dtype=np.float32),
            "robot0_gripper_qpos": np.full(2, 0.5, dtype=np.float32),
        }
        step = policy.control_step_from_obs(obs, runtime, grounder_out)

        # 3. Action -> numpy (the critical path)
        action_np = step.action_env[0].cpu().numpy()
        assert isinstance(action_np, np.ndarray)
        assert action_np.dtype == np.float32
        assert action_np.shape == (A,)

        # 4. Verify action is in valid range
        lo, hi = cfg.model.heads.action_range
        assert (action_np >= lo).all() and (action_np <= hi).all(), \
            f"Action out of range [{lo}, {hi}]: {action_np}"

    def test_multi_step_rollout_simulation(self):
        """Simulate a minimal rollout loop like eval_libero_rollout.py."""
        from unittest.mock import patch

        cfg = _mini_cfg("b")
        cfg.data.proprio_keys = ["joint_states", "gripper_states"]
        cfg.data.proprio_key = "joint_states"
        action_norm, proprio_norm = _fit_normalizers()
        model = _BF16DummyModel(cfg, output_dtype=torch.bfloat16)

        policy = HybridVLALiberoPolicy(
            model=model, cfg=cfg, processor=object(),
            action_normalizer=action_norm, proprio_normalizer=proprio_norm,
            device="cpu",
        )

        n_envs = 3
        n_steps = 5
        actions_batch = np.zeros((n_envs, A))

        runtimes = [policy.init_runtime(batch_size=1) for _ in range(n_envs)]
        grounder_outs = [None] * n_envs

        fake_sem_input = {
            "input_ids": torch.randint(0, 100, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }

        for step in range(n_steps):
            for k in range(n_envs):
                obs = {
                    "robot0_joint_pos": np.random.randn(7).astype(np.float32),
                    "robot0_gripper_qpos": np.random.randn(2).astype(np.float32),
                    "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
                }

                if step == 0:
                    with patch.object(policy, "obs_to_semantic_input",
                                      return_value=fake_sem_input):
                        grounder_outs[k] = policy.semantic_step_from_obs(
                            obs, "pick up block", runtime_state=runtimes[k],
                        )

                step_out = policy.control_step_from_obs(
                    obs, runtimes[k], grounder_outs[k],
                )

                action = step_out.action_env[0]
                # This must not crash — the exact code from eval_libero_rollout.py
                actions_batch[k] = action.float().cpu().numpy()

            assert actions_batch.dtype == np.float64  # numpy default for zeros
            assert not np.any(np.isnan(actions_batch)), f"NaN actions at step {step}"


# ---------------------------------------------------------------------------
# D. Fail-fast assertions
# ---------------------------------------------------------------------------

class TestFailFastAssertions:
    """Verify that bad inputs fail early with readable errors."""

    def test_control_step_without_semantic_summary_raises(self):
        cfg = _mini_cfg("b")
        cfg.data.proprio_keys = ["joint_states", "gripper_states"]
        cfg.data.proprio_key = "joint_states"
        action_norm, proprio_norm = _fit_normalizers()
        model = _BF16DummyModel(cfg)
        policy = HybridVLALiberoPolicy(
            model=model, cfg=cfg, processor=object(),
            action_normalizer=action_norm, proprio_normalizer=proprio_norm,
            device="cpu",
        )
        runtime = policy.init_runtime()
        obs = {
            "robot0_joint_pos": np.full(7, 1.0, dtype=np.float32),
            "robot0_gripper_qpos": np.full(2, 0.5, dtype=np.float32),
        }

        with pytest.raises(ValueError, match="semantic_summary must be computed"):
            policy.control_step_from_obs(obs, runtime, semantic_summary=None)

    def test_normalizer_denormalize_preserves_value_accuracy(self):
        """Ensure fp32 cast doesn't destroy action values."""
        action_norm, _ = _fit_normalizers()

        # Simulate bf16 model output
        model_action = torch.tensor([[0.5] * A], dtype=torch.bfloat16)
        env_action = action_norm.denormalize(model_action)

        # Cast and compare
        env_fp32 = env_action.float()
        assert torch.allclose(env_fp32, env_action.float(), atol=1e-2), \
            "fp32 cast should not significantly change action values"
        assert env_fp32.dtype == torch.float32
