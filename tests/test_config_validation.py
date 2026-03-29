"""Tests for config validation (fail-fast on bad configs).

Covers:
1. proprio_key / proprio_keys consistency
2. V1-style proprio_key with LIBERO format
3. Multi-camera self-consistency
4. proprio_dim vs proprio_keys mismatch
5. Invalid stage
6. grad_accum_steps < 1
7. Valid configs pass without error
"""

from __future__ import annotations

import pytest

from vla_hybrid_v2.config import (
    ActionExpertConfig,
    DataConfig,
    HybridVLAv2Config,
    ModelConfig,
    MultiCameraConfig,
    TrainConfig,
    validate_config,
)


def _base_libero_cfg() -> HybridVLAv2Config:
    """A valid LIBERO config — all checks should pass."""
    return HybridVLAv2Config(
        model=ModelConfig(
            multi_camera=MultiCameraConfig(
                enable=True,
                num_cameras=2,
                camera_names=["agentview", "eye_in_hand"],
            ),
            action_expert=ActionExpertConfig(action_dim=7),
            proprio_dim=9,
        ),
        data=DataConfig(
            format="libero_hdf5",
            proprio_key="joint_states",
            proprio_keys=["joint_states", "gripper_states"],
            camera_keys=["agentview_rgb", "eye_in_hand_rgb"],
        ),
        train=TrainConfig(grad_accum_steps=4),
        stage="b",
    )


class TestValidConfigPasses:
    """Sanity: valid configs should not raise."""

    def test_valid_libero_multicam(self):
        cfg = _base_libero_cfg()
        validate_config(cfg)  # should not raise

    def test_valid_default_config(self):
        """Default HybridVLAv2Config (dummy data, single cam) is valid."""
        cfg = HybridVLAv2Config()
        validate_config(cfg)

    def test_valid_singlecam_no_proprio_keys(self):
        """Single-cam, no proprio_keys list — just proprio_key."""
        cfg = HybridVLAv2Config(
            data=DataConfig(proprio_key="robot0_joint_pos", proprio_keys=[]),
        )
        validate_config(cfg)


class TestProprioKeyValidation:
    """Fail-fast on proprio_key / proprio_keys mismatches."""

    def test_proprio_key_not_in_proprio_keys(self):
        cfg = _base_libero_cfg()
        cfg.data.proprio_key = "wrong_key"
        with pytest.raises(ValueError, match="proprio_key='wrong_key'.*not in.*proprio_keys"):
            validate_config(cfg)

    def test_v1_proprio_key_with_libero_format(self):
        cfg = _base_libero_cfg()
        cfg.data.proprio_key = "robot0_joint_pos"
        cfg.data.proprio_keys = []  # clear to avoid double-error
        with pytest.raises(ValueError, match="robot0_joint_pos.*V1-style"):
            validate_config(cfg)

    def test_proprio_dim_mismatch(self):
        cfg = _base_libero_cfg()
        cfg.model.proprio_dim = 14  # wrong: should be 9 for LIBERO
        with pytest.raises(ValueError, match="proprio_dim=14.*sum to 9"):
            validate_config(cfg)


class TestMultiCameraValidation:
    """Fail-fast on multi-camera config inconsistencies."""

    def test_multicam_num_cameras_lt_2(self):
        cfg = _base_libero_cfg()
        cfg.model.multi_camera.num_cameras = 1
        with pytest.raises(ValueError, match="num_cameras=1.*requires.*>= 2"):
            validate_config(cfg)

    def test_camera_names_count_mismatch(self):
        cfg = _base_libero_cfg()
        cfg.model.multi_camera.camera_names = ["agentview"]  # only 1, but num_cameras=2
        with pytest.raises(ValueError, match="camera_names has 1.*num_cameras=2"):
            validate_config(cfg)

    def test_camera_keys_count_mismatch(self):
        cfg = _base_libero_cfg()
        cfg.data.camera_keys = ["agentview_rgb"]  # only 1, but num_cameras=2
        with pytest.raises(ValueError, match="camera_keys has 1.*num_cameras=2"):
            validate_config(cfg)

    def test_num_cameras_exceeds_max(self):
        cfg = _base_libero_cfg()
        cfg.model.multi_camera.num_cameras = 10
        cfg.model.multi_camera.max_cameras = 8
        with pytest.raises(ValueError, match="num_cameras=10.*exceeds.*max_cameras=8"):
            validate_config(cfg)

    def test_multicam_disabled_skips_checks(self):
        """When multi_camera.enable=False, camera count mismatches are ignored."""
        cfg = HybridVLAv2Config(
            model=ModelConfig(
                multi_camera=MultiCameraConfig(
                    enable=False,
                    num_cameras=1,  # would fail if enabled
                    camera_names=["x", "y", "z"],  # mismatch — but disabled
                ),
            ),
        )
        validate_config(cfg)  # should not raise


class TestStageValidation:
    def test_invalid_stage(self):
        cfg = HybridVLAv2Config(stage="d")
        with pytest.raises(ValueError, match="stage must be"):
            validate_config(cfg)


class TestGradAccumValidation:
    def test_grad_accum_zero(self):
        cfg = HybridVLAv2Config(train=TrainConfig(grad_accum_steps=0))
        with pytest.raises(ValueError, match="grad_accum_steps must be >= 1"):
            validate_config(cfg)

    def test_grad_accum_negative(self):
        cfg = HybridVLAv2Config(train=TrainConfig(grad_accum_steps=-1))
        with pytest.raises(ValueError, match="grad_accum_steps must be >= 1"):
            validate_config(cfg)


class TestMultipleErrors:
    """Validate that all errors are collected and reported together."""

    def test_reports_multiple_issues(self):
        cfg = _base_libero_cfg()
        cfg.stage = "z"
        cfg.data.proprio_key = "wrong_key"
        cfg.model.proprio_dim = 100
        cfg.train.grad_accum_steps = 0
        with pytest.raises(ValueError, match=r"4 issue\(s\)"):
            validate_config(cfg)
