"""Tests for eval_libero_rollout.py config resolution logic.

Covers:
- Auto-discovery of resolved_config.yaml from checkpoint directory
- Symlink (checkpoint-latest) resolution
- Mismatch detection between --config and resolved_config.yaml
- Failure when no config is available
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def _minimal_config(multi_camera_enable: bool = False) -> dict:
    """Minimal YAML config dict sufficient for load_config()."""
    return {
        "model": {
            "backbone": {"name": "mock"},
            "multi_camera": {"enable": multi_camera_enable},
            "action_expert": {"action_dim": 7},
            "proprio_dim": 9,
        },
        "train": {"output_dir": "/tmp/test_out"},
        "data": {"format": "dummy"},
        "stage": "a",
    }


@pytest.fixture
def stage_dir(tmp_path):
    """Create a mock stage output directory with checkpoint + resolved_config."""
    stage = tmp_path / "outputs" / "libero_spatial" / "stage_c"

    # resolved_config.yaml (singlecam)
    _write_yaml(stage / "resolved_config.yaml", _minimal_config(multi_camera_enable=False))

    # checkpoint-5000/meta.json
    ckpt = stage / "checkpoint-5000"
    ckpt.mkdir(parents=True)
    (ckpt / "meta.json").write_text(json.dumps({"step": 5000}))

    # checkpoint-latest -> checkpoint-5000
    latest = stage / "checkpoint-latest"
    latest.symlink_to("checkpoint-5000")

    return stage


@pytest.fixture
def multicam_stage_dir(tmp_path):
    """Stage dir where training used multi_camera.enable=True."""
    stage = tmp_path / "outputs" / "libero_spatial" / "stage_c"
    _write_yaml(stage / "resolved_config.yaml", _minimal_config(multi_camera_enable=True))

    ckpt = stage / "checkpoint-5000"
    ckpt.mkdir(parents=True)
    (ckpt / "meta.json").write_text(json.dumps({"step": 5000}))

    latest = stage / "checkpoint-latest"
    latest.symlink_to("checkpoint-5000")

    return stage


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFindResolvedConfig:
    """Test _find_resolved_config discovery logic."""

    def test_finds_from_checkpoint_dir(self, stage_dir):
        from libero_hybrid.scripts.eval_libero_rollout import _find_resolved_config

        # Direct checkpoint path: checkpoint-5000/
        result = _find_resolved_config(str(stage_dir / "checkpoint-5000"))
        assert result is not None
        assert result.name == "resolved_config.yaml"

    def test_finds_from_symlink(self, stage_dir):
        from libero_hybrid.scripts.eval_libero_rollout import _find_resolved_config

        # Symlink: checkpoint-latest -> checkpoint-5000
        result = _find_resolved_config(str(stage_dir / "checkpoint-latest"))
        assert result is not None
        assert result.name == "resolved_config.yaml"

    def test_returns_none_when_missing(self, tmp_path):
        from libero_hybrid.scripts.eval_libero_rollout import _find_resolved_config

        # No resolved_config.yaml anywhere
        orphan = tmp_path / "no_config" / "checkpoint-100"
        orphan.mkdir(parents=True)
        result = _find_resolved_config(str(orphan))
        assert result is None


class TestConfigMismatchDetection:
    """Test that load_hybridvla_policy rejects multi_camera mismatches."""

    def test_singlecam_config_against_multicam_checkpoint_raises(
        self, multicam_stage_dir, tmp_path,
    ):
        """Passing a singlecam --config when checkpoint was trained multicam must fail."""
        from libero_hybrid.scripts.eval_libero_rollout import load_hybridvla_policy

        singlecam_yaml = _write_yaml(
            tmp_path / "singlecam.yaml",
            _minimal_config(multi_camera_enable=False),
        )

        with pytest.raises(RuntimeError, match="Config mismatch"):
            load_hybridvla_policy(
                checkpoint_path=str(multicam_stage_dir / "checkpoint-5000"),
                config_path=str(singlecam_yaml),
                device="cpu",
            )

    def test_multicam_config_against_singlecam_checkpoint_raises(
        self, stage_dir, tmp_path,
    ):
        """Passing a multicam --config when checkpoint was trained singlecam must fail."""
        from libero_hybrid.scripts.eval_libero_rollout import load_hybridvla_policy

        multicam_yaml = _write_yaml(
            tmp_path / "multicam.yaml",
            _minimal_config(multi_camera_enable=True),
        )

        with pytest.raises(RuntimeError, match="Config mismatch"):
            load_hybridvla_policy(
                checkpoint_path=str(stage_dir / "checkpoint-5000"),
                config_path=str(multicam_yaml),
                device="cpu",
            )


class TestNoConfigFails:
    """Test that omitting --config without resolved_config.yaml fails clearly."""

    def test_no_config_no_resolved_raises(self, tmp_path):
        from libero_hybrid.scripts.eval_libero_rollout import load_hybridvla_policy

        orphan = tmp_path / "no_config" / "checkpoint-100"
        orphan.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="resolved_config.yaml"):
            load_hybridvla_policy(
                checkpoint_path=str(orphan),
                config_path=None,
                device="cpu",
            )
