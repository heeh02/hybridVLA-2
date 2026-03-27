"""Train HybridVLA on LIBERO with explicit real-data overrides.

This wrapper exists to avoid silently falling back to dummy data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import yaml

from libero_hybrid.utils import resolve_libero_suite_dir, suite_output_root
from scripts.train_unified import train
from vla_hybrid_v2.config import HybridVLAv2Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBERO_ROOT = REPO_ROOT / "libero_hybrid"
DEFAULT_CONFIGS = {
    "a": LIBERO_ROOT / "configs/train/libero_stage_a.yaml",
    "b": LIBERO_ROOT / "configs/train/libero_stage_b.yaml",
    "c": LIBERO_ROOT / "configs/train/libero_stage_c.yaml",
}


def _apply_variant(cfg: HybridVLAv2Config, variant: str) -> None:
    # LIBERO Franka Panda: 7-dim actions, 9-dim proprio (joint_states + gripper_states)
    cfg.model.action_expert.action_dim = 7
    cfg.model.proprio_dim = 9
    cfg.data.proprio_key = "joint_states"
    cfg.data.proprio_keys = ["joint_states", "gripper_states"]
    cfg.data.image_key = "agentview_rgb"

    if variant == "singlecam":
        cfg.model.multi_camera.enable = False
        cfg.data.max_text_length = 256
        return

    if variant != "multicam":
        raise ValueError(f"Unknown variant: {variant}")

    cfg.model.multi_camera.enable = True
    cfg.model.multi_camera.num_cameras = 2
    cfg.model.multi_camera.camera_names = [
        "agentview",
        "eye_in_hand",
    ]
    cfg.data.camera_keys = [
        "agentview_rgb",
        "eye_in_hand_rgb",
    ]
    cfg.data.max_text_length = 1024


def _resolve_stage_paths(output_root: Path, stage: str) -> tuple[Path, Optional[Path]]:
    stage_dir = output_root / f"stage_{stage}"
    resume = None
    if stage == "b":
        resume = output_root / "stage_a" / "checkpoint-latest"
    elif stage == "c":
        resume = output_root / "stage_b" / "checkpoint-latest"
    return stage_dir, resume


def _save_resolved_config(cfg: HybridVLAv2Config) -> Path:
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / "resolved_config.yaml"
    with open(dst, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
    return dst


def build_cfg(
    stage: str,
    data_dir: Optional[str],
    suite: str,
    output_root: str,
    stats_dir: Optional[str],
    variant: str,
    val_data_dir: Optional[str],
    max_episodes: Optional[int],
    config_path: Optional[str],
    resume_from: Optional[str],
) -> HybridVLAv2Config:
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIGS[stage]
    cfg = load_config(cfg_path)

    cfg.stage = stage
    cfg.data.format = "libero_hdf5"
    if data_dir is not None:
        cfg.data.data_dir = str(resolve_libero_suite_dir(data_dir, suite))
    if val_data_dir is not None:
        cfg.data.val_data_dir = str(resolve_libero_suite_dir(val_data_dir, suite))
    if max_episodes is not None:
        cfg.data.max_episodes = max_episodes

    _apply_variant(cfg, variant)

    output_root_p = suite_output_root(output_root, suite)
    stage_dir, auto_resume = _resolve_stage_paths(output_root_p, stage)
    cfg.train.output_dir = str(stage_dir)
    cfg.data.normalizer_stats_dir = (
        str(Path(stats_dir).resolve())
        if stats_dir is not None
        else str((output_root_p / "normalizer_stats").resolve())
    )

    if stage in ("b", "c"):
        cfg.train.resume_from = (
            str(Path(resume_from).resolve())
            if resume_from is not None
            else str(auto_resume.resolve())
        )
    else:
        cfg.train.resume_from = None

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HybridVLA on LIBERO")
    parser.add_argument("--stage", required=True, choices=["a", "b", "c"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--val-data-dir", type=str, default=None)
    parser.add_argument("--stats-dir", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="outputs/libero_hybrid")
    parser.add_argument("--variant", choices=["singlecam", "multicam"], default="singlecam")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and args.data_dir is None:
        parser.error("--data-dir is required unless --dry-run is used")

    cfg = build_cfg(
        stage=args.stage,
        data_dir=args.data_dir,
        suite=args.suite,
        output_root=args.output_root,
        stats_dir=args.stats_dir,
        variant=args.variant,
        val_data_dir=args.val_data_dir,
        max_episodes=args.max_episodes,
        config_path=args.config,
        resume_from=args.resume_from,
    )

    resolved = _save_resolved_config(cfg)
    print(f"Resolved config saved to: {resolved}")
    print(f"Stage: {cfg.stage}")
    print(f"Suite: {args.suite}")
    print(f"Variant: {args.variant}")
    print(f"Data format: {cfg.data.format}")
    print(f"Data dir: {cfg.data.data_dir}")
    print(f"Val data dir: {cfg.data.val_data_dir}")
    print(f"Stats dir: {cfg.data.normalizer_stats_dir}")
    print(f"Output dir: {cfg.train.output_dir}")
    print(f"Resume from: {cfg.train.resume_from}")

    if args.dry_run:
        return

    train(cfg)


if __name__ == "__main__":
    main()
