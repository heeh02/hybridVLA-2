"""Compute LIBERO normalization statistics from official task HDF5 files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import h5py
import numpy as np

from libero_hybrid.utils import resolve_libero_suite_dir, sorted_libero_demo_keys, suite_output_root
from vla_hybrid_v2.config import load_config
from vla_hybrid_v2.data.normalizer import ActionNormalizer, ProprioNormalizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = REPO_ROOT / "libero_hybrid/configs/train/libero_stage_a.yaml"
logger = logging.getLogger(__name__)


def compute_libero_stats(
    task_paths: List[Path],
    action_key: str,
    proprio_keys: List[str],
    output_dir: Path,
    action_range: tuple[float, float],
    proprio_range: tuple[float, float],
    max_episodes: int | None,
) -> None:
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")

    all_actions: List[np.ndarray] = []
    all_proprio: List[np.ndarray] = []
    demos_seen = 0
    skipped = 0

    for path in task_paths:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                logger.warning("Skipping %s: no 'data' group", path)
                skipped += 1
                continue
            data_grp = f["data"]
            for demo_key in sorted_libero_demo_keys(data_grp):
                demo_grp = data_grp[demo_key]
                if action_key not in demo_grp or "obs" not in demo_grp:
                    skipped += 1
                    continue
                obs_grp = demo_grp["obs"]
                missing = [k for k in proprio_keys if k not in obs_grp]
                if missing:
                    skipped += 1
                    continue

                all_actions.append(demo_grp[action_key][:].astype(np.float32))
                # Concatenate all proprio keys (e.g., joint_states + gripper_states)
                parts = [obs_grp[pk][:].astype(np.float32) for pk in proprio_keys]
                all_proprio.append(np.concatenate(parts, axis=-1))
                demos_seen += 1
                if max_episodes is not None and demos_seen >= max_episodes:
                    break

        if max_episodes is not None and demos_seen >= max_episodes:
            break

    if not all_actions:
        raise RuntimeError(
            f"No valid LIBERO demos found. Checked {len(task_paths)} task files, skipped={skipped}."
        )

    actions_np = np.concatenate(all_actions, axis=0)
    proprio_np = np.concatenate(all_proprio, axis=0)
    logger.info(
        "Loaded %d demos (%d skipped). action_shape=%s proprio_shape=%s",
        demos_seen, skipped, actions_np.shape, proprio_np.shape,
    )

    action_norm = ActionNormalizer(target_range=action_range)
    proprio_norm = ProprioNormalizer(target_range=proprio_range)
    action_norm.fit(actions_np)
    proprio_norm.fit(proprio_np)

    stats_dir = output_dir if output_dir.name == "normalizer_stats" else output_dir / "normalizer_stats"
    action_norm.save(stats_dir / "action_stats.json")
    proprio_norm.save(stats_dir / "proprio_stats.json")
    logger.info("Stats saved to %s", stats_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LIBERO normalization stats")
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--suite", default="libero_spatial", type=str)
    parser.add_argument("--output-root", default="outputs/libero_hybrid", type=str)
    parser.add_argument("--config", default=str(DEFAULT_CFG), type=str)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    data_dir = resolve_libero_suite_dir(args.data_dir, args.suite)
    output_root = suite_output_root(args.output_root, args.suite)
    stats_dir = output_root / "normalizer_stats"
    task_paths = sorted(data_dir.glob("*.hdf5"))

    if not task_paths:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")

    # Use proprio_keys if set, else fall back to single key
    proprio_keys = cfg.data.proprio_keys if cfg.data.proprio_keys else [cfg.data.proprio_key]
    compute_libero_stats(
        task_paths=task_paths,
        action_key=cfg.data.action_key,
        proprio_keys=proprio_keys,
        output_dir=stats_dir,
        action_range=cfg.model.heads.action_range,
        proprio_range=cfg.model.proprio_range,
        max_episodes=args.max_episodes,
    )

    print(f"Suite: {args.suite}")
    print(f"Resolved data dir: {data_dir}")
    print(f"Saved normalizer stats to: {stats_dir}")


if __name__ == "__main__":
    main()
