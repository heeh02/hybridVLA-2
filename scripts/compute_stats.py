"""Compute normalizer statistics from HDF5 episodes.

Reads all action and proprio data, fits normalizers, and saves statistics
for use during training via build_dataset().

Usage:
    # From config:
    python -m scripts.compute_stats --config configs/train/stage_a.yaml

    # Manual paths:
    python -m scripts.compute_stats \
        --data-dir /path/to/episodes \
        --output-dir outputs/v2_stage_a

    # Limit episodes for quick testing:
    python -m scripts.compute_stats --config ... --max-episodes 100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_stats(
    episode_paths: List[Path],
    action_key: str,
    proprio_key: str,
    output_dir: Path,
    action_strategy: str = "min_max",
    proprio_strategy: str = "min_max",
    action_range: Tuple[float, float] = (-1.0, 1.0),
    proprio_range: Tuple[float, float] = (-1.0, 1.0),
    max_episodes: Optional[int] = None,
) -> None:
    """Iterate over HDF5 episodes, fit normalizers, save stats."""
    import h5py

    from vla_hybrid_v2.data.normalizer import ActionNormalizer, ProprioNormalizer

    if max_episodes:
        episode_paths = episode_paths[:max_episodes]

    all_actions: List[np.ndarray] = []
    all_proprio: List[np.ndarray] = []
    skipped = 0

    for path in episode_paths:
        try:
            with h5py.File(path, "r") as f:
                if "data" not in f:
                    logger.warning("Skipping %s: no 'data' group", path)
                    skipped += 1
                    continue
                data = f["data"]
                if action_key not in data:
                    logger.warning(
                        "Skipping %s: missing '%s'. Available: %s",
                        path, action_key, list(data.keys()),
                    )
                    skipped += 1
                    continue
                if proprio_key not in data:
                    logger.warning(
                        "Skipping %s: missing '%s'. Available: %s",
                        path, proprio_key, list(data.keys()),
                    )
                    skipped += 1
                    continue
                all_actions.append(data[action_key][:].astype(np.float32))
                all_proprio.append(data[proprio_key][:].astype(np.float32))
        except Exception as e:
            logger.warning("Skipping %s: %s", path, e)
            skipped += 1

    if not all_actions:
        raise RuntimeError(
            f"No valid episodes found. Checked {len(episode_paths)} files, "
            f"skipped {skipped}."
        )

    logger.info(
        "Loaded %d episodes (%d skipped), fitting normalizers...",
        len(all_actions), skipped,
    )

    actions_np = np.concatenate(all_actions, axis=0)
    proprio_np = np.concatenate(all_proprio, axis=0)
    logger.info(
        "Action data: shape=%s, range=[%.4f, %.4f]",
        actions_np.shape, actions_np.min(), actions_np.max(),
    )
    logger.info(
        "Proprio data: shape=%s, range=[%.4f, %.4f]",
        proprio_np.shape, proprio_np.min(), proprio_np.max(),
    )

    action_norm = ActionNormalizer(strategy=action_strategy, target_range=action_range)
    proprio_norm = ProprioNormalizer(strategy=proprio_strategy, target_range=proprio_range)

    action_norm.fit(actions_np)
    proprio_norm.fit(proprio_np)

    stats_dir = output_dir if output_dir.name == "normalizer_stats" else output_dir / "normalizer_stats"
    action_norm.save(stats_dir / "action_stats.json")
    proprio_norm.save(stats_dir / "proprio_stats.json")

    logger.info("Stats saved to %s", stats_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute normalizer stats from HDF5 episodes",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config (reads data.data_dir, data.action_key, etc.)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override: directory containing .hdf5 files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override: where to save normalizer stats",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.config:
        from vla_hybrid_v2.config import load_config

        cfg = load_config(args.config)
        data_dir = Path(args.data_dir or cfg.data.data_dir)
        # Use explicit stats dir from config if set, else fall back to output_dir
        output_dir = Path(
            args.output_dir
            or cfg.data.normalizer_stats_dir
            or cfg.train.output_dir
        )
        action_key = cfg.data.action_key
        proprio_key = cfg.data.proprio_key
        action_range = cfg.model.heads.action_range
        proprio_range = cfg.model.proprio_range
        max_episodes = args.max_episodes or cfg.data.max_episodes
    else:
        if not args.data_dir or not args.output_dir:
            parser.error("Provide --config OR both --data-dir and --output-dir")
        data_dir = Path(args.data_dir)
        output_dir = Path(args.output_dir)
        action_key = "actions"
        proprio_key = "robot0_joint_pos"
        action_range = (-1.0, 1.0)
        proprio_range = (-1.0, 1.0)
        max_episodes = args.max_episodes

    episode_paths = sorted(data_dir.glob("*.hdf5"))
    if not episode_paths:
        raise FileNotFoundError(f"No .hdf5 files in {data_dir}")
    logger.info("Found %d HDF5 files in %s", len(episode_paths), data_dir)

    compute_stats(
        episode_paths=episode_paths,
        action_key=action_key,
        proprio_key=proprio_key,
        output_dir=output_dir,
        action_range=action_range,
        proprio_range=proprio_range,
        max_episodes=max_episodes,
    )


if __name__ == "__main__":
    main()
