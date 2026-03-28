"""Stage A training — thin wrapper around train_unified.

DEPRECATED: This script is a legacy entry point. Use train_unified.py instead:

    python -m scripts.train_unified --config configs/train/stage_a.yaml

This wrapper exists only for backward compatibility and will be removed
in a future version.
"""

from __future__ import annotations

import argparse
import logging
import warnings

from vla_hybrid_v2.config import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    warnings.warn(
        "train_stage_a.py is deprecated. Use train_unified.py instead:\n"
        "  python -m scripts.train_unified --config configs/train/stage_a.yaml",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(description="HybridVLA v2 Stage A (deprecated)")
    parser.add_argument("--config", type=str, default="configs/train/stage_a.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.stage != "a":
        logger.warning("Config stage=%s but this is the Stage A entry point.", cfg.stage)
    from scripts.train_unified import train
    train(cfg)


if __name__ == "__main__":
    main()
