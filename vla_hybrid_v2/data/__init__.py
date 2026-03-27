"""HybridVLA v2 data pipeline.

Public API:
    - WindowSample: the data contract between adapters and model
    - Normalizer / ActionNormalizer / ProprioNormalizer: value normalization
    - BaseDatasetAdapter: abstract base for format-specific adapters
    - DummyVLADataset: random-tensor dataset for smoke tests
    - HDF5DatasetAdapter: minimal real-data loader
    - vla_collate_fn: custom collate for variable-length vision fields
    - build_dataset: factory function for dataset construction
"""

from vla_hybrid_v2.data.collate import vla_collate_fn
from vla_hybrid_v2.data.dummy import DummyVLADataset
from vla_hybrid_v2.data.normalizer import ActionNormalizer, Normalizer, ProprioNormalizer
from vla_hybrid_v2.data.schema import WindowSample

__all__ = [
    "WindowSample",
    "Normalizer",
    "ActionNormalizer",
    "ProprioNormalizer",
    "DummyVLADataset",
    "vla_collate_fn",
    "build_dataset",
]


def build_dataset(cfg, split="train", processor=None):
    """Factory: build dataset from config.

    Returns (dataset, collate_fn) tuple ready for DataLoader.

    If cfg.data.format is set, constructs the corresponding adapter.
    Otherwise falls back to DummyVLADataset.
    """
    from vla_hybrid_v2.config import HybridVLAv2Config

    if not isinstance(cfg, HybridVLAv2Config):
        raise TypeError(f"Expected HybridVLAv2Config, got {type(cfg)}")

    fmt = cfg.data.format

    if fmt == "hdf5":
        from vla_hybrid_v2.data.hdf5_adapter import HDF5DatasetAdapter

        action_norm = ActionNormalizer(target_range=cfg.model.heads.action_range)
        proprio_norm = ProprioNormalizer(target_range=cfg.model.proprio_range)

        # Load pre-computed stats — explicit path or fallback to output_dir
        from pathlib import Path
        if cfg.data.normalizer_stats_dir:
            stats_dir = Path(cfg.data.normalizer_stats_dir)
        else:
            stats_dir = Path(cfg.train.output_dir) / "normalizer_stats"
        action_stats = stats_dir / "action_stats.json"
        proprio_stats = stats_dir / "proprio_stats.json"

        if action_stats.exists() and proprio_stats.exists():
            action_norm.load(action_stats)
            proprio_norm.load(proprio_stats)
        else:
            raise FileNotFoundError(
                f"Normalizer stats not found at {stats_dir}. "
                f"Run data preparation to compute action/proprio statistics first."
            )

        dataset = HDF5DatasetAdapter(
            cfg, action_norm, proprio_norm, processor=processor, split=split,
        )
        return dataset, vla_collate_fn

    elif fmt is None or fmt == "dummy":
        size = cfg.train.max_steps * cfg.train.per_device_batch_size * 2
        dataset = DummyVLADataset(size=size, cfg=cfg)
        return dataset, None  # default collate is fine for dummy

    else:
        raise ValueError(
            f"Unknown data format: '{fmt}'. "
            f"Supported: 'hdf5', 'dummy', or None (defaults to dummy)."
        )
