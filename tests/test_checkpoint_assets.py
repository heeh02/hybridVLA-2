"""Tests for checkpoint-side asset packaging."""

from __future__ import annotations

import json

import torch
from torch import nn

from vla_hybrid_v2.utils.checkpointing import save_checkpoint


def test_save_checkpoint_copies_inference_assets(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    resolved_cfg = tmp_path / "resolved_config.yaml"
    resolved_cfg.write_text("stage: a\n")

    stats_dir = tmp_path / "normalizer_stats"
    stats_dir.mkdir()
    (stats_dir / "action_stats.json").write_text(json.dumps({"target_range": [-1, 1]}))
    (stats_dir / "proprio_stats.json").write_text(json.dumps({"target_range": [-1, 1]}))

    ckpt = save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=7,
        output_dir=tmp_path / "out",
        asset_paths={
            "resolved_config.yaml": resolved_cfg,
            "normalizer_stats": stats_dir,
        },
    )

    assert ckpt is not None
    assert (ckpt / "assets" / "resolved_config.yaml").exists()
    assert (ckpt / "assets" / "normalizer_stats" / "action_stats.json").exists()
    assert (ckpt / "assets" / "normalizer_stats" / "proprio_stats.json").exists()
