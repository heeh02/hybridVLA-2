"""Checkpoint save / load / auto-resume for HybridVLA v2.

Supports both FSDP and non-FSDP models.  Saves model, optimizer,
scheduler, EMA, and metadata with atomic writes.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from vla_hybrid_v2.utils.distributed import is_main_process

logger = logging.getLogger(__name__)


def _get_state_dict(model: nn.Module) -> Dict[str, Any]:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        if isinstance(model, FSDP):
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                return model.state_dict()
    except ImportError:
        pass
    return model.state_dict()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: str | Path,
    epoch: int = 0,
    scheduler: Optional[Any] = None,
    ema: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    asset_paths: Optional[Dict[str, str | Path]] = None,
) -> Optional[Path]:
    """Save training checkpoint (rank 0 only)."""
    output_dir = Path(output_dir)
    model_state = _get_state_dict(model)

    if not is_main_process():
        if dist.is_initialized():
            dist.barrier()
        return None

    ckpt_dir = output_dir / f"checkpoint-{step}"
    tmp_dir = output_dir / f".tmp-checkpoint-{step}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        torch.save(model_state, tmp_dir / "model.pt")
        torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")
        if ema is not None:
            torch.save(ema.state_dict(), tmp_dir / "ema.pt")

        meta = {"step": step, "epoch": epoch, **(extra or {})}
        with open(tmp_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if asset_paths:
            assets_dir = tmp_dir / "assets"
            assets_dir.mkdir(exist_ok=True)
            for name, src in asset_paths.items():
                src_path = Path(src)
                if not src_path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint asset '{name}' does not exist: {src_path}"
                    )
                dst = assets_dir / name
                if src_path.is_dir():
                    shutil.copytree(src_path, dst)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst)

        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        tmp_dir.rename(ckpt_dir)

        latest = output_dir / "checkpoint-latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(ckpt_dir.name)

        logger.info("Checkpoint saved: %s", ckpt_dir)
    except Exception as e:
        logger.error("Save failed: %s", e)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    if dist.is_initialized():
        dist.barrier()
    return ckpt_dir


def load_checkpoint(
    ckpt_dir: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    ema: Optional[Any] = None,
    map_location: str = "cpu",
    strict: bool = False,
) -> Dict[str, Any]:
    """Load checkpoint and return metadata."""
    ckpt_dir = Path(ckpt_dir)
    if ckpt_dir.is_symlink():
        ckpt_dir = ckpt_dir.resolve()

    state = torch.load(ckpt_dir / "model.pt", map_location=map_location, weights_only=True)

    # v0.10.10: filter out keys whose shapes don't match the current model
    # (e.g. ActionHistoryEncoder was resized from d=2048 to d=256).
    # This prevents RuntimeError on shape mismatch when strict=False.
    model_state = model.state_dict() if not hasattr(model, 'module') else model.module.state_dict()
    shape_mismatched = []
    for k in list(state.keys()):
        if k in model_state and state[k].shape != model_state[k].shape:
            shape_mismatched.append(
                f"  {k}: ckpt {list(state[k].shape)} vs model {list(model_state[k].shape)}"
            )
            del state[k]
    if shape_mismatched:
        logger.warning(
            "Dropped %d keys with shape mismatch (likely pre-v0.10.10 "
            "checkpoint — ActionHistoryEncoder was resized):\n%s",
            len(shape_mismatched), "\n".join(shape_mismatched[:10]),
        )

    # N1 fix: FSDP models need state_dict_type context to correctly load
    # a FULL_STATE_DICT checkpoint.  Without this, FSDP's default LOCAL
    # state_dict type causes key mismatch and silent param loss.
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )
        if isinstance(model, FSDP):
            fsdp_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_cfg):
                missing, unexpected = model.load_state_dict(state, strict=strict)
        else:
            missing, unexpected = model.load_state_dict(state, strict=strict)
    except ImportError:
        missing, unexpected = model.load_state_dict(state, strict=strict)

    if missing:
        logger.warning("Missing %d keys: %s...", len(missing), missing[:3])
    if unexpected:
        logger.warning("Unexpected %d keys: %s...", len(unexpected), unexpected[:3])

    if optimizer and (ckpt_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(
            torch.load(ckpt_dir / "optimizer.pt", map_location=map_location, weights_only=True)
        )
    if scheduler and (ckpt_dir / "scheduler.pt").exists():
        scheduler.load_state_dict(
            torch.load(ckpt_dir / "scheduler.pt", map_location=map_location, weights_only=True)
        )
    if ema and (ckpt_dir / "ema.pt").exists():
        ema.load_state_dict(
            torch.load(ckpt_dir / "ema.pt", map_location=map_location, weights_only=True)
        )

    meta = {}
    if (ckpt_dir / "meta.json").exists():
        with open(ckpt_dir / "meta.json") as f:
            meta = json.load(f)
    return meta


def find_latest_checkpoint(output_dir: str | Path) -> Optional[Path]:
    output_dir = Path(output_dir)
    latest = output_dir / "checkpoint-latest"
    if latest.exists():
        resolved = latest.resolve() if latest.is_symlink() else latest
        if (resolved / "meta.json").exists():
            return resolved
    return None


def auto_resume(
    output_dir: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    ema: Optional[Any] = None,
    map_location: str = "cpu",
) -> Tuple[int, int]:
    """Resume from latest checkpoint if it exists. Returns (step, epoch)."""
    ckpt = find_latest_checkpoint(output_dir)
    if ckpt is None:
        return 0, 0
    meta = load_checkpoint(ckpt, model, optimizer, scheduler, ema, map_location)
    logger.info("Resumed from %s (step=%d)", ckpt, meta.get("step", 0))
    return meta.get("step", 0), meta.get("epoch", 0)
