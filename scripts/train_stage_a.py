"""Stage A training script for HybridVLA v2.

Trains: backbone LoRA + Grounder + Tri-Rate Core + discrete heads.
Frozen: Action Expert.

Features: cosine LR schedule with warmup, gradient accumulation,
FSDP support, EMA (optional), checkpoint save/load, auto-resume.

Usage:
    # Single GPU:
    python -m scripts.train_stage_a --config configs/train/stage_a.yaml

    # Multi-GPU:
    torchrun --nproc_per_node=8 -m scripts.train_stage_a \\
        --config configs/train/stage_a.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from vla_hybrid_v2.config import HybridVLAv2Config, load_config
from vla_hybrid_v2.data import build_dataset
from vla_hybrid_v2.utils.checkpointing import auto_resume, save_checkpoint
from vla_hybrid_v2.utils.distributed import (
    cleanup_distributed,
    clip_grad_norm_fsdp,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
    wrap_fsdp,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Optional[str] = None) -> None:
    handlers = [logging.StreamHandler()]
    if log_dir and is_main_process():
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / "train.log"))
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARNING,
        format=f"%(asctime)s [R{get_rank()}] %(name)s: %(message)s",
        datefmt="%H:%M:%S", handlers=handlers, force=True,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: HybridVLAv2Config) -> None:
    local_rank = setup_distributed(seed=42)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    setup_logging(cfg.train.output_dir)

    logger.info("Stage A training — v2 Tri-Rate + Hierarchical Grounder")
    logger.info("World size: %d, local rank: %d, device: %s",
                get_world_size(), local_rank, device)

    # ---- Model ----
    from vla_hybrid_v2.models.hybrid_vla_v2 import HybridVLAv2
    model = HybridVLAv2(cfg)

    # Freeze action expert in Stage A
    for p in model.action_expert.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Params: %s trainable / %s total (%.1f%%)",
                f"{trainable:,}", f"{total:,}", 100.0 * trainable / max(total, 1))

    model = model.to(device)

    # ---- FSDP ----
    if cfg.train.fsdp and get_world_size() > 1:
        model = wrap_fsdp(model, mixed_precision=cfg.train.bf16,
                          use_activation_checkpointing=cfg.train.checkpointing)

    # ---- Optimizer (v0.9.1: exclude res_scale/bias/LN from weight decay) ----
    no_decay_keywords = {"bias", "res_scale", "LayerNorm.weight", "layer_norm.weight"}
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    logger.info("Optimizer: %d decay params, %d no-decay params",
                len(decay_params), len(no_decay_params))
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.train.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.train.learning_rate, betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=cfg.train.warmup_steps,
        total_steps=cfg.train.max_steps,
    )

    # ---- EMA ----
    ema = None
    if cfg.model.ema.enable:
        from vla_hybrid_v2.utils.ema import EMAModel
        ema = EMAModel(
            model, initial_decay=cfg.model.ema.initial_decay,
            final_decay=cfg.model.ema.final_decay,
            ramp_steps=cfg.model.ema.ramp_steps,
        )

    # ---- Cross-stage checkpoint loading (v0.9) ----
    if cfg.train.resume_from:
        from pathlib import Path as _Path
        from vla_hybrid_v2.utils.checkpointing import load_checkpoint
        _resume_path = _Path(cfg.train.resume_from)
        if _resume_path.is_symlink():
            _resume_path = _resume_path.resolve()
        if not (_resume_path / "model.pt").exists():
            raise FileNotFoundError(
                f"Cross-stage checkpoint not found: {cfg.train.resume_from}\n"
                f"Resolved path: {_resume_path}\n"
                f"Ensure the prior stage completed and saved a checkpoint."
            )
        logger.info("Loading cross-stage checkpoint: %s", _resume_path)
        load_checkpoint(_resume_path, model, strict=False)
        # Do NOT load optimizer/scheduler from prior stage — they have
        # different total_steps and LR configs.

    # ---- Auto-resume (same-stage) ----
    start_step, start_epoch = auto_resume(
        cfg.train.output_dir, model, optimizer, scheduler, ema,
        map_location=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu",
    )

    # ---- Data (v0.9.3: uses data module) ----
    dataset, collate_fn = build_dataset(cfg, split="train")
    logger.info("Dataset: %s (%d samples)", type(dataset).__name__, len(dataset))
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        if get_world_size() > 1 else None
    )
    loader = DataLoader(
        dataset, batch_size=cfg.train.per_device_batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=2, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )

    # ---- Training ----
    model.train()
    global_step = start_step
    optimizer.zero_grad(set_to_none=True)
    step_start = time.monotonic()
    accum_loss: Dict[str, float] = {}
    grad_accum = cfg.train.grad_accum_steps

    for epoch in range(start_epoch, 9999):
        if sampler is not None:
            sampler.set_epoch(epoch)

        def _to_device(v):
            if isinstance(v, torch.Tensor):
                return v.to(device, non_blocking=True)
            if isinstance(v, list):
                return [_to_device(x) for x in v]
            return v

        for batch_idx, batch in enumerate(loader):
            if global_step >= cfg.train.max_steps:
                break

            batch = {k: _to_device(v) for k, v in batch.items()}

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.train.bf16):
                losses = model.forward_train(batch)

            loss = losses["loss_total"] / grad_accum
            loss.backward()

            for k, v in losses.items():
                accum_loss[k] = accum_loss.get(k, 0.0) + v.detach().item()

            if (batch_idx + 1) % grad_accum == 0:
                grad_norm = clip_grad_norm_fsdp(model, cfg.train.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model, global_step)
                global_step += 1

                # Logging
                if is_main_process() and global_step % cfg.train.log_interval == 0:
                    elapsed = time.monotonic() - step_start
                    sps = cfg.train.log_interval / max(elapsed, 1e-6)
                    lr = scheduler.get_last_lr()[0]
                    avg = {k: v / cfg.train.log_interval for k, v in accum_loss.items()}
                    parts = " | ".join(f"{k}: {v:.4f}" for k, v in avg.items())
                    logger.info("Step %d | %s | gnorm: %.3f | lr: %.2e | %.1f sps",
                                global_step, parts, grad_norm.item(), lr, sps)
                    accum_loss.clear()
                    step_start = time.monotonic()

                # Checkpoint
                if global_step % cfg.train.save_interval == 0:
                    save_checkpoint(model, optimizer, global_step,
                                    cfg.train.output_dir, epoch=epoch,
                                    scheduler=scheduler, ema=ema,
                                    extra={"stage": "a"})

        if global_step >= cfg.train.max_steps:
            break

    save_checkpoint(model, optimizer, global_step, cfg.train.output_dir,
                    epoch=epoch, scheduler=scheduler, ema=ema,
                    extra={"stage": "a", "final": True})
    logger.info("Stage A complete at step %d.", global_step)
    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridVLA v2 Stage A")
    parser.add_argument("--config", type=str, default="configs/train/stage_a.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
