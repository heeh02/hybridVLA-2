"""Unified training script for HybridVLA v2 — Stage A / B / C.

v0.10.3: replaces stage-specific scripts with a single entry point.

Stage semantics:
- A: backbone LoRA + grounder + tri-rate core + discrete heads. Expert frozen.
- B: adds expert (cond_prefix.detach()). EMA starts.
- C: full fine-tune with RTC/FASTER.

Usage:
    # Single GPU, Stage A:
    python -m scripts.train_unified --config configs/train/stage_a.yaml

    # Multi-GPU, Stage B from Stage A checkpoint:
    torchrun --nproc_per_node=8 -m scripts.train_unified \
        --config configs/train/stage_b.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
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
# Stage gate: explicit module freeze/unfreeze per training stage (P0-1a)
# ---------------------------------------------------------------------------

def configure_trainable_modules(
    model: torch.nn.Module, stage: str, cfg: HybridVLAv2Config,
) -> None:
    """Explicitly set requires_grad per module based on training stage.

    Replaces the old implicit approach that relied on PyTorch defaults.
    Called before FSDP wrapping and optimizer creation.
    """
    # Step 1: freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Step 2: re-enable backbone LoRA (trainable in all stages)
    for name, p in model.backbone.named_parameters():
        if "lora" in name.lower():
            p.requires_grad = True

    # Step 3: re-enable backbone multi-scale adapter
    if hasattr(model.backbone, "multi_scale_adapter"):
        for p in model.backbone.multi_scale_adapter.parameters():
            p.requires_grad = True

    # Step 4: modules trainable in ALL stages (A/B/C)
    # M2: include loss modules in case they gain learnable params later
    always_trainable = [
        model.grounder,
        model.temporal_core,
        model.action_history_encoder,
        model.proprio_proj,
        model.prev_action_proj,
        model.embodiment_embedding,
        model.fast_head,
        model.phase_head,
        model.affordance_head,
        model.consistency_loss,
        model.flow_matching_loss,
        model.discrete_loss,
        model.phase_loss,
    ]
    for mod in always_trainable:
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True

    # Step 5: Stage B/C — unfreeze expert + bridging projections
    if stage in ("b", "c"):
        extra_modules = [
            model.action_expert,
            model.cond_builder,
            model.core_to_expert,
            model.proprio_to_expert,
            model.emb_to_expert,
        ]
        for mod in extra_modules:
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True

    # Step 6: Stage C — unfreeze backbone text layers 16-27
    if stage == "c":
        freeze_until = cfg.model.backbone.freeze_text_layers_until  # 16
        for name, p in model.backbone.named_parameters():
            for layer_idx in range(freeze_until, 28):
                if f"layers.{layer_idx}." in name:
                    p.requires_grad = True
                    break

    logger.info("Stage %s: configured trainable modules (explicit gate).", stage.upper())


def sanity_check_trainable_params(
    model: torch.nn.Module, stage: str,
) -> None:
    """Assert trainable parameters match stage expectations (P0-1b)."""
    module_entries = [
        ("backbone", model.backbone),
        ("grounder", model.grounder),
        ("temporal_core", model.temporal_core),
        ("action_history_encoder", model.action_history_encoder),
        ("action_expert", model.action_expert),
        ("fast_head", model.fast_head),
        ("phase_head", model.phase_head),
        ("affordance_head", model.affordance_head),
        ("proprio_proj", model.proprio_proj),
        ("prev_action_proj", model.prev_action_proj),
        ("embodiment_embedding", model.embodiment_embedding),
        ("cond_builder", model.cond_builder),
        ("core_to_expert", model.core_to_expert),
        ("consistency_loss", model.consistency_loss),
    ]
    logger.info("Per-module trainable parameters:")
    for name, mod in module_entries:
        if mod is None:
            logger.info("  %-30s (absent)", name)
            continue
        total = sum(p.numel() for p in mod.parameters())
        train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        pct = 100.0 * train / max(total, 1)
        logger.info("  %-30s trainable=%12s  frozen=%12s  (%.1f%%)",
                     name, f"{train:,}", f"{total - train:,}", pct)

    # --- Stage-specific assertions ---
    expert_trainable = sum(
        p.numel() for p in model.action_expert.parameters() if p.requires_grad
    )
    expert_total = sum(p.numel() for p in model.action_expert.parameters())

    if stage == "a":
        assert expert_trainable == 0, (
            f"Stage A: action_expert should be frozen but has "
            f"{expert_trainable:,} trainable params"
        )
        cond_trainable = sum(
            p.numel() for p in model.cond_builder.parameters()
            if p.requires_grad
        )
        assert cond_trainable == 0, (
            f"Stage A: cond_builder should be frozen but has "
            f"{cond_trainable:,} trainable params"
        )
    elif stage in ("b", "c"):
        assert expert_trainable == expert_total, (
            f"Stage {stage.upper()}: action_expert should be fully trainable "
            f"but {expert_trainable:,}/{expert_total:,} are trainable"
        )

    # Backbone LoRA must always be trainable
    lora_trainable = sum(
        p.numel() for n, p in model.backbone.named_parameters()
        if "lora" in n.lower() and p.requires_grad
    )
    lora_total = sum(
        p.numel() for n, p in model.backbone.named_parameters()
        if "lora" in n.lower()
    )
    if lora_total > 0:
        assert lora_trainable == lora_total, (
            f"Backbone LoRA should be fully trainable but "
            f"{lora_trainable:,}/{lora_total:,} are trainable"
        )

    logger.info("Sanity check passed for Stage %s.", stage.upper())


# ---------------------------------------------------------------------------
# Per-module gradient norm (V5)
# ---------------------------------------------------------------------------

_GRAD_MODULES = [
    "backbone", "grounder", "temporal_core", "action_history_encoder",
    "action_expert", "fast_head", "phase_head", "affordance_head",
    "cond_builder",
]


def _log_per_module_grad_norm(model: torch.nn.Module) -> None:
    """Log L2 gradient norm per major sub-module."""
    parts = []
    for mod_name in _GRAD_MODULES:
        mod = getattr(model, mod_name, None)
        if mod is None:
            continue
        sq_sum = 0.0
        count = 0
        for p in mod.parameters():
            if p.requires_grad and p.grad is not None:
                sq_sum += p.grad.detach().norm(2).item() ** 2
                count += 1
        if count > 0:
            parts.append(f"{mod_name}={sq_sum ** 0.5:.3f}")
    if parts:
        logger.info("  grad_norm: %s", " | ".join(parts))


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cfg: HybridVLAv2Config,
    max_batches: int = 50,
) -> Dict[str, float]:
    """Run offline evaluation and return aggregated metrics.

    Computes average loss components over up to `max_batches` validation
    batches. Returns a dict of metric_name -> value.
    """
    model.eval()
    accum: Dict[str, float] = {}
    count = 0

    def _to_device(v):
        if isinstance(v, torch.Tensor):
            return v.to(device, non_blocking=True)
        if isinstance(v, list):
            return [_to_device(x) for x in v]
        return v

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        batch = {k: _to_device(v) for k, v in batch.items()}

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.train.bf16):
            losses = model.forward_train(batch)

        for k, v in losses.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        count += 1

    model.train()
    if count == 0:
        return {}
    return {k: v / count for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: HybridVLAv2Config) -> None:
    local_rank = setup_distributed(seed=42)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    setup_logging(cfg.train.output_dir)
    stage = cfg.stage

    logger.info("Unified training — Stage %s | v2 Tri-Rate + Hierarchical Grounder", stage.upper())
    logger.info("World size: %d, local rank: %d, device: %s",
                get_world_size(), local_rank, device)

    # ---- Model ----
    from vla_hybrid_v2.models.hybrid_vla_v2 import HybridVLAv2
    model = HybridVLAv2(cfg)

    # P0-1: Explicit stage gate (replaces old implicit if/elif/else)
    configure_trainable_modules(model, stage, cfg)
    sanity_check_trainable_params(model, stage)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Params: %s trainable / %s total (%.1f%%)",
                f"{trainable:,}", f"{total:,}", 100.0 * trainable / max(total, 1))

    model = model.to(device)

    # ---- FSDP ----
    if cfg.train.fsdp and get_world_size() > 1:
        model = wrap_fsdp(model, mixed_precision=cfg.train.bf16,
                          use_activation_checkpointing=cfg.train.checkpointing)

    # ---- Optimizer: per-module LR + weight decay groups (V4) ----
    base_lr = cfg.train.learning_rate
    no_decay_keywords = {"bias", "res_scale", "LayerNorm.weight", "layer_norm.weight"}

    # Classify each param into (module_group, decay/no_decay)
    param_groups_map: Dict[str, Dict[str, list]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Determine module group and LR scale
        if name.startswith("backbone"):
            group = "backbone"
            lr_scale = cfg.train.backbone_lr_scale
        elif name.startswith("action_expert"):
            group = "expert"
            lr_scale = cfg.train.expert_lr_scale
        else:
            group = "core"
            lr_scale = 1.0
        is_no_decay = any(nd in name for nd in no_decay_keywords)
        key = f"{group}_{'nodecay' if is_no_decay else 'decay'}"
        if key not in param_groups_map:
            param_groups_map[key] = {"params": [], "lr": base_lr * lr_scale,
                                     "weight_decay": 0.0 if is_no_decay else cfg.train.weight_decay}
        param_groups_map[key]["params"].append(param)

    param_groups = list(param_groups_map.values())
    for key, pg in param_groups_map.items():
        logger.info("  optim group %-20s  %4d params  lr=%.2e  wd=%.4f",
                     key, len(pg["params"]), pg["lr"], pg["weight_decay"])
    optimizer = torch.optim.AdamW(
        param_groups, lr=base_lr, betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=cfg.train.warmup_steps,
        total_steps=cfg.train.max_steps,
    )

    # ---- EMA (typically enabled from Stage B) ----
    ema = None
    if cfg.model.ema.enable:
        from vla_hybrid_v2.utils.ema import EMAModel
        ema = EMAModel(
            model, initial_decay=cfg.model.ema.initial_decay,
            final_decay=cfg.model.ema.final_decay,
            ramp_steps=cfg.model.ema.ramp_steps,
        )
        logger.info("EMA enabled (decay %.4f → %.4f over %d steps)",
                     cfg.model.ema.initial_decay, cfg.model.ema.final_decay,
                     cfg.model.ema.ramp_steps)

    # ---- Cross-stage checkpoint loading ----
    if cfg.train.resume_from:
        from vla_hybrid_v2.utils.checkpointing import load_checkpoint
        _resume_path = Path(cfg.train.resume_from)
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

    # ---- Auto-resume (same-stage) ----
    start_step, start_epoch = auto_resume(
        cfg.train.output_dir, model, optimizer, scheduler, ema,
        map_location=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu",
    )

    # ---- Processor (v0.10.3 P0-A) ----
    processor = None
    if cfg.data.format and cfg.data.format != "dummy":
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(cfg.model.backbone.name)
        logger.info("Loaded processor: %s", cfg.model.backbone.name)

    # ---- Data ----
    dataset, collate_fn = build_dataset(cfg, split="train", processor=processor)
    logger.info("Train dataset: %s (%d samples)", type(dataset).__name__, len(dataset))
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

    # ---- Validation DataLoader (v0.10.3 P1-E) ----
    val_loader = None
    try:
        val_dataset, val_collate_fn = build_dataset(cfg, split="val", processor=processor)
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset, batch_size=cfg.train.per_device_batch_size,
                shuffle=False, num_workers=1, pin_memory=True,
                drop_last=False, collate_fn=val_collate_fn,
            )
            logger.info("Val dataset: %s (%d samples)", type(val_dataset).__name__, len(val_dataset))
    except (FileNotFoundError, ValueError):
        logger.info("No validation dataset found — eval disabled.")

    # ---- Training ----
    model.train()
    global_step = start_step
    optimizer.zero_grad(set_to_none=True)
    step_start = time.monotonic()
    accum_loss: Dict[str, float] = {}
    grad_accum = cfg.train.grad_accum_steps

    def _to_device(v):
        if isinstance(v, torch.Tensor):
            return v.to(device, non_blocking=True)
        if isinstance(v, list):
            return [_to_device(x) for x in v]
        return v

    for epoch in range(start_epoch, 9999):
        if sampler is not None:
            sampler.set_epoch(epoch)

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
                    # V5: per-module gradient norm (every 5× log_interval to limit overhead)
                    if global_step % (cfg.train.log_interval * 5) == 0:
                        _log_per_module_grad_norm(model)
                    accum_loss.clear()
                    step_start = time.monotonic()

                # Eval (v0.10.3 P1-E)
                if (val_loader is not None
                        and global_step % cfg.train.eval_interval == 0
                        and is_main_process()):
                    metrics = evaluate(model, val_loader, device, cfg)
                    parts = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                    logger.info("Eval step %d | %s", global_step, parts)

                # Checkpoint
                if global_step % cfg.train.save_interval == 0:
                    save_checkpoint(model, optimizer, global_step,
                                    cfg.train.output_dir, epoch=epoch,
                                    scheduler=scheduler, ema=ema,
                                    extra={"stage": stage})

        if global_step >= cfg.train.max_steps:
            break

    save_checkpoint(model, optimizer, global_step, cfg.train.output_dir,
                    epoch=epoch, scheduler=scheduler, ema=ema,
                    extra={"stage": stage, "final": True})
    logger.info("Stage %s complete at step %d.", stage.upper(), global_step)
    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridVLA v2 Unified Training")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (must include 'stage: a|b|c')")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
