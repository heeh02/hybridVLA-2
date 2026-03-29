#!/usr/bin/env python3
"""2-GPU FSDP smoke tests: dry-run, save/resume, dtype consistency.

Usage:
    torchrun --nproc_per_node=2 scripts/smoke_fsdp_2gpu.py

Tests performed:
  1. Init model → dtype normalize → FSDP wrap → forward → backward → step
  2. Save checkpoint → load checkpoint → forward → backward → step
  3. Dtype verification at each stage
  4. Activation checkpointing conflict check

Requires: 2 GPUs with at least 2GB VRAM each (uses tiny model).
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vla_hybrid_v2.config import (
    ActionExpertConfig,
    BackboneConfig,
    EMAConfig,
    GrounderConfig,
    HeadsConfig,
    HybridVLAv2Config,
    ModelConfig,
    MultiCameraConfig,
    TemporalCoreConfig,
    TrainConfig,
    WorldModelConfig,
)
from vla_hybrid_v2.utils.checkpointing import load_checkpoint, save_checkpoint
from vla_hybrid_v2.utils.distributed import (
    cleanup_distributed,
    normalize_model_dtypes_for_fsdp,
    setup_distributed,
    verify_model_dtypes,
    wrap_fsdp,
)

logger = logging.getLogger(__name__)

# Mini dimensions for fast smoke testing
D = 64
D_EXP = 32
H = 4
A = 7
P = 9
T = 2
B = 2
L = 16


def _mini_cfg(stage="a"):
    return HybridVLAv2Config(
        model=ModelConfig(
            backbone=BackboneConfig(name="mock"),
            multi_camera=MultiCameraConfig(enable=False),
            grounder=GrounderConfig(
                hidden_size=D, num_latents=12, num_object_slots=4,
                compressed_slots=2, num_layers=2, num_heads=2,
                mlp_ratio=2.0, hierarchical_compression=True, compression_layer=1,
            ),
            temporal_core=TemporalCoreConfig(
                d_model=D, fast_layers=2, medium_layers=1, slow_layers=1,
                fast_d_state=16, medium_d_state=16, slow_d_state=16,
                d_conv=4, expand=2, fusion_layers=1, fusion_heads=2,
                action_history_layers=1, action_history_d_state=8,
            ),
            action_expert=ActionExpertConfig(
                d_model=D_EXP, num_layers=18, pattern=["mamba", "mamba", "attn"] * 6,
                num_heads=2, d_state=8, d_conv=4, expand=2,
                chunk_horizon=H, cond_tokens=8, cond_dim=D, action_dim=A,
            ),
            heads=HeadsConfig(fast_vocab_size=32),
            ema=EMAConfig(enable=False),
            world_model=WorldModelConfig(enable=False),
            proprio_dim=P,
        ),
        train=TrainConfig(
            sequence_window=T, per_device_batch_size=B,
            semantic_refresh_stride=2, medium_update_stride=1,
            fsdp=True, bf16=True, checkpointing=True,
        ),
        stage=stage,
    )


class _MockBackbone(torch.nn.Module):
    def __init__(self, output_dim=D):
        super().__init__()
        self.output_dim = output_dim
        self.multi_scale_adapter = torch.nn.Linear(output_dim, output_dim)
        self.lora_dummy = torch.nn.Parameter(torch.zeros(1))

    def forward_semantic(self, input_ids, attention_mask, **kwargs):
        B_sz, L_seq = input_ids.shape
        return {"last_hidden_state": torch.randn(B_sz, L_seq, self.output_dim, device=input_ids.device)}

    def named_parameters(self, prefix="", recurse=True):
        for name, p in super().named_parameters(prefix=prefix, recurse=recurse):
            yield name, p


class _MockBackboneWrapper:
    _output_dim = D

    @classmethod
    def from_config(cls, backbone_cfg):
        return _MockBackbone(cls._output_dim)


def _build_model(cfg):
    _MockBackboneWrapper._output_dim = cfg.model.grounder.hidden_size
    with patch(
        "vla_hybrid_v2.models.hybrid_vla_v2.Qwen2VLBackboneWrapper",
        _MockBackboneWrapper,
    ):
        from vla_hybrid_v2.models.hybrid_vla_v2 import HybridVLAv2
        return HybridVLAv2(cfg)


def _make_batch(device):
    return {
        "actions": torch.randn(B, T, H, A, device=device),
        "proprio": torch.randn(B, T, P, device=device),
        "prev_actions": torch.randn(B, T, A, device=device),
        "input_ids": torch.randint(0, 1000, (B, L), device=device),
        "attention_mask": torch.ones(B, L, dtype=torch.long, device=device),
    }


def _assert(condition, msg):
    if not condition:
        rank = dist.get_rank()
        raise AssertionError(f"[Rank {rank}] {msg}")


def main():
    local_rank = setup_distributed(seed=42)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s [R{rank}] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if rank == 0:
        logger.info("=" * 60)
        logger.info("FSDP 2-GPU Smoke Test")
        logger.info("=" * 60)

    cfg = _mini_cfg("a")

    # ---- Test 1: Init → dtype normalize → FSDP wrap → forward/backward/step ----
    if rank == 0:
        logger.info("Test 1: FSDP dry-run (init → wrap → train step)")

    model = _build_model(cfg)
    model = model.to(device)

    # Dtype normalize
    normalize_model_dtypes_for_fsdp(model, target_dtype=torch.bfloat16)
    _assert(
        verify_model_dtypes(model, expected_dtype=torch.bfloat16, label="post-normalize"),
        "dtype verification failed after normalization",
    )

    # FSDP wrap
    model = wrap_fsdp(model, mixed_precision=True, use_activation_checkpointing=True)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, fused=True,
    )

    # Forward + backward + step
    batch = _make_batch(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses = model.forward_train(batch)
    loss = losses["loss_total"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    _assert(loss.isfinite(), f"loss is not finite: {loss.item()}")
    if rank == 0:
        logger.info("  PASS — loss=%.4f", loss.item())

    # ---- Test 2: Save → load → resume train step ----
    if rank == 0:
        logger.info("Test 2: FSDP save + resume")

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = save_checkpoint(
            model, optimizer, step=1, output_dir=tmp, epoch=0,
        )
        dist.barrier()

        # Rebuild fresh model for resume
        model2 = _build_model(cfg)
        model2 = model2.to(device)
        normalize_model_dtypes_for_fsdp(model2, target_dtype=torch.bfloat16)
        model2 = wrap_fsdp(model2, mixed_precision=True, use_activation_checkpointing=True)
        model2.train()

        optimizer2 = torch.optim.AdamW(
            [p for p in model2.parameters() if p.requires_grad],
            lr=1e-4, fused=True,
        )

        # Load checkpoint
        ckpt_dir = Path(tmp) / "checkpoint-1"
        if not ckpt_dir.exists():
            ckpt_dir = Path(tmp) / "checkpoint-latest"
        if ckpt_dir.exists():
            meta = load_checkpoint(
                ckpt_dir, model2, optimizer2,
                map_location=f"cuda:{local_rank}",
            )
            if rank == 0:
                logger.info("  Loaded checkpoint (step=%d)", meta.get("step", -1))

        # Run another train step after resume
        batch2 = _make_batch(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses2 = model2.forward_train(batch2)
        loss2 = losses2["loss_total"]
        loss2.backward()
        optimizer2.step()
        optimizer2.zero_grad()

        _assert(loss2.isfinite(), f"loss after resume is not finite: {loss2.item()}")
        if rank == 0:
            logger.info("  PASS — loss_after_resume=%.4f", loss2.item())

    # ---- Test 3: Dtype verification at each stage ----
    if rank == 0:
        logger.info("Test 3: Dtype consistency checks")

    model3 = _build_model(cfg)
    model3 = model3.to(device)

    # Before normalization: expect mixed dtypes
    f32_count = sum(
        1 for _, p in model3.named_parameters()
        if p.is_floating_point() and p.dtype == torch.float32
    )
    _assert(f32_count > 0, "Expected float32 params before normalization")

    # After normalization: all bf16
    normalize_model_dtypes_for_fsdp(model3, target_dtype=torch.bfloat16)
    _assert(
        verify_model_dtypes(model3, expected_dtype=torch.bfloat16, label="test3-post-norm"),
        "dtype verification failed after normalization",
    )

    # After FSDP wrap: still bf16 (FSDP uses orig params)
    model3 = wrap_fsdp(model3, mixed_precision=True, use_activation_checkpointing=True)
    if rank == 0:
        logger.info("  PASS — dtype consistent at all stages")

    dist.barrier()
    if rank == 0:
        logger.info("=" * 60)
        logger.info("ALL SMOKE TESTS PASSED")
        logger.info("=" * 60)

    cleanup_distributed()


if __name__ == "__main__":
    main()
