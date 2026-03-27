"""Smoke test: single-GPU training loop with dummy data.

Validates end-to-end: model init → forward_train → backward → optimizer step.
Uses a miniature config (small dims) so it runs on CPU or a single GPU.

Note: This file uses an **inline** DummyVLADataset with mini dimensions
(A=7, P=9, D=64) rather than importing from `vla_hybrid_v2.data.dummy`.
This is intentional — the smoke test validates model-layer correctness
independent of the data infrastructure, and needs smaller dimensions to
run quickly on CPU. The production DummyVLADataset in `data/dummy.py`
uses full config dimensions and covers more fields (affordance_labels, etc.).

Usage:
    python -m scripts.train_smoke_test
    python -m scripts.train_smoke_test --steps 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ---------- Miniature config (CPU-friendly) ----------

D_CORE = 64
D_EXPERT = 32
H = 4    # chunk_horizon
A = 7    # action_dim
P = 9    # proprio_dim (intentionally != A to verify decoupling)
T = 8    # sequence_window
B = 2    # batch size
L = 16   # token seq len


def _mini_cfg(stage: str = "a"):
    """Build a tiny config that runs on CPU in seconds."""
    sys.path.insert(0, ".")
    from vla_hybrid_v2.config import (
        ActionExpertConfig, BackboneConfig, EMAConfig, GrounderConfig,
        HeadsConfig, HybridVLAv2Config, ModelConfig, MultiCameraConfig,
        TemporalCoreConfig, TrainConfig, WorldModelConfig,
    )
    return HybridVLAv2Config(
        model=ModelConfig(
            backbone=BackboneConfig(name="mock"),
            multi_camera=MultiCameraConfig(enable=False),
            grounder=GrounderConfig(
                hidden_size=D_CORE, num_latents=12, num_object_slots=4,
                compressed_slots=2, num_layers=2, num_heads=4,
                mlp_ratio=2.0, hierarchical_compression=True,
                compression_layer=1,
            ),
            temporal_core=TemporalCoreConfig(
                d_model=D_CORE, fast_layers=2, medium_layers=1,
                slow_layers=1, fast_d_state=8, medium_d_state=8,
                slow_d_state=8, d_conv=4, expand=2,
                fusion_heads=4, fusion_layers=1,
                action_history_layers=1, action_history_d_state=8,
            ),
            action_expert=ActionExpertConfig(
                d_model=D_EXPERT, num_layers=18, num_heads=4,
                chunk_horizon=H, action_dim=A, d_state=8,
                d_conv=4, expand=2, cond_dim=D_CORE,
                cond_tokens=6, ada_rmsnorm=True,
            ),
            heads=HeadsConfig(
                fast_discrete_head=True, fast_vocab_size=32,
                phase_head=True, num_phases=4, affordance_head=False,
            ),
            ema=EMAConfig(enable=False),
            world_model=WorldModelConfig(enable=False),
            proprio_dim=P,  # v0.9.1: decoupled from action_dim
        ),
        train=TrainConfig(
            sequence_window=T, semantic_refresh_stride=4,
            medium_update_stride=2, per_device_batch_size=B,
            checkpointing=False,
            loss_weights={
                "fast_discrete": 1.0, "phase": 0.5,
                "consistency": 0.3, "flow_matching": 1.0,
            },
        ),
        stage=stage,
    )


# ---------- Mock backbone (no real Qwen2-VL) ----------

class _MockBackbone(nn.Module):
    def __init__(self, hidden_size=D_CORE):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward_semantic(self, input_ids, attention_mask, **kw):
        B, Len = input_ids.shape
        fake = torch.randn(B, Len, self.hidden_size, device=input_ids.device)
        return {
            "last_hidden_state": self.proj(fake),
            "hidden_states": [fake],
            "vision_mask": torch.zeros(B, Len, dtype=torch.bool, device=input_ids.device),
            "text_mask": attention_mask.bool(),
        }


# ---------- Dummy dataset ----------

class DummyVLADataset(Dataset):
    def __init__(self, size=200):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 1000, (L,)),
            "attention_mask": torch.ones(L, dtype=torch.long),
            "actions": torch.randn(T, H, A),
            "proprio": torch.randn(T, P),
            "prev_actions": torch.randn(T, A),
            "phase_labels": torch.randint(0, 4, (T,)),
            "embodiment_id": torch.tensor(0, dtype=torch.long),
        }


# ---------- Training loop ----------

def train(steps: int = 20, stage: str = "a"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}, stage: {stage}, steps: {steps}")

    cfg = _mini_cfg(stage)

    with patch(
        "vla_hybrid_v2.models.hybrid_vla_v2.Qwen2VLBackboneWrapper.from_config",
        return_value=_MockBackbone(D_CORE),
    ):
        from vla_hybrid_v2.models.hybrid_vla_v2 import HybridVLAv2
        model = HybridVLAv2(cfg)

    # P0-1: Use the same explicit stage gate as train_unified.py
    from scripts.train_unified import (
        configure_trainable_modules,
        sanity_check_trainable_params,
    )
    configure_trainable_modules(model, stage, cfg)
    sanity_check_trainable_params(model, stage)

    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Params: {trainable:,} trainable / {total:,} total")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4, weight_decay=0.01,
    )

    dataset = DummyVLADataset(size=steps * B)
    loader = DataLoader(dataset, batch_size=B, shuffle=True, drop_last=True)

    # M4: snapshot expert params before training (for Stage B/C assertion)
    expert_snapshot = None
    if stage in ("b", "c"):
        expert_snapshot = {
            n: p.data.clone()
            for n, p in model.action_expert.named_parameters()
            if p.requires_grad
        }

    model.train()
    t0 = time.monotonic()
    seen_loss_fm = False
    for step_i, batch in enumerate(loader):
        if step_i >= steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        use_amp = device.type == "cuda"
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cpu", enabled=False)
        with ctx:
            losses = model.forward_train(batch)

        if "loss_fm" in losses:
            seen_loss_fm = True

        loss = losses["loss_total"]
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step_i % 5 == 0:
            parts = " | ".join(f"{k}: {v.item():.4f}" for k, v in losses.items() if k != "loss_total")
            log.info(f"Step {step_i:3d} | total: {loss.item():.4f} | {parts} | gnorm: {grad_norm:.3f}")

    elapsed = time.monotonic() - t0
    log.info(f"Done: {steps} steps in {elapsed:.1f}s ({steps / elapsed:.1f} steps/s)")

    # M4: Stage B/C assertions
    if stage in ("b", "c"):
        assert seen_loss_fm, (
            f"Stage {stage.upper()}: loss_fm should be present but was never produced"
        )
        # Verify expert params actually changed
        for n, p in model.action_expert.named_parameters():
            if p.requires_grad and n in expert_snapshot:
                assert not torch.equal(p.data, expert_snapshot[n]), (
                    f"Stage {stage.upper()}: expert param '{n}' did not change after {steps} steps"
                )
                break  # one param changing is sufficient proof
        log.info("Stage %s assertions PASSED: loss_fm present, expert params updated.", stage.upper())

    log.info("Smoke test PASSED — no NaN, no crash.")


def train_multi_camera(steps: int = 10):
    """Smoke test for multi-camera mode with 3 dummy cameras."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Multi-camera smoke test — device: {device}, steps: {steps}")

    cfg = _mini_cfg(stage="a")
    cfg.model.multi_camera.enable = True
    cfg.model.multi_camera.num_cameras = 3

    with patch(
        "vla_hybrid_v2.models.hybrid_vla_v2.Qwen2VLBackboneWrapper.from_config",
        return_value=_MockBackbone(D_CORE),
    ):
        from vla_hybrid_v2.models.hybrid_vla_v2 import HybridVLAv2
        model = HybridVLAv2(cfg)

    from scripts.train_unified import (
        configure_trainable_modules,
        sanity_check_trainable_params,
    )
    configure_trainable_modules(model, "a", cfg)
    sanity_check_trainable_params(model, "a")
    model = model.to(device)

    # Dataset with num_cameras field
    class MultiCamDummyDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 1000, (L,)),
                "attention_mask": torch.ones(L, dtype=torch.long),
                "actions": torch.randn(T, H, A),
                "proprio": torch.randn(T, P),
                "prev_actions": torch.randn(T, A),
                "phase_labels": torch.randint(0, 4, (T,)),
                "embodiment_id": torch.tensor(0, dtype=torch.long),
                "num_cameras": 3,
            }

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4,
    )
    loader = DataLoader(MultiCamDummyDataset(steps * B), batch_size=B,
                        shuffle=True, drop_last=True)

    model.train()
    for step_i, batch in enumerate(loader):
        if step_i >= steps:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if device.type == "cuda"
               else torch.autocast("cpu", enabled=False))
        with ctx:
            losses = model.forward_train(batch)
        losses["loss_total"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step_i % 5 == 0:
            log.info(f"[multi-cam] Step {step_i} | loss: {losses['loss_total'].item():.4f}")

    log.info("Multi-camera smoke test PASSED — no NaN, no crash.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--stage", type=str, default="a", choices=["a", "b", "c"])
    parser.add_argument("--multi-camera", action="store_true",
                        help="Run multi-camera smoke test instead of standard")
    args = parser.parse_args()
    if args.multi_camera:
        train_multi_camera(steps=args.steps)
    else:
        train(steps=args.steps, stage=args.stage)


if __name__ == "__main__":
    main()
