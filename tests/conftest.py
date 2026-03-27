"""Shared fixtures for HybridVLA v2 test suite.

Mirrors OpenPI pattern: lightweight configs + fake data generators
that run on CPU in under a second.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
import torch
from torch import nn

# Ensure project root is importable
sys.path.insert(0, ".")

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

# Mini dimensions for fast CPU testing
D = 64  # d_model
D_EXP = 32  # expert d_model
H = 4  # chunk_horizon
A = 7  # action_dim
P = 9  # proprio_dim
T = 4  # sequence_window
B = 2  # batch_size
L = 16  # token seq len


def _mini_cfg(stage: str = "a") -> HybridVLAv2Config:
    """Build a tiny config that runs on CPU in under a second."""
    return HybridVLAv2Config(
        model=ModelConfig(
            backbone=BackboneConfig(name="mock"),
            multi_camera=MultiCameraConfig(enable=False),
            grounder=GrounderConfig(
                hidden_size=D,
                num_latents=12,
                num_object_slots=4,
                compressed_slots=2,
                num_layers=2,
                num_heads=2,
                mlp_ratio=2.0,
                hierarchical_compression=True,
                compression_layer=1,
            ),
            temporal_core=TemporalCoreConfig(
                d_model=D,
                fast_layers=2,
                medium_layers=1,
                slow_layers=1,
                fast_d_state=16,
                medium_d_state=16,
                slow_d_state=16,
                d_conv=4,
                expand=2,
                fusion_layers=1,
                fusion_heads=2,
                action_history_layers=1,
                action_history_d_state=8,
            ),
            action_expert=ActionExpertConfig(
                d_model=D_EXP,
                num_layers=18,  # FlowActionExpert asserts == 18
                pattern=["mamba", "mamba", "attn"] * 6,
                num_heads=2,
                d_state=8,
                d_conv=4,
                expand=2,
                chunk_horizon=H,
                cond_tokens=8,
                cond_dim=D,
                action_dim=A,
            ),
            heads=HeadsConfig(fast_vocab_size=32),
            ema=EMAConfig(enable=False),
            world_model=WorldModelConfig(enable=False),
            proprio_dim=P,
        ),
        train=TrainConfig(
            sequence_window=T,
            per_device_batch_size=B,
            semantic_refresh_stride=2,
            medium_update_stride=1,
        ),
        stage=stage,
    )


class _MockBackbone(nn.Module):
    """Lightweight backbone that mimics Qwen2VLBackboneWrapper outputs."""

    def __init__(self, output_dim: int = D) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.multi_scale_adapter = nn.Linear(output_dim, output_dim)
        # Dummy LoRA parameter so freeze/unfreeze gate can find it
        self.lora_dummy = nn.Parameter(torch.zeros(1))

    def forward_semantic(self, input_ids, attention_mask, **kwargs):
        B, L_seq = input_ids.shape
        return {"last_hidden_state": torch.randn(B, L_seq, self.output_dim)}

    def named_parameters(self, prefix="", recurse=True):
        for name, p in super().named_parameters(prefix=prefix, recurse=recurse):
            # Inject "lora" into the dummy param name so configure_trainable_modules works
            if "lora_dummy" in name:
                yield name, p
            else:
                yield name, p


class _MockBackboneWrapper:
    """Mock class with from_config classmethod to replace Qwen2VLBackboneWrapper."""

    _output_dim: int = D

    @classmethod
    def from_config(cls, backbone_cfg):
        return _MockBackbone(cls._output_dim)


def _build_model(cfg: HybridVLAv2Config):
    """Build model with mock backbone."""
    _MockBackboneWrapper._output_dim = cfg.model.grounder.hidden_size
    with patch(
        "vla_hybrid_v2.models.hybrid_vla_v2.Qwen2VLBackboneWrapper",
        _MockBackboneWrapper,
    ):
        from vla_hybrid_v2.models.hybrid_vla_v2 import HybridVLAv2

        return HybridVLAv2(cfg)


@pytest.fixture
def mini_config():
    """Minimal config for CPU testing."""
    return _mini_cfg("a")


@pytest.fixture
def dummy_batch():
    """Minimal valid batch dict for forward_train."""
    return {
        "actions": torch.randn(B, T, H, A),
        "proprio": torch.randn(B, T, P),
        "prev_actions": torch.randn(B, T, A),
        "input_ids": torch.randint(0, 1000, (B, L)),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
    }
