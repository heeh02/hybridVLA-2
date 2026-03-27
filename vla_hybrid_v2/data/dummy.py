"""Dummy dataset for smoke testing HybridVLA v2.

Generates random tensors matching the batch protocol. Useful for
verifying model forward/backward without real data.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset

from vla_hybrid_v2.config import HybridVLAv2Config


class DummyVLADataset(Dataset):
    """Random-tensor dataset for smoke tests and debugging.

    Generates all required batch fields with correct shapes.
    Values are random (not meaningful for training).

    Note: Extends Dataset directly (not BaseDatasetAdapter) because dummy
    data does not require normalizers, episode files, or split logic.
    """

    def __init__(
        self,
        size: int = 10_000,
        cfg: Optional[HybridVLAv2Config] = None,
    ) -> None:
        self.size = size
        self.H = cfg.model.action_expert.chunk_horizon if cfg else 24
        self.A = cfg.model.action_expert.action_dim if cfg else 14
        self.P = cfg.model.proprio_dim if cfg else 14
        self.T = cfg.train.sequence_window if cfg else 24
        self.L = 128
        self.num_phases = cfg.model.heads.num_phases if cfg else 16
        self.num_affordance_types = cfg.model.heads.num_affordance_types if cfg else 8

        # Action range for value clamping
        if cfg:
            lo, hi = cfg.model.heads.action_range
        else:
            lo, hi = -1.0, 1.0
        self.lo, self.hi = lo, hi

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        """Return a batch-compatible dict with random tensors."""
        # Clamp actions/proprio to action_range for realistic values
        actions = torch.rand(self.T, self.H, self.A) * (self.hi - self.lo) + self.lo
        proprio = torch.rand(self.T, self.P) * (self.hi - self.lo) + self.lo
        prev_actions = torch.rand(self.T, self.A) * (self.hi - self.lo) + self.lo

        return {
            "input_ids": torch.randint(0, 32000, (self.L,)),
            "attention_mask": torch.ones(self.L, dtype=torch.long),
            "actions": actions,
            "proprio": proprio,
            "prev_actions": prev_actions,
            "phase_labels": torch.randint(0, self.num_phases, (self.T,)),
            "affordance_labels": torch.randint(0, self.num_affordance_types, (self.T,)),
            "embodiment_id": torch.tensor(0, dtype=torch.long),
        }
