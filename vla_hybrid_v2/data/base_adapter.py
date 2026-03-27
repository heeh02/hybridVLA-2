"""Abstract base class for HybridVLA v2 dataset adapters.

Each adapter reads a specific data format (HDF5, RLDS, etc.) and
produces dicts conforming to the WindowSample protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from torch.utils.data import Dataset

from vla_hybrid_v2.config import HybridVLAv2Config
from vla_hybrid_v2.data.normalizer import Normalizer


class BaseDatasetAdapter(Dataset, ABC):
    """Abstract base for all VLA dataset adapters.

    Subclasses implement format-specific reading (HDF5, RLDS, etc.)
    and return dicts matching the WindowSample field contract.
    """

    def __init__(
        self,
        cfg: HybridVLAv2Config,
        action_normalizer: Normalizer,
        proprio_normalizer: Normalizer,
        split: str = "train",
    ) -> None:
        self.cfg = cfg
        self.action_normalizer = action_normalizer
        self.proprio_normalizer = proprio_normalizer
        self.split = split

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abstractmethod
    def episode_lengths(self) -> List[int]:
        """Length of each episode, used for window sampling."""
        ...
