"""HDF5 dataset adapter for HybridVLA v2.

Reads episode data from HDF5 files in the common robotics format:
    episode.hdf5/
        data/
            actions:    [T_ep, A]
            proprio:    [T_ep, P]   (key from cfg.data.proprio_key)
            images/
                agentview_rgb: [T_ep, H, W, C]
                ...
        attrs/
            language_instruction: str

Slices episodes into fixed-length windows for training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from vla_hybrid_v2.config import HybridVLAv2Config
from vla_hybrid_v2.data.base_adapter import BaseDatasetAdapter
from vla_hybrid_v2.data.normalizer import Normalizer

logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]


class HDF5DatasetAdapter(BaseDatasetAdapter):
    """Reads HDF5 episodes and produces training windows.

    Each __getitem__ call returns a fixed-length window from a random
    position within an episode, with actions/proprio normalized.

    Vision processing (pixel_values) requires passing a Qwen2-VL
    processor. If no processor is provided, only text-mode samples
    are returned (suitable for Stage A without vision fine-tuning).
    """

    def __init__(
        self,
        cfg: HybridVLAv2Config,
        action_normalizer: Normalizer,
        proprio_normalizer: Normalizer,
        processor=None,
        split: str = "train",
    ) -> None:
        assert h5py is not None, "h5py required: pip install h5py"
        super().__init__(cfg, action_normalizer, proprio_normalizer, split)

        self.dcfg = cfg.data
        self.processor = processor

        self.window = cfg.train.sequence_window
        self.chunk_H = cfg.model.action_expert.chunk_horizon
        self.action_dim = cfg.model.action_expert.action_dim
        self.proprio_dim = cfg.model.proprio_dim

        # Discover episode files
        data_dir = Path(self.dcfg.data_dir) if self.dcfg.data_dir else None
        if self.dcfg.paths:
            self.episode_paths = [Path(p) for p in self.dcfg.paths]
        elif data_dir and data_dir.exists():
            self.episode_paths = sorted(data_dir.glob("*.hdf5"))
        else:
            raise FileNotFoundError(
                f"No data found. Set data.paths or data.data_dir in config. "
                f"data_dir={self.dcfg.data_dir}"
            )

        if self.dcfg.max_episodes:
            self.episode_paths = self.episode_paths[: self.dcfg.max_episodes]

        # Build index: (episode_idx, start_step) for each valid window
        self._episode_lengths: List[int] = []
        self._index: List[tuple] = []
        self._build_index()
        logger.info(
            "HDF5 adapter: %d episodes, %d windows (T=%d, split=%s)",
            len(self.episode_paths), len(self._index), self.window, split,
        )

    def _build_index(self) -> None:
        """Scan episodes and build (episode_idx, start) pairs."""
        for ep_idx, path in enumerate(self.episode_paths):
            with h5py.File(path, "r") as f:
                # D6: validate HDF5 structure before accessing keys
                if "data" not in f:
                    logger.warning("Episode %s missing 'data' group, skipping.", path)
                    continue
                data_grp = f["data"]
                if self.dcfg.action_key not in data_grp:
                    logger.warning(
                        "Episode %s missing key '%s'. Available: %s",
                        path, self.dcfg.action_key, list(data_grp.keys()),
                    )
                    continue
                T_ep = data_grp[self.dcfg.action_key].shape[0]
            self._episode_lengths.append(T_ep)
            # P0-3 + D4: require window + chunk_H - 1 steps so every
            # chunk within the window has H real future actions.
            min_len = self.window + self.chunk_H - 1
            if T_ep < min_len:
                logger.warning(
                    "Episode %s has %d steps < required %d "
                    "(window=%d + chunk_H=%d - 1), skipping.",
                    path, T_ep, min_len, self.window, self.chunk_H,
                )
                continue
            for start in range(0, T_ep - min_len + 1):
                self._index.append((ep_idx, start))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def episode_lengths(self) -> List[int]:
        return self._episode_lengths

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self._index[idx]
        path = self.episode_paths[ep_idx]
        T = self.window

        with h5py.File(path, "r") as f:
            data = f["data"]
            for required_key in (self.dcfg.action_key, self.dcfg.proprio_key):
                if required_key not in data:
                    raise KeyError(
                        f"Episode {path} missing key '{required_key}'. "
                        f"Available: {list(data.keys())}"
                    )
            # P0-3: read T + H - 1 action steps so every chunk within the
            # window has H real future actions (no padding degradation).
            action_end = start + T + self.chunk_H - 1
            raw_actions = data[self.dcfg.action_key][start:action_end]  # [T+H-1, A]
            raw_proprio = data[self.dcfg.proprio_key][start:start + T]  # [T, P]

            # Language instruction
            lang = self.dcfg.language  # default fallback
            if "attrs" in f and self.dcfg.language_key in f["attrs"]:
                lang = f["attrs"][self.dcfg.language_key][()]
                if isinstance(lang, bytes):
                    lang = lang.decode("utf-8")

        # Normalize
        raw_actions_t = torch.from_numpy(raw_actions.astype(np.float32))
        raw_proprio_t = torch.from_numpy(raw_proprio.astype(np.float32))
        norm_actions_ext = self.action_normalizer.normalize(raw_actions_t)  # [T+H-1, A]
        norm_proprio = self.proprio_normalizer.normalize(raw_proprio_t)

        # Build action chunks from extended buffer: each chunk has H real actions
        A = norm_actions_ext.shape[-1]
        action_chunks = torch.zeros(T, self.chunk_H, A)
        for t in range(T):
            action_chunks[t] = norm_actions_ext[t : t + self.chunk_H]

        # prev_actions from the first T action steps (shift by 1, first uses zeros)
        norm_actions = norm_actions_ext[:T]  # [T, A]
        prev_actions = torch.zeros_like(norm_actions)
        if T > 1:
            prev_actions[1:] = norm_actions[:-1]

        # Tokenize text (without vision for now)
        # If processor is available, use it; otherwise generate placeholder tokens
        if self.processor is not None:
            tok = self.processor(
                text=lang, return_tensors="pt", padding="max_length",
                truncation=True, max_length=128,
            )
            input_ids = tok["input_ids"].squeeze(0)
            attention_mask = tok["attention_mask"].squeeze(0)
        else:
            # Placeholder: will be overridden when processor is available
            input_ids = torch.zeros(128, dtype=torch.long)
            attention_mask = torch.ones(128, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "actions": action_chunks,
            "proprio": norm_proprio,
            "prev_actions": prev_actions,
            "embodiment_id": torch.tensor(
                self.dcfg.embodiment_id, dtype=torch.long
            ),
        }
