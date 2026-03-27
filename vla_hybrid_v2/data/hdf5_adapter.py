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
from PIL import Image

from vla_hybrid_v2.config import HybridVLAv2Config
from vla_hybrid_v2.data.base_adapter import BaseDatasetAdapter
from vla_hybrid_v2.data.normalizer import Normalizer
from vla_hybrid_v2.data.transforms import RobotImageAugmentation

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

        # v0.11: image augmentation (train split only)
        self.augmentation = (
            RobotImageAugmentation(cfg.data.augmentation)
            if split == "train" else None
        )

        self.window = cfg.train.sequence_window
        self.chunk_H = cfg.model.action_expert.chunk_horizon
        self.action_dim = cfg.model.action_expert.action_dim
        self.proprio_dim = cfg.model.proprio_dim
        self.refresh_stride = cfg.train.semantic_refresh_stride
        self.image_key = cfg.data.image_key  # e.g. "agentview_rgb"

        # Multi-camera settings
        self.multi_camera = cfg.model.multi_camera.enable
        self.camera_keys: List[str] = (
            cfg.data.camera_keys if self.multi_camera else [self.image_key]
        )
        self.camera_names: List[str] = cfg.model.multi_camera.camera_names
        self.max_text_length = (
            1024 if self.multi_camera else cfg.data.max_text_length
        )
        if self.multi_camera:
            logger.info(
                "Multi-camera enabled: %d cameras %s → HDF5 keys %s",
                len(self.camera_keys), self.camera_names, self.camera_keys,
            )

        # Discover episode files
        # V1 fix: val split uses separate dir or episode-ratio split
        if split == "val" and self.dcfg.val_data_dir:
            val_dir = Path(self.dcfg.val_data_dir)
            if val_dir.exists():
                self.episode_paths = sorted(val_dir.glob("*.hdf5"))
            else:
                raise FileNotFoundError(
                    f"val_data_dir does not exist: {self.dcfg.val_data_dir}"
                )
        else:
            data_dir = Path(self.dcfg.data_dir) if self.dcfg.data_dir else None
            if self.dcfg.paths:
                all_paths = [Path(p) for p in self.dcfg.paths]
            elif data_dir and data_dir.exists():
                all_paths = sorted(data_dir.glob("*.hdf5"))
            else:
                raise FileNotFoundError(
                    f"No data found. Set data.paths or data.data_dir in config. "
                    f"data_dir={self.dcfg.data_dir}"
                )
            # Episode-ratio split when no separate val_data_dir
            if split == "val" and not self.dcfg.val_data_dir:
                n_val = max(1, int(len(all_paths) * self.dcfg.val_ratio))
                # N3 fix: guard against tiny datasets where n_val >= total
                if n_val >= len(all_paths):
                    logger.warning(
                        "val_ratio=%.2f yields n_val=%d >= total %d episodes. "
                        "Val set will use all episodes (train/val overlap). "
                        "Consider using val_data_dir for proper split.",
                        self.dcfg.val_ratio, n_val, len(all_paths),
                    )
                self.episode_paths = all_paths[-n_val:]
            else:
                if self.dcfg.val_data_dir is None and split == "train":
                    n_val = max(1, int(len(all_paths) * self.dcfg.val_ratio))
                    if n_val >= len(all_paths):
                        logger.warning(
                            "val_ratio=%.2f yields n_val=%d >= total %d episodes. "
                            "Train set will use all episodes (train/val overlap).",
                            self.dcfg.val_ratio, n_val, len(all_paths),
                        )
                        self.episode_paths = all_paths
                    else:
                        self.episode_paths = all_paths[:-n_val]
                else:
                    self.episode_paths = all_paths

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

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _read_image(self, data_grp, image_key: str, frame_idx: int) -> Optional[Image.Image]:
        """Read a single RGB frame from HDF5 as PIL Image, with augmentation."""
        if "images" not in data_grp:
            return None
        images_grp = data_grp["images"]
        if image_key not in images_grp:
            return None
        raw = images_grp[image_key][frame_idx]  # [H, W, C] uint8
        img = Image.fromarray(raw)
        if self.augmentation is not None:
            img = self.augmentation(img)
        return img

    def _read_multi_camera_images(
        self, data_grp, camera_keys: List[str], frame_idx: int,
    ) -> List[Optional[Image.Image]]:
        """Read one RGB frame per camera from HDF5."""
        images: List[Optional[Image.Image]] = []
        for key in camera_keys:
            images.append(self._read_image(data_grp, key, frame_idx))
        return images

    # ------------------------------------------------------------------
    # Multi-camera tokenization
    # ------------------------------------------------------------------

    def _process_text_multi_image(
        self, lang: str, pil_images: List[Optional[Image.Image]],
    ) -> dict:
        """Tokenize text + multiple camera images via Qwen2-VL processor.

        Uses apply_chat_template for proper multi-image formatting.
        Falls back to single-image path when only one valid image exists.
        """
        if self.processor is None:
            return self._process_text_image(lang, None)

        _TARGET = (448, 448)
        valid_images: List[Image.Image] = []
        for img in pil_images:
            if img is not None:
                if img.size != _TARGET:
                    img = img.resize(_TARGET, Image.BILINEAR)
                img = img.convert("RGB")
                valid_images.append(img)

        if len(valid_images) == 0:
            return self._process_text_image(lang, None)
        if len(valid_images) == 1:
            return self._process_text_image(lang, valid_images[0])

        # Build multi-image conversation for Qwen2-VL processor
        content: list = []
        for _ in valid_images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": lang})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        tok = self.processor(
            text=[text], images=valid_images,
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_text_length,
        )
        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "pixel_values": tok["pixel_values"].squeeze(0)
                if "pixel_values" in tok else None,
            "image_grid_thw": tok["image_grid_thw"].squeeze(0)
                if "image_grid_thw" in tok else None,
            "num_cameras": len(valid_images),
        }

    def _process_text_image(
        self, lang: str, pil_image: Optional[Image.Image],
    ) -> dict:
        """Tokenize text (+ image when available) via processor.

        Returns dict with input_ids, attention_mask, and optionally
        pixel_values / image_grid_thw.
        """
        if self.processor is not None and pil_image is not None:
            # P0-3 fix: Force uniform image size so all samples produce the
            # same N_patches, preventing torch.stack crash in collate.
            # 448×448 = 200704 = min_pixels → deterministic patch count.
            _TARGET = (448, 448)
            if pil_image.size != _TARGET:
                pil_image = pil_image.resize(_TARGET, Image.BILINEAR)
            pil_image = pil_image.convert("RGB")
            # Joint text+image — Qwen2-VL processor handles image tokens
            tok = self.processor(
                text=lang, images=pil_image,
                return_tensors="pt", padding="max_length",
                truncation=True, max_length=256,
            )
            return {
                "input_ids": tok["input_ids"].squeeze(0),
                "attention_mask": tok["attention_mask"].squeeze(0),
                "pixel_values": tok["pixel_values"].squeeze(0),
                "image_grid_thw": tok["image_grid_thw"].squeeze(0),
            }
        elif self.processor is not None:
            # Text-only (no image in HDF5)
            tok = self.processor(
                text=lang, return_tensors="pt", padding="max_length",
                truncation=True, max_length=256,
            )
            return {
                "input_ids": tok["input_ids"].squeeze(0),
                "attention_mask": tok["attention_mask"].squeeze(0),
                "pixel_values": None,
                "image_grid_thw": None,
            }
        else:
            # No processor — placeholder tokens (dummy / early dev)
            return {
                "input_ids": torch.zeros(256, dtype=torch.long),
                "attention_mask": torch.ones(256, dtype=torch.long),
                "pixel_values": None,
                "image_grid_thw": None,
            }

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

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

            # ---- Read observation image(s) ----
            if self.multi_camera:
                primary_images = self._read_multi_camera_images(
                    data, self.camera_keys, start,
                )
            else:
                primary_images = None
                primary_image = self._read_image(data, self.image_key, start)

            # ---- Build refresh frames ----
            refresh_steps = list(range(0, T, self.refresh_stride))
            R = len(refresh_steps)
            if self.multi_camera:
                refresh_multi_images: List[List[Optional[Image.Image]]] = []
                for r_step in refresh_steps:
                    refresh_multi_images.append(
                        self._read_multi_camera_images(
                            data, self.camera_keys, start + r_step,
                        )
                    )
            else:
                refresh_images: List[Optional[Image.Image]] = []
                for r_step in refresh_steps:
                    refresh_images.append(
                        self._read_image(data, self.image_key, start + r_step)
                    )

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

        # ---- Primary observation tokenization ----
        if self.multi_camera:
            primary_tok = self._process_text_multi_image(lang, primary_images)
        else:
            primary_tok = self._process_text_image(lang, primary_image)

        sample = {
            "input_ids": primary_tok["input_ids"],
            "attention_mask": primary_tok["attention_mask"],
            "actions": action_chunks,
            "proprio": norm_proprio,
            "prev_actions": prev_actions,
            "embodiment_id": torch.tensor(
                self.dcfg.embodiment_id, dtype=torch.long
            ),
        }

        # Add primary vision fields if available
        if primary_tok.get("pixel_values") is not None:
            sample["pixel_values"] = primary_tok["pixel_values"]
            sample["image_grid_thw"] = primary_tok["image_grid_thw"]

        # Track number of cameras for downstream camera position embeddings
        if self.multi_camera:
            sample["num_cameras"] = primary_tok.get("num_cameras", len(self.camera_keys))
        else:
            sample["num_cameras"] = 1

        # ---- Refresh frame tokenization ----
        if self.multi_camera:
            has_any_image = any(
                any(img is not None for img in imgs)
                for imgs in refresh_multi_images
            )
            if self.processor is not None and R > 1 and has_any_image:
                refresh_input_ids_list = []
                refresh_attention_mask_list = []
                refresh_pv_list: List[Optional[torch.Tensor]] = []
                refresh_thw_list: List[Optional[torch.Tensor]] = []

                for imgs in refresh_multi_images:
                    tok = self._process_text_multi_image(lang, imgs)
                    refresh_input_ids_list.append(tok["input_ids"])
                    refresh_attention_mask_list.append(tok["attention_mask"])
                    refresh_pv_list.append(tok.get("pixel_values"))
                    refresh_thw_list.append(tok.get("image_grid_thw"))

                sample["refresh_input_ids"] = torch.stack(refresh_input_ids_list)
                sample["refresh_attention_mask"] = torch.stack(refresh_attention_mask_list)
                sample["refresh_pixel_values_list"] = refresh_pv_list
                sample["refresh_image_grid_thw_list"] = refresh_thw_list
        else:
            has_any_image = any(img is not None for img in refresh_images)
            if self.processor is not None and R > 1 and has_any_image:
                refresh_input_ids_list = []
                refresh_attention_mask_list = []
                refresh_pv_list_sc: List[Optional[torch.Tensor]] = []
                refresh_thw_list_sc: List[Optional[torch.Tensor]] = []

                for img in refresh_images:
                    tok = self._process_text_image(lang, img)
                    refresh_input_ids_list.append(tok["input_ids"])
                    refresh_attention_mask_list.append(tok["attention_mask"])
                    refresh_pv_list_sc.append(tok["pixel_values"])
                    refresh_thw_list_sc.append(tok["image_grid_thw"])

                sample["refresh_input_ids"] = torch.stack(refresh_input_ids_list)
                sample["refresh_attention_mask"] = torch.stack(refresh_attention_mask_list)
                sample["refresh_pixel_values_list"] = refresh_pv_list_sc
                sample["refresh_image_grid_thw_list"] = refresh_thw_list_sc

        return sample
