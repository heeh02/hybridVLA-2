"""LIBERO HDF5 dataset adapter.

Supports the official LIBERO / robomimic-style layout:

    <task_name>_demo.hdf5
      data/
        attrs:
          problem_info
        demo_0/
          actions
          obs/
            agentview_rgb
            eye_in_hand_rgb
            joint_states
            gripper_states
        demo_1/
          ...

Each task file contains many demonstrations, so train / val splitting is done
at the demonstration level instead of the file level.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DemoRef:
    path: Path
    demo_key: str
    length: int
    language: str


class LiberoHDF5DatasetAdapter(BaseDatasetAdapter):
    """Reads official LIBERO task HDF5 files and produces training windows."""

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

        # Image augmentation (train split only)
        self.augmentation = (
            RobotImageAugmentation(cfg.data.augmentation)
            if split == "train" else None
        )

        self.window = cfg.train.sequence_window
        self.chunk_H = cfg.model.action_expert.chunk_horizon
        self.action_dim = cfg.model.action_expert.action_dim
        self.proprio_dim = cfg.model.proprio_dim
        self.refresh_stride = cfg.train.semantic_refresh_stride
        self.image_key = cfg.data.image_key

        # Proprio: support concatenating multiple obs keys (e.g., joint_states + gripper_states)
        self.proprio_keys: List[str] = cfg.data.proprio_keys if cfg.data.proprio_keys else []

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
                "LIBERO multi-camera enabled: %d cameras %s -> obs keys %s",
                len(self.camera_keys), self.camera_names, self.camera_keys,
            )

        self.task_paths = self._discover_task_paths(split)
        self.demo_refs = self._collect_demo_refs(split)
        if self.dcfg.max_episodes:
            self.demo_refs = self.demo_refs[: self.dcfg.max_episodes]

        self._demo_lengths: List[int] = []
        self._index: List[tuple[int, int]] = []
        self._build_index()
        logger.info(
            "LIBERO adapter: %d task files, %d demos, %d windows (T=%d, split=%s)",
            len(self.task_paths), len(self.demo_refs), len(self._index), self.window, split,
        )

    def _discover_task_paths(self, split: str) -> List[Path]:
        if split == "val" and self.dcfg.val_data_dir:
            val_dir = Path(self.dcfg.val_data_dir)
            if not val_dir.exists():
                raise FileNotFoundError(
                    f"val_data_dir does not exist: {self.dcfg.val_data_dir}"
                )
            paths = sorted(val_dir.glob("*.hdf5"))
        else:
            data_dir = Path(self.dcfg.data_dir) if self.dcfg.data_dir else None
            if self.dcfg.paths:
                paths = [Path(p) for p in self.dcfg.paths]
            elif data_dir and data_dir.exists():
                paths = sorted(data_dir.glob("*.hdf5"))
            else:
                raise FileNotFoundError(
                    "No LIBERO data found. Set data.paths or data.data_dir in config. "
                    f"data_dir={self.dcfg.data_dir}"
                )

        if not paths:
            raise FileNotFoundError(
                f"No .hdf5 task files found for split={split}. "
                f"data_dir={self.dcfg.data_dir} val_data_dir={self.dcfg.val_data_dir}"
            )
        return paths

    @staticmethod
    def _sorted_demo_keys(data_grp) -> List[str]:
        demo_keys = [
            key for key in data_grp.keys()
            if key.startswith("demo_") and isinstance(data_grp[key], h5py.Group)
        ]

        def _demo_sort_key(name: str) -> tuple[int, str]:
            suffix = name.split("demo_", 1)[-1]
            return (int(suffix), name) if suffix.isdigit() else (10**9, name)

        return sorted(demo_keys, key=_demo_sort_key)

    def _extract_language(self, data_grp) -> str:
        lang = self.dcfg.language
        if "problem_info" not in data_grp.attrs:
            return lang

        raw = data_grp.attrs["problem_info"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            problem_info = json.loads(raw)
        except Exception:
            logger.warning("Failed to parse problem_info JSON; using fallback language.")
            return lang

        language = problem_info.get(self.dcfg.language_key)
        if isinstance(language, list):
            language = "".join(language)
        if isinstance(language, str) and language.strip():
            return language.strip().strip('"')
        return lang

    def _split_demo_keys(
        self, demo_keys: List[str], path: Path, split: str,
    ) -> List[str]:
        if self.dcfg.val_data_dir:
            return demo_keys

        if len(demo_keys) <= 1:
            if split == "val":
                logger.warning(
                    "Task file %s has only %d demo; val split overlaps train.",
                    path, len(demo_keys),
                )
            return demo_keys

        n_val = max(1, int(len(demo_keys) * self.dcfg.val_ratio))
        if n_val >= len(demo_keys):
            logger.warning(
                "val_ratio=%.2f yields n_val=%d >= total %d demos for %s. "
                "Train/val will overlap for this task file.",
                self.dcfg.val_ratio, n_val, len(demo_keys), path,
            )
            return demo_keys

        if split == "val":
            return demo_keys[-n_val:]
        return demo_keys[:-n_val]

    def _collect_demo_refs(self, split: str) -> List[DemoRef]:
        demo_refs: List[DemoRef] = []
        for path in self.task_paths:
            with h5py.File(path, "r") as f:
                if "data" not in f:
                    logger.warning("Task file %s missing 'data' group, skipping.", path)
                    continue
                data_grp = f["data"]
                demo_keys = self._split_demo_keys(
                    self._sorted_demo_keys(data_grp), path, split,
                )
                if not demo_keys:
                    logger.warning("Task file %s contains no demos for split=%s.", path, split)
                    continue

                lang = self._extract_language(data_grp)
                for demo_key in demo_keys:
                    demo_grp = data_grp[demo_key]
                    if self.dcfg.action_key not in demo_grp:
                        logger.warning(
                            "Demo %s in %s missing '%s', skipping.",
                            demo_key, path, self.dcfg.action_key,
                        )
                        continue
                    if "obs" not in demo_grp:
                        logger.warning("Demo %s in %s missing 'obs', skipping.", demo_key, path)
                        continue
                    obs_grp = demo_grp["obs"]
                    # Validate proprio keys
                    if self.proprio_keys:
                        missing = [k for k in self.proprio_keys if k not in obs_grp]
                        if missing:
                            logger.warning(
                                "Demo %s in %s missing proprio keys %s, skipping.",
                                demo_key, path, missing,
                            )
                            continue
                    elif self.dcfg.proprio_key not in obs_grp:
                        logger.warning(
                            "Demo %s in %s missing proprio key '%s', skipping.",
                            demo_key, path, self.dcfg.proprio_key,
                        )
                        continue
                    demo_refs.append(
                        DemoRef(
                            path=path,
                            demo_key=demo_key,
                            length=demo_grp[self.dcfg.action_key].shape[0],
                            language=lang,
                        )
                    )
        return demo_refs

    def _build_index(self) -> None:
        min_len = self.window + self.chunk_H - 1
        for demo_idx, ref in enumerate(self.demo_refs):
            self._demo_lengths.append(ref.length)
            if ref.length < min_len:
                logger.warning(
                    "Demo %s in %s has %d steps < required %d "
                    "(window=%d + chunk_H=%d - 1), skipping.",
                    ref.demo_key, ref.path, ref.length, min_len, self.window, self.chunk_H,
                )
                continue
            for start in range(0, ref.length - min_len + 1):
                self._index.append((demo_idx, start))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def episode_lengths(self) -> List[int]:
        return self._demo_lengths

    def _read_image(self, obs_grp, image_key: str, frame_idx: int) -> Optional[Image.Image]:
        if image_key not in obs_grp:
            return None
        raw = obs_grp[image_key][frame_idx]
        img = Image.fromarray(raw)
        if self.augmentation is not None:
            img = self.augmentation(img)
        return img

    def _read_multi_camera_images(
        self, obs_grp, camera_keys: List[str], frame_idx: int,
    ) -> List[Optional[Image.Image]]:
        images: List[Optional[Image.Image]] = []
        for key in camera_keys:
            images.append(self._read_image(obs_grp, key, frame_idx))
        return images

    def _process_text_multi_image(
        self, lang: str, pil_images: List[Optional[Image.Image]],
    ) -> dict:
        if self.processor is None:
            return self._process_text_image(lang, None)

        target = (448, 448)
        valid_images: List[Image.Image] = []
        for img in pil_images:
            if img is None:
                continue
            if img.size != target:
                img = img.resize(target, Image.BILINEAR)
            valid_images.append(img.convert("RGB"))

        if len(valid_images) == 0:
            return self._process_text_image(lang, None)
        if len(valid_images) == 1:
            return self._process_text_image(lang, valid_images[0])

        content: list = []
        for _ in valid_images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": lang})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        tok = self.processor(
            text=[text],
            images=valid_images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
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
        if self.processor is not None and pil_image is not None:
            target = (448, 448)
            if pil_image.size != target:
                pil_image = pil_image.resize(target, Image.BILINEAR)
            pil_image = pil_image.convert("RGB")
            tok = self.processor(
                text=lang,
                images=pil_image,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
            )
            return {
                "input_ids": tok["input_ids"].squeeze(0),
                "attention_mask": tok["attention_mask"].squeeze(0),
                "pixel_values": tok["pixel_values"].squeeze(0),
                "image_grid_thw": tok["image_grid_thw"].squeeze(0),
            }

        if self.processor is not None:
            tok = self.processor(
                text=lang,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
            )
            return {
                "input_ids": tok["input_ids"].squeeze(0),
                "attention_mask": tok["attention_mask"].squeeze(0),
                "pixel_values": None,
                "image_grid_thw": None,
            }

        return {
            "input_ids": torch.zeros(self.max_text_length, dtype=torch.long),
            "attention_mask": torch.ones(self.max_text_length, dtype=torch.long),
            "pixel_values": None,
            "image_grid_thw": None,
        }

    def __getitem__(self, idx: int) -> dict:
        demo_idx, start = self._index[idx]
        ref = self.demo_refs[demo_idx]
        T = self.window

        with h5py.File(ref.path, "r") as f:
            demo_grp = f["data"][ref.demo_key]
            obs_grp = demo_grp["obs"]
            if self.dcfg.action_key not in demo_grp:
                raise KeyError(
                    f"Demo {ref.demo_key} in {ref.path} missing key '{self.dcfg.action_key}'. "
                    f"Available: {list(demo_grp.keys())}"
                )
            # Read proprio: concat multiple keys or use single key
            if self.proprio_keys:
                for pk in self.proprio_keys:
                    if pk not in obs_grp:
                        raise KeyError(
                            f"Demo {ref.demo_key} in {ref.path} missing proprio key "
                            f"'{pk}'. Available: {list(obs_grp.keys())}"
                        )
            elif self.dcfg.proprio_key not in obs_grp:
                raise KeyError(
                    f"Demo {ref.demo_key} in {ref.path} missing proprio key "
                    f"'{self.dcfg.proprio_key}'. Available: {list(obs_grp.keys())}"
                )

            action_end = start + T + self.chunk_H - 1
            raw_actions = demo_grp[self.dcfg.action_key][start:action_end]
            if self.proprio_keys:
                parts = [obs_grp[pk][start:start + T] for pk in self.proprio_keys]
                raw_proprio = np.concatenate(parts, axis=-1)  # [T, sum(dims)]
            else:
                raw_proprio = obs_grp[self.dcfg.proprio_key][start:start + T]
            lang = ref.language or self.dcfg.language

            if self.multi_camera:
                primary_images = self._read_multi_camera_images(
                    obs_grp, self.camera_keys, start,
                )
            else:
                primary_images = None
                primary_image = self._read_image(obs_grp, self.image_key, start)

            refresh_steps = list(range(0, T, self.refresh_stride))
            R = len(refresh_steps)
            if self.multi_camera:
                refresh_multi_images: List[List[Optional[Image.Image]]] = []
                for r_step in refresh_steps:
                    refresh_multi_images.append(
                        self._read_multi_camera_images(
                            obs_grp, self.camera_keys, start + r_step,
                        )
                    )
            else:
                refresh_images: List[Optional[Image.Image]] = []
                for r_step in refresh_steps:
                    refresh_images.append(
                        self._read_image(obs_grp, self.image_key, start + r_step)
                    )

        raw_actions_t = torch.from_numpy(raw_actions.astype(np.float32))
        raw_proprio_t = torch.from_numpy(raw_proprio.astype(np.float32))
        norm_actions_ext = self.action_normalizer.normalize(raw_actions_t)
        norm_proprio = self.proprio_normalizer.normalize(raw_proprio_t)

        A = norm_actions_ext.shape[-1]
        action_chunks = torch.zeros(T, self.chunk_H, A)
        for t in range(T):
            action_chunks[t] = norm_actions_ext[t : t + self.chunk_H]

        norm_actions = norm_actions_ext[:T]
        prev_actions = torch.zeros_like(norm_actions)
        if T > 1:
            prev_actions[1:] = norm_actions[:-1]

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
                self.dcfg.embodiment_id, dtype=torch.long,
            ),
        }

        if primary_tok.get("pixel_values") is not None:
            sample["pixel_values"] = primary_tok["pixel_values"]
            sample["image_grid_thw"] = primary_tok["image_grid_thw"]

        sample["num_cameras"] = (
            primary_tok.get("num_cameras", len(self.camera_keys))
            if self.multi_camera else 1
        )

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
                refresh_pv_list: List[Optional[torch.Tensor]] = []
                refresh_thw_list: List[Optional[torch.Tensor]] = []
                for img in refresh_images:
                    tok = self._process_text_image(lang, img)
                    refresh_input_ids_list.append(tok["input_ids"])
                    refresh_attention_mask_list.append(tok["attention_mask"])
                    refresh_pv_list.append(tok["pixel_values"])
                    refresh_thw_list.append(tok["image_grid_thw"])
                sample["refresh_input_ids"] = torch.stack(refresh_input_ids_list)
                sample["refresh_attention_mask"] = torch.stack(refresh_attention_mask_list)
                sample["refresh_pixel_values_list"] = refresh_pv_list
                sample["refresh_image_grid_thw_list"] = refresh_thw_list

        return sample
