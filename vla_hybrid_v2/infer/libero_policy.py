"""Unified LIBERO inference policy for HybridVLA v2.

Centralises:
- checkpoint/config discovery
- normalizer asset loading
- observation -> model input conversion
- model-space <-> env-space action/proprio transforms
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from vla_hybrid_v2.config import HybridVLAv2Config, load_config
from vla_hybrid_v2.data.normalizer import ActionNormalizer, ProprioNormalizer
from vla_hybrid_v2.types import ControlStepOutput, GrounderOutput, RuntimeCache
from vla_hybrid_v2.utils.checkpointing import load_checkpoint

logger = logging.getLogger(__name__)

_HDF5_PROPRIO_TO_ENV = {
    "joint_states": "robot0_joint_pos",
    "gripper_states": "robot0_gripper_qpos",
}


def resolve_checkpoint_dir(checkpoint_path: str | Path) -> Path:
    """Resolve a checkpoint directory, following checkpoint-latest symlinks."""
    ckpt = Path(checkpoint_path)
    return ckpt.resolve() if ckpt.is_symlink() else ckpt


def find_resolved_config(checkpoint_path: str | Path) -> Optional[Path]:
    """Locate resolved_config.yaml for a checkpoint.

    Prefers checkpoint-local copied assets, then falls back to the stage output dir.
    """
    ckpt = resolve_checkpoint_dir(checkpoint_path)
    asset_candidate = ckpt / "assets" / "resolved_config.yaml"
    if asset_candidate.exists():
        return asset_candidate

    for parent in (ckpt.parent, ckpt.parent.parent):
        candidate = parent / "resolved_config.yaml"
        if candidate.exists():
            return candidate
    return None


def resolve_policy_config(
    checkpoint_path: str | Path,
    config_path: Optional[str | Path],
) -> Tuple[HybridVLAv2Config, Optional[Path]]:
    """Resolve and validate the config used for inference."""
    resolved = find_resolved_config(checkpoint_path)

    if config_path is not None:
        cfg = load_config(config_path)
        if resolved is not None:
            resolved_cfg = load_config(resolved)
            if resolved_cfg.model.multi_camera.enable != cfg.model.multi_camera.enable:
                raise RuntimeError(
                    f"Config mismatch: --config has multi_camera.enable="
                    f"{cfg.model.multi_camera.enable} but the checkpoint was "
                    f"trained with multi_camera.enable="
                    f"{resolved_cfg.model.multi_camera.enable} "
                    f"(from {resolved}).\n"
                    f"Use the resolved config or omit --config to auto-discover it."
                )
            if resolved_cfg.model.proprio_dim != cfg.model.proprio_dim:
                logger.warning(
                    "Config mismatch: --config proprio_dim=%d but resolved=%d. "
                    "Using --config value. This may cause shape errors.",
                    cfg.model.proprio_dim, resolved_cfg.model.proprio_dim,
                )
            if (cfg.data.normalizer_stats_dir
                    and resolved_cfg.data.normalizer_stats_dir
                    and cfg.data.normalizer_stats_dir != resolved_cfg.data.normalizer_stats_dir):
                logger.warning(
                    "Config mismatch: normalizer_stats_dir differs. "
                    "--config=%s, resolved=%s. Using --config value.",
                    cfg.data.normalizer_stats_dir,
                    resolved_cfg.data.normalizer_stats_dir,
                )
        return cfg, resolved

    if resolved is not None:
        logger.info("Auto-discovered config: %s", resolved)
        return load_config(resolved), resolved

    raise FileNotFoundError(
        f"No --config provided and no resolved_config.yaml found near "
        f"{checkpoint_path}. Either pass --config explicitly or ensure "
        f"the checkpoint was created by train_libero.py / train_unified.py "
        f"(which save resolved_config.yaml)."
    )


def _candidate_stats_dirs(
    cfg: HybridVLAv2Config,
    checkpoint_path: str | Path,
) -> List[Path]:
    ckpt = resolve_checkpoint_dir(checkpoint_path)
    candidates: List[Path] = []

    if cfg.data.normalizer_stats_dir:
        candidates.append(Path(cfg.data.normalizer_stats_dir).expanduser())

    candidates.extend([
        ckpt / "assets" / "normalizer_stats",
        ckpt.parent / "normalizer_stats",
        ckpt.parent.parent / "normalizer_stats",
    ])

    resolved_cfg = find_resolved_config(checkpoint_path)
    if resolved_cfg is not None:
        candidates.append(resolved_cfg.parent / "normalizer_stats")

    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def load_policy_normalizers(
    cfg: HybridVLAv2Config,
    checkpoint_path: str | Path,
) -> Tuple[ActionNormalizer, ProprioNormalizer]:
    """Load action/proprio normalizers for inference."""
    action_norm = ActionNormalizer(target_range=cfg.model.heads.action_range)
    proprio_norm = ProprioNormalizer(target_range=cfg.model.proprio_range)

    searched: List[str] = []
    for stats_dir in _candidate_stats_dirs(cfg, checkpoint_path):
        action_stats = stats_dir / "action_stats.json"
        proprio_stats = stats_dir / "proprio_stats.json"
        searched.append(str(stats_dir))
        if action_stats.exists() and proprio_stats.exists():
            action_norm.load(action_stats)
            proprio_norm.load(proprio_stats)
            logger.info("Loaded inference normalizers from %s", stats_dir)
            return action_norm, proprio_norm

    raise FileNotFoundError(
        "Normalizer stats not found for inference. Looked in:\n- "
        + "\n- ".join(searched)
    )


def _make_pil_image(img_np):
    """Convert HWC uint8 numpy array to 448x448 RGB PIL Image."""
    from PIL import Image

    if img_np is None or img_np.ndim != 3:
        return None
    img = Image.fromarray(img_np.astype(np.uint8))
    return img.resize((448, 448), Image.BILINEAR).convert("RGB")


@dataclass
class LiberoPolicyRuntime:
    """Runtime state for one closed-loop LIBERO rollout."""

    runtime_cache: RuntimeCache
    prev_action_model: torch.Tensor


@dataclass
class LiberoPolicyStepOutput:
    """Policy output in both model space and environment space."""

    action_env: torch.Tensor
    action_model: torch.Tensor
    control: ControlStepOutput


class HybridVLALiberoPolicy:
    """Inference wrapper that aligns LIBERO rollout with training-time transforms."""

    def __init__(
        self,
        model,
        cfg: HybridVLAv2Config,
        processor,
        action_normalizer: ActionNormalizer,
        proprio_normalizer: ProprioNormalizer,
        device: str | torch.device = "cuda:0",
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.processor = processor
        self.action_normalizer = action_normalizer
        self.proprio_normalizer = proprio_normalizer
        self.device = torch.device(device)
        self.action_dim = cfg.model.action_expert.action_dim
        self.proprio_dim = cfg.model.proprio_dim
        self.multi_camera = cfg.model.multi_camera.enable
        self.num_cameras = cfg.model.multi_camera.num_cameras if self.multi_camera else 1
        self.proprio_keys: List[str] = (
            cfg.data.proprio_keys if cfg.data.proprio_keys else [cfg.data.proprio_key]
        )

        if self.multi_camera and self.num_cameras != 2:
            raise NotImplementedError(
                "LIBERO inference currently supports exactly 2 cameras "
                "(agentview + eye_in_hand)."
            )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config_path: Optional[str | Path] = None,
        device: str = "cuda:0",
    ) -> "HybridVLALiberoPolicy":
        """Build a ready-to-run LIBERO policy from a training checkpoint."""
        from transformers import AutoProcessor

        from vla_hybrid_v2.models.hybrid_vla_v2 import HybridVLAv2

        cfg, _resolved = resolve_policy_config(checkpoint_path, config_path)
        model = HybridVLAv2(cfg)
        load_checkpoint(checkpoint_path, model, strict=False)

        # Prefer EMA weights for inference (typically 5-15% better)
        ckpt_dir = resolve_checkpoint_dir(checkpoint_path)
        ema_path = ckpt_dir / "ema.pt"
        if ema_path.exists():
            ema_state = torch.load(ema_path, map_location="cpu", weights_only=True)
            shadow = ema_state["shadow"]
            applied = 0
            for name, param in model.named_parameters():
                if name in shadow:
                    param.data.copy_(shadow[name])
                    applied += 1
            logger.info("Applied EMA weights from %s (%d params)", ema_path, applied)

        model = model.to(device).eval()

        action_norm, proprio_norm = load_policy_normalizers(cfg, checkpoint_path)

        try:
            processor = AutoProcessor.from_pretrained(cfg.model.backbone.name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load processor for {cfg.model.backbone.name}. "
                "Closed-loop rollout must use the same tokenizer/vision processor "
                "as training; refusing to fall back to zero tokens."
            ) from exc

        return cls(
            model=model,
            cfg=cfg,
            processor=processor,
            action_normalizer=action_norm,
            proprio_normalizer=proprio_norm,
            device=device,
        )

    def init_runtime(self, batch_size: int = 1) -> LiberoPolicyRuntime:
        """Create runtime state for one or more rollout environments."""
        return LiberoPolicyRuntime(
            runtime_cache=self.model.init_runtime(batch_size=batch_size, device=str(self.device)),
            prev_action_model=torch.zeros(
                batch_size,
                self.action_dim,
                device=self.device,
            ),
        )

    def obs_to_semantic_input(self, obs_single: dict, language: str) -> dict:
        """Convert one env observation into semantic_step inputs."""
        if self.processor is None:
            raise RuntimeError(
                "Processor is required for inference. Refusing to substitute zero tokens."
            )

        agentview = _make_pil_image(obs_single.get("agentview_image"))
        eye_in_hand = _make_pil_image(obs_single.get("robot0_eye_in_hand_image"))

        if agentview is None:
            raise RuntimeError(
                "LIBERO observation is missing agentview_image. "
                "Check the rollout env rendering configuration."
            )

        if self.multi_camera:
            if eye_in_hand is None:
                raise RuntimeError(
                    "multi_camera.enable=True but robot0_eye_in_hand_image is missing "
                    "from env observation. Check that the LIBERO env is configured to "
                    "render eye_in_hand camera."
                )
            content = [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": language},
            ]
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            tok = self.processor(
                text=[text],
                images=[agentview, eye_in_hand],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.cfg.data.max_text_length,
            )
        else:
            tok = self.processor(
                text=language,
                images=agentview,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.cfg.data.max_text_length,
            )

        result = {
            "input_ids": tok["input_ids"].to(self.device),
            "attention_mask": tok["attention_mask"].to(self.device),
        }
        if "pixel_values" in tok:
            result["pixel_values"] = tok["pixel_values"].to(self.device)
        if "image_grid_thw" in tok:
            result["image_grid_thw"] = tok["image_grid_thw"].to(self.device)
        return result

    def semantic_step_from_obs(
        self,
        obs_single: dict,
        language: str,
    ) -> GrounderOutput:
        """Run semantic_step() from raw LIBERO observations."""
        sem_input = self.obs_to_semantic_input(obs_single, language)
        return self.model.semantic_step(
            input_ids=sem_input["input_ids"],
            attention_mask=sem_input["attention_mask"],
            pixel_values=sem_input.get("pixel_values"),
            image_grid_thw=sem_input.get("image_grid_thw"),
            num_cameras=self.num_cameras,
        )

    def obs_to_raw_proprio(self, obs_single: dict) -> torch.Tensor:
        """Extract raw proprio features from one env observation."""
        missing = []
        parts = []
        for proprio_key in self.proprio_keys:
            env_key = _HDF5_PROPRIO_TO_ENV.get(proprio_key, proprio_key)
            if env_key not in obs_single:
                missing.append(env_key)
                continue
            parts.append(np.asarray(obs_single[env_key], dtype=np.float32))

        if missing:
            raise KeyError(
                "LIBERO observation is missing proprio keys required by the config: "
                + ", ".join(missing)
            )

        proprio = np.concatenate(parts)
        if proprio.shape[0] != self.proprio_dim:
            raise RuntimeError(
                f"Proprio dim mismatch: obs produced {proprio.shape[0]} dims but "
                f"cfg.model.proprio_dim={self.proprio_dim}"
            )
        return torch.from_numpy(proprio).unsqueeze(0).to(self.device)

    def control_step_from_obs(
        self,
        obs_single: dict,
        runtime_state: LiberoPolicyRuntime,
        semantic_summary: GrounderOutput,
        embodiment_id: Optional[torch.Tensor] = None,
        num_sample_steps: int = 8,
    ) -> LiberoPolicyStepOutput:
        """Run control_step() in training-aligned normalization space."""
        if semantic_summary is None:
            raise ValueError("semantic_summary must be computed before control_step.")

        proprio_raw = self.obs_to_raw_proprio(obs_single)
        proprio_model = self.proprio_normalizer.normalize(proprio_raw)

        control = self.model.control_step(
            proprio=proprio_model,
            prev_action=runtime_state.prev_action_model,
            semantic_summary=semantic_summary,
            runtime_state=runtime_state.runtime_cache,
            embodiment_id=embodiment_id,
            num_sample_steps=num_sample_steps,
        )

        action_model = control.action.detach()
        action_env = self.action_normalizer.denormalize(action_model)
        lo, hi = self.cfg.model.heads.action_range
        action_env = action_env.clamp(lo, hi)
        runtime_state.prev_action_model = action_model

        return LiberoPolicyStepOutput(
            action_env=action_env,
            action_model=action_model,
            control=control,
        )
