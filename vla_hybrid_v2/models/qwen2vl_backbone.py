"""Qwen2-VL-7B backbone wrapper with multi-scale features for v2.

v2 changes from v1:
- Qwen2-VL-7B-Instruct (3584d, 28 layers)
- Multi-scale feature extraction from layers [10, 18, 28]
- Multi-camera support: processes each camera independently
- Output projected from 3584d → 2048d
- LoRA rank=64 on all 28 layers (vs rank=32 on last 8 in v1)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from vla_hybrid_v2.config import BackboneConfig

logger = logging.getLogger(__name__)


class MultiScaleAdapter(nn.Module):
    """Fuses features from multiple backbone layers via learned projection.

    Inspired by FPN — extracts early (spatial detail), mid (intermediate),
    and late (semantic) features, projects each to output_dim, then sums.
    """

    def __init__(self, backbone_dim: int = 3584, output_dim: int = 2048,
                 num_scales: int = 3) -> None:
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, output_dim))
            for _ in range(num_scales)
        ])
        self.gate = nn.Sequential(
            nn.Linear(output_dim * num_scales, num_scales),
            nn.Softmax(dim=-1),
        )
        self.output_dim = output_dim

    def forward(self, multi_scale_features: List[Tensor]) -> Tensor:
        """multi_scale_features: list of [B, N, backbone_dim] from different layers."""
        projected = [proj(feat) for proj, feat in zip(self.projections, multi_scale_features)]
        stacked = torch.stack(projected, dim=-1)  # [B, N, output_dim, num_scales]
        # Learned per-scale gating: pool spatial dim → compute weights → weighted sum
        gate_input = torch.cat(
            [p.mean(dim=1) for p in projected], dim=-1,
        )  # [B, output_dim * num_scales]
        weights = self.gate(gate_input)  # [B, num_scales]
        return (stacked * weights[:, None, None, :]).sum(dim=-1)  # [B, N, output_dim]


class Qwen2VLBackboneWrapper(nn.Module):
    """Wraps Qwen2-VL-7B as feature extractor with multi-scale + multi-camera.

    v2: 7B model, LoRA on all layers, multi-scale extraction, 2048d output.
    """

    DTYPE_MAP = {
        "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32,
    }
    IMAGE_TOKEN_ID: int = 151655
    VIDEO_TOKEN_ID: int = 151656

    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct",
                 max_pixels=401408, min_pixels=200704,
                 lora_cfg=None, freeze_vision_tower=True,
                 freeze_text_layers_until=16,
                 attn_implementation="flash_attention_2",
                 torch_dtype="bfloat16",
                 multi_scale_layers=None, output_dim=2048) -> None:
        super().__init__()
        self.model_name = model_name
        self._dtype = self.DTYPE_MAP.get(torch_dtype, torch.bfloat16)
        self.multi_scale_layers = multi_scale_layers or [10, 18, 28]
        self.output_dim = output_dim

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self._dtype,
            attn_implementation=attn_implementation,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels,
        )

        if hasattr(self.processor, "tokenizer"):
            tok = self.processor.tokenizer
            for name, attr in [("image_token_id", "IMAGE_TOKEN_ID"),
                               ("video_token_id", "VIDEO_TOKEN_ID")]:
                if hasattr(tok, name):
                    setattr(self, attr, getattr(tok, name))

        self._apply_freeze(freeze_vision_tower, freeze_text_layers_until)

        if lora_cfg and lora_cfg.get("enable", False):
            self._apply_lora(lora_cfg, freeze_text_layers_until)

        self.hidden_size: int = self.model.config.hidden_size  # 3584 for 7B

        # Multi-scale adapter
        self.multi_scale_adapter = MultiScaleAdapter(
            self.hidden_size, output_dim, len(self.multi_scale_layers),
        )

    def _apply_freeze(self, freeze_vision, freeze_until):
        # v0.7: Navigate through ForConditionalGeneration -> Model -> visual
        # In HF transformers, Qwen2VLForConditionalGeneration.model is
        # Qwen2VLModel, which holds .visual (not ForConditionalGeneration).
        composite_model = getattr(self.model, "model", self.model)
        if freeze_vision and hasattr(composite_model, "visual"):
            for p in composite_model.visual.parameters():
                p.requires_grad = False

        # v0.7: Handle both old layout (model.model.layers) and new layout
        # (model.model.language_model.layers) for text layer freezing.
        text_model = composite_model
        if hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        if hasattr(text_model, "layers"):
            for i, layer in enumerate(text_model.layers):
                if i < freeze_until:
                    for p in layer.parameters():
                        p.requires_grad = False
        if hasattr(text_model, "embed_tokens"):
            for p in text_model.embed_tokens.parameters():
                p.requires_grad = False
        elif hasattr(composite_model, "embed_tokens"):
            for p in composite_model.embed_tokens.parameters():
                p.requires_grad = False

    def _apply_lora(self, lora_cfg, freeze_until):
        from peft import LoraConfig, get_peft_model
        target_modules = lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ])
        # v0.7: Navigate to layers via language_model if needed
        text_model = getattr(self.model, "model", self.model)
        if hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        total_layers = len(text_model.layers) if hasattr(text_model, "layers") else 28
        # v2: LoRA on ALL layers (not just unfrozen ones)
        lora_layer_indices = list(range(total_layers))
        peft_config = LoraConfig(
            r=lora_cfg.get("rank", 64),
            lora_alpha=lora_cfg.get("alpha", 128),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=target_modules,
            layers_to_transform=lora_layer_indices,
            bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info("LoRA injected — trainable: %d / %d (%.2f%%)",
                     trainable, total, 100.0 * trainable / total)

    @classmethod
    def from_config(cls, cfg: BackboneConfig) -> "Qwen2VLBackboneWrapper":
        lora_dict = {
            "enable": cfg.lora.enable, "rank": cfg.lora.rank,
            "alpha": cfg.lora.alpha, "dropout": cfg.lora.dropout,
            "target_modules": cfg.lora.target_modules,
        }
        return cls(
            model_name=cfg.name, max_pixels=cfg.max_pixels,
            min_pixels=cfg.min_pixels, lora_cfg=lora_dict,
            freeze_vision_tower=cfg.freeze_vision_tower,
            freeze_text_layers_until=cfg.freeze_text_layers_until,
            attn_implementation=cfg.attn_implementation,
            torch_dtype=cfg.torch_dtype,
            multi_scale_layers=cfg.multi_scale_layers,
            output_dim=cfg.output_dim,
        )

    def forward_semantic(self, input_ids, attention_mask,
                         pixel_values=None, image_grid_thw=None) -> Dict[str, Tensor]:
        model_kwargs = dict(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True, use_cache=False,
        )
        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values.to(dtype=self._dtype)
        if image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = image_grid_thw

        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(**model_kwargs)

        # Multi-scale feature extraction
        all_hidden = outputs.hidden_states
        multi_scale = []
        for layer_idx in self.multi_scale_layers:
            idx = min(layer_idx, len(all_hidden) - 1)
            multi_scale.append(all_hidden[idx])

        # Fuse multi-scale features → [B, N, output_dim]
        fused = self.multi_scale_adapter(multi_scale)

        vision_mask = (input_ids == self.IMAGE_TOKEN_ID) | (input_ids == self.VIDEO_TOKEN_ID)
        text_mask = attention_mask.bool() & ~vision_mask

        return {
            "last_hidden_state": fused,  # [B, N, 2048] (projected)
            "hidden_states": list(outputs.hidden_states),
            "vision_mask": vision_mask,
            "text_mask": text_mask,
        }
