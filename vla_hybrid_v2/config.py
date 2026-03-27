"""Configuration for HybridVLA v2.

v2 changes from v1:
- Backbone: Qwen2-VL-7B (3584d → project 2048d), LoRA rank=64, multi-scale features
- Grounder: 96 latents → hierarchical compression → 24 refined slots, 8 layers, 2048d
- Temporal Core: Tri-Rate (Fast 20L + Medium 6L + Slow 10L), d_state up to 256
- Action Expert: 18L 1536d M-M-A×6, AdaRMSNorm, midpoint ODE
- Multi-camera: 3 cameras native, EMA integration
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    enable: bool = True
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class BackboneConfig:
    name: str = "Qwen/Qwen2-VL-7B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    min_pixels: int = 200704
    max_pixels: int = 401408
    freeze_vision_tower: bool = True
    freeze_text_layers_until: int = 16
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    # v2: multi-scale feature extraction
    multi_scale_layers: List[int] = field(default_factory=lambda: [10, 18, 28])
    output_dim: int = 2048  # project from 3584 → 2048


@dataclass
class MultiCameraConfig:
    enable: bool = False  # NOT YET IMPLEMENTED — set True when multi-camera adapter is ready
    num_cameras: int = 3
    camera_names: List[str] = field(default_factory=lambda: [
        "wrist", "shoulder", "overhead",
    ])


@dataclass
class GrounderConfig:
    hidden_size: int = 2048
    num_latents: int = 96
    num_object_slots: int = 48
    compressed_slots: int = 24
    num_layers: int = 8
    num_heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    hierarchical_compression: bool = True
    compression_layer: int = 4  # compress after this layer


@dataclass
class TemporalCoreConfig:
    d_model: int = 2048
    # Tri-Rate streams
    fast_layers: int = 20
    medium_layers: int = 6
    slow_layers: int = 10
    fast_d_state: int = 128
    medium_d_state: int = 128
    slow_d_state: int = 256
    d_conv: int = 4
    expand: int = 2
    # Fusion
    fusion_type: str = "cross_attention"  # "cross_attention" or "gate"
    fusion_heads: int = 8
    fusion_layers: int = 2
    # Action history
    action_history_len: int = 8
    action_history_layers: int = 4
    action_history_d_state: int = 64


@dataclass
class ActionExpertConfig:
    d_model: int = 1536
    num_layers: int = 18
    pattern: List[str] = field(default_factory=lambda: [
        "mamba", "mamba", "attn",
    ] * 6)  # M-M-A × 6 = 18 layers
    num_heads: int = 24
    d_state: int = 96
    d_conv: int = 4
    expand: int = 2
    chunk_horizon: int = 24
    cond_tokens: int = 32
    cond_dim: int = 2048
    action_dim: int = 14
    dropout: float = 0.0
    # v2 enhancements
    ada_rmsnorm: bool = True
    ode_solver: str = "midpoint"  # "euler" or "midpoint"


@dataclass
class HeadsConfig:
    fast_discrete_head: bool = True
    fast_vocab_size: int = 512
    phase_head: bool = True
    num_phases: int = 16
    affordance_head: bool = True
    num_affordance_types: int = 8  # v0.10.2: was hardcoded in AffordanceHead
    subgoal_head: bool = False
    label_smoothing: float = 0.1  # v0.9.2: was hardcoded in model init
    action_range: Tuple[float, float] = (-1.0, 1.0)  # v0.9.2: shared by bin_centers + discretise


@dataclass
class EMAConfig:
    enable: bool = True
    initial_decay: float = 0.999
    final_decay: float = 0.9999
    ramp_steps: int = 20000


@dataclass
class WorldModelConfig:
    """Configuration for the world model components (v0.4)."""
    enable: bool = False
    d_model: int = 2048
    z_dim: int = 4096        # 2 * d_model
    # Stochastic state
    n_categories: int = 48
    n_classes: int = 48
    # Imagination Mamba
    imagination_layers: int = 8
    imagination_d_state: int = 128
    # Object physics
    num_slots: int = 24
    gnn_layers: int = 6
    d_node: int = 512
    # Imagination rollout
    horizon: int = 32
    checkpoint_every: int = 8
    # Noise augmentation
    max_noise_sigma: float = 0.7
    noise_buckets: int = 16
    # Heads
    reward_bins: int = 255
    value_bins: int = 255
    # KL
    kl_free_bits: float = 1.0
    kl_alpha: float = 0.8
    # Optional decoders
    enable_visual_decoder: bool = True
    enable_subgoal_planner: bool = True


@dataclass
class ModelConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    multi_camera: MultiCameraConfig = field(default_factory=MultiCameraConfig)
    grounder: GrounderConfig = field(default_factory=GrounderConfig)
    temporal_core: TemporalCoreConfig = field(default_factory=TemporalCoreConfig)
    action_expert: ActionExpertConfig = field(default_factory=ActionExpertConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    num_embodiments: int = 16
    proprio_dim: int = 14  # v0.9.1: decoupled from action_dim
    proprio_range: Tuple[float, float] = (-1.0, 1.0)  # v0.10: decoupled from action_range


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class RTCTrainConfig:
    enable: bool = False
    execution_horizon: int = 8
    overlap_ratio: float = 0.333
    inpaint_overlap: bool = True


@dataclass
class FASTERTrainConfig:
    enable: bool = False
    near_ratio: float = 0.3
    near_steps: int = 2
    far_steps: int = 8


@dataclass
class TrainConfig:
    max_steps: int = 120000
    warmup_steps: int = 3000
    learning_rate: float = 2e-4
    backbone_lr_scale: float = 0.1  # v0.10.5: backbone LoRA LR = learning_rate × scale
    expert_lr_scale: float = 0.5  # v0.10.5: expert LR = learning_rate × scale (Stage B/C)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optimizer: str = "adamw_torch_fused"
    lr_scheduler: str = "cosine"

    sequence_window: int = 24
    semantic_refresh_stride: int = 6
    medium_update_stride: int = 2
    global_batch_size: int = 64
    per_device_batch_size: int = 2
    grad_accum_steps: int = 4

    bf16: bool = True
    fsdp: bool = True
    checkpointing: bool = True

    trainable: List[str] = field(default_factory=list)
    frozen: List[str] = field(default_factory=list)

    stop_gradient_cond_prefix: bool = False
    block_fm_to_backbone: bool = False
    ema_decay: float = 0.9999
    timestep_schedule: str = "logit_normal"

    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "fast_discrete": 1.0,
        "phase": 0.5,
        "affordance": 0.3,
        "consistency": 0.3,
        "flow_matching": 1.0,
    })

    rtc: RTCTrainConfig = field(default_factory=RTCTrainConfig)
    faster: FASTERTrainConfig = field(default_factory=FASTERTrainConfig)

    resume_from: Optional[str] = None
    log_interval: int = 50
    eval_interval: int = 2000
    save_interval: int = 5000
    output_dir: str = "outputs/v2_default"


# ---------------------------------------------------------------------------
# Inference config
# ---------------------------------------------------------------------------

@dataclass
class RTCInferConfig:
    enable: bool = True
    freeze_prefix_steps: int = 4
    inpaint_overlap: bool = True


@dataclass
class FASTERInferConfig:
    enable: bool = True
    near_ratio: float = 0.3
    near_steps: int = 1
    far_steps: int = 4


@dataclass
class InferConfig:
    control_hz: float = 50.0
    semantic_hz: float = 12.5
    medium_hz: float = 25.0
    chunk_horizon: int = 24
    execution_horizon: int = 8
    rtc: RTCInferConfig = field(default_factory=RTCInferConfig)
    faster: FASTERInferConfig = field(default_factory=FASTERInferConfig)
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    compile: bool = False
    batch_size: int = 1


@dataclass
class DataConfig:
    format: Optional[str] = None
    paths: List[str] = field(default_factory=list)
    data_dir: Optional[str] = None
    dataset_name: Optional[str] = None
    split: str = "train"
    image_key: str = "agentview_rgb"
    proprio_key: str = "robot0_joint_pos"
    action_key: str = "actions"
    language_key: str = "language_instruction"
    language: str = "complete the task"
    embodiment_id: int = 0
    max_episodes: Optional[int] = None
    normalizer_stats_dir: Optional[str] = None  # v0.10.1: explicit path; falls back to {output_dir}/normalizer_stats
    val_data_dir: Optional[str] = None  # v0.10.5: separate val data directory; if None, split by episode ratio
    val_ratio: float = 0.1  # fraction of episodes for val when val_data_dir is None
    # v2: multi-camera
    camera_keys: List[str] = field(default_factory=lambda: [
        "agentview_rgb", "wrist_rgb", "overhead_rgb",
    ])


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class HybridVLAv2Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stage: str = "a"


# ---------------------------------------------------------------------------
# Loader (reused from v1)
# ---------------------------------------------------------------------------

def _merge_dict(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge_dict(base[k], v)
        else:
            base[k] = v
    return base


def _dict_to_dataclass(cls, data: dict):
    if not isinstance(data, dict):
        return data
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in data.items():
        if k not in field_types:
            # v0.9.2: warn on unknown keys to catch YAML typos early (R2)
            warnings.warn(
                f"Unknown config key '{k}' in {cls.__name__}, ignored",
                stacklevel=2,
            )
            continue
        ft = field_types[k]
        if isinstance(ft, str):
            ft = eval(ft, globals(), locals())
        if isinstance(ft, type) and hasattr(ft, "__dataclass_fields__"):
            kwargs[k] = _dict_to_dataclass(ft, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str | Path) -> HybridVLAv2Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if "defaults" in raw:
        base_paths = raw.pop("defaults")
        merged: dict = {}
        for bp in base_paths:
            bp = (path.parent / bp).with_suffix(".yaml")
            with open(bp) as f:
                _merge_dict(merged, yaml.safe_load(f))
        _merge_dict(merged, raw)
        raw = merged
    return _dict_to_dataclass(HybridVLAv2Config, raw)
