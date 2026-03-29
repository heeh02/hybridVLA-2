"""Distributed training utilities for HybridVLA v2.

Provides FSDP wrapping, activation checkpointing, gradient clipping,
and process group helpers — adapted for the v2 module set
(TriRateMambaCore, HierarchicalGrounder, FlowActionExpert).
"""

from __future__ import annotations

import functools
import logging
import os
import random
from typing import Optional, Set, Type

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def setup_distributed(backend: str = "nccl", seed: int = 42) -> int:
    if not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend=backend)
        else:
            logger.info("Non-distributed mode (use torchrun for multi-GPU).")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    seed_everything(seed + (dist.get_rank() if dist.is_initialized() else 0))
    return local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# FSDP
# ---------------------------------------------------------------------------

def _get_v2_wrap_classes() -> Set[Type[nn.Module]]:
    """Module classes that FSDP should wrap individually."""
    from vla_hybrid_v2.models.attention_grounder import GrounderBlock
    from vla_hybrid_v2.models.flow_action_expert import (
        ExpertAttentionBlock,
        ExpertMambaBlock,
    )
    from vla_hybrid_v2.models.mamba_core import MambaBlock
    return {MambaBlock, GrounderBlock, ExpertMambaBlock, ExpertAttentionBlock}


def wrap_fsdp(
    model: nn.Module,
    mixed_precision: bool = True,
    use_activation_checkpointing: bool = True,
    sync_module_states: bool = True,
) -> nn.Module:
    """Wrap model with FSDP using v2-specific auto-wrap policy."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    wrap_cls = _get_v2_wrap_classes()
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=wrap_cls,
    )

    mp = None
    if mixed_precision:
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        sync_module_states=sync_module_states,
        use_orig_params=True,
        limit_all_gathers=True,
    )

    if use_activation_checkpointing:
        _apply_activation_checkpointing(model, wrap_cls)

    if is_main_process() and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        logger.info(f"FSDP wrapped — {alloc:.2f} GB allocated")

    return model


def _apply_activation_checkpointing(
    model: nn.Module, cls: Set[Type[nn.Module]],
) -> None:
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda m: isinstance(m, tuple(cls)),
        )
        logger.info(f"Activation checkpointing on: {[c.__name__ for c in cls]}")
    except ImportError:
        logger.warning("torch checkpoint_wrapper not available.")


# ---------------------------------------------------------------------------
# FSDP dtype normalization & verification
# ---------------------------------------------------------------------------


def normalize_model_dtypes_for_fsdp(
    model: nn.Module,
    target_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Cast all floating-point params/buffers to *target_dtype* before FSDP wrap.

    With ``use_orig_params=True``, FSDP stores params in their original dtype
    and casts to ``param_dtype`` during forward.  Pre-normalizing to bf16:
    1. halves param + optimizer-state memory,
    2. makes checkpoints consistent across save/resume,
    3. eliminates hidden float32 stragglers (e.g. SSM A_log, D).

    Safe for backbone params already in bf16 (no-op cast).
    Integer/bool buffers are untouched by ``is_floating_point()`` guard.

    Call AFTER ``model.to(device)`` + checkpoint load, BEFORE EMA + FSDP wrap.
    """
    converted = 0
    for name, param in model.named_parameters():
        if param.is_floating_point() and param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)
            converted += 1
    for name, buf in model.named_buffers():
        if buf.is_floating_point() and buf.dtype != target_dtype:
            buf.data = buf.data.to(target_dtype)
            converted += 1
    logger.info(
        "normalize_model_dtypes_for_fsdp: converted %d tensors -> %s",
        converted, target_dtype,
    )


def verify_model_dtypes(
    model: nn.Module,
    expected_dtype: torch.dtype = torch.bfloat16,
    label: str = "",
) -> bool:
    """Check all floating-point params have *expected_dtype*.  Returns True if OK.

    Use after checkpoint load to detect dtype regression.
    """
    tag = f" [{label}]" if label else ""
    violations = []
    for name, param in model.named_parameters():
        if param.is_floating_point() and param.dtype != expected_dtype:
            violations.append((name, param.dtype))
    if violations:
        logger.warning(
            "dtype verification%s FAILED: %d params not %s",
            tag, len(violations), expected_dtype,
        )
        for name, dtype in violations[:10]:
            logger.warning("  %s: %s", name, dtype)
        if len(violations) > 10:
            logger.warning("  ... and %d more", len(violations) - 10)
        return False
    logger.info("dtype verification%s: all params %s", tag, expected_dtype)
    return True


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


def clip_grad_norm_fsdp(model: nn.Module, max_norm: float) -> torch.Tensor:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(model, FSDP):
            return model.clip_grad_norm_(max_norm)
    except ImportError:
        pass
    return torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm,
    )
