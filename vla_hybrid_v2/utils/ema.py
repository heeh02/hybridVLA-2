"""Exponential Moving Average with decay schedule ramp.

Usage:
    ema = EMAModel(model, initial_decay=0.999, final_decay=0.9999, ramp_steps=20000)
    for step, batch in enumerate(loader):
        loss = model(batch); loss.backward(); opt.step()
        ema.update(model, step)
    ema.apply(model)        # swap in EMA weights for eval
    evaluate(model)
    ema.restore(model)      # swap back

FSDP compatibility:
    Initialise EMA *before* FSDP wrapping so shadows hold full (unsharded)
    parameters.  update / apply / restore use ``summon_full_params`` when
    the model is FSDP-wrapped.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import torch
from torch import Tensor, nn


_FSDP_PREFIX = "_fsdp_wrapped_module."


def _strip_fsdp_prefix(name: str) -> str:
    """Strip all FSDP wrapper prefixes from a parameter name."""
    while _FSDP_PREFIX in name:
        name = name.replace(_FSDP_PREFIX, "")
    return name


def _is_fsdp(model: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        return isinstance(model, FSDP)
    except ImportError:
        return False


@contextmanager
def _maybe_summon_full_params(model: nn.Module, writeback: bool = False):
    """Summon full params for FSDP models; no-op otherwise."""
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(model, writeback=writeback, rank0_only=False):
            yield
    else:
        yield


class EMAModel:
    """EMA with linear decay ramp: decay(step) interpolates from
    initial_decay to final_decay over ramp_steps.

    Must be initialised *before* FSDP wrapping so that shadow stores
    full (unsharded) parameter tensors with their original names.
    """

    def __init__(
        self,
        model: nn.Module,
        initial_decay: float = 0.999,
        final_decay: float = 0.9999,
        ramp_steps: int = 20_000,
    ) -> None:
        assert ramp_steps > 0, f"ramp_steps must be positive, got {ramp_steps}"
        self.initial_decay = initial_decay
        self.final_decay = final_decay
        self.ramp_steps = ramp_steps
        self.shadow: Dict[str, Tensor] = {}
        self.backup: Dict[str, Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _get_decay(self, step: int) -> float:
        if step >= self.ramp_steps:
            return self.final_decay
        t = step / self.ramp_steps
        return self.initial_decay + t * (self.final_decay - self.initial_decay)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        decay = self._get_decay(step)
        with _maybe_summon_full_params(model, writeback=False):
            for name, param in model.named_parameters():
                clean = _strip_fsdp_prefix(name)
                if param.requires_grad and clean in self.shadow:
                    self.shadow[clean].lerp_(param.data, 1.0 - decay)

    def apply(self, model: nn.Module) -> None:
        """Replace model params with EMA weights. Call restore() to undo."""
        with _maybe_summon_full_params(model, writeback=True):
            for name, param in model.named_parameters():
                clean = _strip_fsdp_prefix(name)
                if clean in self.shadow:
                    self.backup[clean] = param.data.clone()
                    param.data.copy_(self.shadow[clean])

    def restore(self, model: nn.Module) -> None:
        """Restore original weights saved by apply()."""
        with _maybe_summon_full_params(model, writeback=True):
            for name, param in model.named_parameters():
                clean = _strip_fsdp_prefix(name)
                if clean in self.backup:
                    param.data.copy_(self.backup[clean])
        self.backup.clear()

    def state_dict(self) -> dict:
        return {
            "shadow": self.shadow,
            "initial_decay": self.initial_decay,
            "final_decay": self.final_decay,
            "ramp_steps": self.ramp_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        loaded_shadow = state["shadow"]
        # Filter out shape-mismatched keys (e.g. ActionHistoryEncoder resized
        # in v0.10.10).  Mismatched shadows are dropped and will be
        # re-initialised from the current model params at the next update().
        import logging as _logging
        _logger = _logging.getLogger(__name__)
        dropped = []
        for k in list(loaded_shadow.keys()):
            if k in self.shadow and loaded_shadow[k].shape != self.shadow[k].shape:
                dropped.append(
                    f"  {k}: ckpt {list(loaded_shadow[k].shape)} vs "
                    f"model {list(self.shadow[k].shape)}"
                )
                del loaded_shadow[k]
        if dropped:
            _logger.warning(
                "EMA: dropped %d shadow keys with shape mismatch "
                "(pre-v0.10.10 checkpoint):\n%s",
                len(dropped), "\n".join(dropped[:10]),
            )
        self.shadow.update(loaded_shadow)
        self.initial_decay = state.get("initial_decay", self.initial_decay)
        self.final_decay = state.get("final_decay", self.final_decay)
        self.ramp_steps = state.get("ramp_steps", self.ramp_steps)
