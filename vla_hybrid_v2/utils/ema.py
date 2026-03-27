"""Exponential Moving Average with decay schedule ramp.

Usage:
    ema = EMAModel(model, initial_decay=0.999, final_decay=0.9999, ramp_steps=20000)
    for step, batch in enumerate(loader):
        loss = model(batch); loss.backward(); opt.step()
        ema.update(model, step)
    ema.apply(model)        # swap in EMA weights for eval
    evaluate(model)
    ema.restore(model)      # swap back
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


class EMAModel:
    """EMA with linear decay ramp: decay(step) interpolates from
    initial_decay to final_decay over ramp_steps.
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
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - decay)

    def apply(self, model: nn.Module) -> None:
        """Replace model params with EMA weights. Call restore() to undo."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> dict:
        return {
            "shadow": self.shadow,
            "initial_decay": self.initial_decay,
            "final_decay": self.final_decay,
            "ramp_steps": self.ramp_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self.shadow = state["shadow"]
        self.initial_decay = state.get("initial_decay", self.initial_decay)
        self.final_decay = state.get("final_decay", self.final_decay)
        self.ramp_steps = state.get("ramp_steps", self.ramp_steps)
