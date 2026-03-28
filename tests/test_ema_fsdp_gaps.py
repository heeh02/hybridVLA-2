"""Gap tests for EMA round-trip, FSDP param grouping, TriRateMamba, and Grounder.

Covers issues A-D from v1.0 audit:
- A: EMA update/apply/restore with simulated FSDP prefixed names
- B: Per-module LR param group assignment under FSDP prefixed names
- C: Cross-stage resume EMA shadow initialization ordering
- D: TriRateMambaCore and HierarchicalGrounder smoke tests
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tests.conftest import A, B, D, D_EXP, H, L, P, T, _build_model, _mini_cfg

# ---------------------------------------------------------------------------
# A: EMA round-trip (update -> apply -> verify -> restore -> verify)
# ---------------------------------------------------------------------------


class TestEMARoundTrip:
    """Prove EMA update/apply/restore actually changes and restores weights."""

    def _make_model_and_ema(self):
        from vla_hybrid_v2.utils.ema import EMAModel

        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.Linear(16, 4),
        )
        ema = EMAModel(model, initial_decay=0.9, final_decay=0.99, ramp_steps=100)
        return model, ema

    def test_shadow_initialized_from_model(self):
        model, ema = self._make_model_and_ema()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert torch.equal(ema.shadow[name], param.data)

    def test_update_changes_shadow(self):
        model, ema = self._make_model_and_ema()
        original_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Perturb model weights to simulate a training step
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 10.0)

        ema.update(model, step=0)

        # Shadow should have moved toward new weights
        for name in original_shadow:
            assert not torch.equal(ema.shadow[name], original_shadow[name]), \
                f"Shadow for {name} did not change after update"

    def test_apply_overwrites_model_weights(self):
        model, ema = self._make_model_and_ema()

        # Perturb model
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 10.0)
        ema.update(model, step=50)

        # Save pre-apply weights
        pre_apply = {n: p.data.clone() for n, p in model.named_parameters()}

        ema.apply(model)

        # Model weights should now equal EMA shadow
        for name, param in model.named_parameters():
            clean = name
            if clean in ema.shadow:
                assert torch.equal(param.data, ema.shadow[clean]), \
                    f"apply() did not set {name} to shadow value"
                # And they should differ from pre-apply (model was perturbed)
                assert not torch.equal(param.data, pre_apply[name]), \
                    f"apply() had no effect on {name}"

    def test_restore_recovers_original_weights(self):
        model, ema = self._make_model_and_ema()

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 10.0)

        # Save the training weights (after perturbation)
        training_weights = {n: p.data.clone() for n, p in model.named_parameters()}

        ema.update(model, step=50)
        ema.apply(model)

        # Weights are now EMA, not training weights
        for name, param in model.named_parameters():
            if name in ema.shadow:
                assert not torch.equal(param.data, training_weights[name])

        ema.restore(model)

        # Weights should be back to training weights
        for name, param in model.named_parameters():
            assert torch.equal(param.data, training_weights[name]), \
                f"restore() did not recover {name}"

    def test_full_round_trip_apply_restore_is_identity(self):
        """apply -> restore should be a no-op on model weights."""
        model, ema = self._make_model_and_ema()
        original = {n: p.data.clone() for n, p in model.named_parameters()}

        ema.apply(model)
        ema.restore(model)

        for name, param in model.named_parameters():
            assert torch.equal(param.data, original[name]), \
                f"Round-trip changed {name}"


# ---------------------------------------------------------------------------
# A (cont'd): EMA with simulated FSDP prefix names
# ---------------------------------------------------------------------------


class TestEMAWithFSDPPrefix:
    """Simulate FSDP-wrapped parameter names and verify EMA still works."""

    def test_strip_fsdp_prefix_handles_nested(self):
        from vla_hybrid_v2.utils.ema import _strip_fsdp_prefix

        assert _strip_fsdp_prefix("backbone.lora.weight") == "backbone.lora.weight"
        assert _strip_fsdp_prefix(
            "_fsdp_wrapped_module.backbone.lora.weight"
        ) == "backbone.lora.weight"
        assert _strip_fsdp_prefix(
            "_fsdp_wrapped_module._fsdp_wrapped_module.backbone.lora.weight"
        ) == "backbone.lora.weight"

    def test_update_matches_with_fsdp_prefix(self):
        """Simulate: EMA init on clean names, update on FSDP-prefixed names."""
        from vla_hybrid_v2.utils.ema import EMAModel

        model = nn.Linear(4, 4)
        ema = EMAModel(model, initial_decay=0.9, final_decay=0.99, ramp_steps=100)
        assert "weight" in ema.shadow
        assert "bias" in ema.shadow

        # Create a wrapper that produces FSDP-prefixed names
        class FSDPSimWrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self._fsdp_wrapped_module = inner

        wrapped = FSDPSimWrapper(model)
        # Perturb the underlying model
        with torch.no_grad():
            model.weight.add_(torch.ones_like(model.weight) * 5.0)

        original_shadow = {k: v.clone() for k, v in ema.shadow.items()}
        ema.update(wrapped, step=0)

        # Shadow should update despite prefixed names
        assert not torch.equal(ema.shadow["weight"], original_shadow["weight"]), \
            "EMA update failed to match FSDP-prefixed parameter names"


# ---------------------------------------------------------------------------
# B: Per-module LR param group assignment
# ---------------------------------------------------------------------------


class TestPerModuleLRGroups:
    """Verify param groups are correctly classified with/without FSDP prefix."""

    def _classify_param(self, name: str, cfg):
        from vla_hybrid_v2.utils.ema import _strip_fsdp_prefix

        clean = _strip_fsdp_prefix(name)
        if clean.startswith("backbone"):
            return "backbone", cfg.train.backbone_lr_scale
        elif clean.startswith("action_expert"):
            return "expert", cfg.train.expert_lr_scale
        else:
            return "core", 1.0

    def test_clean_names_classified_correctly(self):
        cfg = _mini_cfg("b")
        group, scale = self._classify_param("backbone.lora.weight", cfg)
        assert group == "backbone"
        assert scale == cfg.train.backbone_lr_scale

        group, scale = self._classify_param("action_expert.layers.0.weight", cfg)
        assert group == "expert"
        assert scale == cfg.train.expert_lr_scale

        group, scale = self._classify_param("grounder.latent_queries", cfg)
        assert group == "core"
        assert scale == 1.0

    def test_fsdp_prefixed_names_classified_correctly(self):
        """FSDP prefix must not break module classification."""
        cfg = _mini_cfg("b")
        prefix = "_fsdp_wrapped_module."

        group, scale = self._classify_param(
            prefix + "backbone.lora.weight", cfg
        )
        assert group == "backbone"

        group, scale = self._classify_param(
            prefix + "action_expert.layers.0.weight", cfg
        )
        assert group == "expert"

        group, scale = self._classify_param(
            prefix + "grounder.latent_queries", cfg
        )
        assert group == "core"

    def test_double_fsdp_prefix(self):
        """Nested FSDP wrapping should also be handled."""
        cfg = _mini_cfg("b")
        nested = "_fsdp_wrapped_module._fsdp_wrapped_module.backbone.lora.weight"
        group, _ = self._classify_param(nested, cfg)
        assert group == "backbone"

    def test_all_model_params_get_a_group(self):
        """Every trainable param in a Stage B model must land in a group."""
        cfg = _mini_cfg("b")
        model = _build_model(cfg)
        # Simulate stage gate
        from scripts.train_unified import configure_trainable_modules
        configure_trainable_modules(model, "b", cfg)

        unclassified = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            group, _ = self._classify_param(name, cfg)
            if group not in ("backbone", "expert", "core"):
                unclassified.append(name)

        assert len(unclassified) == 0, f"Unclassified params: {unclassified}"


# ---------------------------------------------------------------------------
# C: Cross-stage resume EMA shadow initialization ordering
# ---------------------------------------------------------------------------


class TestCrossStageResumeEMAOrder:
    """Verify EMA shadow is initialized from resumed (not initial) weights."""

    def test_ema_shadow_matches_resumed_weights(self):
        """Simulate: init model -> resume_from checkpoint -> init EMA.
        EMA shadow should match resumed weights, not random init."""
        from vla_hybrid_v2.utils.ema import EMAModel

        # Step 1: Create model with known init
        model = nn.Linear(8, 4)
        init_weight = model.weight.data.clone()

        # Step 2: Simulate cross-stage checkpoint (different weights)
        resumed_weight = torch.randn_like(model.weight.data) * 100
        with torch.no_grad():
            model.weight.copy_(resumed_weight)

        # Step 3: Init EMA after resume (as train_unified.py does)
        ema = EMAModel(model)

        # Shadow should match resumed weights, NOT init weights
        assert torch.equal(ema.shadow["weight"], resumed_weight), \
            "EMA shadow was initialized from pre-resume weights!"
        assert not torch.equal(ema.shadow["weight"], init_weight)

    def test_same_stage_resume_overwrites_shadow(self):
        """auto_resume -> load_state_dict should overwrite initial shadow."""
        from vla_hybrid_v2.utils.ema import EMAModel

        model = nn.Linear(8, 4)
        ema = EMAModel(model)

        # Simulate saved EMA state with different shadow
        saved_shadow = {"weight": torch.ones(4, 8) * 42.0,
                        "bias": torch.ones(4) * -1.0}
        saved_state = {
            "shadow": saved_shadow,
            "initial_decay": 0.999,
            "final_decay": 0.9999,
            "ramp_steps": 20000,
        }

        ema.load_state_dict(saved_state)

        assert torch.equal(ema.shadow["weight"], saved_shadow["weight"])
        assert torch.equal(ema.shadow["bias"], saved_shadow["bias"])

    def test_ema_save_load_roundtrip(self):
        """state_dict -> load_state_dict preserves shadow exactly."""
        from vla_hybrid_v2.utils.ema import EMAModel

        model = nn.Linear(8, 4)
        ema1 = EMAModel(model, initial_decay=0.99, final_decay=0.999, ramp_steps=5000)

        # Simulate some training
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema1.update(model, step=100)

        # Save and reload
        state = ema1.state_dict()
        ema2 = EMAModel(model)  # fresh EMA
        ema2.load_state_dict(state)

        for key in ema1.shadow:
            assert torch.equal(ema1.shadow[key], ema2.shadow[key]), \
                f"Shadow mismatch after save/load for {key}"
        assert ema2.initial_decay == 0.99
        assert ema2.final_decay == 0.999
        assert ema2.ramp_steps == 5000


# ---------------------------------------------------------------------------
# D: TriRateMambaCore smoke test
# ---------------------------------------------------------------------------


class TestTriRateMambaCoreSmoke:
    """Verify TriRateMambaCore forward pass runs without error."""

    def _make_core(self):
        from vla_hybrid_v2.models.mamba_core import TriRateMambaCore
        return TriRateMambaCore(
            fast_layers=2, medium_layers=1, slow_layers=1,
            d_model=D, fast_d_state=16, medium_d_state=16, slow_d_state=16,
            d_conv=4, expand=2, fusion_layers=1, fusion_heads=2,
            mamba_impl="fallback",
        )

    def test_forward_returns_temporal_output(self):
        from vla_hybrid_v2.types import TriRateTemporalState

        core = self._make_core()
        state = TriRateTemporalState()

        out = core(
            global_token=torch.randn(B, D),
            object_slots=torch.randn(B, 2, D),
            phase_token=torch.randn(B, D),
            uncertainty_token=torch.randn(B, D),
            affordance_token=torch.randn(B, D),
            proprio_token=torch.randn(B, D),
            prev_action_token=torch.randn(B, D),
            stale_token=torch.randn(B, D),
            embodiment_token=torch.randn(B, D),
            action_history_token=torch.randn(B, D),
            state=state,
            semantic_refresh=True,
            medium_update=True,
        )

        assert out.fused_state.shape == (B, D)
        assert out.fast_token.shape == (B, D)
        assert out.medium_token.shape == (B, D)
        assert out.slow_token.shape == (B, D)
        assert out.next_state is not None

    def test_multi_step_state_propagation(self):
        """Run 3 steps, verify state propagates without error."""
        from vla_hybrid_v2.types import TriRateTemporalState

        core = self._make_core()
        state = TriRateTemporalState()

        for t in range(3):
            out = core(
                global_token=torch.randn(B, D),
                object_slots=torch.randn(B, 2, D),
                phase_token=torch.randn(B, D),
                uncertainty_token=torch.randn(B, D),
                affordance_token=torch.randn(B, D),
                proprio_token=torch.randn(B, D),
                prev_action_token=torch.randn(B, D),
                stale_token=torch.randn(B, D),
                embodiment_token=torch.randn(B, D),
                action_history_token=torch.randn(B, D),
                state=state,
                semantic_refresh=(t == 0),
                medium_update=(t % 2 == 0),
            )
            state = out.next_state

        assert not torch.isnan(out.fused_state).any()

    def test_backward_through_core(self):
        """Verify gradients flow through the core."""
        from vla_hybrid_v2.types import TriRateTemporalState

        core = self._make_core()
        state = TriRateTemporalState()

        out = core(
            global_token=torch.randn(B, D),
            object_slots=torch.randn(B, 2, D),
            phase_token=torch.randn(B, D),
            uncertainty_token=torch.randn(B, D),
            affordance_token=torch.randn(B, D),
            proprio_token=torch.randn(B, D),
            prev_action_token=torch.randn(B, D),
            stale_token=torch.randn(B, D),
            embodiment_token=torch.randn(B, D),
            action_history_token=torch.randn(B, D),
            state=state,
            semantic_refresh=True,
            medium_update=True,
        )

        loss = out.fused_state.sum()
        loss.backward()

        grads_found = sum(
            1 for p in core.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grads_found > 0, "No gradients flowed through TriRateMambaCore"


# ---------------------------------------------------------------------------
# D: HierarchicalGrounder smoke test
# ---------------------------------------------------------------------------


class TestGrounderSmoke:
    """Verify HierarchicalAttentionGrounder forward pass."""

    def _make_grounder(self):
        from vla_hybrid_v2.models.attention_grounder import HierarchicalAttentionGrounder
        return HierarchicalAttentionGrounder(
            hidden_size=D,
            num_latents=12,
            num_object_slots=4,
            compressed_slots=2,
            num_layers=2,
            num_heads=2,
            mlp_ratio=2.0,
            hierarchical_compression=True,
            compression_layer=1,
        )

    def test_forward_output_structure(self):
        grounder = self._make_grounder()
        features = torch.randn(B, L, D)
        mask = torch.ones(B, L, dtype=torch.bool)

        out = grounder(features, attention_mask=mask)

        assert out.global_token.shape == (B, D)
        assert out.object_slots.shape == (B, 4, D)
        assert out.compressed_object_slots.shape == (B, 2, D)
        assert out.phase_token.shape == (B, D)
        assert out.uncertainty_token.shape == (B, D)
        assert out.affordance_token.shape == (B, D)

    def test_forward_no_nan(self):
        grounder = self._make_grounder()
        features = torch.randn(B, L, D)
        out = grounder(features)

        assert not torch.isnan(out.global_token).any()
        assert not torch.isnan(out.compressed_object_slots).any()

    def test_backward_through_grounder(self):
        grounder = self._make_grounder()
        features = torch.randn(B, L, D, requires_grad=True)
        out = grounder(features)
        loss = out.global_token.sum() + out.compressed_object_slots.sum()
        loss.backward()

        assert features.grad is not None
        assert features.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# D: FSDP training step simulation (single-process, no actual FSDP)
# ---------------------------------------------------------------------------


class TestFSDPTrainingStepSimulation:
    """Simulate optimizer step with per-module LR groups on a mini model."""

    def test_param_groups_have_correct_lr(self):
        """Build optimizer exactly as train_unified does, verify LR assignment."""
        from vla_hybrid_v2.utils.ema import _strip_fsdp_prefix

        cfg = _mini_cfg("b")
        model = _build_model(cfg)

        from scripts.train_unified import configure_trainable_modules
        configure_trainable_modules(model, "b", cfg)

        base_lr = cfg.train.learning_rate
        no_decay_keywords = {"bias", "res_scale", "LayerNorm.weight", "layer_norm.weight"}

        param_groups_map = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            clean = _strip_fsdp_prefix(name)
            if clean.startswith("backbone"):
                group = "backbone"
                lr_scale = cfg.train.backbone_lr_scale
            elif clean.startswith("action_expert"):
                group = "expert"
                lr_scale = cfg.train.expert_lr_scale
            else:
                group = "core"
                lr_scale = 1.0
            is_no_decay = any(nd in clean for nd in no_decay_keywords)
            key = f"{group}_{'nodecay' if is_no_decay else 'decay'}"
            if key not in param_groups_map:
                param_groups_map[key] = {
                    "params": [], "lr": base_lr * lr_scale,
                    "weight_decay": 0.0 if is_no_decay else cfg.train.weight_decay,
                }
            param_groups_map[key]["params"].append(param)

        param_groups = list(param_groups_map.values())

        # Verify at least core and expert groups exist in Stage B
        group_keys = set(param_groups_map.keys())
        assert any(k.startswith("core") for k in group_keys), \
            f"No core group found. Groups: {group_keys}"
        assert any(k.startswith("expert") for k in group_keys), \
            f"No expert group found in Stage B. Groups: {group_keys}"

        # Verify LR values
        for key, pg in param_groups_map.items():
            if key.startswith("backbone"):
                assert pg["lr"] == pytest.approx(base_lr * cfg.train.backbone_lr_scale)
            elif key.startswith("expert"):
                assert pg["lr"] == pytest.approx(base_lr * cfg.train.expert_lr_scale)
            elif key.startswith("core"):
                assert pg["lr"] == pytest.approx(base_lr)

    def test_one_training_step_updates_weights(self):
        """Full mini training step: forward -> backward -> optimizer.step."""
        cfg = _mini_cfg("b")
        model = _build_model(cfg)
        from scripts.train_unified import configure_trainable_modules
        configure_trainable_modules(model, "b", cfg)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
        )

        batch = {
            "actions": torch.randn(B, T, H, A),
            "proprio": torch.randn(B, T, P),
            "prev_actions": torch.randn(B, T, A),
            "input_ids": torch.randint(0, 1000, (B, L)),
            "attention_mask": torch.ones(B, L, dtype=torch.long),
        }

        # Save initial weights
        init_weights = {
            n: p.data.clone() for n, p in model.named_parameters()
            if p.requires_grad
        }

        losses = model.forward_train(batch)
        losses["loss_total"].backward()
        optimizer.step()

        # At least some weights should have changed
        changed = sum(
            1 for n, p in model.named_parameters()
            if p.requires_grad and not torch.equal(p.data, init_weights[n])
        )
        assert changed > 0, "No weights changed after training step"
