"""HybridVLA v2: Full model assembly.

Composes all v2 sub-modules into the tri-rate VLA pipeline:

1. Qwen2-VL-7B backbone (multi-scale, multi-camera)
2. Hierarchical Attention Grounder (96 latents → 24 compressed slots)
3. Tri-Rate Mamba Core (Fast 20L + Medium 6L + Slow 10L)
4. Flow Action Expert (18L, AdaRMSNorm, midpoint ODE)
5. Discrete heads (FAST 512-bin, Phase 16-class, Affordance)

Stage-gated training (same semantics as v1):
- Stage A: backbone LoRA + grounder + tri-rate core + heads. Expert frozen.
- Stage B: adds expert with cond_prefix.detach(). EMA starts.
- Stage C: full fine-tune with RTC (overlap inpainting) + FASTER (per-step weighting).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vla_hybrid_v2.config import HybridVLAv2Config
from vla_hybrid_v2.losses.consistency_loss import V2ConsistencyLoss
from vla_hybrid_v2.losses.discrete_loss import DiscreteCELoss, PhaseLoss
from vla_hybrid_v2.losses.flow_matching import FlowMatchingLoss
from vla_hybrid_v2.models.attention_grounder import HierarchicalAttentionGrounder
from vla_hybrid_v2.models.discrete_heads import (
    AffordanceHead,
    FASTDiscreteHead,
    PhaseHead,
)
from vla_hybrid_v2.models.flow_action_expert import FlowActionExpert
from vla_hybrid_v2.models.mamba_core import (
    ActionHistoryEncoder,
    TriRateMambaCore,
)
from vla_hybrid_v2.models.qwen2vl_backbone import Qwen2VLBackboneWrapper
from vla_hybrid_v2.types import (
    ActionExpertOutput,
    ActionHistoryBuffer,
    ControlStepOutput,
    GrounderOutput,
    RuntimeCache,
    TemporalOutput,
    TriRateTemporalState,
)

logger = logging.getLogger(__name__)


class HybridVLAv2(nn.Module):
    """Complete HybridVLA v2 model.

    ~9.9B total params (7.6B frozen backbone + 2.3B trainable).
    Designed for 8×H100-80GB.
    """

    def __init__(self, cfg: HybridVLAv2Config) -> None:
        super().__init__()
        self.cfg = cfg
        mcfg = cfg.model

        # ---- Sub-modules ----
        self.backbone = Qwen2VLBackboneWrapper.from_config(mcfg.backbone)

        d_core = mcfg.temporal_core.d_model  # 2048

        self.grounder = HierarchicalAttentionGrounder(
            hidden_size=mcfg.grounder.hidden_size,
            num_latents=mcfg.grounder.num_latents,
            num_object_slots=mcfg.grounder.num_object_slots,
            compressed_slots=mcfg.grounder.compressed_slots,
            num_layers=mcfg.grounder.num_layers,
            num_heads=mcfg.grounder.num_heads,
            mlp_ratio=mcfg.grounder.mlp_ratio,
            dropout=mcfg.grounder.dropout,
            hierarchical_compression=mcfg.grounder.hierarchical_compression,
            compression_layer=mcfg.grounder.compression_layer,
        )

        tcfg = mcfg.temporal_core
        self.temporal_core = TriRateMambaCore(
            fast_layers=tcfg.fast_layers,
            medium_layers=tcfg.medium_layers,
            slow_layers=tcfg.slow_layers,
            d_model=tcfg.d_model,
            fast_d_state=tcfg.fast_d_state,
            medium_d_state=tcfg.medium_d_state,
            slow_d_state=tcfg.slow_d_state,
            d_conv=tcfg.d_conv, expand=tcfg.expand,
            fusion_type=tcfg.fusion_type,
            fusion_heads=tcfg.fusion_heads,
            fusion_layers=tcfg.fusion_layers,
            mamba_impl=tcfg.mamba_impl,
        )

        _force_fb = (tcfg.mamba_impl == "fallback")
        self.action_history_encoder = ActionHistoryEncoder(
            action_dim=mcfg.action_expert.action_dim,
            d_model=d_core,
            d_state=tcfg.action_history_d_state,
            num_layers=tcfg.action_history_layers,
            force_fallback=_force_fb,
        )

        ecfg = mcfg.action_expert
        # v0.5 fix: cond_prefix is already projected to d_expert by
        # core_to_expert in _build_cond_prefix, so cond_dim = d_expert.
        self.action_expert = FlowActionExpert(
            d_model=ecfg.d_model, num_layers=ecfg.num_layers,
            num_heads=ecfg.num_heads, chunk_horizon=ecfg.chunk_horizon,
            action_dim=ecfg.action_dim, d_state=ecfg.d_state,
            d_conv=ecfg.d_conv, expand=ecfg.expand,
            cond_dim=ecfg.d_model, cond_tokens=ecfg.cond_tokens,
            dropout=ecfg.dropout,
        )

        # ---- Discrete heads ----
        self.fast_head: Optional[FASTDiscreteHead] = None
        if mcfg.heads.fast_discrete_head:
            self.fast_head = FASTDiscreteHead(
                input_dim=d_core, action_dim=ecfg.action_dim,
                vocab_size=mcfg.heads.fast_vocab_size,
                chunk_horizon=ecfg.chunk_horizon,
            )

        self.phase_head: Optional[PhaseHead] = None
        if mcfg.heads.phase_head:
            self.phase_head = PhaseHead(
                input_dim=d_core, num_phases=mcfg.heads.num_phases,
            )

        self.affordance_head: Optional[AffordanceHead] = None
        if mcfg.heads.affordance_head:
            self.affordance_head = AffordanceHead(
                input_dim=d_core,
                num_affordance_types=mcfg.heads.num_affordance_types,
            )

        # L-15: warn if phase/affordance heads are enabled — they require
        # labels from the data adapter (phase_labels / affordance_labels).
        # Real adapters (HDF5/LIBERO) currently do not produce these labels,
        # so the heads will receive no supervision and their tokens in
        # cond_prefix will carry no learned semantics.
        if self.phase_head is not None:
            logger.warning(
                "PhaseHead is enabled but requires 'phase_labels' in each "
                "batch. Ensure your data adapter provides them, otherwise "
                "the head receives zero loss and its cond_prefix token is noise."
            )
        if self.affordance_head is not None:
            logger.warning(
                "AffordanceHead is enabled but requires 'affordance_labels' "
                "in each batch. Ensure your data adapter provides them."
            )

        # ---- Projections ----
        # v0.9.1: proprio_dim decoupled from action_dim (Issue 1)
        self.proprio_proj = nn.Linear(mcfg.proprio_dim, d_core)
        self.prev_action_proj = nn.Linear(ecfg.action_dim, d_core)
        self.embodiment_embedding = nn.Embedding(
            mcfg.num_embodiments, d_core,
        )

        # ---- Condition prefix builder ----
        self.cond_builder = nn.Sequential(
            nn.LayerNorm(d_core),
            nn.Linear(d_core, d_core), nn.GELU(),
            nn.Linear(d_core, d_core),
        )

        # ---- Core → Expert projection (2048 → 1536) ----
        d_expert = ecfg.d_model
        if d_expert != d_core:
            self.core_to_expert = nn.Linear(d_core, d_expert)
            self.proprio_to_expert = nn.Linear(d_core, d_expert)
            self.emb_to_expert = nn.Linear(d_core, d_expert)
        else:
            self.core_to_expert = nn.Identity()
            self.proprio_to_expert = nn.Identity()
            self.emb_to_expert = nn.Identity()

        # ---- Precomputed buffers (v0.6: avoid per-forward allocation) ----
        V = mcfg.heads.fast_vocab_size
        # v0.9.2: action_range from config (was hardcoded [-1, 1])
        lo, hi = mcfg.heads.action_range
        self.register_buffer("_fast_bin_centers", torch.linspace(lo, hi, V))

        # ---- Losses ----
        self.flow_matching_loss = FlowMatchingLoss(
            timestep_schedule=cfg.train.timestep_schedule,
        )
        # v0.9.2: label_smoothing from config (was hardcoded 0.1)
        self.discrete_loss = DiscreteCELoss(label_smoothing=mcfg.heads.label_smoothing)
        self.phase_loss = PhaseLoss()
        self.consistency_loss = V2ConsistencyLoss(
            action_dim=ecfg.action_dim,
            temperature=cfg.train.consistency_temperature,
            slow_fast_weight=cfg.train.consistency_slow_fast_weight,
            action_weight=cfg.train.consistency_action_weight,
        )

        # ---- World Model (v0.4 integration) ----
        self.imagination_engine = None
        self.world_model_loss_fn = None
        wmcfg = mcfg.world_model
        if wmcfg.enable:
            from vla_hybrid_v2.experimental.world_model.imagination_engine import (
                ImaginationEngine,
            )
            from vla_hybrid_v2.experimental.world_model.world_model_loss import WorldModelLoss

            self.imagination_engine = ImaginationEngine(
                d_model=wmcfg.d_model,
                action_dim=ecfg.action_dim,
                num_slots=wmcfg.num_slots,
                horizon=wmcfg.horizon,
                checkpoint_every=wmcfg.checkpoint_every,
                n_categories=wmcfg.n_categories,
                n_classes=wmcfg.n_classes,
                enable_visual_decoder=wmcfg.enable_visual_decoder,
                enable_subgoal_planner=wmcfg.enable_subgoal_planner,
            )
            self.world_model_loss_fn = WorldModelLoss(
                kl_free_bits=wmcfg.kl_free_bits,
                kl_alpha=wmcfg.kl_alpha,
                enable_visual_loss=wmcfg.enable_visual_decoder,
            )

    # ------------------------------------------------------------------
    # VLA ↔ World Model interface (v0.4)
    # ------------------------------------------------------------------

    def get_world_model_state(
        self,
        grounder_out: GrounderOutput,
        temporal_out: TemporalOutput,
    ) -> dict:
        """Extract representations needed by the world model.

        Returns z_det (initial state), obs_encoding (for posterior),
        and compressed object slots (for physics loss targets).
        """
        return {
            "z_det": temporal_out.fused_state,                  # [B, d_core]
            "obs_encoding": temporal_out.fused_state,           # [B, d_core]
            "target_slots": grounder_out.compressed_object_slots,  # [B, S, d_core]
        }

    # ------------------------------------------------------------------
    # Condition prefix
    # ------------------------------------------------------------------

    def _build_cond_prefix(
        self, grounder_out: GrounderOutput, temporal_out: TemporalOutput,
    ) -> Tensor:
        """Build [B, C, D_core] → project to [B, C, D_expert]."""
        tokens = [
            grounder_out.global_token.unsqueeze(1),         # [B, 1, D]
            grounder_out.compressed_object_slots,            # [B, 24, D]
            grounder_out.phase_token.unsqueeze(1),           # [B, 1, D]
            grounder_out.uncertainty_token.unsqueeze(1),     # [B, 1, D]
            grounder_out.affordance_token.unsqueeze(1),      # [B, 1, D]
            temporal_out.fused_state.unsqueeze(1),           # [B, 1, D]
            temporal_out.fast_token.unsqueeze(1),            # [B, 1, D]
            temporal_out.medium_token.unsqueeze(1),          # [B, 1, D]
            temporal_out.slow_token.unsqueeze(1),            # [B, 1, D]
        ]
        cond = torch.cat(tokens, dim=1)  # [B, 32, D_core]
        cond = self.cond_builder(cond)

        target_c = self.cfg.model.action_expert.cond_tokens
        B = cond.shape[0]
        if cond.shape[1] < target_c:
            pad = torch.zeros(B, target_c - cond.shape[1], cond.shape[2],
                              device=cond.device, dtype=cond.dtype)
            cond = torch.cat([cond, pad], dim=1)
        elif cond.shape[1] > target_c:
            logger.warning(
                "_build_cond_prefix: truncating %d → %d tokens. "
                "Temporal tokens beyond target_c=%d are discarded — "
                "consider increasing action_expert.cond_tokens.",
                cond.shape[1], target_c, target_c,
            )
            cond = cond[:, :target_c, :]

        return self.core_to_expert(cond)

    # ------------------------------------------------------------------
    # Batch validation (v0.9.1, Issue 4)
    # ------------------------------------------------------------------

    def _validate_batch(self, batch: Dict[str, Any]) -> None:
        """Fail-fast validation of required batch keys and shapes.

        v0.9.1: basic key + dim checks.
        v0.9.3: added T consistency, input/mask shape, vision coupling,
                 embodiment range.
        """
        ecfg = self.cfg.model.action_expert
        mcfg = self.cfg.model

        # Required keys
        for key in ("actions", "proprio", "prev_actions", "input_ids", "attention_mask"):
            if key not in batch or batch[key] is None:
                raise ValueError(f"Missing required batch key: '{key}'")

        actions = batch["actions"]
        if actions.dim() != 4:
            raise ValueError(f"actions must be [B, T, H, A], got {actions.shape}")
        if actions.shape[2] != ecfg.chunk_horizon:
            raise ValueError(
                f"actions horizon={actions.shape[2]}, expected {ecfg.chunk_horizon}"
            )
        if actions.shape[3] != ecfg.action_dim:
            raise ValueError(
                f"actions dim={actions.shape[3]}, expected {ecfg.action_dim}"
            )

        proprio = batch["proprio"]
        if proprio.dim() != 3:
            raise ValueError(f"proprio must be [B, T, P], got {proprio.shape}")
        if proprio.shape[2] != mcfg.proprio_dim:
            raise ValueError(
                f"proprio dim={proprio.shape[2]}, expected {mcfg.proprio_dim}"
            )

        prev_actions = batch["prev_actions"]
        if prev_actions.dim() != 3 or prev_actions.shape[2] != ecfg.action_dim:
            raise ValueError(
                f"prev_actions must be [B, T, {ecfg.action_dim}], got {prev_actions.shape}"
            )

        # v0.9.3: T consistency across temporal fields
        T_act = actions.shape[1]
        if proprio.shape[1] != T_act:
            raise ValueError(f"proprio T={proprio.shape[1]} != actions T={T_act}")
        if prev_actions.shape[1] != T_act:
            raise ValueError(
                f"prev_actions T={prev_actions.shape[1]} != actions T={T_act}"
            )

        # v0.9.3: input_ids and attention_mask must match
        if batch["input_ids"].shape != batch["attention_mask"].shape:
            raise ValueError(
                f"input_ids shape {batch['input_ids'].shape} != "
                f"attention_mask shape {batch['attention_mask'].shape}"
            )

        # v0.9.3: pixel_values and image_grid_thw must co-occur
        has_pv = batch.get("pixel_values") is not None
        has_thw = batch.get("image_grid_thw") is not None
        if has_pv != has_thw:
            raise ValueError(
                "pixel_values and image_grid_thw must both be present or both absent"
            )

        # v0.10.2: step_weights shape check
        sw = batch.get("step_weights")
        if sw is not None:
            if sw.shape != (actions.shape[0], ecfg.chunk_horizon):
                raise ValueError(
                    f"step_weights must be [B, H]=[{actions.shape[0]}, {ecfg.chunk_horizon}], "
                    f"got {sw.shape}"
                )

        # v0.9.3: embodiment_id range check
        emb = batch.get("embodiment_id")
        if emb is not None:
            if emb.max() >= mcfg.num_embodiments:
                raise ValueError(
                    f"embodiment_id max={emb.max().item()} >= "
                    f"num_embodiments={mcfg.num_embodiments}"
                )

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward_train(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        self._validate_batch(batch)
        B = batch["actions"].shape[0]
        T = batch["actions"].shape[1]
        device = batch["actions"].device
        stage = self.cfg.stage

        # ---- Semantic refresh schedule ----
        if batch.get("semantic_refresh_steps") is not None:
            refresh_steps: List[int] = batch["semantic_refresh_steps"]
        else:
            stride = self.cfg.train.semantic_refresh_stride
            refresh_steps = list(range(0, T, stride))

        R = len(refresh_steps)
        refresh_map = {}
        current_r = 0
        for t in range(T):
            if current_r + 1 < R and t >= refresh_steps[current_r + 1]:
                current_r += 1
            refresh_map[t] = current_r

        # Medium update schedule: every medium_update_stride steps
        medium_stride = self.cfg.train.medium_update_stride
        medium_set = set(range(0, T, medium_stride))

        # ---- Multi-camera: resolve num_cameras ----
        num_cameras = 1
        nc = batch.get("num_cameras")
        if nc is not None:
            num_cameras = int(nc[0].item()) if hasattr(nc, '__getitem__') else int(nc)

        # ---- Backbone + Grounder per refresh point ----
        grounder_outputs: List[GrounderOutput] = []
        if batch.get("refresh_input_ids") is not None:
            for r in range(R):
                backbone_out = self.backbone.forward_semantic(
                    input_ids=batch["refresh_input_ids"][:, r],
                    attention_mask=batch["refresh_attention_mask"][:, r],
                    pixel_values=batch.get("refresh_pixel_values_list", [None] * R)[r],
                    image_grid_thw=batch.get("refresh_image_grid_thw_list", [None] * R)[r],
                    num_cameras=num_cameras,
                )
                grounder_mask = batch["refresh_attention_mask"][:, r].bool()
                grounder_outputs.append(self.grounder(
                    backbone_out["last_hidden_state"], attention_mask=grounder_mask,
                ))
        else:
            backbone_out = self.backbone.forward_semantic(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                num_cameras=num_cameras,
            )
            backbone_hidden = backbone_out["last_hidden_state"]
            grounder_mask = batch["attention_mask"].bool()
            single_grounder_out = self.grounder(backbone_hidden, attention_mask=grounder_mask)
            for _ in range(R):
                grounder_outputs.append(single_grounder_out)

        # ---- Temporal processing ----
        temporal_state = TriRateTemporalState()
        fused_states_list: List[Tensor] = []
        fast_tokens_list: List[Tensor] = []
        temporal_outputs: List[TemporalOutput] = []

        emb_id = batch.get("embodiment_id",
                           torch.zeros(B, dtype=torch.long, device=device))
        embodiment_token = self.embodiment_embedding(emb_id)

        refresh_set = set(refresh_steps)

        # Action history buffer
        action_history_buf = ActionHistoryBuffer(
            max_len=self.cfg.model.temporal_core.action_history_len
        )
        # Pre-fill with zeros
        for _ in range(action_history_buf.max_len):
            action_history_buf.push(
                torch.zeros(B, self.cfg.model.action_expert.action_dim, device=device)
            )

        for t in range(T):
            proprio_token = self.proprio_proj(batch["proprio"][:, t])
            prev_action_token = self.prev_action_proj(batch["prev_actions"][:, t])

            semantic_refresh = t in refresh_set
            medium_update = t in medium_set

            steps_since = torch.full(
                (B,), temporal_state.steps_since_refresh,
                device=device, dtype=torch.long,
            )
            stale_token = self.temporal_core.stale_encoder(steps_since)

            r = refresh_map[t]
            grounder_out = grounder_outputs[r]

            # Encode action history
            action_history_token = self.action_history_encoder.encode(
                action_history_buf.get()
            )

            temporal_out = self.temporal_core(
                global_token=grounder_out.global_token,
                object_slots=grounder_out.compressed_object_slots,
                phase_token=grounder_out.phase_token,
                uncertainty_token=grounder_out.uncertainty_token,
                affordance_token=grounder_out.affordance_token,
                proprio_token=proprio_token,
                prev_action_token=prev_action_token,
                stale_token=stale_token,
                embodiment_token=embodiment_token,
                action_history_token=action_history_token,
                state=temporal_state,
                semantic_refresh=semantic_refresh,
                medium_update=medium_update,
                use_checkpoint=self.cfg.train.checkpointing,
            )

            temporal_state = temporal_out.next_state
            fused_states_list.append(temporal_out.fused_state)
            fast_tokens_list.append(temporal_out.fast_token)
            temporal_outputs.append(temporal_out)

            # Update action history (teacher-forcing: uses GT prev_actions
            # during training; control_step uses model output actions instead)
            action_history_buf.push(batch["prev_actions"][:, t])

        fused_states = torch.stack(fused_states_list, dim=1)
        fast_tokens = torch.stack(fast_tokens_list, dim=1)

        # ---- Losses (v0.10.3 P1-C: multi-step supervision) ----
        # Supervise FAST/phase/affordance at ALL T timesteps to improve
        # gradient density. Expert loss stays at t=-1 (expensive, Stage B/C).
        losses: Dict[str, Tensor] = {}
        weights = self.cfg.train.loss_weights
        _lo, _hi = self.cfg.model.heads.action_range
        _V = self.cfg.model.heads.fast_vocab_size

        # fast_continuous at last step — needed for consistency loss
        fast_continuous: Optional[Tensor] = None

        # FAST discrete loss — all T steps
        if self.fast_head is not None:
            # Vectorised: reshape [B, T, D] → [B*T, D], head outputs [B*T, H, A, V]
            BT = B * T
            fast_logits_flat = self.fast_head(fused_states.reshape(BT, -1))  # [B*T, H, A, V]
            fast_targets_flat = FASTDiscreteHead.discretise_actions(
                batch["actions"], lo=_lo, hi=_hi, V=_V,
            ).reshape(BT, self.cfg.model.action_expert.chunk_horizon, -1)    # [B*T, H, A]
            losses["loss_fast"] = self.discrete_loss(
                fast_logits_flat, fast_targets_flat,
            ) * weights.get("fast_discrete", 1.0)
            # fast_continuous at last step for consistency
            fast_logits_last = fast_logits_flat.reshape(
                B, T, *fast_logits_flat.shape[1:]
            )[:, -1]                                                  # [B, H, A, V]
            fast_probs = fast_logits_last.softmax(dim=-1)
            fast_continuous = (fast_probs * self._fast_bin_centers).sum(dim=-1)

        # Phase loss — all T steps
        if self.phase_head is not None and batch.get("phase_labels") is not None:
            phase_losses = []
            for t_sup in range(T):
                r = refresh_map[t_sup]
                phase_logits_t = self.phase_head(grounder_outputs[r].phase_token)
                phase_loss_t = self.phase_loss(
                    phase_logits_t, batch["phase_labels"][:, t_sup],
                )
                phase_losses.append(phase_loss_t)
            losses["loss_phase"] = torch.stack(phase_losses).mean() * weights.get("phase", 0.5)

        # Affordance loss — all T steps
        if self.affordance_head is not None and batch.get("affordance_labels") is not None:
            aff_losses = []
            for t_sup in range(T):
                r = refresh_map[t_sup]
                aff_logits_t = self.affordance_head(grounder_outputs[r].affordance_token)
                aff_loss_t = nn.functional.cross_entropy(
                    aff_logits_t, batch["affordance_labels"][:, t_sup],
                )
                aff_losses.append(aff_loss_t)
            losses["loss_affordance"] = torch.stack(aff_losses).mean() * weights.get("affordance", 0.3)

        # ---- Stage-gated action expert (t=-1 only) ----
        target_actions = batch["actions"][:, -1]  # [B, H, A]
        if stage != "a":
            last_grounder = grounder_outputs[-1]
            cond_prefix = self._build_cond_prefix(last_grounder, temporal_outputs[-1])
            if (self.cfg.train.stop_gradient_cond_prefix
                    or self.cfg.train.block_fm_to_backbone):
                cond_prefix = cond_prefix.detach()

            flow_t = self.flow_matching_loss.sample_timestep(B, device)
            noise = torch.randn_like(target_actions)
            noisy_actions = self.flow_matching_loss.interpolate(noise, target_actions, flow_t)

            proprio_expert = self.proprio_proj(batch["proprio"][:, -1])
            proprio_for_expert = self.proprio_to_expert(proprio_expert)
            emb_for_expert = self.emb_to_expert(embodiment_token)

            expert_out = self.action_expert(
                noisy_actions=noisy_actions, flow_t=flow_t,
                cond_prefix=cond_prefix,
                proprio_token=proprio_for_expert,
                embodiment_token=emb_for_expert,
            )

            # Recover denoised x_1 from flow matching (used by RTC, FASTER aux, consistency)
            # x_t = (1-t)*x_0 + t*x_1, v = x_1 - x_0  =>  x_1 = x_t + (1-t)*v
            expert_denoised = noisy_actions + (1.0 - flow_t[:, None, None]) * expert_out.velocity

            step_weights = batch.get("step_weights", None)

            # ---- FASTER: per-step weighted FM loss (Stage C) ----
            if stage == "c" and self.cfg.train.faster.enable:
                faster_cfg = self.cfg.train.faster
                H = self.cfg.model.action_expert.chunk_horizon
                near_boundary = max(1, int(faster_cfg.near_ratio * H))

                # Near steps get higher weight to prioritise imminent actions
                faster_w = torch.ones(H, device=device)
                far_ratio = max(faster_cfg.far_steps, 1) / max(faster_cfg.near_steps, 1)
                faster_w[:near_boundary] *= far_ratio
                faster_w = faster_w * (H / faster_w.sum())  # normalise so total = H

                target_velocity = target_actions - noise
                per_step_mse = (expert_out.velocity - target_velocity).pow(2).mean(dim=-1)  # [B, H]
                if step_weights is not None:
                    per_step_mse = per_step_mse * step_weights
                loss_fm = (per_step_mse * faster_w.unsqueeze(0)).mean()
            else:
                loss_fm = self.flow_matching_loss(
                    expert_out.velocity, noise, target_actions, flow_t,
                    step_weights=step_weights,
                )
            losses["loss_fm"] = loss_fm * weights.get("flow_matching", 1.0)

            # ---- RTC: overlapping chunk inpainting loss (Stage C) ----
            if stage == "c" and self.cfg.train.rtc.enable:
                rtc_cfg = self.cfg.train.rtc
                exec_H = rtc_cfg.execution_horizon
                overlap = max(1, int(rtc_cfg.overlap_ratio * exec_H))
                H = self.cfg.model.action_expert.chunk_horizon

                if overlap < H and overlap < exec_H:
                    # Generate a "previous chunk" with fast low-precision sampling.
                    # NOTE (L-5): Ideally prev_chunk should use a *different*
                    # cond_prefix (from a prior timestep) to match inference
                    # semantics. As a pragmatic approximation, we add small
                    # noise to cond_prefix to break self-consistency and
                    # encourage cross-boundary robustness.
                    with torch.no_grad():
                        noise_scale = 0.01
                        prev_cond = cond_prefix + noise_scale * torch.randn_like(cond_prefix)
                        prev_chunk = self.action_expert.sample(
                            cond_prefix=prev_cond,
                            proprio_token=proprio_for_expert,
                            embodiment_token=emb_for_expert,
                            num_steps=rtc_cfg.prev_chunk_steps,
                            solver="euler",
                        )  # [B, H, A]

                    # Inpainting: current chunk head should match previous chunk tail
                    prev_tail = prev_chunk[:, exec_H - overlap: exec_H].detach()
                    curr_head = expert_denoised[:, :overlap]
                    loss_rtc = F.mse_loss(curr_head, prev_tail)

                    # Boundary smoothness: penalise acceleration discontinuity
                    # across the prev→curr seam.  Need ≥3 points for a
                    # second-order finite difference, so use full prev_tail
                    # + curr_head (2*overlap points when overlap ≥ 2).
                    if rtc_cfg.smooth_weight > 0 and overlap >= 2:
                        boundary = torch.cat(
                            [prev_tail.detach(), curr_head], dim=1,
                        )  # [B, 2*overlap, A]
                        accel = boundary[:, 2:] - 2 * boundary[:, 1:-1] + boundary[:, :-2]
                        if accel.numel() > 0:
                            loss_rtc = loss_rtc + rtc_cfg.smooth_weight * accel.pow(2).mean()

                    losses["loss_rtc"] = loss_rtc * weights.get("rtc", 0.3)

            # ---- FASTER: near-horizon auxiliary loss (Stage C) ----
            # Penalise the denoised prediction's near-horizon error more
            # heavily.  Previous version used torch.no_grad() multi-step
            # sampling, which produced zero gradient — replaced with the
            # differentiable single-step denoised prediction (expert_denoised).
            if stage == "c" and self.cfg.train.faster.enable:
                faster_cfg = self.cfg.train.faster
                H = self.cfg.model.action_expert.chunk_horizon
                near_boundary = max(1, int(faster_cfg.near_ratio * H))
                if faster_cfg.aux_loss_weight > 0:
                    loss_faster_aux = F.mse_loss(
                        expert_denoised[:, :near_boundary],
                        target_actions[:, :near_boundary],
                    )
                    losses["loss_faster"] = loss_faster_aux * weights.get("faster", 0.2)

            # v2 consistency: contrastive + slow-fast + action agreement
            losses["loss_consistency"] = self.consistency_loss(
                fused_states,
                fast_tokens=fast_tokens,
                slow_token=temporal_outputs[-1].slow_token,
                discrete_actions=fast_continuous,
                continuous_actions=expert_denoised.detach(),
            ) * weights.get("consistency", 0.3)
        else:
            losses["loss_consistency"] = self.consistency_loss(
                fused_states,
                fast_tokens=fast_tokens,
                slow_token=temporal_outputs[-1].slow_token,
            ) * weights.get("consistency", 0.3)

        losses["loss_total"] = torch.stack(list(losses.values())).sum()
        return losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def semantic_step(self, input_ids, attention_mask,
                      pixel_values=None, image_grid_thw=None,
                      num_cameras: int = 1) -> GrounderOutput:
        with torch.no_grad():
            backbone_out = self.backbone.forward_semantic(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                num_cameras=num_cameras,
            )
            return self.grounder(
                backbone_out["last_hidden_state"],
                attention_mask=attention_mask.bool() if attention_mask is not None else None,
            )

    def control_step(self, proprio, prev_action, semantic_summary,
                     runtime_state, embodiment_id=None,
                     num_sample_steps=8) -> ControlStepOutput:
        """50 Hz control step with chunk caching.

        v0.9: Reuses the cached action chunk when steps remain, avoiding
        redundant expert forward passes.  A new chunk is generated when:
        (a) no cached chunk exists, (b) all steps in the chunk are consumed,
        or (c) a semantic refresh occurred (new observation invalidates plan).
        """
        B = proprio.shape[0]
        device = proprio.device
        exec_horizon = self.cfg.infer.execution_horizon

        with torch.no_grad():
            if self.cfg.infer.faster.enable:
                raise NotImplementedError(
                    "cfg.infer.faster.enable=True but FASTER inference is not "
                    "implemented yet. FASTER is train-only (Stage C). "
                    "Set infer.faster.enable=False to use standard sampling."
                )

            # ---- Check if cached chunk is still valid ----
            # v0.9.1: use monotonic counter instead of fragile id() (Issue A2)
            semantic_refresh = (runtime_state.refresh_counter != runtime_state._last_seen_refresh)
            need_new_chunk = (
                runtime_state.current_chunk is None
                or runtime_state.chunk_step >= exec_horizon
                or semantic_refresh
            )

            # ---- Temporal core always runs (updates recurrent state) ----
            proprio_token = self.proprio_proj(proprio)
            prev_action_token = self.prev_action_proj(prev_action)
            if embodiment_id is None:
                embodiment_id = torch.zeros(B, dtype=torch.long, device=device)
            embodiment_token = self.embodiment_embedding(embodiment_id)

            # v0.9.1: derive stride from infer Hz, not train config (Issue 6)
            medium_stride = max(1, round(
                self.cfg.infer.control_hz / self.cfg.infer.medium_hz
            ))
            medium_update = (runtime_state.temporal_state.steps_since_medium
                             >= medium_stride - 1)

            steps_since = torch.full(
                (B,), runtime_state.temporal_state.steps_since_refresh,
                device=device, dtype=torch.long,
            )
            stale_token = self.temporal_core.stale_encoder(steps_since)

            # Action history
            if runtime_state.action_history is None:
                runtime_state.action_history = torch.zeros(
                    B, self.cfg.model.temporal_core.action_history_len,
                    self.cfg.model.action_expert.action_dim, device=device,
                )
            action_history_token = self.action_history_encoder.encode(
                runtime_state.action_history,
            )

            temporal_out = self.temporal_core(
                global_token=semantic_summary.global_token,
                object_slots=semantic_summary.compressed_object_slots,
                phase_token=semantic_summary.phase_token,
                uncertainty_token=semantic_summary.uncertainty_token,
                affordance_token=semantic_summary.affordance_token,
                proprio_token=proprio_token,
                prev_action_token=prev_action_token,
                stale_token=stale_token,
                embodiment_token=embodiment_token,
                action_history_token=action_history_token,
                state=runtime_state.temporal_state,
                semantic_refresh=semantic_refresh,
                medium_update=medium_update,
            )

            runtime_state.temporal_state = temporal_out.next_state
            runtime_state.last_semantic = semantic_summary
            runtime_state._last_seen_refresh = runtime_state.refresh_counter

            # ---- Generate new chunk only when needed (v0.9) ----
            if need_new_chunk:
                cond_prefix = self._build_cond_prefix(semantic_summary, temporal_out)
                proprio_for_expert = self.proprio_to_expert(proprio_token)
                emb_for_expert = self.emb_to_expert(embodiment_token)

                solver = self.cfg.model.action_expert.ode_solver
                denoised = self.action_expert.sample(
                    cond_prefix=cond_prefix,
                    proprio_token=proprio_for_expert,
                    embodiment_token=emb_for_expert,
                    num_steps=num_sample_steps,
                    solver=solver,
                )
                # v0.11: RTC — blend overlap region with previous chunk tail
                rtc_cfg = self.cfg.infer.rtc
                if rtc_cfg.enable and runtime_state.prev_chunk_tail is not None:
                    overlap = runtime_state.prev_chunk_tail.shape[1]
                    if overlap > 0 and overlap <= denoised.shape[1]:
                        alpha = torch.linspace(1, 0, overlap, device=device)
                        alpha = alpha[None, :, None]  # [1, overlap, 1]
                        denoised[:, :overlap] = (
                            alpha * runtime_state.prev_chunk_tail
                            + (1 - alpha) * denoised[:, :overlap]
                        )

                # Save current chunk tail for next RTC blending
                if rtc_cfg.enable:
                    rtc_overlap = max(1, int(rtc_cfg.overlap_ratio * exec_horizon))
                    rtc_overlap = min(rtc_overlap, exec_horizon, denoised.shape[1])
                    runtime_state.prev_chunk_tail = denoised[
                        :, exec_horizon - rtc_overlap: exec_horizon
                    ].clone()

                runtime_state.current_chunk = denoised  # [B, H, A]
                runtime_state.chunk_step = 0

            # ---- Extract current action from chunk ----
            action = runtime_state.current_chunk[:, runtime_state.chunk_step]
            runtime_state.chunk_step += 1

            # Update action history
            if runtime_state.action_history is not None:
                runtime_state.action_history = torch.roll(
                    runtime_state.action_history, -1, dims=1,
                )
                runtime_state.action_history[:, -1] = action

            # v0.9.1: return the single action, not the full chunk (Issue 2)
            return ControlStepOutput(
                action=action,
                chunk=runtime_state.current_chunk,
                chunk_step=runtime_state.chunk_step,
                semantic_refresh=semantic_refresh,
            )

    def init_runtime(self, batch_size=1, device="cuda") -> RuntimeCache:
        return RuntimeCache(
            temporal_state=TriRateTemporalState(),
            device=torch.device(device),
        )
