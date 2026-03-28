# architecture_v0_10_claude.md

# HybridVLA v2 — Complete Architecture Review

**Reviewer**: Claude
**Date**: 2025-03-28
**Codebase version**: v0.10.9 (post FSDP fix + ActionHistoryEncoder shrink)
**Total codebase**: ~8,941 lines, 46 Python files

---

## 1. System-Level Architecture Diagram

```
                         ┌───────────────────────────────────────────────────────────┐
                         │                     HybridVLAv2                            │
                         │            ~9.4B params (7.6B frozen, ~2.2B trainable)     │
                         └───────────────────────────┬───────────────────────────────┘
                                                     │
  ┌──────────────────────────────────────────────────┼──────────────────────────────────────────┐
  │            SEMANTIC PATH                         │              CONTROL PATH                 │
  │       (~12.5 Hz, heavy compute)                  │         (~50 Hz, lightweight)             │
  │                                                  │                                           │
  │  ┌─────────────────────────────┐                 │                                           │
  │  │  1. Qwen2-VL-7B Backbone   │                 │    proprio ──→ proprio_proj ──────────┐   │
  │  │     7.6B (frozen) + ~100M   │                 │    prev_act ──→ prev_action_proj ────┤   │
  │  │     LoRA r=64, all 28 layers│                 │    emb_id ──→ embodiment_emb ────────┤   │
  │  │     Multi-scale [10,18,28]  │                 │    steps_since → StaleTimeEncoding ──┤   │
  │  │     3584d → 2048d           │                 │                                      │   │
  │  └──────────────┬──────────────┘                 │    action_history [B,8,14]            │   │
  │                 │                                │      │                               │   │
  │  ┌──────────────▼──────────────┐                 │      ▼                               │   │
  │  │  2. Hierarchical Grounder   │                 │    ┌─────────────────────────┐       │   │
  │  │     ~200M params            │                 │    │ ActionHistoryEncoder     │       │   │
  │  │     96 latents, 8 layers    │                 │    │ ~1.6M params (v0.10.9)  │       │   │
  │  │     48 → 24 slot compress   │                 │    │ Linear(14,256) →         │       │   │
  │  │                              │                 │    │ 2L Mamba(d=256,st=64) → │       │   │
  │  │  Outputs:                   │                 │    │ Linear(256,2048)         │       │   │
  │  │    global_token     [B,D]   │                 │    └───────────┬─────────────┘       │   │
  │  │    compressed_slots [B,24,D]│                 │                │                      │   │
  │  │    phase_token      [B,D]   │                 │                │ action_history_token  │   │
  │  │    uncertainty_token[B,D]   │                 │                │ [B, 2048]             │   │
  │  │    affordance_token [B,D]   │                 │                │                      │   │
  │  └──────────────┬──────────────┘                 │    ┌───────────▼──────────────────┐  │   │
  │                 │ GrounderOutput                  │    │ compose 33-token input_seq   │◄─┘   │
  │                 │                                │    │ [B, 33, 2048]                │      │
  │                 └────────────────────────────────►│    │ = 9 singletons + 24 obj_slots│      │
  │                                                  │    └───────────┬──────────────────┘      │
  │                                                  │                │                          │
  │                                                  │    ┌───────────▼──────────────────┐      │
  │                                                  │    │ 3. Tri-Rate Mamba Core       │      │
  │                                                  │    │    ~972M params               │      │
  │                                                  │    │  ┌─────────────────────────┐ │      │
  │                                                  │    │  │ Fast  (20L, d_st=128)   │ │      │
  │                                                  │    │  │ every step    → fast_tok │ │      │
  │                                                  │    │  ├─────────────────────────┤ │      │
  │                                                  │    │  │ Medium (6L, d_st=128)   │ │      │
  │                                                  │    │  │ every 2nd step→ med_tok  │ │      │
  │                                                  │    │  ├─────────────────────────┤ │      │
  │                                                  │    │  │ Slow  (10L, d_st=256)   │ │      │
  │                                                  │    │  │ refresh only → slow_tok  │ │      │
  │                                                  │    │  └──────────┬──────────────┘ │      │
  │                                                  │    │  ┌──────────▼──────────────┐ │      │
  │                                                  │    │  │ CrossAttentionFusion     │ │      │
  │                                                  │    │  │ 2L, 8 heads             │ │      │
  │                                                  │    │  │ → fused_state [B,2048]  │ │      │
  │                                                  │    │  └─────────────────────────┘ │      │
  │                                                  │    └───────────┬──────────────────┘      │
  │                                                  │                │                          │
  │                                                  │       ┌────────┼──────────┐               │
  │                                                  │       │        │          │               │
  │                                                  │       ▼        ▼          ▼               │
  │                                                  │    ┌──────┐ ┌──────┐ ┌──────────┐        │
  │                                                  │    │ FAST │ │Phase │ │Affordance│        │
  │                                                  │    │ Head │ │ Head │ │  Head    │        │
  │                                                  │    │ 512V │ │ 16cl │ │  8cl     │        │
  │                                                  │    └──────┘ └──────┘ └──────────┘        │
  │                                                  │                │                          │
  │                                                  │    ┌───────────▼──────────────────┐      │
  │                                                  │    │ _build_cond_prefix           │      │
  │                                                  │    │ 32 tokens [B,32,2048]        │      │
  │                                                  │    │ → cond_builder → proj(1536)  │      │
  │                                                  │    └───────────┬──────────────────┘      │
  │                                                  │                │                          │
  │                                                  │    ┌───────────▼──────────────────┐      │
  │                                                  │    │ 4. Flow Action Expert        │      │
  │                                                  │    │    ~270M params               │      │
  │                                                  │    │    18L (M-M-A × 6)           │      │
  │                                                  │    │    d=1536, AdaRMSNorm         │      │
  │                                                  │    │    Flow Matching + ODE        │      │
  │                                                  │    │    → velocity [B,24,14]       │      │
  │                                                  │    └──────────────────────────────┘      │
  └──────────────────────────────────────────────────┴──────────────────────────────────────────┘
```

---

## 2. Module-by-Module Detailed Analysis

### 2.1 Qwen2-VL-7B Backbone

**File**: `models/qwen2vl_backbone.py` (296 lines)
**Class**: `Qwen2VLBackboneWrapper`
**Params**: ~7.6B total, ~100M trainable (LoRA)

```
Input:  input_ids [B, N], attention_mask [B, N], pixel_values, image_grid_thw
Output: last_hidden_state [B, N, 2048], vision_mask, text_mask
```

**Architecture**:
```
Qwen2-VL-7B-Instruct (28 layers, hidden_size=3584, FlashAttention-2)
  ├── Vision Tower (ViT) ── FROZEN
  ├── Text Layers 0-15 ── FROZEN
  ├── Text Layers 16-27 ── LoRA (rank=64, alpha=128, dropout=0.05)
  │     target: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  │
  ├── Multi-Scale Feature Extraction
  │     Extract hidden_states at layers [10, 18, 28]
  │     ├── Layer 10: spatial detail (early features)
  │     ├── Layer 18: intermediate features
  │     └── Layer 28: semantic features (final)
  │
  ├── MultiScaleAdapter  (~22M params)
  │     3 × (LayerNorm(3584) → Linear(3584, 2048))
  │     Learned softmax gate: Linear(2048×3, 3) → weighted sum
  │     Output: [B, N, 2048]
  │
  └── CameraPositionEmbedding  (optional, multi-camera)
        nn.Embedding(8, 2048) → added to vision token positions
        Identifies which camera each vision token came from
```

**Design rationale**: Qwen2-VL was chosen for its native multi-image support (image tokens interleaved in text sequence) and strong vision-language grounding. Multi-scale extraction from 3 layers approximates FPN without requiring separate decoders.

**Point of attention**: LoRA on all 28 layers (not just unfrozen ones) means gradients flow through ~80% frozen parameters. This is intentional — PEFT's LoRA injects adapters inside the frozen layers, so the frozen base weights participate in forward/backward but are not updated. LoRA trainable params: 28 layers × 7 modules × (64×3584 + 3584×64) ≈ 100M.

---

### 2.2 Hierarchical Attention Grounder

**File**: `models/attention_grounder.py` (261 lines)
**Class**: `HierarchicalAttentionGrounder`
**Params**: ~200M

```
Input:  backbone_hidden [B, N, 2048], attention_mask [B, N]
Output: GrounderOutput {
          global_token          [B, 2048]     — scene-level summary
          object_slots          [B, 48, 2048] — raw object slots (pre-compression)
          compressed_object_slots [B, 24, 2048] — refined object slots
          phase_token           [B, 2048]     — task phase encoding
          uncertainty_token     [B, 2048]     — epistemic uncertainty
          affordance_token      [B, 2048]     — affordance type encoding
        }
```

**Architecture**:
```
96 Learned Latent Queries [1, 96, 2048]
  Layout: [global(1), objects(48), phase(1), uncertainty(1), affordance(1), aux(44)]
  │
  ├── Layers 0-3: GrounderBlock × 4  (each: CrossAttn + SelfAttn + FFN)
  │     Latents [96] cross-attend to backbone features [N]
  │     Uses F.scaled_dot_product_attention (auto Flash)
  │
  ├── SlotCompression (after layer 3)
  │     24 learned routing queries cross-attend to 48 raw object slots
  │     CrossAttn + SelfAttn → compressed [B, 24, 2048]
  │     Latent layout changes: 96 → 72 (48 objects → 24 compressed)
  │
  ├── Layers 4-7: GrounderBlock × 4
  │     Continue processing with 72 latents
  │
  └── final_norm: LayerNorm(2048)
        Carve out named slots by position index
```

**Design commentary**:

The hierarchical compression (48→24 at layer 4) is architecturally sound — it lets early layers capture fine-grained object features while later layers reason over a compact set. The 24 compressed slots become the dominant representation downstream (they're 24 of 32 condition tokens for the action expert).

**Concern**: The `phase_token`, `uncertainty_token`, and `affordance_token` require explicit labels from the data adapter (`phase_labels`, `affordance_labels`). The code itself warns at init (L149-158) that these heads receive zero supervision if labels are missing. Current HDF5/LIBERO adapters do **not** produce these labels, so these tokens are effectively **random noise** in cond_prefix during current training. They occupy 3 of 32 condition tokens.

---

### 2.3 Tri-Rate Mamba Core

**File**: `models/mamba_core.py` (800 lines)
**Class**: `TriRateMambaCore` (composed of `FastMamba`, `MediumMamba`, `SlowMamba`, `CrossAttentionFusion`, `StaleTimeEncoding`) + separate `ActionHistoryEncoder`
**Params**: ~972M (core) + ~1.6M (ActionHistoryEncoder)

```
Input:  10 tokens → composed into input_seq [B, 33, 2048]
          global_token, phase_token, uncertainty_token, affordance_token,
          proprio_token, prev_action_token, stale_token, embodiment_token,
          action_history_token  (9 singletons)
          + object_slots [B, 24, 2048]  (24 compressed grounder slots)
        + TriRateTemporalState (recurrent SSM/conv states)

Output: TemporalOutput {
          fused_state   [B, 2048]  — cross-attention fusion of 3 stream tokens
          fast_token    [B, 2048]  — mean(fast stream output)
          medium_token  [B, 2048]  — mean(medium stream output) or cached
          slow_token    [B, 2048]  — mean(slow stream output) or cached
          next_state    TriRateTemporalState
        }
```

**Architecture — Three Streams**:

```
input_seq [B, 33, 2048]
  │
  ├── Fast Stream: FastMamba (20 layers, d_state=128, expand=2)
  │     Updated EVERY control step (~50 Hz)
  │     Captures: immediate reaction to proprioception + action changes
  │     Output: fast_out.mean(dim=1) → fast_token [B, 2048]
  │     Params: 20 × ~27M = ~540M
  │
  ├── Medium Stream: MediumMamba (6 layers, d_state=128, expand=2)
  │     Updated every 2nd step (~25 Hz)
  │     Captures: short-horizon motion planning
  │     Output: med_out.mean(dim=1) → medium_token [B, 2048]
  │     Params: 6 × ~27M = ~162M
  │
  ├── Slow Stream: SlowMamba (10 layers, d_state=256, expand=2)
  │     Updated on semantic refresh only (~12.5 Hz)
  │     Captures: long-horizon task structure, goal context
  │     Output: slow_out.mean(dim=1) → slow_token [B, 2048]
  │     Params: 10 × ~27M = ~270M
  │     Note: d_state=256 (2× others) — larger SSM state for longer memory
  │
  └── CrossAttentionFusion (2 layers, 8 heads)
        Learned fusion query [1, 1, 2048]
        kv = stack([fast_token, medium_token, slow_token]) + stale_proj(stale)
        2-layer cross-attention → fused_state [B, 2048]
        Params: ~50M

Total Tri-Rate Core (excl. ActionHistoryEncoder): ~540M + 162M + 270M + 50M = ~1,022M
```

**MambaBlock internals** (per block, d_model=2048, expand=2):
```
x → LayerNorm → in_proj (2048 → 8192) → split [4096, 4096]
                                            │         │
                                        x_main       z (gate)
                                            │
                                     Conv1d(4096, k=4, groups=4096)
                                            │
                                        SiLU activation
                                            │
                                   SSM: x_proj → (dt, B, C)
                                        dt_proj → Δ
                                        A_log → A
                                        selective_scan(x, Δ, A, B, C)
                                            │
                                        y = scan_out + D·x
                                        y = y * SiLU(z)    ← gating
                                            │
                                     out_proj (4096 → 2048)
                                            │
                                   res_scale * out + x      ← scaled residual
```

**res_scale initialization**: `1/sqrt(2*N)` where N = number of layers. For 20-layer fast stream: `res_scale_init = 1/sqrt(40) ≈ 0.158`. This prevents activation explosion in deep stacks.

**ActionHistoryEncoder** (v0.10.9: 2 layers, d_inner=256, d_state=64):
```
action_history [B, 8, 14]
  → Linear(14, 256)        [B, 8, 256]        action_proj
  → _MambaStack(2 layers,  [B, 8, 256]        d_model=256, d_state=64, expand=2
       d_inner=512)
  → out[:, -1, :]          [B, 256]           last token as summary
  → Linear(256, 2048)      [B, 2048]          out_proj to core dim
Params: ~1.6M   (v0.10.9 was 108M — reduced 67×)
```

Param breakdown:
- `action_proj`  Linear(14, 256):  14×256+256 = **3,840**
- `_MambaStack` 2 layers (d=256, expand=2, d_inner=512, d_state=64): 2 × ~512K = **~1,024K**
- `out_proj`    Linear(256, 2048): 256×2048+2048 = **526,336**
- **Total: ~1,554,176 ≈ 1.6M**

This is now appropriately sized for the 112-float input (K=8 × A=14).
The 2-layer Mamba at d=256 retains temporal modeling of the action sequence
(causal scan captures velocity/acceleration patterns) while the output
projection broadcasts the summary into the 2048-dim core space.

**StaleTimeEncoding**: Sinusoidal encoding of `steps_since_refresh` (0-256), similar to transformer positional encoding but encoding "staleness" of semantic information.

**Tri-Rate update schedule** (T=24, refresh_stride=6, medium_stride=2):
```
Step:  0  1  2  3  4  5 | 6  7  8  9 10 11 |12 13 14 15 16 17 |18 19 20 21 22 23
Fast:  ●  ●  ●  ●  ●  ● | ●  ●  ●  ●  ●  ● | ●  ●  ●  ●  ●  ● | ●  ●  ●  ●  ●  ●   (24/24)
Med:   ●     ●     ●    | ●     ●     ●     | ●     ●     ●     | ●     ●     ●        (12/24)
Slow:  ●                | ●                  | ●                  | ●                     (4/24)
Stale: 0  1  2  3  4  5 | 0  1  2  3  4  5  | 0  1  2  3  4  5  | 0  1  2  3  4  5
```

---

### 2.4 Flow Action Expert

**File**: `models/flow_action_expert.py` (361 lines)
**Class**: `FlowActionExpert`
**Params**: ~270M

```
Input:  noisy_actions [B, 24, 14], flow_t [B], cond_prefix [B, 32, 1536],
        proprio_token [B, 1536], embodiment_token [B, 1536]
Output: ActionExpertOutput { velocity [B, 24, 14] }
```

**Architecture**:
```
Noise Schedule: Logit-normal t ~ σ(N(0,1)), biasing toward t≈0.5
Interpolation:  x_t = (1-t)·x_0 + t·x_1   (Rectified Flow)

noisy_actions [B, 24, 14]
  │
  ├── action_proj: Linear(14, 1536)
  ├── + action_pos_emb: Learned(24, 1536)
  ├── + timestep_emb: Sinusoidal(t) → MLP(1536→6144→1536)
  │
  ├── Prepend: [proprio_token, embodiment_token, action_tokens]
  │     → x [B, 26, 1536]
  │
  ├── t_cond: t_cond_mlp(timestep_emb) → [B, 1536]  (for AdaRMSNorm)
  ├── cond: cond_proj(cond_prefix) → [B, 32, 1536]    (for cross-attention)
  │
  ├── 18 Layers (M-M-A × 6):
  │     │
  │     ├── ExpertMambaBlock (12 total)
  │     │     AdaRMSNorm(x, t_cond) → Mamba SSM
  │     │     Uses selective_scan_fn CUDA kernel when available
  │     │     d_inner = 3072, d_state = 96
  │     │     Params per block: ~14M
  │     │
  │     └── ExpertAttentionBlock (6 total)
  │           AdaRMSNorm cross-attn: x queries cond_prefix keys/values
  │           AdaRMSNorm self-attn: x self-attends
  │           AdaRMSNorm FFN
  │           d_model=1536, 24 heads, head_dim=64
  │           Params per block: ~28M
  │
  ├── out_norm: LayerNorm(1536)
  └── out_proj: Linear(1536, 14)
        → velocity [B, 24, 14]

Inference (ODE Sampling):
  Euler:     x_{i+1} = x_i + v(x_i, t_i) · dt         (8 steps)
  Midpoint:  x_mid = x_i + 0.5·dt·v(x_i, t_i)          (8 steps, 16 forward passes)
             x_{i+1} = x_i + dt·v(x_mid, t_mid)         2nd-order accuracy
```

**AdaRMSNorm** (from pi-0.5):
```
x_normed = x / RMS(x)
scale, shift, gate = Linear(cond, 3·dim)
output = sigmoid(gate) · (x_normed · (1+scale) + shift)

Key: gate bias initialized to +2 → sigmoid(2) ≈ 0.88
     Prevents 18-layer residual chain from halving activations
```

**Design commentary**:

The M-M-A pattern is well-motivated: Mamba layers provide efficient sequence mixing for the 26-token action sequence, while attention layers every 3rd position enable explicit cross-referencing with the 32-token condition prefix. AdaRMSNorm conditioning on flow timestep is critical — without it, the network cannot distinguish between denoising early (noisy) vs late (clean) actions.

The d_model=1536 (vs core's 2048) is an intentional cost/quality tradeoff — the expert processes shorter sequences (26 tokens) so smaller width suffices.

---

### 2.5 Discrete Heads

**File**: `models/discrete_heads.py` (77 lines)
**Params**: ~16M total

#### FASTDiscreteHead (~8M)
```
fused_state [B, 2048]
  → LayerNorm → Linear(2048, 768) → GELU     (encoder)
  → Linear(768, 24×192)                        (step_proj, factorized)
  → reshape [B×24, 192]
  → LayerNorm → Linear(192, 14×512)            (dim_head)
  → reshape [B, 24, 14, 512]                   (logits over 512 bins)
```

Predicts discretized actions at all 24 horizon steps. Trained with cross-entropy + label smoothing=0.1. This is the "fast" coarse action prediction that provides gradient signal during Stage A when the expert is frozen.

#### PhaseHead (~4M)
```
phase_token [B, 2048]
  → LayerNorm → Linear(2048, 1024) → GELU → Linear(1024, 16)
  → logits over 16 phases
```

#### AffordanceHead (~4M)
```
affordance_token [B, 2048]
  → LayerNorm → Linear(2048, 1024) → GELU → Linear(1024, 8)
  → logits over 8 affordance types
```

---

### 2.6 Projection & Bridge Layers

```
proprio_proj:         Linear(14, 2048)          14D proprio → core space
prev_action_proj:     Linear(14, 2048)          14D prev_action → core space
embodiment_embedding: Embedding(16, 2048)       embodiment ID → core space

core_to_expert:       Linear(2048, 1536)        core → expert projection
proprio_to_expert:    Linear(2048, 1536)        for expert proprio input
emb_to_expert:        Linear(2048, 1536)        for expert embodiment input

cond_builder:         LayerNorm(2048)           condition prefix processing
                      → Linear(2048, 2048)
                      → GELU
                      → Linear(2048, 2048)
```

---

### 2.7 Loss Functions

**File**: `losses/` (3 files, ~130 lines)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Loss Composition                                  │
│                                                                          │
│  loss_total = loss_fast + loss_phase + loss_affordance                   │
│             + loss_fm + loss_rtc + loss_faster                           │
│             + loss_consistency                                           │
│                                                                          │
│  Stage A:  loss_fast(×1.0) + loss_phase(×0.5) + loss_afford(×0.3)      │
│            + loss_consistency(×0.3)  [temporal only, no action]           │
│                                                                          │
│  Stage B:  + loss_fm(×1.0)                                               │
│            + loss_consistency(×0.3) [full: temporal + action agreement]   │
│                                                                          │
│  Stage C:  + loss_rtc(×0.3) + loss_faster(×0.2)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

| Loss | Function | Supervision | Applied at |
|------|----------|-------------|-----------|
| `loss_fast` | CrossEntropy(512 bins, label_smooth=0.1) | All T steps, all actions | Stage A/B/C |
| `loss_phase` | CrossEntropy(16 classes) | All T steps | A/B/C (requires labels) |
| `loss_affordance` | CrossEntropy(8 classes) | All T steps | A/B/C (requires labels) |
| `loss_fm` | MSE(predicted_velocity, target_velocity) | Last step only (t=-1) | Stage B/C |
| `loss_rtc` | MSE(curr_head, prev_tail) + smooth_weight×accel² | Overlap region | Stage C |
| `loss_faster` | MSE(denoised[:near], target[:near]) | Near-horizon (first 30%) | Stage C |
| `loss_consistency` | InfoNCE + SlowFastMSE + ActionCosine | Temporal states | A/B/C |

---

## 3. Three-Stage Training Pipeline

### 3.1 Module Connectivity Per Stage

```
                    ┌─────────────────────────────────────────────┐
                    │   ■ Trainable   □ Frozen   ⊘ Detached grad │
                    └─────────────────────────────────────────────┘

STAGE A: 120K steps, lr=2e-4
  Backbone:    □ frozen base + ■ LoRA (lr × 0.1 = 2e-5)
  Grounder:    ■ (lr × 1.0 = 2e-4)
  Tri-Rate:    ■ (lr × 1.0 = 2e-4)
  Expert:      □ FROZEN (no forward in loss, no grad)
  FAST Head:   ■ (primary action signal)
  Phase Head:  ■ (if labels available)
  Afford Head: ■ (if labels available)
  Projections: ■
  EMA:         ■ (decay ramp 0.999 → 0.9999 over 20K steps)

  Gradient flow:
    loss_fast → FAST Head → fused_state → Tri-Rate Core ← Grounder ← Backbone (LoRA only)
    loss_consistency (temporal only) → Tri-Rate Core ← Grounder ← Backbone (LoRA only)

  Key: Expert is completely frozen. No flow matching loss. FAST head provides
       the only action-level supervision. The grounder and temporal core must
       learn good representations from discrete action bins + consistency alone.


STAGE B: 200K steps, lr=2e-4
  Backbone:    □ frozen base + ■ LoRA (lr × 0.1)
  Grounder:    ■ (lr × 1.0)
  Tri-Rate:    ■ (lr × 1.0)
  Expert:      ■ (lr × 0.5 = 1e-4)
  FAST Head:   ■
  Phase Head:  ■
  Afford Head: ■
  Projections: ■
  EMA:         ■

  Gradient flow:
    loss_fm → Expert ← ⊘ cond_prefix.detach() ← [Grounder + Tri-Rate]
    loss_fast → fused_state → Tri-Rate ← Grounder ← Backbone
    loss_consistency (full) → fused_state, expert_denoised

  Key: cond_prefix.detach() blocks FM gradient from flowing into backbone/grounder/core.
       Expert trains in isolation on the FROZEN representations from Stage A.
       This prevents the expert's high-magnitude FM gradients from destabilizing
       the already-converged semantic representations.


STAGE C: 80K steps, lr=2e-4
  Backbone:    □ frozen base + ■ LoRA (lr × 0.1)
  Grounder:    ■ (lr × 1.0)
  Tri-Rate:    ■ (lr × 1.0)
  Expert:      ■ (lr × 0.5)
  FAST Head:   ■
  Phase Head:  ■
  Afford Head: ■
  Projections: ■
  EMA:         ■
  + RTC:       ■ (overlap inpainting loss)
  + FASTER:    ■ (near-horizon weighting)

  Gradient flow:
    loss_fm → Expert → cond_prefix → [Grounder + Tri-Rate + Backbone]
    (stop_gradient_cond_prefix default=False, block_fm_to_backbone default=False)
    loss_rtc → Expert (overlap consistency across chunks)
    loss_faster → Expert (near-horizon emphasis)

  Key: Full end-to-end fine-tuning. FM gradients now flow through cond_prefix
       all the way to backbone LoRA. RTC teaches chunk-boundary smoothness.
       FASTER upweights imminent actions (first 30% of 24-step chunk).
```

### 3.2 Cross-Stage Checkpoint Loading

```
Stage A output → checkpoint/model.pt + ema.pt + normalizer_stats/ + resolved_config.yaml
                                    ↓
Stage B: train_unified.py --resume_from <stage_a_checkpoint>
         load_checkpoint(model, strict=False)  ← allows new expert params
         EMA initialized AFTER resume (shadow = resumed weights)
         FSDP wrap AFTER EMA init
                                    ↓
Stage C: train_unified.py --resume_from <stage_b_checkpoint>
         Same flow
```

### 3.3 Training Data Flow

```
HDF5 Episode [~400 steps at 50Hz]
  │
  ├── Sample window: T=24 consecutive steps
  │
  ├── Per-window:
  │     actions [B, T=24, H=24, A=14]   — 24 future action chunks per step
  │     proprio [B, T=24, P=14]          — proprioception per step
  │     prev_actions [B, T=24, A=14]     — previous step's action
  │     input_ids, attention_mask         — tokenized language + image
  │     pixel_values, image_grid_thw      — vision tokens
  │
  ├── Semantic refresh: 4 VLM forward passes at steps [0, 6, 12, 18]
  │     Each produces backbone_hidden [B, N, 2048] → GrounderOutput
  │
  ├── Temporal loop: 24 iterations
  │     Each: compose 33-token input → Tri-Rate forward → fused_state
  │
  ├── Loss computation:
  │     FAST: all 24 steps (vectorized [B×24, ...])
  │     Expert: last step only (t=23)
  │     Phase/Affordance: all 24 steps (requires labels)
  │
  └── Normalization:
        ActionNormalizer: raw → [-1, 1] (action_range)
        ProprioNormalizer: raw → [-1, 1] (proprio_range)
        Stats: precomputed from dataset (mean, std per dimension)
```

---

## 4. Inference Pipeline

### 4.1 Two-Rate Inference Architecture

```
Semantic Rate (~12.5 Hz):
  Camera Image → Qwen2-VL-7B → MultiScaleAdapter → Grounder → GrounderOutput
  Stored in RuntimeCache.last_semantic

Control Rate (~50 Hz):
  Proprioception + prev_action + cached GrounderOutput
    → Tri-Rate Mamba Core (always runs, updates SSM state)
    → If need_new_chunk:
        → _build_cond_prefix → FlowActionExpert.sample(midpoint, 8 steps)
        → RTC blending (if enabled): linear interpolation with previous chunk tail
        → Cache chunk [B, 24, 14]
    → Return chunk[chunk_step] → single action [B, 14]

Chunk caching: Only regenerate when:
  (a) No cached chunk
  (b) chunk_step >= execution_horizon (8 steps consumed)
  (c) Semantic refresh occurred (new observation)
```

### 4.2 Full Inference Call Graph

```
HybridVLALiberoPolicy.from_checkpoint()
  ├── load model + EMA weights + normalizers + processor
  │
  ├── semantic_step_from_obs(obs, language)          @ 12.5 Hz
  │     obs_to_semantic_input() → tokenize image+text
  │     model.semantic_step() → GrounderOutput
  │
  └── control_step_from_obs(obs, runtime, grounding) @ 50 Hz
        obs_to_raw_proprio() → concat joint+gripper
        proprio_normalizer.normalize()
        model.control_step() → ControlStepOutput
          ├── temporal_core always runs (update SSM state)
          ├── action_expert.sample() (only when need_new_chunk)
          │     midpoint ODE: 8 steps × 2 forward = 16 expert forwards
          └── RTC blend + chunk cache
        action_normalizer.denormalize()
        clamp to action_range
```

---

## 5. Parameter Budget

### 5.1 Detailed Breakdown

| Module | Total Params | Trainable | % of Trainable |
|--------|-------------|-----------|----------------|
| **Qwen2-VL-7B base** | 7,615M | 0 | 0% |
| **Qwen2-VL LoRA** | ~100M | ~100M | 4.5% |
| **MultiScaleAdapter** | ~22M | ~22M | 1.0% |
| **CameraPositionEmbedding** | ~0.016M | ~0.016M | <0.1% |
| **Grounder (8 layers)** | ~200M | ~200M | 9.1% |
| **SlotCompression** | ~50M | ~50M | 2.3% |
| **Fast Mamba (20L)** | ~540M | ~540M | 24.6% |
| **Medium Mamba (6L)** | ~162M | ~162M | 7.4% |
| **Slow Mamba (10L)** | ~270M | ~270M | 12.3% |
| **CrossAttentionFusion** | ~50M | ~50M | 2.3% |
| **ActionHistoryEncoder (2L)** | **~1.6M** | **~1.6M** | **0.07%** |
| **StaleTimeEncoding** | ~0.5M | ~0.5M | <0.1% |
| **FlowActionExpert (18L)** | ~270M | ~270M | 12.3% |
| **FAST Head** | ~8M | ~8M | 0.4% |
| **Phase Head** | ~4M | ~4M | 0.2% |
| **Affordance Head** | ~4M | ~4M | 0.2% |
| **Projections (6 linears)** | ~50M | ~50M | 2.3% |
| **cond_builder** | ~17M | ~17M | 0.8% |
| **V2ConsistencyLoss** | ~0.1M | ~0.1M | <0.1% |
| **Total** | **~9,364M** | **~2,200M** | **100%** |

> v0.10.9 change: ActionHistoryEncoder 108M → 1.6M (reduced 67×). Trainable params 2,306M → 2,200M (saved ~106M = 4.6%).

### 5.2 Compute Distribution Per Forward

```
Component               FLOPs %    Memory %    Time %
Backbone (4 refreshes)    45%        40%        30-40%
Grounder (4 calls)        10%        10%        5-10%
Tri-Rate Core (24 steps)  30%        35%       30-40%     ← bottleneck (sequential layers)
Action Expert (1 call)     8%        10%        10-15%
Heads + Losses             2%         5%         5%
Communication (FSDP)       5%         —          5-10%
```

---

## 6. Design Commentary

### 6.1 Strengths

**1. Hierarchical temporal decomposition** — The tri-rate design (50/25/12.5 Hz) is well-motivated by the real-time control structure of robot manipulation. Fast reactions (collision avoidance) need high frequency; task planning needs long context. This avoids forcing a single model to serve both.

**2. Flow matching + discrete hybrid** — Using both continuous flow matching (expert) and discrete binning (FAST head) provides complementary gradients. Discrete head gives stable, dense gradients in Stage A; flow matching gives precise continuous actions in Stage B/C. This is the same insight behind pi-0's design.

**3. Stage-gated training** — The `cond_prefix.detach()` in Stage B is a crucial design choice. Without it, the expert's high-magnitude velocity MSE loss would override the weaker discrete/consistency losses and destabilize the already-converged representations. Stage C removes the detach only after the expert has converged.

**4. Cross-attention fusion** — Replacing v1's scalar gate with learned cross-attention fusion (queries attend to 3 stream tokens + stale encoding) is strictly more expressive. The fusion can learn content-dependent blending rather than fixed ratios.

**5. Multi-scale backbone features** — Extracting from layers [10, 18, 28] with learned gating captures both spatial detail (for grasping) and semantic understanding (for task reasoning) simultaneously.

### 6.2 Weaknesses & Risks

**1. Tri-Rate Core is the compute bottleneck** — With `mamba_impl="fallback"` (current default), the core processes sequences layer-by-layer with activation checkpointing — correct but uses JIT Python scan instead of fused CUDA kernels. With `mamba_impl="auto"` (official Mamba2), the token-by-token `step()` loop creates ~19,536 Python iterations per forward pass and bypasses checkpointing. The fallback path is currently the better choice for training (checkpointing works, vectorized forward), but lacks the CUDA acceleration of the official path.

**2. ActionHistoryEncoder (RESOLVED in v0.10.9)** — Previously 108M params for 112 input floats. Now reduced to ~1.6M (2-layer Mamba d=256 + output projection). Appropriately sized — no longer a concern.

**3. Phase/Affordance heads may train on noise** — Without explicit phase_labels and affordance_labels from the data adapter, these heads receive zero loss. Their tokens in `cond_prefix` (2 of 32) carry random information. This doesn't break training but wastes condition capacity.

**4. Consistency losses have collapse risk** — `ContrastiveTemporalLoss` and `ActionConsistencyLoss` both use L2-normalized vectors + cosine similarity. The model can satisfy both by outputting constant vectors. No variance regularization or asymmetric architecture (e.g., BYOL predictor) prevents this. The weight is low (0.3) so collapse won't destroy training, but the intended temporal structure learning may not occur.

**5. RTC training/inference mismatch** — Training generates `prev_chunk` using the SAME `cond_prefix` (with 0.01 noise), but inference generates it from a DIFFERENT semantic observation. The training signal teaches the model to be consistent with itself, not across observations.

**6. Expert only supervised at t=-1** — The flow matching loss applies only to the last temporal step (`batch["actions"][:, -1]`). The 24-step action chunks at steps 0-22 receive no expert-level continuous supervision. Only FAST's discrete loss covers all steps.

### 6.3 Architecture Comparison with Prior Art

| Design Choice | HybridVLA v2 | pi-0 / pi-0.5 | Octo | RT-2 |
|--------------|--------------|----------------|------|------|
| Backbone | Qwen2-VL-7B | PaliGemma / Gemma | Octo-Base (93M) | PaLI-X (55B) |
| Action representation | Flow matching | Flow matching | Diffusion | Discrete tokens |
| Temporal model | Tri-Rate Mamba (36L) | None (single-step) | Transformer | None |
| Discrete-continuous hybrid | Yes (FAST + FM) | Yes (similar) | No | No |
| Multi-camera | Yes (position embedding) | Yes | Yes | No |
| d_model core/expert | 2048 / 1536 | ~2048 / ~1024 | 768 | — |
| ODE solver | Midpoint (2nd order) | Euler | DDIM | — |
| Chunk horizon | 24 | 50 | 4 | 1 |

HybridVLA v2 is among the most complex architectures in this space. The tri-rate temporal processing is unique — no comparable system uses multi-rate SSM for robot control. This is both the main novelty and the main risk.

---

## 7. Data Flow Diagrams

### 7.1 Training Forward Pass (Stage B)

```
batch
  │
  ├── images + text ──→ Backbone × 4 refreshes ──→ Grounder × 4
  │                                                    │
  │                                                    ▼
  │                                              grounder_outputs[4]
  │                                                    │
  ├── proprio[t] ──→ proprio_proj ──────────────────→──┤
  ├── prev_actions[t] ──→ prev_action_proj ─────────→──┤
  ├── embodiment_id ──→ embodiment_embedding ───────→──┤
  ├── action_history ──→ ActionHistoryEncoder(1.6M) ──→──┤
  ├── steps_since ──→ StaleTimeEncoding ────────────→──┤
  │                                                    │
  │                   ┌────────────────────────────────┘
  │                   │ for t in range(24):
  │                   ▼
  │            Tri-Rate Mamba Core
  │            ├── fast_mamba(input_seq, state)
  │            ├── medium_mamba(input_seq, state)  [every 2nd step]
  │            ├── slow_mamba(input_seq, state)    [refresh only]
  │            └── fusion(fast, medium, slow, stale)
  │                   │
  │                   ▼
  │            fused_states [B, 24, 2048]
  │                   │
  │         ┌─────────┼──────────────────┐
  │         │         │                  │
  │         ▼         ▼                  ▼
  │    FAST Head   Phase Head      Affordance Head
  │    loss_fast   loss_phase      loss_affordance
  │
  ├── actions[:, -1] ──→ target_actions [B, 24, 14]
  │                           │
  │                      noise + flow_t
  │                           │
  │                      interpolate → noisy_actions
  │                           │
  │         cond_prefix.detach() ←── _build_cond_prefix(grounder, temporal)
  │                           │
  │                    FlowActionExpert
  │                    ├── action_proj + pos_emb + t_emb
  │                    ├── 18 layers (M-M-A × 6)
  │                    └── out_proj → velocity
  │                           │
  │                      loss_fm = MSE(velocity, target_velocity)
  │                           │
  │                      expert_denoised (for consistency)
  │                           │
  │                      loss_consistency
  │
  └── loss_total = Σ weighted losses
```

### 7.2 Condition Prefix Composition

```
_build_cond_prefix(grounder_out, temporal_out):

  Token 0:   grounder_out.global_token         [B, 1, 2048]
  Token 1-24: grounder_out.compressed_object_slots [B, 24, 2048]
  Token 25:  grounder_out.phase_token          [B, 1, 2048]
  Token 26:  grounder_out.uncertainty_token     [B, 1, 2048]
  Token 27:  grounder_out.affordance_token      [B, 1, 2048]
  Token 28:  temporal_out.fused_state           [B, 1, 2048]
  Token 29:  temporal_out.fast_token            [B, 1, 2048]
  Token 30:  temporal_out.medium_token          [B, 1, 2048]
  Token 31:  temporal_out.slow_token            [B, 1, 2048]
  ───────────────────────────────────────────────────────────
  Total:     [B, 32, 2048]

  → cond_builder: LayerNorm → Linear → GELU → Linear
  → core_to_expert: Linear(2048, 1536)
  → Final: [B, 32, 1536]  (fed to ExpertAttentionBlock as key/value)
```

---

## 8. Key Architectural Numbers

| Parameter | Value | Justification |
|-----------|-------|--------------|
| d_core | 2048 | Efficiency sweet spot for Mamba; matches standard transformer widths |
| d_expert | 1536 | Expert processes shorter sequences (26 tokens), smaller width suffices |
| chunk_horizon H | 24 | At 50Hz, 24 steps = 480ms lookahead; balances reactivity vs planning |
| execution_horizon | 8 | Execute 8 of 24 predicted steps before replanning (160ms) |
| compressed_slots | 24 | 24 object slots = matches cond_tokens for expert (24 of 32 tokens) |
| num_latents | 96 | Perceiver-style latent bottleneck; 96 balances compression vs capacity |
| LoRA rank | 64 | Higher than typical (16-32) to preserve backbone capacity with full-layer LoRA |
| Semantic refresh stride | 6 | Every 6 control steps = 120ms; VLM is too expensive for every step |
| K (action history) | 8 | 160ms at 50Hz; short enough for real-time, long enough for velocity |
| d_inner (history encoder) | 256 | v0.10.9: right-sized for 112-float input; 1.6M vs prior 108M |
| Mamba expand | 2 | Standard Mamba-2 expansion factor; d_inner = 2 × d_model |
| res_scale init | 1/sqrt(2N) | Prevents activation explosion in 20-layer deep Mamba stacks |

---

## 9. Summary

HybridVLA v2 is an ambitious architecture that combines a 7.6B VLM backbone, a novel tri-rate SSM temporal processor, and a flow-matching action expert into a unified system for robot manipulation. The three-stage training pipeline is carefully designed to prevent gradient conflicts between semantic understanding and action generation.

**Core innovation**: The tri-rate temporal decomposition (Fast/Medium/Slow at 50/25/12.5 Hz) is unique in the VLA landscape. It provides an inductive bias that matches the multi-timescale nature of real robot control.

**Main technical risk**: The system's complexity (~9.4B params, 5 major modules, 3 training stages, 7 loss functions) creates a large surface area for integration bugs — as demonstrated by the FSDP issues found in v0.10.9 reviews. The tri-rate core's token-by-token loop (official path) or sequential-layer processing (fallback path) is the primary training throughput bottleneck.

**v0.10.9 improvement**: ActionHistoryEncoder reduced from 108M to 1.6M params (67x), eliminating the most egregious over-parameterization. Trainable budget is now ~2.2B with better allocation.

**Readiness**: With FSDP fixes verified and ActionHistoryEncoder right-sized, the system is mechanically ready for training. The remaining architectural concerns (consistency loss collapse risk, phase/affordance label availability, RTC train/infer mismatch) affect auxiliary signal quality but are not blockers for a first training run.

---

## 10. 中文注释

### 10.1 系统总览

HybridVLA v2 是一个面向机器人操控的视觉-语言-动作（VLA）模型，总参数量约 94 亿（其中 76 亿为冻结的 Qwen2-VL-7B 骨干网络，约 22 亿为可训练参数）。系统分为两条路径：

- **语义路径**（~12.5 Hz）：视觉语言骨干网络 + 层级注意力 Grounder，负责从图像和语言指令中提取场景理解、物体识别、任务阶段等高层语义信息。计算量大，因此低频运行。
- **控制路径**（~50 Hz）：三频率 Mamba 时序核心 + 流匹配动作专家，负责将语义信息转化为具体的机械臂关节动作。轻量级，高频运行以满足实时控制需求。

### 10.2 各模块中文说明

#### 模块 1：Qwen2-VL-7B 骨干网络（~76 亿参数，其中 LoRA ~1 亿可训练）

- **作用**：从摄像头图像和语言指令中提取视觉-语言联合特征
- **关键设计**：
  - 视觉塔（ViT）完全冻结，文本层 0-15 冻结，仅层 16-27 注入 LoRA 适配器（rank=64）
  - 多尺度特征提取：从第 10/18/28 层分别抽取空间细节/中间特征/语义特征，通过学习门控融合为 2048 维输出
  - 多摄像头支持：通过 CameraPositionEmbedding 为不同摄像头的视觉 token 添加可学习位置编码

#### 模块 2：层级注意力 Grounder（~2 亿参数）

- **作用**：将骨干网络的长序列特征压缩为结构化的语义 token（场景摘要、物体槽位、任务阶段等）
- **关键设计**：
  - 96 个可学习潜在查询（latent queries），布局为：全局(1) + 物体(48) + 阶段(1) + 不确定性(1) + 可供性(1) + 辅助(44)
  - 层 0-3：96 个潜在查询与骨干网络特征做交叉注意力
  - 槽位压缩：第 4 层后，24 个路由查询将 48 个原始物体槽位压缩为 24 个精炼槽位
  - 层 4-7：继续处理压缩后的 72 个潜在变量
- **注意**：phase_token 和 affordance_token 需要数据适配器提供标签（phase_labels / affordance_labels）。当前 LIBERO 数据适配器不产生这些标签，这些 token 在训练中实际是随机噪声

#### 模块 3：三频率 Mamba 时序核心（~9.72 亿参数）

- **作用**：在不同时间尺度上处理时序信息，生成融合状态表征用于动作生成
- **三个流**：
  - **快速流**（20 层，d_state=128）：每步更新（50 Hz），捕捉即时本体感觉反应和动作变化，如碰撞避障
  - **中速流**（6 层，d_state=128）：每 2 步更新（25 Hz），捕捉短期运动规划
  - **慢速流**（10 层，d_state=256）：仅在语义刷新时更新（12.5 Hz），捕捉长期任务结构和目标上下文。d_state=256（其他流的 2 倍）以保持更长的记忆
- **交叉注意力融合**（2 层，8 头）：学习查询向量对三个流的 token 做注意力加权融合，输出 fused_state [B, 2048]
- **输入序列组成**（33 个 token）：
  - 9 个单例 token：全局、阶段、不确定性、可供性、本体感觉、上一步动作、陈旧度编码、实体类型、动作历史摘要
  - 24 个压缩物体槽位（来自 Grounder）

#### ActionHistoryEncoder（~160 万参数，v0.10.9 从 1.08 亿缩减）

- **作用**：将最近 8 步动作历史 [B, 8, 14] 编码为单个摘要 token [B, 2048]
- **架构**：Linear(14→256) → 2 层 Mamba(d=256, d_state=64) → Linear(256→2048)
- **设计考量**：输入仅 112 个浮点数（8×14），使用小维度 Mamba 保留时序建模能力（因果扫描捕捉速度/加速度模式），同时避免过参数化。与 prev_action_token（仅编码上一步）互补——前者提供轨迹上下文，后者提供即时状态

#### 模块 4：流匹配动作专家（~2.7 亿参数）

- **作用**：通过流匹配（Flow Matching）生成连续的 24 步动作轨迹
- **关键设计**：
  - 18 层混合架构：Mamba-Mamba-Attention × 6（12 个 Mamba 块 + 6 个 Attention 块）
  - AdaRMSNorm（来自 pi-0.5）：用流时间步 t 条件化归一化层，使网络区分早期（噪声大）和晚期（噪声小）的去噪阶段。门控偏置初始化为 +2 防止 18 层残差链的激活衰减
  - 条件前缀（32 token）：包含 Grounder 输出 + 时序核心输出，通过 ExpertAttentionBlock 的交叉注意力注入
  - 推理采样：中点法 ODE 求解器（二阶精度），8 步采样（16 次专家前向传播）
  - d_model=1536（小于核心的 2048）：专家处理较短序列（26 token），较小宽度即可满足需求
- **噪声调度**：Logit-normal 分布 t ~ σ(N(0,1))，偏向 t≈0.5 附近采样
- **插值公式**：x_t = (1-t)·x_0 + t·x_1（Rectified Flow 线性插值）

#### 模块 5：离散头（~1600 万参数）

- **FAST 离散头**（~800 万）：将 fused_state 映射为 [B, 24, 14, 512] 的分类 logits（24 步 × 14 维 × 512 个离散 bin）。Stage A 中专家冻结时，这是唯一的动作级监督信号
- **阶段头**（~400 万）：预测 16 类任务阶段（如接近、抓取、放置等）
- **可供性头**（~400 万）：预测 8 类操控类型（如抓、推、拉、放等）

### 10.3 三阶段训练策略

| 阶段 | 步数 | 可训练模块 | 冻结模块 | 核心 Loss | 关键机制 |
|------|------|-----------|---------|----------|---------|
| **A** | 120K | LoRA、Grounder、三频率核心、离散头、投影层 | 骨干网络基础权重、动作专家 | FAST 离散 + 阶段 + 可供性 + 一致性 | 专家完全冻结；FAST 头提供唯一的动作监督信号 |
| **B** | 200K | 上述 + 动作专家 (lr×0.5) | 骨干网络基础权重 | 上述 + 流匹配 | `cond_prefix.detach()` 阻断 FM 梯度回传到骨干/Grounder/核心，防止专家的高幅度梯度破坏已收敛的语义表征 |
| **C** | 80K | 全部可训练 | 骨干网络基础权重 | 上述 + RTC + FASTER | 全端到端微调；FM 梯度可回传至 LoRA；RTC 教授 chunk 边界平滑性；FASTER 加权近期动作 |

**跨阶段检查点加载顺序**：
1. 加载上一阶段的 model.pt（`strict=False` 允许新参数）
2. 初始化 EMA（shadow 复制已恢复的权重）
3. 包裹 FSDP

**梯度流向详解**：
- Stage A：`loss_fast → FAST Head → fused_state → 三频率核心 ← Grounder ← Backbone(仅 LoRA)`
- Stage B：`loss_fm → Expert ←⊘ cond_prefix.detach() ← [Grounder + 核心]`（Expert 在冻结表征上独立训练）
- Stage C：`loss_fm → Expert → cond_prefix → [Grounder + 核心 + Backbone]`（全链路端到端优化）

### 10.4 推理流程

```
语义步骤（~12.5 Hz，每 80ms 执行一次）：
  摄像头图像 + 语言指令 → Qwen2-VL-7B → 多尺度适配器 → Grounder → GrounderOutput
  缓存到 RuntimeCache

控制步骤（~50 Hz，每 20ms 执行一次）：
  本体感觉 + 上一步动作 + 缓存的 GrounderOutput
    → 三频率 Mamba 核心（始终运行，更新 SSM 状态）
    → 若需要新 chunk（首次/chunk 用完/语义刷新）：
        → 构建条件前缀 → 动作专家采样（中点法 ODE，8步）
        → RTC 混合（与上一 chunk 尾部线性插值）
        → 缓存 chunk [B, 24, 14]
    → 返回 chunk[当前步] → 单步动作 [B, 14]
    → 反归一化 → clamp → 发送到机器人

Chunk 缓存策略：执行 8 步后重新规划（160ms 执行窗口 / 480ms 预测范围）
```

### 10.5 条件前缀组成（cond_prefix，32 token）

| Token 位置 | 来源 | 语义 |
|-----------|------|------|
| 0 | grounder.global_token | 场景级全局摘要 |
| 1-24 | grounder.compressed_object_slots | 24 个精炼物体槽位（场景中的物体表征） |
| 25 | grounder.phase_token | 任务阶段编码（当前无标签，实际为噪声） |
| 26 | grounder.uncertainty_token | 认知不确定性编码 |
| 27 | grounder.affordance_token | 可供性类型编码（当前无标签，实际为噪声） |
| 28 | temporal.fused_state | 三频率融合状态（核心输出） |
| 29 | temporal.fast_token | 快速流输出（即时反应） |
| 30 | temporal.medium_token | 中速流输出（短期规划） |
| 31 | temporal.slow_token | 慢速流输出（长期目标） |

经过 `cond_builder`（LayerNorm → Linear → GELU → Linear）处理后，投影到专家维度 1536，作为 ExpertAttentionBlock 交叉注意力的 key/value。

### 10.6 Loss 函数中文说明

| Loss | 函数 | 权重 | 说明 |
|------|------|------|------|
| `loss_fast` | 交叉熵（512 bin, 标签平滑=0.1） | 1.0 | 离散动作预测，所有 T 步监督，Stage A 的主要训练信号 |
| `loss_phase` | 交叉熵（16 类） | 0.5 | 任务阶段分类，需要 phase_labels |
| `loss_affordance` | 交叉熵（8 类） | 0.3 | 可供性类型分类，需要 affordance_labels |
| `loss_fm` | MSE(预测速度, 目标速度) | 1.0 | 流匹配损失，仅最后一步(t=-1)，Stage B/C |
| `loss_rtc` | MSE(当前头, 前一尾) + 加速度平滑 | 0.3 | 重叠块一致性（Readout-Time Consistency），Stage C |
| `loss_faster` | MSE(去噪近端, 目标近端) | 0.2 | 近期动作加权（FASTER），Stage C |
| `loss_consistency` | InfoNCE + 慢快 MSE + 动作余弦 | 0.3 | 时序一致性（对比学习 + 慢速流跟随快速流 + 离散/连续动作对齐） |

### 10.7 关键设计决策及其理由

| 决策 | 选择 | 理由 |
|------|------|------|
| 为什么用三频率而非单频率？ | Fast/Medium/Slow 分开处理 | 机器人控制本质上是多时间尺度的：碰撞避障需要毫秒级反应，任务规划需要秒级上下文 |
| 为什么 Stage B 用 `cond_prefix.detach()`？ | 阻断 FM 梯度回传 | FM loss 的梯度幅度远大于离散 loss，若不阻断会破坏 Stage A 已收敛的语义表征 |
| 为什么同时用离散头和流匹配？ | FAST + FlowMatching 互补 | 离散头在 Stage A 提供稳定、密集的梯度；流匹配在 Stage B/C 提供精确连续动作 |
| 为什么 d_expert=1536 小于 d_core=2048？ | 成本/质量权衡 | 专家处理 26 token 的短序列，较小宽度足够；节省约 30% 专家参数 |
| 为什么用中点法而非 Euler？ | 二阶 ODE 求解 | 同样 8 步采样，中点法精度约为 Euler 的 2 倍，代价是每步 2 次前向传播 |
| 为什么 ActionHistoryEncoder 用 d=256 而非 d=2048？ | 输入仅 112 浮点数 | 108M 参数处理 112 个浮点数严重过参数化（96.6 万参数/浮点数），缩减至 1.6M 更合理 |
| 为什么 LoRA rank=64 而非常见的 16？ | 全层 LoRA 需要更高秩 | 28 层全部注入 LoRA，较高秩保证骨干网络有足够的适应能力 |
| 为什么 chunk_horizon=24 而非更短？ | 480ms 预测范围 | 平衡反应性（执行 8 步=160ms 就重新规划）和规划范围（看到 480ms 的未来） |

### 10.8 已知风险与待解决问题

1. **一致性 Loss 可能坍塌**：ContrastiveTemporalLoss 和 ActionConsistencyLoss 使用 L2 归一化后的余弦相似度，模型可通过输出常向量达到低 loss。权重仅 0.3，不会破坏训练但时序结构学习可能失效
2. **Phase/Affordance 无标签训练**：当前数据适配器不提供这些标签，对应头接收零 loss，其在 cond_prefix 中的 token 是噪声（占 32 个中的 2 个）
3. **RTC 训练/推理不匹配**：训练中 prev_chunk 基于相同 cond_prefix（加微小噪声），推理中基于不同时刻的观测。训练信号教的是"自洽"而非"跨观测一致"
4. **专家仅在 t=-1 监督**：流匹配 loss 仅应用于最后一个时间步，前 23 步的动作 chunk 无专家级连续监督（仅有 FAST 离散监督）
5. **三频率核心的计算效率**：fallback 路径支持 activation checkpointing 但使用 JIT Python 扫描；official 路径有 CUDA 加速但绕过 checkpointing 且产生 ~19,536 次 Python 循环
