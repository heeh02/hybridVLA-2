# HybridVLA v2 Final Architecture & Training Design Expert Review

**Reviewer Perspective**: AI Architecture & Training Design Expert
**Scope**: Full-stack review of architecture correctness, training stability, scalability, and design coherence
**Codebase Version**: Post-v0.7 recovery (all prior correctness bugs resolved)

---

## Table of Contents

1. [Architecture Overview & Design Coherence](#1-architecture-overview--design-coherence)
2. [Vision Backbone (Qwen2-VL-7B)](#2-vision-backbone-qwen2-vl-7b)
3. [Hierarchical Attention Grounder](#3-hierarchical-attention-grounder)
4. [Tri-Rate Mamba Core](#4-tri-rate-mamba-core)
5. [Flow Action Expert](#5-flow-action-expert)
6. [World Model (Imagination Engine)](#6-world-model-imagination-engine)
7. [Loss Design & Gradient Flow Analysis](#7-loss-design--gradient-flow-analysis)
8. [Training Pipeline Design](#8-training-pipeline-design)
9. [Inference Pipeline](#9-inference-pipeline)
10. [Scalability & Memory Analysis](#10-scalability--memory-analysis)
11. [Critical Issues Found](#11-critical-issues-found)
12. [Moderate Issues Found](#12-moderate-issues-found)
13. [Minor Issues & Recommendations](#13-minor-issues--recommendations)
14. [Final Verdict](#14-final-verdict)

---

## 1. Architecture Overview & Design Coherence

### 1.1 System Topology

```
                    ┌─────────────────────────────────────────┐
                    │           Qwen2-VL-7B Backbone           │
                    │   (LoRA r=64, multi-scale [10,18,28])    │
                    │     3584d → MultiScaleAdapter → 2048d    │
                    └─────────────┬───────────────────────────┘
                                  │ [B, N, 2048]
                    ┌─────────────▼───────────────────────────┐
                    │   Hierarchical Attention Grounder        │
                    │   96 latents → 8 blocks → compression   │
                    │   → global/phase/unc/aff/24 slots        │
                    └─────────────┬───────────────────────────┘
                                  │
        ┌──────────┬──────────────┼──────────────┬────────────┐
        │          │              │              │             │
   ┌────▼───┐ ┌───▼───┐   ┌─────▼──────┐  ┌───▼───┐   ┌────▼────┐
   │ Fast   │ │Medium │   │CrossAttn   │  │Slow   │   │ Action  │
   │ 20L    │ │ 6L    │   │ Fusion     │  │ 10L   │   │ History │
   │d_s=128 │ │d_s=128│   │ 2-layer    │  │d_s=256│   │ 4L      │
   │ 50 Hz  │ │ 25 Hz │   │            │  │12.5Hz │   │ Mamba   │
   └────┬───┘ └───┬───┘   └─────┬──────┘  └───┬───┘   └────┬────┘
        └──────────┴─────────────┤─────────────┘            │
                                 │ fused_state [B, 2048]     │
                    ┌────────────▼───────────────────────────┘
                    │         Condition Prefix [B, 32, 2048]
                    │              ↓ project to 1536d
                    │   ┌─────────▼──────────────────────┐
                    │   │   Flow Action Expert            │
                    │   │   18L M-M-A×6, 1536d            │
                    │   │   AdaRMSNorm + Midpoint ODE     │
                    │   └─────────┬──────────────────────┘
                    │             │
              ┌─────▼─────┐  ┌───▼────────┐
              │FAST 512-bin│  │Flow Matching│
              │Discrete   │  │Continuous   │
              └───────────┘  └────────────┘
```

### 1.2 Design Coherence Assessment

The architecture follows a well-motivated multi-resolution temporal processing paradigm. The key innovation lies in the tri-rate decomposition: separating reactive (50 Hz), tactical (25 Hz), and strategic (12.5 Hz) processing into distinct Mamba stacks with appropriate state capacities. This maps cleanly to robotics control theory, where different decision loops operate at different frequencies.

**Design strengths**:
- Clean modular decomposition with well-defined interfaces (`GrounderOutput`, `TemporalOutput`, `ActionExpertOutput`)
- Stage-gated training prevents co-adaptation collapse between representation and action modules
- Dual action heads (discrete + continuous) with consistency regularization is state-of-the-art
- Condition prefix as the interface between core and expert enables dimension change (2048→1536)

**Design coherence issues**:
- The world model subsystem is architecturally present but **not connected** to the training loop
- RTC (Receding Temporal Context) and FASTER strategies are configured but **not implemented**
- Multi-camera support is configured but the forward pass uses **single-view** data only

---

## 2. Vision Backbone (Qwen2-VL-7B)

**File**: `vla_hybrid_v2/models/qwen2vl_backbone.py` (199 lines)

### 2.1 Architecture Choices

| Decision | Choice | Assessment |
|----------|--------|------------|
| Base model | Qwen2-VL-7B-Instruct | Strong: native vision-language, 3584d hidden |
| Adaptation | LoRA r=64, α=128 on all 28 layers | Appropriate: high rank for robotics domain gap |
| Multi-scale | FPN-style fusion of layers [10, 18, 28] | Good: captures low/mid/high semantic features |
| Projection | 3584d → 2048d via learned gating | Clean: learned weighted sum with softmax gate |
| Freeze | Vision tower + text layers 0-15 | Standard: preserves pre-trained representations |

### 2.2 Multi-Scale Adapter Design

```python
# Per-scale: LayerNorm → Linear → gate_proj (pooled)
# Fusion: softmax(gates) weighted sum
```

The `MultiScaleAdapter` computes per-scale gate weights via mean-pooled linear projections, then softmax-normalizes them. This is a Squeeze-and-Excitation-style approach applied to FPN scales.

**Concern**: The gate pooling is `mean(proj(ln(h)), dim=1)`, which collapses spatial information. For object-level downstream use (grounder), spatial-aware gating (per-token rather than per-scale) could be more informative. However, per-token gating would increase parameters and computation proportionally to sequence length. The current design is a reasonable trade-off.

### 2.3 LoRA on All Layers

The v2 design applies LoRA to **all 28 layers**, including frozen layers 0-15. This means frozen layers remain partially trainable through their LoRA adapters. This is intentional and correct: the base weights are frozen (preserving LLM knowledge), but LoRA adaptors learn domain-specific adjustments (robotics).

**Note**: With rank=64 and 7 target modules per layer, each layer adds 2 × 64 × 3584 × 7 ≈ 3.2M LoRA parameters. Across 28 layers: ~90M LoRA parameters. This is substantial but appropriate given the domain gap from language to robotics.

---

## 3. Hierarchical Attention Grounder

**File**: `vla_hybrid_v2/models/attention_grounder.py` (261 lines)

### 3.1 Architecture

```
96 learned latent queries → 8 GrounderBlocks (cross-attn + self-attn)
  ├── Layer 0-3: Full 96 latents attend to backbone features
  ├── Compression: 48 object slots → SlotCompression → 24 compressed slots
  └── Layer 4-7: Reduced 72 latents continue processing
```

**Layout**: `[global(1), objects(48→24), phase(1), uncertainty(1), affordance(1), aux(44)] = 96→72`

### 3.2 Design Assessment

**Strengths**:
- Perceiver-style cross-attention bottleneck is computationally efficient
- Mid-layer compression (48→24 slots) is a clever FPN-to-slot bridge
- Separate semantic tokens (phase, uncertainty, affordance) provide structured representations

**Issues Found**:

**(A) Auxiliary tokens never used (44 aux tokens)**

The layout allocates 44 auxiliary tokens, but `GrounderOutput` only extracts `global_token`, `compressed_object_slots`, `phase_token`, `uncertainty_token`, and `affordance_token`. The 44 auxiliary latents participate in self-attention (potentially helping information flow) but their final values are **discarded**. They consume:
- 44 × 8 × (cross-attn + self-attn + FFN) = significant FLOP overhead per block
- 44 × 2048 × 8 blocks of activation memory

These tokens may serve as an implicit "scratchpad" for information routing during self-attention, which is a valid design pattern (similar to register tokens in ViT-22B). However, 44 is excessive — 8-16 register tokens would suffice for this purpose, saving ~30% of grounder FLOPs.

**(B) SlotCompression quality**

The slot compression uses learned routing queries (24) that cross-attend to raw slots (48), then self-attend. This is architecturally sound (similar to Set Transformer's ISAB). However, the compression happens at a **fixed layer** (layer 4), and the routing queries are **not conditioned** on the task or language instruction. Task-conditioned routing (e.g., which objects matter for "pick up the red cup") could significantly improve downstream performance.

---

## 4. Tri-Rate Mamba Core

**File**: `vla_hybrid_v2/models/mamba_core.py` (772 lines)

### 4.1 Stream Architecture

| Stream | Layers | d_state | Update Rate | Role | Params (est.) |
|--------|--------|---------|-------------|------|---------------|
| Fast   | 20     | 128     | Every step (50 Hz) | Reactive control | ~320M |
| Medium | 6      | 128     | Every 2nd step (25 Hz) | Tactical planning | ~96M |
| Slow   | 10     | 256     | Semantic refresh (12.5 Hz) | Strategic reasoning | ~170M |
| Total  | 36     | —       | — | — | ~586M |

### 4.2 Critical Performance Issue: Token-by-Token Processing

**Severity**: HIGH (Performance)

**Location**: `_MambaStack.forward()`, lines 418-440

```python
if uses_official:
    # Token-by-token step() to capture per-layer states
    for t in range(x.shape[1]):       # L = 33 tokens
        x_t = x[:, t, :]
        for i, layer in enumerate(self.layers):   # 20/6/10 layers
            x_t, ssm_states_list[i], conv_states_list[i] = layer.step(...)
```

When using the official Mamba2 CUDA path, the entire sequence is processed **token-by-token** through all layers. This defeats the purpose of Mamba2's fused CUDA kernel, which achieves its speed through chunked parallel scan (SSD algorithm with chunk_size=256).

**Impact quantification**: For the Fast stream (20 layers, L=33 tokens):
- Token-by-token: 20 × 33 = 660 sequential `.step()` Python calls per control step
- Fused forward: 1 CUDA kernel call with internal parallelism
- Per temporal window (24 control steps): 24 × 660 = **15,840** step calls just for Fast

Combined across all 3 streams per window: 24 × (20+6+10) × 33 = **28,512** step calls.

This is the single largest performance bottleneck in the system. The v0.5 fix chose correctness over performance (Mamba2.forward() doesn't expose final states), but a better solution exists:

**Recommended fix**: Use `Mamba2.forward()` for intra-sequence parallelism, then run a single `.step()` pass on the **last token only** to capture the final state for temporal carry:

```python
if uses_official:
    # Fused forward for full sequence (fast)
    out = x.clone()
    for layer in self.layers:
        out, _, _ = layer.forward(out)  # uses _forward_official with fused CUDA

    # Single step on last token to capture state (for temporal carry)
    x_last = x[:, -1, :]
    for i, layer in enumerate(self.layers):
        x_last, ssm_states_list[i], conv_states_list[i] = layer.step(
            x_last, ssm_states_list[i], conv_states_list[i]
        )
    return out, ssm_states_list, conv_states_list
```

This reduces step calls from 28,512 to 864 per window (only last-token state capture), providing ~33× speedup while maintaining state correctness.

**Caveat**: The fused forward and step paths may produce slightly different outputs (numerical precision), and the saved state would correspond to only the last token's context. If exact state tracking across all tokens is required, the current approach is necessary but should be optimized with custom CUDA kernels.

### 4.3 Double Pre-Normalization at Layer 0

**Severity**: LOW (Architectural cleanliness)

The `TriRateMambaCore.forward()` applies stream-level LayerNorm before passing to the Mamba stack:

```python
fast_out, ... = self.fast_mamba(
    self.fast_input_norm(input_seq),  # LayerNorm #1
    ...
)
```

Each `MambaBlock` then applies its own LayerNorm (v0.7 fix):

```python
def _forward_official(self, x):
    residual = x
    out = self.mamba(self.norm(x))  # LayerNorm #2 (at layer 0)
    ...
```

At layer 0, the input undergoes **two consecutive LayerNorm** operations. Consecutive LayerNorms are mathematically **not idempotent** in general (LN(LN(x)) ≠ LN(x) because the first LN changes the statistics), but in practice the effect is minor because the first LN produces well-conditioned input for the second.

**Recommendation**: Remove `fast_input_norm`, `medium_input_norm`, `slow_input_norm` from `TriRateMambaCore` since each `MambaBlock` already normalizes its input. This simplifies the architecture and avoids redundant computation.

### 4.4 Cross-Attention Fusion

The fusion module uses `nn.MultiheadAttention` (PyTorch native) rather than `F.scaled_dot_product_attention`. The native MHA doesn't automatically dispatch to FlashAttention. Since the sequence length is tiny (3 key-value tokens), this has negligible performance impact, but using SDPA would be more consistent with the rest of the codebase.

The stale-time conditioning via additive projection on the key-value tokens is a clean design: `kv = kv + stale_proj(stale_token).unsqueeze(1)`. This enables the fusion to be aware of information freshness without explicit attention masking.

### 4.5 Input Sequence Composition

```python
singles = [global, phase, unc, aff, proprio, prev_action, stale, embodiment, action_history]  # 9 tokens
input_seq = cat([singles, compressed_object_slots])  # [B, 33, D]
```

The ordering matters for the causal conv1d in Mamba. Semantic tokens (global, phase, etc.) come first, followed by object slots. This means Mamba's conv1d window at the first few time steps sees semantic context before objects, which is architecturally appropriate for top-down visual reasoning.

---

## 5. Flow Action Expert

**File**: `vla_hybrid_v2/models/flow_action_expert.py` (340 lines)

### 5.1 Architecture

```
18 layers: [M, M, A] × 6
d_model=1536, num_heads=24 (head_dim=64), d_state=96
AdaRMSNorm conditioned on flow timestep
Midpoint ODE solver (2nd-order) for inference
```

### 5.2 AdaRMSNorm Design

```python
gate.sigmoid() * (RMSNorm(x) * (1 + scale) + shift)
```

This follows the pi-0.5 (Physical Intelligence) adaptive normalization pattern. The gate mechanism allows the network to completely suppress certain dimensions at specific noise levels, which is critical for denoising quality.

**Initialization concern**: The `cond_proj = nn.Linear(cond_dim, 3 * dim)` uses default Kaiming uniform initialization. At init, `gate ≈ 0` → `sigmoid(gate) ≈ 0.5`, so the output magnitude is halved. For a 18-layer network with residual connections, this compounding halving could lead to vanishing activations in early training. Consider initializing `gate` biases to `+2` so that `sigmoid(2) ≈ 0.88`, keeping early-training outputs closer to unit magnitude.

### 5.3 Expert Mamba Block Correctness

Unlike the core `MambaBlock`, the `ExpertMambaBlock` correctly implements pre-norm + residual in **all** code paths:

```python
def forward(self, x, t_cond):
    residual = x
    x = self.norm(x, t_cond)     # AdaRMSNorm (always applied)
    ...
    return residual + self.out_proj(y)  # residual (always applied)
```

The CUDA path uses `selective_scan_fn` directly (from `mamba_ssm.ops.selective_scan_interface`), which is the Mamba-1 selective scan kernel. This is correct for the expert because:
1. The expert processes each chunk independently (no cross-step state needed)
2. The `selective_scan_fn` with `delta_softplus=True` and `D`/`z` gate matches the Mamba-1 architecture

**Note**: The expert uses Mamba-1 architecture (explicit dt_rank projection, `y * SiLU(z)` gating) while the core uses Mamba-2 (official `Mamba2` block with chunked SSD). This is a deliberate design choice — the expert doesn't need cross-step state persistence, so Mamba-1 with `selective_scan_fn` is more straightforward.

### 5.4 ODE Solver Analysis

**Euler**: `x_{i+1} = x_i + dt × v(x_i, t_i)` — 1 forward per step, 1st-order
**Midpoint**: `x_{mid} = x_i + 0.5dt × v(x_i, t_i); x_{i+1} = x_i + dt × v(x_mid, t_mid)` — 2 forwards per step, 2nd-order

With `num_steps=8`:
- Euler: 8 forward passes, ~1st-order accuracy
- Midpoint: 16 forward passes, ~2nd-order accuracy (√error improvement)

The midpoint solver provides significant quality improvement at 2× cost. For real-time control at 50 Hz, the 16 forward passes through 18 expert layers must complete within 20ms. On H100 with 1536d model, this is feasible but tight.

**Alternative**: Consider adaptive step-size solvers (Dormand-Prince RK45) or distillation to reduce steps.

### 5.5 Condition Prefix → Expert Interface

The condition prefix ([B, 32, 2048]) is projected to [B, 32, 1536] via `core_to_expert` linear. The expert's cross-attention layers attend to this prefix. The 32 tokens provide: 1 global + 24 slots + 1 phase + 1 uncertainty + 1 affordance + 1 fused + 1 fast + 1 medium + 1 slow.

The prefix is rich enough to convey the full scene understanding to the expert. However, the expert's cross-attention keys/values use standard `nn.LayerNorm` (not AdaRMSNorm), which is correct — the condition prefix shouldn't be modulated by the flow timestep.

---

## 6. World Model (Imagination Engine)

**Files**: `vla_hybrid_v2/world_model/` (9 files, ~1040 lines total)

### 6.1 Architecture

```
z_full = [z_det (2048d) ; z_sto (2048d)] = 4096d

Imagination step:
  z_noisy = NoiseAugmentation(z_full, step)
  δz = ImaginationMamba(z_noisy, action, noise_emb)  [8-layer Mamba-2]
  z_det_next = z_det + δz
  z_full_next = cat(z_det_next, StochasticState.prior(z_det_next))
  → WorldModelHeads(z_full_next) → reward/value/done
  → ObjectPhysicsEngine(slots, action) → next_slots
```

### 6.2 Critical Issue: World Model Not Connected to Training

**Severity**: HIGH (Incomplete Feature)

The `ImaginationEngine` and `WorldModelLoss` are instantiated in `HybridVLAv2.__init__()` when `wmcfg.enable=True`, but **never called** in `forward_train()`. The method `get_world_model_state()` exists but is never invoked.

```python
# In __init__:
self.imagination_engine = ImaginationEngine(...)  # Created
self.world_model_loss_fn = WorldModelLoss(...)    # Created

# In forward_train:
# ... no reference to imagination_engine or world_model_loss_fn
```

**Impact**: The world model infrastructure occupies ~170M parameters of GPU memory but contributes zero training signal. If `wmcfg.enable=True` (currently `False` by default), these parameters are loaded but waste memory.

**Required integration**:
```python
# In forward_train(), after temporal processing:
if self.imagination_engine is not None and stage == "c":
    wm_state = self.get_world_model_state(grounder_out, temporal_out)
    trajectory = self.imagination_engine.rollout(wm_state["z_det"], policy=...)
    wm_losses = self.world_model_loss_fn(
        posterior_logits=..., prior_logits=trajectory.prior_logits, ...
    )
    losses.update({f"wm_{k}": v for k, v in wm_losses.items()})
```

### 6.3 Component-Level Assessment

| Component | Lines | Params | Design | Status |
|-----------|-------|--------|--------|--------|
| StochasticStateModule | 98 | ~30M | DreamerV3 48×48 categorical, correct | OK |
| ImaginationMamba | 117 | ~80M | 8-layer Mamba-2 via `.step()`, correct | OK |
| ObjectPhysicsEngine | 153 | ~35M | 6-layer attention GNN, inertia bias | OK |
| NoiseAugmentation | 80 | ~5M | GameNGen linear schedule, 16 buckets | OK |
| WorldModelHeads | 116 | ~15M | SymlogTwoHot regression, correct | OK |
| WorldModelLoss | 196 | ~0 | Per-category free bits KL, correct (v0.4 fix) | OK |
| CNNWorldDecoder | 90 | ~40M | 4-stage ConvTranspose2d, 7→112 | OK |
| LatentSubgoalPlanner | 41 | ~20M | Residual MLP, z_full + phase + language | OK |

### 6.4 Imagination Mamba State Persistence

The `ImaginationMamba` correctly uses `MambaBlock.step()` for single-token recurrence across 32 imagination steps. After the v0.7 fix, each step correctly applies pre-norm + residual. The 8-layer Mamba-2 with d_state=128 provides sufficient capacity for modeling world dynamics over a 32-step horizon.

**Concern**: The `input_proj` concatenates `[z_full (4096d), a_emb (2048d), noise_emb (2048d)]` = 8192d → linear → 2048d. This 4:1 compression at the input could bottleneck information flow. Consider using a 2-layer MLP with intermediate dimension.

---

## 7. Loss Design & Gradient Flow Analysis

### 7.1 Loss Landscape

| Loss | Weight | Source | Gradient Flows To |
|------|--------|--------|-------------------|
| `loss_fast` (Discrete CE) | 1.0 | FAST head logits vs discretized GT | Core → Grounder → Backbone LoRA |
| `loss_phase` (CE) | 0.5 | Phase head vs phase labels | Grounder → Backbone LoRA |
| `loss_affordance` (CE) | 0.3 | Affordance head vs labels | Grounder → Backbone LoRA |
| `loss_fm` (Flow Matching MSE) | 1.0 | Expert velocity vs (x_1 - x_0) | Expert only (Stage B: detached prefix) |
| `loss_consistency` | 0.3 | Contrastive + SlowFast + Action | Core + Heads |

### 7.2 Gradient Flow Diagram (Stage B)

```
Backbone ←(LoRA)── Grounder ←── loss_fast, loss_phase, loss_affordance
                        │
                        ▼
                   Tri-Rate Core ←── loss_fast, loss_consistency
                        │
                        │ detach()           ← knowledge insulation
                        ▼
              Condition Prefix (no grad)
                        │
                        ▼
               Flow Action Expert ←── loss_fm
```

The `cond_prefix.detach()` in Stage B is critical: it prevents flow matching gradients from corrupting the backbone/grounder representations that are still learning from discrete labels. This is the "knowledge insulation" pattern from the pi-0 paper.

### 7.3 Contrastive Temporal Loss Concern

**Severity**: MEDIUM (Training Effectiveness)

```python
class ContrastiveTemporalLoss:
    def forward(self, fused_states):  # [B, T, D]
        a = anchors.reshape(B * T_minus_1, D)  # N = B * (T-1) samples
        logits = torch.matmul(a, p.T) / self.temperature
        labels = torch.arange(N, device=...)
        return F.cross_entropy(logits, labels)
```

With per-device batch size B=2 and T=24, N = 2 × 23 = 46 samples. The InfoNCE loss operates on a 46×46 similarity matrix. This is an **extremely small** effective batch for contrastive learning.

For reference:
- CLIP uses 32,768 pairs
- SimCLR recommends ≥4,096 pairs
- MoCo uses 65,536 negative queue

With only 46 samples, the loss has ~45 negatives per positive. Most negatives will be from the same episode (same batch item), making them "easy negatives" that provide weak learning signal. The loss will converge quickly but won't learn meaningful temporal structure.

**Recommendations**:
1. Accumulate contrastive features across gradient accumulation steps (4 accum × 2 per-device = 184 samples)
2. Use a momentum-based feature bank (MoCo-style) to maintain a queue of past fused states
3. Alternatively, use a simpler temporal smoothness loss that doesn't require large batches:
   ```python
   loss = F.mse_loss(fused_states[:, :-1], fused_states[:, 1:].detach())
   ```

### 7.4 Slow-Fast Agreement Loss Analysis

```python
weights = torch.exp(torch.linspace(-2, 0, T, ...))  # exponential decay
fast_ema = (fast_tokens * weights).sum(dim=1)
return F.mse_loss(slow_token, fast_ema.detach())
```

This is well-designed. The exponential weights emphasize recent fast tokens (the last token gets weight ~1.0, the first gets ~0.14). The detach on `fast_ema` means only the slow stream is trained to match, not vice versa. This creates a one-way consistency constraint: slow should agree with fast's temporal consensus.

**Note**: The exponential weights are computed on CPU (`torch.linspace` then `torch.exp`), but since T is small (24), this is negligible.

### 7.5 Action Consistency Loss

```python
d = F.normalize(discrete_proj(discrete_actions), dim=-1)
c = F.normalize(continuous_proj(continuous_actions.detach()), dim=-1)
return 1.0 - (d * c).sum(dim=-1).mean()
```

Gradients flow only through the discrete branch (continuous is detached). This trains the FAST discrete head to produce predictions consistent with the flow expert's continuous outputs. The projection to 256d embedding space with cosine similarity is standard practice for cross-modal alignment.

**Edge case**: When `discrete_actions` is None (Stage A, no expert output), the consistency loss is called without the action term, and only temporal + slow-fast terms contribute. This is correctly handled via the `Optional` checks.

---

## 8. Training Pipeline Design

### 8.1 Three-Stage Curriculum

```
Stage A (120k steps, lr=2e-4):
  Train: Backbone LoRA + Grounder + Core + Discrete Heads
  Freeze: Action Expert
  Losses: discrete + phase + affordance + consistency

Stage B (200k steps, lr=1e-4):
  Train: + Action Expert (detached condition prefix)
  Losses: + flow_matching

Stage C (80k steps, lr=3e-5):
  Train: Full fine-tune
  Features: + RTC + FASTER (NOT IMPLEMENTED)
```

### 8.2 Critical Issue: Cross-Stage Checkpoint Loading Not Implemented

**Severity**: HIGH (Training Pipeline)

**Location**: `scripts/train_stage_a.py:163` + `configs/train/stage_b.yaml:37`

The Stage B config specifies:
```yaml
resume_from: outputs/v2_stage_a/checkpoint-latest
```

But the training script uses:
```python
start_step, start_epoch = auto_resume(
    cfg.train.output_dir, model, optimizer, scheduler, ema,  # output_dir = "outputs/v2_stage_b"
)
```

`auto_resume()` looks for checkpoints in `cfg.train.output_dir` (Stage B's directory: `outputs/v2_stage_b`), **not** `cfg.train.resume_from` (Stage A's checkpoint). The `resume_from` config field is **never referenced** in the training script.

**Impact**: When starting Stage B, the model initializes from **random weights** (plus pre-trained backbone) instead of loading Stage A's trained checkpoint. All 120k steps of Stage A training are wasted.

**Fix**: Add explicit cross-stage loading before `auto_resume`:
```python
if cfg.train.resume_from:
    load_checkpoint(cfg.train.resume_from, model, strict=False)
    logger.info("Loaded cross-stage checkpoint from %s", cfg.train.resume_from)

# Then auto_resume for Stage B's own checkpoints
start_step, start_epoch = auto_resume(cfg.train.output_dir, ...)
```

### 8.3 Scheduler State Conflict Across Stages

Even if cross-stage loading is fixed, loading Stage A's scheduler state into a Stage B scheduler creates problems. Stage A's scheduler used `total_steps=120000`, but Stage B creates a scheduler with `total_steps=200000`. The `LambdaLR` stores `last_epoch` (step counter) in its state dict. If Stage A ended at step 120k, loading this into Stage B's scheduler would compute cosine progress as `(120k - 5k)/(200k - 5k) = 0.59`, starting Stage B from the middle of its cosine cycle.

**Fix**: When doing cross-stage loading, do NOT load the scheduler state. Let it start fresh.

### 8.4 RTC and FASTER Not Implemented

**Severity**: MEDIUM (Incomplete Feature)

The Stage C config enables `rtc.enable: true` and `faster.enable: true`, and the config defines `RTCTrainConfig` and `FASTERTrainConfig` with parameters. However:
- `forward_train()` has no RTC or FASTER logic
- No RTC/FASTER code exists anywhere in the training loop
- The `execution_horizon`, `overlap_ratio`, `inpaint_overlap`, `near_ratio`, etc. are dead config

These are important inference-time optimization strategies that should also have training-time support (e.g., training with overlapping chunks for RTC robustness).

### 8.5 No Evaluation Loop

The config defines `eval_interval: 2000` but the training script has **no evaluation logic**. Best practices for VLA training include:
- Periodic policy rollout in simulation
- Validation loss on held-out demonstrations
- Discrete action accuracy metrics

### 8.6 Only Dummy Dataset Available

The training script falls back to `DummyVLADataset` which generates random tensors. There is no real dataset class, no data preprocessing pipeline, and no multi-camera data handling. The `DataConfig` defines fields for data paths and camera keys but no code consumes them.

### 8.7 Optimizer Configuration

```python
optimizer = AdamW(
    lr=2e-4, weight_decay=0.01,
    betas=(0.9, 0.95), fused=True,
)
```

- `beta2=0.95` is appropriate for large-model training (more aggressive than default 0.999, common in GPT-3/LLaMA training)
- `fused=True` enables fused CUDA AdamW kernel (significant speedup)
- `weight_decay=0.01` is standard
- No differential learning rates between modules (backbone LoRA, grounder, core all use same LR)

**Recommendation**: Consider differential LR for the backbone LoRA (lower, e.g., 0.5× base LR) since pre-trained representations need gentler updates. The grounder and core (randomly initialized) can use the full base LR.

---

## 9. Inference Pipeline

### 9.1 Two-Level Architecture

```
Semantic Step (~12.5 Hz):
  backbone → grounder → GrounderOutput

Control Step (~50 Hz):
  proprio + prev_action + GrounderOutput → TriRateCore → Expert → ActionChunk
```

### 9.2 Chunk-Based Action Execution

The expert produces 24-step action chunks. The `RuntimeCache` tracks `chunk_step` for chunk-based execution. However, the `control_step` method **always regenerates a full chunk** without checking if the current chunk is still valid. There's no chunk caching or re-use logic.

**Expected behavior**: Generate a new chunk, execute steps 0..H-1, then generate the next chunk. The current code regenerates every control step, which is 24× more expensive than necessary.

### 9.3 Missing Action History Update in Inference

In `control_step()`, `runtime_state.action_history` is initialized but **never updated** after executing actions. The action history encoder always receives the same initial zero tensor, providing no useful temporal context.

**Fix**: After sampling actions, push the executed action into the history buffer:
```python
# After denoised = self.action_expert.sample(...)
runtime_state.action_history = torch.roll(runtime_state.action_history, -1, dims=1)
runtime_state.action_history[:, -1] = denoised[:, 0]  # first action of chunk
```

---

## 10. Scalability & Memory Analysis

### 10.1 Parameter Count Estimate

| Module | Parameters | Trainable (Stage A) | Trainable (Stage B) |
|--------|-----------|---------------------|---------------------|
| Qwen2-VL-7B base | 7.6B | 0 (frozen) | 0 (frozen) |
| LoRA adapters | ~90M | 90M | 90M |
| MultiScaleAdapter | ~25M | 25M | 25M |
| Grounder (8 blocks) | ~540M | 540M | 540M |
| Fast Mamba (20L) | ~320M | 320M | 320M |
| Medium Mamba (6L) | ~96M | 96M | 96M |
| Slow Mamba (10L) | ~170M | 170M | 170M |
| ActionHistoryEncoder | ~32M | 32M | 32M |
| CrossAttentionFusion | ~70M | 70M | 70M |
| Flow Action Expert | ~830M | 0 (frozen) | 830M |
| Discrete heads | ~40M | 40M | 40M |
| Embeddings & projections | ~50M | 50M | 50M |
| **Total** | **~9.9B** | **~1.43B** | **~2.26B** |

### 10.2 Memory Budget (8×H100-80GB, FSDP)

```
Model weights (bf16):         ~20 GB total → ~2.5 GB/GPU with FSDP
Optimizer states (fp32):      ~9 GB trainable → ~1.1 GB/GPU
Gradients (bf16):             ~4.5 GB trainable → ~0.6 GB/GPU
Activations (checkpointed):   ~8-15 GB/GPU (varies by batch/seq)
─────────────────────────────────────────────────
Estimated per-GPU:            ~12-20 GB/GPU
Available:                    80 GB/GPU
Headroom:                     60-68 GB/GPU → can increase batch size
```

The memory budget is comfortable. The main bottleneck is likely compute throughput (due to token-by-token Mamba processing) rather than memory.

### 10.3 FSDP Auto-Wrap Policy Concern

**Severity**: MEDIUM

The FSDP auto-wrap policy includes:
```python
{MambaBlock, GrounderBlock, ExpertMambaBlock, ExpertAttentionBlock}
```

But the Qwen2-VL backbone's internal transformer layers are **NOT** included. The entire 7.6B backbone is treated as one FSDP unit. This means:
- Backbone parameters are not sharded across GPUs
- Each GPU holds the full backbone (~15 GB in bf16)
- Memory is wasted on redundant backbone copies

**Fix**: Include the backbone's transformer layer class in the auto-wrap set:
```python
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer
wrap_cls.add(Qwen2VLDecoderLayer)
```

However, since the backbone is frozen (no optimizer states, no gradients), the memory impact is less severe than for trainable modules.

---

## 11. Critical Issues Found

### C1. Cross-Stage Checkpoint Loading Not Implemented

**Impact**: Stage B/C training starts from scratch instead of building on prior stages. The entire 3-stage curriculum is broken.

**Details**: See Section 8.2.

**Fix Priority**: IMMEDIATE — must be fixed before any real training.

### C2. World Model Not Connected to Training

**Impact**: ~170M parameters of world model infrastructure are loaded but never used. No imagination-based training signal is generated.

**Details**: See Section 6.2.

**Fix Priority**: HIGH — either disable the world model by default or integrate it into forward_train().

### C3. Token-by-Token Processing in Official Mamba Path

**Impact**: ~33× performance penalty for the Tri-Rate Core, which is the most frequently called module in the system. This makes training and inference significantly slower than necessary.

**Details**: See Section 4.2.

**Fix Priority**: HIGH — significant training throughput improvement possible.

---

## 12. Moderate Issues Found

### M1. Contrastive Temporal Loss Ineffective at Small Batch Sizes

46 samples per device is too few for InfoNCE to learn meaningful temporal structure. See Section 7.3.

### M2. No Evaluation Loop

Training runs for 400k+ total steps with no validation. No way to detect overfitting, mode collapse, or training instability.

### M3. FSDP Does Not Wrap Backbone Layers

15 GB of backbone weights are not sharded, wasting memory on multi-GPU setups. See Section 10.3.

### M4. Inference Action History Never Updated

`control_step()` always feeds zero action history to the temporal core. See Section 9.3.

### M5. RTC / FASTER Not Implemented

Configured but dead code. Stage C training lacks these critical inference-time optimization training signals.

### M6. 44 Auxiliary Grounder Tokens Discarded

Significant FLOP overhead for tokens whose outputs are never used downstream. See Section 3.2A.

### M7. No Weight Initialization Strategy

All modules use PyTorch default initialization. For 20-layer Mamba stacks, proper initialization (e.g., scaled residual init) is important for training stability. Consider:

```python
# For each MambaBlock's out_proj:
nn.init.normal_(self.out_proj.weight, std=0.02 / math.sqrt(2 * num_layers))
```

---

## 13. Minor Issues & Recommendations

### m1. AdaRMSNorm Gate Initialization

Default initialization gives `sigmoid(gate) ≈ 0.5` at init, halving activations through the expert's 18 layers. Consider initializing gate bias to +2 for `sigmoid ≈ 0.88`. See Section 5.2.

### m2. No Differential Learning Rate

All modules share the same LR. Backbone LoRA should use a lower LR (e.g., 0.5×) to prevent representation drift. See Section 8.7.

### m3. EMA Starts in Stage A

EMA tracking begins in Stage A (120k steps), occupying ~1.4GB for shadow parameters. EMA is most useful in Stage B/C when the action expert is training. Consider deferring EMA to Stage B.

### m4. No torch.compile

`InferConfig.compile` is defined but never used. `torch.compile` can provide 1.3-2× speedup for the expert and grounder.

### m5. Dummy Dataset Only

No real dataset implementation exists. The `DataConfig` defines structure but no code consumes it.

### m6. Checkpoint Save Without FSDP Consolidation

`save_checkpoint` calls `_get_state_dict` which handles FSDP via `FullStateDictConfig`. This gathers all sharded states to rank 0, which can be slow for large models. Consider using `ShardedStateDictConfig` for faster saves during training, with full consolidation only at stage boundaries.

### m7. Sinusoidal Embedding Order

The accuracy.md flags [cos, sin] vs [sin, cos] ordering inconsistency, but upon inspection both `StaleTimeEncoding` and `SinusoidalTimestepEmbedding` use **[cos, sin]** order. The accuracy.md's claim of inconsistency is **incorrect** — no issue exists.

### m8. FASTDiscreteHead Bottleneck

The factorized head uses: hidden(768) → step_proj → chunk_horizon × 192 → per-dim head → action_dim × vocab_size.

The final linear is 192 → 14 × 512 = 7168, a 37× expansion. This is a severe bottleneck that concentrates representation pressure on a 192-dim intermediate. Consider increasing the step dimension to at least 384.

---

## 14. Final Verdict

### Overall Assessment

HybridVLA v2 is an **architecturally ambitious** system that combines a strong vision-language backbone (Qwen2-VL-7B) with a novel tri-rate temporal processing paradigm and dual action heads. The core ideas are sound and well-motivated by robotics control theory.

### Readiness for Training

| Aspect | Status | Blocking? |
|--------|--------|-----------|
| Core architecture (backbone → grounder → core → expert) | **Ready** | — |
| Forward pass correctness (post-v0.7 fixes) | **Ready** | — |
| Cross-stage checkpoint loading | **BROKEN** | YES |
| Training loop (Stage A single-stage) | **Ready** | — |
| Training pipeline (A→B→C curriculum) | **BROKEN** | YES |
| Dataset pipeline | **NOT IMPLEMENTED** | YES |
| Evaluation | **NOT IMPLEMENTED** | Soft block |
| World model integration | **NOT IMPLEMENTED** | Non-blocking (disabled by default) |
| RTC/FASTER training | **NOT IMPLEMENTED** | Non-blocking (Stage C only) |
| Inference pipeline | **Partially ready** (action history bug) | — |

### Priority Fix Order

1. **Cross-stage checkpoint loading** — without this, multi-stage training is impossible
2. **Real dataset pipeline** — without data, no training can happen
3. **Token-by-token Mamba optimization** — 33× performance improvement opportunity
4. **Evaluation loop** — necessary for training monitoring
5. **Inference action history update** — necessary for deployment
6. **Weight initialization strategy** — important for training stability
7. **Contrastive loss fix** — important for representation quality
8. **World model integration** — required for Stage C imagination-based training
9. **RTC/FASTER implementation** — required for competitive inference quality

### Architecture Quality Score

| Dimension | Score (1-10) | Notes |
|-----------|-------------|-------|
| Design coherence | 8/10 | Clean modular design with well-defined interfaces |
| Correctness (post-v0.7) | 9/10 | All known bugs fixed |
| Completeness | 5/10 | Multiple features configured but not implemented |
| Training stability | 7/10 | Stage-gated design is robust, but missing init strategy |
| Scalability | 7/10 | FSDP setup needs backbone wrapping improvement |
| Performance | 4/10 | Token-by-token Mamba processing is a major bottleneck |
| Production readiness | 3/10 | No real data pipeline, no eval, broken cross-stage loading |
| **Overall** | **6.1/10** | Strong design, incomplete implementation |

The architecture is well-designed for its intended purpose. The core innovation (tri-rate temporal processing with cross-attention fusion) is novel and well-motivated. However, the implementation has significant gaps in the training pipeline infrastructure that must be addressed before meaningful training can begin. The most impactful next steps are fixing cross-stage checkpoint loading, implementing a real dataset pipeline, and optimizing the Mamba processing to unlock the system's performance potential.
