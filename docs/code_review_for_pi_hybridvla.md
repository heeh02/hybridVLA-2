# HybridVLA v2 vs OpenPI: Detailed Code Review

> Reviewer: Claude (Automated Code Review)
> Date: 2026-03-28
> Scope: Full codebase review (vla_hybrid_v2/, scripts/, tests/)
> Version: v0.10.9 (~10,714 lines)
> Reference: comparsion_between_pi_and_hybridVLA_3_claude.md

---

## 1. Executive Summary

HybridVLA v2 is an architecturally ambitious VLA (Vision-Language-Action) system with genuine novelty in its tri-rate temporal design and hierarchical grounding. However, the codebase suffers from **zero end-to-end validation**, several **latent correctness bugs** not caught by the comparison document, and **performance bottlenecks** that may render training infeasible on target hardware. This review goes beyond the comparison document to identify code-level issues, verify its claims, and provide actionable fixes.

**Verified claims from comparison doc**: 9/10 accurate (see Section 3.7 for the one partial disagreement).
**New issues found**: 7 additional issues not in the comparison document.

---

## 2. Architecture Overview (Code-Level)

### 2.1 Module Dependency Graph

```
HybridVLAv2 (hybrid_vla_v2.py:55, 816 lines)
├── Qwen2VLBackboneWrapper (qwen2vl_backbone.py:126, 295 lines)
│   ├── Qwen2VLForConditionalGeneration (HuggingFace, 7.6B frozen)
│   ├── MultiScaleAdapter (qwen2vl_backbone.py:24)
│   └── CameraPositionEmbedding (qwen2vl_backbone.py:56)
├── HierarchicalAttentionGrounder (attention_grounder.py:157, 261 lines)
│   ├── 8× GrounderBlock (Cross-Attention + Self-Attention)
│   └── SlotCompression (48→24 via CrossAttention)
├── TriRateMambaCore (mamba_core.py:601, 786 lines)
│   ├── FastMamba (20L, d_state=128, ~340M)
│   ├── MediumMamba (6L, d_state=128, ~100M)
│   ├── SlowMamba (10L, d_state=256, ~170M)
│   ├── CrossAttentionFusion (2L)
│   └── StaleTimeEncoding
├── ActionHistoryEncoder (mamba_core.py:508, 4L Mamba, ~120M)
├── FlowActionExpert (flow_action_expert.py:233, 343 lines)
│   ├── 12× ExpertMambaBlock (M-M pattern)
│   └── 6× ExpertAttentionBlock (A pattern)
├── FASTDiscreteHead (discrete_heads.py:12)
├── PhaseHead (discrete_heads.py:47)
├── AffordanceHead (discrete_heads.py:60)
└── [WorldModel] (world_model/, 1205 lines, enable=false)
```

### 2.2 Parameter Distribution

| Module | Trainable | Frozen | Total |
|--------|-----------|--------|-------|
| Backbone (Qwen2-VL-7B) | ~120M (LoRA) | ~7.5B | ~7.6B |
| Grounder (8L) | ~270M | 0 | ~270M |
| TriRateMambaCore (36L) | ~610M | 0 | ~610M |
| ActionHistoryEncoder (4L) | ~120M | 0 | ~120M |
| FlowActionExpert (18L) | ~750M | 0 | ~750M |
| Discrete Heads | ~50M | 0 | ~50M |
| Projections + Misc | ~80M | 0 | ~80M |
| **Total** | **~2.0B** | **~7.5B** | **~9.5B** |

---

## 3. Correctness Issues

### 3.1 [P0-CRITICAL] Token-by-token loop kills training throughput

**File**: `mamba_core.py:432-454`

When `mamba_ssm` is installed (the official CUDA path), `_MambaStack.forward()` processes tokens **one at a time in Python**:

```python
# Official Mamba2 path — L=33 tokens, N=20 layers (fast stream)
for t in range(x.shape[1]):        # Python loop over L tokens
    x_t = x[:, t, :]
    for i, layer in enumerate(self.layers):  # Python loop over N layers
        x_t, ssm_states_list[i], conv_states_list[i] = layer.step(...)
    out[:, t, :] = x_t
```

This is nested inside the temporal loop in `forward_train` (`hybrid_vla_v2.py:429`):

```python
for t in range(T):  # T=24, outer temporal loop
    temporal_out = self.temporal_core(...)  # calls all 3 streams
```

**Total Python-level CUDA kernel launches per forward pass (fast stream alone)**:
- T × L × N_fast = 24 × 33 × 20 = **15,840 kernel launches**
- Adding medium (when active): + 12 × 33 × 6 = 2,376
- Adding slow (when active): + 3 × 33 × 10 = 990

Each `layer.step()` calls `self.mamba.step()` + norm + residual = ~3 CUDA kernels minimum. At ~10μs overhead per kernel launch, fast stream alone adds **~475ms of overhead per forward pass**, before any actual compute.

**Impact**: Training speed is likely **5-10x slower** than a batched implementation. A 150K-step training run on 8×H100 could take weeks instead of days.

**The comparison document is correct on this point (§3.4, §3.5).**

**Fix**: Implement a `forward_sequence()` method that uses `Mamba2.forward()` for the full sequence, accepting the tradeoff of not capturing intermediate SSM states. States are only needed at temporal boundaries, not within a sequence.

### 3.2 [P0-CRITICAL] Activation checkpointing does NOT work on official Mamba2 path

**File**: `mamba_core.py:432-454` (official path) vs `mamba_core.py:460-468` (fallback path)

The fallback path has:
```python
if use_checkpoint and self.training:
    x, s, c = activation_checkpoint(layer, x, s_i, c_i, use_reentrant=False)
```

The official path (lines 432-454) has **no checkpoint support** — it calls `layer.step()` directly. Since `_MambaStack.forward()` is the method called by `TriRateMambaCore`, and `use_checkpoint` is passed from `forward_train`:

```python
# hybrid_vla_v2.py:464
temporal_out = self.temporal_core(
    ..., use_checkpoint=self.cfg.train.checkpointing,
)
```

The `use_checkpoint` flag is forwarded to `_MambaStack.forward()` which **ignores it** on the official path. This means FSDP's activation checkpointing wrapper (applied in `distributed.py:133-148`) is the only checkpoint mechanism. But FSDP-level checkpointing wraps `MambaBlock`, not `_MambaStack` — and the official path's `step()` call bypasses the wrapped forward.

**Impact**: ~610M parameters of temporal core activations are **never checkpointed** on the official path. Estimated additional memory: 2-4GB per GPU at bfloat16.

**The comparison document correctly identifies this (§3.5).**

### 3.3 [P1-HIGH] RTC train-infer distribution shift

**File**: `hybrid_vla_v2.py:594-602` (train) vs `hybrid_vla_v2.py:770-789` (infer)

Training:
```python
with torch.no_grad():
    prev_chunk = self.action_expert.sample(
        cond_prefix=cond_prefix,  # same cond_prefix as current chunk
        ...
    )
```

Inference:
```python
# Line 772-780: RTC blending uses runtime_state.prev_chunk_tail
# which was generated with PREVIOUS observation's cond_prefix
```

The training simulation generates `prev_chunk` with the **same** observation as the current chunk. At inference, the previous chunk was conditioned on a **different** (older) observation. This creates a train-infer distribution mismatch in the overlap region.

Additionally, the inference blending uses **linear interpolation** (line 775-779):
```python
alpha = torch.linspace(1, 0, overlap, device=device)
denoised[:, :overlap] = alpha * prev_chunk_tail + (1 - alpha) * denoised[:, :overlap]
```

But training uses **MSE loss** on the overlap (line 607):
```python
loss_rtc = F.mse_loss(curr_head, prev_tail)
```

Training penalizes any difference; inference blends them. These are different objectives.

**The comparison document correctly identifies this (§3.7).**

### 3.4 [P1-HIGH] ContrastiveTemporalLoss may collapse representations

**File**: `consistency_loss.py:16-39`

```python
# Line 34-37
a = anchors.reshape(B * T_minus_1, D)  # N samples
p = positives.reshape(B * T_minus_1, D)  # N samples
logits = torch.matmul(a, p.T) / self.temperature  # [N, N]
labels = torch.arange(logits.shape[0], device=logits.device)
```

This InfoNCE formulation treats ALL non-paired states as negatives, including states from the **same trajectory** that are 2, 3, ... steps apart. Since Mamba's recurrent nature means consecutive states are already highly correlated, and states 2-3 steps apart are nearly as correlated:

1. The loss has very hard negatives (same trajectory, nearby timesteps) that are **semantically similar** to positives
2. The temperature=0.1 makes the softmax very peaked, amplifying this issue
3. With B=4, T=24, N=92: the positive signal is overwhelmed by ~91 negatives including ~22 near-duplicates from the same trajectory

**Risk**: Representations may collapse to a constant to minimize all pairwise distances equally, or training may be dominated by noise from within-trajectory false negatives.

**The comparison document partially identifies this (§3.3) but doesn't detail the within-trajectory false negative problem.**

### 3.5 [P1-HIGH] SlowFastAgreementLoss conceptual issue

**File**: `consistency_loss.py:42-57`

```python
# Line 54-57
weights = torch.exp(torch.linspace(-2, 0, T, device=fast_tokens.device))
weights = weights / weights.sum()
fast_ema = (fast_tokens * weights[None, :, None]).sum(dim=1)  # [B, D]
return F.mse_loss(slow_token, fast_ema.detach())
```

Problems:
1. The `.detach()` on `fast_ema` means gradients only flow through `slow_token`. This loss pushes the **slow stream toward the fast stream's EMA**, but the slow stream's role is to capture long-horizon context that is **different from** fast-stream summaries.
2. The exponential weighting (e^{-2} to e^0 = 0.135 to 1.0) gives recent fast tokens ~7.4x more weight. This makes the "EMA" heavily recency-biased, which is what the fast stream already does.
3. If slow_token successfully learns to match fast_ema, it becomes a **redundant representation** — defeating the purpose of having separate streams.

**The comparison document correctly questions this (§3.3).**

### 3.6 [P2-MEDIUM] ActionConsistencyLoss is weak

**File**: `consistency_loss.py:60-73`

```python
self.discrete_proj = nn.Linear(action_dim, embed_dim)   # 14 → 256
self.continuous_proj = nn.Linear(action_dim, embed_dim)  # 14 → 256
```

The input is 14-dimensional actions projected to 256 dimensions. With 14 DoF, the learned projections have huge freedom to map any two vectors to similar directions in 256D. The cosine similarity constraint in such a high-dimensional space is nearly trivially satisfiable — it constrains almost nothing.

A simpler and stronger constraint would be direct L1/L2 loss between the decoded discrete actions and continuous actions in the original 14D space.

### 3.7 [P2-MEDIUM] `eval()` in config.py — lower risk than claimed

**File**: `config.py:379`

```python
if isinstance(ft, str):
    ft = eval(ft, globals(), locals())
```

The comparison document flags this as a security concern (E-2). After review: this `eval()` only processes **type hint strings** from Python dataclass field annotations (e.g., `"List[str]"`, `"Optional[int]"`), not user input. The input comes from `cls.__dataclass_fields__` which is defined in the source code.

**Risk is LOW** — no path from YAML config to this eval. However, replacing with `typing.get_type_hints()` would be cleaner.

**Partial disagreement with comparison document: this is a code smell, not a security issue.**

---

## 4. New Issues (Not in Comparison Document)

### 4.1 [P1-NEW] `_build_cond_prefix` silent truncation/padding

**File**: `hybrid_vla_v2.py:233-260`

```python
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
```

This produces exactly 32 tokens (1+24+1+1+1+1+1+1+1=32). The `target_c = self.cfg.model.action_expert.cond_tokens` defaults to 32. But if the grounder's `compressed_slots` changes (e.g., to 16 or 32), the total will silently pad or truncate:

```python
if cond.shape[1] < target_c:
    pad = torch.zeros(B, target_c - cond.shape[1], cond.shape[2], ...)
    cond = torch.cat([cond, pad], dim=1)
elif cond.shape[1] > target_c:
    cond = cond[:, :target_c, :]  # SILENT TRUNCATION — drops temporal tokens!
```

If someone changes `compressed_slots` to 32, the temporal tokens (fused_state, fast, medium, slow) would be **silently truncated**, removing all temporal information from the expert's conditioning.

**Fix**: Add an assertion or at least a warning when truncation occurs.

### 4.2 [P1-NEW] `CameraPositionEmbedding` applies same embeddings to all batch items

**File**: `qwen2vl_backbone.py:96-122`

```python
# Line 96-106: tokens_per_image computed once from image_grid_thw
tokens_per_image = []
for i in range(image_grid_thw.shape[0]):
    t, h, w = image_grid_thw[i].tolist()
    tokens_per_image.append(int(t * h * w // merge_factor))

# Line 110-111: cam_ids computed once
cam_ids = torch.tensor(cam_indices, device=features.device, dtype=torch.long)
cam_emb = self.camera_embeddings(cam_ids)  # [total_vision_tokens, D]

# Line 116-122: applied identically to all B batch items
for b in range(B):
    vis_idx = vision_mask[b].nonzero(as_tuple=True)[0]
    n = min(n_vis, n_emb)
    if n > 0:
        out[b, vis_idx[:n]] = out[b, vis_idx[:n]] + cam_emb[:n]
```

`image_grid_thw` has shape `[num_images, 3]` — it's **not batched**. For B>1, the same `cam_emb` is applied to all items. This is correct only when all batch items have the same image layout (same number of vision tokens per camera). If images have different resolutions or padding within a batch, the camera embeddings will be **misaligned** for items beyond the first.

Qwen2-VL typically pads to a common length within a batch, but `image_grid_thw` records the **original** grid dimensions, not the padded ones. If two batch items have different numbers of vision tokens, the same `cam_emb` vector is applied to potentially different positions.

**Fix**: `image_grid_thw` should be batch-indexed, or the grid should be computed per-batch-item.

### 4.3 [P2-NEW] `ActionHistoryBuffer.push()` uses `torch.roll` which is not in-place

**File**: `types.py:111-123`

```python
def push(self, action: Tensor) -> None:
    if self.current_len < self.max_len:
        self.buffer[:, self.current_len] = action
        self.current_len += 1
    else:
        self.buffer = torch.roll(self.buffer, -1, dims=1)  # creates new tensor
        self.buffer[:, -1] = action
```

`torch.roll` creates a new tensor every call. With T=24 timesteps and max_len=8, this is called 16 times per forward pass, each allocating a [B, 8, 14] tensor. Not a major issue for small buffers, but the pattern is wasteful.

The `control_step` also has this pattern (line 799-802):
```python
runtime_state.action_history = torch.roll(runtime_state.action_history, -1, dims=1)
```

**Fix**: Use index-based circular buffer instead of roll:
```python
self.buffer[:, self.write_idx % self.max_len] = action
self.write_idx += 1
```

### 4.4 [P2-NEW] `forward_train` loss_total uses Python `sum()` on dict values

**File**: `hybrid_vla_v2.py:654`

```python
losses["loss_total"] = sum(losses.values())
```

This uses Python's built-in `sum()` which starts with integer 0, so the first addition is `0 + tensor`, which works but creates an unnecessary intermediate. More importantly, if any loss key is accidentally added twice or if the dict is mutated during iteration, this silently produces wrong totals.

**Minor**, but best practice is `torch.stack(list(losses.values())).sum()` or explicit accumulation.

### 4.5 [P2-NEW] ExpertMambaBlock doesn't use official Mamba2 even when available

**File**: `flow_action_expert.py:94-161`

Unlike `MambaBlock` in mamba_core.py which detects and uses `mamba_ssm.Mamba2`, `ExpertMambaBlock` always uses the hand-rolled SSM implementation:

```python
class ExpertMambaBlock(nn.Module):
    def __init__(self, d_model=1536, ...):
        # No HAS_MAMBA_SSM check — always uses manual SSM
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(...)
        ...
```

It does check `HAS_MAMBA_CUDA` for the `selective_scan_fn` fast-path (line 133), but doesn't use the full Mamba2 block. This means the expert's 12 Mamba blocks run with the fallback implementation even when official CUDA kernels are available. The expert processes [B, 26, 1536] sequences (24 action tokens + proprio + embodiment), and with 12 Mamba layers, this is a significant throughput hit.

The comparison doc discusses the core's token-by-token issue but doesn't note that the expert also misses optimization opportunities.

### 4.6 [P2-NEW] `evaluate()` missing `dist.barrier()` before model state change

**File**: `train_unified.py:556-571`

```python
# Line 559-562
if ema is not None:
    ema.apply(model)  # Changes model params
metrics = evaluate(model, val_loader, device, cfg)
if ema is not None:
    ema.restore(model)  # Restores model params
```

With FSDP, `ema.apply()` calls `summon_full_params(writeback=True)` which modifies sharded parameters. If ranks enter this section at different times (no explicit barrier), one rank might be computing gradients with the training weights while another has already swapped in EMA weights.

The code relies on all ranks reaching `evaluate()` at the same time (since it's after the same `global_step` check), which is generally true but not guaranteed under FSDP's async all-gather.

**Fix**: Add `barrier()` before `ema.apply()` and after `ema.restore()`.

### 4.7 [P3-NEW] `FlowMatchingLoss.forward()` ignores `t` parameter

**File**: `flow_matching.py:16-21`

```python
def forward(self, velocity_pred, x_0, x_1, t, step_weights=None):
    target_velocity = x_1 - x_0
    loss = (velocity_pred - target_velocity).pow(2)
    if step_weights is not None:
        loss = loss * step_weights.unsqueeze(-1)
    return loss.mean() if self.reduction == "mean" else loss
```

The `t` parameter is accepted but **never used**. In Rectified Flow, the target velocity is indeed `x_1 - x_0` (time-independent), so this is mathematically correct. But the unused parameter is confusing — it suggests someone might later try to add time-dependent weighting and forget to update this function.

---

## 5. Performance & Efficiency Analysis

### 5.1 Training Forward Pass Cost Estimate

| Component | FLOPs (est.) | Python overhead | Notes |
|-----------|-------------|-----------------|-------|
| Backbone (Qwen2-VL-7B) | ~R×14 TFLOPs | Low (single HF call) | R = num semantic refreshes |
| Grounder (8L × 96 latents) | ~R×0.5 TFLOP | Low | |
| **Temporal Core (T=24 steps)** | **~24×1.2 TFLOP** | **HIGH (15,840+ kernel calls)** | **Bottleneck** |
| ActionHistoryEncoder (T=24) | ~24×0.1 TFLOP | Medium | |
| Expert (1 forward, 18L) | ~0.8 TFLOP | Low-Medium | Stage B/C only |
| **Total** | **~45 TFLOP** | **Dominated by temporal core** | |

### 5.2 Memory Estimate (per GPU, bfloat16, 8×H100)

| Component | Memory |
|-----------|--------|
| Model parameters (sharded 8-way) | ~2.4 GB |
| Frozen backbone (sharded) | ~1.9 GB |
| Optimizer states (AdamW, 2x) | ~4.8 GB |
| SSM states (36L × T=24) | ~1.2 GB |
| Activations (no checkpoint on official path) | **~12-20 GB** |
| Batch data (B=4, T=24) | ~0.5 GB |
| **Total** | **~23-31 GB** |

H100-80GB leaves ~49-57 GB headroom. This should fit, but the activation estimate is uncertain because checkpoint behavior on the official Mamba2 path is untested.

### 5.3 OpenPI comparison

| Metric | HybridVLA v2 (est.) | OpenPI |
|--------|---------------------|--------|
| Trainable params | ~2.0B | ~500M |
| Forward FLOPs | ~45 TFLOP | ~8 TFLOP |
| Training steps | 150K-400K | 30K |
| Python loop overhead | 15,840+ calls | 0 (single forward) |
| Inference latency (per chunk) | ~68ms (est.) | ~20ms (est.) |

---

## 6. Code Quality Assessment

### 6.1 Strengths

1. **Type annotations**: Consistent use of type hints throughout. All function signatures are annotated. Dataclass types are well-defined (`types.py`).

2. **Configuration system**: Clean hierarchical dataclass config with YAML inheritance. The `_dict_to_dataclass` recursive loader with unknown-key warnings (`config.py:371-376`) is a good defensive pattern.

3. **Stage gating**: The explicit `configure_trainable_modules()` + `sanity_check_trainable_params()` pattern (`train_unified.py:118-262`) is robust and prevents the most common multi-stage training bug (training wrong modules).

4. **EMA implementation**: Correct FSDP-aware EMA with `summon_full_params` (`ema.py`). The `_strip_fsdp_prefix` pattern handles FSDP name mangling correctly.

5. **Checkpoint asset system**: Normalizer stats and config travel with checkpoints (`train_unified.py:97-111`), ensuring reproducible inference.

6. **Inference config auto-discovery**: `find_resolved_config()` + `_candidate_stats_dirs()` (`libero_policy.py:39-134`) provide good UX for checkpoint loading.

7. **Per-module LR groups**: Well-implemented in `train_unified.py:405-438` with proper FSDP prefix stripping.

8. **Batch validation**: `_validate_batch()` (`hybrid_vla_v2.py:266-341`) catches shape mismatches early with clear error messages.

### 6.2 Weaknesses

1. **No CI/CD**: No GitHub Actions, no automated testing on push. Tests exist (42 tests, 972 lines) but are only run manually.

2. **Dead code**: `world_model/` (1,205 lines, 15 classes) is never enabled. `train_stage_a.py` (278 lines) is superseded by `train_unified.py`.

3. **Missing test coverage for critical paths**:
   - `TriRateMambaCore` — 0 dedicated tests (most complex module, 786 lines)
   - `HierarchicalAttentionGrounder` — 0 dedicated tests (261 lines)
   - `_MambaStack.forward()` official path — 0 tests
   - `control_step()` — only 2 basic tests, no multi-step rollout test
   - RTC blending — 0 tests
   - FASTER weighting — 0 tests

4. **Inconsistent use of asserts vs exceptions**: `_validate_batch()` uses `assert` (disabled in optimized mode), but `resolve_policy_config()` uses proper `raise`. Production code should not rely on `assert` for validation.

5. **Magic numbers**:
   - `mamba_core.py:552`: `0.02` init scale for fusion_query
   - `flow_action_expert.py:47`: `2.0` gate bias init
   - `consistency_loss.py:22`: temperature=0.1 (no config exposure)
   - `consistency_loss.py:92-94`: hardcoded 0.5 weights for sub-losses

6. **Logging verbosity**: `_log_per_module_grad_norm()` only runs every `log_interval * 5` steps (`train_unified.py:532`). During early debugging, per-step gradient norms would be more useful.

### 6.3 Code Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Total lines (Python) | ~10,714 | Reasonable for scope |
| Dead/duplicate code | ~1,483 lines (13.8%) | Should be cleaned |
| Test lines | 972 | Low for codebase size |
| Test/code ratio | 0.09 | Below 0.2 threshold |
| Max function complexity | `forward_train` (~310 lines) | Too large, should split |
| Max file size | `mamba_core.py` (786 lines) | Acceptable |
| Cyclomatic complexity (forward_train) | ~25 | High — multiple nested if/for |

---

## 7. Testing Review

### 7.1 Test Coverage Map

| Module | Test File | Tests | Critical Paths Covered |
|--------|-----------|-------|----------------------|
| FlowActionExpert | test_expert.py | 6 | forward, AdaRMSNorm |
| Losses | test_losses.py | 9 | All 7 loss functions |
| Forward train | test_forward_train.py | 8 | Stage A/B/C forward, backward |
| Config resolution | test_eval_config_resolution.py | 7 | Config discovery, mismatch detection |
| Normalizer | test_normalizer.py | 7 | Normalize/denormalize round-trip |
| Control step | test_control_step.py | 2 | Basic control step, FASTER error |
| Inference policy | test_infer_policy.py | 3 | Policy loading, step output |
| Checkpoint assets | test_checkpoint_assets.py | 1 | Asset copying |

### 7.2 Critical Coverage Gaps

| Missing Test | Risk |
|-------------|------|
| TriRateMambaCore forward + state persistence | State could silently reset between steps |
| Grounder hierarchical compression | Slot indexing could be wrong |
| Official Mamba2 path (requires mamba_ssm) | Only fallback path tested |
| Multi-step rollout (control_step × N) | Chunk caching + RTC could break |
| EMA apply/restore round-trip | Could corrupt model params |
| FSDP wrapping + training step | Multi-GPU training untested |
| Multi-camera forward pass | Camera embedding alignment untested |
| ActionHistoryBuffer circular behavior | Roll-based push could lose data |

---

## 8. Security & Safety

| Issue | Severity | Location | Notes |
|-------|----------|----------|-------|
| `eval()` on type strings | LOW | config.py:379 | Internal type hints only, not user input |
| No input sanitization on YAML | LOW | config.py:389 | Uses `yaml.safe_load` (good) |
| Unbounded file path traversal | LOW | libero_policy.py:49 | `parent.parent` traversal limited to 2 levels |
| No GPU memory limit enforcement | MEDIUM | distributed.py | No `max_split_size_mb` or memory fraction limit |

---

## 9. Comparison Document Verification

| Claim | Verified | Notes |
|-------|----------|-------|
| §2.1: P0 bugs (FSDP deadlock, EMA) were fixed | YES | Code shows correct FSDP evaluate flow, EMA with summon_full_params |
| §3.1: Zero validation | YES | No evidence of any training run in code or git history |
| §3.2: 4.6× more trainable params vs OpenPI | PARTIALLY | My count is ~2.0B vs claimed ~2.3B. Still ~4× OpenPI's ~500M |
| §3.3: ActionHistoryEncoder over-parameterized | YES | 120M params for 8×14=112 floats is extreme |
| §3.4: 15,840 kernel calls per forward | YES | Confirmed by code tracing (see §3.1 above) |
| §3.5: Activation checkpointing broken on official path | YES | Confirmed — no `use_checkpoint` branch in official path |
| §3.6: FAST head is not pi-0-FAST | YES | Single MLP, no autoregressive generation |
| §3.7: RTC train-infer mismatch | YES | Same cond_prefix in train, different in infer |
| §5.1: F-1 (zero validation) | YES | |
| §5.2: B-1 (FASTER NotImplementedError) | YES | hybrid_vla_v2.py:691 |
| §5.3: E-2 (eval security) | PARTIALLY | Lower risk than claimed — only processes internal type hints |
| §5.3: E-3 (world model dead code) | YES | 1,205 lines, enable=false, never imported in training path |

---

## 10. Recommendations (Prioritized)

### P0: Must-fix before first training run

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 1 | **Implement batched Mamba2 forward** for temporal core | 2-3 days | 5-10× training speedup |
| 2 | **Add activation checkpointing** to official Mamba2 path | 1 day | Prevent OOM on 8×H100 |
| 3 | **Run LIBERO 500-step single-GPU smoke test** | 1 day | Prove loss decreases, no NaN |
| 4 | **Measure actual steps/sec and GPU memory** | 0.5 day | Validate feasibility |

### P1: Fix before serious training

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 5 | Fix `_build_cond_prefix` to assert instead of silent truncation | 0.5 hour | Prevent silent information loss |
| 6 | Add `barrier()` around EMA apply/restore in evaluate | 0.5 hour | Prevent FSDP race condition |
| 7 | Reconsider ContrastiveTemporalLoss — add same-trajectory exclusion or increase temperature | 1 day | Prevent representation collapse |
| 8 | Fix CameraPositionEmbedding to handle per-batch-item grid | 1 day | Correct multi-camera behavior |
| 9 | Implement FASTER inference (remove NotImplementedError) | 2 days | Unblock Stage C inference |

### P2: Important cleanup

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 10 | Delete `world_model/` (1,205 lines) | 0.5 hour | Reduce maintenance burden |
| 11 | Delete `train_stage_a.py` (278 lines) | 0.5 hour | Remove duplicate code |
| 12 | Add tests for TriRateMambaCore and Grounder | 2 days | Test most complex modules |
| 13 | Replace `assert` in `_validate_batch` with `ValueError` | 1 hour | Production-safe validation |
| 14 | Add ruff + pre-commit hooks | 1 hour | Automated code quality |
| 15 | Expose consistency loss hyperparams to config | 1 hour | Enable tuning without code changes |
| 16 | Use official Mamba2 block in ExpertMambaBlock when available | 1 day | Expert forward speedup |

### P3: Nice-to-have

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 17 | Replace ActionHistoryEncoder's 4L Mamba with 2L MLP | 1 day | ~120M → ~1M params savings |
| 18 | Add CI/CD (GitHub Actions with smoke test) | 0.5 day | Catch regressions |
| 19 | Add TensorBoard/WandB integration | 1 day | Better training monitoring |
| 20 | Implement inference serving (FastAPI/gRPC) | 2-3 days | Enable real deployment |

---

## 11. Conclusion

HybridVLA v2 demonstrates genuine architectural creativity, particularly in the tri-rate temporal processing concept and hierarchical slot compression. The code quality is above average for a solo research project — type annotations, configuration management, and stage gating are well-executed.

However, the comparison document's core assessment is correct: **this is a well-drawn blueprint, not a tested building**. The most critical issue is not any single bug but the fact that the training pipeline has never been executed end-to-end. The token-by-token loop performance issue (§3.1) and activation checkpoint gap (§3.2) may make training infeasible on the target hardware without architectural changes.

**The 500-step smoke test (Recommendation #3) should be the absolute first priority** — it will reveal whether the performance and memory issues are blockers or manageable.

---

## Appendix A: File-by-File Review Summary

| File | Lines | Quality | Critical Issues |
|------|-------|---------|-----------------|
| hybrid_vla_v2.py | 816 | B+ | forward_train too long, cond_prefix truncation |
| mamba_core.py | 786 | B | Token-by-token loop, no checkpoint on official path |
| flow_action_expert.py | 343 | A- | Doesn't use official Mamba2 for expert blocks |
| attention_grounder.py | 261 | A | Clean, but untested |
| qwen2vl_backbone.py | 295 | B+ | CameraPositionEmbedding batch issue |
| discrete_heads.py | 77 | A | Simple and correct |
| config.py | 400 | A- | eval() on type hints (low risk) |
| types.py | 129 | A | Clean dataclass definitions |
| ema.py | 128 | A | Correct FSDP-aware implementation |
| consistency_loss.py | 96 | C+ | All three sub-losses have design concerns |
| flow_matching.py | 33 | A | Simple, correct (unused `t` is minor) |
| libero_policy.py | 419 | A | Robust config/normalizer resolution |
| distributed.py | 164 | A- | Missing barrier in eval flow |
| train_unified.py | 602 | A- | Well-structured, good stage gating |
| selective_scan.py | 56 | A | JIT-compiled fallback is correct |
| train_stage_a.py | 278 | D | Dead code, duplicate of train_unified.py |
| world_model/*.py | 1,205 | N/A | Dead code (enable=false) |

**Overall Grade: B-** (good code quality for a research prototype, but untested and with performance blockers)
