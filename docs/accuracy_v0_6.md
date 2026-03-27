# HybridVLA v2 Correctness Analysis

Compared against:
- **Qwen2-VL**: [huggingface/transformers `modeling_qwen2_vl.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)
- **Mamba-2**: [state-spaces/mamba `mamba2.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py)
- **Mamba-1**: [state-spaces/mamba `mamba_simple.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py)
- **selective_scan**: [state-spaces/mamba `selective_scan_interface.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py)

---

## 1. Qwen2-VL Backbone (`qwen2vl_backbone.py`)

### 1.1 Model Loading & Architecture Constants

| Item | Project | Official (7B) | Status |
|------|---------|---------------|--------|
| Model class | `Qwen2VLForConditionalGeneration` | `Qwen2VLForConditionalGeneration` | **CORRECT** |
| hidden_size | 3584 | 3584 | **CORRECT** |
| num_layers | 28 | 28 | **CORRECT** |
| IMAGE_TOKEN_ID | 151655 | 151655 (`config.image_token_id`) | **CORRECT** |
| VIDEO_TOKEN_ID | 151656 | 151656 (`config.video_token_id`) | **CORRECT** |
| attn_implementation | `flash_attention_2` | supported via `from_pretrained` kwarg | **CORRECT** |
| `output_hidden_states=True` | used to extract multi-scale features | returns per-layer hidden states | **CORRECT** |

### 1.2 BUG - Vision Tower Freeze Path (HIGH)

**File**: `qwen2vl_backbone.py:110-113`

```python
# Current code:
if freeze_vision and hasattr(self.model, "visual"):
    for p in self.model.visual.parameters():
        p.requires_grad = False
```

In the official HuggingFace implementation:
- `Qwen2VLForConditionalGeneration.model` -> `Qwen2VLModel`
- `Qwen2VLModel.visual` -> `Qwen2VisionTransformerPretrainedModel`

So `self.model` (which is `Qwen2VLForConditionalGeneration`) does **not** have a `.visual` attribute directly. The correct path is `self.model.model.visual`.

**Result**: `hasattr(self.model, "visual")` returns `False` -> vision tower is **never frozen**, consuming ~600M extra trainable parameters and potentially destabilizing LoRA training.

**Fix**:
```python
composite_model = getattr(self.model, "model", self.model)
if freeze_vision and hasattr(composite_model, "visual"):
    for p in composite_model.visual.parameters():
        p.requires_grad = False
```

**Note**: This may be version-dependent. Some older transformers builds used a flatter structure. If the project targets a specific transformers version where `ForConditionalGeneration.visual` exists, this is not a bug. Verify with `print(type(self.model))` and `dir(self.model)`.

### 1.3 BUG - Text Layer Freeze Path (MEDIUM)

**File**: `qwen2vl_backbone.py:114-119`

```python
text_model = self.model.model
if hasattr(text_model, "layers"):
    for i, layer in enumerate(text_model.layers):
```

In the latest transformers, `self.model.model` = `Qwen2VLModel`, which has `self.language_model` (a `Qwen2VLTextModel`) containing `self.layers`. So `text_model.layers` would fail -- it should be `text_model.language_model.layers`.

**Result**: Same version-dependency as 1.2. If using the latest transformers API (`Qwen2VLModel` -> `language_model` -> `layers`), text layers 0-15 are **not frozen**.

### 1.4 Multi-Scale Feature Extraction -- Correct with Caveat

**File**: `qwen2vl_backbone.py:181-186`

```python
all_hidden = outputs.hidden_states
for layer_idx in self.multi_scale_layers:  # [10, 18, 28]
    idx = min(layer_idx, len(all_hidden) - 1)
    multi_scale.append(all_hidden[idx])
```

**Traditional** HuggingFace convention: `hidden_states` = `(embedding, layer_0, ..., layer_27)` -> 29 entries.
**Newer** `@capture_outputs` API: `hidden_states` = `(layer_0, ..., layer_27)` -> 28 entries.

- With 29 entries: index 28 = layer 27 (last layer) -- correct
- With 28 entries: index 28 clamped to 27 = layer 27 -- still correct for last layer, but indices 10 and 18 shift by 1 layer

**Verdict**: Safe due to `min()` clamping. The semantic intent (early/mid/late features) is preserved either way. Layer offset of 1 is negligible.

### 1.5 MultiScaleAdapter -- Correct

**File**: `qwen2vl_backbone.py:24-53`

- FPN-inspired multi-scale fusion with per-scale LayerNorm + Linear projection
- Learned gating via softmax over pooled projections
- Weighted sum fusion

Standard approach, no correctness issues. The gate computes global (mean-pooled) scale weights then broadcasts, which is a reasonable design.

### 1.6 LoRA Application -- Correct

**File**: `qwen2vl_backbone.py:124-146`

- Uses PEFT `LoraConfig` with `layers_to_transform=list(range(total_layers))`
- Targets all standard linear modules: q/k/v/o_proj + gate/up/down_proj
- `task_type="CAUSAL_LM"` matches the model type

**Correct.** The `layers_to_transform` parameter correctly targets all 28 layers per the v2 design.

### 1.7 Processor & Token Detection -- Correct

**File**: `qwen2vl_backbone.py:87-96`

Falls back to hardcoded constants when tokenizer attributes aren't found. Good defensive pattern.

---

## 2. Mamba Core (`mamba_core.py`)

### 2.1 CRITICAL BUG - Official Mamba2 Path Missing Residual + Pre-Norm

**File**: `mamba_core.py:234-243`

```python
def _forward_official(self, x: Tensor) -> Tuple[Tensor, None, None]:
    ...
    out = self.mamba(x)  # comment says "residual included inside Mamba2"
    ...
    return out, None, None
```

The comment **"residual included inside Mamba2"** is **incorrect**.

From the official `mamba_ssm` source, `Mamba2.forward()` performs:
```
in_proj -> conv1d -> SiLU -> SSM_scan -> gated_RMSNorm -> out_proj
```

It does **NOT** include:
1. **Pre-norm** (input LayerNorm) -- must be applied externally by a `Block` wrapper
2. **Residual connection** (`x + out`) -- must be added externally

The fallback path correctly implements both:
```python
def _forward_fallback(self, x, ...):
    residual = x
    x = self.norm(x)          # <- pre-norm
    ...
    out = self.out_proj(y) + residual   # <- residual
```

**Impact**: For the Fast stream (20 layers), Medium stream (6 layers), and Slow stream (10 layers), all layers lack residual connections and pre-norm when using the CUDA path. This means:
- **No skip connections** across a 20-layer stack -> severe gradient flow issues
- **No pre-normalization** -> activation magnitude instability
- The model would produce fundamentally different (and likely worse) outputs compared to the fallback path

**The same bug affects `_step_official()`** (line 202-230): `self.mamba.step()` also lacks residual and pre-norm.

**Fix for `_forward_official`**:
```python
def _forward_official(self, x: Tensor) -> Tuple[Tensor, None, None]:
    is_single = x.dim() == 2
    if is_single:
        x = x.unsqueeze(1)
    residual = x
    out = self.mamba(x)
    out = out + residual       # <- add residual
    if is_single:
        out = out.squeeze(1)
    return out, None, None
```

**Fix for `_step_official`** -- add residual after step:
```python
def _step_official(self, x, ssm_state, conv_state):
    ...
    out, new_conv_state, new_ssm_state = self.mamba.step(x, conv_state, ssm_state)
    out = out + x  # <- add residual
    return out, new_ssm_state, new_conv_state
```

**Note**: The official `Mamba2` module already includes an internal gated RMSNorm on the output (before `out_proj`), so a separate per-layer pre-norm may not be strictly necessary -- but the residual connection is critical.

### 2.2 Architecture Mismatch: Fallback = Mamba-1, Official = Mamba-2

The fallback path implements **Mamba-1** architecture:

| Aspect | Fallback (Mamba-1) | Official (Mamba-2) |
|--------|-------------------|-------------------|
| A shape | `(d_inner, d_state)` | `(nheads,)` -- one scalar per head |
| SSM state | `(B, d_inner, d_state)` | `(B, nheads, headdim, d_state)` |
| dt projection | Low-rank: `dt_rank -> d_inner` | Direct: `nheads` values from `in_proj` |
| Conv1d applied to | `x` only (`d_inner` channels) | `(x, B, C)` concatenated |
| Gating | `y * SiLU(z)` | Gated RMSNorm: `RMSNorm(y) * SiLU(z)` |
| Scan algorithm | Sequential recurrence | Chunked SSD (chunk_size=256) |

**Severity**: LOW for development (the code warns about this). HIGH if someone trains with one backend and deploys with the other -- outputs would be completely different. The two paths are **not interchangeable**.

### 2.3 BUG - Conv State Handling in Fallback Sequence Mode (MEDIUM)

**File**: `mamba_core.py:267-273`

```python
x_conv = x_main.transpose(1, 2)  # [B, d_inner, L]
if conv_state is not None:
    x_conv = torch.cat([conv_state, x_conv], dim=-1)  # [B, d_inner, (d_conv-1)+L]
x_conv = self.conv1d(x_conv)     # Conv1d has padding=d_conv-1 (symmetric!)
x_conv = x_conv[:, :, :L]        # <- truncation
```

`nn.Conv1d(padding=d_conv-1)` pads `d_conv-1` zeros on **both** sides. When `conv_state` is prepended, the left zero-padding overwrites the historical context.

**Example** (d_conv=4, L=2, conv_state=[a,b,c]):
```
Concatenated input: [a, b, c, x1, x2]           (length 5)
After Conv1d pad:   [0, 0, 0, a, b, c, x1, x2, 0, 0, 0]  (length 11)
Output length:      11 - 4 + 1 = 8

Output[0] = conv(0, 0, 0, a)     <- should be irrelevant
Output[3] = conv(a, b, c, x1)    <- what we WANT for x1
Output[4] = conv(b, c, x1, x2)   <- what we WANT for x2

Truncated [:L=2]: takes Output[0] and Output[1]  <- WRONG positions!
```

The correct truncation when `conv_state` is provided should be `[:, :, (d_conv-1):(d_conv-1)+L]`.

**Impact**: Affects first ~3 tokens of every temporal step in the fallback path. Since the input sequence is `[global, phase, uncertainty, affordance, proprio, prev_action, stale, embodiment, action_history, slots...]`, the affected positions include important semantic tokens. Only impacts fallback path (official path uses `Mamba2.step()` which handles conv state correctly).

### 2.4 _MambaStack.init_states() -- Latent Bug

**File**: `mamba_core.py:326-344`

```python
ssm = [torch.zeros(batch_size, self.d_inner, self.d_state, ...) ...]
```

This creates Mamba-1-shaped SSM states `(B, d_inner, d_state)`. For the official Mamba2 path, the correct shape is `(B, nheads, headdim, d_state)`. Currently not called in the main code path (states initialize as `None`), but would produce wrong shapes if used.

### 2.5 Tri-Rate Design -- Correct

| Stream | Layers | d_state | Update Rule | Purpose |
|--------|--------|---------|-------------|---------|
| Fast | 20 | 128 | Every step | Reactive control |
| Medium | 6 | 128 | Every 2nd step | Tactical |
| Slow | 10 | 256 | Semantic refresh | Strategic |

- Update scheduling logic in `TriRateMambaCore.forward()` is correct
- State caching when streams don't update (reuse last token) is correct
- `steps_since_refresh` / `steps_since_medium` counters are correct
- Cross-attention fusion replaces scalar gate (v1->v2 upgrade) -- correct implementation

### 2.6 CrossAttentionFusion -- Correct

**File**: `mamba_core.py:480-537`

- 2-layer cross-attention with learned fusion query attending to [fast, medium, slow] tokens
- Stale-time conditioning via additive projection
- Pre-norm cross-attention with FFN -- standard architecture

### 2.7 ActionHistoryEncoder -- Correct

**File**: `mamba_core.py:451-473`

- Projects actions [B, K, A] -> [B, K, d_model], processes through 4-layer Mamba, takes last token
- Uses `_MambaStack` which inherits the same official/fallback path issues

### 2.8 StaleTimeEncoding -- Correct

**File**: `mamba_core.py:47-72`

Standard sinusoidal positional encoding with MLP. Clamped to `max_staleness=256`.

---

## 3. Selective Scan (`ops/selective_scan.py`)

### 3.1 JIT-compiled `ssm_scan` -- Correct

**File**: `selective_scan.py:37-55`

```python
@torch.jit.script
def ssm_scan(dA, dBx, C, state):
    for t in range(L):
        state = dA[:, t] * state + dBx[:, t]
        y[:, t] = (state * C[:, t].unsqueeze(1)).sum(-1)
    return y, state
```

This is the standard SSM recurrence: `h_t = A_t * h_{t-1} + B_t * x_t; y_t = C_t * h_t`. Pre-discretized (dA, dBx already computed). **Mathematically correct.**

### 3.2 CUDA Imports -- Correct

Imports `selective_scan_fn` from `mamba_ssm.ops.selective_scan_interface` and `causal_conv1d_fn` from `causal_conv1d`. Graceful fallback when unavailable.

---

## 4. Flow Action Expert (`flow_action_expert.py`)

### 4.1 ExpertMambaBlock -- Correct

**File**: `flow_action_expert.py:90-157`

Unlike `MambaBlock`, the expert block always uses its own pre-norm + residual, regardless of backend:

```python
def forward(self, x, t_cond):
    residual = x
    x = self.norm(x, t_cond)  # AdaRMSNorm (always applied)
    ...
    return residual + self.out_proj(y)  # residual (always applied)
```

The CUDA path uses `selective_scan_fn` correctly:
- Tensor layouts match official spec: `u/dt` as `(B, D, L)`, `A` as `(D, N)`, `B/C` as `(B, N, L)`
- `delta_softplus=True` with `D` and `z` gate passed into the kernel
- dt_proj bias is embedded in the Linear layer (mathematically equivalent to passing `delta_bias` separately)

### 4.2 AdaRMSNorm -- Correct

**File**: `flow_action_expert.py:31-49`

Implements `sigmoid(gate) * (RMSNorm(x) * (1 + scale) + shift)` where `(scale, shift, gate)` are projected from the condition vector. Matches the pi-0.5 (Physical Intelligence) style adaptive normalization.

### 4.3 ExpertAttentionBlock -- Correct

**File**: `flow_action_expert.py:164-222`

Three-part block: cross-attention -> self-attention -> FFN, all with AdaRMSNorm. Uses `F.scaled_dot_product_attention` for auto Flash-Attention dispatch. Correct head dimension handling.

### 4.4 ODE Solvers -- Correct

**Euler** (`sample_euler`): `x_{i+1} = x_i + dt * v(x_i, t_i)` -- standard explicit Euler.
**Midpoint** (`sample_midpoint`): `x_mid = x_i + 0.5*dt*v(x_i, t_i); x_{i+1} = x_i + dt*v(x_mid, t_mid)` -- standard RK2 midpoint. ~2x accuracy for same cost.

Both integrate from t=0 to t~1 with `dt = 1/num_steps`. **Correct.**

### 4.5 Timestep Embeddings -- Correct (Minor Inconsistency)

`SinusoidalTimestepEmbedding` uses **[cos, sin]** order.
`StaleTimeEncoding` uses **[sin, cos]** order.

Both are valid; the MLP on top adapts to either. Functionally equivalent but inconsistent.

---

## 5. Attention Grounder (`attention_grounder.py`)

### 5.1 Architecture -- Correct

Perceiver-style: 96 learned latent queries cross-attend to backbone features through 8 GrounderBlocks. Each block = cross-attention + self-attention + FFN.

Slot layout: `[global(1), objects(48), phase(1), unc(1), aff(1), aux(44)] = 96`

### 5.2 Hierarchical Compression -- Correct

At layer 4, object slots (48 -> 24) are compressed via learned cross-attention routing queries. Post-compression layout: `[global(1), compressed(24), phase(1), unc(1), aff(1), aux(44)] = 72`. Layers 5-7 continue with 72 tokens.

The slot carving in the output correctly handles both compressed and non-compressed paths.

### 5.3 Cross/Self Attention -- Correct

Uses `F.scaled_dot_product_attention` with proper masking support. Pre-norm pattern throughout.

---

## 6. Discrete Heads (`discrete_heads.py`)

### 6.1 FASTDiscreteHead -- Correct

- Discretization: `[-1, 1] -> [0, V-1]` linear mapping. Inverse is exact.
- Architecture: `LayerNorm -> Linear -> GELU -> step_proj -> per-dim head`
- Factorized: one shared encoder, then per-step + per-dimension projection

### 6.2 PhaseHead / AffordanceHead -- Correct

Simple MLP classifiers over the respective grounder tokens.

---

## 7. Flow Matching Loss (`losses/flow_matching.py`)

### 7.1 Velocity Target -- Correct

```python
target_velocity = x_1 - x_0  # constant velocity field (optimal transport)
```

For linear interpolation `x_t = (1-t)*x_0 + t*x_1`, the velocity is `dx_t/dt = x_1 - x_0`.

### 7.2 Logit-Normal Sampling -- Correct

```python
torch.sigmoid(torch.randn(batch_size, device=device))
```

Samples from logit-normal distribution centered at t=0.5, concentrating training on intermediate noise levels. Standard practice.

### 7.3 Dead Parameter (COSMETIC)

`sigma_min` parameter in `interpolate()` is accepted but never used.

---

## 8. Consistency Loss (`losses/consistency_loss.py`)

### 8.1 ContrastiveTemporalLoss -- Correct

InfoNCE-style: consecutive fused states form positive pairs, all other pairs are negatives. L2-normalized before dot product. Standard contrastive learning.

### 8.2 SlowFastAgreementLoss -- Correct

Exponentially-weighted moving average of fast tokens should match the slow token. Encourages temporal consistency across streams.

### 8.3 ActionConsistencyLoss -- Correct

Projects discrete and continuous action predictions into shared embedding space, minimizes cosine distance.

---

## 9. World Model Components

### 9.1 StochasticStateModule -- Correct

DreamerV3-style: 48x48 categorical with 1% uniform mixing (`unimix=0.01`). Straight-through gradient via `sample + probs - probs.detach()`. Posterior conditions on `(z_det, obs_encoding)`, prior on `z_det` only.

### 9.2 ImaginationMamba -- Correct

Uses `MambaBlock.step()` for single-token recurrence (v0.4 fix). State persistence across 32 imagination steps is explicitly managed.

**Note**: Inherits the `_step_official` residual bug from `MambaBlock` (section 2.1).

### 9.3 ObjectPhysicsEngine -- Correct

6-layer attention GNN with inertia bias (residual prediction). Intrinsic/extrinsic attribute separation for invariance loss.

### 9.4 NoiseAugmentation -- Correct

GameNGen-style: linear noise schedule `sigma = max_sigma * (step/total)`. Noise level discretized into 16 buckets for the embedding.

### 9.5 WorldModelHeads -- Correct

DreamerV3-style symlog two-hot regression for reward/value, Bernoulli for done. `SymlogTwoHot` encoding and decoding are correct.

### 9.6 KL Loss -- Correct (v0.4 fix verified)

Per-category free-bits clamping before summing. KL balancing with alpha=0.8 (forward) + (1-alpha)=0.2 (reverse). Matches DreamerV3.

---

## 10. Full Model Assembly (`hybrid_vla_v2.py`)

### 10.1 Cond Prefix -- Correct

Token count: `1 + 24 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 32` = `cond_tokens`. Padding/truncation logic handles edge cases.

### 10.2 Core-to-Expert Projection -- Correct

`2048d -> 1536d` via `nn.Linear` when dimensions differ. Applied to cond_prefix, proprio, and embodiment tokens.

### 10.3 Stage Gating -- Correct

- Stage A: expert frozen, only backbone LoRA + grounder + core + heads train
- Stage B: expert unfrozen with `cond_prefix.detach()` (knowledge insulation)
- Stage C: full fine-tune

### 10.4 Training Loop -- Correct

Temporal processing iterates T steps with proper refresh/medium scheduling. Action history buffer maintained across steps.

---

## Summary of Issues

### Critical

| # | Module | Issue | Impact |
|---|--------|-------|--------|
| 1 | `MambaBlock._forward_official()` | Missing residual connection + pre-norm when using official Mamba2 CUDA path | 20/6/10-layer stacks without skip connections. Severely degraded gradient flow and output quality. |
| 2 | `MambaBlock._step_official()` | Same as above for single-token inference path | Affects both temporal core and imagination engine |

### High

| # | Module | Issue | Impact |
|---|--------|-------|--------|
| 3 | `Qwen2VLBackboneWrapper._apply_freeze()` | Vision tower freeze checks wrong attribute path (`self.model.visual` vs `self.model.model.visual`) | Vision tower may not be frozen; ~600M extra trainable params. Version-dependent. |
| 4 | `Qwen2VLBackboneWrapper._apply_freeze()` | Text layer freeze uses `self.model.model.layers` but latest transformers uses `self.model.model.language_model.layers` | Text layers 0-15 may not be frozen. Version-dependent. |

### Medium

| # | Module | Issue | Impact |
|---|--------|-------|--------|
| 5 | `MambaBlock._forward_fallback()` | Conv state + symmetric padding causes wrong truncation | First ~3 tokens of each temporal step corrupted in fallback path |
| 6 | `MambaBlock` | Fallback (Mamba-1) and official (Mamba-2) are architecturally different models | Cannot switch backends without retraining |

### Low / Cosmetic

| # | Module | Issue | Impact |
|---|--------|-------|--------|
| 7 | `_MambaStack.init_states()` | Creates Mamba-1-shaped states; wrong for Mamba-2 | Latent bug (method currently unused) |
| 8 | `FlowMatchingLoss` | `sigma_min` parameter is unused | Dead code |
| 9 | Embeddings | [cos,sin] vs [sin,cos] ordering inconsistency | No functional impact |

---

## Recommendations

1. **Fix the Mamba2 residual** (Critical): Add `out = out + x` in both `_forward_official` and `_step_official`. Consider also adding a LayerNorm for the pre-norm if parity with the fallback path is desired.

2. **Verify transformers version**: Run `python -c "from transformers import Qwen2VLForConditionalGeneration; m = Qwen2VLForConditionalGeneration.__init__.__code__; print(m.co_varnames)"` to check attribute names. Pin the transformers version in requirements.

3. **Fix conv_state truncation**: When `conv_state` is provided, use either `padding=0` on the Conv1d or adjust the output slice to `[:, :, (d_conv-1):(d_conv-1)+L]`.

4. **Add a regression test**: Create a small-scale test that compares official vs fallback path outputs (with residual fix applied) to catch future divergence.

---
---

# HybridVLA v2 正确性分析（中文版）

对比参考:
- **Qwen2-VL**: [huggingface/transformers `modeling_qwen2_vl.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)
- **Mamba-2**: [state-spaces/mamba `mamba2.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py)
- **Mamba-1**: [state-spaces/mamba `mamba_simple.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py)
- **selective_scan**: [state-spaces/mamba `selective_scan_interface.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py)

---

## 1. Qwen2-VL Backbone (`qwen2vl_backbone.py`)

### 1.1 模型加载与架构常量

| 检查项 | 项目实现 | 官方 (7B) | 状态 |
|--------|---------|-----------|------|
| 模型类 | `Qwen2VLForConditionalGeneration` | `Qwen2VLForConditionalGeneration` | **正确** |
| hidden_size | 3584 | 3584 | **正确** |
| num_layers | 28 | 28 | **正确** |
| IMAGE_TOKEN_ID | 151655 | 151655 (`config.image_token_id`) | **正确** |
| VIDEO_TOKEN_ID | 151656 | 151656 (`config.video_token_id`) | **正确** |
| attn_implementation | `flash_attention_2` | `from_pretrained` 支持此参数 | **正确** |
| `output_hidden_states=True` | 用于提取多尺度特征 | 返回逐层 hidden states | **正确** |

### 1.2 BUG - 视觉塔冻结路径错误（高危）

**文件**: `qwen2vl_backbone.py:110-113`

```python
# 当前代码:
if freeze_vision and hasattr(self.model, "visual"):
    for p in self.model.visual.parameters():
        p.requires_grad = False
```

在官方 HuggingFace 实现中:
- `Qwen2VLForConditionalGeneration.model` -> `Qwen2VLModel`
- `Qwen2VLModel.visual` -> `Qwen2VisionTransformerPretrainedModel`

因此 `self.model`（即 `Qwen2VLForConditionalGeneration`）上并**没有** `.visual` 属性。正确路径应为 `self.model.model.visual`。

**结果**: `hasattr(self.model, "visual")` 返回 `False` -> 视觉塔**永远不会被冻结**，导致约 6 亿额外可训练参数参与训练，可能破坏 LoRA 微调的稳定性。

**修复方案**:
```python
composite_model = getattr(self.model, "model", self.model)
if freeze_vision and hasattr(composite_model, "visual"):
    for p in composite_model.visual.parameters():
        p.requires_grad = False
```

**注意**: 此问题可能与 transformers 版本有关。部分旧版本采用更扁平的结构，`ForConditionalGeneration` 上可能直接存在 `.visual` 属性。建议通过 `print(type(self.model))` 和 `dir(self.model)` 验证。

### 1.3 BUG - 文本层冻结路径错误（中危）

**文件**: `qwen2vl_backbone.py:114-119`

```python
text_model = self.model.model
if hasattr(text_model, "layers"):
    for i, layer in enumerate(text_model.layers):
```

在最新版 transformers 中，`self.model.model` = `Qwen2VLModel`，其文本模型在 `self.language_model`（`Qwen2VLTextModel`）中，而 `.layers` 在 `Qwen2VLTextModel` 上。因此 `text_model.layers` 会失败 -- 应为 `text_model.language_model.layers`。

**结果**: 与 1.2 同样依赖版本。若使用最新 transformers API（`Qwen2VLModel` -> `language_model` -> `layers`），文本层 0-15 将**不会被冻结**。

### 1.4 多尺度特征提取 -- 正确（有注意事项）

**文件**: `qwen2vl_backbone.py:181-186`

```python
all_hidden = outputs.hidden_states
for layer_idx in self.multi_scale_layers:  # [10, 18, 28]
    idx = min(layer_idx, len(all_hidden) - 1)
    multi_scale.append(all_hidden[idx])
```

**传统** HuggingFace 约定: `hidden_states` = `(embedding, layer_0, ..., layer_27)` -> 29 个元素。
**较新** `@capture_outputs` API: `hidden_states` = `(layer_0, ..., layer_27)` -> 28 个元素。

- 29 个元素时: index 28 = 第 27 层输出（最后一层）-- 正确
- 28 个元素时: index 28 被 `min()` 截断为 27 = 第 27 层输出 -- 最后一层仍正确，但 index 10 和 18 会偏移 1 层

**结论**: 由于 `min()` 截断机制，代码安全。语义意图（提取早/中/晚期特征）在两种情况下均可保持。1 层偏移可忽略不计。

### 1.5 MultiScaleAdapter -- 正确

**文件**: `qwen2vl_backbone.py:24-53`

- FPN 风格的多尺度融合：每个尺度独立 LayerNorm + Linear 投影
- 通过 softmax 学习每个尺度的门控权重
- 加权求和融合

标准做法，无正确性问题。门控通过全局平均池化计算尺度权重后广播，设计合理。

### 1.6 LoRA 应用 -- 正确

**文件**: `qwen2vl_backbone.py:124-146`

- 使用 PEFT `LoraConfig`，`layers_to_transform=list(range(total_layers))`
- 目标涵盖所有标准线性模块: q/k/v/o_proj + gate/up/down_proj
- `task_type="CAUSAL_LM"` 与模型类型匹配

**正确。** `layers_to_transform` 参数正确覆盖全部 28 层，符合 v2 设计。

### 1.7 Processor 与 Token 检测 -- 正确

**文件**: `qwen2vl_backbone.py:87-96`

当 tokenizer 属性不存在时回退到硬编码常量。良好的防御性编程模式。

---

## 2. Mamba 核心 (`mamba_core.py`)

### 2.1 严重 BUG - 官方 Mamba2 路径缺少残差连接 + 前置归一化

**文件**: `mamba_core.py:234-243`

```python
def _forward_official(self, x: Tensor) -> Tuple[Tensor, None, None]:
    ...
    out = self.mamba(x)  # 注释写着 "residual included inside Mamba2"
    ...
    return out, None, None
```

注释 **"residual included inside Mamba2"** 是**错误的**。

根据官方 `mamba_ssm` 源码，`Mamba2.forward()` 的计算流程为:
```
in_proj -> conv1d -> SiLU -> SSM_scan -> gated_RMSNorm -> out_proj
```

它**不包含**:
1. **前置归一化**（输入 LayerNorm）-- 必须由外部 `Block` 包装器负责
2. **残差连接**（`x + out`）-- 必须由外部添加

回退路径正确实现了两者:
```python
def _forward_fallback(self, x, ...):
    residual = x
    x = self.norm(x)          # <- 前置归一化
    ...
    out = self.out_proj(y) + residual   # <- 残差连接
```

**影响**: 对于 Fast 流（20 层）、Medium 流（6 层）和 Slow 流（10 层），使用 CUDA 路径时所有层均缺少残差连接和前置归一化。这意味着:
- 20 层堆叠中**没有跳跃连接** -> 严重的梯度流问题
- **没有前置归一化** -> 激活值幅度不稳定
- 与回退路径相比，模型将产生截然不同（且大概率更差）的输出

**同样的 bug 影响 `_step_official()`**（第 202-230 行）: `self.mamba.step()` 同样缺少残差和前置归一化。

**`_forward_official` 修复方案**:
```python
def _forward_official(self, x: Tensor) -> Tuple[Tensor, None, None]:
    is_single = x.dim() == 2
    if is_single:
        x = x.unsqueeze(1)
    residual = x
    out = self.mamba(x)
    out = out + residual       # <- 添加残差
    if is_single:
        out = out.squeeze(1)
    return out, None, None
```

**`_step_official` 修复方案** -- 在 step 后添加残差:
```python
def _step_official(self, x, ssm_state, conv_state):
    ...
    out, new_conv_state, new_ssm_state = self.mamba.step(x, conv_state, ssm_state)
    out = out + x  # <- 添加残差
    return out, new_ssm_state, new_conv_state
```

**补充说明**: 官方 `Mamba2` 模块内部已包含 gated RMSNorm（在 `out_proj` 之前），因此单独的逐层前置归一化可能并非严格必要 -- 但残差连接是**必须的**。

### 2.2 架构不匹配: 回退路径 = Mamba-1，官方路径 = Mamba-2

回退路径实现的是 **Mamba-1** 架构:

| 方面 | 回退路径 (Mamba-1) | 官方路径 (Mamba-2) |
|------|-------------------|-------------------|
| A 的形状 | `(d_inner, d_state)` | `(nheads,)` -- 每个头一个标量 |
| SSM 状态 | `(B, d_inner, d_state)` | `(B, nheads, headdim, d_state)` |
| dt 投影 | 低秩: `dt_rank -> d_inner` | 直接: 从 `in_proj` 输出 `nheads` 个值 |
| Conv1d 作用于 | 仅 `x`（`d_inner` 通道） | `(x, B, C)` 拼接后 |
| 门控机制 | `y * SiLU(z)` | Gated RMSNorm: `RMSNorm(y) * SiLU(z)` |
| 扫描算法 | 顺序递归 | 分块 SSD (chunk_size=256) |

**严重性**: 开发阶段为低（代码已有相关说明）。但若使用一种后端训练、另一种后端部署则为高危 -- 输出将完全不同。两条路径**不可互换**。

### 2.3 BUG - 回退路径的 Conv State 处理错误（中危）

**文件**: `mamba_core.py:267-273`

```python
x_conv = x_main.transpose(1, 2)  # [B, d_inner, L]
if conv_state is not None:
    x_conv = torch.cat([conv_state, x_conv], dim=-1)  # [B, d_inner, (d_conv-1)+L]
x_conv = self.conv1d(x_conv)     # Conv1d 使用 padding=d_conv-1（两侧对称填充！）
x_conv = x_conv[:, :, :L]        # <- 截断
```

`nn.Conv1d(padding=d_conv-1)` 在**两侧**各填充 `d_conv-1` 个零。当 `conv_state` 被拼接在前面时，左侧的零填充会覆盖历史上下文。

**示例** (d_conv=4, L=2, conv_state=[a,b,c]):
```
拼接后输入: [a, b, c, x1, x2]                      (长度 5)
Conv1d 填充后: [0, 0, 0, a, b, c, x1, x2, 0, 0, 0]  (长度 11)
输出长度:     11 - 4 + 1 = 8

Output[0] = conv(0, 0, 0, a)     <- 不相关
Output[3] = conv(a, b, c, x1)    <- 我们需要的 x1 对应输出
Output[4] = conv(b, c, x1, x2)   <- 我们需要的 x2 对应输出

截断 [:L=2]: 取 Output[0] 和 Output[1]  <- 错误的位置！
```

当提供 `conv_state` 时，正确的截断应为 `[:, :, (d_conv-1):(d_conv-1)+L]`。

**影响**: 影响回退路径中每个时间步的前 ~3 个 token。由于输入序列为 `[global, phase, uncertainty, affordance, proprio, prev_action, stale, embodiment, action_history, slots...]`，受影响的位置包含重要的语义 token。仅影响回退路径（官方路径使用 `Mamba2.step()` 正确处理 conv state）。

### 2.4 _MambaStack.init_states() -- 潜在 Bug

**文件**: `mamba_core.py:326-344`

```python
ssm = [torch.zeros(batch_size, self.d_inner, self.d_state, ...) ...]
```

创建的是 Mamba-1 形状的 SSM 状态 `(B, d_inner, d_state)`。对于官方 Mamba2 路径，正确形状应为 `(B, nheads, headdim, d_state)`。目前在主代码路径中未被调用（状态初始化为 `None`），但若被使用将产生错误的形状。

### 2.5 三速率设计 -- 正确

| 流 | 层数 | d_state | 更新规则 | 用途 |
|----|------|---------|----------|------|
| Fast | 20 | 128 | 每步更新 | 反应式控制 |
| Medium | 6 | 128 | 每 2 步更新 | 策略规划 |
| Slow | 10 | 256 | 仅语义刷新时 | 战略上下文 |

- `TriRateMambaCore.forward()` 中的更新调度逻辑正确
- 流不更新时的状态缓存（复用上次 token）正确
- `steps_since_refresh` / `steps_since_medium` 计数器正确
- 交叉注意力融合替代标量门控（v1->v2 升级）-- 实现正确

### 2.6 CrossAttentionFusion -- 正确

**文件**: `mamba_core.py:480-537`

- 2 层交叉注意力，学习的融合查询关注 [fast, medium, slow] token
- 通过加性投影注入过时时间条件
- 带 FFN 的 pre-norm 交叉注意力 -- 标准架构

### 2.7 ActionHistoryEncoder -- 正确

**文件**: `mamba_core.py:451-473`

- 将动作 [B, K, A] -> [B, K, d_model] 投影后通过 4 层 Mamba 处理，取最后一个 token
- 使用 `_MambaStack`，继承了相同的官方/回退路径问题

### 2.8 StaleTimeEncoding -- 正确

**文件**: `mamba_core.py:47-72`

标准正弦位置编码 + MLP。截断到 `max_staleness=256`。

---

## 3. 选择性扫描 (`ops/selective_scan.py`)

### 3.1 JIT 编译的 `ssm_scan` -- 正确

**文件**: `selective_scan.py:37-55`

```python
@torch.jit.script
def ssm_scan(dA, dBx, C, state):
    for t in range(L):
        state = dA[:, t] * state + dBx[:, t]
        y[:, t] = (state * C[:, t].unsqueeze(1)).sum(-1)
    return y, state
```

标准 SSM 递推公式: `h_t = A_t * h_{t-1} + B_t * x_t; y_t = C_t * h_t`。已预离散化（dA, dBx 预先计算完成）。**数学上正确。**

### 3.2 CUDA 导入 -- 正确

从 `mamba_ssm.ops.selective_scan_interface` 导入 `selective_scan_fn`，从 `causal_conv1d` 导入 `causal_conv1d_fn`。不可用时优雅回退。

---

## 4. Flow Action Expert (`flow_action_expert.py`)

### 4.1 ExpertMambaBlock -- 正确

**文件**: `flow_action_expert.py:90-157`

与 `MambaBlock` 不同，专家模块始终使用自己的 pre-norm + 残差，与后端无关:

```python
def forward(self, x, t_cond):
    residual = x
    x = self.norm(x, t_cond)  # AdaRMSNorm（始终应用）
    ...
    return residual + self.out_proj(y)  # 残差（始终应用）
```

CUDA 路径正确使用 `selective_scan_fn`:
- 张量布局符合官方规范: `u/dt` 为 `(B, D, L)`, `A` 为 `(D, N)`, `B/C` 为 `(B, N, L)`
- `delta_softplus=True`，`D` 和 `z` 门控传入内核
- dt_proj 的偏置嵌入在 Linear 层中（与单独传递 `delta_bias` 数学等价）

### 4.2 AdaRMSNorm -- 正确

**文件**: `flow_action_expert.py:31-49`

实现 `sigmoid(gate) * (RMSNorm(x) * (1 + scale) + shift)`，其中 `(scale, shift, gate)` 从条件向量投影得到。匹配 pi-0.5 (Physical Intelligence) 风格的自适应归一化。

### 4.3 ExpertAttentionBlock -- 正确

**文件**: `flow_action_expert.py:164-222`

三部分结构: 交叉注意力 -> 自注意力 -> FFN，全部使用 AdaRMSNorm。使用 `F.scaled_dot_product_attention` 自动调度 Flash-Attention。头维度处理正确。

### 4.4 ODE 求解器 -- 正确

**Euler** (`sample_euler`): `x_{i+1} = x_i + dt * v(x_i, t_i)` -- 标准显式 Euler。
**Midpoint** (`sample_midpoint`): `x_mid = x_i + 0.5*dt*v(x_i, t_i); x_{i+1} = x_i + dt*v(x_mid, t_mid)` -- 标准 RK2 中点法。相同代价下精度约为 Euler 的 2 倍。

两者从 t=0 积分到 t~1，`dt = 1/num_steps`。**正确。**

### 4.5 时间步嵌入 -- 正确（轻微不一致）

`SinusoidalTimestepEmbedding` 使用 **[cos, sin]** 顺序。
`StaleTimeEncoding` 使用 **[sin, cos]** 顺序。

两者均有效；后接的 MLP 会自适应任一顺序。功能等价但不一致。

---

## 5. Attention Grounder (`attention_grounder.py`)

### 5.1 架构 -- 正确

Perceiver 风格: 96 个可学习的潜在查询通过 8 个 GrounderBlock 交叉注意力关注骨干特征。每个 Block = 交叉注意力 + 自注意力 + FFN。

槽位布局: `[global(1), objects(48), phase(1), unc(1), aff(1), aux(44)] = 96`

### 5.2 分层压缩 -- 正确

在第 4 层后，目标槽位（48 -> 24）通过可学习的交叉注意力路由查询进行压缩。压缩后布局: `[global(1), compressed(24), phase(1), unc(1), aff(1), aux(44)] = 72`。第 5-7 层以 72 个 token 继续处理。

输出中的槽位切分正确处理了压缩和非压缩两种路径。

### 5.3 交叉/自注意力 -- 正确

使用 `F.scaled_dot_product_attention`，支持正确的掩码处理。全程使用 pre-norm 模式。

---

## 6. 离散头 (`discrete_heads.py`)

### 6.1 FASTDiscreteHead -- 正确

- 离散化: `[-1, 1] -> [0, V-1]` 线性映射。逆映射精确。
- 架构: `LayerNorm -> Linear -> GELU -> step_proj -> per-dim head`
- 分解式: 共享编码器 + 逐步 + 逐维度投影

### 6.2 PhaseHead / AffordanceHead -- 正确

基于对应 grounder token 的简单 MLP 分类器。

---

## 7. Flow Matching 损失 (`losses/flow_matching.py`)

### 7.1 速度场目标 -- 正确

```python
target_velocity = x_1 - x_0  # 常速度场（最优传输）
```

对于线性插值 `x_t = (1-t)*x_0 + t*x_1`，速度为 `dx_t/dt = x_1 - x_0`。

### 7.2 Logit-Normal 采样 -- 正确

```python
torch.sigmoid(torch.randn(batch_size, device=device))
```

从 logit-normal 分布中采样，以 t=0.5 为中心，将训练集中在中间噪声水平。标准做法。

### 7.3 无用参数（外观问题）

`interpolate()` 中的 `sigma_min` 参数被接受但从未使用。

---

## 8. 一致性损失 (`losses/consistency_loss.py`)

### 8.1 ContrastiveTemporalLoss -- 正确

InfoNCE 风格: 相邻融合状态构成正样本对，其他配对为负样本。点积前进行 L2 归一化。标准对比学习。

### 8.2 SlowFastAgreementLoss -- 正确

fast token 的指数加权移动平均应与 slow token 匹配。促进跨流的时间一致性。

### 8.3 ActionConsistencyLoss -- 正确

将离散和连续动作预测投影到共享嵌入空间，最小化余弦距离。

---

## 9. World Model 组件

### 9.1 StochasticStateModule -- 正确

DreamerV3 风格: 48x48 类别分布，1% 均匀混合（`unimix=0.01`）。通过 `sample + probs - probs.detach()` 实现直通梯度。后验以 `(z_det, obs_encoding)` 为条件，先验仅以 `z_det` 为条件。

### 9.2 ImaginationMamba -- 正确

使用 `MambaBlock.step()` 进行单 token 递推（v0.4 修复）。显式管理跨 32 个想象步骤的状态持久化。

**注意**: 继承了 `MambaBlock` 的 `_step_official` 残差 bug（见 2.1 节）。

### 9.3 ObjectPhysicsEngine -- 正确

6 层注意力 GNN，带惯性偏置（残差预测）。内禀/外禀属性分离用于不变性损失。

### 9.4 NoiseAugmentation -- 正确

GameNGen 风格: 线性噪声调度 `sigma = max_sigma * (step/total)`。噪声等级离散化为 16 个桶用于嵌入。

### 9.5 WorldModelHeads -- 正确

DreamerV3 风格 symlog two-hot 回归用于 reward/value，伯努利分布用于 done。`SymlogTwoHot` 的编码和解码均正确。

### 9.6 KL 损失 -- 正确（v0.4 修复已验证）

每个类别独立应用 free-bits 截断后再求和。KL 平衡: alpha=0.8（正向）+ (1-alpha)=0.2（反向）。与 DreamerV3 一致。

---

## 10. 完整模型组装 (`hybrid_vla_v2.py`)

### 10.1 条件前缀 -- 正确

Token 数量: `1 + 24 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 32` = `cond_tokens`。填充/截断逻辑正确处理边界情况。

### 10.2 核心到专家的投影 -- 正确

维度不同时通过 `nn.Linear` 实现 `2048d -> 1536d`。应用于 cond_prefix、proprio 和 embodiment token。

### 10.3 阶段门控 -- 正确

- 阶段 A: 专家冻结，仅训练骨干 LoRA + grounder + core + heads
- 阶段 B: 专家解冻，`cond_prefix.detach()`（知识隔离）
- 阶段 C: 全量微调

### 10.4 训练循环 -- 正确

时间处理遍历 T 步，语义刷新/中速更新调度正确。动作历史缓冲区在步骤间正确维护。

---

## 问题汇总

### 严重 (Critical)

| # | 模块 | 问题 | 影响 |
|---|------|------|------|
| 1 | `MambaBlock._forward_official()` | 使用官方 Mamba2 CUDA 路径时缺少残差连接 + 前置归一化 | 20/6/10 层堆叠无跳跃连接。梯度流和输出质量严重退化。 |
| 2 | `MambaBlock._step_official()` | 同上，影响单 token 推理路径 | 同时影响时序核心和想象引擎 |

### 高危 (High)

| # | 模块 | 问题 | 影响 |
|---|------|------|------|
| 3 | `Qwen2VLBackboneWrapper._apply_freeze()` | 视觉塔冻结检查了错误的属性路径（`self.model.visual` vs `self.model.model.visual`） | 视觉塔可能未被冻结；约 6 亿额外可训练参数。版本依赖。 |
| 4 | `Qwen2VLBackboneWrapper._apply_freeze()` | 文本层冻结使用 `self.model.model.layers`，但最新 transformers 使用 `self.model.model.language_model.layers` | 文本层 0-15 可能未被冻结。版本依赖。 |

### 中危 (Medium)

| # | 模块 | 问题 | 影响 |
|---|------|------|------|
| 5 | `MambaBlock._forward_fallback()` | Conv state + 对称填充导致错误截断 | 回退路径中每个时间步的前 ~3 个 token 被污染 |
| 6 | `MambaBlock` | 回退路径（Mamba-1）与官方路径（Mamba-2）是架构完全不同的模型 | 无法在不重新训练的情况下切换后端 |

### 低危 / 外观问题 (Low / Cosmetic)

| # | 模块 | 问题 | 影响 |
|---|------|------|------|
| 7 | `_MambaStack.init_states()` | 创建 Mamba-1 形状的状态；对 Mamba-2 不正确 | 潜在 bug（该方法目前未使用） |
| 8 | `FlowMatchingLoss` | `sigma_min` 参数未使用 | 死代码 |
| 9 | 嵌入模块 | [cos,sin] vs [sin,cos] 顺序不一致 | 无功能影响 |

---

## 修复建议

1. **修复 Mamba2 残差连接**（严重）: 在 `_forward_official` 和 `_step_official` 中添加 `out = out + x`。如果需要与回退路径保持一致，还应考虑添加 LayerNorm 作为前置归一化。

2. **验证 transformers 版本**: 运行 `python -c "from transformers import Qwen2VLForConditionalGeneration; m = Qwen2VLForConditionalGeneration.__init__.__code__; print(m.co_varnames)"` 检查属性名。在 requirements 中固定 transformers 版本。

3. **修复 conv_state 截断**: 当提供 `conv_state` 时，要么对 Conv1d 使用 `padding=0`，要么将输出切片调整为 `[:, :, (d_conv-1):(d_conv-1)+L]`。

4. **添加回归测试**: 创建小规模测试用例，比较官方路径与回退路径（修复残差后）的输出，以捕获未来的偏差。
