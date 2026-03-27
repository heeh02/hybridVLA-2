# HybridVLA v0.7 Recovery Log

Based on the comprehensive correctness analysis in `accuracy.md`, this document records all fixes applied in v0.7.

---

## Fix Summary

| # | Severity | Module | Issue | File | Status |
|---|----------|--------|-------|------|--------|
| 1 | **CRITICAL** | `MambaBlock._forward_official()` | Missing pre-norm + residual connection | `mamba_core.py:236-247` | **FIXED** |
| 2 | **CRITICAL** | `MambaBlock._step_official()` | Missing pre-norm + residual connection | `mamba_core.py:227-232` | **FIXED** |
| 3 | **HIGH** | `Qwen2VLBackboneWrapper._apply_freeze()` | Vision tower freeze checks wrong attribute path | `qwen2vl_backbone.py:110-134` | **FIXED** |
| 4 | **HIGH** | `Qwen2VLBackboneWrapper._apply_lora()` | Text layer count uses wrong attribute path | `qwen2vl_backbone.py:142-145` | **FIXED** |
| 5 | **MEDIUM** | `MambaBlock._forward_fallback()` | Conv state + symmetric padding causes wrong output truncation | `mamba_core.py:273-286` | **FIXED** |
| 6 | **LOW** | `_MambaStack.init_states()` | Creates Mamba-1-shaped states, wrong for Mamba-2 official path | `mamba_core.py:340-384` | **FIXED** |
| 7 | **COSMETIC** | `FlowMatchingLoss` | Dead `sigma_min` parameter never used | `flow_matching.py:10-32` | **FIXED** |

Items verified as correct (no fix needed):
- Multi-scale feature extraction (safe via `min()` clamping)
- LoRA application (all 28 layers, correct target modules)
- Tri-Rate scheduling (fast/medium/slow update logic)
- CrossAttentionFusion (2-layer cross-attention with stale-time conditioning)
- ExpertMambaBlock (already has its own pre-norm + residual, unaffected)
- ODE solvers (Euler + Midpoint)
- Selective scan JIT kernel
- All loss functions (flow matching, consistency, KL)
- World model components (StochasticStateModule, ObjectPhysicsEngine, etc.)

---

## Fix #1 & #2 (CRITICAL): MambaBlock Official Path Missing Residual + Pre-Norm

### Problem

The official Mamba2 CUDA path (`_forward_official` and `_step_official`) was missing both:
1. **Pre-normalization** (LayerNorm before the SSM)
2. **Residual connection** (`out = out + x`)

The comment `"residual included inside Mamba2"` in `_forward_official` was **incorrect**. The official `mamba_ssm.Mamba2` module performs:

```
in_proj -> conv1d -> SiLU -> SSM_scan -> gated_RMSNorm -> out_proj
```

It does NOT include pre-norm or residual connections. Those are the responsibility of the wrapping `Block` class in `mamba_ssm`, which our code does not use.

The fallback path correctly implemented both:
```python
def _forward_fallback(self, x, ...):
    residual = x
    x = self.norm(x)          # pre-norm
    ...
    out = self.out_proj(y) + residual   # residual
```

### Impact

- **Fast stream** (20 layers), **Medium stream** (6 layers), and **Slow stream** (10 layers) all ran without skip connections when using the CUDA path
- No skip connections across a 20-layer stack causes severe gradient vanishing
- No pre-normalization causes activation magnitude instability
- ImaginationMamba (world model) inherits the same bug via `MambaBlock.step()`
- The model would produce fundamentally different (and degraded) outputs compared to the fallback path

### Fix Applied

**File**: `vla_hybrid_v2/models/mamba_core.py`

1. Added `self.norm = nn.LayerNorm(d_model)` to the official branch of `__init__` (line 112):
```python
if HAS_MAMBA_SSM:
    self.mamba = _OfficialMamba2(...)
    self.norm = nn.LayerNorm(d_model)  # v0.7: pre-norm for official path
    self._use_official = True
```

2. Fixed `_forward_official` (lines 236-247):
```python
def _forward_official(self, x):
    is_single = x.dim() == 2
    if is_single:
        x = x.unsqueeze(1)
    residual = x
    out = self.mamba(self.norm(x))  # v0.7: pre-norm before Mamba2
    out = out + residual            # v0.7: residual connection
    if is_single:
        out = out.squeeze(1)
    return out, None, None
```

3. Fixed `_step_official` (lines 227-232):
```python
out, new_conv_state, new_ssm_state = self.mamba.step(
    self.norm(x), conv_state, ssm_state,  # v0.7: pre-norm
)
out = out + x  # v0.7: residual connection
```

### Verification

Both paths now follow the same pre-norm residual pattern:
```
out = x + Mamba(LayerNorm(x))
```

This matches the docstring on `MambaBlock` itself (line 86-88):
```
The block always follows the pre-norm residual pattern:
    out = x + out_proj(SSM(SiLU(conv1d(in_proj(LN(x))))))
```

---

## Fix #3 & #4 (HIGH): Qwen2VL Vision/Text Freeze Path

### Problem

In HuggingFace transformers, the model hierarchy is:
```
Qwen2VLForConditionalGeneration  (self.model)
  └── .model -> Qwen2VLModel
        ├── .visual -> Qwen2VisionTransformerPretrainedModel
        └── .language_model (newer API) / .layers (older API)
```

The original code checked `hasattr(self.model, "visual")`, but `self.model` is `Qwen2VLForConditionalGeneration`, which does NOT have `.visual` directly. The correct path is `self.model.model.visual`.

Similarly, `self.model.model.layers` may not exist in newer transformers versions where the text model is accessed via `self.model.model.language_model.layers`.

### Impact

- `hasattr(self.model, "visual")` returns `False` -> vision tower **never frozen**
- ~600M extra trainable parameters participate in training, potentially destabilizing LoRA fine-tuning
- Text layers 0-15 may not be frozen if using newer transformers API

### Fix Applied

**File**: `vla_hybrid_v2/models/qwen2vl_backbone.py`

1. Fixed `_apply_freeze` (lines 110-134):
```python
def _apply_freeze(self, freeze_vision, freeze_until):
    # Navigate through ForConditionalGeneration -> Model -> visual
    composite_model = getattr(self.model, "model", self.model)
    if freeze_vision and hasattr(composite_model, "visual"):
        for p in composite_model.visual.parameters():
            p.requires_grad = False

    # Handle both old layout (.layers) and new layout (.language_model.layers)
    text_model = composite_model
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model
    if hasattr(text_model, "layers"):
        for i, layer in enumerate(text_model.layers):
            if i < freeze_until:
                for p in layer.parameters():
                    p.requires_grad = False
    # Also freeze embed_tokens (check both locations)
    if hasattr(text_model, "embed_tokens"):
        for p in text_model.embed_tokens.parameters():
            p.requires_grad = False
    elif hasattr(composite_model, "embed_tokens"):
        for p in composite_model.embed_tokens.parameters():
            p.requires_grad = False
```

2. Fixed `_apply_lora` layer count (lines 142-145):
```python
text_model = getattr(self.model, "model", self.model)
if hasattr(text_model, "language_model"):
    text_model = text_model.language_model
total_layers = len(text_model.layers) if hasattr(text_model, "layers") else 28
```

### Compatibility

The fix uses `getattr(..., "model", self.model)` and `hasattr` checks, making it compatible with both old and new transformers versions.

---

## Fix #5 (MEDIUM): Conv State Truncation in Fallback Path

### Problem

When `conv_state` (historical context of length `d_conv-1`) is prepended to the input, `nn.Conv1d(padding=d_conv-1)` adds `d_conv-1` zeros on **both** sides. The zero-padding on the left overwrites the historical context that was prepended.

Example with `d_conv=4`, `L=2`, `conv_state=[a,b,c]`:
```
Concatenated:     [a, b, c, x1, x2]             (length 5)
After pad:        [0, 0, 0, a, b, c, x1, x2, 0, 0, 0]  (length 11)
Conv output:      8 elements
Truncated [:L=2]: Output[0], Output[1]           <- WRONG! These use zero padding, not history
Correct [3:5]:    Output[3], Output[4]           <- These use (a,b,c,x1) and (b,c,x1,x2)
```

### Impact

First ~3 tokens of every temporal step corrupted in the fallback path. Since the input sequence starts with `[global, phase, uncertainty, affordance, ...]`, important semantic tokens were affected.

### Fix Applied

**File**: `vla_hybrid_v2/models/mamba_core.py`, lines 273-286

```python
if conv_state is not None:
    x_conv = torch.cat([conv_state, x_conv], dim=-1)
    x_conv = self.conv1d(x_conv)
    # Take the correct causal slice after convolution
    x_conv = x_conv[:, :, (self.d_conv - 1):(self.d_conv - 1) + L]
else:
    x_conv = self.conv1d(x_conv)
    x_conv = x_conv[:, :, :L] if x_conv.shape[-1] > L else x_conv
```

When `conv_state` is prepended, the output is now sliced at `[d_conv-1 : d_conv-1+L]`, which skips the zero-padded prefix and takes exactly the `L` causally-correct output positions.

---

## Fix #6 (LOW): `_MambaStack.init_states()` Wrong Shape for Mamba-2

### Problem

`init_states()` always created Mamba-1-shaped SSM states `(B, d_inner, d_state)` and conv states `(B, d_inner, d_conv-1)`. For the official Mamba-2 path, the correct shapes are:
- SSM: `(B, nheads, headdim, d_state)`
- Conv: `(B, d_inner + 2*ngroups*d_state, d_conv)`

This was a latent bug (the method was not called in the main code path, as states initialize as `None`), but would cause shape mismatches if ever used.

### Fix Applied

**File**: `vla_hybrid_v2/models/mamba_core.py`, lines 340-384

`init_states()` now checks `self.layers[0]._use_official` and creates the appropriate state shapes for each backend, reading `nheads`, `headdim`, and `ngroups` from the official Mamba2 module.

---

## Fix #7 (COSMETIC): Dead `sigma_min` Parameter

### Problem

`FlowMatchingLoss.__init__` accepted a `sigma_min` parameter and stored it as `self.sigma_min`, but it was never used anywhere. The `interpolate()` static method also accepted `sigma_min` as an argument but ignored it. This is expected for Rectified Flow (linear interpolation), where `sigma_min` is a VP-SDE concept.

### Fix Applied

**File**: `vla_hybrid_v2/losses/flow_matching.py`

Removed `sigma_min` from `__init__`, removed `self.sigma_min`, and removed the `sigma_min` parameter from `interpolate()`. No callers passed this argument.

---

## Items Not Fixed (By Design)

### Architecture Mismatch: Fallback = Mamba-1, Official = Mamba-2

The fallback path implements Mamba-1 architecture while the official path uses Mamba-2. These are fundamentally different SSM architectures (different A parameterization, different scan algorithm, different gating). The code already warns about this. This is by design -- the fallback exists for environments without CUDA. Training and inference must use the same backend.

### Sinusoidal Embedding Order Inconsistency

`SinusoidalTimestepEmbedding` uses [cos, sin] order while `StaleTimeEncoding` uses [sin, cos] order. Both are valid since the MLP on top adapts to either ordering. No functional impact.

---

## Files Modified

| File | Lines Changed |
|------|--------------|
| `vla_hybrid_v2/models/mamba_core.py` | +45, -15 |
| `vla_hybrid_v2/models/qwen2vl_backbone.py` | +16, -7 |
| `vla_hybrid_v2/losses/flow_matching.py` | +2, -4 |

---

## Recommended Follow-Up

1. **Pin transformers version** in `requirements.txt` and verify the Qwen2VL attribute paths match your pinned version. Run:
   ```python
   from transformers import Qwen2VLForConditionalGeneration
   m = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
   print(hasattr(m.model, "visual"))          # should be True after fix
   print(hasattr(m.model, "language_model"))  # check which text path is correct
   ```

2. **Add a regression test** comparing official vs fallback path outputs (with a small d_model) to catch future divergence.

3. **Verify checkpoint compatibility**: If pre-v0.7 checkpoints were saved while training with the official path (missing residual), the learned weights will have adapted to the broken forward pass. Loading such checkpoints into the fixed code will produce different outputs. Consider:
   - Restarting training from scratch, OR
   - Adding a flag to temporarily disable the residual for inference continuity, then gradually blending in the residual via warmup

4. **Test the conv_state fix** with a sequence of temporal steps to verify that the first few tokens now receive correct historical context.
