# HybridVLA v2 — v1.0 Independent Code Review

**Reviewer**: Claude Opus 4.6 (acting as independent auditor)
**Date**: 2026-03-29
**Scope**: All uncommitted changes on `main` (delta from commit `63f55df`)
**Method**: Read every diff + current file state. Zero trust in prior fix reports.

---

## Executive Summary

Reviewed 29 changed files, ~2400 lines delta. Found **2 P0**, **3 P1**, **4 P2** issues.
The P0s are both related to FSDP + version-upgrade checkpoint resume — fresh training
on a single GPU will not crash, but multi-GPU resume from v0.10.9 checkpoint will.

---

## P0 — Will Crash on Day 1 (Multi-GPU Version Upgrade)

### P0-1: Checkpoint shape-mismatch filtering is ineffective under FSDP auto-resume

**Severity**: P0 (crash on multi-GPU resume after code upgrade)
**Files**: `vla_hybrid_v2/utils/checkpointing.py:126-142`

**Evidence**:

```python
# load_checkpoint() — shape filter runs BEFORE the FSDP context
model_state = model.state_dict() if not hasattr(model, 'module') \
              else model.module.state_dict()
for k in list(state.keys()):
    if k in model_state and state[k].shape != model_state[k].shape:
        del state[k]
```

When `model` is FSDP-wrapped (as it is during `auto_resume` at line 450 of
`train_unified.py`), `model.module.state_dict()` returns state from the inner
module whose sub-modules are themselves FSDP-wrapped. The returned keys will
have FSDP prefixes and/or sharded shapes. The checkpoint `state` has raw
(unprefixed, full) keys. The `if k in model_state` check finds **zero** overlap,
so **no keys are ever filtered**.

Then `model.load_state_dict(state, strict=False)` is called inside the
FSDP `FULL_STATE_DICT` context. `strict=False` ignores missing/unexpected keys
but **does not tolerate shape mismatches** on matching keys. The resized
`action_history_encoder` keys (d=2048→d=256) will raise `RuntimeError`.

**Reproduction scenario**:
1. Train on multi-GPU with v0.10.9 (ActionHistoryEncoder d=2048), save checkpoint
2. Update code to v0.10.10 (ActionHistoryEncoder d=256)
3. Resume training → `auto_resume` → crash

**Why current fix is insufficient**:
The shape filter was designed for this exact scenario but doesn't work when the
model is FSDP-wrapped. The cross-stage path (before FSDP) works correctly because
the model is unwrapped there.

**Suggested minimal fix**:
Move the shape comparison inside the FSDP `state_dict_type(FULL_STATE_DICT)`
context, or collect full state dict within that context for comparison:

```python
if isinstance(model, FSDP):
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_cfg):
        model_state = model.state_dict()
        # Now filter shapes, then load
        ...
        model.load_state_dict(state, strict=strict)
```

**Minimal test**:
```python
def test_load_checkpoint_fsdp_shape_mismatch():
    """Shape-mismatched keys must be dropped even when model is FSDP-wrapped."""
    # Create two models with different ActionHistoryEncoder sizes
    # Wrap one with FSDP, save from old, load into new
    # Assert: no RuntimeError, mismatched keys dropped with warning
```

---

### P0-2: `_dict_to_dataclass` silently fails for nested dataclasses when `typing.get_type_hints()` raises

**Severity**: P0 (latent — currently masked because `get_type_hints()` succeeds)
**Files**: `vla_hybrid_v2/config.py:373-399`

**Evidence**:

`config.py` has `from __future__ import annotations` (line 11), so all
`__dataclass_fields__[k].type` values are **strings** (e.g., `"TrainConfig"`
instead of `<class TrainConfig>`).

```python
try:
    resolved_hints = typing.get_type_hints(cls)
except Exception:
    resolved_hints = {}

# ...
ft = resolved_hints.get(k, cls.__dataclass_fields__[k].type)
if isinstance(ft, type) and hasattr(ft, "__dataclass_fields__"):
    kwargs[k] = _dict_to_dataclass(ft, v)  # recurse
else:
    kwargs[k] = v  # pass raw dict — will TypeError later
```

If `get_type_hints()` fails (circular import, missing dependency at resolve time,
or running in a restricted environment), `resolved_hints = {}`. The fallback
`cls.__dataclass_fields__[k].type` returns `"ModelConfig"` (a string), not the
class. `isinstance("ModelConfig", type)` is `False`, so nested dataclasses are
passed as raw dicts → `TypeError` in the dataclass constructor.

The original code used `eval(ft, globals(), locals())` which resolved the string.
That was removed for security (L-19), but no functional fallback was added.

**Why not immediately crashing**: `typing.get_type_hints(HybridVLAv2Config)` currently
succeeds because all referenced types are defined in the same module. But any future
refactoring that moves config types to a separate module could trigger this.

**Why current fix is insufficient**:
The `except Exception: resolved_hints = {}` catch-all suppresses the error with
**no log message**. A user would see `TypeError: ModelConfig.__init__() got...`
with zero indication that `get_type_hints()` failed.

**Suggested minimal fix**:
```python
except Exception as e:
    warnings.warn(
        f"typing.get_type_hints({cls.__name__}) failed: {e}. "
        f"Nested dataclass fields may not be parsed correctly.",
        stacklevel=2,
    )
    resolved_hints = {}
```

And add a functional fallback for string types:
```python
if isinstance(ft, str):
    # Try resolving from module globals
    ft = vars(sys.modules[cls.__module__]).get(ft, ft)
```

**Minimal test**:
```python
def test_dict_to_dataclass_fallback_when_hints_fail():
    """Config loading must work even if get_type_hints() fails."""
    import unittest.mock
    with unittest.mock.patch('typing.get_type_hints', side_effect=NameError("test")):
        cfg = load_config("configs/train/stage_a.yaml")
        assert isinstance(cfg.model, ModelConfig)
```

---

## P1 — Silent Training Quality Issues

### P1-1: ContrastiveTemporalLoss treats within-trajectory future steps as negatives

**Severity**: P1 (silently hurts representation quality)
**Files**: `vla_hybrid_v2/losses/consistency_loss.py:40-51`

**Evidence**:

```python
a = anchors.reshape(B * T_minus_1, D)    # flatten batch + time
p = positives.reshape(B * T_minus_1, D)
logits = torch.matmul(a, p.T) / self.temperature  # [N, N]
labels = torch.arange(logits.shape[0], device=logits.device)
```

With N = B*(T-1), each anchor (fused_state at time t) has **one** positive
(fused_state at time t+1) and N-1 negatives. The negatives include:
- Same trajectory, t+2, t+3, ... — states that **should** be nearby
- Same trajectory, t-1, t-2, ... — states that **are** nearby
- Different trajectory, same timestep — could be semantically similar

The loss actively pushes apart consecutive states from the SAME trajectory when
they serve as negatives for other pairs. For T=16, each anchor has 15
intra-trajectory negatives and only 1 positive, creating a strong force to make
the representation non-smooth along time — directly conflicting with the purpose
of temporal consistency.

**Why current implementation is insufficient**:
The VICReg variance term (L-12 fix) prevents collapse but doesn't address the
negative-sampling problem. The temporal contrastive loss and the smoothness
objective work at cross-purposes.

**Suggested minimal fix**:
Mask out intra-trajectory negatives so they don't contribute to the denominator:

```python
# Create mask: [N, N] where True means "same trajectory"
traj_ids = torch.arange(B, device=a.device).repeat_interleave(T_minus_1)
mask = (traj_ids.unsqueeze(1) == traj_ids.unsqueeze(0))  # [N, N]
# Set intra-trajectory non-positive entries to -inf
logits = logits.masked_fill(mask & ~torch.eye(N, device=a.device, dtype=torch.bool), -1e9)
```

**Minimal test**:
```python
def test_contrastive_no_intra_trajectory_negatives():
    """Anchors from same trajectory (except positive) should not be negatives."""
    loss_fn = ContrastiveTemporalLoss()
    # Create 2 trajectories with identical representations
    # Loss should be low (no useful negatives within same trajectory)
    fused = torch.randn(1, 5, 64).expand(2, -1, -1)  # B=2, same states
    loss = loss_fn(fused)
    # Current impl: high loss due to intra-trajectory negatives
    # Fixed impl: lower loss
```

---

### P1-2: VICReg variance regularisation uses local-rank statistics under DDP/FSDP

**Severity**: P1 (variance estimate is noisy on small per-device batches)
**Files**: `vla_hybrid_v2/losses/consistency_loss.py:55-58`

**Evidence**:

```python
flat = fused_states.reshape(-1, D)  # [B*T, D] — local batch only
std = flat.std(dim=0)               # [D] — local estimate
var_loss = F.relu(self.variance_target - std).mean()
```

On 8-GPU FSDP with global batch 128, each rank sees B=16 samples. With T=4,
each rank computes std over only 64 vectors. The variance estimate has high
variance itself (~1/sqrt(64) ≈ 12.5% relative error).

When std is near the target (1.0), some ranks may see std > 1.0 (no penalty)
while others see std < 1.0 (penalty). This creates inconsistent gradients
across ranks — FSDP averages gradients, but the local loss computations diverge.

**Why current implementation is insufficient**:
No `all_reduce` is performed on the variance statistics before computing the loss.
This means each rank optimizes toward its own local variance estimate.

**Suggested minimal fix**:
```python
if self.variance_weight > 0:
    flat = fused_states.reshape(-1, D)
    std = flat.std(dim=0)
    if dist.is_initialized():
        dist.all_reduce(std, op=dist.ReduceOp.AVG)
    var_loss = F.relu(self.variance_target - std).mean()
```

**Minimal test**:
```python
def test_variance_regularisation_nonzero_gradient():
    """Variance term must produce gradient when std < target."""
    loss_fn = ContrastiveTemporalLoss(variance_weight=1.0, variance_target=2.0)
    fused = torch.randn(4, 3, 64, requires_grad=True) * 0.1  # low variance
    loss = loss_fn(fused)
    loss.backward()
    assert fused.grad is not None and fused.grad.abs().sum() > 0
```

---

### P1-3: ActionConsistencyLoss now uses MSE but scale mismatch with ContrastiveTemporalLoss

**Severity**: P1 (loss magnitude imbalance may cause one term to dominate)
**Files**: `vla_hybrid_v2/losses/consistency_loss.py:98-99`, `:119-128`

**Evidence**:

The three sub-losses in `V2ConsistencyLoss.forward()`:
1. `self.temporal(fused_states)` — InfoNCE cross-entropy, typical range ~5-8 for
   random inputs (ln(N) where N = B*(T-1))
2. `self.slow_fast(fast_tokens, slow_token)` — MSE between [B, D] vectors,
   typical range ~1-4 for random 2048-dim vectors
3. `self.action(discrete, continuous)` — MSE between [B, H, 14] tensors,
   typical range ~0.5-2.0 for normalized action space

With default weights (slow_fast_weight=0.5, action_weight=0.5):
- Total ≈ 6.5 + 0.5×2.5 + 0.5×1.0 ≈ 8.25
- The InfoNCE term dominates at ~79% of total loss

The old cosine-based `ActionConsistencyLoss` had range [0, 2], so with weight 0.5
it contributed ~0.5. The new MSE-based version contributes ~0.5 as well, so the
action consistency signal is in a similar range. But the temporal InfoNCE still
dominates, meaning the action consistency (discrete ↔ continuous agreement) gets
relatively weak supervision.

**Why this matters**:
If discrete and continuous heads disagree (which is the main failure mode at
deployment), the consistency gradient is ~12× weaker than the temporal contrastive
gradient. The heads may drift apart during training without adequate correction.

**Suggested minimal test**:
```python
def test_consistency_loss_magnitude_balance():
    """Sub-loss magnitudes should be within 10x of each other."""
    loss_fn = V2ConsistencyLoss(action_dim=14)
    fs = torch.randn(8, 4, 2048)
    ft_tok = torch.randn(8, 4, 2048)
    sl_tok = torch.randn(8, 2048)
    d = torch.randn(8, 24, 14)
    c = torch.randn(8, 24, 14)
    # Get individual losses
    l_temporal = loss_fn.temporal(fs)
    l_sf = loss_fn.slow_fast(ft_tok, sl_tok)
    l_act = loss_fn.action(d, c)
    ratio = l_temporal.item() / max(l_act.item(), 1e-8)
    assert ratio < 20, f"Temporal/Action ratio = {ratio:.1f}, too imbalanced"
```

---

## P2 — Latent Risks / Missing Coverage

### P2-1: ContrastiveTemporalLoss allocates O(B^2 T^2) logit matrix

**Severity**: P2 (OOM at large batch sizes)
**Files**: `vla_hybrid_v2/losses/consistency_loss.py:49`

**Evidence**:

```python
logits = torch.matmul(a, p.T) / self.temperature  # [N, N] where N = B*(T-1)
```

| B | T | N | Matrix Size (bf16) |
|---|---|---|-------------------|
| 32 | 16 | 480 | 0.4 MB |
| 128 | 16 | 1920 | 7.0 MB |
| 128 | 33 | 4096 | 32 MB |
| 256 | 33 | 8160 | 127 MB |

At B=256, T=33, this is 127 MB per rank for a single loss term. With gradient
storage, ~380 MB total. Not catastrophic, but unexpected memory pressure during
scaling experiments.

**Minimal test**:
```python
def test_contrastive_loss_large_batch():
    """Verify no OOM up to B=64, T=33 on 8GB GPU."""
    loss_fn = ContrastiveTemporalLoss()
    fused = torch.randn(64, 33, 2048, device="cuda")
    loss = loss_fn(fused)
    loss.backward()  # Should not OOM
```

---

### P2-2: Inference `control_step` still uses `torch.roll` for action history

**Severity**: P2 (inconsistency between train and inference paths, minor perf)
**Files**: `vla_hybrid_v2/models/hybrid_vla_v2.py:844-847`

**Evidence**:

Training path (forward_train) uses the new `ActionHistoryBuffer` ring buffer:
```python
action_history_buf = ActionHistoryBuffer(max_len=...)
action_history_buf.push(batch["prev_actions"][:, t])
```

Inference path (control_step) still uses `torch.roll`:
```python
runtime_state.action_history = torch.roll(
    runtime_state.action_history, -1, dims=1,
)
runtime_state.action_history[:, -1] = action
```

Both produce the same result (chronologically ordered action history), but:
1. Inconsistency makes reasoning about correctness harder
2. `torch.roll` allocates new memory each call (minor at B=1 inference)
3. If ActionHistoryBuffer.get() ever changes ordering semantics, inference
   will silently diverge from training

**Minimal test**:
```python
def test_action_history_train_inference_consistency():
    """Ring buffer and torch.roll must produce identical ordering."""
    K, A = 8, 14
    buf = ActionHistoryBuffer(max_len=K)
    raw = torch.zeros(1, K, A)
    for i in range(12):
        action = torch.full((1, A), float(i))
        buf.push(action)
        raw = torch.roll(raw, -1, dims=1)
        raw[:, -1] = action
    assert torch.allclose(buf.get(), raw)
```

---

### P2-3: No tests for Stage B or Stage C training paths

**Severity**: P2 (critical code paths untested)
**Files**: `tests/test_losses.py`, `tests/test_control_step.py`

**Evidence**:

The entire `forward_train()` method has **three** distinct code paths gated by
`self.cfg.stage`:
- **Stage A**: Only discrete + consistency losses (expert frozen)
- **Stage B**: Adds flow matching + expert losses + RTC/FASTER conditional
- **Stage C**: Adds RTC overlap inpainting + FASTER per-step weighting

No test covers Stage B or C. The conftest fixture uses `_mini_cfg("a")`.

Untested code includes:
- `flow_matching_loss` call with `step_weights` (line 618)
- `expert_denoised` recovery from flow matching (line 596)
- RTC overlap inpainting loss (lines 625-666)
- FASTER near-horizon auxiliary loss (lines 673-682)
- Consistency loss with `discrete_actions` and `continuous_actions` (line 685-691)
- `cond_prefix.detach()` gating (line 575-577)
- `noise_scale = 0.01` added to prev_cond (line 639-640)

**Minimal test**:
```python
@pytest.mark.parametrize("stage", ["a", "b", "c"])
def test_forward_train_all_stages(stage, dummy_batch):
    cfg = _mini_cfg(stage)
    if stage == "c":
        cfg.train.rtc.enable = True
        cfg.train.faster.enable = True
    model = _build_model(cfg)
    losses = model.forward_train(dummy_batch)
    assert "loss_total" in losses
    assert not torch.isnan(losses["loss_total"])
    if stage != "a":
        assert "loss_fm" in losses
```

---

### P2-4: EMA `load_state_dict` can introduce stale shadows for keys not in checkpoint

**Severity**: P2 (silent correctness issue on checkpoint upgrade)
**Files**: `vla_hybrid_v2/utils/ema.py:123-147`

**Evidence**:

```python
def load_state_dict(self, state: dict) -> None:
    loaded_shadow = state["shadow"]
    # ... shape filter ...
    self.shadow.update(loaded_shadow)  # MERGE, not replace
```

`self.shadow` was initialized from the **current** model in `__init__`. Then
`update()` overwrites with loaded values. For keys that exist in `self.shadow`
but NOT in `loaded_shadow` (e.g., new parameters added in v0.10.10), the
EMA shadow retains the `__init__`-time copy. This is correct.

But for keys that were **shape-dropped** (e.g., `action_history_encoder` keys),
the `del loaded_shadow[k]` removes them from the loaded dict. Then
`self.shadow.update(loaded_shadow)` doesn't overwrite the current model's
init-time shadow for those keys. So the EMA shadow for resized layers starts
from the randomly initialized weights (from `__init__`), not from the
cross-stage checkpoint. If cross-stage loading happened before EMA init, those
weights are already loaded and EMA init captures them. But if auto_resume loads
the model weights AFTER EMA init, the EMA shadows are stale.

**Ordering in train_unified.py**:
```
370: model.to(device)
385: cross-stage load → model has resumed weights
391: EMA init → shadow copies resumed weights ✓
402: FSDP wrap
450: auto_resume → loads into FSDP model, then loads EMA state
     → EMA.load_state_dict merges checkpoint shadows over init shadows
     → Dropped keys keep init-time (resumed) values ✓
```

For auto_resume, the sequence is correct: dropped EMA shadows retain values
from step 391 (cross-stage resumed weights), not random init. So this is
**safe for the documented use case**.

But if someone does only `auto_resume` (no cross-stage), the init-time shadows
for resized layers would be random. This scenario occurs when:
1. Train v0.10.9, save checkpoint (EMA has old ActionHistoryEncoder shadows)
2. Update to v0.10.10, resume
3. Cross-stage is not set, auto_resume loads checkpoint
4. EMA init (step 391) uses model with random ActionHistoryEncoder (not yet loaded)
5. auto_resume loads model (step 450, but EMA init at 391 already captured random weights)

Wait — step 391 happens before 450. At step 391, the model was just `.to(device)`
with random weights (no cross-stage and auto_resume hasn't run yet). So EMA
shadows capture random weights. Then at step 450, auto_resume loads the model
AND the EMA. The EMA load merges old shadows, dropping shape-mismatched keys.
The dropped keys retain random-init values from step 391.

This means: **after resuming from v0.10.9 checkpoint, the EMA shadows for resized
ActionHistoryEncoder parameters are random** instead of the resumed model weights.

**Impact**: EMA evaluation/saving for the first `ramp_steps` will have corrupted
shadows for ActionHistoryEncoder. The EMA will gradually correct itself as
`update()` runs, but eval during early resumed training will be degraded.

**Minimal test**:
```python
def test_ema_load_state_dict_shape_mismatch_shadow_correctness():
    """After load with shape mismatch, dropped keys must use current model weights."""
    # Build model v2 (small ActionHistoryEncoder)
    # Init EMA → shadows have current weights
    # Create old EMA state with large ActionHistoryEncoder shadows
    # Load → dropped keys should retain current model weights, not random
```

---

## Verified Claims (Spot-Check)

| Claim | Status | Evidence |
|-------|--------|---------|
| `assert` → `ValueError` in `_validate_batch` | **VERIFIED** | Lines 309-378: all 12 assertions converted |
| World model files moved to `experimental/` | **VERIFIED** | `experimental/world_model/` exists with all 9 files; `world_model/__init__.py` re-exports |
| `loss_total = sum()` → `torch.stack().sum()` | **VERIFIED** | Line 699: `torch.stack(list(losses.values())).sum()` |
| EMA `_strip_fsdp_prefix` in update/apply/restore | **VERIFIED** | Lines 93, 101, 110: all three methods strip prefix |
| Cross-stage load moved before EMA init | **VERIFIED** | train_unified.py: load at 385, EMA at 391 |
| ActionHistoryEncoder resized to d_inner=256 | **VERIFIED** | mamba_core.py:538-543: `d_inner = 256`, 2-layer stack |
| `FASTDiscreteHead` → `DiscreteActionHead` with alias | **VERIFIED** | discrete_heads.py:85: `FASTDiscreteHead = DiscreteActionHead` |
| `batch.get()` replaces `"key" in batch` | **VERIFIED** | 8 instances converted in hybrid_vla_v2.py |
| FSDP prefix stripping in optimizer param groups | **VERIFIED** | train_unified.py:406-428 |
| Eval barriers added for EMA swap | **VERIFIED** | train_unified.py:565, 570, 573: three barriers |
| `ActionHistoryBuffer` ring buffer (training) | **VERIFIED** | types.py:112-131: index-based, no `torch.roll` |
| `torch.roll` still in inference | **VERIFIED** | hybrid_vla_v2.py:844 |
| `FlowMatchingLoss.forward` `t` made optional | **VERIFIED** | flow_matching.py:16: `t=None` |
| SlowFastAgreementLoss `.detach()` removed | **VERIFIED** | consistency_loss.py:82: bidirectional |
| `mamba_impl` config + `force_fallback` propagation | **VERIFIED** | config.py:97, mamba_core.py:97,340,488,498,508,534,651 |

---

## Summary Table

| # | Title | Severity | File | Crash Risk |
|---|-------|----------|------|------------|
| P0-1 | FSDP shape-mismatch filter ineffective on auto-resume | P0 | checkpointing.py | Multi-GPU version upgrade |
| P0-2 | `_dict_to_dataclass` fallback broken for PEP 563 annotations | P0 (latent) | config.py | Triggered by `get_type_hints()` failure |
| P1-1 | InfoNCE intra-trajectory negatives harm smoothness | P1 | consistency_loss.py | Silent quality degradation |
| P1-2 | VICReg variance uses local-rank statistics | P1 | consistency_loss.py | Noisy on small per-device batch |
| P1-3 | Loss magnitude imbalance (InfoNCE 8x > action MSE) | P1 | consistency_loss.py | Weak action consistency supervision |
| P2-1 | O(B^2 T^2) logit matrix in contrastive loss | P2 | consistency_loss.py | OOM at large B*T |
| P2-2 | Inference still uses `torch.roll` for action history | P2 | hybrid_vla_v2.py:844 | Train/infer inconsistency |
| P2-3 | No tests for Stage B or C forward_train | P2 | tests/ | Untested critical paths |
| P2-4 | EMA shadows for resized layers are random after version-upgrade resume | P2 | ema.py | Degraded EMA eval early in resumed training |
