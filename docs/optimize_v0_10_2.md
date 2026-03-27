# HybridVLA v2 — v0.10.2 Optimization Report

> **Date**: 2026-03-27
> **Scope**: 5 targeted fixes from v0.10.1 cross-audit residual issues
> **Method**: Audit → fix → verify → smoke test

---

## Fixes Applied

### I1: Make `num_affordance_types` configurable (was hardcoded=8)

**Problem**: `AffordanceHead.__init__` accepted `num_affordance_types` but the caller in `HybridVLAv2.__init__` didn't pass it. The value 8 was hardcoded in `DummyVLADataset` with a magic-number comment. If an embodiment needed a different count, three locations had to be edited manually.

**Fix**:
- `config.py`: Added `num_affordance_types: int = 8` to `HeadsConfig`
- `hybrid_vla_v2.py`: Pass `mcfg.heads.num_affordance_types` to `AffordanceHead()`
- `data/dummy.py`: Read from `cfg.model.heads.num_affordance_types` instead of hardcoded 8

**Files**: `config.py`, `models/hybrid_vla_v2.py`, `data/dummy.py`

---

### I2: Lift `_to_device` closure out of training loop

**Problem**: In `train_stage_a.py`, the `_to_device()` helper function was redefined on every batch iteration inside the `for batch_idx, batch in enumerate(loader)` loop. While Python is fast at creating closures, it's unnecessary churn and hurts readability.

**Fix**: Moved `_to_device()` definition before the loop. No semantic change.

**File**: `scripts/train_stage_a.py`

---

### I3: Document smoke test `DummyVLADataset` divergence

**Problem**: `train_smoke_test.py` has an inline `DummyVLADataset` with mini dimensions (A=7, P=9, D=64) that is separate from the production `data/dummy.py` (A=14, P=14, D=512). This was intentional for CPU speed, but undocumented — a future contributor might try to unify them and break the smoke test's isolation.

**Fix**: Added docstring explaining the intentional divergence and its rationale.

**File**: `scripts/train_smoke_test.py`

---

### I4: Add `step_weights` shape validation in `_validate_batch`

**Problem**: `step_weights` (optional per-step loss weighting, shape `[B, H]`) was used in `flow_matching_loss` but never shape-checked at the model boundary. A dataset returning `[B]` or `[B, H, 1]` would cause a silent broadcast or a cryptic error deep in the loss computation.

**Fix**: Added assertion in `_validate_batch()` checking `step_weights.shape == (B, H)` with a descriptive error message.

**File**: `models/hybrid_vla_v2.py`

---

### I5: Remove redundant `.to(device)` on `action_history_buf.get()`

**Problem**: In `forward_train()`, the action history buffer result was explicitly moved to device via `.to(device)`, but the buffer is already on the correct device (it's populated from `batch["actions"]` which is moved to device beforehand). The extra `.to()` call is a no-op that adds overhead.

**Fix**: Removed `.to(device)` — call is now just `action_history_buf.get()`.

**File**: `models/hybrid_vla_v2.py`

---

## Verification

```
I1 OK: num_affordance_types in HeadsConfig
I1 OK: AffordanceHead wired to config
I1 OK: DummyVLADataset reads num_affordance_types from config
I2 OK: _to_device defined before loop
I4 OK: step_weights shape validated
I5 OK: redundant .to(device) removed

Smoke test: 20 steps in 31.1s — PASSED (no NaN, no crash)
```

---

## 中文摘要

v0.10.2 修复了 v0.10.1 交叉审计遗留的 5 个问题：

1. **I1**: `num_affordance_types` 从硬编码改为可配置（`HeadsConfig` → `AffordanceHead` → `DummyVLADataset`）
2. **I2**: `_to_device` 闭包从训练循环内提到循环外，减少不必要的重复定义
3. **I3**: 为 smoke test 的独立 `DummyVLADataset` 添加文档说明，防止未来误合并
4. **I4**: 在 `_validate_batch` 中添加 `step_weights` 形状校验 `[B, H]`
5. **I5**: 移除 `action_history_buf.get()` 上多余的 `.to(device)` 调用

所有修改通过验证 + smoke test，无回归。
