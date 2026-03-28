# v1.0 Code Change Report

## Audit Scope

Based on current codebase state (not historical docs), verify and close 4 categories of issues:

| ID | Issue | Verdict |
|----|-------|---------|
| A | FSDP parameter name prefix breaks EMA update/apply/restore | **Already fixed** |
| B | Per-module LR `name.startswith` fails under FSDP names | **Already fixed** |
| C | Cross-stage resume: EMA shadow initialized before `resume_from` | **Already fixed** |
| D | Gap tests missing for EMA, FSDP param groups, TriRateMamba, Grounder | **Fixed (tests added)** |

---

## A. FSDP EMA Prefix — Already Fixed

**Root cause (historical):** FSDP wrapping adds `_fsdp_wrapped_module.` prefix to parameter names. EMA shadow keys (stored pre-FSDP) would not match post-FSDP names, causing silent no-op in `update()`, `apply()`, `restore()`.

**Current code state:**
- `distributed.py:119` — FSDP uses `use_orig_params=True`, which preserves original parameter names
- `ema.py:30-34` — `_strip_fsdp_prefix()` defensively strips `_fsdp_wrapped_module.` if present
- `ema.py:93` — `update()` calls `_strip_fsdp_prefix(name)` before matching shadow keys
- `ema.py:101,110` — `apply()` and `restore()` do the same
- `ema.py:46-53` — `_maybe_summon_full_params()` correctly unshard params for FSDP models

**Evidence:** 7 new tests pass — `TestEMARoundTrip` (5) + `TestEMAWithFSDPPrefix` (2).

**No code change required.**

---

## B. Per-module LR Under FSDP — Already Fixed

**Root cause (historical):** `name.startswith("backbone")` would fail if FSDP prepends `_fsdp_wrapped_module.backbone...`, causing all params to fall into the default "core" group with base LR.

**Current code state:**
- `train_unified.py:406-417` — imports `_strip_fsdp_prefix` and applies it before `startswith` checks:
  ```python
  clean = _strip_fsdp_prefix(name)
  if clean.startswith("backbone"):
      group = "backbone"
  elif clean.startswith("action_expert"):
      group = "expert"
  else:
      group = "core"
  ```

**Evidence:** 4 new tests pass — `TestPerModuleLRGroups`: clean names, single FSDP prefix, double FSDP prefix, full-model param coverage.

**No code change required.**

---

## C. Cross-stage Resume EMA Ordering — Already Fixed

**Root cause (historical):** If EMA shadow is initialized *before* `resume_from` loads the checkpoint, shadow clones random-init weights instead of resumed weights. Subsequent training starts EMA from a wrong baseline.

**Current code state (`train_unified.py`):**
```
L370:  model.to(device)
L372:  if cfg.train.resume_from:  # cross-stage checkpoint loaded FIRST
L385:      _load_ckpt(...)
L387:  # EMA initialized AFTER resume — shadow clones resumed weights
L388:  ema = EMAModel(model, ...)
L401:  model = wrap_fsdp(model, ...)  # FSDP wrapping AFTER EMA init
L450:  auto_resume(...)  # same-stage resume loads saved EMA state_dict
```

Order is correct: `resume_from` → EMA init → FSDP → `auto_resume`.

**Evidence:** 3 new tests pass — `TestCrossStageResumeEMAOrder`: cross-stage shadow correctness, same-stage overwrite, save/load round-trip.

**No code change required.**

---

## D. Gap Tests — Added

### New file: `tests/test_ema_fsdp_gaps.py` (22 tests)

| Test Class | Count | Coverage |
|------------|-------|----------|
| `TestEMARoundTrip` | 5 | `shadow_init`, `update_changes`, `apply_overwrites`, `restore_recovers`, `apply_restore_identity` |
| `TestEMAWithFSDPPrefix` | 2 | `strip_nested_prefix`, `update_with_fsdp_names` |
| `TestPerModuleLRGroups` | 4 | `clean_names`, `fsdp_prefix`, `double_prefix`, `all_params_covered` |
| `TestCrossStageResumeEMAOrder` | 3 | `shadow_matches_resumed`, `same_stage_overwrite`, `save_load_roundtrip` |
| `TestTriRateMambaCoreSmoke` | 3 | `forward_shape`, `multi_step_state`, `backward_gradient_flow` |
| `TestGrounderSmoke` | 3 | `output_structure`, `no_nan`, `backward_gradient_flow` |
| `TestFSDPTrainingStepSimulation` | 2 | `param_group_lr_values`, `training_step_updates_weights` |

### Verification

```
$ pytest tests/ -v
======================== 71 passed, 1 warning in 20.50s ========================
```

Baseline: 49 tests. After: 71 tests (+22). Zero regressions.

---

## D (Secondary): Design-Level Observations

### RTC Train/Infer Distribution Shift

**Status:** Mitigated, not root-fixed. Design limitation with clear documentation.

- **Training** (`hybrid_vla_v2.py:638-640`): Uses `cond_prefix + 0.01 * noise` to simulate a different previous chunk. Comment L-5 acknowledges the approximation.
- **Inference** (`hybrid_vla_v2.py:818-825`): Uses linear alpha blending between previous chunk tail and current chunk head.
- **Risk:** Train-time approximation means the model never sees real cross-chunk distribution during training. Inference blending compensates but may accumulate error over long horizons.
- **To root-fix:** Would require generating the previous chunk from a genuinely different cond_prefix (prior timestep), which needs data pipeline restructuring.

### Phase/Affordance Heads Without Labels

**Status:** Safeguarded. Not a crash risk.

- `hybrid_vla_v2.py:547,559` — Loss computed only when labels exist in batch (`batch.get("phase_labels") is not None`)
- `hybrid_vla_v2.py:150-158` — `logger.warning()` at model init when heads are enabled
- **Risk:** Heads exist but receive no supervision. Their tokens in `cond_prefix` carry no learned semantics. Not harmful (grounder still learns from other gradients), but wasteful.
- **Recommendation:** If dataset does not contain phase/affordance labels, set `heads.phase_head: false` and `heads.affordance_head: false` in config.

---

## Changed Files

| File | Action | Lines |
|------|--------|-------|
| `tests/test_ema_fsdp_gaps.py` | **Created** | 339 |

No production code was modified. All issues were already fixed in the current codebase.

---

## Remaining Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Real multi-GPU FSDP not tested in CI | Medium | Tests simulate FSDP prefix names but don't run actual `torchrun`. Add integration test on GPU CI. |
| RTC distribution shift | Low | Linear blending covers typical execution horizons. Monitor long-horizon rollout accuracy. |
| Unsupervised heads in cond_prefix | Low | Set `phase_head: false` / `affordance_head: false` when labels absent. |
