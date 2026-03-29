# Training Loop & Config Hygiene Fix Report — v1.1

**Branch**: `fix/train-config-hygiene`
**Base**: `dev` @ `46f7f86`
**Date**: 2026-03-29

---

## 1. Summary

Three focused fixes to make the training entry point more stable and fail-fast:

| Fix | Problem | Solution |
|-----|---------|----------|
| A | Gradient accumulation triggers redundant FSDP all-reduce on every micro-step | Added `no_sync()` context for non-final accumulation steps |
| B1 | `configs/data/libero_multicam.yaml` used V1-style `robot0_joint_pos` and wrong camera keys | Fixed to match LIBERO spec (`joint_states`, proper camera names) |
| B2+B3 | No config validation before training; bad configs fail late with cryptic errors | Added `validate_config()` with 7 fail-fast checks, called at top of `train()` |

---

## 2. Files Changed

### Modified
| File | Change |
|------|--------|
| `scripts/train_unified.py` | Added `_maybe_no_sync()` helper wrapping forward+backward; imported and called `validate_config()` at start of `train()` |
| `vla_hybrid_v2/config.py` | Added `validate_config()` function (80 lines) |
| `configs/data/libero_multicam.yaml` | Fixed `proprio_key`, added `proprio_keys`, `format`, `action_dim`, `proprio_dim`; corrected camera key names |

### New
| File | Purpose |
|------|---------|
| `tests/test_grad_accum_nosync.py` | 8 tests: no_sync behavior with FSDP, single-GPU passthrough, accumulation pattern verification |
| `tests/test_config_validation.py` | 15 tests: valid configs pass, invalid proprio_key/keys, multicam mismatches, stage, grad_accum |

---

## 3. Tests Added

### `tests/test_grad_accum_nosync.py` (8 tests)
- `TestMaybeNoSync::test_calls_no_sync_when_accumulating_and_fsdp`
- `TestMaybeNoSync::test_no_no_sync_on_final_step_fsdp`
- `TestMaybeNoSync::test_single_gpu_no_sync_absent`
- `TestAccumulationPattern::test_sync_only_on_final_micro_step[1/2/4/8]`
- `TestAccumulationPattern::test_grad_accum_1_never_calls_no_sync`

### `tests/test_config_validation.py` (15 tests)
- `TestValidConfigPasses::test_valid_libero_multicam`
- `TestValidConfigPasses::test_valid_default_config`
- `TestValidConfigPasses::test_valid_singlecam_no_proprio_keys`
- `TestProprioKeyValidation::test_proprio_key_not_in_proprio_keys`
- `TestProprioKeyValidation::test_v1_proprio_key_with_libero_format`
- `TestProprioKeyValidation::test_proprio_dim_mismatch`
- `TestMultiCameraValidation::test_multicam_num_cameras_lt_2`
- `TestMultiCameraValidation::test_camera_names_count_mismatch`
- `TestMultiCameraValidation::test_camera_keys_count_mismatch`
- `TestMultiCameraValidation::test_num_cameras_exceeds_max`
- `TestMultiCameraValidation::test_multicam_disabled_skips_checks`
- `TestStageValidation::test_invalid_stage`
- `TestGradAccumValidation::test_grad_accum_zero`
- `TestGradAccumValidation::test_grad_accum_negative`
- `TestMultipleErrors::test_reports_multiple_issues`

---

## 4. Validation

```
$ python -m pytest tests/ -v
113 passed, 0 failed, 0 errors (24.02s)
```

Full test suite passes with zero regressions.

---

## 5. Commit History (this branch)

| SHA | Message |
|-----|---------|
| `3184a61` | `fix(train): add no_sync for grad accumulation micro-steps` |
| `3d1c46f` | `fix(config): correct libero_multicam.yaml proprio_key and camera keys` |
| `e1ddf8d` | `fix(config): add validate_config() with fail-fast checks` |

Note: Commits `e17bea7`, `ea3272b`, `94f154a` on this branch are from the `fix/eval-dtype-safety` thread and were merged before this work began.

---

## 6. Remaining Risks

| Risk | Severity | Notes |
|------|----------|-------|
| `no_sync` untested on actual multi-GPU FSDP | Medium | Unit tests verify logic; needs real multi-GPU smoke test on HPC |
| `configs/data/libero_multicam.yaml` 3-camera setup vs LIBERO standard 2 cameras | Low | Fixed keys are correct; whether 3rd camera (`left_view_rgb`) exists in actual LIBERO data depends on task suite |
| `validate_config` covers known misconfigs only | Low | More checks can be added incrementally |
| Unstaged changes in `vla_hybrid_v2/utils/distributed.py` and `checkpointing.py` | Info | These belong to `fix/eval-dtype-safety` thread — not committed on this branch |

---

## 7. Potential Cross-Thread Conflicts

| File | This thread | Other thread | Conflict risk |
|------|-------------|-------------|---------------|
| `scripts/train_unified.py` | `no_sync`, `validate_config` import/call | `fix/eval-dtype-safety`: dtype normalization imports | **Merge conflict likely** — both modify imports and `train()` body |
| `vla_hybrid_v2/utils/distributed.py` | No changes | `fix/eval-dtype-safety`: adds `normalize_model_dtypes_for_fsdp` | None |
| `vla_hybrid_v2/config.py` | Adds `validate_config()` | None known | None |

**Resolution**: When merging both branches, `train_unified.py` will need manual conflict resolution for the import block and `train()` function header.
