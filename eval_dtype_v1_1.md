# Eval / Inference Dtype Safety Fix — v1.1

## 1. Summary

Fixed the critical bf16 -> numpy crash in the eval/rollout/inference main path.

**Root cause**: When model weights are bf16 (standard for FSDP mixed-precision training), `control_step()` produces bf16 action tensors. The normalizer `denormalize()` preserves input dtype. `tensor.cpu().numpy()` then raises `TypeError: Got unsupported ScalarType BFloat16` because numpy has no bfloat16 support.

**Fix strategy**: Two-layer defense:
1. **Primary (centralized)**: `control_step_from_obs` in `libero_policy.py` now casts `action_env` and `action_model` to fp32 before returning — this is the single exit point for all env-facing actions.
2. **Belt-and-suspenders**: `eval_libero_rollout.py` adds `.float()` before `.cpu().numpy()` at the consumption site.

## 2. Files Changed

| File | Change | Lines |
|------|--------|-------|
| `vla_hybrid_v2/infer/libero_policy.py` | Cast action_env/action_model to fp32 in `control_step_from_obs` | +8 |
| `libero_hybrid/scripts/eval_libero_rollout.py` | Add `.float()` before `.cpu().numpy()` | +1/-1 |

## 3. Tests Added/Updated

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_eval_dtype_safety.py` | 12 tests | bf16/fp16/fp32 numpy safety, control_step dtype parametrized, e2e inference smoke, multi-step rollout sim, fail-fast assertions, normalizer accuracy |

### Test breakdown:
- **TestBF16NumpySafety** (4 tests): Raw bf16/fp16/fp32 -> numpy conversion behavior
- **TestControlStepDtypeSafety** (4 tests): `control_step_from_obs` returns fp32 for bf16/fp16/fp32 model outputs; end-to-end numpy conversion
- **TestInferenceSmokeTest** (2 tests): Full semantic -> control -> numpy path; multi-env multi-step rollout simulation
- **TestFailFastAssertions** (2 tests): Missing semantic_summary raises; normalizer accuracy after fp32 cast

## 4. Validation

```
$ python -m pytest tests/test_eval_dtype_safety.py tests/test_infer_policy.py tests/test_control_step.py -v
======================== 19 passed, 0 warnings in 0.24s =========================
```

- 12 new tests: all pass
- 7 existing inference/control tests: no regressions

## 5. Remaining Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Backbone `semantic_step` output may also be bf16 — GrounderOutput tokens passed to temporal core | Low | Temporal core operations are all-torch (no numpy); bf16 math works fine on GPU. Only matters if someone logs/exports intermediate tokens. |
| `imagination_engine.py` rollout stays in tensor space | None | No numpy conversion in that path — no fix needed. |
| `train_unified.py:evaluate()` uses `v.item()` on losses | None | `.item()` works on any dtype, returns Python float. |
| Checkpoint loaded with explicit bf16 dtype after FSDP unflatten | Low | The fp32 cast at env boundary handles this regardless of model internal dtype. |

## 6. Git Info

- **Branch**: `fix/eval-dtype-safety`
- **Commits**:
  - `0b6870a` — `fix(eval): cast bf16 actions to fp32 before numpy conversion`
  - `df8ec22` — `test(eval): add dtype safety + inference smoke tests`
- **Uncommitted changes**: None (only pre-existing unrelated files: LIBERO submodule, 2 comparison .md files)
- **Potential conflicts**: None — changes are isolated to inference path (`libero_policy.py`, `eval_libero_rollout.py`) and new test file.
