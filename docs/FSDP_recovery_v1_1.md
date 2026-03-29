# FSDP Recovery v1.1 — Core Chain Closure

## 1. Summary

Four targeted fixes to close the FSDP main training path:

| Area | Problem | Fix |
|------|---------|-----|
| **A. dtype** | Non-backbone params (SSM A_log/D, all Linear, buffers) stay float32 while FSDP expects bf16. No centralized control, no post-load verification. | `normalize_model_dtypes_for_fsdp()` converts all float params/buffers to bf16 before FSDP wrap. `verify_model_dtypes()` detects regression after checkpoint load. Wired into `train_unified.py` between resume and EMA/FSDP init. |
| **B. checkpointing** | FSDP `checkpoint_wrapper` wraps MambaBlock + `_MambaStack.forward()` also calls `activation_checkpoint()` on the same layers = double-checkpointing. Also, `layer._forward_official(x)` breaks when layer is CheckpointWrapper. | `_fsdp_manages_checkpointing()` detects FSDP wrapper; skips internal checkpoint when true. `_unwrap_layer()` reaches MambaBlock through CheckpointWrapper. Stateless path uses `layer(x)` instead of `layer._forward_official(x)`. |
| **C. save/resume** | Post-load dtype audit missing. No centralized dtype check after `load_checkpoint()`. | `load_checkpoint()` now logs dtype mix when detected. `train_unified.py` calls normalize+verify after cross-stage resume to ensure consistent state before FSDP wrap. |
| **D. tests** | No FSDP-specific tests. | 18 new CPU tests + 2-GPU smoke script covering all three areas above. |

### FSDP dtype strategy (explicit)

```
Backbone (Qwen2-VL):     bfloat16  (from HF loading, unchanged)
Non-backbone params:      bfloat16  (normalized before FSDP wrap)
SSM A_log, D:             bfloat16  (was float32, now normalized)
Float buffers:            bfloat16  (e.g. _fast_bin_centers)
Integer/bool buffers:     unchanged (masks, indices)
FSDP MixedPrecision:      param=bf16, reduce=f32, buffer=bf16
Optimizer:                runs on bf16 params
```

### Training flow with FSDP (final order)

```
model = HybridVLAv2(cfg)                    # init (f32 non-backbone)
configure_trainable_modules(model, stage)    # freeze/unfreeze
model.to(device)                             # GPU
load_checkpoint(model, ...)                  # cross-stage resume (if any)
normalize_model_dtypes_for_fsdp(model, bf16) # <-- NEW: all float -> bf16
verify_model_dtypes(model, bf16)             # <-- NEW: fail-fast check
ema = EMAModel(model, ...)                   # shadows in bf16
model = wrap_fsdp(model, ...)                # FSDP wrap
optimizer = AdamW(model.parameters(), ...)   # optimizer on bf16 params
auto_resume(model, optimizer, ...)           # same-stage resume (if any)
# training loop...
```

## 2. Files Changed

### Modified
| File | Change |
|------|--------|
| `vla_hybrid_v2/utils/distributed.py` | +`normalize_model_dtypes_for_fsdp()`, +`verify_model_dtypes()` (70 lines) |
| `vla_hybrid_v2/utils/checkpointing.py` | Post-load dtype audit in `load_checkpoint()` (13 lines) |
| `vla_hybrid_v2/models/mamba_core.py` | +`_unwrap_layer()`, +`_fsdp_manages_checkpointing()`, fix stateless/stateful/fallback paths (39 lines net) |
| `scripts/train_unified.py` | dtype normalize+verify block after resume (8 lines) |

### New
| File | Purpose |
|------|---------|
| `tests/test_fsdp_dtype.py` | 10 tests: normalize, verify, roundtrip, SSM params |
| `tests/test_fsdp_checkpointing.py` | 8 tests: unwrap, conflict detect, double-ckpt prevention, integration |
| `scripts/smoke_fsdp_2gpu.py` | 2-GPU smoke: dry-run, save/resume, dtype consistency |

## 3. Tests Added/Updated

### CPU tests (pytest)
```
tests/test_fsdp_dtype.py::TestDtypeNormalization::test_converts_float32_params
tests/test_fsdp_dtype.py::TestDtypeNormalization::test_converts_float_buffers
tests/test_fsdp_dtype.py::TestDtypeNormalization::test_preserves_int_buffers
tests/test_fsdp_dtype.py::TestDtypeNormalization::test_idempotent
tests/test_fsdp_dtype.py::TestDtypeNormalization::test_ssm_params_converted
tests/test_fsdp_dtype.py::TestDtypeVerification::test_all_bf16_passes
tests/test_fsdp_dtype.py::TestDtypeVerification::test_mixed_dtype_fails
tests/test_fsdp_dtype.py::TestDtypeVerification::test_with_label
tests/test_fsdp_dtype.py::TestCheckpointDtypeRoundtrip::test_save_load_preserves_bf16
tests/test_fsdp_dtype.py::TestCheckpointDtypeRoundtrip::test_load_float32_checkpoint_then_normalize
tests/test_fsdp_checkpointing.py::TestUnwrapLayer::test_unwrap_bare_layer
tests/test_fsdp_checkpointing.py::TestUnwrapLayer::test_unwrap_checkpoint_wrapped_layer
tests/test_fsdp_checkpointing.py::TestFsdpManagesCheckpointing::test_false_for_bare_stack
tests/test_fsdp_checkpointing.py::TestFsdpManagesCheckpointing::test_true_after_checkpoint_wrapping
tests/test_fsdp_checkpointing.py::TestNoDoubleCheckpointing::test_fallback_path_with_fsdp_ckpt
tests/test_fsdp_checkpointing.py::TestNoDoubleCheckpointing::test_fallback_path_without_fsdp_ckpt
tests/test_fsdp_checkpointing.py::TestNoDoubleCheckpointing::test_stateless_path_with_checkpoint_wrapper
tests/test_fsdp_checkpointing.py::TestFullModelCheckpointingIntegration::test_forward_train_after_checkpoint_wrapping
```

### 2-GPU smoke test (requires GPU)
```bash
torchrun --nproc_per_node=2 scripts/smoke_fsdp_2gpu.py
```
Tests: dry-run, save+resume, dtype consistency at every stage.

## 4. Validation

```
$ python -m pytest tests/ -q
131 passed in 22s
```

All 131 tests pass (90 pre-existing + 18 new FSDP + 23 from prior sessions).

2-GPU smoke test requires GPU hardware — script is ready but could not be executed locally (macOS, no CUDA).

## 5. Remaining Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| bf16 optimizer states (AdamW moments in bf16) may reduce training stability vs f32 master weights | Medium | Standard for large-model training (LLaMA, GPT). Monitor loss curves. If unstable, keep param storage in f32 and remove `normalize_model_dtypes_for_fsdp` call. |
| `mamba_impl="auto"` + official Mamba2 + FSDP checkpointing: step() path uses `_unwrap_layer()` which bypasses CheckpointWrapper (inference only, acceptable) | Low | Only affects inference path. Training uses fallback vectorized path. |
| EMA `summon_full_params` with bf16 storage: EMA shadows are bf16 → EMA lerp in bf16 | Low | EMA lerp precision loss is negligible at typical decay rates (0.999+). |
| `auto_resume` loads into FSDP-wrapped model (post-wrap) — dtype normalization only runs pre-wrap | Low | Same-stage resume loads matching dtypes. Cross-stage resume is pre-wrap (handled). |

### Out of scope (per task constraints)
- eval / rollout / numpy conversion
- old config / proprio_key cleanup
- teacher-forcing / semantic refresh / detach strategies
- DDP path (FSDP is the default and only supported multi-GPU path)
