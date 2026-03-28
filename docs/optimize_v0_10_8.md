# v0.10.8 Optimization Record

Date: 2026-03-28  
Basis: `/Users/geminihe/hybridVLA_2/comparsion_between_pi_and_hybridVLA_2.md`

## Goal

This round focuses on the highest-value engineering gaps called out in the comparison report:

1. align rollout inference with the training-time normalization space
2. remove default online-control breakage and pseudo-capabilities
3. package minimal inference assets with checkpoints
4. add tests around the real online path instead of only training loss sanity

## Problems Addressed

### P0. Train / infer normalization drift

Before this change:

- training samples normalized `actions`, `proprio`, and `prev_actions`
- LIBERO rollout fed raw proprio into `control_step()`
- rollout sent model-space action directly to `env.step()`

Impact:

- closed-loop rollout could run in a numerically different space from training
- success-rate numbers were not trustworthy when stats were non-identity

### P0. Default `control_step()` RTC path could crash

Before this change:

- `RTCInferConfig` lacked `overlap_ratio`
- `HybridVLAv2.control_step()` accessed `cfg.infer.rtc.overlap_ratio`

Impact:

- default online path could raise `AttributeError`

### P0. Infer-side FASTER was exposed but not implemented

Before this change:

- `InferConfig.faster` existed
- `control_step()` ignored it entirely

Impact:

- configuration suggested a capability that the runtime did not honor
- train / infer drift remained silent

### P1. Checkpoints were weights-first, not inference-first

Before this change:

- checkpoints stored model/optimizer/scheduler/ema/meta
- resolved config and normalizer stats were not copied into the checkpoint

Impact:

- inference had to rediscover assets indirectly from stage directories
- portability and reproducibility were weaker than needed

## Implemented Changes

### 1. Added a unified LIBERO inference policy

New file:

- `vla_hybrid_v2/infer/libero_policy.py`

What it does:

- resolves config from checkpoint assets or stage output
- loads action/proprio normalizers for inference
- converts LIBERO observations into semantic inputs
- normalizes proprio before `control_step()`
- denormalizes model-space action before `env.step()`
- keeps `prev_action` in model space so temporal state stays consistent with training

Result:

- rollout now uses the same value space contract as the training dataset adapters

### 2. Fixed the RTC infer schema bug

Changed:

- `vla_hybrid_v2/config.py`

Details:

- added `RTCInferConfig.overlap_ratio = 0.333`
- clamped runtime RTC overlap to valid bounds in `control_step()`

Result:

- default RTC inference path no longer depends on a missing field

### 3. Made infer-side FASTER fail fast instead of fail open

Changed:

- `vla_hybrid_v2/config.py`
- `vla_hybrid_v2/models/hybrid_vla_v2.py`

Details:

- changed `FASTERInferConfig.enable` default from `True` to `False`
- if a caller explicitly enables infer-side FASTER, `control_step()` now raises `NotImplementedError`

Result:

- the runtime no longer advertises a silently ignored feature

### 4. Routed LIBERO rollout through the unified policy

Changed:

- `libero_hybrid/scripts/eval_libero_rollout.py`

Details:

- replaced ad hoc rollout preprocessing with `HybridVLALiberoPolicy`
- `_find_resolved_config()` now also recognizes checkpoint-local copied assets
- removed zero-token and zero-proprio fallback behavior from the rollout path

Result:

- rollout is stricter and closer to OpenPI-style policy wrapping

### 5. Added minimal checkpoint asset packaging

Changed:

- `vla_hybrid_v2/utils/checkpointing.py`
- `scripts/train_unified.py`

Details:

- `train_unified.py` now writes `resolved_config.yaml` into the stage output dir
- `save_checkpoint()` now accepts `asset_paths`
- checkpoints copy:
  - `resolved_config.yaml`
  - `normalizer_stats/` when present

Checkpoint layout now includes:

```text
checkpoint-STEP/
  model.pt
  optimizer.pt
  scheduler.pt
  ema.pt
  meta.json
  assets/
    resolved_config.yaml
    normalizer_stats/
      action_stats.json
      proprio_stats.json
```

Result:

- checkpoints are closer to self-contained inference bundles

### 6. Updated drifted documentation

Changed:

- `README.md`
- `libero_hybrid/README.md`

Details:

- removed stale claim that RTC/FASTER training logic was not wired
- softened the overly strong "closed-loop ready" wording
- documented that generic inference abstraction is still in progress

## Tests Added

New test files:

- `tests/test_infer_policy.py`
- `tests/test_control_step.py`
- `tests/test_checkpoint_assets.py`

Extended:

- `tests/test_eval_config_resolution.py`

Coverage added:

- normalized proprio is passed into online control
- model-space action is denormalized for env execution
- `prev_action` stays in normalized/model space across timesteps
- default RTC infer path executes
- infer-side FASTER raises clearly when enabled
- checkpoint assets are copied
- resolved config can be found from checkpoint-local assets

## Validation

Recommended validation command:

```bash
pytest -q
```

Targeted test areas after this patch:

- online `control_step()` path
- inference policy transform contract
- checkpoint inference asset packaging
- config discovery for LIBERO rollout

## Remaining Gaps

This patch does not yet solve everything from the comparison report.

Still pending:

- a real infer-side FASTER schedule
- a generic non-LIBERO policy wrapper
- broader checkpoint metadata beyond config + normalizers
- deployment/runtime abstractions comparable to OpenPI server/client stack
- CI wiring for these new inference-path tests

## Net Effect

This round does not change the core model architecture. It hardens the system glue around it.

The most important practical improvement is:

- LIBERO rollout now executes in a training-aligned input/output space instead of mixing raw env values with normalized model targets.
