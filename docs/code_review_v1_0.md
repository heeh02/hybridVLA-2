# HybridVLA v2 Independent Code Review v1.0

**Date**: 2026-03-29
**Reviewer**: Independent audit (Claude Opus 4.6)
**Scope**: v0.10.8/v0.10.9/v0.10.10 changes — priority files listed below
**Method**: Read-only; all claims from prior fix reports treated as unverified

## Priority Files Audited

| File | Lines Read |
|---|---|
| `scripts/train_unified.py` | Full |
| `vla_hybrid_v2/utils/ema.py` | Full |
| `vla_hybrid_v2/utils/checkpointing.py` | Full |
| `vla_hybrid_v2/models/hybrid_vla_v2.py` | Full |
| `vla_hybrid_v2/models/mamba_core.py` | Full |
| `vla_hybrid_v2/models/qwen2vl_backbone.py` | Full |
| `vla_hybrid_v2/models/flow_action_expert.py` | Full |
| `vla_hybrid_v2/models/discrete_heads.py` | Full |
| `vla_hybrid_v2/losses/consistency_loss.py` | Full |
| `vla_hybrid_v2/losses/flow_matching.py` | Full |
| `vla_hybrid_v2/infer/libero_policy.py` | Full |
| `vla_hybrid_v2/types.py` | Full |
| `tests/` (all 10 files) | Full |

---

## 1. Claims Verified

| Claim | Verdict | Evidence |
|---|---|---|
| FSDP model state dict saved with `rank0_only=True` | **Confirmed** | `checkpointing.py:29-31` |
| FSDP load uses `state_dict_type` context (N1 fix) | **Confirmed** | `checkpointing.py:147-156` |
| EMA initialised before FSDP wrapping | **Confirmed** | `train_unified.py:391` before `wrap_fsdp` at line 402 |
| EMA apply/restore use `summon_full_params(writeback=True)` | **Confirmed** | `ema.py:99, 108` |
| Per-module LR covers all trainable params | **Confirmed** | `train_unified.py:413-433` iterates all `named_parameters` |
| L-12: VICReg variance term added to ContrastiveTemporalLoss | **Confirmed** | `consistency_loss.py:55-59` |
| L-13: ActionConsistencyLoss replaced projection+cosine with MSE | **Confirmed** | `consistency_loss.py:98-99` |
| L-14: SlowFastAgreementLoss removed `.detach()` | **Confirmed** | `consistency_loss.py:82`, bidirectional gradient |
| Eval barrier placement prevents FSDP deadlock | **Confirmed** | `train_unified.py:563-573`: apply -> barrier -> eval -> barrier -> restore -> barrier |
| `_strip_fsdp_prefix` used consistently in EMA | **Confirmed** | `ema.py:93, 101, 110` |

---

## 2. Issues Confirmed Closed

| ID | Title | Evidence |
|---|---|---|
| N1 | FSDP load key mismatch | `checkpointing.py:153-156` uses `FSDP.state_dict_type` context with `FullStateDictConfig(rank0_only=False)` |
| L-12 | ContrastiveTemporalLoss collapse | `consistency_loss.py:55-59` adds VICReg variance regularisation with configurable weight and target |
| L-13 | ActionConsistency projection collapse | `consistency_loss.py:98-99` is now direct `F.mse_loss(discrete, continuous)` |
| L-14 | SlowFast unidirectional gradient | `consistency_loss.py:82` — no `.detach()`, both streams receive gradients |
| A2 | `id()`-based semantic refresh detection | `hybrid_vla_v2.py:744` uses monotonic `refresh_counter` |
| P0-1b | Stage-specific freeze gate | `train_unified.py:120-186` explicit gate + `sanity_check_trainable_params` assertions |

---

## 3. Issues Still Open

### 3.1 [P0] FSDP Optimizer State Dict Save/Load Corrupted on Multi-GPU Resume

**Severity**: P0 — silent wrong results after resume on >1 GPU

**Evidence**: `checkpointing.py:63`
```python
torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
```

Under FSDP, `optimizer.state_dict()` returns rank-local sharded state. Only rank 0 executes this line (line 52 gate), so only rank 0's shard is saved. On resume (`checkpointing.py:167-170`), every rank loads rank 0's shard via:
```python
optimizer.load_state_dict(
    torch.load(ckpt_dir / "optimizer.pt", map_location=map_location, weights_only=True)
)
```

Each rank gets rank 0's partial Adam momentum/variance buffers. Training continues with silently corrupted optimizer state (wrong momentum, wrong adaptive learning rate), causing convergence degradation that is extremely hard to diagnose.

Note: the model state dict correctly uses `FullStateDictConfig(rank0_only=True)` + `FSDP.state_dict_type` context (`checkpointing.py:28-31`). The optimizer lacks the equivalent `FSDP.full_optim_state_dict()` / `FSDP.optim_state_dict_to_load()`.

**Files**: `vla_hybrid_v2/utils/checkpointing.py:63, 167-170`

**Why current fix is insufficient**: Model save path was fixed for FSDP, optimizer save path was not.

**Minimal test**: 2-GPU: save checkpoint -> load on different rank mapping -> assert optimizer `exp_avg` and `exp_avg_sq` buffers match pre-save values exactly.

---

### 3.2 [P1] Inference Policy Clamps Denormalised Actions to Model-Space Range

**Severity**: P1 — silent action truncation on any dataset with raw actions outside [-1, 1]

**Evidence**: `libero_policy.py:410-412`
```python
action_env = self.action_normalizer.denormalize(action_model)
lo, hi = self.cfg.model.heads.action_range   # (-1.0, 1.0)
action_env = action_env.clamp(lo, hi)
```

`denormalize()` maps from model space `[-1, 1]` back to raw env space. `action_range = (-1.0, 1.0)` is the **model-space** range. Clamping the **env-space** result to `(-1, 1)` silently clips any action whose raw magnitude exceeds 1.0.

Currently harmless on LIBERO (raw actions happen to be in [-1, 1]), but is a latent bug for any other dataset or robot.

**Files**: `vla_hybrid_v2/infer/libero_policy.py:410-412`

**Why current fix is insufficient**: The clamp target should be the normalizer's raw-space min/max, or removed entirely (the model should produce in-range actions by construction).

**Minimal test**: Create normalizer with raw-space range [-5, 5]; normalize -> model -> denormalize -> assert output is NOT clamped to [-1, 1].

---

### 3.3 [P1] `refresh_counter` Never Incremented by Inference Policy

**Severity**: P1 — semantic refresh has no effect during inference unless caller manually increments

**Evidence**: `grep -r "refresh_counter" vla_hybrid_v2/infer/` returns **zero matches**.

`semantic_step_from_obs()` (`libero_policy.py:345-358`) does NOT increment `refresh_counter`. The model's `control_step()` (`hybrid_vla_v2.py:744`) checks `runtime_state.refresh_counter != runtime_state._last_seen_refresh` to detect semantic refreshes and trigger new chunk generation.

The only increment is in the eval rollout script (`eval_libero_rollout.py:160`), which the caller must remember to do manually. The `RuntimeCache` docstring (`types.py:92`) says "Caller increments refresh_counter after each semantic_step()" — this is an API trap.

If a caller uses `HybridVLALiberoPolicy` without manually incrementing, the model will keep replaying stale action chunks, only generating new ones when `chunk_step >= exec_horizon`, ignoring visual observation changes.

**Files**: `vla_hybrid_v2/infer/libero_policy.py:345-358`, `vla_hybrid_v2/types.py:92-93`

**Why current fix is insufficient**: The policy class should either auto-increment in `semantic_step_from_obs`, or raise/warn when `control_step_from_obs` is called without a preceding increment.

**Minimal test**: Call `semantic_step_from_obs` then `control_step_from_obs`; verify `semantic_refresh=True` is detected inside `control_step`.

---

### 3.4 [P1] FSDP Gradient Accumulation Missing `no_sync`

**Severity**: P1 — communication cost multiplied by `grad_accum` (default 4x), no correctness impact

**Evidence**: `train_unified.py:519-528`
```python
loss = losses["loss_total"] / grad_accum
loss.backward()                                    # all-reduce on EVERY micro-batch
...
if (batch_idx + 1) % grad_accum == 0:
    optimizer.step()
```

Under FSDP, each `.backward()` triggers an all-reduce. Intermediate micro-batches perform wasteful all-reduces whose accumulated results are simply added to by the next micro-batch. The default `grad_accum_steps=4` means 4x the necessary FSDP communication.

Gradients are still numerically correct (they accumulate additively), but multi-GPU training is ~4x slower in the communication-bound regime.

**Files**: `scripts/train_unified.py:519-528`

**Why current fix is insufficient**: Non-final micro-batches must be wrapped in `model.no_sync()`.

**Minimal test**: Mock FSDP module, count all-reduce invocations, assert count equals `ceil(total_batches / grad_accum)` not `total_batches`.

---

### 3.5 [P2] `batch_idx` Resets at Epoch Boundary — Gradient Accumulation Misalignment

**Severity**: P2 — cross-epoch gradient mixing + wrong loss scaling on boundary

**Evidence**: `train_unified.py:513, 528`
```python
for batch_idx, batch in enumerate(loader):     # resets to 0 every epoch
    ...
    if (batch_idx + 1) % grad_accum == 0:      # accumulation gate uses batch_idx
```

If `len(loader)` is not a multiple of `grad_accum`:
1. Final micro-batches of epoch N accumulate gradients but never trigger `optimizer.step()`
2. These leftover gradients carry into epoch N+1's first optimizer step
3. That step mixes data from two different epoch shuffles
4. The loss was divided by `grad_accum` but the step has contributions from both epochs

**Files**: `scripts/train_unified.py:513, 528`

**Why current fix is insufficient**: Needs either (a) epoch-end gradient flush, or (b) use `global_step`-based accumulation gate that doesn't reset.

**Minimal test**: `dataset_size=5, batch_size=1, grad_accum=4`, train 2 epochs, verify optimizer steps use exactly `grad_accum` micro-batches each.

---

### 3.6 [P2] EMA `load_state_dict` Merges Orphan Keys from Stale Checkpoints

**Severity**: P2 — GPU memory leak proportional to removed parameters, grows across resumes

**Evidence**: `ema.py:144`
```python
self.shadow.update(loaded_shadow)
```

After the shape-mismatch filter (`ema.py:131-137`) removes keys with wrong shapes, the remaining keys are merged unconditionally. Keys present in the checkpoint but absent from the current model's `self.shadow` (e.g., after architecture changes or module removal between stages) become orphans: they consume GPU memory but are never touched by `update()` (which only iterates `model.named_parameters()`).

**Files**: `vla_hybrid_v2/utils/ema.py:124-144`

**Why current fix is insufficient**: The shape-mismatch filter only handles keys that exist in both dicts with different shapes. Entirely extra keys are not filtered.

**Minimal test**: Create EMA state with extra key `"removed_module.weight"`, load into fresh EMA, assert key is NOT in `ema.shadow`.

---

### 3.7 [P2] EMA `state_dict()` Returns Live References — `load_state_dict` Mutates Input

**Severity**: P2 — latent corruption hazard

**Evidence**: `ema.py:115-121`
```python
def state_dict(self) -> dict:
    return {"shadow": self.shadow, ...}   # same dict object, same tensor objects
```

`load_state_dict` at line 137 does `del loaded_shadow[k]`, which mutates the dict inside the `state` argument. If `state` was obtained from another EMA's `state_dict()`, this deletes keys from the source EMA's shadow.

Currently safe because `torch.save` is the only consumer and it serialises immediately, but any in-memory round-trip (e.g., distillation, multi-model training) would silently corrupt.

**Files**: `vla_hybrid_v2/utils/ema.py:115-121, 124, 137`

**Minimal test**: `state = ema1.state_dict(); ema2.load_state_dict(state); assert len(ema1.shadow) == original_count`.

---

### 3.8 [P2] Inference Missing Autocast / dtype Management

**Severity**: P2 — higher memory usage + numerical drift vs training

**Evidence**: `libero_policy.py:250` does `model.to(device).eval()` without any autocast. Training uses `torch.autocast("cuda", dtype=torch.bfloat16)` (`train_unified.py:519`).

At inference time, the backbone runs in native bf16 but all other modules (grounder, temporal core, expert, projections) run in fp32. Without autocast, intermediate type promotions differ from training, producing a numerical distribution shift. The config field `infer.torch_dtype` exists (`config.py:300`) but is never read by the policy.

**Files**: `vla_hybrid_v2/infer/libero_policy.py:250`, `vla_hybrid_v2/config.py:300`

**Minimal test**: Compare model output under eval with and without `torch.autocast(dtype=bf16)`; assert difference < epsilon.

---

### 3.9 [P2] Multi-Camera `image_grid_thw` Flat Format Assumes Equal Images Per Sample

**Severity**: P2 — wrong camera embeddings if images-per-sample varies within a batch

**Evidence**: `qwen2vl_backbone.py:106-107`
```python
n_img = image_grid_thw.shape[0] // B
grids_b = image_grid_thw[b * n_img: (b + 1) * n_img]
```

When `image_grid_thw` is 2D (flat format, shape `[total_images, 3]`), the code assumes `total_images` divides evenly by batch size `B`. If different samples have different numbers of cameras/images, the integer division produces incorrect slicing. The code would either crash (index out of bounds) or assign wrong camera position embeddings.

**Files**: `vla_hybrid_v2/models/qwen2vl_backbone.py:106-107`

**Minimal test**: Construct a batch where sample 0 has 2 images and sample 1 has 3 images; verify `CameraPositionEmbedding` output is correct per sample.

---

### 3.10 [P2] `_build_cond_prefix` Silently Truncates Temporal Tokens When `compressed_slots != 24`

**Severity**: P2 — silent information loss if config is modified

**Evidence**: `hybrid_vla_v2.py:262-291`

The default cond token budget is:
```
1 (global) + 24 (compressed_slots) + 1 (phase) + 1 (uncertainty)
+ 1 (affordance) + 1 (fused_state) + 3 (fast/medium/slow) = 32
```

`cond_tokens=32` in default config matches exactly. If `compressed_slots` is changed to any value != 24, the total exceeds or falls below 32. The truncation path (lines 278-285) discards the **last** tokens — which are the temporal tokens (fast/medium/slow) — with only a log warning. These carry critical temporal information for the action expert.

**Files**: `vla_hybrid_v2/models/hybrid_vla_v2.py:262-291`

**Minimal test**: Set `compressed_slots=28`, run `_build_cond_prefix`, verify which tokens were dropped and assert a loud warning or error.

---

### 3.11 [P3] `_log_per_module_grad_norm` Reports Wrong Norms Under FSDP

**Severity**: P3 — misleading monitoring, no correctness impact

**Evidence**: `train_unified.py:276-292`

Under FSDP, `p.grad` is the gradient of the local shard only. The computed L2 norm `p.grad.detach().norm(2)` is the shard-local norm, roughly `sqrt(world_size)` lower than the true full-parameter gradient norm. Logged gradient norms on multi-GPU are systematically under-reported.

**Files**: `scripts/train_unified.py:276-292`

**Minimal test**: N/A (monitoring only).

---

### 3.12 [P3] `accum_loss` Never Cleared on Non-Rank-0 Processes

**Severity**: P3 — unbounded memory growth on non-rank-0

**Evidence**: `train_unified.py:525-553`

`accum_loss` dict is populated on every micro-batch (line 526) but `accum_loss.clear()` is inside `is_main_process()` gate (line 542/553). Non-rank-0 processes accumulate loss values forever. Not a correctness issue (only rank 0 logs), but a memory leak proportional to training length.

**Files**: `scripts/train_unified.py:525-553`

---

## 4. New Regressions Introduced

### 4.1 [P2] InfoNCE Logits Matrix Is O(N^2) in Batch × Time

**Source**: v0.10.10 L-12 fix — replaced L2 smoothness with InfoNCE

**Evidence**: `consistency_loss.py:49`
```python
logits = torch.matmul(a, p.T) / self.temperature   # [B*(T-1), B*(T-1)]
```

For `B=64, T=8`, this is `448 x 448` (~200K elements) — manageable. For `B=128, T=16`, this is `1920 x 1920` (~3.7M elements), and backprop requires storing the full logits matrix. The previous L2 smoothness loss was O(B*T*D) with no quadratic term.

**Files**: `vla_hybrid_v2/losses/consistency_loss.py:49`

**Minimal test**: Run with `B=128, T=16`, measure GPU memory delta from consistency loss alone.

---

### 4.2 [P3] `sanity_check_trainable_params` Missing Several Modules

**Source**: Modules were added to `always_trainable` / `extra_modules` but not to the sanity check

**Evidence**: `train_unified.py:195-210` `module_entries` list omits:
- `proprio_to_expert`
- `emb_to_expert`
- `flow_matching_loss`
- `discrete_loss`
- `phase_loss`

These are unfrozen at `train_unified.py:145-177` but the sanity check would not detect if they were accidentally frozen.

**Files**: `scripts/train_unified.py:195-210`

---

## 5. Test Gaps

### 5.1 Critical — No Coverage

| Gap | Severity | Detail |
|---|---|---|
| **Checkpoint save/load round-trip (weights)** | P0 | `test_checkpoint_assets.py` only tests asset file copying. No test saves model weights and verifies they survive `save_checkpoint` -> `load_checkpoint`. |
| **FSDP optimizer save/load** | P0 | Zero test coverage. The P0 bug in section 3.1 has no test that could catch it. |
| **EMA under real FSDP wrapping** | P0 | `test_ema_fsdp_gaps.py:131-170` uses `FSDPSimWrapper` (adds `_fsdp_wrapped_module` attribute only). `_is_fsdp()` returns `False` for this mock, so `summon_full_params` is never exercised. The most dangerous code path (sharded apply/restore with writeback) is completely untested. |
| **Multi-camera batch processing** | P1 | `conftest.py:49` hardcodes `MultiCameraConfig(enable=False)`. The entire `CameraPositionEmbedding`, multi-camera training path, and multi-camera inference path have zero coverage. |
| **Mamba fallback vs official path equivalence** | P1 | Only `mamba_impl="fallback"` is tested (`test_ema_fsdp_gaps.py:345`). No test compares fallback and official outputs for numerical equivalence. |
| **`ActionHistoryBuffer` ring-buffer correctness** | P1 | `types.py:104-131` has modular-arithmetic reordering in `get()`. Zero tests. |

### 5.2 High — Trivially Passing Tests

| Test | Problem |
|---|---|
| `test_losses.py::test_perfect_prediction_zero_loss` | Passes by identity (`target_v = x_1 - x_0`, then `loss(target_v, x_0, x_1)` is zero by construction). Proves nothing about the loss formula for non-trivial inputs. |
| `test_losses.py::test_temporal_loss_runs` | Only asserts `shape == ()` and `not isnan`. A function returning constant `0.0` would pass. No check that InfoNCE with random inputs produces loss ~ `log(N)`. |
| `test_losses.py::test_slow_fast_agreement` | Same: shape + non-NaN only. |
| `test_losses.py::test_action_consistency` | `assert loss.item() >= 0.0` — MSE of random vectors is always positive. |
| `test_losses.py::test_combined_loss` | `assert not torch.isnan(loss)` — no lower bound check. |
| `test_expert.py::test_forward_shape` | Shape-only. Does not check finiteness, `requires_grad`, or value range. |
| `test_expert.py::test_sample_dispatch` | Shape-only. Does not verify Euler vs midpoint produce different outputs. |
| `test_ema_fsdp_gaps.py::test_multi_step_state_propagation` | Only checks `not isnan` at step 3. A no-op core producing constant output would pass. |

### 5.3 Medium — Tests That Mock the Thing They Should Test

| Test | Problem |
|---|---|
| `conftest.py::_MockBackbone` | Backbone is mocked everywhere. Backbone -> grounder -> temporal -> expert data flow is never tested with real tensor relationships. |
| `test_infer_policy.py::_DummyModel` | Replaces `HybridVLAv2` with a model that always returns `action=0.5`. Real `control_step()` (chunk caching, RTC blending, temporal processing) is never exercised through the policy wrapper. |
| `test_ema_fsdp_gaps.py::_classify_param` (line 181) | Re-implements the param classification logic locally instead of importing from `train_unified.py`. If production code changes its prefix rules, the test still passes. |

### 5.4 Other Gaps

| Gap | Severity |
|---|---|
| `FASTDiscreteHead.discretise_actions` / `undiscretise_actions` round-trip | P2 |
| `_build_cond_prefix` truncation/padding paths | P2 |
| `auto_resume` / `find_latest_checkpoint` | P2 |
| Data pipeline (HDF5 adapter, collation, augmentation) | P2 |
| Inference `from_checkpoint` end-to-end | P2 |
| `StaleTimeEncoding` edge cases | P3 |
| `test_full_round_trip_apply_restore_is_identity` does not assert `ema.backup` is cleared after `restore()` | P3 |

---

## 6. Top 5 Items to Fix Immediately

| Rank | Issue | Section | Rationale |
|---|---|---|---|
| **1** | FSDP optimizer state save/load corrupted | 3.1 | Multi-GPU resume silently produces wrong Adam state -> training degrades without any error signal. Hardest to diagnose post-hoc. |
| **2** | Inference denormalize clamp uses model-space range | 3.2 | Blocks any dataset with raw actions outside [-1, 1]. Latent on LIBERO but will surface immediately on new robots/datasets. |
| **3** | `refresh_counter` not auto-incremented by policy | 3.3 | Any external caller of `HybridVLALiberoPolicy` that forgets manual increment gets stale action chunks — a silent correctness failure at inference. |
| **4** | FSDP `no_sync` missing in gradient accumulation | 3.4 | No correctness impact but ~4x communication overhead on default config. Multi-GPU training speed will be significantly worse than expected. |
| **5** | Checkpoint round-trip test coverage = zero | 5.1 | Without this test, there is no automated way to verify that fix #1 above is correct. This is the confidence foundation for all resume-related code. |

---

## Appendix: Verified Non-Issues

Items investigated and confirmed NOT to be bugs:

| Item | Reason |
|---|---|
| `fast_continuous` gradient asymmetry in consistency loss | By design: discrete head gets gradient toward expert; expert is detached to prevent unstable bidirectional action loss |
| `cond_builder` frozen in Stage A but invoked | `_build_cond_prefix` is inside `if stage != "a":` guard — consistent |
| FASTER `far_ratio` weight direction | Math is correct despite confusing naming: `far_steps/near_steps = 4` applied to near positions gives near 4x weight |
| `torch.roll` vs `ActionHistoryBuffer` at inference | Semantics match (both produce chronological order); implementation differs but correctness is preserved |
| `flow_t` passed but unused in `FlowMatchingLoss` | Documented as API compatibility; Rectified Flow target velocity is t-independent by design |
| Cosine schedule `min_lr_ratio` per-group handling | `LambdaLR` applies the multiplier to each group's `initial_lr` — correct |
| `_save_resolved_config` `mkdir` on all ranks | `exist_ok=True` prevents errors; benign race on local FS, minor concern on NFS only |
