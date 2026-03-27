# HybridVLA v2 — v0.10 Comprehensive Code Structure Analysis

Full codebase audit following v0.9.3 infrastructure build. Serves as the baseline for the v0.10 development cycle. (v1.0 reserved for real training start.)

---

## Part 1: Codebase Overview

### Architecture Summary

```
Qwen2-VL-7B (frozen + LoRA r=64)
       │
   MultiScaleAdapter (layers 10/18/28 → 2048d)
       │
   HierarchicalAttentionGrounder (96 → 72 latents, 24 compressed slots)
       │
   TriRateMambaCore
   ├── Fast  (20L, 50 Hz,  d_state=128)
   ├── Medium (6L, 25 Hz,  d_state=128)
   └── Slow  (10L, 12.5 Hz, d_state=256)
       │
   CrossAttentionFusion → fused_state
       │
   ┌───┴───┐
   │       │
FASTDiscrete  FlowActionExpert (18L, M-M-A×6, 1536d)
(512 bins)    (midpoint ODE → 24-step action chunk)
```

### File Inventory

| Layer | Files | Lines | Status |
|-------|-------|-------|--------|
| **Models** | 6 files | ~2,365 | Mature |
| **Types + Config** | 2 files | 497 | Stable |
| **Data** (v0.9.3) | 7 files | ~658 | New — issues found |
| **Losses** | 3 files | ~157 | Stable |
| **Utils** | 3 files | ~402 | Stable |
| **Ops** | 1 file | 55 | Stable |
| **Scripts** | 2 files | ~465 | Partial |
| **World Model** | 9 files | ~1,100 | Inactive (disabled) |
| **Infer** | 1 stub | 2 | Empty |
| **Tests** | 0 files | 0 | Empty |
| **Configs** | 4 YAML | ~200 | Partial (data/infer empty) |
| **Total** | ~41 files | ~5,900+ | |

---

## Part 2: Model Layer Audit

### 2.1 `hybrid_vla_v2.py` (683 lines) — **PASS**

| Component | Lines | Status | Evidence |
|-----------|-------|--------|----------|
| `__init__` | 61-203 | Clean | All sub-modules assembled, projections use correct dims |
| `proprio_proj` | 139 | Fixed (v0.9.1) | `nn.Linear(mcfg.proprio_dim, d_core)` — decoupled from action_dim |
| `_fast_bin_centers` | 166-167 | Fixed (v0.9.2) | Uses `cfg.model.heads.action_range` from config |
| `discrete_loss` | 174 | Fixed (v0.9.2) | `DiscreteCELoss(label_smoothing=mcfg.heads.label_smoothing)` |
| `_build_cond_prefix` | 229-256 | Clean | 32 tokens: 1(global)+24(slots)+7(special), pad/truncate to target |
| `_validate_batch` | 262-328 | Enhanced (v0.9.3) | 5 required keys + T consistency + input/mask match + vision co-occurrence + embodiment range |
| `forward_train` | 334-549 | Clean | Stage-gated expert, correct denoising formula |
| Denoising formula | 532 | Fixed (v0.9.1) | `noisy_actions + (1.0 - flow_t[:, None, None]) * expert_out.velocity` |
| Grounder mask | 369, 383, 563 | Fixed (v0.9.1) | All 3 call sites pass `attention_mask=` |
| `control_step` | 568-676 | Clean (v0.9.1) | Returns `ControlStepOutput(action=...)`, uses refresh_counter |
| Medium stride | 600-602 | Fixed (v0.9.1) | Computed from `infer.control_hz / infer.medium_hz` |

**No issues found.** All v0.9.0–v0.9.2 fixes verified intact.

### 2.2 `mamba_core.py` (786 lines) — **PASS**

| Component | Lines | Status |
|-----------|-------|--------|
| `MambaBlock.res_scale` | 107 | `nn.Parameter(torch.ones(1))` — learnable per-block |
| `_MambaStack` init | 352-354 | `1/sqrt(N)` initialization for all stacks |
| Fast/Medium/Slow stacks | 477-501 | 20/6/10 layers, correct d_state |
| Token-by-token processing | 432-454 | Official Mamba2 path uses `.step()` for state capture, documented |
| Fallback path | 456-474 | `activation_checkpoint` applied when `use_checkpoint=True` |
| `CrossAttentionFusion` | 537-594 | Learned fusion via multi-head cross-attention |
| `TriRateMambaCore` | 601-785 | Correct tri-rate scheduling logic |

**Known limitation**: Token-by-token processing (L=33 Python loop) loses intra-sequence parallelism. Documented as acceptable (33 tokens/step vs ~1000 for full sequence parallelism). Custom CUDA kernel would provide ~33x speedup.

### 2.3 `flow_action_expert.py` (344 lines) — **PASS**

| Component | Lines | Status |
|-----------|-------|--------|
| `AdaRMSNorm` gate bias | 44-47 | `bias[2*dim:].fill_(2.0)` → sigmoid(2.0) ≈ 0.88 |
| `ExpertMambaBlock` | 94-161 | Correct selective scan integration |
| `ExpertAttentionBlock` | 168-226 | Cross-attn to cond_prefix + self-attn + FFN |
| `FlowActionExpert.forward` | 271-306 | 18-layer hybrid (M-M-A×6), proprio+embodiment prepended |
| `sample_euler` | 309-319 | Standard 1st-order: `x += v * dt` |
| `sample_midpoint` | 322-336 | 2nd-order: half-step then full-step |

**No issues found.**

### 2.4 `attention_grounder.py` (261 lines) — **PASS**

| Component | Lines | Status |
|-----------|-------|--------|
| `CrossAttentionLayer` | 42-79 | SDPA with additive float mask (`-inf` for masked) |
| `SlotCompression` | 128-150 | 48→24 via learned routing weights |
| `HierarchicalAttentionGrounder` | 157-260 | 96 latents, 8 layers, compression at layer 4 |

**No issues found.** Latent layout management is explicit and correct.

### 2.5 `qwen2vl_backbone.py` (214 lines) — **PASS** (1 minor)

| Component | Lines | Status |
|-----------|-------|--------|
| `MultiScaleAdapter` | 24-53 | Layers [10,18,28], gated fusion to 2048d |
| `_apply_freeze` | 110-134 | Vision tower + text layers [0, freeze_until) |
| `_apply_lora` | 136-161 | LoRA r=64 on all 28 layers |
| `forward_semantic` | 181-213 | Returns fused features + vision_mask + text_mask |

**Minor**: Line 146 — `total_layers = len(text_model.layers) if hasattr(...) else 28` hardcodes fallback. Low risk since Qwen2-VL-7B is fixed architecture.

### 2.6 `discrete_heads.py` (77 lines) — **PASS**

All three heads (FAST 512-bin, Phase 16-class, Affordance 8-class) are clean. `discretise_actions` uses `lo, hi` parameters consistently.

### 2.7 `types.py` (128 lines) — **PASS**

All 7 dataclasses verified: `GrounderOutput`, `TriRateTemporalState`, `TemporalOutput`, `ActionExpertOutput`, `ControlStepOutput`, `RuntimeCache`, `ActionHistoryBuffer`. Fields match all usage in model code.

### 2.8 `config.py` (369 lines) — **PASS**

All dataclasses complete. v0.9.2 additions (`label_smoothing`, `action_range`, unknown key warning) verified. `proprio_dim` properly decoupled. `DataConfig` has 13 fields, 9 now consumed by data layer (v0.9.3).

---

## Part 3: Data Layer Audit (v0.9.3)

### 3.1 `schema.py` (75 lines) — **CRITICAL BUG**

**Bug D1: Field name mismatch between schema and forward_train()**

WindowSample defines:
```python
# schema.py:40-41
refresh_pixel_values: Optional[List[Tensor]] = None
refresh_image_grid_thw: Optional[List[Tensor]] = None
```

But BATCH_OPTIONAL_KEYS lists:
```python
# schema.py:73-74
"refresh_pixel_values_list",     # with _list suffix
"refresh_image_grid_thw_list",   # with _list suffix
```

And forward_train() accesses:
```python
# hybrid_vla_v2.py:367
batch.get("refresh_pixel_values_list", [None] * R)[r]
```

The collate function preserves dict keys as-is. When an adapter returns a dict with `refresh_pixel_values`, the collated batch will have key `refresh_pixel_values` — but forward_train() looks for `refresh_pixel_values_list`. The key mismatch causes silent fallback to `[None] * R`.

**Impact**: Latent bug. Currently dormant because no adapter produces refresh vision data. Will silently fail when multi-frame training is implemented.

**Fix options**:
- A) Rename WindowSample fields to `refresh_pixel_values_list` / `refresh_image_grid_thw_list` (match consumer)
- B) Add key renaming in collate_fn (fragile)
- C) Rename batch access in forward_train() to drop `_list` suffix (match producer)

**Recommended**: Option A — aligns schema with the batch protocol that forward_train() already expects.

### 3.2 `normalizer.py` (165 lines) — **PASS** (2 edge cases)

**Math verification**:

| Operation | Formula | Correct |
|-----------|---------|---------|
| min_max normalize | `(raw - min) / scale * (hi - lo) + lo` | Yes |
| min_max denormalize | `(normed - lo) / (hi - lo) * scale + min` | Yes, exact inverse |
| mean_std normalize | `tanh((raw - mu) / std) * range/2 + center` | Yes |
| mean_std denormalize | `atanh(centered) * std + mu` | Yes, with clamping |

**Edge case D2**: `eps=1e-6` in mean_std denormalize (line 126). `atanh(1 - 1e-6) ≈ 7.25` — valid but at numerical edge. Values beyond ~4 sigma are compressed by tanh and not fully recoverable. This is inherent to tanh-squash normalization and acceptable if documented.

**Edge case D3**: Zero-std features. If a feature is constant across all training data, `std ≈ 0`, `sd.clamp(min=eps)` produces `eps`. Then `z = (raw - mu) / eps` explodes for any `raw ≠ mu`. Should log a warning when `std < 1e-4`.

### 3.3 `base_adapter.py` (50 lines) — **PASS**

Clean abstract base class. Three required abstract methods: `__getitem__() -> WindowSample`, `__len__()`, `episode_lengths`. Stores cfg, normalizers, split.

### 3.4 `dummy.py` (63 lines) — **PASS**

Correctly uses config values (`action_range`, `proprio_dim`, `num_phases`). Values clamped to `[lo, hi]`. Returns dict (consistent with HDF5 adapter). Does not extend BaseDatasetAdapter (intentional — dummy doesn't need normalizers).

### 3.5 `hdf5_adapter.py` (176 lines) — **3 ISSUES**

**Bug D4: Short-episode window creation** (line 103)

```python
for start in range(0, max(1, T_ep - self.window + 1)):
```

When `T_ep < self.window` (e.g., T_ep=10, window=24):
- `max(1, 10 - 24 + 1) = max(1, -13) = 1`
- Creates `range(0, 1) = [0]` — one window with start=0
- `end = 0 + 24 = 24`, but `data[0:24]` only returns 10 steps (HDF5 silent truncation)
- Result: `actions` shape `[10, H, A]` instead of `[24, H, A]`
- Causes batch dimension mismatch in collation or `_validate_batch` failure

**Fix**:
```python
# Skip episodes shorter than window
if T_ep < self.window:
    logger.warning("Skipping episode %s (length %d < window %d)", path, T_ep, self.window)
    continue
for start in range(0, T_ep - self.window + 1):
    self._index.append((ep_idx, start))
```

**Bug D5: Inheritance mismatch** (line 39)

```python
class HDF5DatasetAdapter(Dataset):  # extends Dataset, NOT BaseDatasetAdapter
```

`BaseDatasetAdapter` was specifically created as the abstract contract. HDF5DatasetAdapter should extend it. Also, `__getitem__` returns `dict`, not `WindowSample` as specified by the abstract method signature.

**Fix**: Change to `class HDF5DatasetAdapter(BaseDatasetAdapter):` and either return `WindowSample` instances or update `BaseDatasetAdapter.__getitem__` return type to `dict`.

**Issue D6: No HDF5 key validation** (lines 100, 120-121)

`_build_index` accesses `f["data"][self.dcfg.action_key]` without try/except. If an HDF5 file is missing the expected key, the entire dataset initialization crashes with an opaque KeyError.

**Fix**: Wrap in try/except with informative error including file path and expected key.

### 3.6 `collate.py` (50 lines) — **1 ISSUE**

**Issue D7: Undocumented None handling** (lines 29-48)

If `values[0]` is `None` (e.g., `pixel_values=None` for text-only samples):
- `isinstance(None, Tensor)` → False
- `isinstance(None, list)` → False
- `isinstance(None, (int, float))` → False
- Falls through to pass-through: `batch[key] = values` → list of Nones

This actually works (forward_train uses `.get()` with defaults for optional keys), but is fragile and undocumented.

**Fix**: Add explicit None handling:
```python
if values[0] is None:
    batch[key] = None  # All samples should agree
    continue
```

### 3.7 `__init__.py` (79 lines) — **2 ISSUES**

**Issue D8: ProprioNormalizer uses action_range** (line 48)

```python
proprio_norm = ProprioNormalizer(target_range=cfg.model.heads.action_range)
```

Both normalizers use `(-1, 1)`. This works for now but is semantically questionable — proprio features (joint angles, velocities) may have different natural ranges than actions. If a future embodiment needs different proprio normalization, there's no config knob.

**Issue D9: No stats computation script**

`build_dataset()` requires pre-computed stats at `{output_dir}/normalizer_stats/`. But no script exists to compute these stats. Users must write their own before training can start with real data.

**Fix**: Add `scripts/compute_stats.py` that iterates over the dataset and calls `normalizer.fit()`.

---

## Part 4: Loss Layer Audit

### 4.1 `flow_matching.py` (32 lines) — **PASS**

- Target velocity: `x_1 - x_0` (line 17)
- Loss: MSE `(v_pred - v_target)^2` (line 18)
- step_weights: optional `[B, H]` broadcast to `[B, H, 1]` (lines 19-20)
- Interpolation: `x_t = (1-t)*x_0 + t*x_1` (line 31-32)
- Timestep: logit-normal via `sigmoid(randn())` (lines 24-26)

### 4.2 `consistency_loss.py` (96 lines) — **PASS**

- `ContrastiveTemporalLoss`: InfoNCE on consecutive fused_states. O(N^2) with N=B*(T-1). Acceptable for current scale.
- `SlowFastAgreementLoss`: EMA of fast tokens vs slow token. `fast_ema.detach()` prevents bi-directional gradient.
- `ActionConsistencyLoss`: Cosine similarity in projected space. Gradient blocked at call site via `expert_continuous.detach()` (model-level decision, not loss-level). v0.9.1 removed redundant detach from loss module.
- `V2ConsistencyLoss`: Combines all three with optional gating.

### 4.3 `discrete_loss.py` (29 lines) — **PASS**

- `DiscreteCELoss`: Reshapes `[B,H,A,V]` → `[*,V]`, applies `cross_entropy` with configurable label_smoothing.
- `PhaseLoss`: Standard cross_entropy wrapper.

---

## Part 5: Infrastructure Audit

### 5.1 `utils/checkpointing.py` (158 lines) — **PASS**

Robust implementation: atomic saves (temp dir → rename), symlink management, FSDP-aware state dict consolidation, `weights_only=True` for secure loading. Auto-resume via latest symlink.

### 5.2 `utils/distributed.py` (162 lines) — **PASS**

FSDP wrap policy covers all 4 block types (MambaBlock, GrounderBlock, ExpertMambaBlock, ExpertAttentionBlock). Mixed precision (bf16 params, fp32 reduce). Activation checkpointing support.

### 5.3 `utils/ema.py` (82 lines) — **PASS**

Linear decay ramp from `initial_decay` to `final_decay` over `ramp_steps`. Shadow params cloned at init, updated via `lerp_()`. Apply/restore for eval swapping.

### 5.4 `ops/selective_scan.py` (55 lines) — **PASS**

Probes for `mamba_ssm` CUDA kernel, falls back to JIT-compiled Python loop. Clean fallback hierarchy.

### 5.5 `scripts/train_stage_a.py` (264 lines) — **PASS** (gaps noted)

- **Data**: Uses `build_dataset(cfg)` (v0.9.3), passes `collate_fn` to DataLoader
- **Optimizer**: Two param groups (decay / no-decay), excludes `res_scale`, `bias`, `LayerNorm.weight`
- **LR Schedule**: Cosine with linear warmup (3000 steps)
- **FSDP**: Conditional wrapping for multi-GPU
- **EMA**: Conditional init, updated on every optimizer step
- **Checkpointing**: Save every 5000 steps, auto-resume, cross-stage loading with `strict=False`

**Gap S1**: No validation/eval loop. `eval_interval=2000` in config but unused.
**Gap S2**: No Stage B/C training scripts. Configs exist (`stage_b.yaml`, `stage_c.yaml`) but no scripts consume them. Reuse of Stage A script with `--config stage_b.yaml` would work but isn't documented.

### 5.6 `scripts/train_smoke_test.py` (201 lines) — **PASS** (intentional isolation)

Still uses inline DummyVLADataset (not the data module). Uses mini dimensions (A=7, P=9, H=4, L=16) with mock backbone. This is intentional — the smoke test validates model-layer correctness independent of data infrastructure.

### 5.7 Config YAMLs — **PASS** (gaps noted)

| File | Status | Key Settings |
|------|--------|-------------|
| `model/v2_qwen2vl_7b_trirate_expert18.yaml` | Complete | Full architecture specification |
| `train/stage_a.yaml` | Complete | 120K steps, lr=2e-4, expert frozen |
| `train/stage_b.yaml` | Complete | 200K steps, lr=1e-4, adds expert + FM loss |
| `train/stage_c.yaml` | Complete | 80K steps, lr=3e-5, RTC/FASTER enabled |
| `data/` | **Empty** | No dataset config templates |
| `infer/` | **Empty** | No inference config |

### 5.8 Repo Hygiene — **IMPROVED** (v0.9.3)

| Item | Status |
|------|--------|
| `.gitignore` | Present (42 lines, comprehensive) |
| `requirements.txt` | Present (torch>=2.1.0, transformers>=4.37.0, etc.) |
| `pyproject.toml` | **Missing** |
| `tests/` | **Empty** |
| `docs/` | 32 files (historical, needs archiving) |

---

## Part 6: Cross-Module Integration Analysis

### 6.1 Data → Model Flow

```
DataConfig → build_dataset() → (HDF5DatasetAdapter | DummyVLADataset, collate_fn)
                                        │
                                    DataLoader
                                        │
                            batch: Dict[str, Tensor]
                                        │
                              _validate_batch(batch)
                                        │
                              forward_train(batch)
```

**Verified**: All required keys from `_validate_batch` are produced by both DummyVLADataset and HDF5DatasetAdapter. Shape constraints match config values.

**Gap**: Vision path (`pixel_values`, `image_grid_thw`) is never populated by any adapter. DummyVLADataset omits these keys. HDF5DatasetAdapter omits them (text-only mode). This means the backbone's vision tower has **never been tested in the training loop**.

### 6.2 Model → Loss Flow

```
forward_train() produces:
├── loss_fast       ← DiscreteCELoss(logits, discretized_targets)
├── loss_phase      ← PhaseLoss(phase_logits, phase_labels)
├── loss_affordance ← CE(aff_logits, aff_labels)
├── loss_fm         ← FlowMatchingLoss(velocity, noise, targets, t, step_weights)
└── loss_consistency ← V2ConsistencyLoss(fused, fast, slow, discrete, continuous)
```

**Verified**: All loss inputs match expected shapes. Stage gating is correct (expert losses only in stage != "a"). Denoising formula `x_t + (1-t)*v` is mathematically correct and matches the interpolation formula.

### 6.3 Config → Everything Flow

`HybridVLAv2Config` is the single source of truth. All components read from it:
- Model dimensions: `action_dim`, `proprio_dim`, `chunk_horizon`, `d_model`, etc.
- Training params: `learning_rate`, `weight_decay`, `loss_weights`, etc.
- Data params: `format`, `paths`, `action_key`, etc.
- Inference params: `control_hz`, `medium_hz`, `execution_horizon`, etc.

**Verified**: No config field is consumed incorrectly. The unknown-key warning (v0.9.2) catches YAML typos.

---

## Part 7: Issue Registry

### CRITICAL (Must fix for v0.10)

| ID | File | Line | Issue | Impact |
|----|------|------|-------|--------|
| D1 | `data/schema.py` | 40, 73 | `refresh_pixel_values` vs `refresh_pixel_values_list` name mismatch | Silent fallback when refresh vision is implemented |
| D4 | `data/hdf5_adapter.py` | 103 | `max(1, T_ep - window + 1)` creates undersized windows for short episodes | Batch shape mismatch / crash |
| D5 | `data/hdf5_adapter.py` | 39 | Extends `Dataset` not `BaseDatasetAdapter`, returns `dict` not `WindowSample` | Contract violation |

### HIGH (Should fix for v0.10)

| ID | File | Line | Issue | Impact |
|----|------|------|-------|--------|
| D6 | `data/hdf5_adapter.py` | 100 | No HDF5 key validation in `_build_index` | Opaque crash on malformed data |
| D7 | `data/collate.py` | 29 | None values handled by pass-through, undocumented | Fragile for future changes |
| D9 | — | — | No stats computation script | Cannot start real data training |
| S1 | `train_stage_a.py` | — | `eval_interval=2000` unused, no eval loop | Cannot measure training progress |

### MEDIUM (Should fix before production)

| ID | File | Line | Issue | Impact |
|----|------|------|-------|--------|
| D2 | `data/normalizer.py` | 126 | atanh near-singularity with eps=1e-6 | Numerical instability for outliers |
| D3 | `data/normalizer.py` | 105 | No warning for zero-std features | Degenerate normalization |
| D8 | `data/__init__.py` | 48 | ProprioNormalizer uses action_range | Inflexible for multi-embodiment |
| S2 | `scripts/` | — | No Stage B/C scripts | Blocks multi-stage training |

### LOW (Future improvement)

| ID | File | Line | Issue | Impact |
|----|------|------|-------|--------|
| — | `infer/` | — | Empty stub | No inference encapsulation |
| — | `eval/` | — | Doesn't exist | No evaluation framework |
| — | `tests/` | — | Empty | No automated testing |
| — | `configs/data/` | — | Empty | No dataset config templates |
| — | `docs/` | — | 32 files unarchived | Context pollution for audits |

---

## Part 8: Scoring

### Dimension Scoring (10-point scale)

| # | Dimension | Score | Justification |
|---|-----------|-------|---------------|
| 1 | **Design coherence** | **8.5** | Clean separation: backbone→grounder→temporal→expert→heads. Data layer follows adapter pattern. Config is single source of truth. Minor: refresh field naming inconsistency. |
| 2 | **Correctness** | **9.0** | All math verified (denoising, interpolation, losses). Grounder mask connected. v0.9.1-v0.9.2 fixes intact. Deductions: HDF5 short-episode bug (D4), schema name mismatch (D1). |
| 3 | **Completeness** | **6.0** | Model complete. Data layer exists with real loader + normalizer. But: no eval, no runtime/policy, no Stage B/C scripts, no tests, vision path untested, no stats script. |
| 4 | **Training stability** | **9.0** | res_scale excluded from weight decay (v0.9.1). Cosine LR + warmup. Grad clipping. EMA with decay ramp. Gate bias init prevents activation collapse. |
| 5 | **Scalability** | **7.0** | FSDP with 4-class wrap policy. Mixed precision bf16. Activation checkpointing. But: backbone not sharded (15GB replicated, acceptable on H100-80GB). |
| 6 | **Performance** | **6.0** | Chunk caching saves 8x expert calls at inference. Token-by-token processing is 33x slower than optimal (needs custom CUDA). |
| 7 | **Production readiness** | **5.5** | .gitignore + requirements.txt + batch validation + secure checkpoint loading. But: no inference encapsulation, no denormalization pipeline, no env adapter. |
| 8 | **Code quality** | **8.0** | Consistent style, type annotations, docstrings on public methods. Config dataclasses well-structured. Minor: HDF5 adapter doesn't follow its own abstract base. |
| 9 | **Documentation** | **4.0** | Extensive audit trail in docs/ (32 files). But: no README, no API docs, no usage examples, no architecture diagram. Inline code comments are sparse. |
| 10 | **Testing** | **1.0** | `train_smoke_test.py` is the only test. No pytest suite. No unit tests for normalizer, collate, schema. `tests/` directory is empty. |
| | **Weighted average** | **6.8** | Weights: correctness(2x), completeness(1.5x), stability(1.5x), others(1x) |

### Historical Comparison

| Version | Weighted Score | Delta | Key Change |
|---------|---------------|-------|------------|
| v0.7.0 | 5.5 | — | Bug fixes (residual, freeze, conv state) |
| v0.7.2 | 7.0 | +1.5 | Cross-stage checkpoint, init, gate bias |
| v0.9.0 | 7.3 | +0.3 | res_scale, chunk caching, double-LN removal |
| v0.9.1 | 7.5 | +0.2 | Denoising fix, proprio, grounder mask, API cleanup |
| v0.9.2 | 7.3 | -0.2 | Config cleanup (re-calibrated scoring) |
| v0.9.3 | **6.8** | -0.5 | Data layer added but brings new bugs; testing/eval/docs now in scope |

**Note**: The apparent score decrease from v0.9.2 (7.3) to v0.9.3 (6.8) reflects expanded audit scope. v0.9.3 added 7 new files (data layer), expanding the surface area. Testing (1.0) and documentation (4.0) dimensions, now properly weighted, pull the average down. The model layer itself improved from 7.3 to 8.5.

---

## Part 9: Recommended Fix Order for v0.10

### Phase 1: Data layer bug fixes (1 day)

```
D1 → Rename schema.py fields to refresh_pixel_values_list / refresh_image_grid_thw_list
D4 → Fix hdf5_adapter._build_index: skip short episodes, use max(0, ...)
D5 → HDF5DatasetAdapter extends BaseDatasetAdapter
D6 → Add try/except with informative error in _build_index
D7 → Add explicit None handling in collate.py
```

### Phase 2: Training completeness (2-3 days)

```
D9 → Add scripts/compute_stats.py (normalizer stats from real data)
S1 → Add eval loop to training script (offline action MSE on validation split)
S2 → Unify Stage A/B/C into single train.py with --stage flag
     (or document reuse: python train_stage_a.py --config stage_b.yaml)
```

### Phase 3: Testing baseline (1-2 days)

```
tests/test_schema.py         — WindowSample field validation
tests/test_normalizer.py     — round-trip, edge cases, save/load
tests/test_collate.py        — fixed tensors, lists, Nones, mixed types
tests/test_hdf5_adapter.py   — window indexing, short episodes, chunk construction
tests/test_forward_train.py  — end-to-end with mock backbone
```

### Phase 4: Runtime/Eval (2-3 days)

```
runtime/policy.py      — PolicyWrapper (semantic_step + control_step + cache lifecycle)
runtime/postprocess.py — action denormalize + clamp
eval/offline_eval.py   — validation set MSE + discrete accuracy
eval/metrics.py        — success_rate, action_mse, phase_accuracy
```

### Phase 5: Cleanup (0.5 day)

```
configs/data/default.yaml    — template data config
configs/infer/default.yaml   — template infer config
docs/archive/                — move historical audit docs
README.md                    — project overview + quick start
```

### Projected Scores After Each Phase

| Phase | Score | Delta |
|-------|-------|-------|
| Current (v0.9.3) | 6.8 | — |
| Phase 1 (bug fixes) | 7.5 | +0.7 |
| Phase 2 (training) | 8.0 | +0.5 |
| Phase 3 (testing) | 8.5 | +0.5 |
| Phase 4 (runtime/eval) | 9.0 | +0.5 |
| Phase 5 (cleanup) | 9.2 | +0.2 |

---

## Part 10: 中文摘要

### 审计结论

v0.9.3 基础设施建设（数据管线、归一化器、HDF5 适配器、collate 函数、.gitignore、requirements.txt）成功将项目从"模型原型"升级为"数据就绪平台"。但新代码引入了若干问题。

### 模型层（8.5/10）

v0.9.0–v0.9.2 的所有修复均完整保留：
- 去噪公式 `x_t + (1-t)*v` 正确
- `proprio_dim` 已从 `action_dim` 解耦
- Grounder 三个调用点均传入 `attention_mask`
- `res_scale` 不参与权重衰减
- 推理 `medium_stride` 从 Hz 计算而非引用训练配置
- `ControlStepOutput` 返回当前步动作 `[B, A]`
- `refresh_counter` 替代了不可靠的 `id()` 比较

**模型层无新问题。**

### 数据层（v0.9.3 新建，发现 9 个问题）

**关键问题**：

1. **D1 字段名不匹配**：`schema.py` 定义 `refresh_pixel_values`，但 `forward_train()` 访问 `refresh_pixel_values_list`（带 `_list` 后缀）。collate 不重命名 key。当实现多帧刷新数据时，key 找不到会静默回退。

2. **D4 短 episode 窗口 Bug**：`hdf5_adapter.py:103` 使用 `max(1, T_ep - window + 1)`。当 T_ep < window 时仍创建窗口，产生尺寸不足的张量，导致 batch 维度不匹配。应改为 `max(0, ...)` 并跳过过短 episode。

3. **D5 继承不匹配**：`HDF5DatasetAdapter` 继承 `Dataset` 而非 `BaseDatasetAdapter`，返回 `dict` 而非 `WindowSample`。违反了专门为此设计的抽象契约。

### 基础设施缺口

| 维度 | 现状 | 影响 |
|------|------|------|
| 评估循环 | 不存在 | 无法衡量训练效果 |
| 推理封装 | 空壳 (`infer/`) | 无法对接 benchmark |
| 测试 | 空目录 | 无回归保障 |
| Stage B/C 脚本 | 不存在 | 阻塞专家网络训练 |
| 归一化统计计算 | 无脚本 | 阻塞真实数据训练 |
| 视觉通路 | 从未测试 | DummyVLADataset 和 HDF5Adapter 均不产生 pixel_values |

### 评分

综合评分 **6.8/10**。模型层 8.5 拉高，但测试 (1.0)、文档 (4.0)、完整度 (6.0) 拖低。与 v0.9.2 (7.3) 相比分数下降，原因是审计范围扩大（现纳入测试和文档维度），非代码质量倒退。

### 推荐优先级

1. **Phase 1**（1 天）：修复数据层 5 个 bug（D1/D4/D5/D6/D7）
2. **Phase 2**（2-3 天）：stats 计算脚本 + eval 循环 + 统一训练入口
3. **Phase 3**（1-2 天）：基础 pytest 测试
4. **Phase 4**（2-3 天）：runtime/policy.py + eval/offline_eval.py
5. **Phase 5**（0.5 天）：config 模板 + docs 归档 + README

完成全部 5 阶段后预计达 **9.2/10**，进入可运行实验阶段。
