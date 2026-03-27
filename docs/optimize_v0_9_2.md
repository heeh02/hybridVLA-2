# HybridVLA v2 — v0.9.2 Optimization Report

Based on `analysis_v0_9_2.md` (cross-audit of v0.9.1 fixes + GPT review findings). 5 code changes applied.

---

## Context

v0.9.2 is a refinement pass based on a cross-audit that:
1. Verified all 9 v0.9.1 fixes — **9/9 PASS**, no regressions
2. Cross-checked GPT's per-file audit — identified 3 new actionable items, corrected 1 mischaracterization, dismissed 1 false positive
3. Found 4 additional minor issues (N1–N4)

This version focuses on engineering quality and config robustness, not model architecture changes.

---

## Changes Applied in v0.9.2

### G1. Config Unknown Key Warning (R2 — P1)

**Problem**: `_dict_to_dataclass` silently skips YAML keys not matching any dataclass field. A typo like `leraning_rate` instead of `learning_rate` is ignored — the default value is used with no indication. This is a classic "debugging black hole" in ML training.

**Fix** (`config.py:338-344`):
```python
if k not in field_types:
    warnings.warn(
        f"Unknown config key '{k}' in {cls.__name__}, ignored",
        stacklevel=2,
    )
    continue
```

**Impact**: Any YAML config typo now produces a visible warning at load time. Zero performance overhead (runs once at startup).

**Files**: `config.py`

---

### G2. label_smoothing Configurable (N1/C1 — P3)

**Problem**: `DiscreteCELoss(label_smoothing=0.1)` was hardcoded in `HybridVLAv2.__init__`. The `DiscreteCELoss` class itself accepts the parameter, but there was no way to control it from YAML config. Different tasks/datasets may need different smoothing values (e.g., 0.0 for small datasets, 0.2 for noisy labels).

**Fix**:
- `config.py:126`: Added `label_smoothing: float = 0.1` to `HeadsConfig`
- `hybrid_vla_v2.py:174`: `DiscreteCELoss(label_smoothing=mcfg.heads.label_smoothing)`

**Backward-compatible**: Default is 0.1, matching previous behavior.

**Files**: `config.py`, `hybrid_vla_v2.py`

---

### G3. Unified action_range Config (N2/R7 — P2)

**Problem**: The action discretization range `[-1, 1]` was hardcoded in two separate places with no shared constant:
- `hybrid_vla_v2.py:165`: `torch.linspace(-1, 1, V)` for bin centers
- `discrete_heads.py:38`: `discretise_actions(actions, lo=-1.0, hi=1.0, V=512)`

If the action normalization range changes (e.g., `[0, 1]` for some embodiments), both must be updated simultaneously. The implicit coupling is a bug waiting to happen.

**Fix**:
- `config.py:127`: Added `action_range: Tuple[float, float] = (-1.0, 1.0)` to `HeadsConfig`
- `hybrid_vla_v2.py:166-167`: `lo, hi = mcfg.heads.action_range; torch.linspace(lo, hi, V)`
- `hybrid_vla_v2.py:435-437`: `discretise_actions(target_actions, lo=_lo, hi=_hi, V=...)`

`FASTDiscreteHead.discretise_actions()` and `undiscretise_actions()` still accept `lo`/`hi` as parameters (for flexibility), but the model now passes config values consistently.

**Files**: `config.py`, `hybrid_vla_v2.py`

---

### G4. AffordanceHead Docstring Fix (G7/R9 — P3)

**Problem**: Docstring claimed "Outputs a spatial affordance map (where to act) and a categorical affordance type (how to act)" but the implementation only has a categorical classification head.

**Fix** (`discrete_heads.py:61-65`):
```python
"""Predicts categorical affordance type from the affordance token.

Returns logits over `num_affordance_types` classes (e.g., grasp, push,
pull, place). Spatial affordance map is not yet implemented.
"""
```

**Files**: `discrete_heads.py`

---

### G5. Teacher-Forcing Documentation (N3)

**Problem**: `forward_train()` pushes `batch["prev_actions"][:, t]` (ground-truth) into action history, while `control_step()` pushes model-output actions. This is standard teacher-forcing but the divergence could confuse readers.

**Fix** (`hybrid_vla_v2.py:418`):
```python
# Update action history (teacher-forcing: uses GT prev_actions
# during training; control_step uses model output actions instead)
action_history_buf.push(batch["prev_actions"][:, t])
```

**Files**: `hybrid_vla_v2.py`

---

## Issues NOT Changed (With Justification)

### R1: Real Data Pipeline (P0)

Remains the single largest gap. `vla_hybrid_v2/data/` is empty, `DataConfig` fields are unconsumed. This is a full-module design task (format parsing, tokenization, image preprocessing, multi-camera handling, episode windowing) that requires its own design iteration, not a patch.

### R3: Stage B/C Training Scripts (P1)

Only `train_stage_a.py` exists. Stage B (add expert with detached cond_prefix) and Stage C (full fine-tune with RTC/FASTER) need their own scripts or a unified script with stage-gated logic. Not a v0.9.2 scope item.

### R4: Evaluation Loop / Metrics (P1)

No evaluation infrastructure exists. Cannot measure training progress. Required before serious experiments.

### R5: RTC/FASTER/step_weights (P2)

Config-only, no implementation. Stage C feature.

### R6: World Model Not Connected to Training (P2)

`imagination_engine` and `world_model_loss_fn` are initialized but never called in `forward_train()`. By design — world model training integration is a separate design phase.

### G8: torch.load weights_only Compatibility (P3)

`checkpointing.py` uses `weights_only=True` (requires PyTorch ≥ 2.1). Solution: add version constraint to `requirements.txt` or add a fallback. Deferred to infrastructure cleanup.

### C3: ContrastiveTemporalLoss O(N²) (P3)

N = B*(T-1) = 1472 at default config. Matrix is ~8.7 MB in fp32 — acceptable. Would need attention if batch size grows significantly.

---

## Summary of All v0.9.2 Changes

| # | File | Change | Addresses |
|---|------|--------|-----------|
| G1 | `config.py` | `warnings.warn` for unknown config keys | R2 (P1) |
| G2 | `config.py`, `hybrid_vla_v2.py` | `label_smoothing` from `HeadsConfig` | N1/C1 (P3) |
| G3 | `config.py`, `hybrid_vla_v2.py` | `action_range` from `HeadsConfig` | N2/R7 (P2) |
| G4 | `discrete_heads.py` | AffordanceHead docstring corrected | G7/R9 (P3) |
| G5 | `hybrid_vla_v2.py` | Teacher-forcing comment | N3 |

**Files modified**: 3 files, ~+15 / -5 lines net.

---

## Updated Scoring (v0.9.2)

| Dimension | v0.9.2 audit (before) | v0.9.2 (after fixes) | Delta | Justification |
|-----------|----------------------|---------------------|-------|---------------|
| Design coherence | 8.5 | **8.5** | — | action_range + label_smoothing configurable, minor improvement |
| Correctness | 9.5 | **9.5** | — | No new correctness issues |
| Completeness | 5.0 | **5.0** | — | Data pipeline still missing |
| Training stability | 9.0 | **9.0** | — | No changes |
| Scalability | 7.0 | **7.0** | — | No changes |
| Performance | 6.0 | **6.0** | — | No changes |
| Production readiness | 6.0 | **6.5** | +0.5 | Config typo detection prevents silent failures |
| **Weighted avg** | **7.3** | **7.4** | +0.1 | |

**Note**: v0.9.2 is a refinement pass — the +0.1 improvement is expected. The big gains are behind us (v0.9.1 fixed the critical correctness/design issues). Further score improvement requires infrastructure work (data pipeline, eval loop, Stage B/C scripts).

---

## Cumulative Change Log (v0.9 → v0.9.2)

| Version | Changes | Key Impact |
|---------|---------|-----------|
| v0.9 | 5 changes (res_scale, remove double-LN, chunk cache, path validation, image guard) | Performance +2.0, Stability +0.5 |
| v0.9.1 | 9 changes (denoising formula, proprio_dim, grounder mask, res_scale decay, refresh counter, API, infer config, batch validation, detach cleanup) | Correctness +1.0, Production +2.0 |
| v0.9.2 | 5 changes (config warning, label_smoothing, action_range, docstring, comment) | Production +0.5 |

**Total since v0.9 start**: 19 changes across 8 files. Score: 7.0 → 7.4 (+0.4).

---

## Remaining Priority Queue

```
P0: Real data pipeline (vla_hybrid_v2/data/)   — blocks all real training
P1: Stage B training script                    — enables expert training
P1: Evaluation loop + metrics                  — measures training progress
P2: step_weights / RTC / FASTER                — enables Stage C
P2: Multi-camera forward path                  — enables multi-view robots
P2: World model training integration           — connects imagination engine
P3: torch.load compatibility guard             — requirements.txt
```

---

## 中文摘要

### v0.9.2 修复内容

共 5 项代码修改，聚焦工程质量和配置鲁棒性。

1. **配置未知键警告（G1, P1）**：`_dict_to_dataclass` 遇到 YAML 中不匹配 dataclass 字段的键时，发出 `warnings.warn`。一行改动防止拼写错误被静默吞掉（如 `leraning_rate` → 使用默认值而无任何提示）。

2. **label_smoothing 可配置（G2, P3）**：`HeadsConfig` 新增 `label_smoothing: float = 0.1`，模型初始化从配置读取，不再硬编码。默认值不变，向后兼容。

3. **action_range 统一配置（G3, P2）**：`HeadsConfig` 新增 `action_range: Tuple[float, float] = (-1.0, 1.0)`。`_fast_bin_centers`（bin 中心）和 `discretise_actions`（离散化）两处均从配置读取，消除隐式耦合。未来切换动作归一化范围只需改一处配置。

4. **AffordanceHead 文档修正（G4, P3）**：docstring 声称输出"空间 affordance 地图 + 分类"，实际仅有分类输出。修正为准确描述。

5. **Teacher-forcing 注释（G5）**：训练循环中 `action_history` 推入 GT 动作（teacher-forcing），推理时推入模型输出。添加注释说明此设计差异。

### 审计结论

- v0.9.1 的 9 项修复**全部正确**，无遗漏或回归
- GPT 审计报告**质量良好**，3 项新发现有价值，1 项表述不精确，1 项误判（temporal core ≠ cond_prefix 不是 bug）
- 校准评分 **7.4/10**（v0.9.1 的 7.5 略偏乐观 → 审计校准 7.3 → v0.9.2 修复后 7.4）
- **最大阻断项仍然是真实数据管线**，所有审计一致

### 下一步

模型层面的代码质量已接近天花板（correctness 9.5, stability 9.0, design 8.5）。进一步提升需要基础设施建设：
- 真实数据 loader → completeness +1.5
- 评估循环 → production readiness +1.0
- Stage B 脚本 → completeness +0.5

完成这三项后预计评分 **8.8/10**，进入"可运行实验平台"阶段。
