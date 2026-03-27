# HybridVLA v2 — v0.10.3 Analysis (Claude x GPT Deep Cross-Audit)

> **Method**: v0.10.2 fix verification + GPT 6.6 score analysis cross-verification + independent deep code audit + detailed fix guidance
>
> **Date**: 2026-03-27

---

## Part 1: v0.10.2 Fix Verification

All 5 fixes from `optimize_v0_10_2.md` verified.

| Fix | Status | Evidence |
|-----|--------|----------|
| I1: `num_affordance_types` configurable | **PASS** | `config.py:125` — field in HeadsConfig; `hybrid_vla_v2.py:137` — passed to AffordanceHead; `dummy.py:39` — reads from config |
| I2: `_to_device` outside loop | **PASS** | `train_stage_a.py` — function defined before loop body |
| I3: Smoke test divergence documented | **PASS** | `train_smoke_test.py` — docstring explains intentional isolation |
| I4: `step_weights` shape validated | **PASS** | `hybrid_vla_v2.py:325-331` — asserts `(B, H)` shape |
| I5: Redundant `.to(device)` removed | **PASS** | `hybrid_vla_v2.py:441` — `action_history_buf.get()` without `.to(device)` |

Model layer: all prior fixes intact. No regressions.

---

## Part 2: GPT Score Analysis Cross-Verification

GPT gave **6.6/10** (vs our prior 7.3). Below is a point-by-point verdict on GPT's critique.

### 2.1 "正确性 9.5 偏高" — PARTIALLY AGREE

GPT argues three reasons to lower correctness:

| GPT Reason | My Verdict | Explanation |
|-----------|-----------|-------------|
| (a) 语言条件真实缺失 | **Completeness issue, not correctness** | Code correctly implements placeholder fallback; it does what the docstring says. The gap is that processor isn't connected, which is a completeness/design gap. |
| (b) 视觉输入真实缺失 | **Completeness issue, not correctness** | Same reasoning — the adapter is designed for text-only Stage A. |
| (c) 只监督最后一个 chunk | **VALID design concern** | Even after P0-3 chunk padding fix, `target_actions = batch["actions"][:, -1]` means the expert and FAST head only get supervision from the last window position. All 24 temporal steps run, but only the last produces a loss signal. This is a **training efficiency/quality concern**, not a code bug. |

**My revised position**: Correctness remains **9.5** (code does what it intends), but I add a new design issue **D1** (single-step supervision density) to the gap registry with severity P1.

### 2.2 "代码质量 8.5 偏高" — PARTIALLY AGREE

| GPT Reason | My Verdict |
|-----------|-----------|
| WindowSample 软约束 | Valid — schema exists but not enforced. P2. |
| collate 规则脆弱 | Valid — list transpose assumes uniform R across batch. P2. |
| 仓库 hygiene 差 | `.gitignore` now exists, but `tests/` empty, no `pyproject.toml`. P3. |
| tests 完全空白 | Valid — but this is scored under "Testing" dimension separately. |

**My position**: Code quality stays at **8.5** for the code that exists (model layer is genuinely clean). The repo-level issues are real but don't reduce code quality of the written modules.

### 2.3 "数据通路 4.9/10" — AGREE

GPT's data path scoring closely matches our prior assessment. The breakdown in `analysis_v10_extra.md` is accurate:

| Sub-dimension | GPT Score | My Score | Notes |
|--------------|-----------|----------|-------|
| Action/Proprio path | 9.0 | 9.0 | Fully correct after P0-3 fix |
| Text path | 4.0 | 4.0 | Architecture ready but processor disconnected |
| Vision path | 1.0 | 1.0 | Completely missing |
| Normalization | 8.5 | 8.5 | compute_stats + path decoupling + range warning |

### 2.4 "训练方法 6.5/10" — AGREE

GPT correctly identifies the dichotomy: **design is 9.0, implementation is 3.0**. Three-stage gating, EMA ramp, five-way loss system are well-designed; but only Stage A has a script, no eval loop exists.

### 2.5 GPT's New Critical Finding: Single-Step Supervision

**This is the most important new insight from the GPT analysis.**

GPT points out (quoted): *"temporal core 全窗口 rollout 的收益被弱化 — 更像'窗口上下文辅助最后一步预测'，而不是序列级 chunk 学习"*

**Code evidence** (`hybrid_vla_v2.py:466-498`):

```
Temporal loop: for t in range(T=24):    ← all 24 steps processed
    ...temporal_core(...)

Loss computation:
    target_actions = batch["actions"][:, -1]           ← only t=23
    fast_logits = self.fast_head(fused_states[:, -1])  ← only t=23
    flow matching uses target_actions                   ← only t=23
    phase_labels[:, -1]                                ← only t=23
    affordance_labels[:, -1]                           ← only t=23
```

**All five losses operate on the LAST timestep only.** The temporal core processes 24 steps, but the gradient signal only flows back through the last step's contribution. Earlier steps influence the final state through recurrence but receive no direct supervision.

**Impact analysis**:
- Temporal core learns to predict a good final state, but intermediate states are not directly supervised
- This is analogous to training an LSTM but only computing loss at the final token
- Phase labels at intermediate steps are wasted
- The grounder's output at refresh points 0, 6, 12, 18 only gets supervision through the recurrent pathway from t=18/23 back to t=0

**This is NOT a bug but a significant training quality limitation.** Severity: **P1-Design**.

### 2.6 Score Reconciliation

| Dimension | Our v0.10.2 | GPT 6.6 Analysis | Reconciled |
|-----------|-------------|-------------------|-----------|
| Architecture | 8.5 | 8.4 | **8.5** |
| Correctness | 9.5 | 7.6 | **9.5** (code correctness; design gap tracked separately) |
| Completeness | 6.5 | ~5.0 (data path) | **6.0** (lowered: accepting GPT's stricter "training-ready" standard) |
| Training stability | 9.0 | included in "模型实现" | **9.0** |
| Code quality | 8.5 | 7.1 | **8.0** (lowered: accepting collate fragility concern) |
| Training method | 6.5 | 6.3 | **6.5** |
| Testing | 1.5 | included in quality | **1.5** |

**Final reconciled score: 6.8/10**

GPT's 6.6 and our 6.8 are within 0.2 — strong agreement. The difference is GPT applies a "training-ready" lens (harsher on gaps that block real training), while we apply a "code quality" lens (credits clean implementation).

---

## Part 3: Full Issue Registry (Post v0.10.2)

### P0 — Blocks Meaningful Training

| ID | Issue | File:Line | Impact | Fix Effort |
|----|-------|-----------|--------|-----------|
| **P0-A** | Processor not connected — all-zero `input_ids` | `train_stage_a.py:176` | Backbone produces identical hidden states for all instructions; grounder cannot learn instruction-conditioned grounding | **~5 lines** |
| **P0-B** | No image reading in HDF5 adapter | `hdf5_adapter.py:139-147` | Not a VLA — only language+state+action | **~150 lines** |

### P1 — Blocks Full Training Quality / Multi-Stage

| ID | Issue | File:Line | Impact | Fix Effort |
|----|-------|-----------|--------|-----------|
| **P1-C** | Single-step supervision — all losses at t=-1 only | `hybrid_vla_v2.py:466-498` | Temporal core intermediate states unsupervised; gradient signal sparse | ~50 lines |
| **P1-D** | No Stage B/C training scripts | `scripts/` | Expert never trains; three-stage pipeline only 1/3 implemented | ~200 lines |
| **P1-E** | No eval loop | `config.py:248` unused | No validation, no checkpoint selection, no metrics | ~150 lines |
| **P1-F** | Refresh frame data never produced | `hdf5_adapter.py` | Multi-refresh semantic pathway untested | Included in P0-B |

### P2 — Quality / Robustness

| ID | Issue | File:Line | Impact |
|----|-------|-----------|--------|
| P2-G | WindowSample not enforced as return type | `base_adapter.py:38` | Schema is documentation-only |
| P2-H | `split` parameter unused | `hdf5_adapter.py:55` | No train/val separation |
| P2-I | Collate list-transpose fragile for mixed-R batches | `collate.py:35-43` | Assumes all samples have same R |
| P2-J | Empty `infer/` — no PolicyWrapper | `infer/__init__.py` | No deployment path |

### P3 — Cosmetic

| ID | Issue | File:Line | Impact |
|----|-------|-----------|--------|
| P3-K | No `pyproject.toml`, empty `tests/` | Root | Package not installable, no CI |
| P3-L | `num_workers=2` in DataLoader | `train_stage_a.py:185` | Potentially IO-bound for real data |

---

## Part 4: Detailed Fix Guidance

### Fix 1: Connect Processor (P0-A) — **Highest ROI, ~5 lines**

**Why this is #1**: With all-zero `input_ids`, the backbone produces **identical** hidden states regardless of the language instruction. This means:
- Grounder cannot learn to differentiate "pick up the red cup" from "push the blue box"
- Stage A training is essentially wasted for language conditioning
- The ~15-22 hours of GPU time produces a model with no instruction understanding

**File**: `scripts/train_stage_a.py`

**Current code** (line 176):
```python
dataset, collate_fn = build_dataset(cfg, split="train")
```

**Fix** — insert before line 176:
```python
# Connect Qwen2-VL processor for real tokenization
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(cfg.model.backbone.name)
dataset, collate_fn = build_dataset(cfg, split="train", processor=processor)
```

**Verification**: After fix, check `dataset[0]["input_ids"]` — should contain non-zero token IDs corresponding to the language instruction. `attention_mask` should have 1s for real tokens and 0s for padding.

**Note**: `build_dataset()` already accepts `processor=` (line 29 of `data/__init__.py`) and passes it through to `HDF5DatasetAdapter` (line 66). The plumbing exists — only the call site needs the change.

---

### Fix 2: HDF5 Image Reading (P0-B) — **Critical path, ~150 lines**

**Why this matters**: Without vision, the model is "LA" not "VLA". This is the single largest functional gap.

**File**: `vla_hybrid_v2/data/hdf5_adapter.py`

**Implementation plan**:

**Step 1**: Read image from HDF5 in `__getitem__()` (after line 153):
```python
# Read image for backbone processing
# Use the frame at the START of the window as the observation
img_key = self.dcfg.image_key  # e.g., "agentview_rgb"
if img_key and img_key in data.get("images", {}):
    raw_image = data["images"][img_key][start]  # [H, W, C] uint8
    pil_image = Image.fromarray(raw_image)
else:
    pil_image = None
```

**Step 2**: Process image+text jointly via processor (replace lines 174-185):
```python
if self.processor is not None and pil_image is not None:
    tok = self.processor(
        text=lang, images=pil_image,
        return_tensors="pt", padding="max_length",
        truncation=True, max_length=256,
    )
    input_ids = tok["input_ids"].squeeze(0)
    attention_mask = tok["attention_mask"].squeeze(0)
    pixel_values = tok["pixel_values"].squeeze(0)       # NEW
    image_grid_thw = tok["image_grid_thw"].squeeze(0)   # NEW
elif self.processor is not None:
    # Text-only (no image in HDF5)
    tok = self.processor(text=lang, return_tensors="pt", ...)
    input_ids, attention_mask = tok["input_ids"].squeeze(0), tok["attention_mask"].squeeze(0)
    pixel_values, image_grid_thw = None, None
else:
    # Placeholder
    input_ids = torch.zeros(256, dtype=torch.long)
    attention_mask = torch.ones(256, dtype=torch.long)
    pixel_values, image_grid_thw = None, None
```

**Step 3**: Add vision fields to return dict (extend line 187):
```python
sample = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "actions": action_chunks,
    "proprio": norm_proprio,
    "prev_actions": prev_actions,
    "embodiment_id": ...,
}
if pixel_values is not None:
    sample["pixel_values"] = pixel_values
    sample["image_grid_thw"] = image_grid_thw
return sample
```

**Step 4**: Construct refresh frames (if multiple observations within window):
```python
# For each refresh point, read the corresponding frame
refresh_stride = self.cfg.train.semantic_refresh_stride  # e.g., 6
refresh_steps = list(range(0, T, refresh_stride))
R = len(refresh_steps)

if self.processor is not None and R > 1 and pil_images_available:
    refresh_input_ids_list = []
    refresh_attention_mask_list = []
    refresh_pv_list = []
    refresh_thw_list = []
    for r_step in refresh_steps:
        frame_idx = start + r_step
        img = Image.fromarray(data["images"][img_key][frame_idx])
        tok = self.processor(text=lang, images=img, ...)
        refresh_input_ids_list.append(tok["input_ids"].squeeze(0))
        refresh_attention_mask_list.append(tok["attention_mask"].squeeze(0))
        refresh_pv_list.append(tok["pixel_values"].squeeze(0))
        refresh_thw_list.append(tok["image_grid_thw"].squeeze(0))

    sample["refresh_input_ids"] = torch.stack(refresh_input_ids_list)
    sample["refresh_attention_mask"] = torch.stack(refresh_attention_mask_list)
    sample["refresh_pixel_values_list"] = refresh_pv_list
    sample["refresh_image_grid_thw_list"] = refresh_thw_list
```

**Collate compatibility**: `vla_collate_fn` already handles:
- Tensor stacking (for `refresh_input_ids`, `refresh_attention_mask`)
- List-of-tensors transpose (for `refresh_pixel_values_list`, `refresh_image_grid_thw_list`)
- Pass-through for None values

**_to_device compatibility**: `train_stage_a.py` already has recursive `_to_device` (F3 fix) that handles list-of-tensors.

**forward_train compatibility**: `hybrid_vla_v2.py:362-388` already handles refresh fields via `batch.get("refresh_pixel_values_list", [None]*R)`.

---

### Fix 3: Multi-Step Supervision (P1-C) — **~50 lines**

**Current design**: Only t=-1 is supervised.

**Recommended approach**: Supervise at ALL timesteps where the chunk is fully real (which is all T after the P0-3 fix). Compute loss at each t, average.

**File**: `vla_hybrid_v2/models/hybrid_vla_v2.py`

**Current** (line 466):
```python
target_actions = batch["actions"][:, -1]  # [B, H, A]
```

**Proposed change**:
```python
# Multi-step supervision: compute loss at every timestep and average
T = batch["actions"].shape[1]
all_losses_fast = []
all_losses_phase = []
all_losses_aff = []

for t_sup in range(T):
    target_t = batch["actions"][:, t_sup]  # [B, H, A]

    # FAST discrete
    if self.fast_head is not None:
        fast_logits_t = self.fast_head(fused_states[:, t_sup])
        fast_targets_t = FASTDiscreteHead.discretise_actions(target_t, lo=_lo, hi=_hi, V=...)
        all_losses_fast.append(self.discrete_loss(fast_logits_t, fast_targets_t))

    # Phase
    if self.phase_head is not None and "phase_labels" in batch:
        grounder_r = grounder_outputs[refresh_map[t_sup]]
        phase_logits_t = self.phase_head(grounder_r.phase_token)
        all_losses_phase.append(self.phase_loss(phase_logits_t, batch["phase_labels"][:, t_sup]))

losses["loss_fast"] = torch.stack(all_losses_fast).mean() * weights.get("fast_discrete", 1.0)
# ... similar for phase, affordance
```

**Expert loss**: Keep at t=-1 only (Stage B/C). The expert sees the final cond_prefix which aggregates the full window context. Multi-step expert loss would require building cond_prefix at every step, which is expensive.

**Trade-off**: This increases Stage A compute by ~T× for FAST/phase/affordance losses, but these losses are lightweight relative to the temporal core forward pass. Net training time increase: ~10-20%.

**Alternative (simpler)**: Supervise at 4 evenly-spaced timesteps (t=0, 8, 16, 23) instead of all 24. This gives 4× supervision density at ~4× loss cost.

---

### Fix 4: Unified Training Script (P1-D) — **~200 lines**

**File**: `scripts/train_unified.py` (new)

**Design**: Copy `train_stage_a.py` as the base, add:

1. **Stage argument**: `--stage a|b|c` (or read from YAML)
2. **Processor creation**: If stage != "a", instantiate processor and pass to `build_dataset()`
3. **Expert unfreezing**: If stage != "a", unfreeze `action_expert` parameters
4. **Stage-gated loss**: Already handled by `if stage != "a":` in `forward_train()`
5. **Cross-stage resume**: Already handled by `cfg.train.resume_from` + `load_checkpoint(strict=False)`
6. **EMA activation**: If `cfg.model.ema.initial_decay > 0`, enable EMA

Most of the code (optimizer setup, FSDP, training loop, checkpointing) can be reused verbatim from `train_stage_a.py`.

---

### Fix 5: Eval Loop (P1-E) — **~150 lines**

**File**: `scripts/train_unified.py` (add to training loop) or `eval/offline_eval.py` (standalone)

**Minimum viable eval**:
```python
@torch.no_grad()
def evaluate(model, val_loader, device, cfg):
    model.eval()
    total_mse, total_discrete_acc, count = 0, 0, 0
    for batch in val_loader:
        batch = {k: _to_device(v) for k, v in batch.items()}
        losses = model.forward_train(batch)
        # Action MSE
        pred_actions = ...  # from FAST discrete softmax expected value
        target = batch["actions"][:, -1]
        total_mse += F.mse_loss(pred_actions, target).item()
        count += 1
    model.train()
    return {"action_mse": total_mse / count}
```

**Integrate into training loop** (in `train_unified.py`):
```python
if global_step % cfg.train.eval_interval == 0 and val_loader is not None:
    metrics = evaluate(model, val_loader, device, cfg)
    logger.info("Eval step %d: %s", global_step, metrics)
```

---

## Part 5: Recommended Execution Order

```
Priority  Task                              Effort    Impact
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 #1  P0-A  Connect Processor                 5 lines   ★★★★★
            → Stage A becomes useful

 #2  P0-B  HDF5 Image Reading               150 lines  ★★★★★
            → System becomes VLA

 #3  P1-C  Multi-step supervision            50 lines  ★★★★☆
            → Training quality improves significantly

 #4  P1-D  Unified training script           200 lines  ★★★★☆
            → Stage B/C become runnable

 #5  P1-E  Eval loop                         150 lines  ★★★☆☆
            → Can measure and compare checkpoints
```

**Total**: ~555 lines of new code to reach "complete VLA training system."

**Projected scores**:

| After | Score | Delta |
|-------|-------|-------|
| Current (v0.10.2+) | 6.8 | — |
| #1 Processor | 7.2 | +0.4 |
| #2 Vision | 8.0 | +0.8 |
| #3 Multi-step loss | 8.3 | +0.3 |
| #4 Unified script | 8.7 | +0.4 |
| #5 Eval loop | 9.0 | +0.3 |

---

## Part 6: 中文摘要

### v0.10.2 修复验证

全部 5 项修复通过验证（I1 affordance 可配置 ✓, I2 _to_device 提出 ✓, I3 smoke test 文档 ✓, I4 step_weights 校验 ✓, I5 冗余 .to 移除 ✓）。无回退。

### GPT 6.6 分分析交叉验证

GPT 6.6 与我们 6.8 在 0.2 分内——**高度一致**。

**GPT 对的地方**：
- "数据通路 5.0/10"判断准确，action/proprio 9.0 但视觉 1.0 拖低均分
- "训练方法 6.5/10"准确——设计 9.0 但实现只有 Stage A
- **"只监督最后一个时间步"是本轮最重要的新发现**——即使 chunk 填充修复后，所有 5 路损失仍仅作用于 t=T-1，temporal core 24 步运算只在最后一步产生梯度信号

**GPT 偏严的地方**：
- "正确性 7.6" 过低——语言/视觉缺失属于完备性问题，不是正确性问题。代码在其声明的范围内是正确的。
- "代码质量 7.1" 过低——模型层代码质量确实接近 8.5，repo 层面的问题（tests 空、无 pyproject.toml）是独立维度。

### 当前最关键的 5 个修复

1. **连接 Processor（P0-A, 5 行）**——ROI 最高。修复后 Stage A 从"无语言条件"升级为"有意义的预训练"
2. **HDF5 图像读取（P0-B, ~150 行）**——修复后系统从 "LA" 升级为 "VLA"
3. **多步监督（P1-C, ~50 行）**——修复后训练信号密度从 1/24 提升到 24/24
4. **统一训练脚本（P1-D, ~200 行）**——修复后 Stage B/C 可运行
5. **评估循环（P1-E, ~150 行）**——修复后可量化模型性能

总计约 **555 行新代码**，即可达到完整 VLA 训练系统。预计评分从 6.8 提升至 **9.0/10**。

### 最终定位

> **代码架构已成熟（8.5/10），正确性无 bug（9.5/10）。瓶颈完全在"真实训练路径的完备性"上。连接 Processor（5 行代码）是投入产出比最高的一步——它让 Stage A 训练从"演示"变成"有价值的预训练"。**
