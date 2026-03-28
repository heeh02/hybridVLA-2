# HybridVLA v2 — v0.10.5 Analysis

> **标准**: "真实训练第一天会不会崩" + "训练结果是否有意义"
>
> **Date**: 2026-03-27

---

## Part 1: v0.10.4 P0 Fix Verification

| Fix | Status | 验证证据 |
|-----|--------|---------|
| P0-3: pixel_values resize | **PASS** | `hdf5_adapter.py:159-162` — `_TARGET=(448,448)`, resize + RGB convert。`collate.py` 有 `_safe_stack_vision()` 安全网。N_patches 确定性一致。|
| P0-1a: 显式 stage 门控 | **PASS** | `train_unified.py:87-150` — `configure_trainable_modules()` 先冻结全部再按 stage 解冻。Stage A expert 冻结 ✓; Stage B expert 解冻 ✓; Stage C backbone text 16-27 解冻 ✓。调用顺序: configure(292) → to(device)(300) → FSDP(303) → load_checkpoint(360)。`requires_grad` 在 `load_state_dict` 中不被覆盖，顺序正确。|
| P0-1b: sanity check | **PASS** | `train_unified.py:153-224` — 14 模块逐个统计。Stage A 断言 expert_trainable==0; Stage B/C 断言 expert_trainable==expert_total; 所有 stage 断言 LoRA 可训练。|
| P0-4: MultiCamera=False | **PASS** | `config.py:54` — `enable: bool = False`。|

**Smoke test 结果** (optimize_v0_10_4.md): Stage A/B/C 三阶段全部通过，loss 收敛，freeze/unfreeze 行为正确，Stage B/C 有 loss_fm。

**"第一天崩溃测试"**: 4 项 P0 修复后，以下场景不再崩溃：
- ✅ 不同分辨率图像 batch → resize 统一
- ✅ Stage B checkpoint 加载后 expert 解冻 → 显式 configure
- ✅ MultiCamera 误开 → 默认关闭

---

## Part 2: 新发现

### V1: Validation Split 被忽略 — P1

`train_unified.py:392` 调用 `build_dataset(cfg, split="val", processor=processor)`，但 `hdf5_adapter.py` 的 `_build_index()` **不使用 split 参数**（只存储不过滤）。结果：val_loader 加载的是**全量训练数据**，validation metrics 是在训练集上算的。

**不会崩**，但 eval 结果无意义——过拟合指标无法反映真实泛化能力。

**修复**: 要么在 `_build_index()` 中按 episode 比例切分（如 90/10），要么要求数据目录中有独立的 val split 文件夹。

---

### V2: Refresh Frame Smoke Test 缺失 — P2

`train_smoke_test.py` 的 DummyVLADataset 不产生 `refresh_input_ids` / `refresh_pixel_values_list`。这意味着 refresh 路径（`forward_train:362-374` 中 `if "refresh_input_ids" in batch` 分支）从未在 smoke test 中执行过。

代码阅读上看逻辑正确，但缺乏运行时验证。

---

### V3: 单步监督设计限制仍在 — P1-Design

`hybrid_vla_v2.py:466`:
```python
target_actions = batch["actions"][:, -1]  # 只用 t=T-1
```

所有 5 路损失（FAST/Phase/Affordance/FM/Consistency）仅作用于最后一个时间步。temporal core 处理 24 步但梯度只从 t=23 回传。这不是 bug 但是训练效率限制。

---

### V4: Per-Module LR 未实现 — P1

`train_unified.py:307-327` 只按 decay/no_decay 分组，不按模块分组。Stage B 的设计意图（expert 低 LR / backbone 高 LR）无法实现。所有可训练参数共享 `cfg.train.learning_rate`。

---

### V5: Per-Module Gradient Norm 未记录 — P1

`train_unified.py:438` 只记录全局 `grad_norm`。Stage B 的 `cond_prefix.detach()` 梯度隔离效果无法验证——没有 backbone/grounder 与 expert 的分 module gnorm 对比。

---

## Part 3: "训练结果是否有意义" 评估

### Stage A（当前可启动）

| 条件 | 状态 | 意义 |
|------|------|------|
| Processor 连接 | ✅ (train_unified.py:368-372) | 骨干接收真实语言 token |
| 图像读取 | ✅ (hdf5_adapter.py 图像读取+resize) | 骨干接收真实视觉输入 |
| Action chunk 完整 | ✅ (T+H-1 扩展读取) | 监督目标无填充退化 |
| Stage 门控 | ✅ (expert 冻结, LoRA+grounder+core 训练) | 正确的参数分区 |
| 归一化 | ✅ (compute_stats → normalizer) | action/proprio 标准化 |
| Refresh 帧 | ✅ (代码存在) / ⚠️ (未被 smoke test 覆盖) | 多视角语义刷新 |
| Eval | ⚠️ (val split 未过滤) | 指标在训练集上，不反映泛化 |

**结论**: Stage A 训练**可以产生有意义的结果**——LoRA 学习语言-视觉对齐，grounder 学习指令条件化物体接地，temporal core 学习动力学。但 eval 指标需要手动准备 val split 数据。

### Stage B（需 Stage A 完成后启动）

| 条件 | 状态 | 风险 |
|------|------|------|
| Expert 解冻 | ✅ sanity check 保证 | 低 |
| cond_prefix.detach() | ✅ 代码存在 | 梯度隔离效果不可验证（P1-4） |
| Per-module LR | ❌ | Expert/backbone 共享 LR，可能次优 |
| 回归测试 | ❌ (tests/ 空) | Stage B 特有逻辑无自动化验证 |

**结论**: Stage B 可以运行但**结果可能次优**（LR 不分组）且**正确性无自动化保证**（无回归测试）。

### Stage C（需 Stage B 完成后启动）

同 Stage B 风险，额外：backbone text 16-27 解冻后的训练稳定性未验证。

---

## Part 4: 评分

| # | 维度 | v0.10.4 前 | v0.10.4 后 | 理由 |
|---|------|:---------:|:---------:|------|
| 1 | 设计一致性 | 7.5 | **8.0** | MultiCamera 声明修正；stage 门控显式化 |
| 2 | 正确性 | 8.0 | **9.0** | pixel_values 不再崩；stage freeze/unfreeze 有断言 |
| 3 | 完备性 | 6.0 | **7.0** | train_unified + eval loop + processor 连接 + 视觉读取 |
| 4 | 训练稳定性 | 7.5 | **8.0** | sanity check 防止静默冻结；但无 per-module gnorm |
| 5 | 可扩展性 | 6.5 | **6.5** | 未变化 |
| 6 | 性能设计 | 6.0 | **6.0** | 未变化 |
| 7 | 生产就绪度 | 5.0 | **6.0** | pixel_values 不崩，但无 runtime wrapper |
| 8 | 代码质量 | 7.5 | **8.0** | 显式 stage 门控；消除隐式依赖 |
| 9 | 文档 | 4.5 | **4.5** | 未变化 |
| 10 | 测试 | 1.5 | **2.0** | smoke test 用新 configure/sanity；但 tests/ 仍空 |
| | **加权均分** | **6.2** | **7.2** | |

**与 optimize_v0_10_4.md 声称的 7.5 对比**: 我给 7.2，略低 0.3 分。原因：
- Val split 未过滤降低 eval 有效性（完备性打折）
- 无 per-module LR 和 gnorm 降低训练质量置信度
- tests/ 仍空（测试维度受限）

---

## Part 5: 按优先级排序的后续修复

### 训练前建议修复（提升训练结果质量）

| # | ID | 项目 | 行数 | 影响 |
|---|-----|------|------|------|
| 1 | V1 | **Val split 过滤** — `_build_index()` 按 episode 比例切分 | ~20行 | eval 指标从无意义变为有意义 |
| 2 | V5 | **Per-module gradient norm** — 分 backbone/grounder/core/expert 记录 | ~30行 | 可验证 Stage B 梯度隔离 |

### 可与 Stage A 并行开发

| # | ID | 项目 | 行数 |
|---|-----|------|------|
| 3 | V4 | Per-module optimizer LR 分组 | ~40行 |
| 4 | V3 | 多步监督（可选，改变训练 dynamics） | ~50行 |
| 5 | — | Stage B/C 回归测试 | ~120行 |
| 6 | — | Tri-rate ablation 开关 | ~100行 |
| 7 | — | Inference runtime wrapper | ~200行 |
| 8 | V2 | Refresh frame smoke test | ~30行 |

---

## Part 6: 中文摘要

### v0.10.4 修复验证

全部 4 项 P0 修复通过验证。**真实数据训练不再有"第一天崩溃"风险。** pixel_values resize 保证 N_patches 一致；stage 门控显式化并有断言保护；MultiCamera 默认关闭。

### 新发现

5 项新问题（均为 P1-P2，不阻塞训练）：
1. **V1 (P1)**: val split 被忽略——eval 在训练集上算，指标无意义
2. **V2 (P2)**: refresh frame 路径未被 smoke test 覆盖
3. **V3 (P1)**: 单步监督（所有损失只在 t=T-1）——训练效率限制
4. **V4 (P1)**: 无 per-module LR 分组——Stage B expert/backbone 共享 LR
5. **V5 (P1)**: 无 per-module gradient norm——梯度隔离效果不可验证

### 评分

**7.2/10**（v0.10.4 前 6.2，修复后提升 1.0 分）。

与 optimize_v0_10_4.md 声称的 7.5 相比低 0.3 分——我在 eval 有效性（val split）和训练质量置信度（per-module LR/gnorm 缺失）上更保守。

### 训练就绪度

| 场景 | 可否启动 | 条件 |
|------|:-------:|------|
| Stage A | **✅ 可以** | 准备好 HDF5 数据 + compute_stats |
| Stage A eval | **⚠️ 有条件** | 需准备独立 val split 或手动切分 |
| Stage B | **⚠️ 可以但次优** | 缺 per-module LR，建议先修 V4 |
| Stage C | **⚠️ 同上** | 同上 |

**建议**: 先修 V1（val split, ~20行）和 V5（per-module gnorm, ~30行），然后启动 Stage A。总额外工作 ~50 行，半天内完成。
