# HybridVLA v2 — analysis_v0_10_5.md 审核报告

> **日期**: 2026-03-27
> **目标**: 逐项审核 analysis_v0_10_5.md 的 5 个新发现 (V1-V5), 判断每项的准确性, 并补充遗漏问题

---

## Part 1: V1-V5 逐项审核

### V1: Validation Split 被忽略 — **TRUE, P1 确认**

**analysis_v0_10_5 声称**: `_build_index()` 不使用 split 参数, val_loader 加载全量训练数据。

**代码验证**:

`hdf5_adapter.py:56` 接收 `split` → `base_adapter.py:35` 存储为 `self.split` → 但 `_build_index()` (lines 95-123) **零引用** `self.split`。全部 episode 无条件索引。

`train_unified.py:392`:
```python
val_dataset, val_collate_fn = build_dataset(cfg, split="val", processor=processor)
```

`build_dataset` (`data/__init__.py:68-69`) 传递 `split="val"` 给 `HDF5DatasetAdapter`, 但 adapter 不过滤。结果: val 数据 = train 数据。

**影响**: eval 指标在训练集上计算, 只能观察 loss 趋势, 不能判断泛化能力。不会崩, 但 eval 结果无真实参考价值。

**判定**: **TRUE**。建议在训练前修复 (~20 行)。最简方案: 用独立 val 数据目录, 在 `DataConfig` 中加 `val_data_dir`。

---

### V2: Refresh Frame Smoke Test 缺失 — **TRUE, P2 确认**

`train_smoke_test.py` 使用 `DummyVLADataset`, 不产生 `refresh_input_ids` 等字段 → `forward_train()` 走 `else` 分支 (line 386-399, 单 backbone 调用复用 R 次), refresh 路径 (line 373-385) 从未运行时验证。

**判定**: **TRUE**。代码审查上 refresh 路径逻辑正确, 但缺运行时测试。P2 — 不阻塞训练。

---

### V3: 单步监督设计限制 — **PARTIALLY FALSE, 需修正**

**analysis_v0_10_5 声称**: "所有 5 路损失仅作用于最后一个时间步" (`hybrid_vla_v2.py:466`)

**代码验证** — 这个声明与当前代码**不符**:

| 损失 | 监督范围 | 行号 | 证据 |
|------|---------|------|------|
| FAST Discrete | **全 T 步** | 484-494 | `fused_states.reshape(BT, -1)` — vectorized over B×T |
| Phase | **全 T 步** | 502-512 | `for t_sup in range(T)` 循环 |
| Affordance | **全 T 步** | 514-524 | `for t_sup in range(T)` 循环 |
| Flow Matching | **仅 t=-1** | 527 | `target_actions = batch["actions"][:, -1]` |
| Consistency (Temporal) | **全 T 步** | `fused_states [B,T,D]` 全窗口 |
| Consistency (SlowFast) | 仅最后 | slow_token vs fast_tokens |
| Consistency (Action) | 仅最后 | fast_continuous vs expert_continuous |

`hybrid_vla_v2.py:473` 的注释明确说:
```python
# ---- Losses (v0.10.3 P1-C: multi-step supervision) ----
# Supervise FAST/phase/affordance at ALL T timesteps to improve
# gradient density. Expert loss stays at t=-1 (expensive, Stage B/C).
```

**analysis_v0_10_5 引用的 "line 466" 已不存在** — 该行号对应 v0.10.2 代码。v0.10.3 的 P1-C 修复已将 FAST/Phase/Affordance 改为全步监督。当前代码中 line 527 的 `target_actions = batch["actions"][:, -1]` 仅供 **FM expert loss** 使用。

**判定**: **PARTIALLY FALSE**。3/5 路损失已全步监督 (v0.10.3 P1-C)。仅 FM expert 和部分 consistency 是单步。analysis_v0_10_5 基于过时的代码理解, 遗漏了 v0.10.3 修复。

**实际状态**:
- FAST/Phase/Affordance: 24/24 步监督 ✓ (已修复)
- FM Expert: 1/24 步 (设计选择, expert 前向开销大)
- Consistency Action: 1/24 步 (依赖 expert 输出)

这不是"所有损失单步"的问题, 而是"FM expert 单步"的设计权衡。当前梯度密度已从 v0.10.2 的 1/24 提升到 ~18/24 (加权)。

---

### V4: Per-Module LR 未实现 — **TRUE, P1 确认**

`train_unified.py:307-327` 验证:
```python
no_decay_keywords = {"bias", "res_scale", "LayerNorm.weight", "layer_norm.weight"}
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if not param.requires_grad: continue
    if any(nd in name for nd in no_decay_keywords):
        no_decay_params.append(param)
    else:
        decay_params.append(param)
optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": cfg.train.weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=cfg.train.learning_rate, ...)
```

只有 weight decay 分组, 无 per-module LR。Stage B 的 backbone LoRA / grounder / expert 全部共享 `learning_rate: 1e-4`。

**缓解**: stage_b.yaml 的全局 LR = 1e-4 (低于 Stage A 的 2e-4), 是对 expert 初始不稳定的一种全局缓解。但无法为不同模块设置不同 LR。

**判定**: **TRUE**。P1 — 训练可以进行但可能次优, 建议 Stage B 前补上。

---

### V5: Per-Module Gradient Norm 未记录 — **TRUE, P1 确认**

`train_unified.py:438`:
```python
grad_norm = clip_grad_norm_fsdp(model, cfg.train.max_grad_norm)
```
`train_unified.py:453`:
```python
logger.info("... gnorm: %.3f ...", grad_norm.item())
```

只有全局 gnorm。无法分辨 backbone/grounder/expert 各自的梯度量级。

**判定**: **TRUE**。P1 — 不阻塞训练, 但阻塞 "knowledge insulation 有效" 的论证。

---

## Part 2: analysis_v0_10_5 遗漏的问题

### M1: `build_dataset` 对 val split 的 data_dir 未区分 — P1

`build_dataset()` (`data/__init__.py:44-71`) 无论 `split="train"` 还是 `split="val"`, 都从同一个 `cfg.data.data_dir` 加载数据。没有 `val_data_dir` 配置字段。即使 `_build_index` 加了 split 过滤, 也需要一个独立 val 数据源。

当前 `DataConfig` (`config.py:288-306`) 没有 `val_data_dir` 或 `val_paths` 字段。

**两种可行方案**:
- A: 在 `DataConfig` 加 `val_data_dir`, `build_dataset` 根据 split 选择目录
- B: 在 `_build_index` 中按 episode 比例切分 (如前 90% train, 后 10% val)

方案 A 更干净 (训练/验证数据物理分离)。

---

### M2: `configure_trainable_modules` 未处理 `flow_matching_loss` 和 `discrete_loss` — P2

`configure_trainable_modules()` (train_unified.py:110-120) 的 `always_trainable` 列表:
```python
always_trainable = [
    model.grounder, model.temporal_core, model.action_history_encoder,
    model.proprio_proj, model.prev_action_proj, model.embodiment_embedding,
    model.fast_head, model.phase_head, model.affordance_head,
    model.consistency_loss,
]
```

缺少: `model.flow_matching_loss`, `model.discrete_loss`, `model.phase_loss`。

这些是 `nn.Module` 子类 (继承自 `nn.Module`):
- `FlowMatchingLoss` (`losses/flow_matching.py:9`) — 无可学习参数 (只有 MSE), 但如果将来加参数会被冻结
- `DiscreteCELoss` (`losses/discrete_loss.py`) — 无可学习参数
- `PhaseLoss` — 无可学习参数

**当前影响**: 零。这些 loss 模块没有可学习参数, 冻结/解冻无差别。
**潜在风险**: 如果将来给 loss 加可学习参数 (如 learnable loss weights), 会被 Step 1 冻结且不在解冻列表中。

**判定**: P2。当前无影响, 建议加上以防后续扩展。

---

### M3: `_fast_bin_centers` 注册为 buffer 但不在 `configure` 中处理 — OK

`hybrid_vla_v2.py` 中 `_fast_bin_centers` 是 `register_buffer` — buffer 不受 `requires_grad` 影响, 不在优化器中。无问题。

---

### M4: Smoke test Stage B/C 不验证 expert 梯度非零 — P2

`train_smoke_test.py` 对 Stage B/C:
- 调用 `configure_trainable_modules` + `sanity_check_trainable_params` ✓
- 执行 20 步 forward/backward ✓
- 检查 loss 不 NaN ✓
- **不检查** expert 参数在 step 后是否变化
- **不检查** `loss_fm` 是否存在

比 analysis_v0_10_5 报告的 "Stage B/C 有 loss_fm" 更准确地说: loss_fm 出现在 `optimize_v0_10_4.md` 的手动验证中, 但 smoke test 代码本身没有断言。

**判定**: P2。sanity check + loss 下降已提供合理置信度, 但显式断言更好。

---

## Part 3: 评分审核

### analysis_v0_10_5 评分 vs 审核修正

| # | 维度 | v0_10_5 评分 | 修正 | 理由 |
|---|------|:-----------:|:----:|------|
| 1 | 设计一致性 | 8.0 | **8.0** | 同意 |
| 2 | 正确性 | 9.0 | **9.0** | 同意 — V3 的"单步监督"描述不准确, 但不影响正确性评分 |
| 3 | 完备性 | 7.0 | **7.5** | V3 误判拉低了原评分; 实际 FAST/Phase/Affordance 已全步 |
| 4 | 训练稳定性 | 8.0 | **8.0** | 同意 |
| 5 | 可扩展性 | 6.5 | **6.5** | 同意 |
| 6 | 性能设计 | 6.0 | **6.5** | V3 修正: 多步监督已提供 24× 梯度密度提升 |
| 7 | 生产就绪度 | 6.0 | **6.0** | 同意 |
| 8 | 代码质量 | 8.0 | **8.0** | 同意 |
| 9 | 文档 | 4.5 | **4.5** | 同意 |
| 10 | 测试 | 2.0 | **2.0** | 同意 |

**修正综合评分**: 约 **7.4/10** (analysis_v0_10_5 给 7.2, 修正后略高, 因为 V3 的扣分不成立)

---

## Part 4: 结论

### analysis_v0_10_5 准确性评估

| 项目 | 准确? | 说明 |
|------|:-----:|------|
| P0 修复验证 (Part 1) | ✅ 完全准确 | 4 项 P0 全部正确验证 |
| V1 Val split | ✅ 准确 | split 参数确实被忽略 |
| V2 Refresh smoke test | ✅ 准确 | DummyVLADataset 不产生 refresh 字段 |
| **V3 单步监督** | **❌ 部分错误** | FAST/Phase/Affordance 已全步 (v0.10.3 P1-C), 仅 FM 单步。引用行号过时 |
| V4 Per-module LR | ✅ 准确 | 只有 decay/no-decay 分组 |
| V5 Per-module gnorm | ✅ 准确 | 只有全局 gnorm |
| 评分 7.2 | ⚠️ 偏低 0.2 | 因 V3 误判导致完备性/性能设计扣分过多 |

**结论**: analysis_v0_10_5 整体可靠, 4/5 新发现准确。**V3 是一个基于过时代码的误判** — v0.10.3 已修复多步监督, analysis 作者可能参考了旧版本代码或旧行号。

### 训练就绪度 — 与 analysis_v0_10_5 一致

| 场景 | 判定 | 与 v0_10_5 一致? |
|------|:----:|:---------------:|
| Stage A | ✅ 可以启动 | ✅ 一致 |
| Stage A eval | ⚠️ 需独立 val 数据 | ✅ 一致 |
| Stage B | ⚠️ 可运行但缺 per-module LR | ✅ 一致 |
| Stage C | ⚠️ 同上 | ✅ 一致 |

### 建议的训练前修复

同意 analysis_v0_10_5 的建议, 优先级微调:

| 优先级 | 项目 | 行数 | 理由 |
|--------|------|------|------|
| **1** | V1: Val split — 加 `val_data_dir` 到 DataConfig + build_dataset 分支 | ~25 行 | eval 指标从无意义变为有意义 |
| **2** | V5: Per-module gradient norm | ~30 行 | 验证 Stage B 梯度隔离 |
| 可选 | V4: Per-module LR (Stage B 前) | ~40 行 | 提升 Stage B 训练质量 |

**V3 不需要修复** — 多步监督已在 v0.10.3 实现。
