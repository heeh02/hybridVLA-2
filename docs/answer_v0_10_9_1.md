# HybridVLA v2 v0.10.10 — 问题修复报告 (Round 2)

> 修复日期: 2026-03-28
> 对应问题文档: `docs/problem_v0_10_9.md`
> 测试结果: **49/49 通过** (`pytest tests/ -q`)
> 本轮修复包含第一轮修复 + 第二轮 review 反馈修复

---

## 一、修复总览

### 第一轮: 21 项问题修复 (L-1 ~ L-21)
### 第二轮: 5 项 review 反馈修复

| 轮次 | 严重性 | 数量 | 关键内容 |
|------|--------|------|---------|
| R1 | P0 | 1 | Optional field contract bug |
| R1 | P1 | 6 | Camera, EMA barrier, RTC, FASTER, ActionHistoryEncoder |
| R1 | P2 | 8 | Ring buffer, loss sum, consistency, dead code |
| R1 | P3 | 5 | API cleanup, config, rename |
| R2 | P1 | 3 | Camera ndim fix, barrier reposition, EMA ckpt compat |
| R2 | P2 | 1 | World model import paths |
| R2 | P3 | 1 | Consistency knobs wiring |

---

## 二、第二轮修复详情

### R2-1 [P1]: CameraPositionEmbedding 多相机 batch 崩溃

**根因**: collate 通过 `_safe_stack_vision` 将 per-sample `[N_images, 3]` 堆叠为 `[B, N_images, 3]`，但第一轮修复仍按 `[total_images, 3]` 二维张量索引。`image_grid_thw[img_idx].tolist()` 在 3D tensor 上返回嵌套 list，导致 `TypeError`。

**修复**: 检测 `image_grid_thw.ndim`，分两路处理：
- ndim == 3: `grids_b = image_grid_thw[b]` (batched)
- ndim == 2: `grids_b = image_grid_thw[b*n_img:(b+1)*n_img]` (flat)

逐元素 `.item()` 取值，避免 `.tolist()` 返回嵌套结构。

**验证**: 手动测试 B=2, num_cameras=3 的 batched `[2,3,3]` 和 flat `[6,3]` 两种输入均通过。

**文件**: `vla_hybrid_v2/models/qwen2vl_backbone.py:90-130`

---

### R2-2 [P1]: EMA barrier 放置位置错误

**根因**: 第一轮 barrier 在 `apply()` 前和 `restore()` 后。实际危险窗口是：
- fast rank 在 slow rank 还未 apply 时就进入 eval (用的是训练权重)
- fast rank 在 slow rank 还在 eval 时就 restore (破坏对方的 EMA 权重)

**修复**: 3 个 barrier 重新放置：
1. `ema.apply(model)` **之后** → 所有 rank 都已切换到 EMA 权重
2. `evaluate()` **之后**, `ema.restore()` **之前** → 所有 rank 都已完成 eval
3. `ema.restore(model)` **之后** → 所有 rank 都已恢复训练权重

**文件**: `scripts/train_unified.py:556-571`

---

### R2-3 [P1]: EMA checkpoint 旧权重 shape mismatch

**根因**: `load_checkpoint()` 的 shape filter 只处理 `model.pt`，不处理 `ema.pt`。`EMAModel.load_state_dict()` 直接赋值 shadow dict，导致首次 `apply()`/`update()` 时 `param.data.copy_(shadow)` 因 shape 不匹配而崩溃。

**修复**: `EMAModel.load_state_dict()` 新增 shape filter — 对比 loaded shadow 与当前 shadow 的 shape，不匹配的 key 删除并 warning。被删除的 key 保持当前模型的初始 shadow 值，在下一次 `update()` 时自然被 EMA 更新覆盖。

**文件**: `vla_hybrid_v2/utils/ema.py:123-145`

---

### R2-4 [P2]: World model 迁移后内部 import 断裂

**根因**: `imagination_engine.py` 和 `world_model_loss.py` 内部 7 处 `from vla_hybrid_v2.world_model.*` 指向已删除的旧路径。

**修复**: 全部替换为 `from vla_hybrid_v2.experimental.world_model.*`。

**验证**: `from vla_hybrid_v2.experimental.world_model.imagination_engine import ImaginationEngine` 通过。

**文件**: `vla_hybrid_v2/experimental/world_model/imagination_engine.py`, `world_model_loss.py`

---

### R2-5 [P3]: Consistency loss 配置旋钮未接线

**根因**: `V2ConsistencyLoss` 新增了 `temperature/slow_fast_weight/action_weight` 参数，但 `HybridVLAv2.__init__` 仍用 `V2ConsistencyLoss(ecfg.action_dim)` 默认构造，config 调不了。

**修复**:
- `TrainConfig` 新增 `consistency_temperature`, `consistency_slow_fast_weight`, `consistency_action_weight`
- `HybridVLAv2.__init__` 将这些值传入 `V2ConsistencyLoss()`

**文件**: `vla_hybrid_v2/config.py:255-257`, `vla_hybrid_v2/models/hybrid_vla_v2.py:200-205`

---

## 三、诚实状态评估

### 已确认修复 (verified fixed)

| ID | 问题 | 验证方式 |
|----|------|---------|
| L-1 | Optional field contract bug | 测试覆盖 + 代码审查确认所有 `.get()` |
| L-2 | cond_prefix 静默截断 | 测试触发 warning |
| L-3 | CameraPositionEmbedding batch | 手动测试 batched + flat |
| L-4 | evaluate() EMA barrier | 代码审查确认 3 barrier 位置 |
| L-6 | FASTER NotImplementedError | 测试覆盖 |
| L-7 | ActionHistoryBuffer ring | 手动测试 6 push → 正确顺序 |
| L-8 | loss_total torch.stack | 代码审查 |
| L-10 | assert → ValueError | 代码审查确认全部替换 |
| L-11 | ActionHistoryEncoder 缩减 | 参数量从 108M → ~1.7M |
| L-17 | FAST head 重命名 | 测试确认 alias |
| L-19 | eval() → get_type_hints | 手动测试 config 加载 |
| L-20 | world_model 迁移 | import 测试 + 内部 import 修复 |
| L-21 | train_stage_a 薄包装 | 代码审查 |
| R2-1 | Camera ndim 处理 | 手动测试 |
| R2-2 | Barrier 重新放置 | 代码审查 |
| R2-3 | EMA ckpt shape filter | 手动测试 shape mismatch |
| R2-4 | World model import 修复 | import 测试 |
| R2-5 | Consistency config 接线 | 测试覆盖 |

### 已缓解但非完整修复 (mitigated, not fully closed)

| ID | 问题 | 当前状态 | 为什么不算完整修复 |
|----|------|---------|------------------|
| **L-5** | RTC train-infer 分布不一致 | 对 cond_prefix 加 0.01 噪声 | 加噪声只是近似 — 真正修复需要使用前一时间步的 cond_prefix，需要数据管线提供 T-1 观测。当前方案是合理的工程近似但不等于消除了分布差异。 |
| **L-12** | ContrastiveTemporalLoss collapse | 添加 VICReg variance term | InfoNCE 仍将同轨迹近邻当 negative 的根本问题未解决。variance term 是防护网，不是根因修复。完整方案需要排除同轨迹 negative 或使用不同的 contrastive 策略。 |
| **L-14** | SlowFastAgreementLoss 单向训练 | 移除 .detach() | 双向对齐是否合理需要训练验证 — slow stream 设计目标就应与 fast 不同，双向 MSE 可能导致两者退化为同一表示。需要 ablation 确认。 |
| **L-15** | Phase/Affordance 无监督 | 添加 startup warning | 警告不等于修复。真正修复是：(A) 数据 adapter 补标签，或 (B) 默认关闭这两个头，或 (C) 启动时断言头+标签一致性。当前只做了 (warning)。 |

### 未修复 (not fixed, design decision needed)

| ID | 问题 | 为什么未修 |
|----|------|----------|
| **L-16** | ExpertMambaBlock 未使用 official Mamba2 | 添加了检测和 log，但 Expert 的 12 层 Mamba 仍走自定义实现。真正替换需要适配 AdaRMSNorm conditioning 到 official Mamba2 block 的 interface，这是非 trivial 的架构工作。当前已使用 `selective_scan_fn` CUDA kernel 加速。 |
| **L-13** | ActionConsistencyLoss 换成 MSE | MSE loss 的 scale 与原 cosine loss 不同。权重 0.5 在 cosine [0,2] 范围合理，但 MSE 值域无界。需要训练时观察 loss 曲线并可能调整 `consistency_action_weight`。 |
| **L-18** | Sub-weights 暴露到 config | 已完成接线，但默认值 (temperature=0.1, weights=0.5) 是否合理需要训练验证。 |

### 测试覆盖缺口

当前 49 个测试全部通过，但以下路径 **没有自动化测试覆盖**：
- 多相机 batch (B>1, num_cameras>1) 的端到端 forward
- 分布式 EMA eval (需多 GPU)
- 旧 checkpoint auto_resume + shape mismatch
- world_model enable=True 的构造路径
- RTC 训练路径 (Stage C)
- Consistency loss 的梯度是否真的反传到双方

---

## 四、修改文件清单

| 文件 | 变更类型 | 涉及 ID |
|------|---------|---------|
| `vla_hybrid_v2/models/hybrid_vla_v2.py` | L-1,2,5,6,8,11,15 + R2-5 | bug fix, warning, config |
| `vla_hybrid_v2/models/qwen2vl_backbone.py` | L-3 + R2-1 | bug fix |
| `vla_hybrid_v2/models/mamba_core.py` | L-11 | architecture |
| `vla_hybrid_v2/models/flow_action_expert.py` | L-16 | detection |
| `vla_hybrid_v2/models/discrete_heads.py` | L-17 | rename |
| `vla_hybrid_v2/losses/consistency_loss.py` | L-12,13,14,18 | loss redesign |
| `vla_hybrid_v2/losses/flow_matching.py` | L-9 | API |
| `vla_hybrid_v2/types.py` | L-7 | ring buffer |
| `vla_hybrid_v2/config.py` | L-19 + R2-5 | eval→hints, config fields |
| `vla_hybrid_v2/utils/ema.py` | R2-3 | ckpt compat |
| `vla_hybrid_v2/utils/checkpointing.py` | ckpt compat | shape filter |
| `scripts/train_unified.py` | L-4 + R2-2 | barrier |
| `scripts/train_stage_a.py` | L-21 | deprecate |
| `vla_hybrid_v2/experimental/world_model/*.py` | L-20 + R2-4 | relocate + imports |
| `vla_hybrid_v2/world_model/__init__.py` | L-20 | compat re-export |
| `tests/test_control_step.py` | L-6 test update | test |
| `tests/test_losses.py` | L-13 test update | test |

---

## 五、下一步建议

### 训练前必须完成
1. **L-15 决策**: phase/affordance head 是否在无标签时默认关闭 — 当前 warning 不够，应改为 config 控制
2. **补测试**: 多相机 batch forward, EMA round-trip, ActionHistoryBuffer edge cases

### 训练初期
3. **观察 consistency loss 曲线**: MSE 替代 cosine 后 scale 是否合理，variance term 是否有效防止 collapse
4. **L-5 验证**: RTC 噪声近似是否足以让重叠区平滑过渡

### HPC3 上机后
5. **H-2/H-4**: 500 步 + 2×GPU smoke test
6. **H-7**: official vs fallback Mamba benchmark
7. **H-12**: consistency loss 有效性验证
