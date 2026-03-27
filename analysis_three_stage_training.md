# HybridVLA v2 — 三阶段训练完备性分析

> **标准**: 逐 stage 验证代码路径 × 配置 × 梯度流 × 损失组成，判断能否正确训练

---

## 1. Stage A — 感知对齐

### 1.1 设计意图

训练 backbone LoRA + grounder + temporal core + discrete heads。Expert 冻结。学习语言-视觉对齐、物体接地、时序动力学。

### 1.2 代码路径验证

| 检查项 | 文件:行 | 状态 |
|--------|---------|:----:|
| Expert 冻结 | `train_unified.py:96-97` 全冻结 + `:134` 不解冻 expert | ✅ |
| Backbone LoRA 可训练 | `train_unified.py:100-102` `"lora" in name` | ✅ |
| Grounder/Core/Heads 可训练 | `train_unified.py:114-132` always_trainable | ✅ |
| cond_builder 冻结（Stage A 不需要） | `train_unified.py:134` 仅 B/C 解冻 | ✅ |
| Sanity check 断言 | `train_unified.py:160-231` expert_trainable==0 | ✅ |

### 1.3 损失组成

| 损失 | 权重 | 代码路径 | 正确性 |
|------|------|---------|:------:|
| FAST discrete (全 T 步) | 1.0 | `hybrid_vla_v2.py:491-506` | ✅ 向量化全 T |
| Phase (全 T 步) | 0.5 | `:509-518` | ✅ 逐步 grounder token |
| Affordance (全 T 步) | 0.3 | `:521-530` | ✅ 同上 |
| Consistency (无 action 分支) | 0.3 | `:648-652` 只传 fused_states + fast/slow | ✅ Stage A 无 expert → 无 discrete-continuous 对齐 |
| **Flow matching** | — | `:534 stage != "a"` 跳过 | ✅ 不产生 |
| **RTC** | — | `:587 stage == "c"` 跳过 | ✅ 不产生 |
| **FASTER** | — | `:563 stage == "c"` 跳过 | ✅ 不产生 |

### 1.4 梯度流

```
Backbone (LoRA) ← Grounder ← Temporal Core (24步) ← FAST/Phase/Aff Loss
                                                    ← Consistency Loss (temporal + slow-fast)
```

Expert 完全冻结，无梯度。`cond_builder` / `core_to_expert` 也冻结，不浪费 optimizer 内存。**正确。**

### 1.5 配置验证 (`stage_a.yaml`)

| 字段 | 值 | 正确性 |
|------|-----|:------:|
| `stage: a` | ✓ | ✅ |
| `loss_weights` | fast=1.0, phase=0.5, aff=0.3, consistency=0.3 | ✅ 无 flow_matching |
| `resume_from` | 无 | ✅ 从头训练 |
| `stop_gradient_cond_prefix` | 默认 false | ✅ Stage A 不涉及 expert |
| `learning_rate` | 2e-4 | ✅ |
| `backbone_lr_scale` | 0.1 → backbone LR = 2e-5 | ✅ |

### 1.6 Stage A 判定: ✅ 完备

---

## 2. Stage B — Expert 训练

### 2.1 设计意图

解冻 expert + bridging projections。通过 flow matching 训练 expert 生成动作 chunk。`cond_prefix.detach()` 阻断梯度从 expert 回传到 backbone/grounder（知识隔离）。EMA 开始。

### 2.2 代码路径验证

| 检查项 | 文件:行 | 状态 |
|--------|---------|:----:|
| Expert 解冻 | `train_unified.py:135-146` stage "b" → 解冻 expert + cond_builder + projections | ✅ |
| cond_prefix.detach() | `hybrid_vla_v2.py:537-539` `stop_gradient_cond_prefix: true` | ✅ |
| block_fm_to_backbone | `stage_b.yaml:27` = true | ✅ |
| EMA 启动 | `train_unified.py:384-409` `ema_decay: 0.999` | ✅ |
| Cross-stage checkpoint | `stage_b.yaml:37` `resume_from: outputs/v2_stage_a/checkpoint-latest` | ✅ |
| `load_checkpoint(strict=False)` | `train_unified.py:410` | ✅ 新增 expert 权重不 crash |

### 2.3 损失组成

| 损失 | 权重 | 代码路径 | 正确性 |
|------|------|---------|:------:|
| FAST discrete (全 T) | 1.0 | `:491-506` | ✅ |
| Phase (全 T) | 0.5 | `:509-518` | ✅ |
| Affordance (全 T) | 0.3 | `:521-530` | ✅ |
| **Flow matching** | **1.0** | `:534 stage != "a"` → `:541-583` | ✅ 标准 FM loss |
| Consistency (含 action 分支) | 0.3 | `:640-646` 传入 discrete_actions + continuous_actions | ✅ |
| RTC | — | `:587 stage == "c"` 跳过 | ✅ 不产生 |
| FASTER | — | `:563 stage == "c"` 跳过 | ✅ 不产生 |

### 2.4 梯度流

```
                                    cond_prefix.detach()
                                          ↓ (梯度阻断)
Backbone LoRA ← Grounder ← Core ←→ FAST/Phase/Aff Loss + Consistency
                                   (梯度不从 expert 回传)

Expert ← Flow Matching Loss
Expert ← Consistency Loss (continuous_actions = expert_denoised.detach())
```

**关键验证**: `expert_denoised.detach()` (line 645) 确保 consistency loss 不向 expert 传梯度——expert 只从 FM loss 获得梯度。**正确。**

**但注意**: `cond_prefix.detach()` 只阻断从 expert 到 backbone/grounder 的梯度。FAST/Phase/Aff loss 仍然向 backbone/grounder 传梯度，不受影响。**这是正确的设计——感知模块从离散头获得梯度，expert 独立学习。**

### 2.5 配置验证 (`stage_b.yaml`)

| 字段 | 值 | 正确性 |
|------|-----|:------:|
| `stage: b` | ✓ | ✅ |
| `stop_gradient_cond_prefix: true` | ✓ | ✅ 知识隔离 |
| `block_fm_to_backbone: true` | ✓ | ✅ |
| `loss_weights` | + flow_matching=1.0 | ✅ |
| `resume_from` | `outputs/v2_stage_a/checkpoint-latest` | ✅ |
| `learning_rate` | 1e-4 (降半) | ✅ |
| `expert_lr_scale` | 0.5 → expert LR = 5e-5 | ✅ |
| `ema_decay` | 0.999 | ✅ |

### 2.6 潜在问题

**B1: Stage B 的 `loss_weights` 中无 `flow_matching` 键名验证**

`stage_b.yaml:35`: `flow_matching: 1.0`。代码中 `weights.get("flow_matching", 1.0)` (line 584)。如果 YAML 拼写错误（如 `flow_match`），会静默使用默认值 1.0。不会崩，但拼写错误不被发现。**P3 — 不影响训练。**

**B2: Per-module gnorm 可验证梯度隔离**

`train_unified.py:238-261` 每 5×log_interval 输出 backbone/grounder/expert 各自的 gnorm。Stage B 中应观察到 expert gnorm > 0 且 backbone gnorm 来自 FAST/Phase/Aff 而非 FM。**可验证，无问题。**

### 2.7 Stage B 判定: ✅ 完备

---

## 3. Stage C — 全微调 + RTC + FASTER

### 3.1 设计意图

解冻 backbone text 16-27。去掉梯度隔离（`stop_gradient_cond_prefix: false`）。启用 RTC（重叠 chunk inpainting）和 FASTER（近步加权）。

### 3.2 代码路径验证

| 检查项 | 文件:行 | 状态 |
|--------|---------|:----:|
| Backbone text 16-27 解冻 | `train_unified.py:149-155` | ✅ |
| Expert 保持解冻 | `train_unified.py:135-146` stage "c" | ✅ |
| `stop_gradient_cond_prefix: false` | `stage_c.yaml:26` | ✅ 梯度端到端流通 |
| RTC enable | `stage_c.yaml:30` + `hybrid_vla_v2.py:587` | ✅ |
| FASTER enable | `stage_c.yaml:38` + `:563` | ✅ |
| Cross-stage checkpoint | `stage_c.yaml:53` `resume_from: outputs/v2_stage_b/checkpoint-latest` | ✅ |

### 3.3 损失组成

| 损失 | 权重 | 代码路径 | 正确性 |
|------|------|---------|:------:|
| FAST discrete (全 T) | 1.0 | `:491-506` | ✅ |
| Phase (全 T) | 0.5 | `:509-518` | ✅ |
| Affordance (全 T) | 0.3 | `:521-530` | ✅ |
| **FM (FASTER weighted)** | **1.0** | `:563-578` near_ratio=0.3, far_ratio=4.0 | ✅ |
| **RTC inpainting** | **0.3** | `:587-621` prev_chunk + overlap MSE + smooth | ✅ |
| **FASTER aux** | **0.2** | `:628-637` denoised 近步 MSE | ✅ |
| Consistency (含 action) | 0.3 | `:640-646` | ✅ |

### 3.4 RTC 训练逻辑详细验证

**Line 586-621**:

1. `prev_chunk = action_expert.sample(cond_prefix, ..., num_steps=4, solver="euler")` — 低精度采样生成 "前一 chunk"。`torch.no_grad()` 内，不贡献梯度。✅

2. `prev_tail = prev_chunk[:, exec_H - overlap: exec_H].detach()` — 取前一 chunk 的执行尾部。`exec_H=8, overlap=max(1, int(0.333*8))=2`。切片 `[:, 6:8]` → `[B, 2, A]`。✅

3. `curr_head = expert_denoised[:, :overlap]` — 当前 chunk 的头部 `[:, :2]`。**有梯度**（来自 `expert_denoised` 的可微分路径）。✅

4. `loss_rtc = F.mse_loss(curr_head, prev_tail)` — 当前头部应匹配前一尾部。**梯度只流向 expert（curr_head）**，不流向 prev_chunk（detached）。✅

5. 平滑正则 (overlap≥2): `accel = boundary[:, 2:] - 2*boundary[:, 1:-1] + boundary[:, :-2]`。`boundary = [prev_tail(detached), curr_head]` → `[B, 4, A]`。二阶差分 = 3 点 → `accel.shape = [B, 2, A]`。✅

**数学验证**: overlap=2, exec_H=8, H=24:
- `prev_tail = prev_chunk[:, 6:8]` → 2 个动作
- `curr_head = expert_denoised[:, 0:2]` → 2 个动作
- `boundary = cat([2, 2]) = [B, 4, A]`
- `accel = [B, 2, A]` (4-2=2 个二阶差分点)
- ✅ 数值正确

### 3.5 FASTER 训练逻辑详细验证

**Per-step weighted FM loss (line 562-583)**:

H=24, near_ratio=0.3 → near_boundary=7

```
faster_w = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
faster_w[:7] *= 4.0 → [4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
sum = 28+17 = 45
normalise: *= 24/45
→ near: 4*24/45 ≈ 2.13  far: 1*24/45 ≈ 0.53
总和 = 24 ✅
```

**target_velocity = target_actions - noise** (line 574): 这里 `noise = x_0`（噪声起点），`target_actions = x_1`（干净目标）。velocity = x_1 - x_0 是 Rectified Flow 的正确目标速度。✅

**per_step_mse** (line 575): `(v_pred - v_target).pow(2).mean(dim=-1)` → `[B, H]`。对动作维度 A 取均值，保留步维度 H。✅

**Aux loss (line 628-637)**:
`expert_denoised[:, :near_boundary]` vs `target_actions[:, :near_boundary]` — 近 7 步的 MSE。`expert_denoised` 是可微分的（line 558 `noisy_actions + (1-t)*velocity`），梯度回传到 expert。✅

### 3.6 梯度流

```
Stage C: 梯度端到端流通（stop_gradient=false）

Backbone text 16-27 ← Grounder ← Core ← FAST/Phase/Aff Loss
                                        ← Consistency Loss
                                        ↕
                              cond_prefix (不 detach)
                                        ↓
Expert ← FM Loss (FASTER weighted) + RTC Loss + FASTER Aux Loss + Consistency
```

**Stage C 与 Stage B 的关键区别**:
1. `cond_prefix` 不再 detach → expert 梯度可以回传到 backbone/grounder（端到端微调）
2. RTC loss 额外约束 chunk 重叠一致性
3. FASTER 加权 FM loss 对近步权重更高
4. backbone text 16-27 解冻 → 更大的模型容量

### 3.7 配置验证 (`stage_c.yaml`)

| 字段 | 值 | 正确性 |
|------|-----|:------:|
| `stage: c` | ✓ | ✅ |
| `stop_gradient_cond_prefix: false` | ✓ | ✅ 端到端 |
| `rtc.enable: true` | ✓ | ✅ |
| `rtc.smooth_weight: 0.1` | ✓ | ✅ |
| `rtc.prev_chunk_steps: 4` | ✓ | ✅ 低精度快速采样 |
| `faster.enable: true` | ✓ | ✅ |
| `faster.near_ratio: 0.3` | ✓ | ✅ |
| `faster.aux_loss_weight: 0.2` | ✓ | ✅ |
| `loss_weights.rtc: 0.3` | ✓ | ✅ |
| `loss_weights.faster: 0.2` | ✓ | ✅ |
| `resume_from` | `outputs/v2_stage_b/checkpoint-latest` | ✅ |
| `learning_rate` | 3e-5 (再降) | ✅ |

### 3.8 潜在问题

**C1: RTC prev_chunk 使用当前 cond_prefix (P3-Design)**

`line 596`: 用**同一个** `cond_prefix` 生成 prev_chunk 和评估 expert_denoised。真实推理中前一 chunk 使用**前一个** semantic summary 的 cond_prefix。训练学到的是"同一条件下的重叠一致性"，而非"跨条件的过渡平滑性"。

**影响**: 训练中学到的 RTC 约束比实际推理时的约束更强（同条件 vs 跨条件）。模型可能在推理中跨条件过渡时 RTC 效果略弱于训练时的表现。**可接受——这是一个常见的训练简化，实际影响取决于语义变化速度。**

**C2: RTC 额外前向成本**

每个训练 step，RTC 在 `torch.no_grad()` 下额外执行一次 `action_expert.sample(num_steps=4, solver="euler")`。对于 18 层 expert，这相当于 4 次前向 pass。增加约 15-20% 的 Stage C 训练时间。**可接受的开销。**

**C3: FASTER `faster_w` 每步重新计算 (P3-Performance)**

`line 569`: `faster_w = torch.ones(H, device=device)` 每次 forward 重新创建。可以用 `register_buffer` 预计算。**不影响正确性，微小性能浪费。**

### 3.9 Stage C 判定: ✅ 完备

---

## 4. 跨 Stage 流程验证

### 4.1 Checkpoint 传递链

```
Stage A (从头训练)
  → outputs/v2_stage_a/checkpoint-latest/
    ├── model.pt       ← backbone LoRA + grounder + core + heads (已训练)
    ├── optimizer.pt   ← 不被 Stage B 加载
    └── meta.json      ← {"stage": "a", "step": 120000}

Stage B (resume_from Stage A)
  → load_checkpoint(strict=False) ← expert 权重是随机初始化的（Stage A 没有训练 expert）
  → configure_trainable_modules(stage="b") ← 解冻 expert
  → 新建 optimizer ← expert 参数首次进入优化器
  → outputs/v2_stage_b/checkpoint-latest/

Stage C (resume_from Stage B)
  → load_checkpoint(strict=False) ← 全部权重已有有意义的值
  → configure_trainable_modules(stage="c") ← 额外解冻 backbone text 16-27
  → 新建 optimizer
```

**验证**: `load_checkpoint(strict=False)` 允许缺失键（Stage A→B 时 expert 权重可能不匹配）和多余键。`configure_trainable_modules` 在 checkpoint 加载**之前**设置 `requires_grad`，但 `load_state_dict` 不修改 `requires_grad`，所以顺序安全。

**注意**: 实际执行顺序是 `configure → FSDP → optimizer → load_checkpoint`（`train_unified.py:329-410`）。optimizer 在 checkpoint 加载前创建，意味着 cross-stage resume 时 optimizer 状态是全新的（正确——不同 stage 有不同 LR/total_steps）。auto-resume（同 stage）会加载 optimizer 状态。✅

### 4.2 损失演进

| 损失 | Stage A | Stage B | Stage C |
|------|:-------:|:-------:|:-------:|
| loss_fast | ✅ 全 T | ✅ 全 T | ✅ 全 T |
| loss_phase | ✅ 全 T | ✅ 全 T | ✅ 全 T |
| loss_affordance | ✅ 全 T | ✅ 全 T | ✅ 全 T |
| loss_fm | ❌ | ✅ 标准 FM | ✅ FASTER 加权 FM |
| loss_rtc | ❌ | ❌ | ✅ 重叠 inpainting + 平滑 |
| loss_faster | ❌ | ❌ | ✅ 近步 aux MSE |
| loss_consistency | ✅ temporal+SF | ✅ +action agreement | ✅ +action agreement |

**Stage A→B 损失增量**: +flow_matching, +action_consistency。Expert 从冻结到活跃。
**Stage B→C 损失增量**: +rtc, +faster_aux。FM 从标准变为加权。梯度隔离解除。

### 4.3 LR 演进

| 模块组 | Stage A | Stage B | Stage C |
|--------|---------|---------|---------|
| Backbone LoRA | 2e-4 × 0.1 = 2e-5 | 1e-4 × 0.1 = 1e-5 | 3e-5 × 0.1 = 3e-6 |
| Core (grounder/core/heads) | 2e-4 | 1e-4 | 3e-5 |
| Expert | 冻结 | 1e-4 × 0.5 = 5e-5 | 3e-5 × 0.5 = 1.5e-5 |

**递减趋势合理**: 随着训练推进，LR 逐阶段降低，防止过拟合。

---

## 5. 推理路径验证

### 5.1 control_step RTC blending (`hybrid_vla_v2.py:763-780`)

```python
if rtc_cfg.enable and runtime_state.prev_chunk_tail is not None:
    overlap = runtime_state.prev_chunk_tail.shape[1]
    alpha = torch.linspace(1, 0, overlap, device=device)  # 线性衰减
    denoised[:, :overlap] = alpha * prev_tail + (1-alpha) * denoised[:, :overlap]

if rtc_cfg.enable:
    rtc_overlap = max(1, int(rtc_cfg.overlap_ratio * exec_horizon))
    runtime_state.prev_chunk_tail = denoised[:, exec_H - rtc_overlap: exec_H].clone()
```

- 首次调用 `prev_chunk_tail is None` → 跳过 blending → 正确 ✅
- 后续调用用线性 alpha 从 prev_tail 过渡到 curr_head → 平滑 ✅
- `overlap_ratio` 从 config 读取（已修复硬编码） → ✅

### 5.2 推理 FASTER

推理时 FASTER 影响 `num_sample_steps`：近步用少步去噪（快但粗），远步用多步（慢但精）。但当前 `control_step` 中的 `num_sample_steps` 由调用方传入，没有自动 FASTER 分步逻辑。

**这是一个推理侧未完整实现的部分**——训练中 FASTER 改变了损失权重，但推理中的 near/far 自适应步数需要在 `infer/runtime.py` 中实现。**P2——不阻塞基础推理，但 FASTER 的推理加速效果暂时无法体现。**

---

## 6. 总结

| Stage | 训练损失 | 梯度流 | 配置 | Checkpoint | 判定 |
|-------|---------|--------|------|-----------|:----:|
| **A** | 4 路 (FAST/Phase/Aff/Consistency) | backbone←grounder←core←heads | ✅ | 从头 | **✅ 完备** |
| **B** | 5 路 (+FM) | 知识隔离: cond_prefix.detach() | ✅ | resume A | **✅ 完备** |
| **C** | 7 路 (+RTC+FASTER) | 端到端 (无 detach) | ✅ | resume B | **✅ 完备** |

### 已发现问题

| ID | 问题 | 严重性 | 影响 |
|----|------|:------:|------|
| C1 | RTC prev_chunk 与 curr 用同一 cond_prefix | P3 | 训练简化，推理中跨条件 RTC 可能略弱 |
| C2 | RTC 每步额外 4 次 expert forward | P3 | ~15-20% Stage C 训练时间增加 |
| C3 | `faster_w` 每步重建而非 register_buffer | P3 | 微小性能浪费 |
| I1 | 推理侧 FASTER 自适应步数未实现 | P2 | FASTER 推理加速暂不可用 |

**无 P0/P1 问题。三阶段训练链路完备，可以开始训练。**

---

## 7. 中文摘要

三阶段训练代码路径经逐行验证，**全部完备**：

- **Stage A**: 4 路损失（FAST 全步 + Phase 全步 + Affordance 全步 + Consistency temporal/slow-fast），expert 冻结，backbone LoRA 0.1× LR。
- **Stage B**: 新增 flow matching loss，expert 解冻 (0.5× LR)，`cond_prefix.detach()` 实现知识隔离（per-module gnorm 可验证），EMA 启动。
- **Stage C**: 新增 RTC inpainting loss（重叠一致性 + 边界平滑）+ FASTER 加权 FM loss + FASTER 近步 aux loss。梯度隔离解除，backbone text 16-27 解冻。

跨 stage checkpoint 传递链正确（`strict=False` + 新建 optimizer）。LR 逐阶段递减。损失从 4 路渐进到 7 路。

**4 个 P2-P3 小问题不影响训练正确性。三阶段可以启动。**
