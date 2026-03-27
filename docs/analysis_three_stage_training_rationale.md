# HybridVLA v2 三阶段训练范式: 原理分析与判断

> **日期**: 2026-03-28
> **范围**: 训练范式的理论依据、收益、风险、与替代方案的对比
> **基于**: 代码实际实现 (修复后版本)

---

## 0. 架构速览

```
Qwen2-VL-7B (vision frozen, text 0-15 frozen, LoRA rank=64)
    ↓  [B, seq_len, 2048]
Hierarchical Attention Grounder (96 latents → 48 slots → 24 compressed)
    ↓  global_token + 24 compressed_slots + phase + uncertainty + affordance
Tri-Rate Mamba Core (Fast 20L + Medium 6L + Slow 10L)
    ↓  fused_state [B, 2048] per step
    ├─→ FAST Discrete Head (512-bin, 24×14)          ← 离散动作
    ├─→ Phase Head (16-class)                         ← 任务阶段
    ├─→ Affordance Head                               ← 可操作性
    └─→ cond_builder → core_to_expert → [B, 32, 1536]
            ↓
        Flow Action Expert (18L, Mamba+CrossAttn, AdaRMSNorm)
            ↓  velocity [B, 24, 14]                   ← 连续动作
```

---

## 1. 三阶段设计的核心思想

### 1.1 课程学习 (Curriculum Learning)

三阶段训练本质上是**模块级课程学习**——按模块职责分批引入训练目标, 而非一次性端到端优化所有参数。

```
Stage A ──── 学 "看" ────→ Stage B ──── 学 "做" ────→ Stage C ──── 学 "协调" ──→
感知表征 + 离散动作         连续动作生成 (固定感知)       全链路端到端微调
```

这遵循人类技能习得的规律: 先建立感知理解 (看懂任务), 再学精细控制 (执行动作), 最后整体协调优化。

### 1.2 为什么不能一步到位?

HybridVLA v2 的特殊性在于**双头架构** (离散 FAST + 连续 Flow Matching), 且二者共享 backbone 和 grounder。如果从第一步就联合训练:

| 问题 | 原因 |
|------|------|
| **梯度冲突** | 离散 CE loss 和连续 MSE/velocity loss 对 backbone 的梯度方向可能相反 |
| **感知表征未稳定** | Expert 接收的 conditioning 来自 grounder, 若 grounder 同时在变化, expert 学到的映射不稳定 |
| **Loss 量级差异** | FM loss 初始 ~1.0 (随机 velocity), 而 CE loss 初始 ~log(512)≈6.2, 梯度规模差距大 |
| **调参维度爆炸** | 同时调 backbone LR、expert LR、loss 权重、warmup 等, 搜索空间过大 |

---

## 2. 各阶段详解

### 2.1 Stage A: 感知表征建立

```
可训练:  backbone LoRA (rank=64) + grounder (8L, 96 latents)
         + temporal_core (36L Mamba) + 离散 heads
冻结:    vision tower, text layers 0-15, action expert
损失:    FAST CE (×1.0) + Phase CE (×0.5) + Affordance CE (×0.3) + Consistency (×0.3)
超参:    LR=2e-4, 120K steps, warmup=3K
```

**目的**: 建立从视觉-语言输入到**结构化场景理解**的映射。

**设计依据**:

1. **LoRA 而非全参数微调**: 7B backbone 全参数微调需要 >8×A100, LoRA (rank=64, ~50M 参数) 将 backbone 可训练参数量降低 ~99%, 在保持预训练知识的同时适配机器人领域。

2. **离散动作作为感知锚点**: FAST head 将连续动作离散化为 512-bin 分类问题。这不是最终的动作输出方式, 而是一个**代理任务** (proxy task) ——迫使 backbone+grounder 学会提取与动作决策相关的视觉特征。类似 BERT 的 MLM: 目标不是做完形填空, 而是学出好的表征。

3. **Phase + Affordance 辅助任务**: 多任务学习提供额外的归纳偏置, 防止表征过度拟合到 FAST 一个目标。Phase (16-class 任务阶段分类) 迫使模型理解任务进度, Affordance 迫使模型理解物体的可操作性。

4. **高学习率 (2e-4)**: 因为 grounder 和 temporal_core 是随机初始化的, 需要较大 LR 快速收敛。Backbone LoRA 通过 `backbone_lr_scale=0.1` 降到 2e-5, 保护预训练权重。

5. **120K steps**: 对 grounder (8L attention) + temporal_core (36L Mamba) 来说, 这是建立稳定表征所需的最低步数。过少则 Stage B 的 expert 在未稳定的特征上学习, 过多则浪费算力。

**本阶段结束时, 模型应具备**:
- 从 448×448 图像中提取 24 个 compressed object slots 的能力
- 用 tri-rate temporal 状态追踪任务进展
- 用离散 head 粗粒度预测动作 (512-bin 精度)
- Phase/Affordance 辅助理解

### 2.2 Stage B: 连续动作专家训练

```
新增训练:  action_expert (18L, 1536d) + cond_builder + 投影层
关键设计:  cond_prefix.detach() — FM 梯度不回传到 backbone
损失:      Stage A 全部 + Flow Matching MSE (×1.0)
超参:      LR=1e-4, expert_lr_scale=0.5, 200K steps, warmup=5K
           EMA decay=0.999
```

**目的**: 在**已稳定的感知特征**之上, 训练连续动作生成器 (Flow Matching Expert)。

**设计依据**:

1. **知识绝缘 (Knowledge Insulation)**: `cond_prefix.detach()` 是本阶段的核心设计。它实现了:
   - Expert 通过 cond_prefix 的**前向连接**接收 grounder 的感知特征 (信息流通)
   - FM loss 的**反向梯度被阻断**, 不影响已训练好的 backbone/grounder (梯度隔离)
   - 离散 head 的 loss 继续训练 backbone/grounder (感知路径持续优化)

   这等价于将模型分为两个独立的训练单元:
   ```
   单元 1 (离散路径): backbone → grounder → temporal → FAST head    [受 CE loss 驱动]
   单元 2 (连续路径): [frozen features] → expert                  [受 FM loss 驱动]
   ```

2. **Expert 冷启动**: Expert 从随机初始化开始, 这是标准做法 (类似 BERT + 随机 classification head)。它接收的 cond_prefix 来自 Stage A 训练好的模块, 提供了丰富的 32-token conditioning (global + 24 slots + phase + uncertainty + affordance + temporal fused/fast/medium/slow)。

3. **更低基础 LR (1e-4 vs 2e-4)**: Stage A 的模块已部分收敛, 不需要太大 LR。Expert 通过 `expert_lr_scale=0.5` 得到 5e-5, 适合从头训练 18L transformer。

4. **最长阶段 (200K steps)**: Flow Matching 的 velocity 回归比离散分类更难收敛。Expert 需要学习从 32 维 conditioning 到 24×14 维动作轨迹的复杂映射, 且训练信号来自随机 timestep 采样 (logit-normal), 每步只在一个 t 值上提供监督。

5. **EMA 启动 (decay=0.999)**: Expert 训练波动较大 (FM loss 的随机性), EMA 平滑模型权重, 提供更稳定的推理模型。

**本阶段结束时, 模型应具备**:
- 高精度的连续动作生成能力 (通过 ODE 求解)
- 离散和连续双头输出的能力
- 但两个头独立运作, 未做端到端协调

### 2.3 Stage C: 端到端协调微调

```
新增训练:  backbone text layers 16-27 (12 层)
关键变更:  stop_gradient_cond_prefix=false — FM 梯度可回传到 backbone
新增损失:  RTC (×0.3) + FASTER aux (×0.2)
超参:      LR=3e-5 (全局最低), 80K steps, warmup=2K
           EMA decay=0.9999 (更保守)
```

**目的**: **打通全链路端到端梯度**, 让 FM loss 的信号可以影响 backbone, 实现感知和控制的协同优化。

**设计依据**:

1. **移除梯度阻断 (`stop_gradient_cond_prefix=false`)**: Stage B 的 detach 保护了感知模块, 但也意味着 expert 无法告诉 backbone "我需要什么样的特征"。Stage C 移除这道墙, 让 FM loss 产生的梯度沿以下路径回传:

   ```
   FM loss → expert → cond_prefix → cond_builder → grounder → backbone text 16-27 + LoRA
   ```

   这使得 backbone 可以根据 expert 的需求调整感知输出——例如, 如果 expert 发现某些物体特征不够细致, 梯度会驱动 grounder 和 backbone 提取更有区分度的特征。

2. **极低学习率 (3e-5, backbone 实际 3e-6)**: 全部模块已收敛到较好的状态, Stage C 是精细调整。过大 LR 会破坏 Stage A/B 积累的知识。EMA decay 也从 0.999 升到 0.9999 (更长记忆), 进一步抑制波动。

3. **解冻 text layers 16-27**: Qwen2-VL 的 28 层 text transformer 中:
   - Layers 0-15: 编码通用语言理解, **保持冻结**以防灾难性遗忘
   - Layers 16-27: 编码高层语义, 解冻后可以适配机器人域特有的指令理解

   这遵循 "深层可调、浅层冻结" 的经典策略——浅层特征更通用 (语法、词义), 深层特征更任务相关 (指令→动作映射)。

4. **RTC (Receding Temporal Consistency)**: 解决 chunk-based 动作预测的核心问题——相邻 chunk 在交接处的不连续。
   ```
   时间轴: [---- chunk k ----][---- chunk k+1 ----]
                        ↑ overlap 区域
                        │ 无约束 → 速度/加速度跳变
                        │ RTC → 迫使 overlap 区域匹配 + 平滑过渡
   ```
   - **Inpainting loss**: 当前 chunk 的头部 ≈ 前一个 chunk 的尾部
   - **Smoothness loss**: 惩罚交接处的加速度突变 (二阶差分)
   - 仅在 Stage C 启用, 因为 Stage B 的 expert 还不够稳定

5. **FASTER (Frequency-Adaptive Step-size Training)**: 机器人控制中, 近期动作比远期动作重要得多 (近期预测直接执行, 远期预测会被下一个 chunk 覆盖)。
   - **加权 FM loss**: 近期步 (前 30%) 权重 = 远期步的 4 倍
   - **近期辅助 loss**: 额外约束 denoised 预测在近期的精度
   - 这引导 expert 的 velocity field 在近期更准确, 允许远期略微模糊

**本阶段结束时, 模型应具备**:
- 端到端协调的感知-控制能力
- 时间一致的动作输出 (无 chunk 交接跳变)
- 近期动作高精度、远期动作合理

---

## 3. 为什么是三阶段而不是其他方案

### 3.1 与替代方案的对比

| 方案 | 优势 | 劣势 | 适用性 |
|------|------|------|--------|
| **一阶段端到端** | 简单, 无转换开销 | 梯度冲突, 调参困难, FM 可能破坏 backbone | 小模型 (<1B) 可行 |
| **二阶段 (A→C)** | 省一步 | Expert 冷启动在变化中的特征上训练, 不稳定 | 若 expert 很小可行 |
| **二阶段 (A→B)** | 避免全参微调风险 | 缺少端到端协调, 离散/连续头脱节 | 若追求稳定性 |
| **三阶段 (A→B→C)** | 渐进式引入复杂度, 每步风险可控 | 总训练步数多 (400K), 需管理转换 | **当前方案** |
| **交替训练 (Alternating)** | 梯度不冲突 | 实现复杂, 每步只用一半数据 | 不适合大模型 |
| **蒸馏 (离散→连续)** | 无双头竞争 | 需额外蒸馏阶段, 且离散精度上限限制连续 | 若离散精度足够 |

### 3.2 三阶段的关键优势

**渐进式参数解冻 (Progressive Unfreezing)**:

```
Stage A:  LoRA (50M) + grounder/core/heads (~200M)
Stage B:  + expert (~180M) + cond_builder (~12M)
Stage C:  + text 16-27 (~2.4B)
           总可训练参数: ~50M → ~440M → ~2.9B
```

每阶段的可训练参数量逐步增大, 但已训练参数已有良好初始化。这使得:
1. 每阶段的优化景观更平坦 (更少 local minima)
2. 调试成本更低 (可以独立验证每阶段)
3. 失败时可以从上一个 checkpoint 重试, 不必从头来

**梯度路径渐进扩展**:

```
Stage A: loss → heads → temporal → grounder → LoRA
Stage B: loss → heads → temporal → grounder → LoRA
         loss_fm → expert (隔离)
Stage C: loss → heads → temporal → grounder → LoRA → text 16-27
         loss_fm → expert → cond → grounder → LoRA → text 16-27  (打通)
```

每一步都在前一步的基础上扩展梯度路径, 而非一次性建立所有连接。

---

## 4. 优势总结

### 4.1 训练稳定性

- **避免 FM loss 破坏感知表征**: Stage A 的 120K 步让 backbone 建立稳定的视觉-语言-动作映射, 这些知识在 Stage B 被 detach 保护, 在 Stage C 以极低 LR 微调
- **避免 Expert 在不稳定特征上学习**: Stage B 的 expert 接收已稳定的 cond_prefix, 学到的映射不会因为 backbone 剧烈变化而失效
- **避免突然释放过多梯度**: Stage C 的 LR 是 Stage A 的 1/7, 且有 grad clipping (max_norm=1.0)

### 4.2 可调试性

- 每阶段可独立验证:
  - Stage A 结束: 检查 FAST 离散精度、grounder attention 可视化
  - Stage B 结束: 检查 FM 重建质量、ODE 采样轨迹
  - Stage C 结束: 检查端到端执行成功率
- 某阶段失败不必从头训练, 可从上一阶段 checkpoint 调参重试

### 4.3 计算效率

- Stage A 冻结 expert (180M 参数), 省 ~30% 显存和计算
- Stage B 的 detach 减少了 FM loss 的反传深度 (不需要穿过 grounder→backbone)
- Stage C 最短 (80K steps), 因为只是微调已收敛的模型

### 4.4 离散-连续协同

- 离散头 (FAST) 提供快速粗粒度预测, 可用于实时 fallback
- 连续头 (FM) 提供高精度轨迹, 用于主要控制
- Consistency loss 让两者在训练中保持一致
- 推理时可用离散头做实时安全检查 (confidence gating)

---

## 5. 潜在问题与风险

### 5.1 阶段转换中的信息断裂

**问题**: 跨阶段不继承 optimizer state (动量、自适应 LR)。

```
Stage A optimizer (momentum for backbone/grounder) → 丢弃
Stage B optimizer (新建, 含 expert group)          → 丢弃
Stage C optimizer (新建, 含 text layer group)
```

**影响**: 每个阶段开头有一段 "冷启动期", optimizer 需要重新积累二阶矩估计。warmup 步数 (3K/5K/2K) 部分缓解此问题, 但不完美。

**判断**: **可接受**。不同阶段的参数组和 LR 配置差异较大, 旧的 AdamW state 反而可能有害。这是课程学习的标准做法。

### 5.2 Expert 冷启动 + 已训练 Backbone 的协调难度

**问题**: Stage B 的 expert 从随机初始化开始, 但它接收的 conditioning 来自已训练 120K 步的 grounder。这意味着:
- Expert 的输入分布从第一步就是 "高质量" 的 (而非随机), 好处是加速收敛
- 但 grounder 在 Stage B 仍在被离散头 loss 继续优化, 所以 cond_prefix 分布会缓慢漂移

**影响**: Expert 训练前期可能出现 loss 不降 (在漂移的特征上适应), 直到两者达到动态平衡。

**判断**: **风险低**。`cond_prefix.detach()` 确保漂移只来自离散头路径 (梯度远小于 FM), 且 Stage B 的 LR (1e-4 for core) 低于 Stage A (2e-4), 漂移速度有限。200K 步的训练长度足以让 expert 追上。

### 5.3 Stage C 梯度竞争

**问题**: Stage C 中, backbone text layers 16-27 同时从**两个方向**接收梯度:
- 离散路径: CE loss → FAST head → temporal → grounder → backbone
- 连续路径: FM loss → expert → cond_prefix → cond_builder → grounder → backbone

这两个梯度可能方向不一致。离散头可能希望 backbone 输出更 "分类友好" 的特征, 而 expert 可能希望更 "回归友好" 的特征。

**影响**: 可能导致 backbone 梯度震荡, 最终学到两个任务的折中表征, 而非对某一个最优。

**判断**: **风险中等, 但已被缓解**:
1. Stage C LR 极低 (backbone 实际 3e-6), 梯度竞争的影响被压缩
2. max_grad_norm=1.0 限制了最坏情况
3. Consistency loss (×0.3) 在训练中对齐离散和连续预测, 减少两个路径对 backbone 需求的分歧
4. EMA (decay=0.9999) 平滑最终权重

**建议**: 监控 Stage C 早期 backbone 的 per-layer gradient norm, 如果离散路径和连续路径的梯度 cosine similarity 持续为负, 考虑降低 FM loss 权重或恢复 detach。

### 5.4 总训练开销

**问题**: 三阶段总计 **400K steps** (120K+200K+80K), 每步处理 batch=64, 即 2560 万样本。

与一阶段方案相比:
- 一阶段可能只需 200-300K steps (但每步更贵, 需全参数梯度)
- 三阶段总 FLOP 可能更多, 但每步 FLOP 更少 (前两阶段冻结大量参数)

**判断**: **可接受的权衡**。单步效率 × 稳定性的乘积更高。且 Stage A/B 的显存占用更低 (冻结 expert/text layers), 允许使用更大 batch 或更少 GPU。

### 5.5 RTC 的额外推理开销

**问题**: RTC 训练时需要在每个 batch 执行一次 `action_expert.sample()` (4-step Euler ODE) 来生成 "前一个 chunk", 增加 ~20% 的 expert 前向计算。

**判断**: **可接受**。仅在 Stage C (80K steps) 启用, 且 `prev_chunk_steps=4` 是最粗粒度的采样, 计算量远小于主训练的 backbone 前向。推理时 RTC blending 几乎零开销 (只是 alpha 加权)。

### 5.6 离散头的长期价值

**问题**: FAST 离散头在 Stage A 作为 proxy task 非常有价值, 但到 Stage C, 连续 expert 已经收敛, 离散头的监督信号是否还有意义? 继续训练它会占用 backbone 梯度预算。

**判断**: **有争议, 但当前设计合理**:
- **保留理由**: 离散头提供了更丰富的 backbone 梯度 (512 类分类的梯度比 MSE 的梯度信息量更大), 且推理时可用作 safety check
- **移除理由**: 可以在 Stage C 降低 `fast_discrete` 权重 (如 0.3), 让 FM loss 占主导
- **建议**: 可实验性地对比 Stage C 中 `fast_discrete: 1.0` vs `0.3` 的效果

---

## 6. 关键设计判断

### 6.1 正确且关键的设计

| 设计 | 理由 | 判断 |
|------|------|------|
| Stage B 的 `cond_prefix.detach()` | 保护已训练的感知模块, 让 expert 在稳定特征上学习 | **核心设计, 不可移除** |
| Stage C 的低 LR (3e-5) | 端到端微调已收敛模型, 需极保守 | **正确** |
| Text layers 0-15 始终冻结 | 保护通用语言/视觉理解 | **正确** |
| EMA 在 Stage B 启动 | Expert 训练初期波动大, EMA 提供稳定推理权重 | **正确** |
| Consistency loss 中 `expert_denoised.detach()` | 防止 FM 梯度通过 consistency 绕行到 temporal_core | **正确** |

### 6.2 合理但可商榷的设计

| 设计 | 当前值 | 可选范围 | 建议 |
|------|--------|----------|------|
| Stage A 步数 | 120K | 80K-150K | 看 FAST loss 收敛曲线, 如果 80K 已平坦可缩短 |
| Stage B 步数 | 200K | 150K-250K | FM loss 收敛慢, 200K 是安全值 |
| Stage C 步数 | 80K | 50K-100K | 如果 end-to-end 稳定可缩短, 如果 RTC 收敛慢需加长 |
| `backbone_lr_scale` | 0.1 | 0.05-0.2 | Stage C 中 0.1 可能偏保守 |
| `expert_lr_scale` | 0.5 | 0.3-1.0 | 0.5 是合理的, 1.0 可能让 expert 训练更快但更不稳定 |
| Stage C 的 `fast_discrete` 权重 | 1.0 | 0.3-1.0 | FM 已收敛后可考虑降低 |

### 6.3 需要实验验证的假设

1. **Stage C detach 移除是否真的有益**: 我们将 `stop_gradient_cond_prefix` 改为 false, 理论上更好 (端到端), 但实际效果需要对比实验。如果 Stage C 的 FM loss 反弹或 FAST loss 劣化, 可能需要恢复 detach。

2. **RTC 的实际贡献**: RTC 增加了训练开销和复杂性。需要做 ablation: 有/无 RTC 的推理 action smoothness 对比。

3. **FASTER 的 near_ratio=0.3 是否最优**: 这意味着前 7 步 (7/24=29%) 被视为 "近期"。具体值应根据控制频率和执行延迟调整。

---

## 7. 总体判断

### 优势 (相比一阶段端到端)

```
✓ 训练稳定性: 消除了 CE/FM 梯度冲突对 backbone 的破坏风险
✓ 可调试性:   每阶段独立验证, 失败可局部重试
✓ 计算效率:   前两阶段显存占用更低, 可更大 batch
✓ 理论完备:   渐进解冻 + 知识绝缘是成熟的训练范式
```

### 劣势 (相比一阶段端到端)

```
✗ 总训练步数更多 (400K vs ~250K)
✗ 需管理 3 个 YAML 配置 + 2 次 checkpoint 转换
✗ 阶段转换存在 optimizer 冷启动
✗ 最终性能依赖阶段间衔接, 而非全局最优
```

### 最终评价

> HybridVLA v2 的三阶段训练是一个**工程上成熟、理论上有据**的设计。它用可控的额外训练开销换取了显著的稳定性和可调试性收益。对于 7B backbone + 双头架构这一复杂组合, 分阶段训练是比一步端到端**更务实的选择**。
>
> 主要风险点在 Stage C 的梯度竞争 (离散 vs 连续路径), 需要通过 gradient norm 监控和 loss 曲线来验证。如果竞争严重, 可以通过降低 `fast_discrete` 权重或恢复部分 detach 来缓解。
>
> **建议的关键 ablation 实验:**
> 1. Stage C `stop_gradient_cond_prefix`: true vs false
> 2. Stage C `fast_discrete` 权重: 1.0 vs 0.5 vs 0.3
> 3. RTC/FASTER 的 on/off ablation
>
> 这三个实验的结果将决定三阶段范式的最终收益上限。
