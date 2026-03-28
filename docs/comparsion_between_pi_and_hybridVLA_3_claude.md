# HybridVLA v2 vs OpenPI 对比分析报告 v3 (客观版)

> 分析日期: 2026-03-28
> HybridVLA 版本: v0.10.9 (~10,714 行, 单人开发, 零真实训练)
> OpenPI 版本: 开源发布版 (PI 团队, 10K+ 小时真实机器人数据验证)
> 前序: v2 报告基于 v0.10.7

---

## 1. 一句话结论

**HybridVLA v2 是一个架构想法丰富但从未被验证的研究原型。OpenPI 是一个已在真实世界证明能力的工程系统。两者不在同一个发展阶段, 直接打分对比有误导性。**

---

## 2. v0.10.7 → v0.10.9 变更: 实事求是

本轮改动是纯 bug fix, 修了 10 项问题中的 9 项。修复本身质量好, 但需要正视一个事实: **这些 bug 本不该存在**。

### 2.1 修复的是什么

| 问题 | 严重性 | 含义 |
|------|--------|------|
| FSDP evaluate 死锁 | **P0** | **多卡训练从来没真正跑通过** — 否则早就发现了 |
| EMA/FSDP 语义错误 | **P0** | **EMA 权重在多卡下一直是垃圾** — shadow 读的是不完整的 shard |
| 推理不加载 EMA 权重 | P1 | 推理管线写了但没人用真模型跑过 |
| Action 无 clipping | P2 | 同上 |
| Val 无 DistributedSampler | P2 | 多卡验证指标一直不对 |
| Gnorm 在 zero_grad 后 | P2 | per-module gnorm 日志一直是全零 |

**这些 bug 的共同指向: 整个训练 + 推理管线从未被端到端执行过。** 它们不是边界情况, 而是主流程上的基础错误。如果真正跑过一次 8×GPU Stage B 训练, FSDP 死锁会立即暴露; 如果真正跑过一次推理, 缺 EMA 和缺 clipping 也会立即暴露。

### 2.2 修复后真正改变了什么

- 之前: **多卡训练声称可用但实际会死锁**, 推理声称可用但输出不正确
- 现在: 多卡训练 + 推理在代码层面正确了, **但仍然没人跑过**

这是一个从 "不可能工作" 到 "理论上能工作" 的进步, 不是从 "能工作" 到 "工作得更好"。

---

## 3. 诚实的差距评估

### 3.1 最根本的差距: 零验证

| | HybridVLA v2 | OpenPI |
|--|-------------|--------|
| 真实机器人数据训练 | **0 小时** | 10,000+ 小时 |
| 完整训练 run (任何数据) | **0 次** | 无数次 |
| 端到端推理 (真实环境) | **0 次** | 产品级部署 |
| 已知能收敛的证据 | **无** | 论文 + 开源权重 |

这不是"差一些", 这是**质的鸿沟**。一个从未训练过的模型的所有架构"优势"都是纸上谈兵。三频率 Mamba 可能在实际训练中梯度爆炸; 96 个潜在标记的 grounder 可能学不出有意义的物体表示; 混合一致性损失可能让两个头互相拖累而非互补。**在没有实验数据之前, 这些都只是假设。**

OpenPI 的"简单"方案 (无时序, 无 grounder, 3B 参数, 30K 步) 之所以有资格被称为"足够好", 是因为它被证明了。HybridVLA 的"复杂"方案之所以不能被称为"更好", 是因为它没被证明。

### 3.2 参数效率: 一个被忽视的劣势

| | HybridVLA v2 | OpenPI | HybridVLA 代价 |
|--|-------------|--------|---------------|
| 可训练参数 | ~2.3B | ~500M | **4.6× 更多** |
| 冻结参数 (GPU 显存) | 7.6B | ~2B | **3.8× 更多** |
| 训练步数 | 150K-400K | 30K | **5-13× 更多** |
| GPU 需求 | 8×H100-80GB | 少得多 | **显存门槛高** |
| 单 chunk 推理延迟 (估算) | ~68ms | ~20ms | **3.4× 更慢** |

HybridVLA 用了 OpenPI 3-5 倍的计算资源, 但**没有任何证据表明这些额外资源带来了回报**。在资源有限的研究环境下, 这种设计选择需要非常强的理由。

### 3.3 过度工程化的具体证据

**ActionHistoryEncoder (4L Mamba @ 2048d, ~120M 参数)**:
编码 8 个 14 维动作向量。输入总共 8×14=112 个浮点数。用 120M 参数的 4 层 Mamba 处理这 112 个数, 参数/数据比超过 100 万:1。一个 2 层 MLP (~50K 参数) 大概率就够了。OpenPI 没有动作历史编码, 证明这不是必要的。

**Consistency Loss (3 组件)**:
- ContrastiveTemporalLoss: InfoNCE on fused_states — 鼓励相邻时间步表示相似。但 Mamba 的递归本质**已经**让相邻输出高度相关, 这个 loss 可能在推梯度进入一个 trivial 解 (所有 fused_state 趋于常量)。
- SlowFastAgreementLoss: slow_token ≈ EMA(fast_tokens) — 但 slow stream 更新频率就是低于 fast, 它**不应该**等于 fast 的 EMA。这个约束的物理含义值得质疑。
- ActionConsistencyLoss: 余弦相似度在 256d 空间。14 维动作投影到 256 维再做余弦 — 信息瓶颈在 14→256 的投影, 而非余弦约束。这是一个**弱到几乎无用**的约束。

这三个 loss 组合起来增加了代码复杂度, 但每一个都有理论上的疑问。OpenPI 不需要这些辅助 loss。

**世界模型 (1,130 行, enable=false)**:
ImaginationEngine + ObjectPhysicsEngine + StochasticState + VisualDecoder + SubgoalPlanner + 5 个相关文件。写了但从未开启, 从未测试, 不在任何训练阶段的代码路径中。这不是"远见", 是**死代码**。

### 3.4 forward_train 的计算效率问题

`hybrid_vla_v2.py:429-474` — 时间步循环:

```python
for t in range(T):  # T=24 (sequence_window)
    # 每个 t: ActionHistoryEncoder forward (4L Mamba)
    action_history_token = self.action_history_encoder.encode(action_history_buf.get())
    # 每个 t: TriRateMambaCore forward (36L Mamba total, 分三流)
    temporal_out = self.temporal_core(...)
```

这意味着单次 forward_train 要执行 **T=24 次 temporal_core forward**。每次 temporal_core forward 至少经过 fast stream (20L), 可能加 medium (6L) 和 slow (10L)。在 official Mamba2 路径下, 每个 stream 还会退化为 **逐 token 的 Python for-loop** (`_MambaStack.forward()` L443-449), 因为需要保存 SSM 状态:

```python
for t in range(x.shape[1]):        # L=33 tokens
    for i, layer in enumerate(self.layers):  # 20 layers (fast stream)
        x_t, ssm_states_list[i], ... = layer.step(x_t, ...)
```

**一次训练 forward 中, fast stream 的 layer.step() 被调用 24 × 33 × 20 = 15,840 次。** 这是 Python-level 循环调用 CUDA kernel, 每次调用的 kernel launch overhead 累积可达数百毫秒。

OpenPI 的 forward 是**一次 Transformer forward pass**, 无时间步循环, 无 Python-level per-token dispatch。

### 3.5 训练 token-by-token loop 带来的隐性风险

`_MambaStack.forward()` 的 official 路径 (L432-454) 用 `layer.step()` 逐 token 调用。这个 workaround 是为了捕获 SSM 状态, 但它引入了一个严重问题: **激活检查点在此路径下可能不生效**。

看 fallback 路径 (L460-468):
```python
if use_checkpoint and self.training:
    x, s, c = activation_checkpoint(layer, x, s_i, c_i, use_reentrant=False)
```

但 official 路径 (L432-454) **没有 use_checkpoint 分支** — 它直接在 for loop 中调用 `layer.step()`。这意味着用 official Mamba2 库时, **activation checkpointing 对 temporal_core 的三个 Mamba stack 不起作用**。考虑到这三个 stack 有 ~800M 参数, 这对显存影响巨大。

### 3.6 FAST 头: 名不副实

叫 "FAST" 但跟 pi-0-FAST 的机制完全不同:
- pi-0-FAST: 自回归, 逐 token 生成, 维度间有序列依赖, 利用 KV-cache
- HybridVLA FAST: 单次 MLP, 一次输出 [B, 24, 14, 512], 无维度间依赖

这不是"不同的设计选择", 这是**用了同一个名字的不同东西**。如果要声称 FAST 是辅助信号, 那它不应该叫 FAST — 这会误导读者以为它有 pi-0-FAST 的能力。

### 3.7 RTC 的 train-infer 不一致

训练时 (`hybrid_vla_v2.py:594-602`):
```python
with torch.no_grad():
    prev_chunk = self.action_expert.sample(
        cond_prefix=cond_prefix,  # ← 当前步的 cond_prefix
        ...
    )
```

推理时 (`hybrid_vla_v2.py:770-789`): prev_chunk_tail 来自**上一次** chunk 生成, 用的是**上一次观测**的 cond_prefix。

训练时前序块和当前块用**同一个** cond_prefix (同一帧观测), 推理时用**不同帧观测**。这种分布偏移意味着 RTC 的 overlap inpainting 在推理时的效果可能远低于训练时, 甚至可能引入伪影。

---

## 4. 相对 OpenPI 的真实优势

在上面指出了大量问题后, 需要公平地说: HybridVLA 确实有一些架构想法是 OpenPI 没有的, 其中一部分在学术上有价值。

### 4.1 确有价值的设计 (但需要实验验证)

| 设计 | 学术价值 | 为什么需要验证 |
|------|---------|---------------|
| **三频率时序分离** | 高 — 从物理第一性原理出发 | 36L Mamba 可能过参数化; 频率分配 (20/6/10) 是拍脑袋的; slow stream 每 ~8 步才更新一次, 信息可能过度陈旧 |
| **分层槽压缩 (48→24)** | 中高 — Perceiver 变体 | 48 个原始槽在小数据集上可能学不出有意义的物体表示; 压缩可能丢弃关键信息 |
| **Midpoint ODE** | 中 — 低成本 2 阶精度 | 这是一个纯数学改进, 无争议地比 Euler 好, 但推理延迟翻倍 |
| **Per-module LR** | 中 — 好的工程实践 | 合理且没有风险 |
| **AdaRMSNorm gate bias** | 中 — 防激活塌陷 | 来自 pi-0.5, 是借鉴不是创新 |

### 4.2 不确定是否有价值的设计

| 设计 | 风险 |
|------|------|
| 多尺度视觉特征 (FPN-style) | Qwen2-VL-7B 的最终层已有强语义, 提前层特征可能引入噪声 |
| 混合离散+连续动作 | 两个头互相约束, 但也互相干扰。OpenPI 故意分开做 (pi-0 连续, pi-0-FAST 离散) |
| StaleTimeEncoding | 概念巧妙, 但增加了正弦编码 → MLP → 注意力调制的计算路径, 可能过度复杂 |
| Phase/Affordance/Uncertainty tokens | 需要弱监督标签 (phase_labels, affordance_labels), 数据集可能没有 |

### 4.3 本轮新增的工程优势 (修复后)

| 优势 | 实际意义 |
|------|---------|
| EMA summon_full_params 方案 | PyTorch FSDP + EMA 的正确实现, 有参考价值 |
| 推理 EMA 自动发现 | 好的 UX 设计 |
| FSDP use_orig_params=True | 正确的选择, 解决了参数名匹配问题 |

---

## 5. 仍存在的全部问题

### 5.1 致命级 (不解决则项目无意义)

| # | 问题 | 说明 |
|---|------|------|
| **F-1** | **零验证** | 从未在任何数据上完成过一次完整训练。所有架构讨论都是纸上谈兵。 |
| **F-2** | **计算效率未知** | token-by-token Python loop 的实际训练速度未知。可能慢到无法在合理时间内完成 150K 步。 |
| **F-3** | **显存可行性未验证** | 9.9B 参数 + 36L Mamba SSM 状态 × T=24 + activation checkpointing 在 official 路径可能失效 → 8×H100-80GB 是否够用完全未知。 |

### 5.2 阻塞级

| # | 问题 |
|---|------|
| **B-1** | FASTER 推理 NotImplementedError — Stage C 训练的 FASTER 在推理时无法使用 |
| **B-2** | RTC train-infer 分布不一致 — inpainting 约束在推理时可能失效 |
| **B-3** | Official Mamba2 路径下 activation checkpointing 不生效 — 显存可能爆 |

### 5.3 工程级

| # | 问题 |
|---|------|
| **E-1** | 无 CI/CD, 无 linter |
| **E-2** | config.py `eval()` 安全隐患 |
| **E-3** | 1,130 行世界模型死代码 |
| **E-4** | train_stage_a.py 278 行重复代码 |
| **E-5** | Grounder/TriRate Mamba 无单元测试 (最复杂的两个组件) |
| **E-6** | Phase/Affordance heads 需要标签, 但数据集可能没有 → 训练时这些 loss 可能为 0 或乱标签 |
| **E-7** | ContrastiveTemporalLoss 可能推向 trivial 解 |
| **E-8** | ActionHistoryEncoder 过度参数化 (120M 参数处理 112 个浮点数) |
| **E-9** | 单 benchmark (LIBERO only), 无通用 Policy 抽象 |
| **E-10** | 无推理服务化能力 |

---

## 6. 综合评价

### 6.1 评分 (严格标准)

评分标准: **"如果明天要用这个系统做真实机器人实验, 它能给你多少信心?"**

| 维度 | HybridVLA v0.10.9 | OpenPI | 说明 |
|------|-------------------|--------|------|
| 架构创新性 | 7.5 | 6.0 | 有创新想法, 但部分过度工程化, 且创新价值完全未被验证 |
| 训练管线代码正确性 | 7.5 | 9.0 | 修了 P0 bug, 但这些 bug 本身说明管线从未被实际使用; OpenPI 已被无数次使用 |
| 训练管线实际可用性 | **2.0** | **9.5** | HybridVLA 从未完成过一次训练; OpenPI 日常使用 |
| 推理管线 | 4.0 | 9.0 | 有代码但从未在真实环境跑过; EMA/clipping 修复后理论正确, 但缺 FASTER 推理 |
| 参数效率 | 3.0 | 8.0 | 4.6× 可训练参数, 5-13× 训练步数, 3.4× 推理延迟, 零证据表明有回报 |
| 验证深度 | **0.5** | **10.0** | 0 小时 vs 10,000+ 小时 |
| 代码质量 | 6.0 | 8.0 | 类型注解好, 配置清晰, 但无 CI/linter, 有死代码, 部分设计可疑 |
| 测试覆盖 | 4.0 | 7.5 | 972 行测试, 但最复杂的组件 (Grounder, TriRate Mamba) 无测试 |
| 部署就绪 | 2.0 | 9.0 | 无服务化, 单 benchmark, 无多机器人 |
| **综合 (加权)** | **4.0** | **8.5** | **差距 4.5 分** |

### 6.2 为什么综合分是 4.0 而不是之前的 7.1

之前的评分犯了一个错误: **把代码层面的"正确性"等同于系统层面的"可用性"**。

一个从未训练过的模型, 无论代码写得多正确, 它的实际价值接近零。FSDP 死锁修了, 但如果没人跑过训练, "死锁修了"和"死锁没修"的区别只是理论上的。

OpenPI 的代码可能没有 per-module LR, 没有 Midpoint ODE, 没有 StaleTimeEncoding — 但它**被证明能在真实机器人上工作**。这一条就压过所有纸面上的优势。

### 6.3 公平的类比

HybridVLA v0.10.9 类似于一份**精心设计的建筑蓝图** — 图纸画得好, 承重计算做了, 材料清单列了, 甚至连施工流程 (三阶段训练) 都规划了。但地基还没挖。

OpenPI 是一栋**已经建成并住了人的楼** — 也许外观没那么花哨, 但它立着, 有人住, 经受住了风雨。

在蓝图变成建筑之前, 讨论"哪栋楼更好"是没有意义的。

---

## 7. 建议 (优先级严格排序)

### 第一优先: 证明它能工作

1. **LIBERO 500 步单卡验证** — 确认 loss 能下降, 无 NaN, GPU 显存够用
2. **LIBERO 5K 步单卡验证** — 确认收敛趋势
3. **LIBERO 15K 步 Stage A → B 验证** — 确认跨阶段检查点加载正确, expert 能学到东西

如果这三步中任何一步失败, 后面的所有工作都不重要。

### 第二优先: 降低风险

4. 验证 official Mamba2 路径下的 activation checkpointing 是否生效 (如果不生效, 显存可能爆)
5. 测量实际训练速度 (steps/sec) — 如果 token-by-token loop 太慢, 可能需要放弃 official Mamba2 用 fallback 路径
6. 实现 FASTER 推理 (消除最后一个 NotImplementedError)

### 第三优先: 清理

7. 删除 world_model/ (1,130 行死代码)
8. 删除 train_stage_a.py
9. 添加 ruff + pre-commit
10. 修复 config.py `eval()`

---

## 附录: 代码量分布

| 模块 | 行数 | 状态 |
|------|------|------|
| `vla_hybrid_v2/models/` | 2,948 | 核心, 从未在真实数据上跑过 |
| `vla_hybrid_v2/world_model/` | 1,132 | **死代码** |
| `vla_hybrid_v2/data/` | 1,220 | 理论上完整, 从未被 10+ episode 数据测试 |
| `vla_hybrid_v2/losses/` | 128 | 部分 loss 设计可疑 (见 §3.3) |
| `vla_hybrid_v2/infer/` | 437 | v0.10.9 修复后理论正确, 但 FASTER 推理缺失 |
| `vla_hybrid_v2/utils/` | 352 | v0.10.9 修复后 FSDP/EMA 正确 |
| `vla_hybrid_v2/` 其他 | 1,329 | config + types + ops |
| `scripts/` | 1,370 | 含 278 行重复的 train_stage_a.py |
| `tests/` | 972 | Grounder/Mamba 无覆盖 |
| `libero_hybrid/` | ~826 | 从未被实际训练/评估执行 |
| **总计** | **~10,714** | 其中 ~1,410 行是死代码或重复 (13%) |
