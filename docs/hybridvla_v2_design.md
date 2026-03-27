# HybridVLA v2 — 激进架构升级设计文档

> 日期: 2026-03-25
> 硬件目标: 8 × H100 80GB SXM
> 核心: 从 v1 的保守设计全面升级到充分利用 8×H100 算力的激进设计

---

## 1. v1 的六大瓶颈

### 1.1 Backbone 瓶颈: 2B 太小

v1 使用 Qwen2-VL-2B (1536d, 28 层)。对比:
- π₀ 的 PaliGemma: Gemma-2B (2048d) + SigLIP So400m = ~2.5B
- π₀ 的特征维度 2048 比我们的 1536 宽 33%

更关键的是 v1 只 LoRA fine-tune 最后 8 层 (rank=32)。这意味着前 20 层的表征完全是预训练的通用 VLM 特征，没有任何机器人领域的适应。

**v1 还缺少多尺度特征**——只用 last_hidden_state。物体边界、抓取点等精细空间信息在早期层更丰富。

### 1.2 Grounder 瓶颈: 32 latents 不够

16 个 object slots 在桌面操作（4-8 个物体）中勉强够用，但在厨房场景（15+ 物体、餐具、食材）或仓库拣选中严重不足。4 层处理深度也偏浅——Perceiver 文献中 6-8 层是常见配置。

### 1.3 Temporal Core 瓶颈: 双频率间隔太大

50 Hz (fast) 和 12.5 Hz (slow) 之间存在 4× 的频率间隙。v1 用一个简单的 sigmoid 门来融合——这等于让模型用一个标量在"完全信任 fast"和"完全信任 slow"之间做二选一，无法表达"对这个维度信任 fast、对那个维度信任 slow"的细粒度融合。

d_state=64 是保守的。Mamba 文献中 d_state=128-256 显著提升序列建模质量，代价是仅微小的计算增加（SSM scan 对 d_state 几乎线性）。

### 1.4 Expert 瓶颈: 弱于 π₀

v1 的 12 层 1024d expert 约 152M 参数，而 π₀ 的 Gemma-300M expert 有 18 层 1024d (311M)。更关键的差距是 **timestep conditioning**——v1 只用加法嵌入 `action_tokens + t_emb`，而 π₀.5 证明了 AdaRMSNorm (乘法调制) 对 flow matching 质量至关重要。

### 1.5 缺失功能

| 功能 | π₀ | v1 | 影响 |
|------|-----|-----|------|
| 多相机 | 3 cameras × 256 tokens | 单相机 | 丧失多视角空间推理 |
| EMA | 0.99 集成 | 定义了未集成 | 5-15% 动作质量损失 |
| 动作历史 | 无 (但 chunk 覆盖长) | 无 | 缺少短期动作上下文 |
| 2nd-order ODE | — | Euler (1st-order) | 需要更多步才达同等精度 |

### 1.6 参数浪费

v1 在 8×H100 上每 GPU 仅用 ~11 GB / 80 GB。这意味着 86% 的显存被浪费。8×H100 集群的能力远未被发挥。

---

## 2. v2 解决方案总览

### 2.1 总体升级路线

| 维度 | v1 | v2 | 升级倍数 |
|------|-----|-----|---------|
| Backbone | Qwen2-VL-2B (1536d) | **Qwen2-VL-7B** (3584d→2048d) | 3.5× 参数 |
| Grounder | 32 latents, 4L, 1536d | **96 latents → 24 compressed**, 8L, 2048d | 3× latents, 2× 深度 |
| Temporal Core | Dual-Rate 12+6L, d_state=64 | **Tri-Rate 20+6+10L**, d_state 128/128/256 | 2× 深度, 4× 状态 |
| Expert | 12L 1024d, Euler | **18L 1536d**, AdaRMSNorm, midpoint | 2.3× 参数 |
| Fusion | Scalar sigmoid gate | **Cross-attention** (2L, 8 heads) | 质变 |
| chunk_horizon | 16 | **24** | 1.5× |
| cond_tokens | 24 | **32** | 1.33× |
| FAST bins | 256 | **512** | 2× 精度 |
| 多相机 | ✗ | **✓ (3 cameras)** | 新功能 |
| EMA | ✗ | **✓ (decay schedule)** | 新功能 |
| 动作历史 | ✗ | **✓ (4L Mamba, K=8)** | 新功能 |
| 物理 affordance | ✗ | **✓ (affordance head)** | 新功能 |

### 2.2 参数规模对比

| 组件 | v1 参数 | v2 参数 | 说明 |
|------|---------|---------|------|
| Backbone (总) | 2,200M | **7,600M** | 7B vs 2B |
| Backbone LoRA | 8M | **90M** | rank 64 on all 28 layers |
| Grounder | 57M | **~200M** | 96 latents, 8L, 2048d, compression |
| Fast Mamba | 142M | **~330M** | 20L, 2048d, d_state=128 |
| Medium Mamba | — | **~100M** | 6L, 2048d, d_state=128 (新) |
| Slow Mamba | 71M | **~200M** | 10L, 2048d, d_state=256 |
| Fusion | ~7M | **~35M** | Cross-attention 2L |
| Action History Encoder | — | **~65M** | 4L Mamba, 2048d (新) |
| Flow Expert | 152M | **~350M** | 18L 1536d, AdaRMSNorm |
| FAST Head | 2.3M | **~8M** | 512 bins, 2048d input |
| Phase Head | 0.6M | **~2M** | 16 classes, 2048d |
| Affordance Head | — | **~2M** | 新增 |
| Projections/Misc | 18M | **~50M** | 2048→1536 等 |
| StaleTimeEncoding | ~5M | **~8M** | 2048d |
| Embodiment Emb | ~0.5M | **~0.5M** | 同 |
| **总参数** | **~2,600M** | **~9,040M** | 3.5× |
| **总冻结** | **~2,160M** | **~7,510M** | Backbone frozen 部分 |
| **总可训练** | **~440M** | **~1,530M** | 3.5× |

### 2.3 8×H100 内存适配

```
Per GPU (FSDP full-shard):
  冻结参数 (bf16 replicated):     ~15.0 GB  (7.5B × 2B / 1 — 部分 offload)
  可训练参数 shard:               ~0.38 GB  (1530M × 2B / 8)
  梯度 shard:                     ~0.38 GB
  优化器 shard (fp32 m+v+master): ~2.3 GB   (1530M × 12B / 8)
  ─────────────────────────────────────
  静态总计:                       ~18 GB

  激活 (checkpointed, bs=2):      ~25-30 GB
  CUDA overhead:                  ~3-5 GB
  ─────────────────────────────────────
  总使用:                         ~50-55 GB
  剩余:                           ~25-30 GB (安全余量)
```

**结论: 完全在 80 GB 限制内**，且有充足余量用于增大 batch 或加入世界模型。

---

## 3. 每个模块的设计理由

### 3.1 为什么 7B 而不是 2B 或 72B？

- **2B → 7B**：3.5× 参数提供显著更强的视觉-语言理解。Qwen2-VL-7B 在多模态 benchmark 上比 2B 高 15-20%。
- **为什么不 72B**：72B 需要 ~144 GB bf16，即使 FSDP 8 卡 shard 后也需要 18 GB/GPU 仅存参数，加上优化器和激活后超出 80 GB 限制。
- **7B 的 3584d 输出过宽**：通过 MultiScaleAdapter 投影到 2048d，这是 Mamba 和 Expert 的效率甜点。

### 3.2 为什么 Tri-Rate 而不是更简单的 Dual-Rate？

50 Hz 和 12.5 Hz 之间的 4× 间隙意味着 fast stream 在两次语义刷新之间独自运行 ~4 步而没有新的语义信息注入。Medium stream (25 Hz) 将这个间隙缩短到 2 步，显著提高了对中频变化（如物体被推动后的位移）的响应速度。

**计算开销评估**：
- Medium stream (6L) vs Fast stream (20L)：参数仅增加 30%
- 但只运行 50% 的步数：实际计算增加 ~15%
- 换来的是更精细的时间建模

### 3.3 为什么 AdaRMSNorm 是必要的？

Flow matching 在不同噪声水平 t 下需要完全不同的处理策略：
- t ≈ 0（接近纯噪声）：需要大幅度修正
- t ≈ 1（接近干净信号）：只需微调

加法嵌入 `x + t_emb` 无法改变特征的分布——它只能平移。AdaRMSNorm 的 `(1+scale) * x_norm + shift` 可以缩放和平移特征分布，让模型在每个噪声水平使用不同的特征动态范围。

π₀.5 的消融实验证明这带来了显著的动作质量提升。

### 3.4 为什么 midpoint 而不是 Euler？

Midpoint (2nd-order Runge-Kutta):
```
v1 = f(x, t)
x_mid = x + 0.5*dt*v1
v2 = f(x_mid, t+0.5*dt)
x_next = x + dt*v2
```

与 Euler 相比：
- 每步 2 次前向传播 vs 1 次
- 但局部截断误差 O(dt³) vs O(dt²)
- 8 步 midpoint ≈ 16 步 Euler 的精度
- 总前向传播次数相同 (8×2 = 16×1)，但 midpoint 更准确

### 3.5 层次化 Grounder 压缩的意义

48 个 raw object slots → 24 个 compressed slots 的过程本质上是一个**learned attention pooling**。它让模型学会：
- 合并相似物体（如一排杯子→"一组杯子"）
- 过滤无关物体（如背景物体→被忽略）
- 聚焦任务相关物体（如"要拿的杯子"→保留高分辨率 slot）

这比固定 24 个 slots 更灵活——初始的 48 个 slots 允许每个物体都被检测到，压缩阶段只保留对当前任务重要的物体。

---

## 4. 训练策略

| 阶段 | 步数 | 学习率 | 训练组件 | 壁钟时间 (8×H100) |
|------|------|--------|---------|-------------------|
| A | 120K | 2e-4 | LoRA + Grounder + Tri-Rate + Heads | ~36h |
| B | 200K | 1e-4 | + Expert (知识隔离) + EMA | ~72h |
| C | 80K | 3e-5 | 全部 + RTC/FASTER | ~30h |
| **总计** | **400K** | | | **~138h ≈ 5.75 天** |

**Stage B 最长**因为 Expert 需要从随机初始化收敛，且 EMA 需要足够步数积累稳定的权重。

**Global batch size = 64**：2 per GPU × 8 GPUs × 4 grad_accum。这比 v1 的 32 更大，能提供更稳定的梯度估计。

---

## 5. v1 → v2 的预期性能提升

| Benchmark | v1 预测 | v2 预测 | 提升来源 |
|-----------|---------|---------|---------|
| LIBERO-Spatial | ~75% | ~85% | 7B backbone + 多相机 + EMA |
| LIBERO-Object | ~70% | ~82% | 48 object slots + affordance |
| LIBERO-Long | ~72% | ~83% | Tri-Rate + 动作历史 + 更长 chunk |
| Calvin (ABC→D) | ~2.5 chains | ~3.5 chains | 更强的泛化（7B + LoRA 全层） |

**最大的提升来源是 7B backbone + EMA + AdaRMSNorm 的组合**——这三者各自贡献 3-5%，组合效果可能达 10-15%。

---

## 6. 文件结构

```
hybridVLA_2/
  vla_hybrid_v2/
    __init__.py
    config.py                              # 全部 v2 配置
    types.py                               # TriRateTemporalState 等
    models/
      __init__.py
      hybrid_vla_v2.py                     # 主模型组装
      qwen2vl_backbone.py                  # 7B + multi-scale + multi-camera
      attention_grounder.py                # 层次化 96→24 压缩
      mamba_core.py                        # Tri-Rate (20+6+10) + CrossAttention fusion
      flow_action_expert.py                # 18L AdaRMSNorm + midpoint ODE
      discrete_heads.py                    # FAST 512-bin + Phase 16 + Affordance
    losses/
      flow_matching.py
      discrete_loss.py                     # + label smoothing
      consistency_loss.py                  # contrastive + slow-fast + action
    ops/
      selective_scan.py                    # JIT SSM scan
    data/                                  # multi-camera dataset
    infer/                                 # tri-rate runtime loop
    utils/                                 # EMA + distributed
  configs/
    model/v2_qwen2vl_7b_trirate_expert18.yaml
    train/stage_{a,b,c}.yaml
```

---

*v2 是 v1 的全面升级：从 2.6B/440M trainable 到 9.0B/1.5B trainable，充分利用 8×H100 的计算和显存预算。每一个模块都基于 π₀、DreamerV3、DIAMOND 等 SOTA 方法的经验教训进行了有针对性的强化。*
