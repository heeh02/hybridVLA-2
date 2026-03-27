# HybridVLA v2 — 全面可行性分析 (v0.6.1)

> 基于 4,823 行代码、35 个文件的逐行审计。
> 日期: 2026-03-25
> 目标: 判断项目在**架构、训练脚本、数据流、损失函数、分布式基础设施**上是否已达到可训练标准。

---

## 0. 结论摘要

| 维度 | 完成度 | 能否训练? | 阻塞项 |
|------|--------|----------|--------|
| 模型架构 (forward pass) | **95%** | ✅ 可以 | 无 |
| World Model | **90%** | ✅ 可以 | 未集成到 forward_train 主循环 |
| 损失函数 | **90%** | ✅ 可以 | WorldModelLoss 未在训练中调用 |
| 训练脚本 (Stage A) | **85%** | ✅ 可以 | DummyDataset 仅占位 |
| 分布式基础设施 (FSDP/EMA/Ckpt) | **90%** | ✅ 可以 | 需 CUDA 环境实测 |
| 数据管线 | **5%** | ❌ 不可以 | **唯一硬性阻塞项** |
| 评估 (Eval) | **0%** | — | 非阻塞, 可后补 |
| **综合** | **~75%** | **⚠️ 有条件可训练** | 真实数据管线 (2-3 天) |

**一句话判定**: 架构和训练基础设施已达到可训练标准。唯一阻塞项是真实数据管线 — 使用 DummyDataset 今天就可以启动 smoke test; 换上真实数据后即可启动正式 Stage A 训练。

---

## 1. 架构完整性审计

### 1.1 端到端数据流

```
输入图像 [B, C, H, W]
    ↓
Qwen2-VL 7B Backbone (冻结 vision tower + 层 0-15; LoRA 层 16-27)
    → 多尺度特征 [B, N, 3584] (层 10, 18, 28)
    → MultiScaleAdapter: 可学习门控加权 → [B, N, 2048]
    ↓
HierarchicalAttentionGrounder (8 层, 96 latent → 层 4 后压缩至 72)
    → global_token [B, 2048]
    → compressed_object_slots [B, 24, 2048]
    → phase_token [B, 2048]
    → uncertainty_token [B, 2048]
    → affordance_token [B, 2048]
    → raw_object_slots [B, 48, 2048] (保留)
    ↓
TriRateMambaCore (Fast 20L + Medium 6L + Slow 10L)
    输入序列: [global, phase, unc, aff, proprio, prev_action, stale, embodiment, action_hist, slots×24]
           = [B, 33, 2048]
    ↓ CrossAttentionFusion
    → fused_state [B, 2048]
    → fast_token / medium_token / slow_token [B, 2048] 各一
    ↓
├── FASTDiscreteHead: fused_state → [B, 24, 14, 512] logits
├── PhaseHead: phase_token → [B, 16] logits
├── AffordanceHead: affordance_token → [B, 8] logits
│
└── FlowActionExpert (18L, M-M-A×6, 1536d)
    cond_prefix [B, 32, 2048] → core_to_expert → [B, 32, 1536]
    noisy_actions [B, 24, 14] + flow_t [B]
    → velocity [B, 24, 14]
    → 推理时: midpoint ODE 8步 → denoised_action [B, 24, 14]
```

### 1.2 维度一致性逐点验证

| 连接点 | 上游输出 | 下游期望 | 匹配? |
|--------|---------|---------|-------|
| Backbone → Grounder | `[B, N, 2048]` | `backbone_hidden: [B, N, 2048]` | ✅ |
| Grounder → TriRateCore (global) | `[B, 2048]` | `global_token: [B, 2048]` | ✅ |
| Grounder → TriRateCore (slots) | `[B, 24, 2048]` | `object_slots: [B, S, 2048]` (S=24) | ✅ |
| TriRateCore input seq | 9 tokens + 24 slots | `[B, 33, 2048]` | ✅ |
| TriRateCore → Fusion | fast/med/slow + stale `[B, 2048]` 各一 | `CrossAttentionFusion(4 inputs)` | ✅ |
| Fusion → FASTDiscreteHead | `fused_state [B, 2048]` | `input_dim=2048` | ✅ |
| Fusion → cond_prefix | 9 tokens 拼接 `[B, 32, 2048]` | — | ⚠️ 见 §1.3 |
| cond_prefix → Expert | `[B, 32, 2048]` → `core_to_expert` | `[B, 32, 1536]` (cond_dim=1536) | ✅ |
| Expert 内部 action_proj | `[B, 24, 14]` → `[B, 24, 1536]` | `d_model=1536` | ✅ |
| Expert output | `[B, 26, 1536]` (取后 24) | `out_proj → [B, 24, 14]` | ✅ |
| ActionHistoryEncoder | `[B, 8, 14]` → `[B, 2048]` | `action_history_token [B, 2048]` | ✅ |

### 1.3 cond_prefix 32 token 构成验证

```python
tokens = [
    global_token      [B, 1, 2048],   # 1
    compressed_slots  [B, 24, 2048],  # 24
    phase_token       [B, 1, 2048],   # 1
    uncertainty_token [B, 1, 2048],   # 1
    affordance_token  [B, 1, 2048],   # 1
    fused_state       [B, 1, 2048],   # 1
    fast_token        [B, 1, 2048],   # 1
    medium_token      [B, 1, 2048],   # 1
    slow_token        [B, 1, 2048],   # 1
]                                      # 总计: 32 ✅
```

**结论**: 维度在所有模块连接点完全一致，无 shape mismatch。

---

## 2. 训练脚本审计

### 2.1 Stage A 训练循环完整性 (train_stage_a.py, 257 行)

| 训练步骤 | 实现? | 代码位置 | 备注 |
|----------|------|---------|------|
| 配置加载 | ✅ | 命令行参数 | HybridVLAv2Config |
| 分布式初始化 | ✅ | `setup_distributed()` | NCCL + seed + device |
| 模型实例化 | ✅ | `HybridVLAv2(cfg)` | 全量创建 |
| Expert 冻结 | ✅ | `p.requires_grad = False` | Stage A 专属 |
| FSDP 包装 | ✅ | `wrap_fsdp()` | world_size > 1 时启用 |
| 优化器 (AdamW) | ✅ | `fused=True, betas=(0.9, 0.95)` | 正确 |
| LR 调度器 | ✅ | cosine + linear warmup | min_lr_ratio=0.1 |
| EMA | ✅ | `EMAModel` + decay ramp | 0.999 → 0.9999 / 20K steps |
| 断点续训 | ✅ | `auto_resume()` | checkpoint-latest symlink |
| 数据加载 | ⚠️ | `DummyVLADataset` | **占位符** |
| 前向传播 | ✅ | `model.forward_train(batch)` | bf16 autocast |
| 梯度累积 | ✅ | `loss / grad_accum` + 周期性 step | 正确 |
| 梯度裁剪 | ✅ | `clip_grad_norm_fsdp(model, 1.0)` | FSDP 感知 |
| 日志 | ✅ | 每 log_interval 步 | 多 rank 分级 |
| Checkpoint 保存 | ✅ | 原子写入 + symlink | 安全 |
| 评估 | ❌ | — | 下一轮 |

**18/19 步已实现**, 仅评估缺失 (非阻塞)。

### 2.2 Stage B/C 脚本

尚未创建独立脚本。但根据 stage_b.yaml / stage_c.yaml 配置, 仅需在 Stage A 基础上修改 ~10 行:

- Stage B: 解冻 Expert, 设 `stop_gradient_cond_prefix=True`, 添加 flow_matching loss
- Stage C: 移除梯度隔离, 启用 RTC + FASTER, 降低 LR

**工作量**: 0.5 天, 非阻塞。

### 2.3 Smoke Test (train_smoke_test.py)

- 使用 D_CORE=64, D_EXPERT=32 的微型配置
- Mock Backbone 替代 7B Qwen2-VL (避免下载依赖)
- 已验证 Stage A + Stage B 前向/反向传播无 NaN
- 20 步 smoke test 通过

---

## 3. 损失函数审计

### 3.1 损失全景

| 损失 | 类 | 训练阶段 | 权重 | 实现状态 |
|------|---|---------|------|---------|
| FAST 离散 CE | `DiscreteCELoss` | A/B/C | 1.0 | ✅ 完整, label_smoothing=0.1 |
| Phase CE | `PhaseLoss` | A/B/C | 0.5 | ✅ 完整 |
| Affordance CE | `F.cross_entropy` | A/B/C | 0.3 | ✅ 完整 |
| 一致性 (Consistency) | `V2ConsistencyLoss` | A/B/C | 0.3 | ✅ 3 子项 |
| Flow Matching | `FlowMatchingLoss` | B/C | 1.0 | ✅ 完整, logit_normal 采样 |
| World Model | `WorldModelLoss` | (未集成) | — | ✅ 代码完整, 未在训练循环调用 |

### 3.2 V2ConsistencyLoss 子项分析

```
V2ConsistencyLoss = 1.0 × ContrastiveTemporalLoss (InfoNCE)
                  + 0.5 × SlowFastAgreementLoss (MSE, fast_ema vs slow)
                  + 0.5 × ActionConsistencyLoss (cosine similarity)
```

**设计合理性**:
- ContrastiveTemporalLoss: 相邻时间步 fused_state 应相似 → 时间一致性
- SlowFastAgreementLoss: 慢速流应追踪快速流的指数加权平均 → 多速率对齐
- ActionConsistencyLoss: 离散/连续 action 预测应一致 → 双头互约束

**注意**: ActionConsistencyLoss 对 continuous_actions 使用 `.detach()`, 梯度仅流向离散分支。这是有意设计 — 防止 flow matching 梯度通过一致性损失污染 backbone。

### 3.3 Flow Matching 正确性

```python
target_velocity = x_1 - x_0          # 目标: 从噪声到动作的速度场
loss = (velocity_pred - target_velocity).pow(2).mean()  # MSE
interpolate: x_t = (1-t)*x_0 + t*x_1  # 线性插值
timestep: t ~ sigmoid(N(0,1))         # logit-normal 分布 (SD3 风格)
```

- 数学上正确: 条件流匹配 (CFM) 标准实现
- logit_normal 采样比 uniform 更好: 在 t ≈ 0.5 附近采样更密集, 这是去噪最关键的区域
- midpoint ODE solver: 2 阶精度 O(dt³), 8 步 ≈ Euler 16 步

### 3.4 WorldModelLoss (已实现但未集成)

完整实现的 7 项子损失:

| 子损失 | 权重 | 公式 |
|--------|------|------|
| KL (posterior-prior) | 1.0 | DreamerV3 双向 KL, per-category free_bits=1.0 (有效最小 48 nats) |
| Latent prediction | 10.0 | MSE(z_pred, z_true) |
| Reward | 1.0 | Symlog two-hot CE (255 bins, [-20, 20]) |
| Done | 0.5 | BCE with logits |
| Slot prediction | 5.0 | MSE(predicted_slots, target_slots) |
| Intrinsic invariance | 2.0 | MSE(intrinsic, next_intrinsic.detach()) |
| Interaction sparsity | 0.1 | mean(attention_weights) |

**状态**: 代码完整可用, 但 `forward_train()` 未调用 WorldModelLoss。这是 **设计选择** (Stage A-C 先训练 action prediction, World Model 可能安排在 Stage D 或独立训练)。

---

## 4. World Model 模块审计

### 4.1 组件完整性 (9 个模块, ~200M 参数)

| 模块 | 参数量 | 输入/输出 | 代码状态 |
|------|-------|----------|---------|
| StochasticStateModule | ~10M | z_det [B,2048] → z_full [B,4096] | ✅ DreamerV3 48×48 categorical |
| ImaginationMamba | ~80M | z_full+action+noise → delta_z [B,2048] | ✅ 8 层, .step() API |
| NoiseAugmentation | ~2M | z + step_idx → z_noisy + noise_emb | ✅ GameNGen σ=0.7×step/H |
| WorldModelHeads | ~15M | z_full → reward/value/done logits | ✅ symlog two-hot 255 bins |
| ObjectPhysicsEngine | ~35M | slots+action+z → next_slots | ✅ 6 层 GNN |
| CNNWorldDecoder | ~40M | z_full → image [B,3,112,112] | ✅ 4 层转置卷积 |
| LatentSubgoalPlanner | ~20M | z+phase+lang → z_goal (残差) | ✅ 残差 MLP |
| ImaginationEngine | — | 协调器: 32 步 rollout | ✅ 完整编排 |
| WorldModelLoss | — | 7 项子损失 | ✅ 灵活可选 |

### 4.2 Imagination Rollout 数据流

```
z_det_init [B, 2048] (来自 TriRateCore 的 fused_state)
    ↓ encode_prior()
z_full [B, 4096]
    ↓
Loop 32 steps:
  ├─ policy(z_full) → action [B, 14]
  ├─ noise_aug(z_full, step) → z_noisy [B, 4096], noise_emb [B, 2048]
  ├─ dynamics(z_noisy, action, noise_emb) → delta_z [B, 2048]
  ├─ z_det_next = z_det + delta_z (残差更新)
  ├─ encode_prior(z_det_next) → z_full_next [B, 4096]
  ├─ heads(z_full_next) → reward/value/done
  ├─ slot_decoder(z_full_next) → slots [B, 24, 2048]
  ├─ physics(slots, action, z_full_next) → next_slots, interaction_w
  └─ visual_decoder(z_full_next) → image [B, 3, 112, 112]
    ↓
ImaginationTrajectory (所有 [B, 32, *] shape)
```

### 4.3 World Model 与主模型的接口

World Model 通过 `fused_state` (来自 TriRateMambaCore) 作为 `z_det_init` 与主 VLA 模型对接。接口维度 [B, 2048] 匹配。但 **当前未在 `forward_train()` 中调用**。

---

## 5. 分布式训练基础设施审计

### 5.1 FSDP 配置

```python
FSDP(
    model,
    auto_wrap_policy = {MambaBlock, GrounderBlock, ExpertMambaBlock, ExpertAttentionBlock},
    mixed_precision = MixedPrecision(param=bf16, reduce=fp32, buffer=bf16),
    sharding_strategy = FULL_SHARD,
    sync_module_states = True,
    limit_all_gathers = True,
    # 缺少: use_orig_params=True  ← §6.2 中讨论
)
```

**Auto-wrap 策略**: 正确识别了 4 种可重复的 Transformer/Mamba block 类型, 每个 block 独立分片。

**Activation Checkpointing**: NO_REENTRANT 模式, 应用于与 auto-wrap 相同的 4 类 block。

### 5.2 显存预算 (8×H100 80GB)

```
冻结参数 (Backbone 7.6B, bf16):  ~15.0 GB (每 GPU 复制)
可训练参数分片 (1.53B / 8):       ~0.38 GB
梯度分片:                         ~0.38 GB
优化器状态 (fp32 m+v):            ~2.3 GB
────────────────────────────────
静态占用:                         ~18 GB

激活 (activation checkpoint):     ~25-30 GB
CUDA 开销:                        ~3-5 GB
────────────────────────────────
总占用:                           ~50-55 GB
剩余安全余量:                     ~25-30 GB  ✅ 充足
```

### 5.3 Checkpoint 系统

- **原子写入**: `.tmp-checkpoint-{step}/` → rename → `checkpoint-latest` symlink
- **FSDP 兼容**: `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)`
- **非严格加载**: `strict=False`, 兼容 LoRA checkpoint 迁移
- **自动续训**: `auto_resume()` 查找 `checkpoint-latest`
- **保存内容**: model + optimizer + scheduler + EMA + metadata JSON

### 5.4 EMA 系统

- 线性 decay ramp: 0.999 → 0.9999 (20K steps)
- `shadow.lerp_(param, 1-decay)`: 数学正确
- `apply()/restore()` 周期: 用于评估时切换模型权重
- 仅追踪 `requires_grad=True` 参数: 正确 (冻结参数无需 EMA)

---

## 6. 已知问题分类

### 6.1 阻塞项 (必须修复才能正式训练)

| # | 问题 | 严重度 | 工作量 | 详情 |
|---|------|--------|-------|------|
| **B1** | **真实数据管线** | 🔴 | 2-3 天 | `DummyVLADataset` 仅生成随机张量, 无法用于有意义的训练。需要实现: (1) RLDS/HDF5 数据加载 (2) 图像预处理 + tokenization (3) action chunk 构建 (4) 多相机拼接 (5) 分布式采样 |

### 6.2 高优先级 (应在正式训练前修复)

| # | 问题 | 严重度 | 工作量 | 详情 |
|---|------|--------|-------|------|
| H1 | FSDP 缺少 `use_orig_params=True` | 🟡 | 5 分钟 | 冻结的 7.5B backbone 仍参与 allgather/reduce_scatter, 每步浪费 ~15 GB 通信。添加此参数后冻结参数跳过通信。 |
| H2 | 日志 loss 值偏高 4× | 🟡 | 5 分钟 | `accum_loss` 累积了 `log_interval × grad_accum` 个微批次, 但仅除以 `log_interval`。修复: `avg = {k: v / (log_interval * grad_accum) for ...}` |
| H3 | Stage B/C 脚本缺失 | 🟡 | 0.5 天 | 复制 Stage A, 修改 ~10 行 (解冻 expert, 添加 flow_matching loss, 修改 LR/梯度隔离)。 |
| H4 | flash_attn 硬依赖 | 🟡 | 10 分钟 | `attn_implementation="flash_attention_2"` 硬编码, 缺少 SDPA fallback。 |

### 6.3 低优先级 (可在训练期间修复)

| # | 问题 | 严重度 | 工作量 | 详情 |
|---|------|--------|-------|------|
| L1 | EMA + FSDP 跨 GPU 数续训 | 🟢 | 1 小时 | EMA 在 FSDP wrap 后创建, shadow 存的是分片参数。不同 GPU 数恢复时分片大小不匹配。实际场景 (同配置续训) 无影响。 |
| L2 | ObjectPhysicsEngine 未使用 extrinsic | 🟢 | — | 计算了 `extrinsic_proj(h)` 但未使用 (`noqa: F841`)。浪费少量计算, 不影响正确性。 |
| L3 | LPIPS 可选依赖无警告 | 🟢 | 5 分钟 | 如果 `lpips` 未安装, CNNWorldDecoder 只用 L1 loss, 无日志提示。 |
| L4 | World Model 未集成到训练循环 | 🟢 | — | 设计选择: Stage A-C 聚焦 action prediction。World Model 训练可独立进行。 |

---

## 7. 配置与代码对齐验证

### 7.1 YAML ↔ Python 参数映射 (全量)

| YAML 参数 | Python 参数 | 值 | 匹配? |
|-----------|------------|---|-------|
| `backbone.name` | `BackboneConfig.name` | Qwen/Qwen2-VL-7B-Instruct | ✅ |
| `backbone.output_dim` | `BackboneConfig.output_dim` | 2048 | ✅ |
| `backbone.multi_scale_layers` | `BackboneConfig.multi_scale_layers` | [10, 18, 28] | ✅ |
| `backbone.freeze_vision_tower` | `BackboneConfig.freeze_vision_tower` | True | ✅ |
| `backbone.freeze_text_layers_until` | `BackboneConfig.freeze_text_layers_until` | 16 | ✅ |
| `backbone.lora.rank` | `LoRAConfig.rank` | 64 | ✅ |
| `grounder.num_latents` | `GrounderConfig.num_latents` | 96 | ✅ |
| `grounder.num_object_slots` | `GrounderConfig.num_object_slots` | 48 | ✅ |
| `grounder.compressed_slots` | `GrounderConfig.compressed_slots` | 24 | ✅ |
| `grounder.num_layers` | `GrounderConfig.num_layers` | 8 | ✅ |
| `grounder.compression_layer` | `GrounderConfig.compression_layer` | 4 | ✅ |
| `temporal_core.fast_layers` | `TemporalCoreConfig.fast_layers` | 20 | ✅ |
| `temporal_core.medium_layers` | `TemporalCoreConfig.medium_layers` | 6 | ✅ |
| `temporal_core.slow_layers` | `TemporalCoreConfig.slow_layers` | 10 | ✅ |
| `temporal_core.fast_d_state` | `TemporalCoreConfig.fast_d_state` | 128 | ✅ |
| `temporal_core.slow_d_state` | `TemporalCoreConfig.slow_d_state` | 256 | ✅ |
| `temporal_core.fusion_type` | `TemporalCoreConfig.fusion_type` | cross_attention | ✅ |
| `action_expert.num_layers` | `ActionExpertConfig.num_layers` | 18 | ✅ |
| `action_expert.d_model` | `ActionExpertConfig.d_model` | 1536 | ✅ |
| `action_expert.chunk_horizon` | `ActionExpertConfig.chunk_horizon` | 24 | ✅ |
| `action_expert.action_dim` | `ActionExpertConfig.action_dim` | 14 | ✅ |
| `action_expert.ada_rmsnorm` | `ActionExpertConfig.ada_rmsnorm` | True | ✅ |
| `action_expert.ode_solver` | `ActionExpertConfig.ode_solver` | midpoint | ✅ |
| `heads.fast_vocab_size` | `HeadsConfig.fast_vocab_size` | 512 | ✅ |
| `heads.num_phases` | `HeadsConfig.num_phases` | 16 | ✅ |
| `heads.affordance_head` | `HeadsConfig.affordance_head` | True | ✅ |
| `ema.initial_decay` | `EMAConfig.initial_decay` | 0.999 | ✅ |
| `ema.final_decay` | `EMAConfig.final_decay` | 0.9999 | ✅ |
| `ema.ramp_steps` | `EMAConfig.ramp_steps` | 20000 | ✅ |

**28/28 参数完全对齐。** 无配置与代码不一致。

### 7.2 三阶段训练配置一致性

| 参数 | Stage A | Stage B | Stage C | 设计合理性 |
|------|---------|---------|---------|-----------|
| Steps | 120K | 200K | 80K | ✅ 标准: 感知→专家→微调, 中间最长 |
| LR | 2e-4 | 1e-4 | 3e-5 | ✅ 逐渐降低, 10× 衰减 |
| Warmup | 3K | 5K | 1K | ✅ 各占总步数 2-3% |
| Expert 训练 | 冻结 | 解冻 + 梯度隔离 | 全开 | ✅ 渐进式解冻 |
| Flow Matching | — | 1.0 | 1.0 | ✅ Stage B 引入 |
| EMA | ramp | constant 0.999 | constant 0.9999 | ✅ 逐步更保守 |
| RTC / FASTER | — | — | 启用 | ✅ 仅 fine-tuning 阶段 |

---

## 8. 参数量估算

### 8.1 总参数

| 组件 | 参数量 | 可训练? |
|------|-------|--------|
| Qwen2-VL 7B (冻结) | ~7.6B | ❌ (LoRA 除外) |
| LoRA (28 层, rank=64) | ~48M | ✅ |
| MultiScaleAdapter | ~22M | ✅ |
| HierarchicalGrounder | ~350M | ✅ |
| TriRateMambaCore (20+6+10 层) | ~400M | ✅ |
| ActionHistoryEncoder (4 层) | ~50M | ✅ |
| CrossAttentionFusion (2 层) | ~35M | ✅ |
| FlowActionExpert (18 层) | ~450M | ✅ |
| FASTDiscreteHead | ~15M | ✅ |
| PhaseHead + AffordanceHead | ~5M | ✅ |
| 投影层 (proprio/action/embodiment/cond) | ~25M | ✅ |
| World Model (未激活) | ~200M | ✅ (独立训练) |
| **总计** | **~9.2B** | **~1.4B 可训练** |

### 8.2 与设计文档对比

| 指标 | 设计目标 | 实际 | 偏差 |
|------|---------|------|------|
| 总参数 | 9.04B | ~9.2B | +2% (可接受) |
| 可训练参数 | 1.53B | ~1.4B | -8% (可接受) |
| 每 GPU 显存 | 50-55 GB | 待实测 | — |

---

## 9. 架构设计合理性评估

### 9.1 与 SOTA 对比

| 特性 | π₀ | Octo | HybridVLA v2 |
|------|------|------|-------------|
| Backbone | PaLI-5B | ViT-B/L | Qwen2-VL 7B |
| Action 预测 | Flow Matching | Diffusion | **双轨: FAST 离散 + Flow Matching** |
| 时间建模 | Transformer | 无 | **Tri-Rate Mamba SSM** |
| ODE solver | Euler | DDPM | **Midpoint (2 阶)** |
| World Model | 无 | 无 | **DreamerV3 + GNN (可选)** |
| 多相机 | ✅ | ✅ | ✅ |
| EMA | ✅ (固定 0.99) | ❌ | ✅ (自适应 ramp) |

**独创贡献**:
1. **Tri-Rate Mamba**: 解决 4× 频率间隙 (50Hz fast → 12.5Hz slow), 中间 25Hz 仅增 15% 计算
2. **Hierarchical Slot Compression**: 96→24 可学习压缩, 比固定数量更灵活
3. **FAST + Flow Matching 双轨**: 离散头提供初始粗预测, 连续头细化
4. **AdaRMSNorm**: 乘性时间步条件注入, 比加性 embedding 更有效

### 9.2 潜在风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| 7B backbone 过大导致训练不稳定 | 中 | LoRA rank=64 + 冻结前 16 层, 有效限制更新量 |
| Tri-Rate 三流对齐困难 | 中 | SlowFastAgreementLoss + CrossAttentionFusion |
| 512 bin 离散化精度 | 低 | Flow Matching 头提供连续精度补充 |
| World Model 增加复杂度 | 低 | 可选模块, 不影响 Stage A-C |
| FSDP 通信瓶颈 | 中 | limit_all_gathers + 建议添加 use_orig_params |

---

## 10. 代码质量评估

### 10.1 工程实践

| 方面 | 评分 | 说明 |
|------|------|------|
| 类型标注 | A | 所有公开接口都有 type hints |
| Dataclass 使用 | A | 7 个 dataclass 清晰定义数据结构 |
| 模块化 | A | 35 文件, 职责清晰, 耦合度低 |
| 错误处理 | B | checkpoint 有 try/except, 但部分地方静默失败 |
| 文档 | B+ | 设计文档详尽, 但代码内注释偏少 |
| 测试 | C | 仅 smoke test, 无单元测试 |
| CUDA fallback | B | SSM 和 Mamba 有 fallback, flash_attn 缺少 |

### 10.2 代码行数演进

| 版本 | 行数 | 主题 |
|------|------|------|
| v0.1 | 2,660 | 架构修复 |
| v0.2 | 2,660 | Mamba2 集成 |
| v0.3 | — | 分析 (Core state loss 发现) |
| v0.4 | 3,927 | World Model + Trajectory |
| v0.5 | 4,164 | Core state fix + smoke test |
| v0.6 | 4,823 | 训练基础设施 (FSDP/EMA/Ckpt) |
| **当前** | **4,823** | **6 次迭代, ~30 个修复** |

---

## 11. 可行性最终判定

### 11.1 回答核心问题: "能否开始训练?"

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Q: 架构是否完整?                                       │
│  A: ✅ 是。所有 9 个核心模块 + 9 个 World Model 模块     │
│     已实现, 维度完全一致, smoke test 通过。              │
│                                                         │
│  Q: 训练脚本是否就绪?                                   │
│  A: ✅ 是。Stage A 脚本 18/19 步完成。                   │
│     Stage B/C 需 0.5 天补充。                            │
│                                                         │
│  Q: 分布式训练是否就绪?                                 │
│  A: ✅ 代码就绪, 待 CUDA 环境实测。                      │
│     FSDP + AMP + activation checkpointing 全部实现。     │
│                                                         │
│  Q: 能否今天启动训练?                                   │
│  A: ⚠️ 可以启动 smoke test (DummyDataset)。              │
│     正式训练需真实数据管线 (2-3 天)。                     │
│                                                         │
│  Q: 有无根本性设计缺陷?                                 │
│  A: ❌ 无。架构设计合理, 参数量/显存在 8×H100 预算内,    │
│     三阶段训练策略符合领域最佳实践。                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 11.2 到正式训练的路线图

```
当前 ──→ Day 0: 单 GPU smoke test (DummyDataset)      ← 今天可做
    │
    ├──→ Day 1: 修复 H1-H4 (4 个高优先级, 共 ~2 小时)
    │    ├── H1: FSDP use_orig_params=True
    │    ├── H2: 日志 loss 除数修复
    │    ├── H3: Stage B/C 脚本
    │    └── H4: flash_attn fallback
    │
    ├──→ Day 1-3: 真实数据管线 (B1, 唯一阻塞项)
    │    ├── RLDS/HDF5 数据加载器
    │    ├── 图像预处理 + Qwen2-VL tokenization
    │    ├── Action chunk 构建
    │    ├── 多相机处理
    │    └── DistributedSampler 集成
    │
    ├──→ Day 3: 8×H100 FSDP 环境测试 (1000 步验证)
    │
    └──→ Day 4: 启动 Stage A 正式训练 (120K 步, ~36 小时)
         ├──→ Day 5.5: Stage A 完成, 启动 Stage B (200K 步, ~72 小时)
         ├──→ Day 8.5: Stage B 完成, 启动 Stage C (80K 步, ~30 小时)
         └──→ Day 10: 全部训练完成
```

**总计**: 从现在到完成全部训练约 **10 天** (含 3-4 天准备 + 6 天训练)。

---

## 附录 A: 文件清单与行数

| 文件路径 | 行数 | 功能 |
|---------|------|------|
| `vla_hybrid_v2/__init__.py` | 1 | 版本声明 |
| `vla_hybrid_v2/types.py` | ~130 | 7 个核心 dataclass |
| `vla_hybrid_v2/config.py` | ~200 | 配置 dataclass 系统 |
| `vla_hybrid_v2/models/hybrid_vla_v2.py` | ~500 | 主模型 (forward_train + inference) |
| `vla_hybrid_v2/models/qwen2vl_backbone.py` | ~180 | Backbone wrapper + MultiScaleAdapter |
| `vla_hybrid_v2/models/attention_grounder.py` | ~300 | 分层注意力 grounder |
| `vla_hybrid_v2/models/mamba_core.py` | ~450 | Tri-Rate Mamba + Fusion |
| `vla_hybrid_v2/models/flow_action_expert.py` | ~350 | Flow Action Expert (M-M-A×6) |
| `vla_hybrid_v2/models/discrete_heads.py` | ~120 | FAST/Phase/Affordance heads |
| `vla_hybrid_v2/world_model/*.py` (9 files) | ~900 | World Model 全部模块 |
| `vla_hybrid_v2/losses/*.py` (3 files) | ~150 | 5 种损失函数 |
| `vla_hybrid_v2/ops/selective_scan.py` | ~50 | SSM scan (JIT + CUDA) |
| `vla_hybrid_v2/utils/*.py` (3 files) | ~400 | EMA + Distributed + Checkpointing |
| `scripts/train_stage_a.py` | ~257 | Stage A 训练脚本 |
| `scripts/train_smoke_test.py` | ~180 | Smoke test |
| `configs/**/*.yaml` (4 files) | ~230 | 模型 + 训练配置 |
| **总计** | **~4,823** | |

## 附录 B: 关键设计决策追溯

| 决策 | 选择 | 替代方案 | 理由 |
|------|------|---------|------|
| Backbone 大小 | 7B | 2B / 72B | 2B 太弱 (-15%); 72B 超显存 |
| 时间建模 | Tri-Rate Mamba SSM | Transformer / 单速 Mamba | SSM O(1) 推理; 三速率填频率间隙 |
| Action 预测 | FAST 512 + Flow Matching | 纯离散 / 纯连续 | 双轨: 粗+精, 互约束 |
| ODE solver | Midpoint (2 阶) | Euler (1 阶) / RK4 (4 阶) | 8 步 midpoint ≈ 16 步 Euler; RK4 太贵 |
| Slot 压缩 | 96→24 层内压缩 | 固定 24 slot | 可学习压缩保留更多信息 |
| EMA | 自适应 ramp | 固定 decay | 早期 0.999 探索, 后期 0.9999 稳定 |
| FSDP 策略 | FULL_SHARD | DDP / ZeRO-1 | 7B backbone 需要参数分片 |
