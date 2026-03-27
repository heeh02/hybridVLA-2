# HybridVLA v2 — 三阶段训练数据流闭环分析

> **日期**: 2026-03-27
> **标准**: 追踪每个 tensor 从产生到消费的完整路径, 验证三阶段是否真正打通
> **方法**: 逐行代码追踪 + 精确 tensor shape 标注 + 梯度流验证

---

## 1. Stage A 完整数据流

### 1.1 数据产生: HDF5 → Sample Dict

```
HDF5 episode (≥47 steps)
  ├─ data/actions[start : start+47]           → raw_actions [47, 14]
  ├─ data/robot0_joint_pos[start : start+24]  → raw_proprio [24, 14]
  ├─ data/images/agentview_rgb[start]         → primary_image (PIL 448×448)
  ├─ data/images/agentview_rgb[start+0,6,12,18] → refresh_images ×4
  └─ attrs/language_instruction               → lang (str)

归一化: normalize(raw_actions) → [47, 14]
分块:   action_chunks[t] = norm[t : t+24]   → [24, 24, 14]  (每个 chunk 24 个真实动作)
位移:   prev_actions[1:] = norm[:23]         → [24, 14]
```

**`hdf5_adapter.__getitem__` 返回** (`hdf5_adapter.py:392-457`):
```
sample = {
  "input_ids":           [256]              # Qwen2-VL tokenized text+image
  "attention_mask":      [256]
  "pixel_values":        [N, patch_dim]     # N_patches 固定 (448² resize)
  "image_grid_thw":      [N_img, 3]
  "actions":             [24, 24, 14]       # [T, H, A]
  "proprio":             [24, 14]           # [T, P]
  "prev_actions":        [24, 14]           # [T, A]
  "embodiment_id":       scalar
  "num_cameras":         1
  "refresh_input_ids":         [4, 256]     # [R, L]
  "refresh_attention_mask":    [4, 256]
  "refresh_pixel_values_list": List[Tensor] ×4
  "refresh_image_grid_thw_list": List[Tensor] ×4
}
```

### 1.2 Collate → Batch

`collate.py:vla_collate_fn` — 固定 shape tensor stack, 视觉 tensor safe_stack, list transpose:

```
batch = {
  "input_ids":           [B, 256]
  "attention_mask":      [B, 256]
  "pixel_values":        [B, N, patch_dim]    # safe_stack 保证 shape 一致
  "image_grid_thw":      [B, N_img, 3]
  "actions":             [B, 24, 24, 14]      # [B, T, H, A]
  "proprio":             [B, 24, 14]
  "prev_actions":        [B, 24, 14]
  "embodiment_id":       [B]
  "num_cameras":         [B]
  "refresh_input_ids":         [B, 4, 256]
  "refresh_attention_mask":    [B, 4, 256]
  "refresh_pixel_values_list": [[B,N,D], [B,N,D], [B,N,D], [B,N,D]]  # len=R=4
  "refresh_image_grid_thw_list": [[B,N_img,3], ...]
}
```

### 1.3 forward_train (Stage A)

```
┌─────────────────── forward_train() ───────────────────┐
│                                                        │
│  ① _validate_batch: shape 校验 [B,T,H,A] etc.         │
│                                                        │
│  ② Refresh schedule: stride=6 → steps=[0,6,12,18]     │
│     refresh_map: {0-5→0, 6-11→1, 12-17→2, 18-23→3}   │
│     medium_set: {0,2,4,...,22}                         │
│                                                        │
│  ③ Backbone + Grounder ×4 (per refresh point):         │
│     for r in [0,1,2,3]:                                │
│       backbone.forward_semantic(                       │
│         refresh_input_ids[:,r],     [B, 256]            │
│         refresh_attention_mask[:,r], [B, 256]           │
│         refresh_pixel_values_list[r], [B, N, D]        │
│         num_cameras=1                                  │
│       ) → backbone_out["last_hidden_state"] [B, L, 2048]│
│       grounder(backbone_hidden) → GrounderOutput:      │
│         global_token          [B, 2048]                │
│         compressed_slots      [B, 24, 2048]            │
│         phase_token           [B, 2048]                │
│         uncertainty_token     [B, 2048]                │
│         affordance_token      [B, 2048]                │
│                                                        │
│  ④ Temporal loop × T=24 steps:                         │
│     for t in 0..23:                                    │
│       inputs = [grounder_out[refresh_map[t]],          │
│                 proprio[:,t], prev_actions[:,t],        │
│                 embodiment_token, action_history,       │
│                 stale_encoding]                         │
│       → temporal_core(                                 │
│           33 tokens: [9 special + 24 compressed slots] │
│           state, semantic_refresh, medium_update        │
│         )                                              │
│       → fused_state [B, 2048]                          │
│                                                        │
│  ⑤ Stack: fused_states [B, 24, 2048]                   │
│           fast_tokens  [B, 24, 2048]                   │
│                                                        │
│  ⑥ Losses (Stage A — 无 FM):                           │
│     loss_fast:       FAST head(fused_states[B*T,D])    │
│                      → logits [B*T, 24, 14, 512]       │
│                      CE vs discretized actions          │
│                      全 T 步监督 ✓                      │
│     loss_phase:      Phase head(grounder.phase_token)  │
│                      → [B, 16] per step, 全 T 步 ✓     │
│     loss_affordance: Aff head(grounder.aff_token)      │
│                      → [B, 8] per step, 全 T 步 ✓      │
│     loss_consistency: InfoNCE + SlowFast (无 Action)    │
│     loss_fm:         ✗ 不计算 (expert 冻结)             │
│                                                        │
│  ⑦ loss_total = sum(losses)                            │
└────────────────────────────────────────────────────────┘
```

### 1.4 Stage A 梯度流

```
loss_fast (×1.0)  ──→ fast_head ──→ fused_states ──→ temporal_core ──→ grounder ──→ backbone LoRA
loss_phase (×0.5) ──→ phase_head ──→ grounder.phase_token ──→ grounder ──→ backbone LoRA
loss_aff (×0.3)   ──→ aff_head ──→ grounder.aff_token ──→ grounder ──→ backbone LoRA
loss_cons (×0.3)  ──→ consistency_loss ──→ fused_states/fast_tokens/slow_token ──→ temporal_core

expert: ✗ 冻结, 无梯度
cond_builder: ✗ 冻结, 无梯度
```

**Optimizer groups** (`train_unified.py:344-374`):
```
backbone_decay:   LoRA weights,   LR = 2e-4 × 0.1 = 2e-5
backbone_nodecay: LoRA biases,    LR = 2e-5, WD=0
core_decay:       grounder/core/heads weights, LR = 2e-4
core_nodecay:     biases,         LR = 2e-4, WD=0
(expert_* groups: 不存在 — expert 冻结)
```

### 1.5 Checkpoint 内容

```
outputs/v2_stage_a/checkpoint-120000/
  model.pt      ← 全部模块权重 (含 frozen expert 的随机 init 权重)
  optimizer.pt  ← backbone + core param groups 的 AdamW state
  scheduler.pt  ← cosine schedule 状态
  ema.pt        ← (如果 enabled) shadow dict
  meta.json     ← {"step": 120000, "epoch": N, "stage": "a"}
```

---

## 2. Stage A → B 转换

### 2.1 Checkpoint 加载

```
Stage B 启动:
  model = HybridVLAv2(cfg)              # 全新模型, 所有参数随机 init
  configure_trainable_modules("b", cfg)  # 冻结全部 → 解冻 LoRA + core + expert + cond_builder
  sanity_check("b")                      # assert expert_trainable == expert_total ✓
  model.to(device)
  wrap_fsdp(model)
  optimizer = AdamW(...)                 # 新 optimizer, 含 expert group
  load_checkpoint("stage_a/checkpoint-latest", model, strict=False)
    ↓
  FSDP context: FULL_STATE_DICT
  model.load_state_dict(state, strict=False)
    ↓
  结果: backbone LoRA = Stage A 训练值 ✓
        grounder = Stage A 训练值 ✓
        temporal_core = Stage A 训练值 ✓
        heads = Stage A 训练值 ✓
        expert = Stage A 的随机 init 值 (checkpoint 中有, 但从未训练)
        cond_builder = Stage A 的随机 init 值
```

**关键**: `strict=False` 不会报 missing key, 因为 Stage A checkpoint **包含全部模块** (含 frozen expert)。所有 key 都匹配。

### 2.2 Stage B 新增行为

**Optimizer** 新增 expert group (`train_unified.py:354-356`):
```python
elif name.startswith("action_expert"):
    group = "expert"
    lr_scale = cfg.train.expert_lr_scale  # 0.5
```
→ Expert LR = 1e-4 × 0.5 = **5e-5**

**forward_train Stage B 分支** (`hybrid_vla_v2.py:528-572`):
```
if stage != "a":  ← Stage B 进入此分支

  cond_prefix = _build_cond_prefix(grounder_out[-1], temporal_out[-1])
    ← 组成: global(1) + compressed_slots(24) + phase(1) + uncertainty(1)
            + affordance(1) + fused(1) + fast(1) + medium(1) + slow(1)
    ← 32 tokens @ D_core=2048
    ← cond_builder transform → core_to_expert project → [B, 32, 1536]

  cond_prefix = cond_prefix.detach()  ← stop_gradient_cond_prefix=true
                                       ← 梯度阻断: FM → expert ✓, FM → backbone ✗

  flow_t = sample_timestep(B)          ← logit_normal: sigmoid(randn())
  noise = randn_like(target_actions)   ← [B, 24, 14]
  noisy_actions = (1-t)*noise + t*target ← 线性插值

  expert_out = action_expert(
    noisy_actions, flow_t,
    cond_prefix (detached),
    proprio_for_expert, emb_for_expert
  )  → velocity [B, 24, 14]

  loss_fm = MSE(velocity, target - noise) × 1.0

  expert_continuous = noisy_actions + (1-t)*velocity  ← 去噪恢复
  loss_consistency = consistency_loss(
    fused_states, fast_tokens, slow_token,
    discrete_actions=fast_continuous,        ← FAST head 软预测
    continuous_actions=expert_continuous.detach()  ← expert 预测 (detached)
  ) × 0.3
```

### 2.3 Stage B 梯度流

```
loss_fast (×1.0)  ──→ fast_head ──→ temporal_core ──→ grounder ──→ backbone LoRA
loss_phase (×0.5) ──→ phase_head ──→ grounder ──→ backbone LoRA
loss_aff (×0.3)   ──→ aff_head ──→ grounder ──→ backbone LoRA
loss_cons (×0.3)  ──→ InfoNCE + SlowFast ──→ temporal_core
                  ──→ ActionConsistency ──→ consistency_loss params (投影层)
                      (discrete_actions 有梯度 → fast_head → temporal)
                      (continuous_actions detached → expert 不受此 loss 影响)
loss_fm (×1.0)    ──→ action_expert (18L, 1536d) ──→ ✓ 梯度更新
                  ──→ proprio_to_expert, emb_to_expert ──→ ✓ 梯度更新
                  ──✗ cond_prefix DETACHED → grounder/backbone 不受 FM 影响

Stage B Optimizer:
  backbone_decay:   LR = 1e-4 × 0.1 = 1e-5  (来自离散头 loss)
  core_decay:       LR = 1e-4                 (来自离散头 loss)
  expert_decay:     LR = 1e-4 × 0.5 = 5e-5   (来自 FM loss)
  expert_nodecay:   LR = 5e-5, WD=0           (来自 FM loss)
```

---

## 3. Stage B → C 转换

### 3.1 Checkpoint 加载

同 A→B 模式: `load_checkpoint("stage_b/checkpoint-latest", model, strict=False)`
- 所有模块权重从 Stage B 继承 (含训练过的 expert)

### 3.2 Stage C 新增行为

**额外解冻** (`train_unified.py:148-155`):
```python
if stage == "c":
    for name, p in model.backbone.named_parameters():
        for layer_idx in range(16, 28):  # text layers 16-27
            if f"layers.{layer_idx}." in name:
                p.requires_grad = True
```

**Stage C 梯度流** — 与 Stage B 相同, 额外:
```
loss_fast  ──→ ... ──→ backbone LoRA ──→ backbone text 16-27 (新增 ✓)
loss_phase ──→ ... ──→ backbone LoRA ──→ backbone text 16-27 (新增 ✓)
loss_fm    ──→ expert ──✗ cond_prefix.detach() (仍然阻断)
```

backbone text 16-27 **仅从离散头 loss 接收梯度, 不从 FM loss 接收**。

---

## 4. 发现的问题

### F1: RTC/FASTER 在 forward_train 中未实现 — 中等

**证据**: `grep "rtc\|faster\|RTC\|FASTER" hybrid_vla_v2.py` → 仅 line 14 docstring。

`stage_c.yaml:29-39` 定义了:
```yaml
rtc:
  enable: true
  execution_horizon: 8
  overlap_ratio: 0.333
faster:
  enable: true
  near_ratio: 0.3
```

但 `forward_train()` 零引用。**Stage C 训练行为 = Stage B + 更多可训练参数 + 更低 LR。** RTC/FASTER 是死配置。

**影响**: Stage C 能跑, 但缺少论文声称的训练增强。如果 RTC 是训练时 chunk overlap 采样 (如 ACT 论文), 当前未实现意味着 Stage C 的动作预测没有时间一致性约束。

---

### F2: Stage C cond_prefix 仍然 detach — 需确认设计意图

**证据**: `stage_c.yaml:26`: `stop_gradient_cond_prefix: true`

`hybrid_vla_v2.py:536-538`:
```python
if (self.cfg.train.stop_gradient_cond_prefix
        or self.cfg.train.block_fm_to_backbone):
    cond_prefix = cond_prefix.detach()
```

Stage C 中 `stop_gradient_cond_prefix=true` → detach 生效 → FM 梯度不到 grounder。

**两种解读**:
- **有意**: 保护 backbone 在 Stage C 不被 FM loss 破坏, 只通过离散头 loss 精调
- **无意**: 遗漏, 应改为 `stop_gradient_cond_prefix: false` 实现真正 end-to-end

**当前代码行为**: backbone text 16-27 只从 FAST/Phase/Affordance 更新, expert 只从 FM 更新, 二者通过 cond_prefix 的前向连接但梯度隔离。

---

### F3: Expert 冷启动 — 设计选择

Stage A checkpoint 中 expert 权重 = 模型创建时的随机 init。Stage B expert 从零开始学习。

**这不是 bug** — 类似 BERT 中在预训练后随机初始化分类 head 再微调。expert 的 `cond_prefix` 输入来自 Stage A 训练好的 grounder/core, 提供了丰富的感知特征, expert 只需学习从这些特征到动作的映射。

---

### F4: EMA 跨阶段行为 — 正确

`train_unified.py:410`: `load_checkpoint(path, model, strict=False)` — 不传 `ema`。

EMA 在每个 stage 重新创建 (`train_unified.py:384-395`), 从当前模型权重初始化 shadow dict。跨阶段不继承 EMA, 这是正确的 (不同 stage 训练不同参数集, 旧 EMA 不适用)。

---

### F5: Optimizer 动量丢失 — 预期行为

Cross-stage 不加载 optimizer state。Stage B 从零动量开始。标准做法 — 不同 LR 和参数组配置下, 旧动量无意义。warmup_steps=5000 缓解冷启动。

---

## 5. 结论: 三阶段是否打通

### 打通的部分 ✅

| 链路 | 状态 | 证据 |
|------|:----:|------|
| Stage A 数据→模型→损失→梯度→参数更新 | ✅ | 4 路 loss 全步监督, 梯度到 LoRA+grounder+core |
| Stage A checkpoint 保存 | ✅ | FSDP FULL_STATE_DICT, 含全部模块 |
| A→B checkpoint 加载 | ✅ | strict=False, FSDP context, key 完全匹配 |
| Stage B expert 解冻 + 训练 | ✅ | configure + sanity_check + FM loss backward |
| Stage B cond_prefix.detach() | ✅ | FM→expert ✓, FM→backbone ✗ (设计意图) |
| B→C checkpoint 加载 | ✅ | 同 A→B 模式 |
| Stage C backbone 部分解冻 | ✅ | text 16-27 从离散头 loss 接收梯度 |
| Eval loop | ✅ | val split 独立, evaluate() 定期运行 |

### 未打通/有问题的部分

| 链路 | 状态 | 严重性 | 影响 |
|------|:----:|:------:|------|
| **F1: RTC/FASTER 训练** | ❌ 配置存在但代码未实现 | 中 | Stage C 无时间一致性约束 |
| **F2: Stage C FM→grounder 梯度** | ⚠️ detach 阻断, 可能非预期 | 中 | "Full fine-tune" 不含 FM→backbone |
| F3: Expert 冷启动 | ✅ 设计选择 | 低 | 标准做法, 无需修改 |
| F4: EMA 跨阶段 | ✅ 正确 | 低 | 每 stage 重新初始化 |
| F5: Optimizer 动量 | ✅ 预期 | 低 | warmup 缓解 |

### 最终判定

> **Stage A 和 Stage B 的训练链路已完全打通。** 数据流从 HDF5 读取到梯度更新, 每个 tensor 的 shape 和梯度路径都可追踪。A→B 的 checkpoint 过渡正确。
>
> **Stage C 功能上打通但存在两个设计缺口:**
> 1. RTC/FASTER 为死配置 — Stage C 训练行为等价于 "低 LR Stage B + 更多可训练参数"
> 2. cond_prefix.detach() 阻止 FM loss 训练 grounder — "full fine-tune" 只适用于离散头路径
>
> 这两个缺口**不阻塞训练** (Stage C 能正常运行), 但影响训练质量和论文叙事。

### 建议

| 优先级 | 动作 | 工作量 |
|--------|------|--------|
| **训练前确认** | F2: 确认 Stage C `stop_gradient_cond_prefix` 是否应改为 false | 1 行 YAML |
| 训练后 | F1: 实现 RTC chunk overlap 训练增强 (如需要) | ~100 行 |
| 文档 | F3: 记录 expert 冷启动设计决策 | README 补充 |
