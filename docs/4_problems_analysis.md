# HybridVLA v2 — 四项关键问题深度分析

> **分析日期**: 2026-03-29（修订 2026-03-30）
> **代码版本**: v1.1.0 (commit 9a362ec)
> **分析范围**: expert 多时刻监督、WindowSample 强类型化、validation loop 落地、LR scale 接入 optimizer
>
> **修订说明**: v2 修正 Expert 多时刻监督的优先级判断。初版将其列为 P1，经深入论证（动作块 95.8% 重叠、与 pi-0 SOTA 一致、Stage B detach 设计）后下调至 P3，作为训练后 ablation 选项。

---

## 目录

1. [Expert 多时刻监督](#1-expert-多时刻监督)
2. [WindowSample 强类型化](#2-windowsample-强类型化)
3. [Validation Loop 落地](#3-validation-loop-落地)
4. [LR Scale 真正接入 Optimizer](#4-lr-scale-真正接入-optimizer)
5. [总结与优先级](#5-总结与优先级)

---

## 1. Expert 多时刻监督

### 1.1 现状

当前 flow matching expert 的监督信号 **仅来自窗口最后一个时刻 (t=-1)**，而 FAST 离散头、Phase 头、Affordance 头 **已在全部 T 个时刻计算损失**。这是一个有意的设计选择（README 中标注"due to its computational cost"），但造成了梯度密度的不对称。

**关键代码位置**：

| 组件 | 文件 | 行号 | 监督范围 |
|------|------|------|---------|
| Flow expert | `hybrid_vla_v2.py` | 570-622 | **仅 t=-1** |
| FAST discrete | `hybrid_vla_v2.py` | 528-545 | 全部 T（向量化 B*T） |
| Phase head | `hybrid_vla_v2.py` | 546-556 | 全部 T（逐步循环） |
| Affordance head | `hybrid_vla_v2.py` | 558-568 | 全部 T（逐步循环） |

**Expert t=-1 only 的核心代码** (`hybrid_vla_v2.py:570-622`)：

```python
# ---- Stage-gated action expert (t=-1 only) ----
target_actions = batch["actions"][:, -1]          # [B, H, A]  ← 仅取最后时刻
flow_t = self.flow_matching_loss.sample_timestep(B, device)
noise = torch.randn_like(target_actions)
noisy_actions = self.flow_matching_loss.interpolate(noise, target_actions, flow_t)
expert_out = self.action_expert(
    noisy_actions=noisy_actions, flow_t=flow_t,
    cond_prefix=cond_prefix, ...
)
loss_fm = self.flow_matching_loss(expert_out.velocity, noise, target_actions, flow_t)
```

**对比：FAST discrete 的全 T 向量化** (`hybrid_vla_v2.py:528-545`)：

```python
BT = B * T
fast_logits_flat = self.fast_head(fused_states.reshape(BT, -1))  # [B*T, H, A, V]
fast_targets_flat = FASTDiscreteHead.discretise_actions(
    batch["actions"], ...
).reshape(BT, chunk_horizon, -1)                                   # [B*T, H, A]
losses["loss_fast"] = self.discrete_loss(fast_logits_flat, fast_targets_flat)
```

### 1.2 数据结构支持

数据管线 **已经为多时刻监督提供了完整的数据**：

- `batch["actions"]` 形状为 `[B, T, H, A]`，其中每个时刻 `t` 都有独立的 H 步动作块
- HDF5 adapter 构建时读取 `T + H - 1` 个原始动作，为每个窗口位置 `t` 生成 `actions[t] = raw[t:t+H]`
- `cond_prefix` 可以从 `grounder_outputs[t]` 和 `temporal_outputs[t]` 为每个时刻独立构建

换句话说，**数据端不需要任何改动**，唯一需要改的是 `forward_train()` 中 expert loss 的计算逻辑。

### 1.3 实现方案对比

#### 方案 A：逐步循环（类似 Phase/Affordance）

```python
losses_fm = []
for t_sup in range(T):
    target_t = batch["actions"][:, t_sup]                    # [B, H, A]
    cond_t = self._build_cond_prefix(grounder_outputs[refresh_map[t_sup]],
                                      temporal_outputs[t_sup])
    flow_t = self.flow_matching_loss.sample_timestep(B, device)
    noise_t = torch.randn_like(target_t)
    noisy_t = self.flow_matching_loss.interpolate(noise_t, target_t, flow_t)
    expert_out_t = self.action_expert(noisy_t, flow_t, cond_t, ...)
    losses_fm.append(self.flow_matching_loss(expert_out_t.velocity, noise_t, target_t, flow_t))
losses["loss_fm"] = torch.stack(losses_fm).mean()
```

- **优点**：实现简单，梯度密度 T 倍提升
- **缺点**：T 次 expert forward pass，计算量 ~T 倍（T=20 → 20x）

#### 方案 B：向量化（类似 FAST）

```python
BT = B * T
targets_flat = batch["actions"].reshape(BT, H, A)
cond_flat = all_cond_prefixes.reshape(BT, C, D)
flow_t = self.flow_matching_loss.sample_timestep(BT, device)
noise_flat = torch.randn_like(targets_flat)
noisy_flat = self.flow_matching_loss.interpolate(noise_flat, targets_flat, flow_t)
expert_out = self.action_expert(noisy_flat, flow_t, cond_flat, ...)
losses["loss_fm"] = self.flow_matching_loss(expert_out.velocity, noise_flat, targets_flat, flow_t)
```

- **优点**：GPU 并行效率高
- **缺点**：显存 ~T 倍（expert 18 层 × 1536d × B*T，可能超出 80GB）

#### 方案 C：随机采样 K 个时刻（推荐折中）

```python
K = min(cfg.train.expert_supervision_steps, T)   # e.g., K=4
t_indices = torch.randperm(T)[:K]
losses_fm = []
for t_sup in t_indices:
    # ... same as 方案 A but only K iterations
losses["loss_fm"] = torch.stack(losses_fm).mean()
```

- **优点**：计算量可控（K 倍而非 T 倍），仍覆盖多个时间上下文
- **缺点**：引入采样方差，需调 K

### 1.4 影响评估

| 指标 | 当前 (t=-1) | 全 T (方案 A/B) | 采样 K=4 (方案 C) |
|------|------------|----------------|-------------------|
| Expert 梯度密度 | 1x | T× (~20×) | 4× |
| 计算量 / step | 基线 | ~T× | ~4× |
| 显存 (bs=2, 8GPU) | ~50-55 GB | 可能溢出 | ~65-70 GB |
| 有效 batch size | B | B×T | B×K |
| 训练时间 / epoch | 基线 | ~5-6× | ~2× |
| 实现复杂度 | — | 低 | 中（需配置项） |

### 1.5 再审视：真的需要多时刻 expert 监督吗？

初版分析将此问题列为 P1。经深入论证后，结论修正为 **当前阶段不需要实现**。

#### 支持"维持 t=-1"的论点

**1. 架构设计意图本身就是 t=-1**

Flow matching expert 的职责是：给定当前条件前缀（cond_prefix），去噪生成一个 24 步动作块。推理时 expert 只在最新状态下调用，训练用 t=-1 与推理对齐。这不是偶然的简化，而是有意的设计选择。

FAST/Phase/Affordance 头是轻量 MLP（几万参数），全 T 监督的增量计算几乎为零。Expert 是 18 层 1536d 的重型网络（~350M 参数），两者的全 T 代价差 3-4 个数量级。将它们类比是不恰当的。

**2. 窗口内动作块高度重叠——多时刻信号近乎冗余**

当前配置 `sequence_window (T) = 24`，`chunk_horizon (H) = 24`。相邻时刻的动作块：

```
actions[t]   = raw[t   : t+24]
actions[t+1] = raw[t+1 : t+25]
重叠 = 23/24 = 95.8%
```

多时刻监督提供的不是"T 倍更多的信息"，而是"T 个 95.8% 重叠的信号"。有效信息增益远低于 T 倍。相邻时刻的 cond_prefix 也高度相似（同一抓取动作的连续帧），梯度方向近乎平行。

**3. pi-0 等 SOTA 方法同样是单时刻监督**

pi-0（Physical Intelligence）的 flow matching expert 在单个时刻计算 loss。这不是 HybridVLA 的独特缺陷，而是 flow matching action expert 的标准做法。没有已发表的工作证明多时刻 expert 监督能显著提升性能。

**4. Stage B 的 cond_prefix.detach() 进一步削弱了多时刻的意义**

Stage B 配置为 `stop_gradient_cond_prefix: true` + `block_fm_to_backbone: true`，expert 梯度不回传到 Temporal Core。在这种设计下，多时刻监督的唯一收益是"expert 自身看到更多条件变体"。但由于动作块 95.8% 重叠，这些"变体"几乎相同。Stage C 虽然放开了梯度，但只有 80K 步，此时 expert 已从 Stage B 的 200K 步中充分学习。

**5. 梯度密度不足可以通过更廉价的手段补偿**

- 增大 batch size（通过梯度累积，已有 `no_sync()` 优化）
- 调高 `expert_lr_scale`（当前 0.5，可尝试 0.7-1.0）
- 延长 Stage B 训练步数
- 以上都不增加单步计算量

**6. 计算代价不容忽视**

即使方案 C（K=4），训练时间 ~2×，显存逼近 80GB 上限。Stage B 的 200K 步 × 2 = 多出数天 H100 时间，而收益不确定。

#### 支持"需要"的论点

**1. Temporal Core 中间状态缺少 expert 级梯度反馈**

FAST 头在全 T 监督下将梯度回传到 `fused_states` 的每个时刻，帮助 Temporal Core 学习有意义的中间表示。Expert 梯度只从 t=-1 的 cond_prefix 回传（Stage B 还 detach 了）。Temporal Core 可能为离散头优化而非为 expert 优化。

**反驳**：这是一个真实的理论风险，但 FAST 头和 expert 都监督同一种目标（动作预测），方向一致。consistency loss 进一步对齐两者。且 Stage C 端到端微调时梯度会重新流通。

**2. 一致性损失只在 t=-1 对齐**

如果 FAST 在 t=0..T-2 学到的表示与 expert 在 t=-1 学到的不一致，consistency loss 无法弥合。

**反驳**：同上，这是理论风险。需要实际 loss 曲线来判断是否触发。

### 1.6 修正结论

**当前阶段不需要实现多时刻 expert 监督。优先级从 P1 下调至 P3。**

正确的策略是数据驱动：

1. 先用 t=-1 跑完 Stage A → B（~320K 步）
2. 观察 `loss_fm` 曲线：
   - 正常下降 → 不需要多时刻监督
   - 停滞/震荡 → 先尝试调 `expert_lr_scale`（0.7-1.0）和增大有效 batch size
   - 以上无效 → 再考虑 K=2~4 的多时刻监督作为 ablation
3. 多时刻监督作为 **ablation 实验选项**而非默认配置

**风险等级**：低 — 当前设计与 SOTA 一致，无需预优化

---

## 2. WindowSample 强类型化

### 2.1 现状

`WindowSample` 在 `schema.py` 中定义为一个 **完整的 dataclass**，但在整个数据管线中 **从未被实例化**。所有 adapter 返回的是 **plain dict**，collate 函数处理的也是 dict，`forward_train()` 接收的仍然是 `Dict[str, Any]`。

**WindowSample dataclass 定义** (`schema.py:16-51`)：

```python
@dataclass
class WindowSample:
    """One training window from an episode."""
    actions: Tensor          # [T, H, A]
    proprio: Tensor          # [T, P]
    prev_actions: Tensor     # [T, A]
    input_ids: Tensor        # [L]
    attention_mask: Tensor   # [L]
    pixel_values: Optional[Tensor] = None
    image_grid_thw: Optional[Tensor] = None
    refresh_input_ids: Optional[Tensor] = None
    # ... 更多可选字段
    phase_labels: Optional[Tensor] = None
    affordance_labels: Optional[Tensor] = None
    step_weights: Optional[Tensor] = None
```

**实际 adapter 返回值** (`hdf5_adapter.py:401-466`)：

```python
def __getitem__(self, idx: int) -> dict:    # ← 返回类型是 dict，不是 WindowSample
    sample = {
        "input_ids": primary_tok["input_ids"],
        "actions": action_chunks,
        "proprio": norm_proprio,
        # ...
    }
    return sample                            # ← plain dict
```

### 2.2 数据流中的类型状态

```
HDF5Adapter.__getitem__() → dict         # 无类型检查
        ↓
vla_collate_fn(List[dict]) → dict        # 按 key 遍历 + stack，无校验
        ↓
training loop: batch = dict              # Dict[str, Any]
        ↓
model.forward_train(batch: Dict) →       # _validate_batch() 做运行时形状检查
```

| 阶段 | 类型 | 校验 |
|------|------|------|
| adapter 输出 | `dict` | 无 |
| collate 输入 | `List[dict]` | 无（按 key 自动 stack） |
| collate 输出 | `dict` | 无 |
| forward_train 入口 | `Dict[str, Any]` | `_validate_batch()` 运行时检查 |

### 2.3 现有运行时校验

`_validate_batch()` (`hybrid_vla_v2.py:297-378`) 是 **唯一的校验点**，检查：

- 5 个必需 key 存在性 (`actions`, `proprio`, `prev_actions`, `input_ids`, `attention_mask`)
- `actions` 4D 形状 `[B, T, H, A]`，H = chunk_horizon，A = action_dim
- `proprio` 3D 形状 `[B, T, P]`，P = proprio_dim
- T 一致性（actions、proprio、prev_actions 的 T 维度匹配）
- vision 字段共现（`pixel_values` 和 `image_grid_thw` 必须同时有或无）
- `step_weights` 形状 `[B, H]`
- `embodiment_id` 范围检查

### 2.4 风险分析

**Schema 常量（`BATCH_REQUIRED_KEYS` 等）定义了但从未被使用来做校验**——纯粹是文档。

| 风险 | 严重性 | 示例 | 何时发现 |
|------|--------|------|---------|
| adapter 中 key 拼写错误 | **高** | `"proprioceptive"` vs `"proprio"` | `forward_train()` 崩溃 |
| 可选字段缺失导致静默 fallback | **中** | 忘记设 `num_cameras` → 默认 1 | 永远不会发现（静默错误） |
| refresh 字段只添加了部分 | **中** | 添加了 `refresh_input_ids` 但漏了 `refresh_pixel_values_list` | 运行时才发现（如果走到那个分支） |
| 形状维度顺序错误 | **中** | `[T, A, H]` vs `[T, H, A]` | `_validate_batch()` 拦截 |
| IDE 无法提供自动补全 | **低** | `sample["actons"]` 拼写错误无提示 | 运行时 |

### 2.5 改进方案

#### 方案 A：强制返回 WindowSample 实例

```python
# adapter 改为：
def __getitem__(self, idx: int) -> WindowSample:
    return WindowSample(
        actions=action_chunks,
        proprio=norm_proprio,
        ...
    )

# collate 改为：
def vla_collate_fn(samples: List[WindowSample]) -> Dict[str, Any]:
    batch = {}
    for field in dataclasses.fields(WindowSample):
        values = [getattr(s, field.name) for s in samples]
        # ... stack logic
    return batch
```

- **优点**：构造时即校验字段存在性和类型，IDE 自动补全
- **缺点**：需改动所有 adapter + dummy dataset + 测试 fixture

#### 方案 B：TypedDict（静态类型检查 + 运行时兼容）

```python
class WindowSampleDict(TypedDict, total=False):
    actions: Tensor
    proprio: Tensor
    prev_actions: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    pixel_values: Optional[Tensor]
    # ...

def __getitem__(self, idx: int) -> WindowSampleDict:
    ...
```

- **优点**：dict 用法不变，mypy/pyright 可静态检查
- **缺点**：运行时无校验（除非加 typeguard），Optional 字段处理不如 dataclass 自然

#### 方案 C：在 collate 中添加前置校验（最小改动，推荐）

```python
def vla_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 前置校验
    for i, s in enumerate(samples):
        missing = BATCH_REQUIRED_KEYS - s.keys()
        if missing:
            raise ValueError(f"Sample {i} missing keys: {missing}")
    # ... 原有 stack 逻辑
```

- **优点**：改动最小（~10 行），利用已有的 `BATCH_REQUIRED_KEYS` 常量
- **缺点**：仅校验 key 存在性，不检查类型/形状

### 2.6 结论

**WindowSample 是一个"文档型" dataclass——定义了但从不实例化**。推荐分两步改进：

1. **短期（方案 C）**：在 `collate.py` 顶部加前置校验，利用已有的 `BATCH_REQUIRED_KEYS`
2. **中期（方案 A）**：重构 adapter 返回 `WindowSample` 实例，collate 接收 `List[WindowSample]`

**风险等级**：中 — 当前 `_validate_batch()` 覆盖了最危险的情况，但错误发现太晚

---

## 3. Validation Loop 落地

### 3.1 现状：已完整实现

与前两个问题不同，**validation loop 已经是完整的生产就绪实现**。

**核心流程** (`train_unified.py:489-610`)：

```
1. 构建 val_loader（graceful fallback）         # L489-505
2. 每 eval_interval 步触发                       # L590-591
3. ema.apply(model) + dist.barrier()            # L592-596
4. evaluate(model, val_loader, device, cfg)     # L597
5. ema.restore(model) + dist.barrier()          # L598-602
6. dist.all_reduce(metrics, AVG)                # L603-606
7. Rank 0 日志输出                               # L607-610
```

### 3.2 已实现的功能

| 功能 | 状态 | 位置 |
|------|------|------|
| 验证数据加载 | ✅ 已实现 | `train_unified.py:489-505` |
| DistributedSampler（多 GPU） | ✅ 已实现 | `train_unified.py:494-497` |
| 定期触发（eval_interval） | ✅ 已实现 | `config.py:265`（默认 2000 步） |
| EMA 权重切换 | ✅ 已实现 | `ema.py:97-113`，apply/restore + FSDP summon |
| FSDP barrier 同步 | ✅ 已实现 | 3 处 barrier（apply 后 / restore 前 / restore 后） |
| 多 GPU 指标归约 | ✅ 已实现 | `dist.all_reduce(..., ReduceOp.AVG)` |
| model.eval() 模式 | ✅ 已实现 | `evaluate()` 函数内 |
| 所有 loss 组件计算 | ✅ 已实现 | 调用 `forward_train()` 获取全部 loss |

**evaluate() 函数** (`train_unified.py:301-340`)：

```python
@torch.no_grad()
def evaluate(model, val_loader, device, cfg, max_batches=50):
    model.eval()
    accum = {}
    count = 0
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        batch = {k: _to_device(v) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.train.bf16):
            losses = model.forward_train(batch)
        for k, v in losses.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        count += 1
    model.train()
    return {k: v / count for k, v in accum.items()} if count > 0 else {}
```

### 3.3 可改进项

虽然核心功能完整，有几个增强点：

| 改进项 | 优先级 | 说明 |
|--------|--------|------|
| `max_batches` 可配置化 | 中 | 当前硬编码 50，应加入 `TrainConfig` |
| 最优 loss 检查点保存 | 中 | 当前仅按 `save_interval` 固定保存，不追踪 best val loss |
| 验证指标持久化 | 低 | 当前仅日志输出，未保存到文件或 wandb |
| 分阶段 eval_interval | 低 | Stage C 可能需要更频繁的 eval（当前统一 2000 步） |
| evaluate() 函数单测 | 低 | EMA apply/restore 已测，但 evaluate 函数本身无直接测试 |

### 3.4 结论

**Validation loop 是四个问题中实现最完整的**。核心 EMA 切换 → 评估 → 恢复 → 多 GPU 归约流程完整可靠。改进项都是锦上添花而非必需。

**风险等级**：低 — 已是生产就绪状态

---

## 4. LR Scale 真正接入 Optimizer

### 4.1 现状：已完整接入

**LR scale 已完整实现且经过测试**。从 config 定义到 optimizer 构建到测试断言，链路完整。

### 4.2 实现链路

#### 4.2.1 Config 层 (`config.py:221-222`)

```python
backbone_lr_scale: float = 0.1  # backbone LoRA LR = learning_rate × scale
expert_lr_scale: float = 0.5    # expert LR = learning_rate × scale (Stage B/C)
```

#### 4.2.2 Optimizer 构建 (`train_unified.py:418-455`)

```python
base_lr = cfg.train.learning_rate

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    clean = _strip_fsdp_prefix(name)

    if clean.startswith("backbone"):
        group = "backbone"
        lr_scale = cfg.train.backbone_lr_scale     # 0.1
    elif clean.startswith("action_expert"):
        group = "expert"
        lr_scale = cfg.train.expert_lr_scale       # 0.5
    else:
        group = "core"
        lr_scale = 1.0

    param_groups_map[key] = {
        "params": [],
        "lr": base_lr * lr_scale,                   # ← 实际乘以 scale
        "weight_decay": 0.0 if is_no_decay else cfg.train.weight_decay,
    }
    param_groups_map[key]["params"].append(param)

optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.95))
```

#### 4.2.3 实际 LR 值

| 阶段 | base_lr | Backbone (×0.1) | Core (×1.0) | Expert (×0.5) |
|------|---------|-----------------|-------------|---------------|
| A | 2e-4 | 2e-5 | 2e-4 | frozen |
| B | 1e-4 | 1e-5 | 1e-4 | 5e-5 |
| C | 3e-5 | 3e-6 | 3e-5 | 1.5e-5 |

#### 4.2.4 测试验证 (`test_ema_fsdp_gaps.py:561-613`)

```python
def test_param_groups_have_correct_lr():
    # ... 复现 optimizer 构建逻辑 ...
    assert pg["lr"] == pytest.approx(base_lr * cfg.train.backbone_lr_scale)
    assert pg["lr"] == pytest.approx(base_lr * cfg.train.expert_lr_scale)
    assert pg["lr"] == pytest.approx(base_lr)
```

#### 4.2.5 日志可观测

训练启动时输出：
```
optim group backbone_decay    XXX params  lr=2.00e-05  wd=0.0100
optim group core_decay        YYY params  lr=2.00e-04  wd=0.0100
optim group expert_decay      ZZZ params  lr=5.00e-05  wd=0.0100
```

### 4.3 FSDP 兼容性

`_strip_fsdp_prefix(name)` 在参数名分类前剥离 `_fsdp_wrapped_module.` 前缀，确保 FSDP 包裹后参数仍被正确归组。

### 4.4 附加特性

- **Weight decay 分组**：bias、LayerNorm 等参数自动 `wd=0.0`
- **可配置覆盖**：`stage_b_compressed.yaml` 中 `expert_lr_scale: 0.7` 可覆盖默认 0.5
- **6 个 param groups**：backbone_decay / backbone_nodecay / core_decay / core_nodecay / expert_decay / expert_nodecay

### 4.5 结论

**LR scale 是四个问题中实现最彻底的——完全不存在问题**。Config → optimizer → test → logging 链路闭环。

**风险等级**：无 — 已完整实现并测试

---

## 5. 总结与优先级

| 问题 | 现状 | 风险 | 行动建议 |
|------|------|------|---------|
| **Expert 多时刻监督** | 仅 t=-1，与 SOTA 一致 | **低** | 不实现。训练后根据 `loss_fm` 曲线决定是否做 ablation |
| **WindowSample 强类型化** | dataclass 存在但未使用，全链路 plain dict | **中** | 训练前在 collate 加前置校验；训练后重构 adapter 返回类型 |
| **Validation loop 落地** | **已完整实现** | **低** | 可选改进：`max_batches` 可配置、best-loss checkpoint、指标持久化 |
| **LR scale 接入 optimizer** | **已完整实现并测试** | **无** | 无需改动 |

### 优先级排序（修正版）

```
P1 — WindowSample collate 前置校验
     理由：防止数据管线静默错误，真实数据比 dummy 更容易触发
     工作量：~10 行代码
     时机：训练前

P2 — Validation loop 增强（max_batches 可配置、best checkpoint）
     理由：首轮训练需要追踪 val loss 选最优检查点
     工作量：~30 行
     时机：训练前可选

P3 — Expert 多时刻监督（方案 C，ablation）
     理由：当前设计与 pi-0 一致，动作块 95.8% 重叠使多时刻信号冗余
     前置条件：loss_fm 停滞 + 调 LR/batch 无效后再考虑
     工作量：~50 行代码 + 配置项 + 测试
     时机：Stage B 训练后视数据决定

P4 — WindowSample adapter 返回类型重构
     理由：类型安全提升，但 _validate_batch() 已覆盖高危场景
     工作量：~100 行跨 4 个文件
     时机：训练跑通后

P5 — LR scale（无需操作）
```

### 开始训练前的必要改动

**仅 P1 是训练前的硬性要求**：

- **P1**：数据管线的静默错误（key 拼写、字段缺失）在真实 LIBERO 数据上更容易触发，debug 成本远高于 10 行校验代码
- **P2**：可选但推荐——首轮训练的 val loss 趋势是后续所有决策的基础

**P3（Expert 多时刻监督）不应在首轮训练前实现**——连 t=-1 的收敛曲线都没有，无法判断 expert 梯度密度是否是瓶颈。正确做法是先跑通 320K 步，用数据说话。
