# HybridVLA v2 — v0.10.3 深度代码审计 (Extra-2-1)

> **日期**: 2026-03-27
> **范围**: 基于外部专家意见的逐项代码验证，覆盖 P0×4 + P1×4 + P2×3
> **方法**: 全量源码验证 + 精确到行号的证据链
> **前序**: 本报告回应 extra-2 遗留的结构性问题。extra-2 过于乐观地将系统评为"可以开始训练"——本报告逐项验证后给出修正结论。

---

## P0-1: 训练入口未做到真正 Stage-Aware

### 现状分析

当前有两个训练入口:

| 脚本 | 用途 | Stage B/C Expert? |
|------|------|:-----------------:|
| `train_stage_a.py` | Stage A 专用 | **无条件冻结** (line 99) |
| `train_unified.py` | A/B/C 统一 | **隐式依赖默认值** |

#### `train_unified.py` 冻结逻辑 (lines 147-157):

```python
if stage == "a":
    for p in model.action_expert.parameters():
        p.requires_grad = False          # ← 显式冻结
    logger.info("Stage A: action_expert frozen.")
elif stage == "b":
    logger.info("Stage B: all trainable (expert unfrozen, cond_prefix detached).")
    # ← 没有任何 requires_grad 操作
else:
    logger.info("Stage C: full fine-tune.")
    # ← 没有任何 requires_grad 操作
```

#### 实际行为验证

`train_unified.py` 每次运行创建**全新模型** → 所有参数默认 `requires_grad=True` → Stage B/C 的 `elif/else` 分支不做冻结操作 → **Expert 确实可训练**。

**表面上没有 bug。** 但这是通过"什么都不做"来实现"解冻"——依赖 PyTorch 默认行为，而不是显式声明。

### 问题清单

**问题 1: 隐式解冻 = 脆弱**

如果以下任一变化发生，Stage B/C 将静默失效:
- 模型 `__init__` 中某处将 expert 默认冻结
- Checkpoint 加载恢复了 `requires_grad` 状态
- 有人复制 `train_stage_a.py` 改成 Stage B

**问题 2: 无可训练参数校验**

`train_unified.py:159-162` 打印了总数但不分模块:
```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info("Params: %s trainable / %s total (%.1f%%)", ...)
```

没有断言 "Stage B 下 expert 参数必须可训练"。训练可以在 expert 静默冻结的状态下跑完全程，只有 loss 不降时才被发现。

**问题 3: Stage C "全微调" 名不副实**

骨干初始化时 (`qwen2vl_backbone.py:110-131`) 无条件执行:
```python
# 视觉塔永久冻结
for p in composite_model.visual.parameters():
    p.requires_grad = False

# 文本层 0-15 永久冻结
for i, layer in enumerate(text_model.layers):
    if i < freeze_until:  # freeze_until=16
        p.requires_grad = False

# embed_tokens 永久冻结
for p in text_model.embed_tokens.parameters():
    p.requires_grad = False
```

Stage C 的 `else` 分支 **不会覆盖这些冻结**。日志说 "full fine-tune" 但实际是 "除 vision tower + text 0-15 + embeddings 外的 fine-tune"。

**问题 4: 无 per-module optimizer 参数组**

`train_unified.py:171-191` 只区分 decay/no-decay，不区分模块:
```python
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if any(nd in name for nd in no_decay_keywords):
        no_decay_params.append(param)
    else:
        decay_params.append(param)
```

Stage B 的设计意图 (expert 低 LR / backbone 低 LR) 无法通过当前代码实现。所有可训练参数共享同一 LR。

### 缺失的三个函数

当前代码没有以下任何一个:

| 函数 | 用途 | 状态 |
|------|------|:----:|
| `configure_trainable_modules(model, stage)` | 按 stage 显式设置 requires_grad | ✗ 不存在 |
| `build_optimizer_param_groups(model, stage)` | 按 module 分组, 支持 per-module LR | ✗ 不存在 |
| `sanity_check_trainable_params(model, stage)` | 断言可训练参数与预期一致 | ✗ 不存在 |

### 验收差距

| 验收标准 | 当前状态 |
|---------|---------|
| Stage A/B/C 各跑 5 step | smoke test 支持 `--stage a/b/c` ✓ |
| 打印可训练模块清单 | 只打印总数, 不分模块 ✗ |
| Stage B/C 下 expert 梯度非零 | 未验证 (无 per-module grad 日志) ✗ |
| checkpoint resume 后 stage 切换正确 | 功能存在但无断言 ⚠️ |

### 严重性评定: **P0 确认**

虽然当前代码在 "happy path" 下 Stage B/C expert 是可训练的，但缺乏任何防护机制。一旦投入真实训练，可能出现 "跑了 200K 步才发现 expert 没学" 的灾难性后果。**必须在训练前修复。**

---

## P0-2: 缺少 Stage B/C 最小可运行测试

### 现状

| 测试 | 文件 | Stage A | Stage B | Stage C |
|------|------|:-------:|:-------:|:-------:|
| Smoke test | `scripts/train_smoke_test.py` | ✓ (默认) | ✓ (`--stage b`) | ✓ (`--stage c`) |
| 单元测试 | `tests/` | ✗ 目录空 | ✗ | ✗ |

`train_smoke_test.py:152-154` 对 Stage B/C 确实不冻结 expert:
```python
if stage == "a":
    for p in model.action_expert.parameters():
        p.requires_grad = False
# stage b/c: 不冻结, expert 可训练
```

### 问题

**Smoke test 不验证关键不变量:**

1. ✗ 不检查 `loss_fm` 是否存在 (Stage B/C 应产生 flow matching loss)
2. ✗ 不检查 expert 参数是否更新 (只检查总 loss 不 NaN)
3. ✗ 不检查 `cond_prefix.detach()` 是否正确阻断梯度
4. ✗ 不检查 consistency loss 中 discrete-continuous 分支是否激活

**没有专门的 Stage B/C 回归测试:**

如果有人修改 `forward_train()` 的 stage 门控逻辑，当前 smoke test **不会检测到** Stage B/C 静默回退到 Stage A 行为。

### 缺失的测试场景

```
tests/test_stage_b_minimal.py:
  - cfg.stage = "b"
  - forward_train → assert "loss_fm" in losses
  - assert "loss_consistency" in losses
  - assert loss_fm.requires_grad == True
  - backward + step → assert expert params changed
  - assert backbone LoRA params unchanged (stop_gradient_cond_prefix)

tests/test_stage_c_minimal.py:
  - cfg.stage = "c"
  - forward_train → assert "loss_fm" in losses
  - backward + step → assert expert params changed
  - assert backbone LoRA params can change (if cond_prefix not detached)
```

### 严重性评定: **P0 确认**

没有这些测试，Stage B/C 的"正确性"完全依赖代码审查。真实训练中 expert 不更新会浪费数天 GPU 时间。

---

## P0-3: Refresh Vision Batching 存在 Shape 不一致风险

### 问题根源链

```
Qwen2-VL Processor
  → 根据图像分辨率/宽高比动态分 patch
  → 输出 pixel_values shape: [N_patches, patch_dim]
  → N_patches 因图像不同而不同
       ↓
hdf5_adapter._process_text_image()
  → 直接返回 processor 输出
  → 不做 shape 校验或 padding
       ↓
collate.py (line 45)
  → torch.stack(frame_vals, dim=0)
  → 同一 batch 内不同样本的 N_patches 不同
  → ❌ RuntimeError: shape mismatch
```

### 代码验证

**`hdf5_adapter.py:155-167`** — 无 shape 保证:
```python
if self.processor is not None and pil_image is not None:
    tok = self.processor(
        text=lang, images=pil_image,
        return_tensors="pt", padding="max_length",
        truncation=True, max_length=256,
    )
    return {
        ...
        "pixel_values": tok["pixel_values"].squeeze(0),  # ← shape 不固定!
        "image_grid_thw": tok["image_grid_thw"].squeeze(0),
    }
```

`max_length=256` 只控制 **text token** 长度，不控制 **pixel patch** 数量。

**`collate.py:36-48`** — 无条件 stack:
```python
elif isinstance(values[0], list):
    R = len(values[0])
    transposed = []
    for r in range(R):
        frame_vals = [v[r] for v in values]
        if isinstance(frame_vals[0], Tensor):
            transposed.append(torch.stack(frame_vals, dim=0))  # ← 崩溃点
```

**`BackboneConfig`** (`config.py:42-43`):
```python
min_pixels: int = 200704   # ~448×448
max_pixels: int = 401408   # ~634×634
```

在此像素预算内，不同宽高比的图像产生不同 patch 数。

### 崩溃场景

```python
# batch_size=2, 一张 640×480, 一张 320×240
sample_0["pixel_values"].shape = [2304, 1176]  # 多 patch
sample_1["pixel_values"].shape = [1024, 1176]  # 少 patch
torch.stack([sample_0["pixel_values"], sample_1["pixel_values"]])
# → RuntimeError: stack expects each tensor to be equal size
```

**此问题在 dummy smoke test 中不会暴露** — DummyVLADataset 不产生视觉字段。只有真实 HDF5 数据才会触发。

### 修复路线

**路线 A (保守, 推荐先做):**
- 在 `_process_text_image()` 中强制统一 resize: 所有图像 resize 到固定尺寸 (如 448×448) 再送 processor
- 保证 processor 输出 shape 一致

**路线 B (通用):**
- collate 中不 stack 变长视觉张量，保留 list
- backbone 逐样本处理 (或 pad + mask)

### 严重性评定: **P0 确认**

这是"真实训练第一天就爆"的问题。Dummy 测试不暴露, 必须在投入 GPU 前修复。

---

## P0-4: 单相机实现 vs 多相机配置/叙事 脱节

### 代码证据

**配置层** — 声称 3 相机 (`config.py:52-58, 304-306`):
```python
class MultiCameraConfig:
    enable: bool = True
    num_cameras: int = 3
    camera_names: List[str] = ["wrist", "shoulder", "overhead"]

class DataConfig:
    camera_keys: List[str] = ["agentview_rgb", "wrist_rgb", "overhead_rgb"]
```

**数据层** — 实际只用 1 相机 (`hdf5_adapter.py:69`):
```python
self.image_key = cfg.data.image_key  # 单个 key: "agentview_rgb"
```

**全代码搜索**: `camera_keys` **零引用** (除定义处)。`MultiCameraConfig` **零引用** (除定义处)。

### 影响

1. README 和设计文档中 "multi-camera native" 的描述与实现不符
2. 如果按 "三相机系统" 理解实验结果, 实际是在误读
3. 后续如果有人看代码, 会被配置误导

### 推荐

在真正实现多相机前:
```python
# 训练时加断言:
assert len(cfg.data.camera_keys) == 1 or not cfg.multi_camera.enable, \
    "Multi-camera not yet implemented. Set multi_camera.enable=False."
```

或者更简单: 把 `MultiCameraConfig.enable` 默认改为 `False`, 文档标注 "planned"。

### 严重性评定: **P0 确认** (不是代码崩溃, 是认知错误风险)

---

## P1-1: Tri-Rate 无 Clean Ablation 支持

### 现状

`TemporalCoreConfig` (`config.py:76-94`) 没有以下字段:

| 字段 | 用途 | 存在? |
|------|------|:-----:|
| `temporal_mode: str` | `"single" / "dual" / "tri"` | ✗ |
| `use_action_history: bool` | 启用/禁用动作历史编码器 | ✗ |
| `use_stale_encoding: bool` | 启用/禁用 stale time 编码 | ✗ |

`TriRateMambaCore.__init__` (`mamba_core.py:613-641`) 无条件创建三个流:
```python
self.fast_mamba = ...   # 20 layers — 总是创建
self.medium_mamba = ... # 6 layers  — 总是创建
self.slow_mamba = ...   # 10 layers — 总是创建
```

`forward()` 用 bool flag 控制 medium/slow 是否更新，但不能完全禁用:
```python
# medium_update=False → 复用 last_medium_token，但 medium_mamba 占显存
# semantic_refresh=False → 复用 last_slow_token，但 slow_mamba 占显存
```

### 需要的实验矩阵

| 对照组 | 配置方式 | 当前是否支持 |
|--------|---------|:----------:|
| Single-rate Transformer | baseline | ✗ (需新代码) |
| Single-rate Mamba | `fast_layers=20, medium=0, slow=0` | ✗ |
| Dual-rate Mamba | `fast_layers=20, slow_layers=10, medium=0` | ✗ |
| **Tri-rate Mamba** | 当前实现 | ✓ |
| Tri-rate w/o action history | `use_action_history=False` | ✗ |
| Tri-rate w/o stale encoding | `use_stale_encoding=False` | ✗ |
| Tri-rate gate fusion | `fusion_type="gate"` | ⚠️ (config 有, code 未实现) |

### 严重性评定: **P1 确认**

没有 clean ablation，Tri-Rate 的主贡献论证力弱。但这不阻塞训练，可以在 Stage A 训练期间并行开发。

---

## P1-2: 缺少长距相关统计指标

### 现状

训练日志 (`train_unified.py:317-318`):
```python
logger.info("Step %d | %s | gnorm: %.3f | lr: %.2e | %.1f sps",
            global_step, parts, grad_norm.item(), lr, sps)
```

仅记录: loss 分量, 总梯度范数, 学习率, 吞吐量。

评估函数 (`train_unified.py:87-126`): 计算验证集上的平均 loss, 不分桶。

### 缺失的长距指标

| 指标 | 用途 | 存在? |
|------|------|:-----:|
| 不同 horizon bucket 的 loss | 区分短/中/长距表现 | ✗ |
| steps_since_refresh 分桶表现 | 验证 stale tolerance | ✗ |
| medium/slow token 激活频率 | 分析流使用模式 | ✗ |
| chunk reuse 次数 | 量化 chunk caching 效率 | ✗ |
| semantic refresh 触发频率 | 验证 refresh schedule 合理性 | ✗ |
| task length vs success rate | 长距优势核心证据 | ✗ |

### 严重性评定: **P1 确认**

不阻塞训练, 但要发 Tri-Rate 长距优势论文, 这些是必需的分析数据。

---

## P1-3: 无 Inference Runtime 封装

### 现状

`vla_hybrid_v2/infer/__init__.py`: 1 行, 空 docstring。

模型提供了推理 API:
- `semantic_step()` (`hybrid_vla_v2.py:582`)
- `control_step()` (`hybrid_vla_v2.py:595`)
- `init_runtime()` (`hybrid_vla_v2.py:705`)

但没有封装层。用户需要手动:
1. 调 `init_runtime()` 初始化状态
2. 每 `1/semantic_hz` 秒调一次 `semantic_step()`
3. 每 `1/control_hz` 秒调一次 `control_step()`
4. 管理 `RuntimeCache` 的生命周期
5. 处理 action denormalization
6. 管理 refresh schedule

### 缺失内容

| 组件 | 用途 | 状态 |
|------|------|:----:|
| `infer/runtime.py` | 封装 semantic/control 循环 | ✗ |
| PolicyWrapper | 环境交互接口 | ✗ |
| Rollout demo | 100 步连续控制示例 | ✗ |
| Action denormalization | 从 [-1,1] 还原真实动作 | ✗ |
| Latency profiling | 推理时间统计 | ✗ |

### 严重性评定: **P1 确认**

不阻塞训练, 但阻塞评估和部署。如果要在 LIBERO/CALVIN 上测试, 必须有 runtime wrapper。

---

## P1-4: 梯度隔离无可验证手段

### 现状

Stage B 的 `cond_prefix.detach()` (`hybrid_vla_v2.py:531-533`):
```python
if (self.cfg.train.stop_gradient_cond_prefix
        or self.cfg.train.block_fm_to_backbone):
    cond_prefix = cond_prefix.detach()
```

`stage_b.yaml:26-27`:
```yaml
stop_gradient_cond_prefix: true
block_fm_to_backbone: true
```

**设计意图**: Expert 训练的 flow matching 梯度不应回传到 backbone/grounder, 避免 expert 初期不稳定梯度破坏已训练的感知模块。

### 问题

1. 只有 `cond_prefix.detach()` 一处隔离, 没有验证手段
2. 无 per-module gradient norm 日志
3. 无法证明 "detach 后 perception branch 更稳定"

### 当前梯度日志

```python
# train_unified.py:302
grad_norm = clip_grad_norm_fsdp(model, cfg.train.max_grad_norm)
# train_unified.py:318
logger.info("... gnorm: %.3f ...", grad_norm.item())
```

**只有全局 gnorm**。没有:
- backbone LoRA gnorm
- grounder gnorm
- temporal core gnorm
- action expert gnorm

### 需要的日志

```python
# 分模块梯度范数
for name, module in [("backbone_lora", model.backbone),
                      ("grounder", model.grounder),
                      ("temporal_core", model.temporal_core),
                      ("action_expert", model.action_expert)]:
    gnorm = sum(p.grad.norm()**2 for p in module.parameters()
                if p.grad is not None).sqrt()
    logger.info("  %s gnorm: %.4f", name, gnorm)
```

### 严重性评定: **P1 确认**

Reviewer 必问 "knowledge insulation 的证据是什么"。没有这张图, 论点缺乏数据支撑。

---

## P2-1: 多相机 — 建议推迟

`camera_keys` 配置已存在但为死代码。如果短期主线是 Tri-Rate 长距:
- 论文先不打多相机
- Repo 标注 "planned"
- 后续做时至少需要: per-camera backbone pass + camera token embedding + camera-aware fusion

### 严重性评定: **P2**

---

## P2-2: Phase/Affordance 应做"可缺省"

当前 `forward_train()` 已经用 `"phase_labels" in batch` 门控:
```python
if self.phase_head is not None and "phase_labels" in batch:
    ...
if self.affordance_head is not None and "affordance_labels" in batch:
    ...
```

**已部分实现**, 但缺少:
- 启动时日志 "phase supervision: active/skipped"
- 评估时区分有/无标签的指标

### 严重性评定: **P2** (已基本可缺省)

---

## P2-3: Backbone Refresh 重复开销

当前 refresh 路径 (`hybrid_vla_v2.py:373-385`) 逐 refresh 帧重跑完整骨干:
```python
for r in range(R):
    backbone_out = self.backbone.forward_semantic(
        input_ids=..., attention_mask=...,
        pixel_values=..., image_grid_thw=...,
    )
```

R=4 时, 骨干前向 4 次 (7B 模型)。这是训练计算的主要瓶颈。

可能的优化 (P2, 不急):
- 只在关键 refresh 点跑完整骨干, 中间帧复用特征
- 缓存骨干低层特征, 只更新高层
- 减少 `output_hidden_states=True` 的额外开销

### 严重性评定: **P2**

---

## 综合评分修正

### extra-2 评分 vs extra-2-1 修正

| # | 维度 | extra-2 评分 | 修正评分 | Δ | 修正理由 |
|---|------|:-----------:|:-------:|:-:|---------|
| 1 | 设计一致性 | 8.5 | **7.5** | -1.0 | 多相机配置 vs 实现脱节, Stage C "全微调" 名不副实 |
| 2 | 正确性 | 9.5 | **8.0** | -1.5 | refresh batching 会崩, stage 门控缺校验 |
| 3 | 完备性 | 9.0 | **7.5** | -1.5 | 无 Stage B/C 测试, 无消融配置, 无推理封装 |
| 4 | 训练稳定性 | 9.0 | **7.5** | -1.5 | 无 per-module gradient 监控, 梯度隔离不可验证 |
| 5 | 可扩展性 | 7.5 | **6.5** | -1.0 | 无消融开关, 不能方便做实验矩阵 |
| 6 | 性能设计 | 7.0 | **6.5** | -0.5 | refresh 重复开销, 无长距指标 |
| 7 | 生产就绪度 | 8.0 | **5.5** | -2.5 | 无 runtime 封装, 无 rollout 评估, vision batch 不稳定 |
| 8 | 代码质量 | 8.5 | **7.5** | -1.0 | 死代码 (camera_keys/MultiCameraConfig), 隐式解冻 |
| 9 | 文档 | 5.5 | **5.0** | -0.5 | 多相机叙事失实 |
| 10 | 测试 | 2.0 | **2.0** | — | 仍为空 tests/ |

**修正综合评分**: 加权后约 **6.5/10** (extra-2 报告的 7.4 过于乐观)

---

## 是否可以开始训练 — 修正结论

### extra-2 结论 vs 修正

| extra-2 结论 | 修正结论 | 理由 |
|:------------:|:-------:|------|
| Stage A ✅ 可以 | **⚠️ 需先修 P0-3** | 视觉 batch 在真实数据上会崩 |
| Stage B ✅ 可以 | **❌ 需修 P0-1 + P0-2** | 缺显式解冻+校验+测试 |
| Stage C ✅ 可以 | **❌ 需修 P0-1 + P0-2** | 同上 |

### 修正后的启动前提

**必须修复 (阻塞训练):**

| 优先级 | 项目 | 预计工作量 |
|--------|------|-----------|
| **P0-1** | `configure_trainable_modules()` + `build_optimizer_param_groups()` + `sanity_check_trainable_params()` | ~80 行 |
| **P0-2** | `tests/test_stage_b_minimal.py` + `tests/test_stage_c_minimal.py` | ~120 行 |
| **P0-3** | 统一 resize 或 pad+mask 解决 pixel_values 变长问题 | ~30 行 |
| **P0-4** | `MultiCameraConfig.enable` 默认改 False + 断言 | ~10 行 |

**P0 总工作量: ~240 行, 预计 0.5-1 天**

**训练中修复 (不阻塞启动, 但阻塞论文):**

| 优先级 | 项目 | 预计工作量 |
|--------|------|-----------|
| **P1-1** | `temporal_mode` + `use_action_history` + `use_stale_encoding` 消融开关 | ~150 行 |
| **P1-2** | 长距指标日志 (horizon bucket, chunk reuse, refresh 统计) | ~100 行 |
| **P1-3** | `infer/runtime.py` + 最小 demo | ~200 行 |
| **P1-4** | per-module gradient norm 日志 | ~30 行 |

### 修正后的训练启动路径

```
今天:
  P0-1 显式 stage 门控 + sanity check      ← 半天
  P0-2 Stage B/C 最小测试                  ← 半天
  P0-3 统一图像 resize                      ← 2 小时
  P0-4 多相机断言                           ← 30 分钟
       ↓
P0 全部完成后:
  准备数据 + compute_stats
  Stage A 开始训练 (120K steps, ~15-22h)
       ↓
Stage A 训练期间并行开发:
  P1-1 消融配置
  P1-2 长距指标
  P1-4 per-module gradient logging
       ↓
Stage A 完成后:
  P1-3 runtime wrapper (Stage B 前可选)
  Stage B 开始 (200K steps)
```

---

## 附录: 逐项代码位置索引

| 问题 ID | 文件 | 行号 | 关键代码 |
|---------|------|------|---------|
| P0-1a | `train_unified.py` | 147-157 | Stage 冻结逻辑 |
| P0-1b | `train_unified.py` | 171-191 | Optimizer 参数组 |
| P0-1c | `train_unified.py` | 159-162 | 可训练参数打印 |
| P0-1d | `qwen2vl_backbone.py` | 110-134 | 骨干冻结 (Stage C 不覆盖) |
| P0-2 | `scripts/train_smoke_test.py` | 152-154 | Stage A 冻结 (B/C 无校验) |
| P0-3a | `hdf5_adapter.py` | 155-167 | _process_text_image 无 shape 保证 |
| P0-3b | `collate.py` | 36-48 | torch.stack 变长张量 |
| P0-3c | `config.py` | 42-43 | min/max_pixels (不控制 patch 数) |
| P0-4a | `config.py` | 52-58 | MultiCameraConfig (死代码) |
| P0-4b | `config.py` | 304-306 | camera_keys (零引用) |
| P0-4c | `hdf5_adapter.py` | 69 | self.image_key (单相机) |
| P1-1 | `config.py` | 76-94 | TemporalCoreConfig (无 mode 字段) |
| P1-1 | `mamba_core.py` | 613-641 | 三流硬编码初始化 |
| P1-2 | `train_unified.py` | 317-318 | 日志 (无长距指标) |
| P1-3 | `vla_hybrid_v2/infer/__init__.py` | 1 | 空 stub |
| P1-4 | `train_unified.py` | 302 | 只有全局 gnorm |
