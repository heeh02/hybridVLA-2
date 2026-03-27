# HybridVLA v2 — v0.10.5 全面代码审计

> **日期**: 2026-03-27
> **范围**: (1) 历史修复回归验证 (2) 多相机实现审计 (3) v0.10.5 新修复验证 (4) 新发现问题
> **标准**: "8×H100 真实训练第一天会不会崩"

---

## Part 1: 历史修复回归验证 — 全部完好

| # | 修复 | 版本 | 文件:行 | 状态 |
|---|------|------|---------|:----:|
| 1 | Action chunk T+H-1 | v0.10.1 | `hdf5_adapter.py:142,309,356` | ✅ |
| 2 | pixel_values resize 448² | v0.10.4 | `hdf5_adapter.py:250-256` | ✅ |
| 3 | collate safe_stack_vision | v0.10.4 | `collate.py:23-43,65-83` | ✅ |
| 4 | configure_trainable_modules | v0.10.4 | `train_unified.py:87-154` | ✅ |
| 5 | sanity_check_trainable_params | v0.10.4 | `train_unified.py:157-228` | ✅ |
| 6 | Processor 连接 | v0.10.3 | `train_unified.py:415-419` | ✅ |
| 7 | HDF5 图像读取 | v0.10.3 | `hdf5_adapter.py:167-175` | ✅ |
| 8 | 多步监督 FAST/Phase/Aff | v0.10.3 | `hybrid_vla_v2.py:484-524` | ✅ |
| 9 | normalizer_stats_dir 解耦 | v0.10.1 | `config.py:306` | ✅ |
| 10 | 递归 _to_device | v0.10.1 | `train_unified.py:458-463` | ✅ |
| 11 | MultiCamera.enable=False | v0.10.4 | `config.py:54` | ✅ |

**无回归。** 所有历史修复均在当前代码中完好。

---

## Part 2: v0.10.5 新修复验证

### V1: Val Split — ✅ 已修复

`hdf5_adapter.py:81-111` 现在正确处理 split:

```python
if split == "val" and self.dcfg.val_data_dir:
    # 独立 val 目录
    self.episode_paths = sorted(val_dir.glob("*.hdf5"))
elif split == "val" and not self.dcfg.val_data_dir:
    # Episode 比例切分 (最后 val_ratio 比例)
    n_val = max(1, int(len(all_paths) * self.dcfg.val_ratio))
    self.episode_paths = all_paths[-n_val:]
elif split == "train" and self.dcfg.val_data_dir is None:
    # 排除 val episodes
    n_val = max(1, int(len(all_paths) * self.dcfg.val_ratio))
    self.episode_paths = all_paths[:-n_val]
```

新增配置字段 (`config.py:307-308`):
```python
val_data_dir: Optional[str] = None
val_ratio: float = 0.1
```

**判定**: ✅ 闭环。train/val 数据不再重叠。

### V4: Per-Module LR — ✅ 已修复

`train_unified.py:341-374` 现在按模块分组:

```python
if name.startswith("backbone"):
    group = "backbone"; lr_scale = cfg.train.backbone_lr_scale    # 0.1
elif name.startswith("action_expert"):
    group = "expert"; lr_scale = cfg.train.expert_lr_scale         # 0.5
else:
    group = "core"; lr_scale = 1.0
```

新增配置字段 (`config.py:214-215`):
```python
backbone_lr_scale: float = 0.1
expert_lr_scale: float = 0.5
```

**效果**: Stage B 下 backbone LoRA LR = 1e-4 × 0.1 = 1e-5, expert LR = 1e-4 × 0.5 = 5e-5, core LR = 1e-4。

**判定**: ✅ 闭环。模块间 LR 差异化已实现。

### V5: Per-Module Gradient Norm — ✅ 已修复

`train_unified.py` 新增 `_log_per_module_grad_norm()` 函数, 记录 backbone / grounder / temporal_core / action_expert 等 9 个模块的独立 gnorm, 每 5×log_interval 步触发。

**判定**: ✅ 闭环。Stage B 的 knowledge insulation 现在可通过日志验证。

---

## Part 3: 多相机实现审计

### 3.1 实现范围

| 层 | 组件 | 状态 | 证据 |
|----|------|:----:|------|
| **配置** | `MultiCameraConfig` | ✅ 完整 | `config.py:53-60` — enable, num_cameras, camera_names, fusion, max_cameras |
| **配置** | `camera_keys` 在 DataConfig | ✅ 引用 | `config.py:310-314` — 3 camera keys |
| **配置** | YAML 多相机 | ✅ 存在 | `configs/data/libero_multicam.yaml` — enable=true |
| **数据** | `_read_multi_camera_images()` | ✅ 实现 | `hdf5_adapter.py:177-184` — 循环读取多 camera |
| **数据** | `_process_text_multi_image()` | ✅ 实现 | `hdf5_adapter.py:190-239` — 联合 text+多图 tokenize |
| **数据** | `__getitem__` 多相机分支 | ✅ 实现 | `hdf5_adapter.py:321-391` — if multi_camera 分支完整 |
| **数据** | `num_cameras` 跟踪 | ✅ 实现 | `hdf5_adapter.py:388-391`, `schema.py:44` |
| **骨干** | `CameraPositionEmbedding` | ✅ 实现 | `qwen2vl_backbone.py:56-123` — 可学习 per-camera 嵌入 |
| **骨干** | `forward_semantic` 接受 num_cameras | ✅ 实现 | `qwen2vl_backbone.py:256-295` |
| **模型** | `forward_train` 传递 num_cameras | ✅ 实现 | `hybrid_vla_v2.py:371-398` |
| **模型** | `semantic_step` 接受 num_cameras | ✅ 实现 | `hybrid_vla_v2.py:587-599` |
| **Collate** | 处理 num_cameras | ✅ 通用 | `collate.py:89-90` — int 自动转 tensor |

### 3.2 多相机调用链

```
config: multi_camera.enable = true
  ↓
hdf5_adapter.__init__: self.camera_keys = cfg.data.camera_keys (3 keys)
  ↓
__getitem__:
  _read_multi_camera_images(data, camera_keys, frame)
    → [PIL, PIL, PIL] (3 cameras)
  _process_text_multi_image(lang, [PIL, PIL, PIL])
    → processor(text, images=[img1, img2, img3])
    → pixel_values, image_grid_thw (含 3 张图信息)
  sample["num_cameras"] = 3
  ↓
collate:
  stack pixel_values → [B, N_patches_total, D]
  stack num_cameras → [B] (all = 3)
  ↓
forward_train:
  num_cameras = batch["num_cameras"][0]  # = 3
  backbone.forward_semantic(..., num_cameras=3)
    → Qwen2-VL 处理含 3 张图的 token 序列
    → CameraPositionEmbedding 给不同图的 token 加 camera ID 嵌入
    → multi_scale_adapter 融合多尺度特征
  grounder(backbone_features) → 正常处理 (含 camera 上下文)
```

### 3.3 残留问题

| 问题 | 严重性 | 说明 |
|------|:------:|------|
| `camera_names` 字段未使用 | P3 | 定义了但代码中零引用, 纯装饰 |
| `fusion: str = "concat"` 未使用 | P2 | 配置字段存在但无对应 fusion 模块。实际融合是隐式的 (token 拼接在序列中) |
| Smoke test 未测多相机模式 | P2 | `train_smoke_test.py` 使用 `enable=False` |
| 多相机 resize 逻辑 | **P1** | `_process_text_multi_image` 中 resize 应用于每张图, 但 3 张图拼接后 token 序列更长, 显存增加约 3× |

### 3.4 多相机结论

**实现完成度: ~90%。** 核心链路 (数据读取 → tokenize → camera embedding → backbone → grounder → model) 已打通。默认关闭, 启用只需改 YAML。

**"完美解决" 的差距:**
1. 无独立 fusion 模块 — 依赖 Qwen2-VL processor 的隐式 token 拼接 + CameraPositionEmbedding 的可学习区分
2. 显存预算未验证 — 3 相机 token 序列约 3× 单相机, 需要降低 batch size 或增加 grad_accum
3. 无多相机 smoke test — 功能未经运行时验证

---

## Part 4: 新发现问题

### N1: FSDP 多卡下 Checkpoint 加载会静默失效 — **P0-CRITICAL**

**这是当前最严重的未修复问题。**

**证据链**:

`wrap_fsdp()` (`distributed.py:112-119`) 不使用 `use_orig_params=True`:
```python
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    ...
    # 无 use_orig_params=True → FSDP 改变 state_dict key 名
)
```

`save_checkpoint()` (`checkpointing.py:24-34`) 使用 `FULL_STATE_DICT`:
```python
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
    return model.state_dict()  # 保存时正确: 用原始 key
```

`load_checkpoint()` (`checkpointing.py:107-108`) **没有** state_dict_type context:
```python
state = torch.load(ckpt_dir / "model.pt", ...)
missing, unexpected = model.load_state_dict(state, strict=strict)
# ← FSDP 模型的默认 state_dict 类型可能是 LOCAL (sharded)
# ← 全量 state dict 的 key 与 FSDP sharded key 不匹配
# ← strict=False → 静默跳过所有参数
```

**影响**: 在 8×H100 (FSDP 启用) 下:

| 场景 | 后果 |
|------|------|
| Stage A auto-resume | 中断续训失效 → 从头开始 |
| Stage B cross-stage 加载 | Stage A 权重丢失 → expert 从随机初始化训练 |
| Stage C cross-stage 加载 | Stage B 权重丢失 → 全部从零开始 |

**日志表现**: `load_checkpoint` 会打印 `"Missing N keys"` 和 `"Unexpected M keys"` 的 warning, 但 `strict=False` 不抛异常, 训练继续进行。用户可能不会注意到 warning。

**修复方案**:
```python
# checkpointing.py:load_checkpoint 中加:
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    if isinstance(model, FSDP):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            missing, unexpected = model.load_state_dict(state, strict=strict)
        return ...
except ImportError:
    pass
# fallback to regular load
```

**单卡训练不受影响** — FSDP 仅在 `world_size > 1` 时启用。

---

### N2: collate 中 refresh list 的 None/Tensor 混合风险 — P2

`collate.py:78-87`:
```python
for r in range(R):
    frame_vals = [v[r] for v in values]
    if frame_vals[0] is None:          # ← 只检查第一个样本
        transposed.append(None)
    elif isinstance(frame_vals[0], Tensor):
        transposed.append(_safe_stack_vision(frame_vals, ...))  # 假设全是 Tensor
```

如果 batch 中 sample 0 的 refresh frame r 有图像但 sample 1 没有 (某一帧图像缺失), `frame_vals = [Tensor, None]` → `_safe_stack_vision([Tensor, None])` → crash。

**当前缓解**: `_read_image` 对缺失图像返回 None, `_process_text_image(lang, None)` 返回 text-only token (无 pixel_values=None)。如果所有 refresh 帧要么全有图要么全无图, 不会出问题。

**触发条件**: HDF5 中部分帧缺少 images group 或特定 camera key。实际数据中不太常见, 但不是不可能。

**严重性**: P2 — 不太可能在干净数据上触发, 但没有防护。

---

### N3: val_ratio 切分的边界情况 — P2

`hdf5_adapter.py:108-109`:
```python
n_val = max(1, int(len(all_paths) * self.dcfg.val_ratio))
self.episode_paths = all_paths[:-n_val] if n_val < len(all_paths) else all_paths
```

当 `n_val >= len(all_paths)` (如只有 1-2 个 episode), train 集 = 全量 = val 集。无断言。

**严重性**: P2 — 只在极小数据集上触发。

---

### N4: Gradient 累积 loss 日志轻微膨胀 — P3

`train_unified.py:478-498`:
```python
loss = losses["loss_total"] / grad_accum   # 缩放用于 backward
loss.backward()
for k, v in losses.items():
    accum_loss[k] = accum_loss.get(k, 0.0) + v.detach().item()  # 原始未缩放值
...
avg = {k: v / cfg.train.log_interval for k, v in accum_loss.items()}
```

每个 log_interval 内累积 `log_interval × grad_accum` 个 micro-batch loss, 但只除以 `log_interval`。**日志中 loss 值膨胀 `grad_accum` 倍** (默认 4×)。

不影响训练 (backward 用的是正确缩放的 loss), 只影响日志可读性。趋势正确, 绝对值偏大。

**严重性**: P3 — 纯日志展示问题。

---

## Part 5: 综合评估

### 问题严重性总览

| 级别 | 问题 | 影响 | 阻塞训练? |
|------|------|------|:---------:|
| **P0** | N1: FSDP checkpoint 加载 | 多卡训练 checkpoint 静默失效 | **阻塞多卡** |
| P2 | N2: refresh None/Tensor 混合 | 脏数据可能 crash collate | 否 |
| P2 | N3: val_ratio 极小数据集 | train=val | 否 |
| P2 | 多相机无 smoke test | 功能未经运行时验证 | 否 |
| P3 | N4: loss 日志膨胀 | 绝对值偏大但趋势正确 | 否 |

### 训练就绪度

| 场景 | 判定 | 阻塞项 |
|------|:----:|--------|
| **单卡 Stage A** | ✅ 可以 | 无 |
| **8×H100 Stage A** | **❌** | N1: FSDP checkpoint load |
| **Stage B/C** (单卡) | ✅ 可以 | 无 |
| **Stage B/C** (多卡) | **❌** | N1: 跨阶段 checkpoint 失效 |
| **多相机模式** | ⚠️ 未验证 | 无 smoke test, 显存未评估 |

### N1 修复工作量

`checkpointing.py:load_checkpoint` 中加约 **10 行** FSDP state_dict_type context。这是唯一的 P0 阻塞项。

---

## Part 6: 评分

| # | 维度 | v0.10.4 | v0.10.5 | 说明 |
|---|------|:-------:|:-------:|------|
| 1 | 设计一致性 | 8.0 | **8.5** | 多相机实现闭环, per-module LR 分组, val split |
| 2 | 正确性 | 9.0 | **8.5** | -0.5: FSDP load 静默失效 |
| 3 | 完备性 | 7.5 | **8.5** | 多相机 + val split + per-module LR + gnorm |
| 4 | 训练稳定性 | 8.0 | **8.5** | per-module gnorm 可验证梯度隔离 |
| 5 | 可扩展性 | 6.5 | **7.0** | 多相机架构就绪 |
| 6 | 性能设计 | 6.5 | **6.5** | 未变 |
| 7 | 生产就绪度 | 6.0 | **6.5** | 多相机可启用, 但无 inference runtime |
| 8 | 代码质量 | 8.0 | **8.5** | per-module optimizer, 显式 stage gate |
| 9 | 文档 | 4.5 | **5.0** | 多相机 YAML 配置 |
| 10 | 测试 | 2.0 | **2.5** | V1/V4/V5 修复, 但 tests/ 仍空 |

**综合: ~7.6/10** (受 FSDP checkpoint 问题拖累, 否则 ~8.0)

---

## Part 7: 中文摘要

### 历史修复
全部 11 项历史修复无回归。

### v0.10.5 新修复
V1 (val split), V4 (per-module LR), V5 (per-module gnorm) 三项 P1 全部正确修复。analysis_v0_10_5 中 V3 (单步监督) 的声称是**错误的** — FAST/Phase/Affordance 在 v0.10.3 已改为全步监督, 仅 FM expert 是单步 (设计选择)。

### 多相机
实现完成度约 90%。核心链路已打通: 多相机数据读取 → 联合 tokenize → CameraPositionEmbedding → backbone → grounder。默认关闭, 启用只需改 YAML。残留: `camera_names`/`fusion` 配置字段未使用、无 smoke test、显存未评估。

### 新发现
**P0-CRITICAL: FSDP 多卡 checkpoint 加载静默失效。** `checkpointing.py:load_checkpoint` 不使用 `FSDP.state_dict_type` context, 导致 FSDP 模型无法正确加载 FULL_STATE_DICT 格式的 checkpoint。**修复: ~10 行代码。** 单卡训练不受影响。

### 结论
**单卡训练可以立即启动。多卡 (8×H100) 训练需要先修 N1 (~10 行)。** 多相机功能已就绪但需 smoke test 验证后再启用。
