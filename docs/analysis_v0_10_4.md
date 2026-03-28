# HybridVLA v2 — v0.10.4 Analysis (Rigorous Cross-Audit)

> **前序**: v0.10.3 analysis 被外部专家指出"过于乐观"。本报告对照 `analysis_v10_extra_2_1.md` 逐项验证，以"真实训练第一天会不会崩"为标准重新评估。
>
> **Date**: 2026-03-27

---

## Part 1: GPT extra-2-1 逐项验证

### P0-1: Stage 门控隐式依赖默认值 — **TRUE, 确认 P0**

**代码证据** (`train_unified.py:147-157`):
```python
if stage == "a":
    for p in model.action_expert.parameters():
        p.requires_grad = False         # ← 显式冻结
elif stage == "b":
    logger.info("Stage B: all trainable ...")   # ← 什么都不做
else:
    logger.info("Stage C: full fine-tune.")     # ← 什么都不做
```

**问题 1a — 隐式解冻**: Stage B/C 通过"不冻结"来实现"解冻"，依赖 PyTorch 默认 `requires_grad=True`。一旦模型初始化逻辑变更（例如某处默认冻结 expert），Stage B/C 会静默失效。

**问题 1b — Stage C "全微调"名不副实** (`qwen2vl_backbone.py:110-134`):
```python
def _apply_freeze(self, freeze_vision, freeze_until):
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

这些冻结在 `__init__` 中**无条件执行**，Stage C 的 `else` 分支**不会覆盖**。所以 Stage C 实际上是 "除 vision tower + text 0-15 + embed_tokens 以外的 fine-tune"，而非日志声称的 "full fine-tune"。

**问题 1c — 无 per-module LR** (`train_unified.py:171-191`): 所有可训练参数共享同一学习率。Stage B 的设计意图（expert 低 LR / backbone 高 LR）无法实现。

**问题 1d — 无可训练参数校验**: 只打印总数（line 159-162），不分模块验证。Expert 可以在完全冻结状态下跑完 200K 步而不被发现。

**GPT 推荐的三个缺失函数**:

| 函数 | 用途 | 当前状态 |
|------|------|:-------:|
| `configure_trainable_modules(model, stage)` | 按 stage 显式设置 requires_grad | **不存在** |
| `build_optimizer_param_groups(model, stage)` | 按 module 分组，支持 per-module LR | **不存在** |
| `sanity_check_trainable_params(model, stage)` | 断言可训练参数与预期一致 | **不存在** |

---

### P0-2: 无 Stage B/C 回归测试 — **TRUE, 确认 P0**

`tests/` 目录**完全为空**。`train_smoke_test.py` 支持 `--stage b/c` 但不验证：
- `loss_fm` 是否存在（Stage B/C 应产生 flow matching loss）
- Expert 参数是否实际更新
- `cond_prefix.detach()` 是否阻断梯度
- Discrete-continuous consistency 分支是否激活

**风险**: 有人修改 `forward_train()` 的 stage 门控逻辑后，smoke test **不会检测到** Stage B/C 静默退化为 Stage A 行为。

---

### P0-3: Refresh Vision Batching 变长 Shape 会崩 — **TRUE, 确认 P0-Critical**

**这是最严重的问题——真实数据训练第一天就会崩。**

**根因链**:
```
Qwen2-VL Processor
  → 根据图像分辨率动态分 patch
  → pixel_values shape: [N_patches, patch_dim]
  → N_patches 因图像不同而不同
       ↓
hdf5_adapter._process_text_image() (line 155-167)
  → 直接返回 processor 输出，无 shape 控制
       ↓
collate.py (line 40)
  → torch.stack(frame_vals, dim=0)
  → ❌ RuntimeError: shape mismatch
```

**代码验证**:
- `hdf5_adapter.py:157-161`: `self.processor(text=lang, images=pil_image, ...)` — `max_length=256` 只控制 text token，不控制 pixel patch 数
- `config.py:42-43`: `min_pixels=200704, max_pixels=401408` — 不同宽高比图像产生不同 patch 数
- `collate.py:40`: `torch.stack(frame_vals, dim=0)` — 无条件 stack，变长必崩

**此问题在 dummy smoke test 中不暴露**——DummyVLADataset 不产生视觉字段。只有真实 HDF5 数据才触发。

---

### P0-4: MultiCameraConfig.enable=True 但为死代码 — **TRUE, 确认 P0**

- `config.py:54`: `enable: bool = True` — **默认开启**
- 全代码搜索: `camera_keys` **零引用**（除定义处）。`MultiCameraConfig` **零引用**（除定义处）。
- `hdf5_adapter.py:69`: `self.image_key = cfg.data.image_key` — 只用单个 key
- 日志/文档声称 "multi-camera native" 但实现完全不存在

**不是代码崩溃，是认知错误风险。** 用户会以为三相机系统在工作，但实际只有一个。

---

### P1-1: Tri-Rate 无 Clean Ablation 支持 — **TRUE, 确认 P1**

`TemporalCoreConfig` 没有 `temporal_mode` / `use_action_history` / `use_stale_encoding` 字段。`TriRateMambaCore.__init__` 无条件创建三流（`mamba_core.py:613-641`）。无法方便做实验矩阵。

---

### P1-2: 缺少长距相关统计指标 — **TRUE, 确认 P1**

训练日志只记录: loss 分量、全局 grad norm、LR、吞吐量。无 horizon bucket loss、无 steps_since_refresh 分桶、无 medium/slow token 激活频率。

---

### P1-3: 无 Inference Runtime 封装 — **TRUE, 确认 P1**

`infer/__init__.py` 仍为空 stub（1 行 docstring）。模型提供了 `semantic_step()` / `control_step()` / `init_runtime()`，但无封装层、无 PolicyWrapper、无 action denormalization、无 rollout demo。

---

### P1-4: 梯度隔离无验证手段 — **TRUE, 确认 P1**

`hybrid_vla_v2.py:531-533`: `cond_prefix = cond_prefix.detach()` 是唯一隔离手段。训练日志只有全局 `grad_norm`（`train_unified.py:302`），无 per-module gradient norm。无法证明 backbone/grounder 在 Stage B 下确实不受 expert 梯度影响。

---

### GPT 验证总结

| ID | GPT 声明 | 验证结果 | 严重性 |
|----|---------|---------|--------|
| P0-1 | Stage 门控隐式，Stage C 名不副实 | **TRUE** | P0 |
| P0-2 | 无 Stage B/C 回归测试 | **TRUE** | P0 |
| P0-3 | pixel_values 变长 collate 崩溃 | **TRUE — 最严重** | P0-Critical |
| P0-4 | MultiCamera 默认开但死代码 | **TRUE** | P0 (认知) |
| P1-1 | 无 ablation 开关 | **TRUE** | P1 |
| P1-2 | 缺长距指标 | **TRUE** | P1 |
| P1-3 | 无 runtime 封装 | **TRUE** | P1 |
| P1-4 | 梯度隔离不可验证 | **TRUE** | P1 |

**GPT 准确率: 8/8 全部正确。** 我先前的分析确实过于乐观——将"代码结构合理"等同于"可以训练"，忽视了多个"训练第一天就暴露"的问题。

---

## Part 2: 修正评分

### 先前评分 vs 修正评分

| # | 维度 | v0.10.3 评分 | 修正评分 | Δ | 修正理由 |
|---|------|:-----------:|:-------:|:-:|---------|
| 1 | 设计一致性 | 8.5 | **7.5** | -1.0 | MultiCamera 声明 vs 实现脱节; Stage C "全微调"名不副实 |
| 2 | 正确性 | 9.5 | **8.0** | -1.5 | pixel_values collate 会崩; stage 门控缺校验 |
| 3 | 完备性 | 6.0 | **6.0** | 0 | train_unified + eval 已加，但视觉通路仍有致命 bug |
| 4 | 训练稳定性 | 9.0 | **7.5** | -1.5 | 无 per-module gradient 监控; 梯度隔离不可验证 |
| 5 | 可扩展性 | 7.0 | **6.5** | -0.5 | 无 ablation 开关; 无实验矩阵支持 |
| 6 | 性能设计 | 6.0 | **6.0** | 0 | refresh 重复开销未变 |
| 7 | 生产就绪度 | 6.5 | **5.0** | -1.5 | 无 runtime 封装; vision batch 不稳定 |
| 8 | 代码质量 | 8.0 | **7.5** | -0.5 | 死代码 (camera_keys); 隐式解冻 |
| 9 | 文档 | 4.5 | **4.5** | 0 | 无变化 |
| 10 | 测试 | 1.5 | **1.5** | 0 | tests/ 仍为空 |

**修正综合评分**: 加权后 **6.2/10** (先前 6.8 确实过于乐观)

**核心原因**: 先前评分以"代码结构是否合理"为标准，本次以"真实训练是否能跑"为标准。两个标准的差距即为 0.6 分的修正量。

---

## Part 3: 按优先级排序的修复清单

### 阻塞真实训练（必须在 Stage A 前修复）

| # | ID | 修复项 | 文件 | 行数 | 详细指导 |
|---|----|----|------|------|---------|
| 1 | **P0-3** | pixel_values 统一 resize | `hdf5_adapter.py` | ~20行 | 在 `_read_image()` 后强制 `pil_image.resize((448, 448))`。或在 processor 调用时传 `size={"height": 448, "width": 448}`，保证所有图像输出相同 N_patches。**不修就崩。** |
| 2 | **P0-1a** | 显式 stage 门控函数 | `train_unified.py` | ~40行 | 新增 `configure_trainable_modules(model, stage, cfg)`: Stage A 冻结 expert; Stage B 显式 `p.requires_grad_(True)` on expert; Stage C 解冻 backbone text 16-27（覆盖 `_apply_freeze`）|
| 3 | **P0-1b** | 可训练参数 sanity check | `train_unified.py` | ~20行 | Stage B: assert expert params 可训练且 > 0; Stage C: assert backbone 高层可训练。打印分模块参数数量。|
| 4 | **P0-4** | MultiCamera.enable 改 False | `config.py:54` | 1行 | `enable: bool = False`。加断言: 如果 `enable=True`，raise NotImplementedError。|

**P0 总工作量: ~80 行, ~0.5 天**

### 阻塞论文/评估（可在训练期间并行开发）

| # | ID | 修复项 | 文件 | 行数 |
|---|----|----|------|------|
| 5 | **P0-2** | Stage B/C 最小测试 | `tests/test_stage_gates.py` | ~120行 |
| 6 | **P1-4** | Per-module gradient norm 日志 | `train_unified.py` | ~30行 |
| 7 | **P0-1c** | Per-module optimizer 参数组 | `train_unified.py` | ~40行 |
| 8 | **P1-1** | Tri-rate ablation 开关 | `config.py` + `mamba_core.py` | ~100行 |
| 9 | **P1-2** | 长距指标日志 | `train_unified.py` | ~80行 |
| 10 | **P1-3** | Inference runtime wrapper | `infer/runtime.py` | ~200行 |

---

## Part 4: 关键修复的具体代码指导

### Fix 1: pixel_values 统一 resize (P0-3) — **不修就崩**

**文件**: `vla_hybrid_v2/data/hdf5_adapter.py`

**方案 A（保守，推荐先做）**: 在 `_read_image()` 中强制统一尺寸

```python
@staticmethod
def _read_image(data_grp, image_key: str, frame_idx: int,
                target_size: tuple = (448, 448)) -> Optional[Image.Image]:
    if "images" not in data_grp:
        return None
    images_grp = data_grp["images"]
    if image_key not in images_grp:
        return None
    raw = images_grp[image_key][frame_idx]
    img = Image.fromarray(raw)
    if target_size:
        img = img.resize(target_size, Image.BILINEAR)  # 统一尺寸
    return img
```

这保证所有图像经过 processor 后产生相同的 `N_patches`，`torch.stack` 不会崩。

**方案 B（通用）**: 修改 collate 中 pixel_values 的处理——不 stack，保留 list，backbone 逐样本处理。这更灵活但改动更大。

### Fix 2: 显式 stage 门控 (P0-1a)

**文件**: `scripts/train_unified.py`

在 line 147 前新增函数:

```python
def configure_trainable_modules(model, stage: str, cfg) -> None:
    """Explicitly set requires_grad per module based on training stage."""
    # Start: freeze everything, then selectively unfreeze
    for p in model.parameters():
        p.requires_grad = False

    # Always trainable: grounder, temporal_core, discrete heads, projections
    always_trainable = [
        model.grounder, model.temporal_core,
        model.fast_head, model.phase_head, model.affordance_head,
        model.proprio_proj, model.prev_action_proj,
        model.embodiment_embedding, model.cond_builder,
    ]
    for module in always_trainable:
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True

    # Backbone LoRA: always trainable (LoRA params only)
    for name, p in model.backbone.named_parameters():
        if "lora" in name.lower():
            p.requires_grad = True

    # Stage-specific
    if stage in ("b", "c"):
        for p in model.action_expert.parameters():
            p.requires_grad = True
        for p in model.core_to_expert.parameters():
            p.requires_grad = True

    if stage == "c":
        # Unfreeze backbone text layers 16-27 (override __init__ freeze)
        # This makes Stage C a true "broader fine-tune"
        text_model = getattr(model.backbone.model, "model", model.backbone.model)
        if hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        if hasattr(text_model, "layers"):
            for i, layer in enumerate(text_model.layers):
                if i >= 16:  # unfreeze layers 16-27
                    for p in layer.parameters():
                        p.requires_grad = True
```

### Fix 3: Sanity check (P0-1b)

```python
def sanity_check_trainable_params(model, stage: str) -> None:
    """Assert trainable parameters match stage expectations."""
    module_counts = {}
    for name, p in model.named_parameters():
        module = name.split(".")[0]
        if module not in module_counts:
            module_counts[module] = {"trainable": 0, "frozen": 0}
        if p.requires_grad:
            module_counts[module]["trainable"] += p.numel()
        else:
            module_counts[module]["frozen"] += p.numel()

    for mod, counts in sorted(module_counts.items()):
        logger.info("  %s: %s trainable, %s frozen",
                     mod, f"{counts['trainable']:,}", f"{counts['frozen']:,}")

    # Stage-specific assertions
    expert_trainable = sum(
        p.numel() for p in model.action_expert.parameters() if p.requires_grad
    )
    if stage == "a":
        assert expert_trainable == 0, \
            f"Stage A: expert should be frozen but has {expert_trainable:,} trainable params"
    elif stage in ("b", "c"):
        assert expert_trainable > 0, \
            f"Stage {stage}: expert should be trainable but has 0 trainable params"
```

### Fix 4: MultiCamera 默认 False (P0-4)

**文件**: `config.py:54`

```python
enable: bool = False  # NOT YET IMPLEMENTED — set True when multi-camera adapter is ready
```

---

## Part 5: 修正后的训练启动路径

```
今天 (P0 修复, ~80 行, ~0.5 天):
  Fix 1: pixel_values 统一 resize          ← 不修就崩
  Fix 2: 显式 stage 门控                    ← 隐式依赖太脆弱
  Fix 3: 可训练参数 sanity check            ← 防止 200K 步白跑
  Fix 4: MultiCamera.enable=False           ← 1 行改动
       ↓
P0 修复后:
  准备数据 + compute_stats
  Stage A 开始训练 (120K steps, ~15-22h)
       ↓
Stage A 训练期间并行开发:
  Fix 5: Stage B/C 最小测试
  Fix 6: Per-module gradient logging
  Fix 7: Per-module optimizer param groups
       ↓
Stage A 完成后:
  Fix 8-10 (ablation, metrics, runtime) 可选
  Stage B 开始 (200K steps)
```

---

## Part 6: 中文摘要

### 核心结论

**我先前的分析确实过于乐观。** GPT `analysis_v10_extra_2_1.md` 的 8 项声明经逐项验证**全部正确**。最严重的发现：

1. **P0-3 (Critical)**: `pixel_values` 变长导致 collate `torch.stack` 必崩——真实数据训练第一个 batch 就会 RuntimeError。dummy smoke test 完全不暴露此问题。
2. **P0-1**: Stage B/C 的 expert 解冻依赖 PyTorch 默认行为（"不冻结=可训练"），无 sanity check。Stage C 声称"全微调"但 backbone vision tower + text 0-15 + embeddings 永久冻结。
3. **P0-2**: `tests/` 完全空白，smoke test 不验证 Stage B/C 的关键不变量（loss_fm 存在、expert 参数更新、梯度隔离生效）。
4. **P0-4**: `MultiCameraConfig.enable=True` 但零实现——认知误导。

### 评分修正

| 先前 (v0.10.3) | 修正 (v0.10.4) | 差值 | 原因 |
|:-------------:|:-------------:|:---:|------|
| 6.8/10 | **6.2/10** | -0.6 | "代码结构合理" ≠ "真实训练能跑"。pixel_values 崩溃、stage 门控脆弱、无测试是硬伤 |

### 最高优先级

**4 项 P0 修复（~80 行, 0.5 天）是训练前的刚性前提。** 其中 pixel_values resize 是"不修就崩"，configure_trainable_modules 是"不修可能白跑 200K 步"。

### 对自身分析方法论的反思

先前几轮分析的系统性偏差：
- 过度关注"代码写得是否正确"（数学公式、API 调用、shape 计算），忽视"真实数据路径是否走通"
- 用 dummy smoke test 通过来推断"训练可以跑"，但 dummy 不产生视觉字段、不触发变长 collate、不测试 stage 门控
- 将"代码结构合理"的分数直接赋予"生产就绪度"，二者有本质区别

**修正方法论**: 未来评估应增加"第一天崩溃测试"维度——逐项列出真实训练第一个 batch 会经过的代码路径，验证每条路径在非 dummy 数据上的行为。
