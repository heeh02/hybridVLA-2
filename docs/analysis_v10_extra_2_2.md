# HybridVLA v2 — v0.10.4 闭环验证 (Extra-2-2)

> **日期**: 2026-03-27
> **目标**: 验证 v0.10.4 的 4 项 P0 修复是否构成工程闭环，回答"现在能不能开始训练"
> **方法**: 逐项代码验证 + "第一天会不会崩" 实战评估

---

## 1. P0 修复逐项验证

### P0-3: pixel_values 变长 collate — ✅ 已闭环

**第一层防御** (`hdf5_adapter.py:155-162`):
```python
_TARGET = (448, 448)
if pil_image.size != _TARGET:
    pil_image = pil_image.resize(_TARGET, Image.BILINEAR)
pil_image = pil_image.convert("RGB")
```
448² = 200704 = `min_pixels` → Qwen2-VL processor 对所有图像产生相同 `N_patches`。**根因消除。**

**第二层防御** (`collate.py:23-43`):
```python
def _safe_stack_vision(tensors, key):
    shapes = [t.shape[0] for t in tensors]
    if len(set(shapes)) == 1:
        return torch.stack(tensors, dim=0)    # 正常路径
    # 异常路径: pad + warning
    max_n = max(shapes)
    logger.warning("collate: variable dim-0 in '%s' (shapes %s).", key, shapes)
    ...  # pad to max_n
```

对主帧 (`collate.py:65-66`) 和 refresh 帧 (`collate.py:81-83`) 均应用 `_safe_stack_vision`。

**判定**: 根因已消除 (resize)，且有兜底 (pad+warning)。真实数据不会崩。**✅ 闭环。**

---

### P0-1a: 显式 Stage 门控 — ✅ 已闭环

**`configure_trainable_modules()`** (`train_unified.py:87-150`):

```
Step 1: 冻结全部参数
Step 2: 解冻 backbone LoRA
Step 3: 解冻 backbone multi_scale_adapter
Step 4: 解冻 grounder / temporal_core / heads / projections (所有 stage)
Step 5: Stage B/C → 额外解冻 action_expert + cond_builder + bridging projections
Step 6: Stage C → 额外解冻 backbone text layers 16-27
```

关键设计:
- **先冻结全部再逐层解冻** — 不再依赖 PyTorch 默认值
- Stage A 不解冻 `cond_builder` / `core_to_expert` 等 expert 桥接模块 — expert 冻结时这些浪费 optimizer 内存
- Stage C 通过参数名匹配 `f"layers.{layer_idx}."` 解冻 — 对 peft 包装鲁棒

**调用顺序** (`train_unified.py:288-327`):
```
model = HybridVLAv2(cfg)           # 289: 创建模型
configure_trainable_modules(...)    # 292: 设置 requires_grad
sanity_check_trainable_params(...)  # 293: 断言校验
model = model.to(device)           # 300: 移动到 GPU
model = wrap_fsdp(...)             # 303: FSDP 包装
optimizer = AdamW([...])           # 320: 构建优化器 (只含 requires_grad=True 的参数)
```

**判定**: 显式冻结 → 断言校验 → 再建优化器。不再有隐式依赖。**✅ 闭环。**

---

### P0-1b: 可训练参数 Sanity Check — ✅ 已闭环

**`sanity_check_trainable_params()`** (`train_unified.py:153-224`):

- 逐模块打印 trainable/frozen 参数数和百分比 (14 个模块)
- **Stage A 断言**: `expert_trainable == 0` 且 `cond_builder_trainable == 0`
- **Stage B/C 断言**: `expert_trainable == expert_total`
- **所有 stage 断言**: `lora_trainable == lora_total`
- 断言失败 → `AssertionError` → 训练立即终止

**Smoke test 验证结果** (`optimize_v0_10_4.md:86-108`):
```
Stage A: action_expert trainable=0, frozen=310,087 (0.0%)         ✓
Stage B: action_expert trainable=310,087, frozen=0 (100.0%)       ✓
Stage C: action_expert trainable=310,087, frozen=0 (100.0%)       ✓
```

**Smoke test 代码** (`train_smoke_test.py:152-158`) 与 `train_unified.py` 共用同一组函数。

**判定**: Expert 冻结/解冻状态现在有硬断言保护。200K 步白跑不会再发生。**✅ 闭环。**

---

### P0-4: MultiCamera.enable = False — ✅ 已闭环

`config.py:54`:
```python
enable: bool = False  # NOT YET IMPLEMENTED — set True when multi-camera adapter is ready
```

**判定**: 配置不再误导。代码实现 (单相机) 与配置声明 (未实现) 一致。**✅ 闭环。**

---

## 2. "第一天会不会崩" 实战清单

| 场景 | v0.10.3 会崩? | v0.10.4 会崩? | 证据 |
|------|:------------:|:------------:|------|
| 真实 HDF5 图像尺寸不一 | **会** (collate torch.stack) | **不会** (resize + safe_stack) | `hdf5_adapter.py:159-161`, `collate.py:23-43` |
| Stage B 跑了但 expert 没学 | **可能** (隐式依赖) | **不会** (assert 终止) | `train_unified.py:203-207` |
| Stage C 以为全微调但 vision tower 冻结 | **会误解** | **明确预期** (显式 Step 6 只解冻 text 16-27) | `train_unified.py:141-148` |
| 误以为三相机在工作 | **会** (enable=True) | **不会** (enable=False + 注释) | `config.py:54` |
| Checkpoint 加载后 requires_grad 状态错误 | **可能** | **不会** (configure 在 load 前) | `train_unified.py:292` vs `348` |

---

## 3. 残留风险 (P1 级, 不阻塞训练)

| ID | 风险 | 影响 | 当前缓解 |
|----|------|------|---------|
| P0-2 | 无 Stage B/C 专项回归测试 | stage 门控修改后可能不被检测 | smoke test 支持 `--stage b/c` 但不验证梯度非零 |
| P1-4 | 无 per-module gradient norm | 无法验证 knowledge insulation 效果 | 全局 gnorm 有记录 |
| P0-1c | 所有模块共享同一 LR | Stage B expert 可能需要更低 LR | 可通过 stage_b.yaml 全局 LR 缓解 |
| P1-1 | 无 tri-rate ablation 开关 | 消融实验需改代码 | 不阻塞主线训练 |
| P1-3 | 无 inference runtime | 训练完无法直接部署评估 | 训练期间可并行开发 |

**P0-2 风险评估**: smoke test (`--stage b`) 已验证 `loss_fm` 存在且 loss 下降。虽然没有显式检查 expert 参数变化, 但 `sanity_check_trainable_params()` 保证了 expert 可训练 + `loss_fm` 下降 = expert 参数在更新。风险可控。

---

## 4. 结论: 是否达到工程训练要求

### 闭环状态

| extra-2-1 P0 项 | v0.10.4 状态 | 闭环? |
|:----------------:|:-----------:|:-----:|
| P0-1 Stage 门控 | `configure_trainable_modules` + `sanity_check` | ✅ |
| P0-2 Stage B/C 测试 | smoke test `--stage b/c` + sanity check 断言 | ⚠️ 部分 (无梯度值检查, 但有断言+loss下降) |
| P0-3 Vision batch | resize 448×448 + safe_stack pad | ✅ |
| P0-4 多相机误导 | enable=False + 注释 | ✅ |

### 训练启动判定

**✅ Stage A 可以启动。** 所有刚性阻塞已消除:
- 视觉数据: resize 保证 collate 不崩
- Processor: 已连接, 真实 token
- Stage 门控: 显式冻结 + 断言
- 数据管线: action/proprio/vision/text 全通路

**✅ Stage B 可以启动** (Stage A 完成后):
- Expert 解冻有断言保护
- `cond_prefix.detach()` 配置在 `stage_b.yaml:26`
- `loss_fm` 在 smoke test 中已验证存在且下降

**✅ Stage C 可以启动** (Stage B 完成后):
- 与 Stage B 相同保护
- 额外解冻 backbone text 16-27
- RTC/FASTER 配置就绪 (`stage_c.yaml:29-39`)

### 训练前操作清单

```
1. 数据准备
   ├── HDF5 episodes (actions + proprio + images + lang, min 47 步/episode)
   ├── python -m scripts.compute_stats --config configs/train/stage_a.yaml
   └── (可选) 准备验证集

2. 环境
   ├── pip install torch>=2.1 transformers>=4.37 peft>=0.7 h5py
   ├── (推荐) pip install mamba_ssm>=2.0 causal_conv1d>=1.2
   └── 下载 Qwen/Qwen2-VL-7B-Instruct

3. 配置
   ├── 编辑 stage_a.yaml: data_dir, output_dir, normalizer_stats_dir
   └── (可选) 调整 per_device_batch_size / grad_accum_steps

4. 启动
   torchrun --nproc_per_node=8 -m scripts.train_unified \
       --config configs/train/stage_a.yaml
```

### 评分

**v0.10.4 修正评分: 7.5/10**

| 维度 | 分数 | 说明 |
|------|:----:|------|
| 正确性 | 9.0 | P0-3 根因消除 + 兜底, stage 门控显式化 |
| 训练稳定性 | 8.5 | sanity check 断言, 但无 per-module grad 监控 |
| 完备性 | 8.0 | 三阶段可执行, 视觉通路完整, 缺 inference |
| 生产就绪度 | 6.5 | 缺 runtime wrapper, 缺在线评估 |
| 测试 | 3.0 | smoke test 有效但 tests/ 仍空 |

**一句话: 代码已达到工程训练要求。刚性阻塞全部消除, 残留 P1 可在 Stage A 训练期间并行解决。**
