# HybridVLA v2 工程完善建议 — 参考 OpenPI 最佳实践

> 基于 OpenPI (pi-0/pi-0-FAST/pi-0.5) 的测试套件、CI/CD、数据管线、部署架构的深度分析，为 HybridVLA v2 提供分级工程改进建议。

---

## 目录

- [P0: 生死攸关 — 不做就无法训练](#p0-生死攸关--不做就无法训练)
  - [P0-1: 补全 Stage C RTC/FASTER 训练实现](#p0-1-补全-stage-c-rtcfaster-训练实现)
  - [P0-2: 降低训练风险 — 快速验证管线 + 步长压缩方案](#p0-2-降低训练风险--快速验证管线--步长压缩方案)
  - [P0-3: 添加数据增强](#p0-3-添加数据增强)
- [P1: 工程健壮性 — 防止回归和隐性崩溃](#p1-工程健壮性--防止回归和隐性崩溃)
  - [P1-1: 测试套件设计](#p1-1-测试套件设计)
  - [P1-2: 代码质量工具链](#p1-2-代码质量工具链)
  - [P1-3: 训练管线健壮性](#p1-3-训练管线健壮性)
- [P2: 功能完备性 — 从原型到可部署系统](#p2-功能完备性--从原型到可部署系统)
  - [P2-1: 推理管线实现](#p2-1-推理管线实现)
  - [P2-2: 机器人策略适配层](#p2-2-机器人策略适配层)
  - [P2-3: 评估基准框架](#p2-3-评估基准框架)
- [P3: 锦上添花 — 长期可维护性](#p3-锦上添花--长期可维护性)
- [附录A: 训练步长压缩的详细计算](#附录a-训练步长压缩的详细计算)
- [附录B: Stage C RTC/FASTER 完整实现设计](#附录b-stage-c-rtcfaster-完整实现设计)

---

## P0: 生死攸关 — 不做就无法训练

### P0-1: 补全 Stage C RTC/FASTER 训练实现

#### 当前问题

`configs/train/stage_c.yaml` 声明 `rtc.enable: true` 和 `faster.enable: true`，但 `forward_train()` (`hybrid_vla_v2.py:345-581`) 中**没有任何代码读取这两个配置**。`RTCTrainConfig` (`config.py:194-198`) 和 `FASTERTrainConfig` (`config.py:201-206`) 仅是空壳 dataclass。

#### 什么是 RTC (Real-Time Correction)

RTC 是一种**重叠动作块修正机制**。标准 action chunking 一次生成 H=24 步动作，执行 execution_horizon=8 步后丢弃剩余 16 步。RTC 的核心思想是：

- 新块生成时，其前几步（overlap 区域）与上一块的尾部**部分重叠**
- 重叠区域使用 **inpainting** 融合两个预测，利用上一块的动态一致性信息约束新块的起始段
- 训练时需要学习这种重叠约束

#### RTC 训练实现设计

在 `forward_train()` 的 Stage C 分支中添加 RTC 损失：

```python
# === 在 hybrid_vla_v2.py forward_train() 中, loss_fm 计算之后 ===

if stage == "c" and self.cfg.train.rtc.enable:
    rtc_cfg = self.cfg.train.rtc
    exec_H = rtc_cfg.execution_horizon          # 8
    overlap = int(rtc_cfg.overlap_ratio * exec_H)  # floor(0.333 * 8) = 2
    H = self.cfg.model.action_expert.chunk_horizon   # 24

    # 1. 用当前 cond_prefix 生成一个"前序块"(模拟上一步已有的预测)
    with torch.no_grad():
        prev_chunk = self.action_expert.sample(
            cond_prefix=cond_prefix, proprio_token=proprio_for_expert,
            embodiment_token=emb_for_expert, num_steps=4, solver="euler",
        )  # [B, H, A] — 快速 4-step Euler, 不需要高精度

    # 2. 生成当前块 (已有 expert_out.velocity, 可复用或重新采样)
    current_chunk = noisy_actions + (1.0 - flow_t[:, None, None]) * expert_out.velocity

    # 3. 重叠区域 inpainting 损失
    # prev_chunk 的步 [exec_H - overlap : exec_H] 应与
    # current_chunk 的步 [0 : overlap] 一致
    if rtc_cfg.inpaint_overlap and overlap > 0:
        prev_tail = prev_chunk[:, exec_H - overlap : exec_H]  # [B, overlap, A]
        curr_head = current_chunk[:, :overlap]                  # [B, overlap, A]
        loss_rtc_inpaint = F.mse_loss(curr_head, prev_tail.detach())

        # 4. 边界平滑约束: 当前块前 overlap 步的加速度应平滑
        if overlap >= 2:
            jerk = curr_head[:, 1:] - 2 * curr_head[:, :-1]  # 非标准, 简化
            # 更好的做法: 拼接后检查跨边界加速度
            concat_boundary = torch.cat([prev_tail[:, -1:], curr_head[:, :2]], dim=1)
            accel = concat_boundary[:, 2:] - 2 * concat_boundary[:, 1:-1] + concat_boundary[:, :-2]
            loss_rtc_smooth = accel.pow(2).mean()
        else:
            loss_rtc_smooth = torch.tensor(0.0, device=device)

        losses["loss_rtc"] = (loss_rtc_inpaint + 0.1 * loss_rtc_smooth) * weights.get("rtc", 0.3)
```

**关键配置**:
```python
# config.py RTCTrainConfig 补充字段:
@dataclass
class RTCTrainConfig:
    enable: bool = False
    execution_horizon: int = 8
    overlap_ratio: float = 0.333    # overlap = floor(ratio * exec_H)
    inpaint_overlap: bool = True
    smooth_weight: float = 0.1      # 新增: 边界平滑正则权重
    prev_chunk_steps: int = 4       # 新增: 生成前序块的 ODE 步数 (快速低精度)
```

**loss_weights 更新** (`stage_c.yaml`):
```yaml
loss_weights:
  fast_discrete: 1.0
  phase: 0.5
  affordance: 0.3
  consistency: 0.3
  flow_matching: 1.0
  rtc: 0.3            # 新增
  faster: 0.2         # 新增
```

#### 什么是 FASTER (Fast Adaptive Sampling for Temporal Efficient Refinement)

FASTER 是一种**自适应去噪步数分配策略**，核心观察是：动作块内不同步的预测难度不一样。

- **近端步 (near)**: chunk 的前 `near_ratio * H` 步（约步 0-7），即将执行，对精度要求**最高** → 用更多去噪步
- **远端步 (far)**: chunk 的后部步（约步 8-23），还没执行就会被下一个 chunk 替换 → 用更少去噪步

#### FASTER 训练实现设计

FASTER 改变的是**训练时的损失加权**，而非架构。它教模型对近端步投入更多注意力：

```python
# === 在 hybrid_vla_v2.py forward_train() 中, flow matching 损失计算时 ===

if stage == "c" and self.cfg.train.faster.enable:
    faster_cfg = self.cfg.train.faster
    H = self.cfg.model.action_expert.chunk_horizon  # 24
    near_boundary = int(faster_cfg.near_ratio * H)  # floor(0.3 * 24) = 7

    # 1. 构造 per-step 权重: 近端高权重, 远端低权重
    faster_weights = torch.ones(H, device=device)
    # 近端步: 额外乘以 (far_steps / near_steps) 使梯度密度更高
    faster_weights[:near_boundary] *= (faster_cfg.far_steps / faster_cfg.near_steps)
    # 归一化使总权重 = H (不改变整体 loss 量级)
    faster_weights = faster_weights * (H / faster_weights.sum())
    # faster_weights shape: [H] → broadcast to [B, H, A]

    # 2. 替换 step_weights 或与已有的 step_weights 相乘
    if step_weights is not None:
        combined_weights = step_weights * faster_weights.unsqueeze(0)  # [B, H]
    else:
        combined_weights = faster_weights.unsqueeze(0).expand(B, -1)   # [B, H]

    # 3. 重新计算加权 flow matching loss
    target_velocity = target_actions - noise
    per_step_loss = (expert_out.velocity - target_velocity).pow(2)  # [B, H, A]
    per_step_loss = per_step_loss.mean(dim=-1)  # [B, H]
    loss_fm_faster = (per_step_loss * combined_weights).mean()

    # 4. FASTER 还可以对近端/远端用不同数量的去噪步训练 (多分辨率训练)
    # 近端: 额外用 near_steps=2 步的粗采样做辅助损失 (教模型在少步时也能近端精确)
    if faster_cfg.near_steps < 8:  # 只在 near_steps 较少时添加辅助
        with torch.no_grad():
            coarse_chunk = self.action_expert.sample(
                cond_prefix=cond_prefix, proprio_token=proprio_for_expert,
                embodiment_token=emb_for_expert,
                num_steps=faster_cfg.near_steps, solver="midpoint",
            )
        # 近端粗采样应接近精采样的近端
        loss_faster_aux = F.mse_loss(
            coarse_chunk[:, :near_boundary],
            target_actions[:, :near_boundary],
        )
        losses["loss_faster"] = loss_faster_aux * weights.get("faster", 0.2)

    # 替换原始 loss_fm
    losses["loss_fm"] = loss_fm_faster * weights.get("flow_matching", 1.0)
```

**关键配置**:
```python
@dataclass
class FASTERTrainConfig:
    enable: bool = False
    near_ratio: float = 0.3        # 前 30% 步为近端
    near_steps: int = 2            # 近端辅助损失的粗采样步数
    far_steps: int = 8             # 远端标准步数 (仅用于权重计算比例)
    aux_loss_weight: float = 0.2   # 新增: 近端粗采样辅助损失权重
```

#### FASTER 推理实现

FASTER 推理更直接 — 对近端用更多步去噪，远端用更少步：

```python
# === 在 control_step() 中, 生成新 chunk 时 ===

if self.cfg.infer.faster.enable:
    faster_cfg = self.cfg.infer.faster
    H = self.cfg.model.action_expert.chunk_horizon
    near_boundary = int(faster_cfg.near_ratio * H)

    # 用 far_steps (少) 先快速采样完整 chunk
    coarse = self.action_expert.sample(
        cond_prefix=cond_prefix, proprio_token=proprio_for_expert,
        embodiment_token=emb_for_expert,
        num_steps=faster_cfg.far_steps, solver="midpoint",
    )

    # 对近端步, 用 near_steps (多) 重新精采样, 覆盖粗采样结果
    # 这需要 action_expert.sample 支持 partial horizon — 或者:
    # 方案 B (更简单): 完整采样两次, 拼接
    fine = self.action_expert.sample(
        cond_prefix=cond_prefix, proprio_token=proprio_for_expert,
        embodiment_token=emb_for_expert,
        num_steps=faster_cfg.near_steps + faster_cfg.far_steps, solver="midpoint",
    )
    # 近端用精细, 远端用粗略
    denoised = torch.cat([fine[:, :near_boundary], coarse[:, near_boundary:]], dim=1)
```

#### RTC 推理实现

```python
# === 在 control_step() 中, need_new_chunk 逻辑后 ===

if self.cfg.infer.rtc.enable and runtime_state.current_chunk is not None:
    rtc_cfg = self.cfg.infer.rtc
    overlap = int(0.333 * self.cfg.infer.execution_horizon)

    # 新块的前 overlap 步与旧块的尾部做线性插值融合
    if overlap > 0 and runtime_state.prev_chunk_tail is not None:
        alpha = torch.linspace(1, 0, overlap, device=device)  # [overlap]
        alpha = alpha[None, :, None]  # [1, overlap, 1]
        denoised[:, :overlap] = (
            alpha * runtime_state.prev_chunk_tail +
            (1 - alpha) * denoised[:, :overlap]
        )

    # 保存当前块尾部用于下次融合
    exec_H = self.cfg.infer.execution_horizon
    runtime_state.prev_chunk_tail = denoised[:, exec_H - overlap : exec_H].clone()
```

#### 需要修改的文件清单

| 文件 | 修改内容 |
|------|----------|
| `config.py:194-206` | 补充 RTCTrainConfig/FASTERTrainConfig 字段 |
| `hybrid_vla_v2.py:531-580` | 在 Stage C 分支添加 RTC/FASTER 训练损失 |
| `hybrid_vla_v2.py:601-709` | control_step 中添加 RTC/FASTER 推理逻辑 |
| `types.py` | RuntimeCache 添加 `prev_chunk_tail: Optional[Tensor]` |
| `configs/train/stage_c.yaml` | 更新 loss_weights 加入 rtc/faster |

---

### P0-2: 降低训练风险 — 快速验证管线 + 步长压缩方案

#### 核心问题

当前方案: 400K 步 × 8×H100 ≈ **数天到数周**的计算，没有任何真实数据预实验。这是一个昂贵且高风险的赌博。

#### 方案: 三级验证金字塔

```
                    ┌─────────────┐
                    │  Level 3    │  全量训练 (8×H100)
                    │  400K→150K  │  Stage A 60K + B 70K + C 20K
                    ├─────────────┤
                │   Level 2        │  中规模验证 (1-2×H100)
                │   15K steps      │  Stage A 8K + B 5K + C 2K
                ├──────────────────┤
            │     Level 1              │  烟雾验证 (单 GPU / CPU)
            │     500 steps            │  确认收敛趋势
            └──────────────────────────┘
```

#### Level 1: 烟雾验证 (30 分钟, 单 GPU)

**目标**: 确认损失能下降，梯度无 NaN/explosion

```bash
# 已有 train_smoke_test.py 但用 MockBackbone — 不够
# 需要新的 scripts/quick_validation.py:
python -m scripts.quick_validation \
    --config configs/train/stage_a.yaml \
    --data-dir /path/to/mini_libero_10ep \  # 10 个 episode 即可
    --max-steps 200 \
    --per-device-batch-size 1 \
    --log-interval 10 \
    --no-fsdp
```

**检查清单**:
- [ ] loss_fast 在 200 步内从 ~6.2 (log(512) 随机) 下降
- [ ] loss_fm 在 200 步内从初始值下降 (Stage B)
- [ ] 无 NaN 梯度
- [ ] GPU 内存峰值 < 70GB (留余量给 FSDP)
- [ ] 每步时间合理 (< 5s/step on single H100)

#### Level 2: 中规模验证 (1-2 天, 1-2×H100)

**目标**: 确认三阶段衔接工作，损失持续下降，checkpoint 跨阶段加载正确

```yaml
# configs/train/level2_stage_a.yaml
train:
  max_steps: 8000
  warmup_steps: 500
  learning_rate: 2.0e-4
  global_batch_size: 8       # 小 batch
  per_device_batch_size: 2
  grad_accum_steps: 1        # 无累积
  save_interval: 2000
  output_dir: outputs/level2_stage_a
```

**关键验证项**:
- [ ] Stage A 8K 步后, loss_fast < 初始值的 50%
- [ ] Stage B 加载 Stage A checkpoint 后, loss_fm 能下降
- [ ] Stage C 加载 Stage B checkpoint 后, 所有 loss 继续下降
- [ ] EMA checkpoint 与标准 checkpoint 的 loss 对比
- [ ] Phase/affordance loss 有意义 (如果数据集有标注)

#### Level 3: 全量训练 — 步长压缩方案

**原方案**: Stage A 120K + Stage B 200K + Stage C 80K = **400K 步**

**压缩方案**: 基于以下分析大幅减少步长

##### 压缩依据

1. **OpenPI 全量训练仅 30K 步**, 在 10K+ 小时数据上。HybridVLA 的数据量远小于此, 更长的训练反而过拟合。
2. **Stage A 的目标是建立感知**, 不需要完全收敛 — 它只需要给 Stage B 一个合理的初始化。120K 步对于 LoRA + grounder 的感知预训练过多。
3. **Stage B 的 200K 步是最大的浪费点**。Flow matching expert 的收敛通常在 50-80K 步内完成 (参考扩散模型文献)。知识隔离 (detach) 意味着 expert 和 backbone 独立训练, 不需要漫长的联合适应。
4. **Stage C 缩短到 20K** 足够进行端到端微调。低 LR (3e-5) 下 20K 步已能充分调整。

##### 推荐压缩方案

| 阶段 | 原步数 | 压缩步数 | 压缩比 | 理由 |
|------|--------|----------|--------|------|
| Stage A | 120K | **50K** | 2.4x | 感知预热, 不需完全收敛; LoRA rank=64 在 50K 步内足够学习特征提取 |
| Stage B | 200K | **80K** | 2.5x | Flow expert 独立训练 (detach), 80K 步是扩散模型的典型收敛区间 |
| Stage C | 80K | **20K** | 4x | 低 LR 端到端微调, RTC/FASTER 是轻量附加损失 |
| **总计** | **400K** | **150K** | **2.67x** | 从数周降至约 1 周 |

##### 更激进的压缩: 与 OpenPI 对齐

如果数据量有限 (< 1000 小时), 更激进的方案:

| 阶段 | 步数 | 关键调整 |
|------|------|----------|
| Stage A | **30K** | 提高 LR 到 3e-4, warmup 1500 步 |
| Stage B | **40K** | 提高 LR 到 2e-4, expert_lr_scale=1.0 (不打折) |
| Stage C | **10K** | LR 5e-5, 仅微调 |
| **总计** | **80K** | 约 2-3 天 (8×H100) |

##### 早停策略

不要盲目跑满预定步数。在每个 Stage 设置**早停条件**:

```python
# train_unified.py 中添加早停逻辑:

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def should_stop(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
```

在 `eval_interval` 检查点上评估验证集 loss, 如果连续 5 次 eval 没有改善则提前结束当前 Stage。

##### 具体配置修改

```yaml
# configs/train/stage_a_compressed.yaml
train:
  max_steps: 50000           # 120K → 50K
  warmup_steps: 2000
  learning_rate: 2.5e-4      # 略提高补偿步数减少
  save_interval: 5000
  eval_interval: 2000        # 更频繁评估

# configs/train/stage_b_compressed.yaml
train:
  max_steps: 80000           # 200K → 80K
  warmup_steps: 3000
  learning_rate: 1.5e-4      # 略提高
  expert_lr_scale: 0.7       # expert 给更高 LR
  save_interval: 5000

# configs/train/stage_c_compressed.yaml
train:
  max_steps: 20000           # 80K → 20K
  warmup_steps: 1000
  learning_rate: 5.0e-5      # 略提高
  save_interval: 2000
```

##### Batch Size 与步数的权衡

OpenPI 用 batch_size=32, 30K 步 = ~1M 样本。HybridVLA 用 batch_size=64, 150K 步 = ~9.6M 样本。即使压缩后, HybridVLA 看到的样本数仍是 OpenPI 的近 10 倍。

如果想进一步压缩, 可以**增大 batch size** (如 128) 并按线性缩放 LR:
- batch 64 → 128: LR × 2, 步数 / 2
- 但需要确认 8×H100 的显存能承受

---

### P0-3: 添加数据增强

#### 当前问题

HybridVLA 的 `hdf5_adapter.py` 没有任何图像增强。OpenPI 使用:
- Random crop 95% (训练时随机裁剪到原图 95% 大小)
- Random rotation ±5°
- Color jitter (亮度/对比度/饱和度随机扰动)

#### 实现方案

在 `data/transforms.py` (新文件) 中实现, 由 `hdf5_adapter.py` 调用:

```python
"""Image augmentations for HybridVLA v2 training.

Reference: OpenPI transforms.py — random crop, rotation, color jitter.
"""
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from typing import Optional

class RobotImageAugmentation:
    """Training-time image augmentation for robot manipulation.

    Designed to be applied BEFORE Qwen2-VL processor tokenization,
    operating on PIL Images.
    """

    def __init__(
        self,
        random_crop_scale: float = 0.95,
        random_rotation_degrees: float = 5.0,
        color_jitter: bool = True,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.02,
    ):
        transforms = []
        if random_crop_scale < 1.0:
            transforms.append(T.RandomResizedCrop(
                size=(224, 224),
                scale=(random_crop_scale, 1.0),
                ratio=(0.95, 1.05),
                antialias=True,
            ))
        if random_rotation_degrees > 0:
            transforms.append(T.RandomRotation(degrees=random_rotation_degrees))
        if color_jitter:
            transforms.append(T.ColorJitter(
                brightness=brightness, contrast=contrast,
                saturation=saturation, hue=hue,
            ))
        self.transform = T.Compose(transforms) if transforms else None

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.transform is None:
            return image
        return self.transform(image)
```

在 `hdf5_adapter.py` 的 `__getitem__` 中, 图像从 HDF5 读取后、送入 processor 前调用:

```python
# hdf5_adapter.py __getitem__ 中:
if self.augmentation is not None and self.split == "train":
    pil_image = self.augmentation(pil_image)
```

**配置集成** (`config.py` DataConfig 添加):
```python
@dataclass
class AugmentationConfig:
    enable: bool = True          # 训练时启用
    random_crop_scale: float = 0.95
    random_rotation: float = 5.0
    color_jitter: bool = True

@dataclass
class DataConfig:
    # ... 已有字段
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
```

---

## P1: 工程健壮性 — 防止回归和隐性崩溃

### P1-1: 测试套件设计

#### OpenPI 测试体系参考

OpenPI 的测试结构:
```
tests/
├── conftest.py                 # 共享 fixtures (fake data, mini configs)
├── model_test.py               # 模型前向/反向
├── pi0_test.py                 # pi-0 特定测试
├── tokenizer_test.py           # 分词器正确性
├── transforms_test.py          # 数据变换
├── normalize_test.py           # 归一化往返
├── image_tools_test.py         # 图像处理
├── data_loader_test.py         # 数据加载
├── lora_test.py                # LoRA 适配
├── policy_test.py              # 策略推理
└── download_test.py            # 资源下载
```

CI 命令: `uv run pytest --strict-markers -m "not manual"`
- 自动测试不含 `@pytest.mark.manual` 标记
- 手动测试 (需要 GPU / 大文件) 单独执行

#### HybridVLA 推荐测试结构

```
tests/
├── conftest.py                     # [新建] 共享 fixtures
├── unit/
│   ├── test_normalizer.py          # [P1-Critical] 归一化往返
│   ├── test_grounder.py            # [P1-Critical] 接地器形状
│   ├── test_expert.py              # [P1-Critical] 动作专家形状 + ODE
│   ├── test_mamba_core.py          # [P1-High] 时序核心三流
│   ├── test_discrete_heads.py      # [P1-High] 离散头量化/反量化
│   ├── test_losses.py              # [P1-High] 损失函数数值
│   ├── test_collate.py             # [P1-Medium] collate 函数
│   └── test_config.py              # [P1-Medium] 配置加载
├── integration/
│   ├── test_forward_train.py       # [P1-Critical] 端到端前向+反向
│   ├── test_stage_transition.py    # [P1-Critical] Stage A→B→C checkpoint
│   ├── test_control_step.py        # [P1-High] 推理 control_step
│   └── test_data_pipeline.py       # [P1-High] HDF5 → batch → model
└── markers.py                      # pytest markers 定义
```

#### 核心测试实现详解

##### conftest.py — 共享 Fixtures

```python
"""Shared fixtures for HybridVLA v2 test suite.

Mirrors OpenPI pattern: lightweight configs + fake data generators.
"""
import pytest
import torch
from vla_hybrid_v2.config import (
    HybridVLAv2Config, ModelConfig, TrainConfig, BackboneConfig,
    GrounderConfig, TemporalCoreConfig, ActionExpertConfig, HeadsConfig,
)

# Mini dimensions for fast CPU testing
D = 64       # d_model
D_EXP = 32   # expert d_model
H = 4        # chunk_horizon
A = 7        # action_dim
P = 9        # proprio_dim
T = 4        # sequence_window
B = 2        # batch_size

@pytest.fixture
def mini_config():
    """Minimal config that runs on CPU in < 1 second."""
    return HybridVLAv2Config(
        model=ModelConfig(
            backbone=BackboneConfig(name="mock"),
            grounder=GrounderConfig(
                hidden_size=D, num_latents=8, num_object_slots=4,
                compressed_slots=2, num_layers=2, num_heads=2,
            ),
            temporal_core=TemporalCoreConfig(
                d_model=D, fast_layers=2, medium_layers=1, slow_layers=1,
                d_state=16, fusion_layers=1, fusion_heads=2,
            ),
            action_expert=ActionExpertConfig(
                d_model=D_EXP, num_layers=3, num_heads=2, d_state=8,
                chunk_horizon=H, action_dim=A,
            ),
            heads=HeadsConfig(fast_vocab_size=32),
        ),
        train=TrainConfig(
            sequence_window=T, per_device_batch_size=B,
            semantic_refresh_stride=2, medium_update_stride=1,
        ),
        stage="a",
    )

@pytest.fixture
def dummy_batch():
    """Minimal valid batch dict for forward_train."""
    return {
        "actions": torch.randn(B, T, H, A),
        "proprio": torch.randn(B, T, P),
        "prev_actions": torch.randn(B, T, A),
        "input_ids": torch.randint(0, 1000, (B, 16)),
        "attention_mask": torch.ones(B, 16, dtype=torch.long),
    }

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

##### test_normalizer.py — 归一化往返测试

```python
"""Test normalizer roundtrip: normalize → denormalize = identity."""
import pytest, torch, json, tempfile
from pathlib import Path
from vla_hybrid_v2.data.normalizer import Normalizer

class TestNormalizer:
    def test_min_max_roundtrip(self):
        norm = Normalizer(dim=7, strategy="min_max")
        data = torch.randn(100, 7) * 5 + 3  # 非标准分布
        norm.fit(data)
        normalized = norm.normalize(data)
        recovered = norm.denormalize(normalized)
        assert torch.allclose(data, recovered, atol=1e-5)

    def test_mean_std_roundtrip(self):
        norm = Normalizer(dim=7, strategy="mean_std")
        data = torch.randn(100, 7) * 5 + 3
        norm.fit(data)
        normalized = norm.normalize(data)
        recovered = norm.denormalize(normalized)
        assert torch.allclose(data, recovered, atol=1e-4)

    def test_save_load_persistence(self, tmp_path):
        norm = Normalizer(dim=7, strategy="min_max")
        norm.fit(torch.randn(50, 7))
        norm.save(tmp_path / "stats.json")

        norm2 = Normalizer(dim=7, strategy="min_max")
        norm2.load(tmp_path / "stats.json")
        x = torch.randn(5, 7)
        assert torch.allclose(norm.normalize(x), norm2.normalize(x))

    def test_min_max_range(self):
        norm = Normalizer(dim=3, strategy="min_max")
        norm.fit(torch.randn(200, 3))
        normalized = norm.normalize(torch.randn(10, 3))
        # 多数值应在 [-1, 1] 范围内 (不是所有, 因为测试数据可能超出 fit 范围)
        within_range = ((normalized >= -1.5) & (normalized <= 1.5)).float().mean()
        assert within_range > 0.8
```

##### test_forward_train.py — 端到端集成测试

```python
"""Integration test: forward_train → backward → gradient check.

This is the single most important test — if this passes, the model
can train. If it fails, nothing else matters.
"""
import pytest, torch
from unittest.mock import patch

class TestForwardTrain:
    @pytest.mark.parametrize("stage", ["a", "b", "c"])
    def test_forward_backward_all_stages(self, mini_config, dummy_batch, stage):
        """Core sanity: loss computes and gradients flow for all 3 stages."""
        mini_config.stage = stage
        # Mock backbone to avoid loading 7B weights
        with patch("vla_hybrid_v2.models.hybrid_vla_v2.Qwen2VLBackboneWrapper") as mock_bb:
            mock_bb.return_value = _create_mock_backbone(mini_config)
            from vla_hybrid_v2.models.hybrid_vla_v2 import HybridVLAv2
            model = HybridVLAv2(mini_config)

        losses = model.forward_train(dummy_batch)

        assert "loss_total" in losses
        assert losses["loss_total"].requires_grad
        assert not torch.isnan(losses["loss_total"])
        assert not torch.isinf(losses["loss_total"])

        # Backward should not crash
        losses["loss_total"].backward()

        # At least one parameter should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        assert has_grad

    def test_stage_a_expert_frozen(self, mini_config, dummy_batch):
        """Stage A must NOT produce flow matching loss."""
        mini_config.stage = "a"
        # ... build model ...
        losses = model.forward_train(dummy_batch)
        assert "loss_fm" not in losses
        assert "loss_fast" in losses

    def test_stage_b_has_flow_loss(self, mini_config, dummy_batch):
        """Stage B must produce flow matching loss."""
        mini_config.stage = "b"
        # ... build model ...
        losses = model.forward_train(dummy_batch)
        assert "loss_fm" in losses

    def test_no_nan_gradients(self, mini_config, dummy_batch):
        """No parameter should have NaN gradients after backward."""
        # ... forward + backward ...
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
```

##### test_stage_transition.py — 跨阶段 Checkpoint 测试

```python
"""Test Stage A → B → C checkpoint loading chain.

This verifies the most fragile part of the pipeline: loading a
checkpoint trained in Stage X into a model configured for Stage Y.
"""
import pytest, torch, tempfile
from pathlib import Path

class TestStageTransition:
    def test_a_to_b_checkpoint(self, mini_config, dummy_batch, tmp_path):
        """Stage A checkpoint loads correctly into Stage B model."""
        # 1. Train Stage A for 1 step
        mini_config.stage = "a"
        model_a = build_model(mini_config)
        losses = model_a.forward_train(dummy_batch)
        losses["loss_total"].backward()

        # 2. Save checkpoint
        ckpt_path = tmp_path / "stage_a"
        ckpt_path.mkdir()
        torch.save(model_a.state_dict(), ckpt_path / "model.pt")

        # 3. Load into Stage B
        mini_config.stage = "b"
        model_b = build_model(mini_config)
        state = torch.load(ckpt_path / "model.pt", weights_only=True)
        missing, unexpected = model_b.load_state_dict(state, strict=False)

        # Expert params should be in missing (newly initialized in Stage B)
        expert_missing = [k for k in missing if "action_expert" in k]
        assert len(expert_missing) > 0, "Expert params should be missing from Stage A ckpt"

        # Backbone/grounder params should NOT be missing
        backbone_missing = [k for k in missing if "backbone" in k or "grounder" in k]
        assert len(backbone_missing) == 0, f"Core params missing: {backbone_missing}"

        # Stage B should be trainable
        losses_b = model_b.forward_train(dummy_batch)
        assert "loss_fm" in losses_b
        losses_b["loss_total"].backward()

    def test_b_to_c_checkpoint(self, mini_config, dummy_batch, tmp_path):
        """Stage B checkpoint loads into Stage C without missing keys."""
        # Similar to above, but B → C should have NO missing keys
        # (Stage C uses same modules as B, just unfreezes more)
        ...
```

##### test_expert.py — 动作专家测试

```python
"""Test FlowActionExpert: forward, ODE solvers, AdaRMSNorm."""
import pytest, torch

class TestFlowActionExpert:
    def test_output_shape(self, mini_config):
        from vla_hybrid_v2.models.flow_action_expert import FlowActionExpert
        cfg = mini_config.model.action_expert
        expert = FlowActionExpert(cfg, core_dim=64)
        B, H, A = 2, cfg.chunk_horizon, cfg.action_dim
        out = expert(
            noisy_actions=torch.randn(B, H, A),
            flow_t=torch.rand(B),
            cond_prefix=torch.randn(B, 8, 64),
            proprio_token=torch.randn(B, 64),
            embodiment_token=torch.randn(B, 64),
        )
        assert out.velocity.shape == (B, H, A)

    def test_euler_vs_midpoint_consistency(self, mini_config):
        """Midpoint with many steps should ≈ Euler with many steps."""
        expert = build_expert(mini_config)
        expert.eval()
        cond = torch.randn(1, 8, 64)
        prop = torch.randn(1, 64)
        emb = torch.randn(1, 64)

        euler_32 = expert.sample(cond, prop, emb, num_steps=32, solver="euler")
        midpoint_16 = expert.sample(cond, prop, emb, num_steps=16, solver="midpoint")
        # 两者不需要完全相同, 但应该在同一量级
        assert euler_32.shape == midpoint_16.shape
        # 两者的均值不应差太多 (随机初始化, 只检查不崩溃)
        assert not torch.isnan(euler_32).any()
        assert not torch.isnan(midpoint_16).any()

    def test_adarmsorm_gate_init(self):
        """AdaRMSNorm gate bias should init to +2 (sigmoid ≈ 0.88)."""
        from vla_hybrid_v2.models.flow_action_expert import AdaRMSNorm
        norm = AdaRMSNorm(dim=64, cond_dim=32)
        gate_bias = norm.cond_proj.bias.data[128:]  # 第三个 chunk
        assert torch.allclose(gate_bias, torch.full_like(gate_bias, 2.0))
```

#### pytest 配置

```toml
# pyproject.toml 添加:
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "manual: 需要 GPU 或大文件的测试 (不在 CI 中运行)",
    "slow: 运行时间 > 30 秒的测试",
]
addopts = "--strict-markers -m 'not manual'"
```

---

### P1-2: 代码质量工具链

#### 参考 OpenPI 的工具链

OpenPI 使用:
1. **ruff** (linting + formatting, 替代 flake8 + black + isort)
2. **pre-commit hooks** (每次 commit 自动检查)
3. **GitHub Actions CI** (PR 触发测试 + 代码检查)

#### HybridVLA 推荐配置

##### pyproject.toml

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "I",    # isort
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
]
ignore = ["E501"]  # 行长度由 formatter 控制

[tool.ruff.format]
quote-style = "double"
```

##### .pre-commit-config.yaml (新建)

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

##### .github/workflows/test.yml (新建)

```yaml
name: Tests
on: [pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -e ".[dev]"
      - run: pytest --strict-markers -m "not manual" -v
```

---

### P1-3: 训练管线健壮性

#### 问题 1: 验证集静默禁用

`train_unified.py:450` 中, 找不到验证数据时静默降级:
```python
except (FileNotFoundError, ValueError):
    logger.info("No validation dataset found — eval disabled.")
```

**修复**: 区分"用户选择不验证"和"配置错误":

```python
# 推荐修改:
if cfg.data.val_data_dir is not None:
    # 用户显式指定了验证路径 — 找不到应该报错
    val_dataset = build_dataset(cfg, split="val")
    if val_dataset is None:
        raise FileNotFoundError(
            f"Validation data not found at {cfg.data.val_data_dir}. "
            f"Set data.val_data_dir to null to disable validation."
        )
elif cfg.data.val_ratio > 0:
    # 从训练集自动分割 — 找不到可以降级
    try:
        val_dataset = build_dataset(cfg, split="val")
    except FileNotFoundError:
        logger.warning("Auto-split validation failed — eval disabled.")
        val_dataset = None
```

#### 问题 2: 删除冗余训练脚本

`scripts/train_stage_a.py` (279行) 是 `train_unified.py` (550行) 的功能子集。两个脚本维护同一逻辑会导致分歧。

**修复**: 删除 `train_stage_a.py`, 仅保留 `train_unified.py`。

#### 问题 3: 归一化统计前置依赖

当前必须手动运行 `scripts/compute_stats.py` 后才能训练。如果忘记，训练时 `FileNotFoundError`。

**修复**: 在 `train_unified.py` 中添加自动检测:

```python
# train_unified.py 数据加载前:
stats_dir = Path(cfg.data.data_dir) / "norm_stats"
if not (stats_dir / "action_stats.json").exists():
    logger.info("Normalizer stats not found. Computing automatically...")
    from scripts.compute_stats import compute_and_save_stats
    compute_and_save_stats(cfg)
```

#### 问题 4: 训练指标监控增强

参考 OpenPI 的 PyTorch 训练脚本中的内存分析:

```python
# 每 log_interval 步记录 GPU 内存:
if step % cfg.train.log_interval == 0 and torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    logger.info(f"GPU mem: {allocated:.1f}GB alloc, {reserved:.1f}GB rsv, {peak:.1f}GB peak")
    # 也记录各 loss 分量:
    for k, v in losses.items():
        logger.info(f"  {k}: {v.item():.4f}")
```

---

## P2: 功能完备性 — 从原型到可部署系统

### P2-1: 推理管线实现

#### OpenPI 推理架构参考

```
OpenPI Inference:
  Policy (policy.py)
    ├── input_transforms:  观测 → 模型输入
    ├── model.infer():     前向推理
    └── output_transforms: 模型输出 → 动作

  WebsocketPolicyServer (websocket_policy_server.py)
    ├── _handler(): 每客户端连接处理
    ├── msgpack 序列化
    ├── /healthz 健康检查
    └── 性能计时 (推理耗时 + 往返)
```

#### HybridVLA 推理管线设计

`model.py` 中已有 `semantic_step()` 和 `control_step()` — 缺少的是**编排层**。

```python
# vla_hybrid_v2/infer/policy.py (新建)

"""HybridVLA v2 inference policy — orchestrates dual-frequency control.

Usage:
    policy = HybridVLAPolicy.from_checkpoint("outputs/v2_stage_c/checkpoint-latest")
    policy.reset()  # 新 episode
    for obs in environment:
        action = policy.act(obs)
"""

import time, torch, logging
from pathlib import Path
from typing import Dict, Optional, Any
from vla_hybrid_v2.config import HybridVLAv2Config
from vla_hybrid_v2.models.hybrid_vla_v2 import HybridVLAv2
from vla_hybrid_v2.data.normalizer import Normalizer
from vla_hybrid_v2.types import RuntimeCache

logger = logging.getLogger(__name__)


class HybridVLAPolicy:
    """Dual-frequency VLA policy for real-time robot control.

    Manages the semantic loop (12.5 Hz) and control loop (50 Hz)
    internally, exposing a simple act() interface.
    """

    def __init__(self, model: HybridVLAv2, config: HybridVLAv2Config,
                 action_normalizer: Normalizer, proprio_normalizer: Normalizer,
                 processor=None, device="cuda"):
        self.model = model.eval().to(device)
        self.config = config
        self.action_norm = action_normalizer
        self.proprio_norm = proprio_normalizer
        self.processor = processor
        self.device = device

        # Runtime state
        self.runtime: Optional[RuntimeCache] = None
        self.last_semantic = None
        self.step_count = 0
        self.semantic_interval = int(config.infer.control_hz / config.infer.semantic_hz)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, device="cuda"):
        """Load policy from a training checkpoint directory."""
        ckpt_path = Path(checkpoint_dir)
        config = HybridVLAv2Config.from_yaml(ckpt_path / "config.yaml")
        model = HybridVLAv2(config)
        state = torch.load(ckpt_path / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        # Load normalizers
        action_norm = Normalizer(dim=config.model.action_expert.action_dim)
        action_norm.load(ckpt_path / "action_stats.json")
        proprio_norm = Normalizer(dim=config.model.heads.proprio_dim)
        proprio_norm.load(ckpt_path / "proprio_stats.json")
        return cls(model, config, action_norm, proprio_norm, device=device)

    def reset(self):
        """Call at the start of each episode."""
        self.runtime = self.model.init_runtime(batch_size=1, device=self.device)
        self.last_semantic = None
        self.step_count = 0

    @torch.inference_mode()
    def act(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Single-step inference: observation → action.

        Args:
            observation: dict with keys:
                - "image": PIL Image or np.ndarray [H, W, 3]
                - "proprio": np.ndarray [P]
                - "language_instruction": str
                - "prev_action": np.ndarray [A] (optional)
        Returns:
            action: np.ndarray [A] in original (unnormalized) space
        """
        assert self.runtime is not None, "Call reset() before act()"

        # --- Semantic step (12.5 Hz) ---
        need_semantic = (
            self.last_semantic is None or
            self.step_count % self.semantic_interval == 0
        )
        if need_semantic:
            input_ids, attention_mask, pixel_values, image_grid_thw = (
                self._preprocess_vision(observation)
            )
            self.last_semantic = self.model.semantic_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            self.runtime.refresh_counter += 1

        # --- Control step (50 Hz) ---
        proprio = self._preprocess_proprio(observation["proprio"])
        prev_action = self._preprocess_prev_action(
            observation.get("prev_action")
        )

        control_out = self.model.control_step(
            proprio=proprio,
            prev_action=prev_action,
            semantic_summary=self.last_semantic,
            runtime_state=self.runtime,
        )

        self.step_count += 1

        # Denormalize to original action space
        action = self.action_norm.denormalize(control_out.action.cpu())
        return action.squeeze(0).numpy()

    def _preprocess_vision(self, obs):
        """Convert raw observation to model input tensors."""
        # Use Qwen2-VL processor
        image = obs["image"]
        text = obs.get("language_instruction", "perform the task")
        # ... processor tokenization ...
        # Returns: input_ids, attention_mask, pixel_values, image_grid_thw
        ...

    def _preprocess_proprio(self, proprio):
        proprio_t = torch.tensor(proprio, dtype=torch.float32, device=self.device)
        return self.proprio_norm.normalize(proprio_t).unsqueeze(0)

    def _preprocess_prev_action(self, prev_action):
        if prev_action is None:
            return torch.zeros(1, self.config.model.action_expert.action_dim,
                             device=self.device)
        pa = torch.tensor(prev_action, dtype=torch.float32, device=self.device)
        return self.action_norm.normalize(pa).unsqueeze(0)
```

---

### P2-2: 机器人策略适配层

#### OpenPI 的适配模式

```python
# OpenPI: 每个机器人平台一个 Policy adapter
class ALOHAPolicy:
    input_transforms = [resize, normalize, repack_aloha_obs]
    output_transforms = [unpack_aloha_actions, unnormalize]

class DROIDPolicy:
    input_transforms = [resize, normalize, repack_droid_obs]
    output_transforms = [unpack_droid_actions, unnormalize]
```

核心思想: **模型不知道具体机器人**, 适配层完成坐标空间、传感器布局、动作空间的转换。

#### HybridVLA 推荐适配层

```python
# vla_hybrid_v2/infer/adapters/base.py (新建)

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

class RobotAdapter(ABC):
    """Adapter between robot-specific I/O and model's normalized space."""

    @abstractmethod
    def obs_to_model(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert robot observation to model input format."""
        ...

    @abstractmethod
    def action_to_robot(self, model_action: np.ndarray) -> np.ndarray:
        """Convert model output action to robot command."""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @property
    @abstractmethod
    def proprio_dim(self) -> int: ...


# vla_hybrid_v2/infer/adapters/libero.py (新建)
class LIBEROAdapter(RobotAdapter):
    """Adapter for LIBERO simulation benchmark."""

    def obs_to_model(self, raw_obs):
        return {
            "image": raw_obs["agentview_rgb"],
            "proprio": raw_obs["robot0_joint_pos"],
            "language_instruction": raw_obs.get("language_instruction", ""),
        }

    def action_to_robot(self, model_action):
        return model_action  # LIBERO uses direct joint action

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def proprio_dim(self) -> int:
        return 9
```

---

### P2-3: 评估基准框架

```python
# scripts/evaluate.py (新建)

"""Evaluate HybridVLA v2 policy on a benchmark.

Usage:
    python -m scripts.evaluate \
        --checkpoint outputs/v2_stage_c/checkpoint-latest \
        --benchmark libero_spatial \
        --num-episodes 50 \
        --render
"""

import argparse, logging, torch
import numpy as np
from pathlib import Path
from vla_hybrid_v2.infer.policy import HybridVLAPolicy

logger = logging.getLogger(__name__)

def evaluate(policy, env, num_episodes=50):
    successes = 0
    episode_returns = []

    for ep in range(num_episodes):
        obs = env.reset()
        policy.reset()
        episode_return = 0
        done = False

        while not done:
            action = policy.act(obs)
            obs, reward, done, info = env.step(action)
            episode_return += reward

        if info.get("success", False):
            successes += 1
        episode_returns.append(episode_return)

        logger.info(
            f"Episode {ep+1}/{num_episodes}: "
            f"return={episode_return:.2f}, "
            f"success={info.get('success', False)}, "
            f"running_sr={successes/(ep+1):.1%}"
        )

    return {
        "success_rate": successes / num_episodes,
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "num_episodes": num_episodes,
    }
```

---

## P3: 锦上添花 — 长期可维护性

### P3-1: 清理死代码

| 文件/目录 | 状态 | 建议 |
|-----------|------|------|
| `scripts/train_stage_a.py` | 与 `train_unified.py` 功能重叠 | 删除 |
| `vla_hybrid_v2/world_model/` | ~1,130 行, `enable: false`, 从未调用 | 移至独立分支 `feature/world-model` |
| `config.py` WorldModelConfig | 仅在 enable 时初始化模块但从未调用 | 标注 `# TODO: not implemented` |

### P3-2: 类型检查

OpenPI 使用 `jaxtyping + beartype` 做运行时形状验证。HybridVLA 可以更轻量:

```python
# 在关键函数入口添加断言:
def forward_train(self, batch):
    B, T = batch["actions"].shape[:2]
    H, A = self.cfg.model.action_expert.chunk_horizon, self.cfg.model.action_expert.action_dim
    assert batch["actions"].shape == (B, T, H, A), \
        f"Expected actions shape ({B}, {T}, {H}, {A}), got {batch['actions'].shape}"
```

### P3-3: 文档与示例

参考 OpenPI 的 `examples/` 结构:

```
examples/
├── inference_demo.py          # 加载 checkpoint → 单步推理
├── libero_eval.py             # LIBERO 评估完整示例
└── convert_data.py            # 将自定义数据转为 HDF5 格式
```

### P3-4: Checkpoint 格式改进

OpenPI 使用 SafeTensors 格式 (更安全, 无 pickle 风险)。当前 HybridVLA 使用 `torch.save/load`:

```python
# 推荐: 保存时同时输出 safetensors 格式
from safetensors.torch import save_file, load_file

def save_checkpoint(model, path):
    save_file(model.state_dict(), path / "model.safetensors")
    # 同时保存旧格式兼容:
    torch.save(model.state_dict(), path / "model.pt")
```

---

## 附录A: 训练步长压缩的详细计算

### 训练样本量对比

| 配置 | Batch | 步数 | 样本量 | 相对 OpenPI |
|------|-------|------|--------|-------------|
| OpenPI | 32 | 30K | 960K | 1.0x |
| HybridVLA 原方案 | 64 | 400K | 25.6M | 26.7x |
| 压缩方案 | 64 | 150K | 9.6M | 10.0x |
| 激进方案 | 64 | 80K | 5.1M | 5.3x |
| 激进+大batch | 128 | 40K | 5.1M | 5.3x |

### 为什么可以大幅压缩

1. **HybridVLA 每步的信息量更大**: 24 步窗口 × 多步监督 (T 个时间步的 loss), 每步比 OpenPI 提供更多梯度信号

2. **LoRA 收敛快**: rank=64 的 LoRA 参数量约 90M, 在 50K 步 × batch 64 ≈ 3.2M 样本内通常足够收敛

3. **知识隔离缩短 Stage B**: `cond_prefix.detach()` 意味着 expert 独立训练, 不需要等 backbone 适应。Expert 的 ~500M 参数在 80K 步内可以充分训练

4. **Stage C 微调本质上是 fine-tuning**: LR 3e-5 下的端到端微调, 20K 步已经足够 (类似 NLP 的 fine-tuning 通常只需 3-5 epochs)

### 推荐的步长选择决策树

```
数据量 < 100 episodes (< 50 小时):
  → 激进方案 (80K): 数据不够, 长训练只会过拟合
  → 添加更强的增强和正则化

100 < 数据量 < 1000 episodes:
  → 压缩方案 (150K): 平衡训练充分度和过拟合风险
  → 监控 val loss, 使用早停

数据量 > 1000 episodes (> 500 小时):
  → 接近原方案 (300K): 数据足够, 可以更充分训练
  → 但仍建议从压缩方案开始, 看到收敛后决定是否延长
```

---

## 附录B: Stage C RTC/FASTER 完整实现设计

### 修改总览

```
需要修改的文件:
├── vla_hybrid_v2/config.py              # 补充 config 字段
├── vla_hybrid_v2/types.py               # RuntimeCache 添加 prev_chunk_tail
├── vla_hybrid_v2/models/hybrid_vla_v2.py # forward_train + control_step
└── configs/train/stage_c.yaml           # loss_weights 更新
```

### config.py 修改

```python
@dataclass
class RTCTrainConfig:
    enable: bool = False
    execution_horizon: int = 8
    overlap_ratio: float = 0.333
    inpaint_overlap: bool = True
    smooth_weight: float = 0.1         # 新增
    prev_chunk_steps: int = 4          # 新增

@dataclass
class FASTERTrainConfig:
    enable: bool = False
    near_ratio: float = 0.3
    near_steps: int = 2
    far_steps: int = 8
    aux_loss_weight: float = 0.2       # 新增
```

### types.py 修改

```python
@dataclass
class RuntimeCache:
    temporal_state: TriRateTemporalState = field(default_factory=TriRateTemporalState)
    current_chunk: Optional[Tensor] = None
    chunk_step: int = 0
    refresh_counter: int = 0
    _last_seen_refresh: int = -1
    last_semantic: Any = None
    action_history: Optional[Tensor] = None
    device: Optional[torch.device] = None
    prev_chunk_tail: Optional[Tensor] = None   # 新增: RTC 用
```

### forward_train 中 Stage C 分支的完整逻辑

```python
# 在 losses["loss_fm"] = ... 之后, losses["loss_consistency"] = ... 之前:

# ---- Stage C: RTC loss ----
if stage == "c" and self.cfg.train.rtc.enable:
    rtc_cfg = self.cfg.train.rtc
    exec_H = rtc_cfg.execution_horizon
    overlap = max(1, int(rtc_cfg.overlap_ratio * exec_H))

    # 生成前序块 (低精度快速采样, detached)
    with torch.no_grad():
        prev_chunk = self.action_expert.sample(
            cond_prefix=cond_prefix,
            proprio_token=proprio_for_expert,
            embodiment_token=emb_for_expert,
            num_steps=rtc_cfg.prev_chunk_steps,
            solver="euler",
        )

    # 当前块的预测 (从 expert_out 恢复)
    current_pred = noisy_actions + (1.0 - flow_t[:, None, None]) * expert_out.velocity

    # Overlap inpainting loss
    prev_tail = prev_chunk[:, exec_H - overlap : exec_H].detach()
    curr_head = current_pred[:, :overlap]
    loss_rtc = F.mse_loss(curr_head, prev_tail)

    # 边界平滑 (可选)
    if rtc_cfg.smooth_weight > 0 and overlap >= 1:
        boundary = torch.cat([prev_tail[:, -1:].detach(), curr_head[:, :1]], dim=1)
        accel = boundary[:, 2:] - 2 * boundary[:, 1:-1] + boundary[:, :-2]
        if accel.numel() > 0:
            loss_rtc = loss_rtc + rtc_cfg.smooth_weight * accel.pow(2).mean()

    losses["loss_rtc"] = loss_rtc * weights.get("rtc", 0.3)

# ---- Stage C: FASTER weighted FM loss ----
if stage == "c" and self.cfg.train.faster.enable:
    faster_cfg = self.cfg.train.faster
    H = self.cfg.model.action_expert.chunk_horizon
    near_boundary = max(1, int(faster_cfg.near_ratio * H))

    # 构造 per-step 权重
    faster_w = torch.ones(H, device=device)
    far_ratio = max(faster_cfg.far_steps, 1) / max(faster_cfg.near_steps, 1)
    faster_w[:near_boundary] *= far_ratio
    faster_w = faster_w * (H / faster_w.sum())  # 归一化

    # 重新计算加权 FM loss (替换之前的均匀权重)
    target_v = target_actions - noise
    per_step = (expert_out.velocity - target_v).pow(2).mean(dim=-1)  # [B, H]
    losses["loss_fm"] = (per_step * faster_w.unsqueeze(0)).mean() * weights.get("flow_matching", 1.0)
```

### 实现优先级

| 组件 | 复杂度 | 依赖 | 建议 |
|------|--------|------|------|
| FASTER 训练 (加权 FM loss) | 低 (~20 行) | 无 | **先实现**: 仅修改损失权重, 无新模块 |
| RTC 训练 (overlap inpainting) | 中 (~40 行) | 无 | **再实现**: 需要额外一次 expert 采样 |
| FASTER 推理 (自适应步数) | 中 (~30 行) | P2-1 推理管线 | 与推理管线一起实现 |
| RTC 推理 (chunk 融合) | 低 (~15 行) | P2-1 推理管线 | 与推理管线一起实现 |

---

## 总结: 实施路线图

```
Week 1 (P0):
├── P0-3: 添加数据增强 (transforms.py + hdf5_adapter 集成)
├── P0-2 Level 1: 快速烟雾验证 (quick_validation.py)
└── P0-1 FASTER 训练: 加权 FM loss (~20 行改动)

Week 2 (P0 + P1):
├── P0-1 RTC 训练: overlap inpainting loss (~40 行改动)
├── P0-2 Level 2: 中规模三阶段验证 (1-2 GPU)
├── P1-1: 核心测试 (conftest + test_normalizer + test_forward_train)
└── P1-2: ruff + pre-commit 配置

Week 3 (P1 + P2):
├── P1-1: 完整测试套件 (剩余 unit + integration tests)
├── P1-3: 训练管线健壮性修复 (验证集, 自动stats, 冗余脚本)
├── P2-1: 推理管线 (policy.py)
└── P2-2: LIBERO 适配器

Week 4 (P2 + P3 + 训练):
├── P2-3: 评估框架 (evaluate.py)
├── P0-2 Level 3: 全量训练 (压缩方案 150K 步)
├── P3-1: 清理死代码
└── P3-4: SafeTensors 支持
```

---

*本文档基于 OpenPI 公开代码库的工程最佳实践，为 HybridVLA v2 v0.10.5+ 提供可操作的改进建议。所有代码示例已针对 HybridVLA 的具体架构和接口定制。*
