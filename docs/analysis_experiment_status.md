# HybridVLA v2 实验状态分析

> 分析日期: 2026-03-29
> 代码版本: v0.10.9 (fix_review_v3 后)
> 代码规模: 10,772 行 Python (65 文件)
> 训练状态: **尚未启动正式训练**

---

## 1. 从 v0.6 到 v0.10.9 的演进

自 v0.6 以来，代码库经历了 **15+ 轮 audit-fix 迭代**，从 4,823 行增长到 10,772 行 (+123%)。

### 1.1 重大变更清单

| 版本区间 | 关键变化 |
|---------|---------|
| v0.7-v0.9 | MambaBlock 添加 `res_scale` 防激活爆炸；pre-norm 修复；conv1d 因果切片修正；`force_fallback` 配置项；ActionHistoryEncoder 重构；`proprio_dim` 与 `action_dim` 解耦；`ControlStepOutput` 替代 `ActionExpertOutput`；`ActionHistoryBuffer` 改用索引（去 `torch.roll`）；`RuntimeCache` 使用单调计数器替代 `id()` 比较 |
| v0.10.0-v0.10.2 | 7 项旧 bug 修复；schema/normalizer/HDF5 adapter/collate 数据层；affordance 可配置 |
| v0.10.3 | **关键**: processor 连接、HDF5 图像读取、多步监督、统一训练脚本 `train_unified.py`、eval loop |
| v0.10.4 | pixel_values resize（不修就崩）；显式 stage 门控；sanity check |
| v0.10.5 | val split 过滤；per-module LR (backbone×0.1, expert×0.5)；per-module grad_norm；smoke test B/C 断言 |
| v0.10.6 | 多相机支持；compressed configs |
| v0.10.9 | **5 个 FSDP bug 修复**: EMA/FSDP name prefix、per-module LR grouping、EMA init ordering、eval deadlock、grad norm timing |
| v0.10.10 (linter) | ConsistencyLoss 修复（VICReg variance + bidirectional slow-fast + MSE action consistency）；配置化 sub-weights |

### 1.2 新增基础设施

| 组件 | 状态 | 文件 |
|------|------|------|
| 统一训练脚本 (A/B/C) | ✅ | `scripts/train_unified.py` (612行) |
| HDF5 数据加载 | ✅ | `data/hdf5_adapter.py` (466行) |
| LIBERO 专用加载 | ✅ | `data/libero_hdf5_adapter.py` (526行) |
| 数据归一化 | ✅ | `data/normalizer.py` (185行) |
| Collate + Schema | ✅ | `data/collate.py` + `data/schema.py` |
| FSDP wrapping | ✅ | `utils/distributed.py` (163行) |
| EMA (FSDP-aware) | ✅ | `utils/ema.py` (147行) |
| Checkpoint save/load | ✅ | `utils/checkpointing.py` (158行) |
| LIBERO eval rollout | ✅ | `libero_hybrid/scripts/eval_libero_rollout.py` |
| LIBERO stats compute | ✅ | `libero_hybrid/scripts/compute_libero_stats.py` |
| Smoke test (Stage A/B/C) | ✅ | `scripts/train_smoke_test.py` (313行) |

---

## 2. 实验现状

### 2.1 已完成

| 步骤 | 状态 | 证据 |
|------|------|------|
| 模型代码编写 | ✅ 100% | 10,772 行，65 文件 |
| FSDP bug 修复 | ✅ | v0.10.9 fix_review_v3 确认 |
| Consistency loss 修复 | ✅ | v0.10.10 linter 修正 |
| LIBERO 数据格式适配 | ✅ | `libero_hdf5_adapter.py` |
| 训练配置生成 | ✅ | `outputs/libero_hybrid/libero_spatial/stage_a/resolved_config.yaml` |
| Eval 脚本准备 | ✅ | `eval_libero_rollout.py` |

### 2.2 未完成（阻塞训练启动）

| 步骤 | 状态 | 阻塞原因 |
|------|------|---------|
| LIBERO 数据下载 | ❓ 未确认 | 需确认 `/path/to/LIBERO/datasets` 是否存在 |
| Normalizer stats 计算 | ❌ | `normalizer_stats/` 目录为空 |
| 2×GPU smoke test | ❌ | v0.10.9 分析要求但未执行 |
| Official vs fallback benchmark | ❌ | 未执行 |
| **正式 Stage A 训练** | ❌ | 上述前置步骤未完成 |

---

## 3. v0.10.9 最终分析的关键发现

### 3.1 评分: 6.9/10

| 维度 | 分数 | 关键问题 |
|------|------|---------|
| 正确性 | 7.5 | FSDP bugs 已修，consistency loss 已修（v0.10.10） |
| 完整性 | 8.0 | 三阶段 + RTC + FASTER + EMA + inference pipeline 完整 |
| **效率** | **5.5** | **token-by-token loop 19,536 calls/forward** |
| 可维护性 | 6.5 | ~16% dead code (world_model/ + train_stage_a.py) |
| 鲁棒性 | 7.0 | 错误处理完善但从未端到端 GPU 验证 |
| 架构设计 | 6.5 | ActionHistoryEncoder 108M 过参数化 |

### 3.2 训练时间估算

| 路径 | Stage A | 三阶段总计 | 瓶颈 |
|------|---------|-----------|------|
| Official Mamba2 (当前) | ~17 天 | ~55-64 天 | 19,536 Python loop/forward |
| **Fallback path (推荐)** | **~8-10 天** | **~28-35 天** | JIT scan 但消除 loop + 启用 checkpointing |
| 理论最优 (fork mamba_ssm) | ~5-7 天 | ~18-22 天 | 序列并行 + CUDA kernel |

### 3.3 8 个未关闭问题

| ID | 严重度 | 问题 | 推荐处理 |
|----|--------|------|---------|
| R1 | P1 | Token-by-token loop | **切换 `mamba_impl: fallback`** 训练 |
| R2 | P1 | Activation checkpoint 断裂 | 随 R1 同时解决 |
| R3 | P2 | ActionHistoryEncoder 108M | 缩减到 d=256, 2 layers (~1.7M) |
| R4 | P2 | ContrastiveTemporalLoss collapse | ✅ 已修 (v0.10.10 VICReg) |
| R5 | P2 | ActionConsistencyLoss collapse | ✅ 已修 (v0.10.10 MSE) |
| R6 | P2 | world_model/ 死代码 | 已移至 `experimental/` |
| R7 | P3 | train_stage_a.py 冗余 | 已改为 train_unified wrapper |
| R8 | P3 | 未使用 imports | 低优先级 |

---

## 4. 距离第一次真正训练的最短路径

### 4.1 推荐立即执行（~1 天）

```bash
# Step 1: 确认数据存在
ls /path/to/LIBERO/datasets/libero_spatial/*.hdf5

# Step 2: 计算归一化统计量
python -m libero_hybrid.scripts.compute_libero_stats \
  --data-dir /path/to/LIBERO/datasets \
  --suite libero_spatial \
  --output-root outputs/libero_hybrid

# Step 3: Dry-run 验证配置
python -m libero_hybrid.scripts.train_libero \
  --stage a --data-dir /path/to/LIBERO/datasets \
  --suite libero_spatial --output-root outputs/libero_hybrid --dry-run

# Step 4: 单 GPU smoke test (fallback path)
python -m scripts.train_smoke_test --steps 20 --stage a
python -m scripts.train_smoke_test --steps 10 --stage b

# Step 5: 启动 Stage A 训练 (fallback path, 推荐)
# 在 configs/train/stage_a.yaml 或 temporal_core 配置中确认:
#   mamba_impl: fallback
python -m libero_hybrid.scripts.train_libero \
  --stage a --data-dir /path/to/LIBERO/datasets \
  --suite libero_spatial --output-root outputs/libero_hybrid
```

### 4.2 关键配置建议

```yaml
# 推荐 Stage A 配置修改
model:
  temporal_core:
    mamba_impl: fallback  # 消除 Python loop + 启用 checkpointing
    # 如果显存允许可尝试:
    # fast_layers: 12  # 减少 20→12 层加速
    # slow_layers: 6   # 减少 10→6 层加速

train:
  per_device_batch_size: 4  # fallback 启用 checkpointing 后可增大
  grad_accum_steps: 2       # 保持 global_batch = 4*8*2 = 64
  max_steps: 60000          # 先跑一半看 loss 趋势
  save_interval: 2000
  eval_interval: 5000
```

---

## 5. 总体评估

### 5.1 工程成熟度

代码库经历了 **27+ 轮 audit-fix 迭代**，从最初的 2,660 行增长到 10,772 行。这是一个**架构完整、bug 修复充分、但从未真正在 GPU 上训练过的系统**。

| 优势 | 风险 |
|------|------|
| 三阶段训练设计完整 (A/B/C) | 从未在真实数据上验证 loss 下降 |
| FSDP 5 个 bug 全部修复 | 可能还有未触发的分支 bug |
| 数据 pipeline 完整 (LIBERO HDF5) | 未确认数据加载正确性 |
| EMA FSDP-aware | EMA 从未被真正运行过 |
| Per-module LR 已实现 | 未调过超参数 |

### 5.2 核心建议

1. **立即用 fallback path 开始 Stage A 训练**。不要等 fork mamba_ssm 的序列并行方案。fallback 路径综合效率（消除 Python loop + 启用 checkpointing + 更大 batch）可能反而更好。

2. **先跑 5K 步看 loss 趋势**。如果 FAST CE loss 明显下降就说明数据 pipeline 和模型是对的。如果 loss 平坦或 NaN，优先检查 LR、数据归一化、梯度分布。

3. **ActionHistoryEncoder 缩减可以并行做**，不阻塞训练。108M→1.7M 节省 ~5% 参数和显存。

4. **World model 部分 (`experimental/`) 暂不启用**。等 VLA 基线训练成功后再考虑集成。

---

*代码已到 "可以训练" 的状态。最大的差距不是代码质量，而是 **还没有人按下那个训练按钮**。推荐立即用 LIBERO-Spatial Stage A + fallback path 启动第一次真正的训练，哪怕只是 5K 步的验证性运行。*
