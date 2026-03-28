# HybridVLA v2 v0.10.10 — 问题修复报告

> 修复日期: 2026-03-28
> 对应问题文档: `docs/problem_v0_10_9.md`
> 测试结果: **49/49 通过** (`pytest tests/ -q`)

---

## 一、修复总览

本次修复覆盖 `problem_v0_10_9.md` 中 **第二节 (本地可解决)** 的全部问题项 L-1 至 L-21 (共 21 项)，按严重性 P0 → P3 逐步实施。HPC3 相关问题 (H-1 至 H-21) 需 GPU 集群，本次不涉及。

| 严重性 | 修复数 | 涉及文件 |
|--------|--------|---------|
| **P0** | 1 | hybrid_vla_v2.py |
| **P1** | 6 | hybrid_vla_v2.py, qwen2vl_backbone.py, train_unified.py, mamba_core.py, checkpointing.py |
| **P2** | 8 | types.py, hybrid_vla_v2.py, consistency_loss.py, flow_action_expert.py, train_stage_a.py, world_model/ |
| **P3** | 5 | flow_matching.py, discrete_heads.py, config.py, consistency_loss.py |

---

## 二、P0 修复

### L-1: Optional field contract bug

**问题**: `forward_train()` 用 `"key" in batch` 分支判断可选字段，但 collate 可能产出 `batch[key]=None`，导致 `TypeError`。`_validate_batch()` 使用 `assert` 在生产模式下会被跳过。

**修复**:
- `_validate_batch()`: 所有 `assert` 改为 `raise ValueError(...)`
- 所有可选字段检查改为 `batch.get("key") is not None` 模式
- 涉及字段: `semantic_refresh_steps`, `num_cameras`, `refresh_input_ids`, `phase_labels`, `affordance_labels`, `pixel_values`, `image_grid_thw`, `step_weights`, `embodiment_id`

**文件**: `vla_hybrid_v2/models/hybrid_vla_v2.py:280-345,357-413`

---

## 三、P1 修复

### L-2: `_build_cond_prefix` 静默截断

**问题**: `compressed_slots` 数量变化时，temporal tokens 被静默丢弃。

**修复**: 截断分支增加 `logger.warning()`，输出被截断的 token 数和 `target_c` 值。

**文件**: `vla_hybrid_v2/models/hybrid_vla_v2.py:260-270`

---

### L-3: CameraPositionEmbedding 未 per-batch-item 索引

**问题**: `image_grid_thw` 对所有 batch item 共享同一个 camera embedding 计算，不同分辨率/camera 数的 batch item 会错位。

**修复**: 重写 `CameraPositionEmbedding.forward()`，按 `images_per_batch = total_images // B` 为每个 batch item 独立计算 `cam_indices` 和 `cam_emb`。

**文件**: `vla_hybrid_v2/models/qwen2vl_backbone.py:71-123`

---

### L-4: `evaluate()` 缺少 `dist.barrier()`

**问题**: EMA apply/restore 期间 rank 不同步，FSDP async all-gather 可能导致 rank 间权重不一致。

**修复**: 在 `ema.apply()` 前和 `ema.restore()` 后各加 `dist.barrier()`（仅在 `dist.is_initialized()` 时）。

**文件**: `scripts/train_unified.py:559-567`

---

### L-5: RTC train-infer 分布不一致

**问题**: 训练时 `prev_chunk` 使用当前帧 `cond_prefix`，推理时使用上一帧 — 训练学到的是同观测自一致性而非跨步边界一致性。

**修复**: 训练 RTC 时对 `cond_prefix` 添加小幅噪声 (`noise_scale=0.01`) 以打破自一致性，迫使模型学习跨条件边界的重叠对齐。添加 NOTE 注释说明理想方案需要前一时间步条件。

**文件**: `vla_hybrid_v2/models/hybrid_vla_v2.py:609-622`

---

### L-6: FASTER 推理 `NotImplementedError`

**问题**: Stage C 训练可开启 FASTER，但推理端直接 raise。

**修复**: 保留 `NotImplementedError` 的 fail-fast 行为（明确 FASTER 是 train-only），改善错误信息指导用户设置 `infer.faster.enable=False`。

**文件**: `vla_hybrid_v2/models/hybrid_vla_v2.py:730-735`

---

### L-11: ActionHistoryEncoder 严重过参数化

**问题**: 108M 参数处理 112 个浮点数 (8 actions × 14 dims)，参数/输入比 966K:1。

**修复**:
- 内部 Mamba 从 `d_model=2048, 4 layers` 缩减为 `d_inner=256, 2 layers`
- 新增 `out_proj: Linear(256, 2048)` 将输出投影回核心维度
- 参数量从 ~108M 降至 ~1.7M
- 同步添加 checkpoint 兼容性处理（见下）

**文件**: `vla_hybrid_v2/models/mamba_core.py:519-550`

**Checkpoint 兼容性** (补充修复):
`load_checkpoint()` 新增 shape-mismatch 过滤器，对 pre-v0.10.10 checkpoint 中形状不匹配的 key 自动跳过并 warning，避免 `RuntimeError`。

**文件**: `vla_hybrid_v2/utils/checkpointing.py:124-139`

---

## 四、P2 修复

### L-7: ActionHistoryBuffer 用 `torch.roll` 分配新 tensor

**问题**: T=24 时每次调用 `push()` 都通过 `torch.roll` 创建新 tensor，效率低。

**修复**: 改为 index-based 环形缓冲，`push()` 使用 `self._write_idx % self.max_len` 直接写入；`get()` 仅在 buffer 满时通过 `torch.cat` 重排序。

**文件**: `vla_hybrid_v2/types.py:104-131`

---

### L-8: `loss_total` 用 Python built-in `sum`

**问题**: `sum(losses.values())` 以 int 0 为起始值。

**修复**: 改为 `torch.stack(list(losses.values())).sum()`。

**文件**: `vla_hybrid_v2/models/hybrid_vla_v2.py:693`

---

### L-12: ContrastiveTemporalLoss collapse 风险

**问题**: InfoNCE 将同轨迹近邻时间步当 negative，Mamba 递归输出高度相关，无 collapse 防护。

**修复**: 添加 VICReg-style variance regularisation — 当 `fused_states` 的 per-dimension std 低于 `variance_target=1.0` 时施加惩罚，防止所有表示退化为常数。

**文件**: `vla_hybrid_v2/losses/consistency_loss.py:24-61`

---

### L-13: ActionConsistencyLoss 约束过弱

**问题**: 14D action 投影到 256D 做 cosine，两个独立投影层可输出常向量达到 loss=0。

**修复**: 移除投影层，改为原始 14D 空间的 MSE loss，直接度量 discrete/continuous actions 的一致性。

**文件**: `vla_hybrid_v2/losses/consistency_loss.py:85-99`

---

### L-14: SlowFastAgreementLoss `.detach()` 单向训练

**问题**: `.detach()` 只训练 slow stream 向 fast EMA 靠拢，slow stream 无法反向影响 fast stream。

**修复**: 移除 `.detach()`，改为双向对齐 (bidirectional MSE)。

**文件**: `vla_hybrid_v2/losses/consistency_loss.py:64-82`

---

### L-15: Phase/Affordance 头缺少监督

**问题**: HDF5/LIBERO adapter 均未生成 `phase_labels`/`affordance_labels`，头在训练中无 loss 输入。

**修复**:
- 模型 `__init__` 中添加 `logger.warning()` 提醒用户这些头需要标签数据
- `forward_train()` 中已使用 `batch.get("phase_labels") is not None` 安全跳过无标签场景 (L-1 修复)

**文件**: `vla_hybrid_v2/models/hybrid_vla_v2.py:143-158`

---

### L-16: ExpertMambaBlock 未使用 official Mamba2

**问题**: 即使 `mamba_ssm` 可用，FlowActionExpert 也只走 fallback 实现。

**修复**:
- 添加 `HAS_MAMBA2_MODULE` 检测 (`mamba_ssm.modules.mamba2.Mamba2`)
- `ExpertMambaBlock` 已通过 `HAS_MAMBA_CUDA` 使用 `selective_scan_fn` CUDA kernel 加速
- 完整 Mamba2 block 替换需要适配 AdaRMSNorm conditioning，标记为 future work
- 初始化时 log 检测到的 Mamba2 可用性

**文件**: `vla_hybrid_v2/models/flow_action_expert.py:26-31,266-276`

---

### L-20: `world_model/` 死代码

**问题**: ~1,200 行代码 (RSSM+GNN+visual decoder) 但 `enable=false`，`forward_train()` 从未调用。

**修复**: 迁移至 `vla_hybrid_v2/experimental/world_model/`，原位置保留 backward-compatible re-export `__init__.py`。模型中 import 更新为 `vla_hybrid_v2.experimental.world_model.*`。

**文件**: `vla_hybrid_v2/experimental/world_model/` (新位置), `vla_hybrid_v2/world_model/__init__.py` (兼容层)

---

### L-21: `train_stage_a.py` 危险遗留入口

**问题**: 278 行代码与 `train_unified.py` 分叉，缺少 validation/evaluate、per-module LR 等修正。

**修复**: 替换为 ~30 行薄包装，发出 `DeprecationWarning` 后调用 `scripts.train_unified.train(cfg)`。

**文件**: `scripts/train_stage_a.py`

---

## 五、P3 修复

### L-9: `FlowMatchingLoss.forward()` 接受未使用的 `t` 参数

**修复**: 将 `t` 参数改为 `t=None`（可选），添加注释说明 Rectified Flow 下 target velocity 与 t 无关。

**文件**: `vla_hybrid_v2/losses/flow_matching.py:16-18`

---

### L-10: `_validate_batch()` 使用 `assert`

**修复**: 全部改为 `raise ValueError(...)` (已在 L-1 修复中完成)。

**文件**: `vla_hybrid_v2/models/hybrid_vla_v2.py:280-345`

---

### L-17: FAST 头命名误导

**修复**: 重命名为 `DiscreteActionHead`，保留 `FASTDiscreteHead = DiscreteActionHead` backward-compatible alias。添加 docstring 说明与 pi-0-FAST 的区别。

**文件**: `vla_hybrid_v2/models/discrete_heads.py:12-18,79-80`

---

### L-18: Consistency loss 子权重硬编码

**修复**: `V2ConsistencyLoss.__init__()` 新增参数 `temperature`, `slow_fast_weight`, `action_weight`，取代原硬编码的 `0.1` 和 `0.5`。

**文件**: `vla_hybrid_v2/losses/consistency_loss.py:102-132`

---

### L-19: `config.py` 中 `eval()` 解析 dataclass annotation

**修复**: 替换为 `typing.get_type_hints(cls)`，附带 fallback 到原 `__dataclass_fields__`。

**文件**: `vla_hybrid_v2/config.py:368-396`

---

## 六、测试更新

| 测试文件 | 变更 |
|---------|------|
| `tests/test_control_step.py` | `test_faster_infer_fails_fast_until_implemented`: 保持 `NotImplementedError` 断言 |
| `tests/test_losses.py` | `test_action_consistency`: range 断言从 `[0, 2]` cosine 改为 `>= 0` MSE |

---

## 七、未修复项 (本次范围外)

### 测试覆盖缺失 (L-25 ~ L-34)

新增单测需要更多时间，建议在 HPC3 smoke test 后逐步补充：
- L-25: TriRateMambaCore 专项测试
- L-26: HierarchicalAttentionGrounder 专项测试
- L-27: Official Mamba2 路径测试
- L-28~L-34: 其余测试项

### 代码质量 (L-22 ~ L-24)

- L-22: 未使用 import — 建议 `ruff --fix` 一键清理
- L-23: `forward_train` 过长 (~310 行) — 建议拆分为子方法
- L-24: 无 CI/CD — 建议添加 `.github/workflows/` 和 `.pre-commit-config.yaml`

### HPC3 验证 (H-1 ~ H-21)

需 GPU 集群，建议按以下顺序执行：
1. H-2: 500 步单卡 smoke test
2. H-4: 2×GPU 100 步 smoke test
3. H-7: official vs fallback Mamba benchmark
4. H-3: 5K 步收敛验证

---

## 八、关键数字对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| ActionHistoryEncoder 参数 | 108M | ~1.7M |
| Collapse 防护 | 无 | VICReg variance term |
| `_validate_batch` 模式 | `assert` (可被 -O 跳过) | `raise ValueError` |
| World model 死代码位置 | `vla_hybrid_v2/world_model/` | `experimental/world_model/` |
| `train_stage_a.py` | 278 行独立入口 | ~30 行薄包装 |
| Checkpoint 兼容性 | shape mismatch → crash | 自动跳过 + warning |
| 测试通过率 | 49/49 | 49/49 |
