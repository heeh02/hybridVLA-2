# HybridVLA v2 vs OpenPI (pi-0 / pi-0-FAST / pi-0.5) 深度对比分析报告 v2

> 分析日期: 2026-03-28
> HybridVLA 版本: v0.10.7 (~9,870 行代码, 单人开发)
> 对比基线: v0.10.5 时的首次对比报告 (comparsion_between_pi_and_hybridVLA.md)
> OpenPI 版本: 开源发布版 (Physical Intelligence 团队)

---

## 目录

1. [Executive Summary](#1-executive-summary)
2. [v0.10.5 → v0.10.7 改进追踪](#2-v0105--v0107-改进追踪)
3. [架构对比（更新）](#3-架构对比更新)
4. [工程质量对比（更新）](#4-工程质量对比更新)
5. [训练流水线对比（更新）](#5-训练流水线对比更新)
6. [推理与部署对比](#6-推理与部署对比)
7. [仍存在的问题（按严重度分级）](#7-仍存在的问题按严重度分级)
8. [相对 OpenPI 的优势](#8-相对-openpi-的优势)
9. [综合评分与建议](#9-综合评分与建议)

---

## 1. Executive Summary

### 与上次对比报告的核心变化

v0.10.5 → v0.10.7 是一次**工程成熟度的跨越式进步**:

| 维度 | v0.10.5 (上次报告) | v0.10.7 (当前) | 变化 |
|------|-------------------|----------------|------|
| **RTC/FASTER** | 配置空壳, 训练代码未实现 | **训练+推理均已实现** | 从 vaporware → 可用 |
| **推理管线** | `infer/__init__.py` 仅一行 | **LIBERO 策略推理完整实现** (`libero_policy.py` 250+行) | 从零到可用 |
| **测试覆盖** | `tests/` 为空 | **7 个测试文件, 554 行, 35+ test cases** | 从零到基础覆盖 |
| **数据增强** | 完全没有 | **RandomResizedCrop + ColorJitter + Rotation** | 补齐关键缺失 |
| **Benchmark 集成** | 无 | **LIBERO 全链路** (train + eval + stats + validate) | 从原型到可评估 |
| **代码量** | ~5,250 行 | ~9,870 行 (+88%) | 接近翻倍 |

### 当前核心结论

HybridVLA v2 已从 "概念验证原型" 演进为 "可训练、可评估的研究系统"。上次报告中指出的 **P0 级 (生死攸关) 问题已全部修复**。但与 OpenPI 的差距从 "代际差距" 缩小到 "工程深度差距" — 主要体现在部署完备性、多机器人适配、和大规模验证三个方面。

### 关键指标速览 (更新)

| 维度 | HybridVLA v2 (v0.10.7) | OpenPI | 差距状态 |
|------|------------------------|--------|---------|
| **总参数量** | ~9.9B (7.6B 冻结 + 2.3B 可训练) | ~3B | 未变 |
| **RTC/FASTER** | **已实现** (训练+推理) | N/A | **已修复** |
| **推理管线** | LIBERO 策略 + chunk 缓存 + RTC | WebSocket 策略服务器 | **大幅改善**, 仍差部署就绪 |
| **测试覆盖** | 7 文件 / 35+ cases | pytest 完整套件 + CI | **从零改善**, 仍不够 |
| **数据增强** | Crop + Rotation + ColorJitter | 相同三项 | **已对齐** |
| **Benchmark** | LIBERO (单 benchmark) | LIBERO + DROID + ALOHA + ... | 有基础, 差多机器人 |
| **真实数据验证** | 无 | 10K+ 小时 | **未变**, 最大差距 |

---

## 2. v0.10.5 → v0.10.7 改进追踪

### 2.1 上次报告 P0 问题修复状态

| P0 问题 | 上次状态 | 当前状态 | 实现位置 |
|---------|---------|---------|---------|
| **RTC 训练实现** | 配置空壳 | **已实现**: overlap inpainting + boundary smoothness | `hybrid_vla_v2.py:586-621` |
| **FASTER 训练实现** | 配置空壳 | **已实现**: per-step weighted FM loss + near-horizon aux | `hybrid_vla_v2.py:562-637` |
| **数据增强** | 完全缺失 | **已实现**: `RobotImageAugmentation` | `data/transforms.py:28-64` |
| **快速验证管线** | 仅 MockBackbone smoke test | **改善**: 完整 conftest.py + 7 个测试文件 | `tests/` |

### 2.2 上次报告 P1 问题修复状态

| P1 问题 | 上次状态 | 当前状态 |
|---------|---------|---------|
| **测试套件** | 0 个测试 | 7 文件 / 35+ cases (normalizer, losses, expert, forward_train, control_step, checkpoint, eval_config) |
| **推理管线** | 不存在 | `libero_policy.py`: 完整策略封装, config 自动发现, normalizer 资产加载 |
| **Benchmark 集成** | 不存在 | `libero_hybrid/`: train + eval_rollout + compute_stats + validate_hdf5 |

### 2.3 新增代码质量评估

#### RTC 实现质量分析

```python
# hybrid_vla_v2.py:586-621 — Stage C RTC 训练
if stage == "c" and self.cfg.train.rtc.enable:
    # 1. 用当前 cond_prefix 快速生成 "前序块" (4-step Euler)
    with torch.no_grad():
        prev_chunk = self.action_expert.sample(...)
    # 2. Inpainting: curr_head 应匹配 prev_tail
    loss_rtc = F.mse_loss(curr_head, prev_tail)
    # 3. Boundary smoothness: 二阶有限差分加速度惩罚
    accel = boundary[:, 2:] - 2*boundary[:, 1:-1] + boundary[:, :-2]
    loss_rtc += smooth_weight * accel.pow(2).mean()
```

**评价**: 实现忠实于 advice_with_pi.md 的设计。`with torch.no_grad()` 生成前序块避免梯度爆炸, boundary smoothness 用标准二阶有限差分。**一个隐患**: 前序块与当前块共享 `cond_prefix` — 真实推理中两个块的 cond_prefix 来自不同时间步的观测。训练时用相同条件做 inpainting 可能导致 train-inference 不一致。OpenPI 无此组件, 无法直接对比。

#### FASTER 实现质量分析

```python
# hybrid_vla_v2.py:562-584 — per-step weighted FM loss
faster_w = torch.ones(H, device=device)
faster_w[:near_boundary] *= far_ratio  # 近端更高权重
faster_w = faster_w * (H / faster_w.sum())  # 归一化

# hybrid_vla_v2.py:628-637 — near-horizon auxiliary loss
loss_faster_aux = F.mse_loss(
    expert_denoised[:, :near_boundary],
    target_actions[:, :near_boundary],
)
```

**评价**: 上次报告指出辅助损失用了 `torch.no_grad()` 导致零梯度 — **已修复**, 现在用 `expert_denoised` (单步解析解, 可微)。归一化权重保持总和 = H 不改变 loss 量级是好的工程实践。**但**: `cfg.infer.faster.enable` 在推理时仍 raise `NotImplementedError` (`hybrid_vla_v2.py:691-695`)。

#### 推理管线质量分析

`infer/libero_policy.py` 实现了:
- 检查点+配置自动发现 (`resolve_policy_config`)
- Normalizer 资产搜索 (checkpoint-local → output_dir → config)
- multi_camera.enable 一致性校验
- 完整的 obs → action 转换

`hybrid_vla_v2.py:675-809` 的 `control_step()` 实现了:
- Chunk 缓存 + 按需生成 (避免冗余 expert forward)
- RTC overlap 线性插值融合 (`alpha * prev_tail + (1-alpha) * curr_head`)
- Action history 更新
- 语义刷新触发新 chunk

**与 OpenPI 对比**: OpenPI 有 WebSocket 策略服务器 (`serving/websocket_policy_server.py`), 支持多客户端并发, 前缀缓存加速。HybridVLA 的推理是库级 API (需集成到用户脚本), 尚无独立服务化能力。

---

## 3. 架构对比（更新）

### 3.1 整体架构变化

v0.10.7 的架构与 v0.10.5 完全相同 — 没有新增或删除模块。变化集中在**工程层**:

```
                    HybridVLA v2 (v0.10.7)                          OpenPI
    ┌──────────────────────────────────────────┐    ┌──────────────────────────────┐
    │ Qwen2-VL-7B (3584d, LoRA r=64, 28层)     │    │ PaliGemma (SigLIP + Gemma 2B) │
    │ + MultiScaleAdapter [10,18,28] → 2048d   │    │ 单尺度                        │
    │ + CameraPositionEmbedding (≤8 cameras)   │    │                              │
    ├──────────────────────────────────────────┤    ├──────────────────────────────┤
    │ Hierarchical Grounder                     │    │ 无 (直接对接)                  │
    │ 96 latents → 8L → 48→24 压缩             │    │                              │
    ├──────────────────────────────────────────┤    ├──────────────────────────────┤
    │ TriRate Mamba Core                        │    │ 无时序建模                     │
    │ Fast(20L) + Medium(6L) + Slow(10L)       │    │                              │
    │ + CrossAttentionFusion + StaleTime        │    │                              │
    ├──────────────────────────────────────────┤    ├──────────────────────────────┤
    │ FlowActionExpert (18L M-M-A×6, 1536d)    │    │ Gemma 300M (扩散/自回归)       │
    │ + AdaRMSNorm + Midpoint ODE              │    │ Euler ODE / KV-cache AR      │
    │ + RTC overlap ✓  + FASTER weighted ✓     │    │                              │
    ├──────────────────────────────────────────┤    ├──────────────────────────────┤
    │ 混合离散 (512-bin) + 连续 (Flow)          │    │ 分离: pi-0 连续 / FAST 自回归  │
    │ + Consistency Loss (cosine embed space)   │    │                              │
    └──────────────────────────────────────────┘    └──────────────────────────────┘
```

### 3.2 FAST 头: 根本性架构差异 (未修复)

这是上次报告指出但**仍未改变**的关键差异:

| 方面 | HybridVLA FAST | pi-0-FAST |
|------|---------------|-----------|
| 生成方式 | **单次前向** (MLP) | **自回归** (逐 token, KV-cache) |
| 维度间依赖 | 无 (并行输出 `[B, H, A, V]`) | 有 (前 token 条件化后 token) |
| 参数分布 | encoder → step_proj → dim_head | 共享 VLM 词汇表末 128 token |
| 序列建模 | 不建模 | 完整自回归序列建模 |

`discrete_heads.py:30-35`:
```python
def forward(self, fused_state: Tensor) -> Tensor:
    h = self.encoder(fused_state)                    # [B, 768]
    h = self.step_proj(h).view(B * chunk_horizon, -1)  # [B*24, 192]
    logits = self.dim_head(h)                        # [B*24, 14*512]
    return logits.view(B, 24, 14, 512)               # 一次性全部输出
```

**影响**: HybridVLA 的 "FAST" 本质上是一个**独立的分类头**, 丧失了 pi-0-FAST 自回归建模动作维度间依赖的核心优势。对于需要 joint 间协调的精细操作 (如双臂协作), 这种独立维度预测可能产生不协调的动作。

**但这不一定是缺陷**: HybridVLA 的 FAST 头定位是**辅助监督信号** + **一致性约束的锚点**, 而非主要动作生成器。主动作来自 FlowActionExpert (连续 flow matching), FAST 头的作用是提供离散视角的梯度信号增强感知模块训练。从这个角度看, 单次前向的简单 FAST 头是**合理的设计选择**, 降低了推理时自回归的延迟开销。

### 3.3 参数量深度对比

| 模块 | HybridVLA v2 | OpenPI pi-0 | 备注 |
|------|-------------|-------------|------|
| Vision Backbone | 7.6B (冻结) | ~2B (SigLIP) | HybridVLA 3.8× |
| LoRA | ~90M (rank=64, 28层, 7 target modules) | 更少 | HybridVLA 激进 |
| MultiScaleAdapter | ~25M | 无 | HybridVLA 独有 |
| Grounder | ~280M (96 latents, 8L, 2048d, 含 FPN 压缩) | 无 | HybridVLA 独有 |
| Temporal Core | ~800M (36L Mamba-2 @ 2048d + 融合) | 无 | HybridVLA 独有 |
| ActionHistoryEncoder | ~120M (4L Mamba @ 2048d) | 无 | 可能 over-sized |
| Action Expert | ~500M (18L @ 1536d) | ~300M | HybridVLA 1.7× |
| Discrete Heads | ~15M | 共用 VLM | |
| **可训练总计** | **~2.3B** | **~500M** | HybridVLA 4.6× |
| **参数总计** | **~9.9B** | **~3B** | HybridVLA 3.3× |

**关键观察**: HybridVLA 的 ActionHistoryEncoder 用 4 层 Mamba-2 @ 2048d (~120M 参数) 编码 8 个动作。这是上次报告指出的 over-engineering — 一个 2 层 MLP (~2M 参数) 可能就够了。OpenPI 没有显式动作历史编码, 证明这不是成功的必要条件。

---

## 4. 工程质量对比（更新）

### 4.1 测试覆盖 (大幅改善, 仍不完整)

| 测试文件 | 覆盖内容 | test cases |
|---------|---------|------------|
| `conftest.py` | Mock backbone, mini config, batch fixture | 基础设施 |
| `test_forward_train.py` | 三阶段 loss 计算, backward, NaN 检查, RTC/FASTER | 8 |
| `test_expert.py` | AdaRMSNorm gate bias, ODE solver 形状/NaN | 6 |
| `test_normalizer.py` | roundtrip, range, save/load, edge cases | 7 |
| `test_losses.py` | flow matching 公式, consistency 4 组件, discrete | 7 |
| `test_control_step.py` | 推理 control_step 基本功能 | ~3 |
| `test_checkpoint_assets.py` | 检查点资产完整性 | ~3 |
| `test_eval_config_resolution.py` | 评估配置解析 | ~2 |

**与 OpenPI 对比**:

| 维度 | HybridVLA | OpenPI |
|------|-----------|--------|
| 测试文件数 | 7 | 10+ |
| 测试用例数 | ~35 | 50+ |
| 数据管线测试 | 无 | data_loader_test, transforms_test |
| 策略测试 | 无 | policy_test (多机器人) |
| 分词器测试 | 无 | tokenizer_test |
| CI/CD | **无** | GitHub Actions + ruff + pre-commit |
| 标记系统 | **无** | `@pytest.mark.manual` 分层 |

**仍缺失的关键测试**:
1. **Grounder 形状测试**: 96 latents → 压缩 → 24 slots 的维度链路
2. **TriRate Mamba 状态持久化测试**: 跨 step 的 SSM 状态传递是否正确
3. **跨阶段 checkpoint 加载测试**: Stage A→B→C 的 `strict=False` 加载
4. **HDF5 数据管线端到端测试**: 从文件读取到 batch 组装
5. **多相机 tokenization 测试**: CameraPositionEmbedding 是否正确注入

### 4.2 代码质量工具链

| 工具 | HybridVLA | OpenPI |
|------|-----------|--------|
| Linter | **无** | ruff |
| Formatter | **无** | ruff format |
| Type checker | **无** (有注解但未检查) | mypy/pyright |
| Pre-commit hooks | **无** | pre-commit |
| CI/CD pipeline | **无** | GitHub Actions |
| Dependency lock | requirements.txt (无版本锁) | pyproject.toml + uv |

**评价**: 这是 HybridVLA 与 OpenPI 差距最大的维度之一。单人项目可以不做 CI, 但**没有 linter 意味着没有自动化代码质量保障**。随着代码量接近 10K 行, 这会成为维护负担。

### 4.3 Config 安全隐患 (未修复)

`config.py:379`:
```python
if isinstance(ft, str):
    ft = eval(ft, globals(), locals())  # ← 代码注入风险
```

上次报告已指出。`eval()` 在此处用于解析 dataclass 字段类型注解字符串 (如 `"Optional[str]"`)。虽然输入来自代码定义而非用户 YAML, 但仍是不安全的模式。推荐替换为 `typing.get_type_hints()` 或显式类型映射。

### 4.4 训练脚本重复 (部分改善)

`train_unified.py` (583行) 已成为主训练脚本, 但 `train_stage_a.py` (278行) 仍然存在且包含大量重复代码。应删除或标记为 deprecated。`libero_hybrid/scripts/train_libero.py` (173行) 是一个轻量包装, 正确委托给 `train_unified.py` 的逻辑。

---

## 5. 训练流水线对比（更新）

### 5.1 三阶段训练完整性 (关键改善)

```
v0.10.5 (上次):                         v0.10.7 (当前):
Stage A ── 可运行          ✓             Stage A ── 可运行 + 测试覆盖      ✓
Stage B ── 可运行          ✓             Stage B ── 可运行 + 测试覆盖      ✓
Stage C ── 功能残缺        △             Stage C ── RTC+FASTER 已实现     ✓✓
           RTC/FASTER 未实现                        推理 FASTER 未实现     △
```

### 5.2 损失函数完整性

| 损失 | Stage A | Stage B | Stage C | 实现状态 |
|------|---------|---------|---------|---------|
| FAST 离散 CE (多步) | ✓ | ✓ | ✓ | **完整** |
| Phase CE (多步) | ✓ | ✓ | ✓ | **完整** |
| Affordance CE (多步) | ✓ | ✓ | ✓ | **完整** |
| Consistency (InfoNCE + SlowFast + ActionAgreement) | ✓ | ✓ | ✓ | **完整** |
| Flow Matching | ✗ | ✓ | ✓ | **完整** |
| FASTER per-step weighted FM | ✗ | ✗ | ✓ | **新增, 完整** |
| FASTER near-horizon auxiliary | ✗ | ✗ | ✓ | **新增, 完整** |
| RTC overlap inpainting | ✗ | ✗ | ✓ | **新增, 完整** |
| RTC boundary smoothness | ✗ | ✗ | ✓ | **新增, 完整** |

### 5.3 数据管线对比 (改善)

| 方面 | HybridVLA v0.10.5 | HybridVLA v0.10.7 | OpenPI |
|------|-------------------|-------------------|--------|
| 数据格式 | HDF5 | HDF5 + LIBERO HDF5 | LeRobot + RLDS |
| 图像增强 | **无** | **Crop + Rotation + ColorJitter** | 相同三项 |
| 多相机 | 支持但未启用 | **已集成** (CameraPositionEmbedding) | 完整支持 |
| Val split | 无 | **episode ratio / separate dir** | 标准 split |
| Normalization | min_max / mean_std | min_max / mean_std + **stats 持久化** | mean_std + 分位数裁剪 |
| HDF5 校验 | 无 | **validate_libero_hdf5.py** | 内置 |

**仍然缺失**: OpenPI 的分位数裁剪 (quantile clipping) — 比 min_max 更鲁棒于离群值。HybridVLA 的 Normalizer 有 `eps` 保护但没有 percentile 截断。

### 5.4 LIBERO 集成质量分析

新增的 `libero_hybrid/` 模块实现了:

1. **`libero_hdf5_adapter.py`** (526行): 最大的新数据层
   - 支持 LIBERO/robomimic 多 demo 分组结构
   - multi-proprio 拼接 (`proprio_keys` → concat)
   - 多相机逐帧 tokenization
   - 与基础 `hdf5_adapter.py` 共享 Normalizer 接口

2. **`eval_libero_rollout.py`** (411行): 完整的 rollout 评估
   - Per-env 隔离 (每个 LIBERO task 独立 env)
   - 多相机观测获取
   - Chunk 缓存推理 (避免每步重新生成)

3. **`train_libero.py`** (173行): 训练包装
   - Variant 系统 (suite → task 映射)
   - Config 保存用于推理时自动发现

**评价**: 这是一个**良好的 benchmark 集成**, 足以在 LIBERO 上获得可比较的结果。但 OpenPI 的策略适配层是**通用抽象** (`Policy` 基类 + ALOHA/DROID/LIBERO 子类), 而 HybridVLA 的集成是**LIBERO-specific** — 换一个机器人需要重写大部分代码。

---

## 6. 推理与部署对比

### 6.1 推理管线完整性

| 功能 | HybridVLA | OpenPI | 评价 |
|------|-----------|--------|------|
| 模型加载 | `load_checkpoint()` + config 自动发现 | 模型权重 + config | 对等 |
| Chunk 缓存 | ✓ (runtime_state.current_chunk) | ✓ (前缀缓存) | 对等 |
| RTC 推理 | ✓ (线性插值融合) | N/A | HybridVLA 独有 |
| FASTER 推理 | **NotImplementedError** | N/A | **未完成** |
| 策略封装 | `HybridVLALiberoPolicy` (LIBERO only) | `Policy` 基类 (多机器人) | OpenPI 更通用 |
| 服务化 | **无** | WebSocket 策略服务器 | OpenPI 完胜 |
| 动作反归一化 | ✓ | ✓ | 对等 |
| Torch compile | 配置有但未验证 | 无 | 实际均未用 |

### 6.2 FASTER 推理缺失 (剩余 P0 问题)

`hybrid_vla_v2.py:690-695`:
```python
if self.cfg.infer.faster.enable:
    raise NotImplementedError(
        "cfg.infer.faster.enable=True but FASTER inference is not "
        "implemented yet."
    )
```

FASTER 训练已实现 (per-step 加权 + near-horizon 辅助损失), 但推理时的**自适应去噪步数分配**未实现。advice_with_pi.md 中设计的方案 (近端精采样 + 远端粗采样 + 拼接) 尚未编码。这意味着用 Stage C FASTER 训练的模型在推理时无法利用 FASTER 加速。

### 6.3 推理延迟估算

| 组件 | HybridVLA 估算 | OpenPI 估算 |
|------|---------------|-------------|
| Backbone forward | ~20ms (7B, bf16, H100) | ~5ms (2.5B) |
| Grounder | ~3ms (8L, 96 latents) | 0ms |
| Temporal Core (full) | ~15ms (36L Mamba) | 0ms |
| Expert sample (midpoint, 8步) | ~30ms (16 forward × 18L) | ~15ms (8 forward × Euler) |
| **单 chunk 总计** | **~68ms** | **~20ms** |
| Chunk 缓存命中 | ~18ms (仅 temporal) | ~5ms |
| **50 Hz 预算** | 20ms | 20ms |

**关键问题**: HybridVLA 的单 chunk 生成 (~68ms) 大幅超出 50 Hz 控制预算 (20ms)。即使 chunk 缓存避免了大部分 expert 调用, 每次新 chunk 生成仍会产生 ~68ms 的延迟尖峰。Midpoint ODE (8步=16 forward) 是 Euler (8步=8 forward) 延迟的 2×, 在实时场景下可能需要退回到 Euler + 更少步数。

OpenPI 的 ~20ms 总延迟在 50 Hz 预算内, 且通过前缀缓存进一步降低。

---

## 7. 仍存在的问题（按严重度分级）

### 7.1 P0: 阻塞性问题

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| **P0-1** | **FASTER 推理未实现** | Stage C FASTER 训练的模型推理时无法使用 FASTER 加速, 训练投入的 FASTER 损失无法兑现推理收益 | 实现 advice_with_pi.md 中的近端精采样 + 远端粗采样方案 |
| **P0-2** | **真实数据零验证** | 所有架构设计和参数选择均基于理论, 未经过任何真实机器人数据的训练和评估。与 OpenPI 的 10K+ 小时验证差距是最根本的差距 | 在 LIBERO 上完成至少 Level 2 验证 (15K steps, 1-2 GPU) |
| **P0-3** | **RTC 训练-推理不一致** | 训练时前序块与当前块共享 cond_prefix (同一观测), 推理时两块来自不同时间步的观测。这种分布不匹配可能导致 RTC 训练的 inpainting 约束在推理时失效 | 训练时用时间偏移的 cond_prefix 或随机扰动模拟时序差异 |

### 7.2 P1: 工程健壮性问题

| # | 问题 | 影响 | 状态 |
|---|------|------|------|
| **P1-1** | **无 CI/CD** | 代码回归无自动检测; 10K 行代码无持续集成 | 未修复 |
| **P1-2** | **无 linter/formatter** | 代码风格不一致, typo/import 错误无自动捕获 | 未修复 |
| **P1-3** | **config.py 的 `eval()` 调用** | 安全隐患 (虽然风险低, 但是不良实践) | 未修复 |
| **P1-4** | **世界模型死代码** | ~1,130 行代码 (`world_model/`) 始终 `enable: false`, 增加维护负担 | 未清理 |
| **P1-5** | **Grounder/Mamba Core 无测试** | 最复杂的两个组件 (96 latents 压缩 + 36L 三流 Mamba) 没有单元测试 | 未添加 |
| **P1-6** | **train_stage_a.py 重复** | 278 行与 train_unified.py 重复, 容易分叉 | 未清理 |
| **P1-7** | **EMA 不兼容 FSDP** | `EMAModel` 基于 `named_parameters()` 的 shadow 字典在 FSDP full-shard 下可能无法正确匹配参数名, 因为 FSDP 会修改参数名前缀 | 需要用 FSDP 感知的 EMA (如 `torch.distributed.fsdp.ShardedOptimStateDictConfig`) |
| **P1-8** | **official Mamba2 token-by-token 回退** | `_MambaStack.forward()` 在 official 路径下用 Python for-loop 逐 token 调用 `.step()` (丧失 intra-sequence 并行性), 对 L=33 序列是 ~33× 调用开销 | 需要验证性能影响; 考虑用 `forward()` + 手动状态提取 |

### 7.3 P2: 功能完备性

| # | 问题 | 与 OpenPI 差距 |
|---|------|---------------|
| **P2-1** | **单 benchmark** | HybridVLA 仅集成 LIBERO; OpenPI 支持 LIBERO + DROID + ALOHA + Bridge |
| **P2-2** | **无通用 Policy 抽象** | `HybridVLALiberoPolicy` 是 LIBERO-specific; OpenPI 有 `Policy` 基类 |
| **P2-3** | **无策略服务器** | 无 WebSocket/gRPC 推理服务; OpenPI 可直接部署 |
| **P2-4** | **无 JAX 实现** | OpenPI 同时有 JAX (高性能训练) + PyTorch (兼容); HybridVLA 仅 PyTorch |
| **P2-5** | **Normalizer 无分位数裁剪** | 对离群值敏感; OpenPI 用分位数裁剪更鲁棒 |

---

## 8. 相对 OpenPI 的优势

### 8.1 架构创新 (HybridVLA 独有)

| 创新 | 描述 | 学术价值 | 实际价值 (待验证) |
|------|------|---------|-------------------|
| **三频率时序核心** | 匹配机器人控制物理频率带 (50/25/12.5 Hz) 的分离式 Mamba-2 架构 | **高** — 第一性原理驱动, 机器人控制领域新颖 | 未知 — OpenPI 无时序建模也能工作 |
| **分层槽压缩** | Perceiver-style 96→24 的学习型路由压缩, 产生结构化中间表示 | **中高** — 可解释的物体/相位/可供性分离 | 可能在复杂多物体场景有优势 |
| **CrossAttentionFusion + StaleTimeEncoding** | 内容相关的逐维度融合 + 信息陈旧度编码 | **中** — 比标量门控更灵活 | 增加了 ~5M 参数和延迟 |
| **混合动作一致性** | 离散/连续双头在共享嵌入空间的余弦约束 | **中** — 新颖的多模态动作表示约束 | 弱约束, 效果存疑 |
| **RTC + FASTER** | 重叠块修正 + 自适应去噪步数 (现已实现) | **中** — 动作 chunking 的改进方向 | RTC 有 train-infer 不一致; FASTER 推理未实现 |

### 8.2 工程设计优势

| 优势 | 描述 | OpenPI 对应 |
|------|------|------------|
| **多尺度视觉特征** | FPN 风格的学习门控融合 (层 10/18/28) | 单尺度 SigLIP |
| **Midpoint ODE 求解器** | 2 阶精度, 与 Euler 相同前向次数 | 仅 Euler (1 阶) |
| **Per-module LR scaling** | backbone ×0.1, expert ×0.5, core ×1.0 | 统一 LR |
| **Per-module gradient norm 监控** | 每 5× log_interval 输出各模块梯度范数 | 无 |
| **三阶段渐进训练** | 感知 → 专家 → 端到端, 知识隔离 (`detach()`) | 单阶段微调 |
| **严格批次验证** | `_validate_batch()` 检查所有键、形状、维度一致性 | 较少显式验证 |
| **Logit-normal 时间步采样** | 偏向中间噪声水平 (最困难区域) | 均匀采样 |
| **AdaRMSNorm gate bias +2** | 防止 18 层残差激活塌陷 (sigmoid(2)≈0.88 初始透过率) | 标准初始化 |
| **Per-block residual scale** | `1/sqrt(N)` 初始化, 防止深层 Mamba 栈激活爆炸 | 无显式控制 |
| **压缩训练配置** | `stage_*_compressed.yaml` 提供 2.67× 步数压缩方案 | 固定 30K 步 |

### 8.3 理论深度优势

HybridVLA 的核心设计哲学是 **"机器人控制需要时序记忆和结构化感知"**, 这与 OpenPI 的 **"简单端到端足以"** 形成对比:

1. **时序建模的论点**: 机器人操作任务 (如倒水、穿针) 需要跨多个时间步的状态积累。OpenPI 每步独立推理, 丢弃历史信息。HybridVLA 的 SSM 状态理论上能捕获动态趋势。但 OpenPI 通过 action chunking (一次生成 24 步) 隐式包含了短期轨迹规划。

2. **结构化感知的论点**: 96 个分类型潜在标记 (global/object/phase/uncertainty/affordance) 提供了可解释的中间表示, 有助于 debug 和分析模型决策。OpenPI 的端到端方式是黑盒的。

3. **多频率的论点**: 50 Hz 电机控制、25 Hz 物体动力学、12.5 Hz 任务规划的频率分离匹配了物理系统的内在时间尺度。这在长时序任务 (数分钟级) 中可能提供优势。

**但所有这些论点都缺乏实验验证**。这是 HybridVLA 最大的弱点: 它有丰富的理论动机, 但零实验证据。OpenPI 的 "简单方案" 已经被证明能在真实机器人上工作。

---

## 9. 综合评分与建议

### 9.1 维度评分

| 维度 | v0.10.5 评分 | v0.10.7 评分 | OpenPI | 说明 |
|------|-------------|-------------|--------|------|
| 架构创新性 | 8.0 | 8.0 | 6.0 | 不变 — 架构未修改 |
| 工程成熟度 | 4.0 | **6.5** | 9.0 | 测试 + 推理 + LIBERO 大幅提升 |
| 训练管线完整性 | 5.5 | **8.0** | 9.0 | RTC/FASTER 实现补齐 |
| 推理/部署能力 | 1.0 | **5.5** | 9.0 | 从零到可用, 仍缺服务化 |
| 验证深度 | 0.5 | **1.5** | 10.0 | 有 LIBERO 集成但未训练 |
| 代码质量 | 6.0 | 6.5 | 8.5 | 仍缺 linter/CI |
| 文档/可维护性 | 5.0 | 6.0 | 8.0 | 配置注释好, 但无 API 文档 |
| **综合** | **4.3** | **6.0** | **8.5** | **+1.7 提升** |

### 9.2 差距概括

```
              HybridVLA v0.10.5          HybridVLA v0.10.7          OpenPI
                    │                          │                      │
  概念验证原型 ─────┤                          │                      │
                    │                          │                      │
  可训练系统 ───────┤──────────────────────────┤                      │
                    │                          │                      │
  可评估系统 ───────┤──────────────────────────┤                      │
                    │                          │                      │
  可部署系统 ───────┤──────────────────────────┤──────────────────────┤
                    │                          │                      │
  生产就绪 ─────────┤──────────────────────────┤──────────────────────┤
```

### 9.3 下一步优先级建议

**立即 (本周)**:
1. 在 LIBERO 上跑 Level 1 验证 (500 steps, 单 GPU) — 确认损失能下降
2. 实现 FASTER 推理 (近端精采样 + 远端粗采样)
3. 添加 ruff + pre-commit (10 分钟工作, 永久收益)

**短期 (两周内)**:
4. 修复 RTC train-infer 不一致 (用时间偏移的 cond_prefix)
5. 添加 Grounder + TriRate Mamba 的单元测试
6. 删除 `train_stage_a.py` 和世界模型死代码
7. 验证 EMA 与 FSDP 的兼容性

**中期 (一个月)**:
8. LIBERO Level 2 验证 (15K steps) — 三阶段衔接
9. 抽象出通用 `Policy` 基类 (为多机器人做准备)
10. 推理延迟优化 (Euler 替代 Midpoint 选项, torch.compile)

---

## 附录 A: 文件级变更清单 (v0.10.5 → v0.10.7)

### 新增文件

| 文件 | 行数 | 类别 |
|------|------|------|
| `vla_hybrid_v2/data/libero_hdf5_adapter.py` | 526 | 数据层 |
| `vla_hybrid_v2/data/transforms.py` | 64 | 数据增强 |
| `vla_hybrid_v2/infer/libero_policy.py` | ~250 | 推理层 |
| `tests/conftest.py` | 164 | 测试 |
| `tests/test_forward_train.py` | 133 | 测试 |
| `tests/test_expert.py` | 78 | 测试 |
| `tests/test_normalizer.py` | 77 | 测试 |
| `tests/test_losses.py` | 102 | 测试 |
| `tests/test_control_step.py` | ~50 | 测试 |
| `tests/test_checkpoint_assets.py` | ~50 | 测试 |
| `tests/test_eval_config_resolution.py` | ~40 | 测试 |
| `libero_hybrid/scripts/eval_libero_rollout.py` | 411 | 评估 |
| `libero_hybrid/scripts/train_libero.py` | 173 | 训练包装 |
| `libero_hybrid/scripts/compute_libero_stats.py` | 133 | 工具 |
| `libero_hybrid/scripts/validate_libero_hdf5.py` | 145 | 工具 |
| `libero_hybrid/utils.py` | 72 | 工具 |

### 修改文件

| 文件 | 关键变化 |
|------|---------|
| `hybrid_vla_v2.py` | +RTC 训练 +FASTER 训练 +RTC 推理 +FASTER 推理 (NotImplemented) |
| `config.py` | +AugmentationConfig +RTCTrainConfig 补充字段 +FASTERTrainConfig 补充字段 |
| `qwen2vl_backbone.py` | +CameraPositionEmbedding |
| `data/__init__.py` | +libero_hdf5 format 分支 |
| `infer/__init__.py` | 从 1 行 → 导出 LIBERO 策略 |
| `types.py` | +prev_chunk_tail for RTC |

## 附录 B: 核心模块参数量估算

| 模块 | 计算方式 | 估算值 |
|------|---------|--------|
| Qwen2-VL-7B 冻结 | 官方参数量 | 7.6B |
| LoRA (r=64, α=128, 28层 × 7 modules) | 28 × 7 × 2 × 64 × 3584 | ~90M |
| MultiScaleAdapter | 3 × (3584→2048 + LN) + gate | ~25M |
| CameraPositionEmbedding | 8 × 2048 | ~16K |
| Grounder (8L, 96 lat, 2048d) | 8 × (cross+self+FFN) × 2048² × 4 | ~280M |
| SlotCompression | cross + self + route_queries | ~35M |
| Fast Mamba (20L, 2048d, expand=2) | 20 × (in_proj + conv + SSM + out_proj) | ~400M |
| Medium Mamba (6L, 2048d) | 6 × same | ~120M |
| Slow Mamba (10L, 2048d, d_state=256) | 10 × (略大于 Fast 单层) | ~220M |
| CrossAttentionFusion (2L) | 2 × (MHA + FFN) + stale_proj | ~50M |
| StaleTimeEncoding | 2 × Linear(2048, 2048) | ~8M |
| ActionHistoryEncoder (4L, 2048d) | 4 × Mamba block + action_proj | ~120M |
| FlowActionExpert (18L, 1536d) | 12×Mamba + 6×Attn(cross+self+FFN) | ~500M |
| Discrete Heads | FAST + Phase + Affordance | ~15M |
| Projections | proprio + prev_action + embodiment + cond_builder + core_to_expert | ~30M |
| **可训练总计** | | **~2.3B** |
| **总计 (含冻结)** | | **~9.9B** |
