# HybridVLA v2 — v0.10.6 版本总结

> **标准**: "8×H100 真实训练会不会崩" + "训练产出是否有意义" + "工程闭环是否完整"

---

## 1. 版本演进回顾

v0.10.0 → v0.10.6 经历 6 轮 audit-fix-rescore，累计约 20 项修复。核心进展：

| 版本 | 评分 | 关键变化 |
|------|------|---------|
| v0.9.3 | 6.8 | 数据层出现（schema/normalizer/HDF5 adapter/collate） |
| v0.10.0 | 6.5 | 7 项旧 bug 修复（D1-D8），compute_stats |
| v0.10.1 | 7.3 | **P0-3 chunk 退化修复**（T+H-1），stats 解耦，_to_device 递归 |
| v0.10.2 | 7.3 | affordance 可配置，step_weights 校验，代码清理 |
| v0.10.3 | — | **processor 连接**，**HDF5 图像读取**，**多步监督**，**统一训练脚本**，**eval loop** |
| v0.10.4 | 7.5 | **pixel_values resize**（不修就崩），**显式 stage 门控**，**sanity check**，MultiCamera=False |
| v0.10.5 | 8.0 | **val split 过滤**，**per-module LR**，**per-module gnorm**，smoke test B/C 断言 |
| **v0.10.6** | — | 多相机支持，compressed configs，本次审计 |

---

## 2. 当前代码结构

```
vla_hybrid_v2/                          总计 ~4,200 行 Python
├── models/
│   ├── hybrid_vla_v2.py     (792 行)   主模型：forward_train + control_step
│   ├── mamba_core.py        (785 行)   三速率 Mamba 时序核心
│   ├── flow_action_expert.py(344 行)   18 层 M-M-A×6 流匹配 expert
│   ├── attention_grounder.py(260 行)   层次注意力接地器 96→24 槽
│   ├── qwen2vl_backbone.py  (213 行)   Qwen2-VL-7B + LoRA + 多尺度
│   ├── discrete_heads.py    (76 行)    FAST 512-bin / Phase / Affordance
│   └── types.py             (127 行)   7 核心数据类
├── data/
│   ├── hdf5_adapter.py      (466 行)   真实数据加载：图像+动作+语言+refresh
│   ├── collate.py           (113 行)   安全 collate（_safe_stack_vision）
│   ├── normalizer.py        (185 行)   action/proprio 归一化
│   ├── schema.py            (79 行)    WindowSample 协议
│   ├── dummy.py             (67 行)    smoke test 用随机数据
│   ├── base_adapter.py      (49 行)    抽象基类
│   └── __init__.py          (83 行)    build_dataset 工厂
├── losses/                  (156 行)   flow_matching + discrete + consistency
├── utils/                   (402 行)   checkpointing + distributed + ema
├── infer/                   (1 行)     空 stub
├── config.py                (397 行)   全量配置
└── ops/                     (55 行)    selective_scan

scripts/
├── train_unified.py         (549 行)   统一训练入口 A/B/C
├── train_smoke_test.py      (313 行)   三阶段 smoke test + 断言
├── compute_stats.py         (186 行)   归一化统计量

configs/
├── model/v2_qwen2vl_7b_trirate_expert18.yaml
├── train/stage_a.yaml, stage_b.yaml, stage_c.yaml
├── train/stage_{a,b,c}_compressed.yaml
└── data/libero_multicam.yaml
```

---

## 3. v0.10.0 以来的全部改进清单

### 数据层（从"空壳"到"完整 VLA 数据管线"）

| # | 改进 | 版本 | 影响 |
|---|------|------|------|
| 1 | Action chunk 读取 T+H-1 步 | v0.10.1 | 消除监督目标 96% 填充退化 |
| 2 | normalizer_stats_dir 路径解耦 | v0.10.1 | 跨 stage 共享 stats |
| 3 | HDF5 图像读取 + Qwen2-VL processor 连接 | v0.10.3 | 系统从 "LA" 升级为 "VLA" |
| 4 | Refresh 帧构造 (多观测语义刷新) | v0.10.3 | 12.5Hz 视觉更新可训练 |
| 5 | pixel_values 统一 448×448 resize | v0.10.4 | 消除 collate torch.stack 崩溃 |
| 6 | collate _safe_stack_vision 安全网 | v0.10.4 | 变长 patch 二级防御 |
| 7 | Val split 过滤 (val_data_dir + val_ratio) | v0.10.5 | eval 不再在训练集上算 |
| 8 | 多相机数据读取 (_process_text_multi_image) | v0.10.6 | 3 相机联合 tokenization |
| 9 | collate None/Tensor 混合处理 | v0.10.6 | refresh 帧缺失图像不崩 |

### 训练管线（从"Stage A 唯一脚本"到"三阶段统一训练"）

| # | 改进 | 版本 | 影响 |
|---|------|------|------|
| 10 | 统一训练脚本 train_unified.py | v0.10.3 | A/B/C 单一入口 |
| 11 | Eval loop 集成 | v0.10.3 | 验证集 loss 自动计算 |
| 12 | 显式 configure_trainable_modules | v0.10.4 | 冻结/解冻不再隐式 |
| 13 | sanity_check_trainable_params | v0.10.4 | 断言防止 expert 静默冻结 |
| 14 | Per-module LR 分组 | v0.10.5 | backbone 0.1×, expert 0.5×, core 1.0× |
| 15 | Per-module gradient norm 日志 | v0.10.5 | 可验证 Stage B 梯度隔离 |
| 16 | Smoke test Stage B/C 断言 | v0.10.5 | loss_fm 存在 + expert 参数更新 |

### 模型层

| # | 改进 | 版本 | 影响 |
|---|------|------|------|
| 17 | 多步监督 FAST/Phase/Affordance 全 T | v0.10.3 | 梯度密度从 1/24 到 24/24 |
| 18 | num_affordance_types 可配置 | v0.10.2 | 不再硬编码 |
| 19 | step_weights 形状校验 | v0.10.2 | 防止 silent broadcast |
| 20 | MultiCameraConfig.enable=False | v0.10.4 | 消除认知误导 |

---

## 4. 当前潜在问题

以"8×H100 真实训练"为标准，按严重程度分级。

### 无"第一天崩溃"风险 ✅

以下路径已验证安全：
- 图像大小不一致 → 448×448 resize + _safe_stack_vision 兜底
- Stage 冻结/解冻 → configure_trainable_modules 显式 + sanity_check 断言
- collate 变长 → 防御性 padding + warning
- 设备转移 → 递归 _to_device 处理嵌套 list

### P1 — 影响训练质量但不崩溃

| ID | 问题 | 文件 | 当前状态 | 建议 |
|----|------|------|---------|------|
| **R1** | Expert loss 仅在 t=-1 | `hybrid_vla_v2.py:533` | 设计选择（expert forward 昂贵）| 可接受。FAST/Phase/Aff 已全步监督 |
| **R2** | tests/ 目录空 | — | 无 pytest 单元测试 | 训练期间并行补充 |
| **R3** | infer/ 空 stub | `infer/__init__.py` | 无 PolicyWrapper / rollout | 阻塞 benchmark 评估，不阻塞训练 |
| **R4** | Tri-rate 无 ablation 开关 | `config.py`, `mamba_core.py` | 无法关闭 medium/slow 做对照 | 论文提交前需补 |
| **R5** | 长距指标未记录 | `train_unified.py` | 无 horizon bucket / chunk reuse 统计 | 论文分析时需补 |
| **R6** | Phase/Aff 标签 HDF5 中缺失 | `hdf5_adapter.py` | 返回 dict 无此字段 → loss 自动跳过 | 正确行为，无数据就不算 |

### P2 — 代码质量 / 长期维护

| ID | 问题 | 说明 |
|----|------|------|
| R7 | WindowSample 仍为文档协议 | adapter 返回 dict，不强制类型 |
| R8 | split 排序依赖文件名 | episode-ratio split 按 sorted(glob) 切分，换数据集可能不稳定 |
| R9 | backbone refresh 重复前向 | R=4 时骨干跑 4 次（7B 模型），训练瓶颈 |
| R10 | `num_workers=2` 可能 IO 受限 | 大数据集建议 4-8 |

---

## 5. "是否可以开始训练"判定

### Stage A（backbone LoRA + grounder + temporal core）

| 条件 | 状态 | 证据 |
|------|:----:|------|
| Processor 连接 | ✅ | `train_unified.py:415-419` AutoProcessor 创建并传入 |
| 图像读取+resize | ✅ | `hdf5_adapter.py:188-196, 250-256` 448×448 强制 |
| Action chunk 完整 | ✅ | `hdf5_adapter.py:309,356` T+H-1 扩展 |
| Refresh 帧 | ✅ | `hdf5_adapter.py:350-455` 多帧构造 |
| 多步监督 | ✅ | `hybrid_vla_v2.py:490-530` FAST/Phase/Aff 全 T |
| Expert 冻结 | ✅ | `train_unified.py:96-97,134` 显式冻结 + 断言 |
| Val split | ✅ | `hdf5_adapter.py:95-139` val_data_dir 或比例切分 |
| Per-module LR | ✅ | `train_unified.py:344-377` backbone 0.1×, core 1.0× |
| Gradient 监控 | ✅ | `train_unified.py:238-261` 9 模块 gnorm |
| FSDP + bf16 | ✅ | `train_unified.py:340-342` |

**Stage A 判定: ✅ 可以开始训练。**

### Stage B（+ expert 训练）

| 条件 | 状态 | 证据 |
|------|:----:|------|
| Expert 解冻 | ✅ | `train_unified.py:135-146` + sanity_check 断言 |
| cond_prefix.detach() | ✅ | `hybrid_vla_v2.py:537-539` stage_b.yaml 配置 |
| loss_fm 产生 | ✅ | smoke test 验证 (optimize_v0_10_5.md) |
| Expert 参数更新 | ✅ | smoke test 快照对比验证 |
| Per-module LR | ✅ | expert 0.5× |
| EMA | ✅ | `train_unified.py:384-409` initial_decay→final_decay ramp |
| 跨 stage checkpoint | ✅ | `train_unified.py:411-424` strict=False |

**Stage B 判定: ✅ 可以开始训练（Stage A 完成后）。**

### Stage C（全微调 + RTC/FASTER）

| 条件 | 状态 | 说明 |
|------|:----:|------|
| Backbone text 16-27 解冻 | ✅ | `train_unified.py:149-155` |
| RTC/FASTER config | ✅ | `config.py:192-204` + `stage_c.yaml` |
| RTC/FASTER 训练逻辑 | ⚠️ | config 就绪，但 `forward_train()` 中 RTC/FASTER **未实际使用** — `rtc.enable` 和 `faster.enable` 字段存在但无代码读取 |

**Stage C 判定: ⚠️ 可以训练基础版本。RTC/FASTER 增强逻辑尚未接入 forward_train，需在 Stage B 训练期间补充。**

---

## 6. 评分

| # | 维度 | v0.10.0 | v0.10.6 | Δ | 理由 |
|---|------|:-------:|:-------:|:-:|------|
| 1 | 设计一致性 | 8.5 | **8.5** | 0 | 多相机实现，MultiCamera=False 修正 |
| 2 | 正确性 | 8.5 | **9.5** | +1.0 | chunk 修复、pixel resize、stage 断言、多步监督 |
| 3 | 完备性 | 6.0 | **8.0** | +2.0 | 视觉通路、processor、统一脚本、eval、val split |
| 4 | 训练稳定性 | 9.0 | **9.0** | 0 | per-module LR/gnorm 加强；基础未变 |
| 5 | 可扩展性 | 7.0 | **7.0** | 0 | 无 ablation 开关仍是短板 |
| 6 | 性能设计 | 6.0 | **6.5** | +0.5 | 多相机；refresh 重复开销仍在 |
| 7 | 生产就绪度 | 5.0 | **7.0** | +2.0 | 三阶段可运行，val split 有效，但无 infer runtime |
| 8 | 代码质量 | 7.5 | **8.5** | +1.0 | 显式 stage 门控、安全 collate、per-module 分组 |
| 9 | 文档 | 4.5 | **5.0** | +0.5 | 多份 audit/optimize 文档；无 README |
| 10 | 测试 | 1.5 | **3.0** | +1.5 | smoke test 三阶段断言；但 tests/ 仍空 |
| | **加权均分** | **6.5** | **7.8** | **+1.3** | |

---

## 7. 训练启动检查清单

```
准备阶段:
  □ 准备 HDF5 数据 (actions + proprio + images/{camera_key})
  □ 运行 compute_stats: python -m scripts.compute_stats --config configs/train/stage_a.yaml
  □ 确认 normalizer_stats/ 下有 action_stats.json + proprio_stats.json
  □ (可选) 准备独立 val 数据目录，配置 data.val_data_dir

Stage A:
  □ torchrun --nproc_per_node=8 -m scripts.train_unified --config configs/train/stage_a.yaml
  □ 确认启动日志: "Stage A: configured trainable modules"
  □ 确认 sanity check: "action_expert trainable=0"
  □ 确认 per-module gnorm 每 250 步输出
  □ 确认 eval 每 2000 步运行

Stage B (Stage A 完成后):
  □ torchrun --nproc_per_node=8 -m scripts.train_unified --config configs/train/stage_b.yaml
  □ 确认 "action_expert trainable=..." > 0
  □ 确认 loss_fm 出现在日志中
  □ 确认 expert gnorm > 0 且 backbone gnorm 较小 (梯度隔离)

Stage C (Stage B 完成后):
  □ torchrun --nproc_per_node=8 -m scripts.train_unified --config configs/train/stage_c.yaml
  □ 确认 backbone text layers 16-27 解冻
```

---

## 8. 中文摘要

### 版本定位

v0.10.6 是从 v0.9.3（"数据层出现"）到训练就绪的关键里程碑。经过 6 轮 audit-fix-rescore（含 Claude × GPT 交叉审计），代码从"结构合理但真实训练会崩"提升为"三阶段统一训练可启动"。

### 核心进步（20 项改进）

**数据层**（9 项）：从空壳到完整 VLA 管线——图像读取 + resize + refresh 帧 + 多相机 + val split + 安全 collate。

**训练管线**（7 项）：从 Stage A 唯一脚本到三阶段统一训练——显式 stage 门控 + sanity check + per-module LR/gnorm + eval loop + smoke test 断言。

**模型层**（4 项）：多步监督 + affordance 可配置 + step_weights 校验 + MultiCamera 修正。

### 潜在问题

- **无"第一天崩溃"风险** ✅
- **P1（不阻塞训练）**: Expert loss 仅 t=-1（设计选择）、tests/ 空、infer/ 空、无 ablation 开关、无长距指标
- **Stage C 注意**: RTC/FASTER 配置存在但训练逻辑未接入 forward_train

### 是否可以开始训练

| 阶段 | 判定 | 条件 |
|------|:----:|------|
| **Stage A** | **✅ 可以** | 准备 HDF5 数据 + compute_stats |
| **Stage B** | **✅ 可以** | Stage A 完成后 |
| **Stage C** | **⚠️ 基础版可以** | RTC/FASTER 增强需补充 |

### 评分

**7.8/10**（v0.10.0 → v0.10.6: +1.3 分）。最大提升来自**完备性**（+2.0，视觉通路 + 统一脚本）和**生产就绪度**（+2.0，三阶段可运行 + val split）。

剩余 2.2 分差距主要来自：tests/ 空白（-1.5）、无 infer runtime（-0.5）、无 ablation 开关（-0.2）。这些可在 Stage A 训练期间并行补充。
