# HybridVLA v2 — 架构审查地图 v1.0.1

> **审查者**: Claude Opus 4.6
> **日期**: 2026-03-29
> **代码版本**: v1.0.0 (pyproject.toml) / dev 分支
> **审查范围**: 全仓库 ~12,000 行 Python（不含 docs/experimental）

---

## A. 仓库模块划分

```
hybridVLA_2/
├── vla_hybrid_v2/                 # 核心库（~5,500 行）
│   ├── config.py                  #   配置体系（dataclass + YAML loader）
│   ├── types.py                   #   核心数据类型定义
│   ├── models/                    #   模型子模块
│   │   ├── hybrid_vla_v2.py       #     顶层模型：forward_train + control_step
│   │   ├── qwen2vl_backbone.py    #     7B VLM 骨干 + MultiScale + LoRA
│   │   ├── attention_grounder.py  #     层次化 Perceiver 注意力
│   │   ├── mamba_core.py          #     三速率 Mamba 时序核心
│   │   ├── flow_action_expert.py  #     Flow Matching 动作专家
│   │   └── discrete_heads.py      #     离散头（FAST/Phase/Affordance）
│   ├── losses/                    #   损失函数
│   │   ├── flow_matching.py       #     Rectified Flow MSE
│   │   ├── discrete_loss.py       #     Cross-Entropy + label smoothing
│   │   └── consistency_loss.py    #     对比 + 慢快一致 + 动作一致
│   ├── data/                      #   数据管线
│   │   ├── schema.py              #     WindowSample 数据契约
│   │   ├── base_adapter.py        #     抽象适配器
│   │   ├── hdf5_adapter.py        #     通用 HDF5 适配器
│   │   ├── libero_hdf5_adapter.py #     LIBERO 格式适配器
│   │   ├── normalizer.py          #     动作/本体感知归一化
│   │   ├── collate.py             #     视觉感知批次拼装
│   │   ├── transforms.py          #     图像增强
│   │   └── dummy.py               #     合成数据（冒烟测试）
│   ├── infer/                     #   推理封装
│   │   └── libero_policy.py       #     LIBERO 闭环推理策略
│   ├── utils/                     #   工具
│   │   ├── checkpointing.py       #     存/读/恢复检查点 + 资产打包
│   │   ├── distributed.py         #     FSDP + 梯度裁剪 + 分布式
│   │   └── ema.py                 #     EMA + FSDP summon_full_params
│   ├── ops/                       #   底层算子
│   │   └── selective_scan.py      #     JIT SSM scan + CUDA dispatch
│   └── experimental/              #   实验模块（未连接主管线）
│       └── world_model/           #     DreamerV3 风格世界模型（~1,200 行）
│
├── scripts/                       # 入口脚本
│   ├── train_unified.py           #   统一三阶段训练（627 行）
│   ├── train_stage_a.py           #   废弃包装器
│   ├── train_smoke_test.py        #   冒烟测试（CPU/GPU 端到端验证）
│   └── compute_stats.py           #   归一化统计计算
│
├── libero_hybrid/                 # LIBERO 基准集成
│   ├── scripts/
│   │   ├── train_libero.py        #   LIBERO 训练包装器（变体切换）
│   │   ├── eval_libero_rollout.py #   闭环 rollout 评测
│   │   ├── compute_libero_stats.py#   LIBERO 归一化统计
│   │   └── validate_libero_hdf5.py#   HDF5 结构验证
│   └── utils.py                   #   Suite 路径解析/demo 排序
│
├── configs/                       # YAML 配置
│   ├── model/                     #   模型架构（v2_qwen2vl_7b_trirate_expert18）
│   ├── train/                     #   训练配置（stage_a/b/c + compressed 变体）
│   └── data/                      #   数据配置（libero_multicam/singlecam）
│
├── tests/                         # 单元测试（10 个文件，~1,600 行）
│   ├── conftest.py                #   Mini 配置 + MockBackbone
│   ├── test_forward_train.py      #   三阶段前向/反向测试
│   ├── test_losses.py             #   损失函数单元测试
│   ├── test_normalizer.py         #   归一化往返测试
│   ├── test_expert.py             #   ODE 求解器 + AdaRMSNorm 初始化
│   ├── test_control_step.py       #   在线推理（RTC/FASTER fail-fast）
│   ├── test_checkpoint_assets.py  #   检查点资产打包
│   ├── test_eval_config_resolution.py # 配置发现与校验
│   ├── test_ema_fsdp_gaps.py      #   EMA/FSDP/梯度累积全面测试
│   └── test_infer_policy.py       #   推理策略归一化对齐
│
└── docs/                          # 文档（30+ 文件）
```

---

## B. 每个模块/目录的职责

### B.1 核心模型层 (`vla_hybrid_v2/models/`)

| 文件 | 行数 | 职责 | 关键类 |
|------|------|------|--------|
| `hybrid_vla_v2.py` | 862 | 顶层编排：forward_train (多步时序训练) + control_step (50Hz推理) + 损失计算 + 条件前缀构建 | `HybridVLAv2` |
| `qwen2vl_backbone.py` | ~300 | Qwen2-VL-7B加载 + LoRA注入 + 多尺度特征提取[L10,L18,L28] + 相机位置嵌入 | `Qwen2VLBackboneWrapper`, `MultiScaleAdapter`, `CameraPositionEmbedding` |
| `attention_grounder.py` | 260 | 96 latent Perceiver → 层4压缩48→24 slots → 结构化输出tokens | `HierarchicalAttentionGrounder`, `SlotCompression` |
| `mamba_core.py` | 807 | 三速率SSM核心(Fast 20L/Medium 6L/Slow 10L) + 交叉注意力融合 + 过时时间编码 + 动作历史编码器 | `TriRateMambaCore`, `MambaBlock`, `CrossAttentionFusion`, `ActionHistoryEncoder` |
| `flow_action_expert.py` | 360 | 18层 M-M-A×6 + AdaRMSNorm + Euler/Midpoint采样 | `FlowActionExpert`, `AdaRMSNorm`, `ExpertMambaBlock`, `ExpertAttentionBlock` |
| `discrete_heads.py` | 85 | FAST 512-bin离散化/反离散化 + Phase 16类 + Affordance 8类 | `DiscreteActionHead`, `PhaseHead`, `AffordanceHead` |

### B.2 数据管线 (`vla_hybrid_v2/data/`)

| 文件 | 行数 | 职责 | 关键点 |
|------|------|------|--------|
| `schema.py` | ~80 | `WindowSample` dataclass — 数据适配器与模型之间的契约 | 定义 actions/proprio/prev_actions/pixel_values 等字段形状 |
| `base_adapter.py` | ~60 | 抽象基类 `BaseDatasetAdapter` | 约束 `__len__`, `__getitem__` 返回 `WindowSample` |
| `hdf5_adapter.py` | ~250 | 通用 HDF5 适配器：滑窗采样 + Qwen2VL processor 分词 + 图像缩放 | 处理 `data/{action_key}`, `obs/{image_key}` 结构 |
| `libero_hdf5_adapter.py` | ~200 | LIBERO robomimic 格式：多 demo 分组 + 多 proprio 键拼接 + 语言指令提取 | `demo_0/obs/agentview_rgb` → 多键 proprio concat |
| `normalizer.py` | ~200 | `Normalizer` 基类 + `ActionNormalizer` + `ProprioNormalizer`：min_max / mean_std 策略 + JSON 持久化 | 支持 fit/transform/inverse_transform + save/load |
| `collate.py` | ~100 | `vla_collate_fn`：处理变长 pixel_values 的批次 padding | 对齐 vision token 序列长度 |
| `transforms.py` | ~80 | `RobotImageAugmentation`：RandomResizedCrop + Rotation + ColorJitter | 训练时增强，推理时禁用 |
| `dummy.py` | ~100 | `DummyVLADataset`：随机张量数据集，用于冒烟测试 | 生成符合 WindowSample 契约的假数据 |

### B.3 损失函数 (`vla_hybrid_v2/losses/`)

| 文件 | 行数 | 职责 |
|------|------|------|
| `flow_matching.py` | ~100 | Rectified Flow MSE + logit-normal 时间步采样 + 线性插值 |
| `discrete_loss.py` | ~60 | 带 label smoothing 的 Cross-Entropy + PhaseLoss (无 smoothing) |
| `consistency_loss.py` | ~150 | 三项一致性：InfoNCE 时序对比 + 慢快指数移动平均一致 + 离散-连续动作 cosine 一致 |

### B.4 工具 (`vla_hybrid_v2/utils/`)

| 文件 | 行数 | 职责 | FSDP 感知 |
|------|------|------|-----------|
| `checkpointing.py` | ~220 | 原子写检查点 + 资产打包(config + normalizer stats) + 跨阶段恢复 + 形状不匹配过滤 | 是：`FullStateDictConfig`, `full_optim_state_dict` |
| `distributed.py` | ~150 | FSDP 包装(auto_wrap MambaBlock/GrounderBlock/ExpertBlock) + 激活检查点 + 梯度裁剪 + 种子 | 是：`use_orig_params=True`, `MixedPrecision` bf16 |
| `ema.py` | ~160 | EMA 衰减线性爬坡 + FSDP `summon_full_params` 读/写 + 形状不匹配 shadow 过滤 | 是：FSDP 前初始化，update/apply/restore 用 summon |

### B.5 推理 (`vla_hybrid_v2/infer/`)

| 文件 | 行数 | 职责 |
|------|------|------|
| `libero_policy.py` | ~350 | 完整 LIBERO 闭环策略：检查点发现 → 配置对齐 → 归一化加载 → 观测→模型输入 → 动作反归一化 |

### B.6 实验模块 (`vla_hybrid_v2/experimental/world_model/`)

| 文件 | 行数 | 状态 |
|------|------|------|
| `imagination_engine.py` | ~200 | 32步 latent rollout 编排器 |
| `imagination_mamba.py` | ~100 | 8层 Mamba 动力学 (用 .step() 保持状态) |
| `stochastic_state.py` | ~100 | 48×48 categorical latent + straight-through Gumbel |
| `object_physics.py` | ~150 | 6层 GNN 物体交互 |
| `noise_augmentation.py` | ~80 | GameNGen 风格噪声鲁棒性 |
| `world_model_heads.py` | ~120 | Reward/Value/Done 预测 (symlog two-hot) |
| `world_model_loss.py` | ~180 | KL + Physics + Visual 联合损失 |
| `visual_decoder.py` | ~100 | CNN 112×112 解码器 |
| `subgoal_planner.py` | ~40 | 相位级 latent 子目标预测 |

**状态**: 全部 `enable=False`，未连接 `forward_train()`。代码完整但零训练验证。

---

## C. 主要调用链

### C.1 训练调用链

```
scripts/train_unified.py::train()
 ├─ setup_distributed() → init_process_group + seed
 ├─ HybridVLAv2(cfg) → 构建所有子模块
 ├─ configure_trainable_modules(model, stage) → 冻结/解冻
 ├─ sanity_check_trainable_params(model, stage) → 断言
 ├─ model.to(device)
 ├─ load_checkpoint() [跨阶段恢复: B ← A, C ← B]
 ├─ EMAModel(model) [FSDP 前初始化, shadow 克隆 full params]
 ├─ wrap_fsdp(model) [auto_wrap MambaBlock/GrounderBlock/ExpertBlock]
 ├─ AdamW(param_groups) [per-module LR: backbone×0.1, expert×0.5, core×1.0]
 ├─ cosine_schedule_with_warmup()
 ├─ auto_resume() [同阶段检查点恢复]
 ├─ build_dataset(cfg) → HDF5/LIBERO/Dummy 适配器
 │
 └─ 训练循环:
     for batch in loader:
      ├─ model.forward_train(batch) ──────────────────────────┐
      │   ├─ _validate_batch() → 形状/键校验                   │
      │   ├─ backbone.forward_semantic() × R次 [语义刷新]       │
      │   │   ├─ Qwen2-VL forward → multi_scale_layers 提取    │
      │   │   ├─ MultiScaleAdapter → 门控融合 → [B, N, 2048]   │
      │   │   └─ CameraPositionEmbedding (如果 multi_cam)       │
      │   ├─ grounder() × R次 → GrounderOutput                 │
      │   │   ├─ 层0-3: 96 latents cross-attend backbone       │
      │   │   ├─ 层4: SlotCompression 48→24                    │
      │   │   └─ 层4-7: 72 latents self-attend                 │
      │   ├─ for t in range(T): [时序展开]                       │
      │   │   ├─ proprio_proj + prev_action_proj + stale_encoder│
      │   │   ├─ action_history_encoder.encode()                │
      │   │   ├─ temporal_core() → TemporalOutput               │
      │   │   │   ├─ FastMamba (每步)                            │
      │   │   │   ├─ MediumMamba (每 medium_stride 步)           │
      │   │   │   ├─ SlowMamba (语义刷新时)                      │
      │   │   │   └─ CrossAttentionFusion                        │
      │   │   └─ action_history_buf.push()                       │
      │   ├─ fast_head(fused_states) → FAST loss [全 T 步]       │
      │   ├─ phase_head/affordance_head → 对应 loss [全 T 步]    │
      │   ├─ [Stage B/C]: _build_cond_prefix() → [B, 32, D_exp] │
      │   │   ├─ if stop_gradient: cond_prefix.detach()          │
      │   │   ├─ FlowMatchingLoss.sample_timestep()              │
      │   │   ├─ action_expert(noisy, t, cond) → velocity       │
      │   │   ├─ loss_fm = MSE(velocity, target_velocity)        │
      │   │   ├─ [Stage C]: RTC inpainting loss                  │
      │   │   └─ [Stage C]: FASTER near-horizon aux loss          │
      │   └─ consistency_loss() → 三项一致性                      │
      │                                                          │
      ├─ loss.backward() / grad_accum                            │
      ├─ clip_grad_norm_fsdp()                                   │
      ├─ optimizer.step() + scheduler.step()                     │
      ├─ ema.update(model, step)                                 │
      ├─ [定期] evaluate() with EMA weights → all_reduce 同步    │
      └─ [定期] save_checkpoint() + asset_paths                  │
```

### C.2 数据处理调用链

```
build_dataset(cfg, split, processor)
 ├─ [format="hdf5"] HDF5DatasetAdapter(data_dir, processor, ...)
 │   ├─ glob("*.hdf5") → episode_paths
 │   ├─ _index_episodes() → (path, start_idx, length) 索引
 │   ├─ __getitem__(idx):
 │   │   ├─ h5py.File → 读取 action/obs/proprio 窗口
 │   │   ├─ PIL Image resize(448×448) → processor tokenize
 │   │   ├─ RobotImageAugmentation (训练时)
 │   │   └─ → WindowSample dict
 │   └─ ActionNormalizer.fit() + ProprioNormalizer.fit()
 │
 ├─ [format="libero_hdf5"] LiberoHDF5Adapter
 │   ├─ 读取 libero_{suite}/ 下所有 HDF5
 │   ├─ _index_demos() → 按 demo_0, demo_1 ... 索引
 │   ├─ __getitem__(idx):
 │   │   ├─ 多 proprio 键拼接 (joint_states + gripper_states)
 │   │   ├─ language = parse problem_info JSON
 │   │   └─ → WindowSample dict
 │   └─ train/val split 按 demo 级别划分
 │
 └─ [format=None] DummyVLADataset → 随机张量 (冒烟测试)

归一化预计算: scripts/compute_stats.py 或 libero_hybrid/scripts/compute_libero_stats.py
 └─ 遍历 HDF5 → fit normalizer → save JSON (action_stats.json / proprio_stats.json)
```

### C.3 推理调用链

```
HybridVLALiberoPolicy.from_checkpoint(ckpt_path)
 ├─ resolve_policy_config() → 发现/验证配置
 ├─ HybridVLAv2(cfg) + load_checkpoint()
 ├─ [如有] 加载 EMA shadow weights → 覆盖模型参数
 ├─ load_policy_normalizers() → ActionNormalizer + ProprioNormalizer
 └─ AutoProcessor.from_pretrained(backbone_name)

闭环 rollout (eval_libero_rollout.py):
 for step in range(max_steps):
  ├─ [每 ~4步] policy.semantic_step_from_obs(obs, language)
  │   ├─ obs_to_semantic_input() → processor tokenize
  │   ├─ model.semantic_step() → backbone + grounder
  │   └─ runtime_state.refresh_counter += 1
  │
  └─ policy.control_step_from_obs(obs, runtime, semantic)
      ├─ obs_to_raw_proprio() → 多键拼接 + dim 校验
      ├─ proprio_normalizer.normalize()
      ├─ model.control_step() ──────────────────────┐
      │   ├─ 检查 chunk cache 是否有效              │
      │   ├─ temporal_core() 始终运行(更新 SSM 状态) │
      │   ├─ [需要新 chunk]:                        │
      │   │   ├─ _build_cond_prefix()               │
      │   │   ├─ action_expert.sample(midpoint, 8步) │
      │   │   └─ [RTC] 与 prev_chunk_tail 混合       │
      │   └─ action = chunk[:, chunk_step]            │
      ├─ action_normalizer.denormalize()
      └─ action.clamp(lo, hi)
```

### C.4 评测调用链

```
libero_hybrid/scripts/eval_libero_rollout.py::main()
 ├─ load_hybridvla_policy(ckpt, config) → HybridVLALiberoPolicy
 ├─ for task_id in tasks:
 │   └─ evaluate_task(policy, suite, task_id, n_eval=20, max_steps=600)
 │       ├─ SubprocVectorEnv([make_env] * n_eval) → 并行环境
 │       ├─ env.set_init_state(official_initial_states)
 │       ├─ policy.init_runtime(batch_size=n_eval) → per-env RuntimeCache
 │       ├─ for step in range(max_steps):
 │       │   ├─ policy.semantic_step_from_obs() [按 refresh 调度]
 │       │   ├─ policy.control_step_from_obs()
 │       │   ├─ env.step(action_env)
 │       │   └─ 检查 success/done
 │       └─ 报告 per-task success rate
 └─ 汇总所有 task 成功率
```

---

## D. 高风险审查区域

### D.1 致命风险（训练第一天可能崩溃）

| # | 风险点 | 文件:行 | 严重度 | 原因 |
|---|--------|---------|--------|------|
| **D1** | `control_step` 中 `action_history` 用 `torch.roll` 而 `forward_train` 用 `ActionHistoryBuffer`（基于索引） | `hybrid_vla_v2.py:844` vs `hybrid_vla_v2.py:458-465` | **HIGH** | 训练/推理语义不一致。推理时 `torch.roll` 每步分配新张量，但更关键的是两种实现对"最旧/最新"的定义不同，可能导致训练-推理分布偏移 |
| **D2** | FSDP `_log_per_module_grad_norm` 在 FSDP 包装后通过 `getattr(model, mod_name)` 访问子模块 | `train_unified.py:280` | **MEDIUM** | FSDP 包装后子模块名可能变为 `_fsdp_wrapped_module.backbone` 等，`getattr` 可能返回 None，导致 grad_norm 日志全部跳过。不会崩溃但会静默失去监控 |
| **D3** | Stage B/C `resume_from` 使用 `strict=False` 加载，形状不匹配键被静默丢弃 | `train_unified.py:385`, `checkpointing.py:145-158` | **MEDIUM** | 如果 Stage A→B 之间修改了模型结构（如 ActionHistoryEncoder 维度），被丢弃的键不会报错，但对应模块会用随机初始化，可能导致训练发散但难以诊断 |
| **D4** | `forward_train` 尾部 `micro_step % grad_accum != 0` flush 逻辑中，`epoch` 变量可能未定义 | `train_unified.py:597-611` | **LOW** | 如果训练 0 epoch（数据集为空或 max_steps=0），`epoch` 未绑定。极端边界情况 |
| **D5** | 多 GPU 下 `evaluate()` 中 `model.eval()` / `model.train()` 模式切换与 FSDP 的交互 | `train_unified.py:312,335` | **MEDIUM** | 在 EMA apply/restore 之间切换 eval/train 模式，如果 BatchNorm 存在（当前模型不含 BN，但 future-proof 考虑），可能导致不一致 |

### D.2 正确性风险（不崩溃但结果错误）

| # | 风险点 | 文件:行 | 严重度 | 原因 |
|---|--------|---------|--------|------|
| **D6** | `forward_train` 中 `fast_head` 在所有 T 步上计算但 `action_expert` 只在 `t=-1` 计算 | `hybrid_vla_v2.py:529-544` vs `571-622` | **HIGH** | FAST head 看到所有 T 步的 fused_state 生成离散预测，但一致性 loss 只比较 t=-1 的 FAST 连续输出与 expert 去噪输出。如果 T>1，FAST loss 和 consistency loss 的信息密度不对称，可能导致离散头与连续头发散 |
| **D7** | `PhaseHead` 和 `AffordanceHead` 已启用但数据适配器不提供 `phase_labels` / `affordance_labels` | `hybrid_vla_v2.py:149-158`, 所有 adapter | **HIGH** | 两个头的参数占用 cond_prefix 中各 1 个 token 位置，但训练时永远收不到梯度信号。这些 token 本质上是噪声，污染 expert 的条件输入。模型不会崩溃但 expert 的 2/32 条件 token 无意义 |
| **D8** | RTC 训练 loss 中 `prev_chunk` 用同一 `cond_prefix` + 小噪声生成，而非真正的前一步条件 | `hybrid_vla_v2.py:638-647` | **MEDIUM** | 训练时 prev_chunk ≈ curr_chunk（因为条件几乎相同），RTC loss 可能退化为自一致性正则化而非真正的跨 chunk 连续性约束 |
| **D9** | `ActionHistoryBuffer.get()` 在 `current_len < max_len` 时返回包含零填充的原始 buffer | `types.py:131-132` | **MEDIUM** | 前 K 步中，buffer 包含零填充行，`ActionHistoryEncoder` 会把零行当作"静止"动作处理。这可能引入偏置但不一定是 bug（取决于设计意图） |
| **D10** | `cosine_schedule_with_warmup` 在跨阶段恢复后从 step=0 重新开始 warmup | `train_unified.py:450,444-447` | **MEDIUM** | Stage B 的 `auto_resume` 返回 (0, 0)（因为 B 阶段目录里没有 checkpoint），scheduler 从 step=0 开始。这意味着 B 阶段的前 5000 步 LR 从 0 线性爬到目标值，但 expert 刚解冻、权重是从 A 加载的，这个 warmup 可能合理，但也可能不是设计意图 |
| **D11** | `evaluate()` 调用 `model.forward_train(batch)` 但在 eval 模式下 | `train_unified.py:329` | **LOW** | `forward_train` 内部有 dropout (Grounder 配置 dropout=0.0 所以当前无害)，但如果未来 dropout>0，eval 模式会关闭它，导致训练/验证 loss 不可比 |

### D.3 FSDP / 分布式风险

| # | 风险点 | 文件:行 | 严重度 | 原因 |
|---|--------|---------|--------|------|
| **D12** | EMA `update` 在所有 rank 上运行，`summon_full_params` 触发 all-gather，但 shadow 只在 rank 0 有意义 | `ema.py:96-100` | **MEDIUM** | 每个 rank 都维护完整 shadow 副本 + 每步 all-gather，内存开销 = N_rank × shadow_size。对 8×H100 可能额外消耗 ~12GB/GPU（shadow ~1.5B params × 2 bytes bf16）。功能正确但资源低效 |
| **D13** | `save_checkpoint` 中 `_get_optim_state_dict` 是 COLLECTIVE 操作，但随后的文件写入在 `is_main_process()` guard 内 | `checkpointing.py:86-90` | **OK** | 当前实现正确：COLLECTIVE ops 在 guard 前，只有 rank 0 写文件，最后 barrier。这个模式是正确的 |
| **D14** | `load_checkpoint` 中 FSDP 模型使用 `FullStateDictConfig(rank0_only=False)` 加载 | `checkpointing.py:165` | **LOW** | 所有 rank 加载完整 state_dict（~9B params），内存峰值可能很高。但由于 `map_location="cpu"`，应该不会 OOM |

### D.4 数据管线风险

| # | 风险点 | 文件:行 | 严重度 | 原因 |
|---|--------|---------|--------|------|
| **D15** | `vla_collate_fn` 对 `pixel_values` 做 padding 但对 `image_grid_thw` 不做 padding | 需要查看 collate.py | **MEDIUM** | 如果批次内不同样本的图像分辨率不同，pixel_values 维度可能不一致。Qwen2-VL 的 processor 通常已经处理了 padding，但需要验证 |
| **D16** | `hdf5_adapter.py` 在每次 `__getitem__` 时打开/关闭 HDF5 文件 | hdf5_adapter.py | **MEDIUM** | 高频 I/O 开销。h5py 文件句柄不是线程安全的，multi-worker DataLoader 下每个 worker 独立打开文件，可能导致文件锁竞争 |
| **D17** | `libero_hdf5_adapter.py` 中 `val_ratio` 按 demo 索引前 90% 训练 / 后 10% 验证 | libero_hdf5_adapter.py | **LOW** | 如果 demo 按难度/环境排序，train/val 分布不均。但 `sorted_libero_demo_keys` 做了数字排序，降低了这个风险 |

### D.5 推理/评测风险

| # | 风险点 | 文件:行 | 严重度 | 原因 |
|---|--------|---------|--------|------|
| **D18** | `eval_libero_rollout.py` 中 per-env 状态管理使用独立 RuntimeCache 但共享模型 | eval_libero_rollout.py | **HIGH** | SubprocVectorEnv 并行 20 个环境，但 semantic_step 对每个环境单独调用（batch_size=1）。这意味着 backbone forward 被调用 20 次而非一次 batch forward，推理速度可能比预期慢 20 倍 |
| **D19** | `HybridVLALiberoPolicy` 中 `multi_camera` 限制为 exactly 2 cameras | `libero_policy.py:206-209` | **LOW** | 配置文件 `libero_multicam.yaml` 定义 3 cameras，但推理策略硬编码限制 2 cameras。需要在推理前修复或确认这是有意为之 |
| **D20** | `action_env.clamp(lo, hi)` 在 denormalize 之后 clamp | `libero_policy.py:315-316` | **MEDIUM** | clamp 到 `action_range` (-1, 1) 但 denormalize 后的 action 应该在环境动作空间内。如果归一化 target_range=(-1,1) 且 denormalize 正确，clamp 是冗余的；但如果环境动作空间 ≠ [-1,1]，这个 clamp 会截断有效动作 |

---

## E. 缺失文件/信息对审查准确性的影响

| # | 缺失项 | 影响 | 严重度 |
|---|--------|------|--------|
| **E1** | 实际训练日志 / wandb 记录 | 无法验证 loss 曲线是否正常、梯度是否爆炸、LR schedule 是否合理 | **HIGH** |
| **E2** | `.github/workflows/*.yml` 的 CI 实际运行结果 | 无法确认 CI 测试是否全部通过 | **MEDIUM** |
| **E3** | `configs/train/stage_*_compressed.yaml` 的内容 | 不清楚 "compressed" 变体与标准变体的区别，可能隐藏配置错误 | **MEDIUM** |
| **E4** | `libero_hybrid/configs/` 下的 LIBERO 专用训练配置 | 已看到但需要交叉验证与主 config 的一致性 | **LOW** |
| **E5** | 实际 HDF5 数据文件样本 | 无法验证 adapter 解析逻辑在真实数据上的正确性 | **HIGH** |
| **E6** | `outputs/` 目录下的 `resolved_config.yaml` 示例 | 无法验证配置序列化/反序列化的往返一致性 | **LOW** |
| **E7** | Mamba CUDA kernel 版本兼容性 | `selective_scan.py` 有 fallback 但未验证 CUDA path 是否与 mamba_ssm>=2.0 API 兼容 | **MEDIUM** |
| **E8** | 多 GPU 下的端到端测试结果 | 当前测试全在 CPU/单 GPU mock 上运行，FSDP 路径未被真正测试 | **HIGH** |

---

## F. 审查优先级建议

基于以上分析，建议后续 code review 按以下优先级进行：

### P0 — 必须立即审查（训练第一天阻断）

1. **D1**: 训练/推理动作历史实现不一致
2. **D7**: Phase/Affordance 头启用但无监督信号 → cond_prefix 噪声污染
3. **D6**: FAST/expert 信息密度不对称
4. **D18**: 评测推理速度瓶颈（20× 预期外的 backbone forward）

### P1 — 重要审查（可能导致训练质量问题）

5. **D8**: RTC loss 退化为自一致性
6. **D3**: 跨阶段 resume 静默丢弃参数
7. **D10**: 跨阶段 LR warmup 行为
8. **D12**: EMA 内存效率

### P2 — 建议审查（代码健壮性）

9. **D2**: FSDP 下 grad_norm 日志失效
10. **D15-D17**: 数据管线边界情况
11. **D19-D20**: 推理策略硬编码限制

---

## G. 代码质量总览

| 维度 | 评价 | 评分 |
|------|------|------|
| 架构设计 | 三速率 Mamba + 层次化 Grounder + Flow Expert 的组合有理论支撑，模块边界清晰 | 9/10 |
| 配置管理 | dataclass + YAML 继承 + 未知键警告，完备 | 9/10 |
| 阶段门控 | 显式冻结/解冻 + 断言检查 + 日志，robust | 9/10 |
| FSDP 集成 | use_orig_params + activation_checkpointing + EMA summon，基本正确 | 8/10 |
| 数据管线 | 多格式适配器 + 归一化持久化 + collate，但缺少端到端验证 | 7/10 |
| 推理策略 | 归一化对齐 + 检查点发现 + EMA 加载，但评测循环效率低 | 7/10 |
| 测试覆盖 | 10 个测试文件覆盖主流程，但全在 CPU mock 上，FSDP 路径未测 | 7/10 |
| 死代码 | world_model ~1,200 行 enable=False，phase/affordance heads 无监督信号 | 6/10 |
| 文档 | README 极其详尽，但内联注释偏少 | 8/10 |

**整体就绪度**: 8.2/10 — 架构成熟、配置完备、阶段门控严密。主要风险集中在 训练/推理一致性（D1/D7）和评测效率（D18），这些在首次实际训练运行前应优先解决。

---

*本文件为代码审查的第一步——建立仓库理解地图。后续将按 P0→P1→P2 优先级逐项深入审查。*
