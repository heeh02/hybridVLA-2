# HybridVLA v2 仓库理解地图

版本: `architecture_gpt_v1_0_1`

范围: 基于当前仓库目录、`README.md`、依赖文件、配置文件、主入口脚本及核心模块源码建立审查地图；本文件不包含修复建议，只为后续完整 code review 提供结构化上下文。

## 0. 总体判断

这个仓库不是“单一训练脚本 + 若干模型文件”的简单项目，而是一个分成三层的研究型 Python VLA 系统：

| 层级 | 路径 | 角色 | 是否主路径 |
|---|---|---|---|
| 核心研究包 | `vla_hybrid_v2/` | 模型、数据、loss、推理、checkpoint、分布式训练 | 是 |
| Benchmark/场景适配层 | `libero_hybrid/` | LIBERO 数据布局、训练包装、闭环 rollout 评测 | 是 |
| 运行入口层 | `scripts/`、`libero_hybrid/scripts/`、`configs/` | 训练、统计、验证、评测、配置装配 | 是 |
| 实验/预留层 | `vla_hybrid_v2/experimental/world_model/`、`vla_hybrid_v2/world_model/` | 世界模型试验性代码，当前默认关闭 | 否 |
| 质量保障层 | `tests/` | 单测、smoke test、EMA/FSDP/推理回归 | 是 |
| 历史分析层 | `docs/` | 架构迭代、修复记录、历史 audit 文档 | 辅助 |

我更倾向把它理解成：

1. `vla_hybrid_v2` 是“通用 VLA 平台内核”。
2. `libero_hybrid` 是“首个真实 benchmark 适配实现”。
3. `scripts/train_unified.py` 是真实训练主控器，不再是早期多脚本散装训练。
4. `HybridVLALiberoPolicy` 是训练空间和 rollout 空间对齐的关键胶水层。

## A. 仓库模块划分

### A1. 顶层模块划分

| 模块 | 主要路径 | 核心对象/脚本 | 作用 |
|---|---|---|---|
| 配置系统 | `vla_hybrid_v2/config.py`, `configs/`, `libero_hybrid/configs/` | `HybridVLAv2Config`, `load_config` | 用 dataclass + YAML defaults 组织 model/train/infer/data 配置，并被训练/推理入口复用 |
| 模型系统 | `vla_hybrid_v2/models/` | `HybridVLAv2`, `Qwen2VLBackboneWrapper`, `HierarchicalAttentionGrounder`, `TriRateMambaCore`, `FlowActionExpert` | 实现 VLA 主网络和各子模块 |
| 损失系统 | `vla_hybrid_v2/losses/` | `FlowMatchingLoss`, `DiscreteCELoss`, `V2ConsistencyLoss` | 统一定义 Stage A/B/C 训练目标 |
| 数据系统 | `vla_hybrid_v2/data/` | `build_dataset`, `HDF5DatasetAdapter`, `LiberoHDF5DatasetAdapter`, `vla_collate_fn`, `Normalizer` | 负责 HDF5/LIBERO 数据读取、归一化、切窗、tokenize、batch 组装 |
| 推理系统 | `vla_hybrid_v2/infer/` | `HybridVLALiberoPolicy` | 将 checkpoint/config/normalizer/runtime state 拼成 rollout 可调用策略 |
| 训练基础设施 | `vla_hybrid_v2/utils/` | `save_checkpoint`, `load_checkpoint`, `EMAModel`, `wrap_fsdp` | FSDP、EMA、checkpoint、auto-resume |
| 低层算子 | `vla_hybrid_v2/ops/` | `ssm_scan`, `selective_scan_fn` | Mamba fallback/CUDA path 的底层 selective scan |
| Benchmark 适配 | `libero_hybrid/` | `train_libero.py`, `eval_libero_rollout.py`, `compute_libero_stats.py`, `validate_libero_hdf5.py` | 对接 LIBERO 的数据目录、训练阶段包装、官方 rollout |
| 训练入口 | `scripts/` | `train_unified.py`, `compute_stats.py`, `train_smoke_test.py` | 通用训练/统计/smoke 验证 |
| 测试 | `tests/` | 多个 `test_*.py` | 对 loss、expert、forward_train、control_step、EMA/FSDP、infer policy 做回归 |

### A2. `vla_hybrid_v2` 子模块再划分

| 子模块 | 路径 | 职责 |
|---|---|---|
| 配置与类型 | `config.py`, `types.py` | 定义所有训练/推理/模型 dataclass 配置和 runtime 类型 |
| Backbone | `models/qwen2vl_backbone.py` | Qwen2-VL 加载、冻结、LoRA、多尺度特征提取、多相机位置嵌入 |
| Grounder | `models/attention_grounder.py` | Perceiver 风格 latent grounder，含对象槽压缩 |
| 时序核心 | `models/mamba_core.py` | Fast/Medium/Slow 三频 Mamba、动作历史编码、陈旧时间编码、跨频融合 |
| 动作专家 | `models/flow_action_expert.py` | Flow matching 动作去噪网络、AdaRMSNorm、Euler/Midpoint 采样 |
| 离散头 | `models/discrete_heads.py` | FAST 离散动作头、phase 头、affordance 头 |
| 总装配 | `models/hybrid_vla_v2.py` | 训练 forward、语义 step、控制 step、Stage A/B/C loss 组合 |
| 数据适配 | `data/hdf5_adapter.py`, `data/libero_hdf5_adapter.py` | 通用 HDF5 与 LIBERO/robomimic HDF5 两套 schema 读取 |
| 归一化与 batch | `data/normalizer.py`, `data/collate.py`, `data/schema.py` | 统计读取、数值归一化、batch contract |
| 推理封装 | `infer/libero_policy.py` | config/checkpoint/stats 自动发现，obs -> model 输入对齐 |
| 训练基建 | `utils/checkpointing.py`, `utils/distributed.py`, `utils/ema.py` | 多卡包装、EMA shadow、保存恢复 |
| 试验性世界模型 | `experimental/world_model/` | 与主 VLA 的未来接口预留，当前未纳入主训练路径 |

## B. 每个目录/模块职责

### B1. 顶层目录职责

| 目录 | 职责 | 备注 |
|---|---|---|
| `configs/` | 通用模型/训练配置模板 | 面向 generic pipeline，不绑定 LIBERO 数据布局 |
| `scripts/` | 通用训练与数据准备入口 | `train_unified.py` 是当前标准入口；`train_stage_a.py` 仅兼容旧路径 |
| `vla_hybrid_v2/` | 核心 Python 包 | 绝大多数核心审查都应围绕这里展开 |
| `libero_hybrid/` | LIBERO 专用包装层 | 将 generic core 约束到真实 benchmark 环境和数据结构上 |
| `tests/` | 单元与回归测试 | 覆盖面不算低，但更多是功能回归，不是完整真实数据 E2E |
| `docs/` | 历史架构文档与修复记录 | 对理解演化过程有帮助，但不等于当前代码正确 |
| `outputs/` | 运行产物目录 | 不是源码，但后续若审查恢复/评测一致性，需要看真实产物结构 |

### B2. `scripts/` 职责

| 文件 | 职责 | 评语 |
|---|---|---|
| `scripts/train_unified.py` | Stage A/B/C 统一训练主入口 | 当前最关键的 orchestration 文件 |
| `scripts/train_stage_a.py` | 已弃用的 Stage A wrapper | 只做兼容，不是主审查对象 |
| `scripts/compute_stats.py` | 通用 HDF5 数据统计 | 更适合 generic HDF5，不适合 LIBERO 多 proprio 拼接场景 |
| `scripts/train_smoke_test.py` | 小模型 smoke test | 验证图很重要，但不能替代真实数据 E2E |

### B3. `libero_hybrid/` 职责

| 文件/目录 | 职责 | 评语 |
|---|---|---|
| `libero_hybrid/configs/` | LIBERO 单/多相机、Stage A/B/C 配置 | 与通用 `configs/` 有策略差异，后续 review 必须对齐 |
| `libero_hybrid/scripts/train_libero.py` | 根据 `suite`/`variant`/`stage` 组装真实训练配置 | 实际 benchmark 训练从这里进，不是直接跑 `train_unified.py` |
| `libero_hybrid/scripts/compute_libero_stats.py` | LIBERO 统计计算 | 处理多 proprio key 拼接 |
| `libero_hybrid/scripts/validate_libero_hdf5.py` | 训练前 HDF5 完整性检查 | 属于数据侧“预防性入口” |
| `libero_hybrid/scripts/eval_libero_rollout.py` | 官方风格 rollout 评测 | 闭环评测主路径 |
| `libero_hybrid/utils.py` | suite 解析、demo 排序、`problem_info` 语言提取 | 轻量但关键的 benchmark 目录兼容层 |

### B4. `vla_hybrid_v2/models/` 职责

| 文件 | 职责 | 关键点 |
|---|---|---|
| `hybrid_vla_v2.py` | 主模型总装配与 train/infer 两条主逻辑 | 最重要的系统级控制点 |
| `qwen2vl_backbone.py` | 加载 Qwen2-VL、应用 LoRA、提取多尺度特征 | 外部依赖最重，版本敏感 |
| `attention_grounder.py` | 96 latent grounder + 48->24 object slot compression | 承担变长语义到定长 token 的收缩 |
| `mamba_core.py` | Tri-rate temporal core、history encoder、cross-attn fusion | 承担时序缓存和 recurrent state |
| `flow_action_expert.py` | Flow matching 动作块去噪与 ODE 采样 | 训练和推理差异最大 |
| `discrete_heads.py` | 离散 chunk 预测与 phase/affordance 分类 | 与 flow head 形成双头监督 |

### B5. `vla_hybrid_v2/data/` 职责

| 文件 | 职责 | 关键点 |
|---|---|---|
| `__init__.py` | `build_dataset` 工厂 | 连接 normalizer stats 与具体 adapter |
| `hdf5_adapter.py` | 通用 episode.hdf5 读取 | 预期 schema 与 LIBERO 不同 |
| `libero_hdf5_adapter.py` | 官方 LIBERO/robomimic 结构读取 | 当前真实 benchmark 主路径 |
| `collate.py` | 处理 vision tensor 与 refresh list 的组 batch | 多相机/refresh frame 会经过这里 |
| `normalizer.py` | 统计读写、归一化与反归一化 | 训练与 rollout 对齐核心 |
| `schema.py` | WindowSample contract | 是 data/model 的接口契约定义 |
| `transforms.py` | 训练时图像增强 | 只在 train split 生效 |
| `dummy.py` | dummy data | 用于最小闭环与 smoke 测试 |

### B6. `vla_hybrid_v2/utils/` 职责

| 文件 | 职责 | 关键点 |
|---|---|---|
| `checkpointing.py` | 保存/加载/自动恢复/资产打包 | 同时处理 FSDP full state 与普通模型 |
| `distributed.py` | 分布式初始化、FSDP wrap、activation checkpointing、grad clip | 训练稳定性关键 |
| `ema.py` | EMA 更新、apply/restore、FSDP name/param 对齐 | 与 eval 和 resume 深度耦合 |

## C. 训练、数据处理、评测、推理的主要调用链

### C1. 通用训练主链

```text
configs/train/stage_{a|b|c}.yaml
  -> vla_hybrid_v2.config.load_config()
  -> scripts.train_unified.train(cfg)
      -> setup_distributed()
      -> _save_resolved_config()
      -> _checkpoint_assets()
      -> HybridVLAv2(cfg)
      -> configure_trainable_modules(stage)
      -> optional load_checkpoint(resume_from)
      -> optional EMAModel(model)
      -> optional wrap_fsdp(model)
      -> AutoProcessor.from_pretrained(backbone.name)
      -> build_dataset(cfg, split="train"/"val", processor)
      -> DataLoader(...)
      -> for each batch:
           -> model.forward_train(batch)
           -> backward
           -> clip_grad_norm_fsdp()
           -> optimizer.step()
           -> scheduler.step()
           -> ema.update()
           -> periodic evaluate()
           -> periodic save_checkpoint()
```

### C2. `forward_train()` 内部主链

```text
HybridVLAv2.forward_train(batch)
  -> _validate_batch(batch)
  -> 计算 semantic refresh schedule / medium update schedule
  -> backbone.forward_semantic(...)
  -> grounder(...)
  -> 初始化 TriRateTemporalState / ActionHistoryBuffer
  -> 对 T 个时间步循环:
       -> proprio_proj / prev_action_proj / embodiment_embedding
       -> stale_encoder(steps_since_refresh)
       -> action_history_encoder.encode()
       -> temporal_core(...)
       -> 累积 fused_states / fast_tokens / temporal_outputs
  -> FAST discrete loss (all T)
  -> phase/affordance loss (all T, 但依赖 batch 是否提供 label)
  -> Stage B/C:
       -> _build_cond_prefix(last_grounder, last_temporal)
       -> optional cond_prefix.detach()
       -> FlowMatchingLoss.sample_timestep()
       -> action_expert(...)
       -> flow matching loss
       -> optional RTC loss
       -> optional FASTER loss
       -> consistency loss
  -> 汇总 loss_total
```

### C3. 数据处理主链

#### 通用 HDF5

```text
scripts.compute_stats
  -> 遍历 .hdf5
  -> ActionNormalizer / ProprioNormalizer.fit()
  -> 保存 normalizer_stats/*.json

train_unified.train()
  -> build_dataset(cfg, processor)
  -> HDF5DatasetAdapter(...)
      -> 发现 episode_paths
      -> _build_index() 建立 (episode_idx, start)
      -> __getitem__():
           -> 读取 action/proprio/image
           -> 切出 [T + H - 1] 动作
           -> 归一化
           -> 组装 action_chunks / prev_actions
           -> processor 文本/图像 tokenize
           -> 生成 refresh_* 字段
  -> vla_collate_fn()
  -> forward_train()
```

#### LIBERO HDF5

```text
libero_hybrid.scripts.validate_libero_hdf5
  -> 检查 data/demo_x/obs/... 是否完整

libero_hybrid.scripts.compute_libero_stats
  -> 对每个 task file 遍历 demo_x
  -> 拼接 proprio_keys
  -> 保存 normalizer_stats/*.json

libero_hybrid.scripts.train_libero
  -> build_cfg(stage, suite, variant, stats_dir, output_root, ...)
      -> load_config(libero_stage_*.yaml)
      -> _apply_variant(singlecam|multicam)
      -> resolve_libero_suite_dir()
      -> 覆盖 data.format / data_dir / val_data_dir / resume_from / output_dir
  -> scripts.train_unified.train(cfg)
      -> build_dataset(...)
      -> LiberoHDF5DatasetAdapter(...)
```

### C4. 评测调用链

#### 训练中的离线评测

```text
scripts.train_unified.evaluate()
  -> val_loader
  -> model.forward_train(batch)
  -> 聚合 loss_* 平均值
```

这是“离线 batch 验证”，不是机器人闭环 rollout。

#### LIBERO 闭环 rollout 评测

```text
libero_hybrid.scripts.eval_libero_rollout.main()
  -> load_hybridvla_policy()
      -> HybridVLALiberoPolicy.from_checkpoint()
          -> resolve_policy_config()
          -> HybridVLAv2(cfg)
          -> load_checkpoint()
          -> optional apply EMA
          -> load_policy_normalizers()
          -> AutoProcessor.from_pretrained()
  -> get_benchmark(...)
  -> evaluate_task(...)
      -> SubprocVectorEnv(...)
      -> 每个 env 独立 policy.init_runtime()
      -> 按 refresh_interval:
           -> policy.semantic_step_from_obs()
      -> 每个 control step:
           -> policy.control_step_from_obs()
               -> obs_to_raw_proprio()
               -> proprio_normalizer.normalize()
               -> model.control_step(...)
               -> action_normalizer.denormalize()
      -> env.step(actions_batch)
      -> 汇总 success_rate
```

### C5. 在线推理主链

```text
HybridVLALiberoPolicy.control_step_from_obs()
  -> obs_to_raw_proprio()
  -> proprio_normalizer.normalize()
  -> HybridVLAv2.control_step(
       proprio,
       prev_action,
       semantic_summary,
       runtime_cache,
     )
      -> temporal_core(...)
      -> 若 chunk cache 失效:
           -> _build_cond_prefix(...)
           -> action_expert.sample(midpoint/euler)
           -> optional RTC overlap blend
      -> 取 current_chunk[:, chunk_step]
      -> 更新 runtime_cache.action_history / chunk_step
  -> action_normalizer.denormalize()
  -> clip 到 action_range
```

## D. 后续必须重点审查的高风险区域

### D1. 高风险清单

| 风险级别 | 模块 | 原因 |
|---|---|---|
| 高 | `vla_hybrid_v2/models/hybrid_vla_v2.py` | 这里把 backbone、grounder、temporal core、expert、loss、stage gate、teacher forcing、RTC/FASTER 全部拼在一起；任何一个时间维、detach、refresh schedule、chunk 对齐错误都会“训练能跑但语义错” |
| 高 | `scripts/train_unified.py` | 真实训练总控，涵盖 FSDP、EMA、resume、param groups、eval barrier、checkpoint 资产；这是最容易出现分布式死锁、错误恢复、训练阶段冻结错误的地方 |
| 高 | `vla_hybrid_v2/data/libero_hdf5_adapter.py` | LIBERO 真实数据主路径；涉及 demo 级切分、`T + H - 1` 对齐、语言提取、多 proprio 拼接、多相机 refresh tokenize，极易出现 silent data bug |
| 高 | `vla_hybrid_v2/infer/libero_policy.py` | 训练空间与 rollout 空间的归一化、配置发现、EMA 应用、obs key 映射都在这里；任何错位都直接导致“训练看似正常但 rollout 失败” |
| 高 | `libero_hybrid/scripts/eval_libero_rollout.py` | 闭环 benchmark 的实际落点；per-env runtime state、refresh 频率、向量环境状态隔离很敏感，且成功率是对外结果指标 |
| 高 | `vla_hybrid_v2/utils/checkpointing.py` + `utils/ema.py` | FSDP full state、shape mismatch 过滤、EMA shadow 命名、cross-stage resume 顺序都可能造成参数静默丢失或加载不全 |
| 中高 | `vla_hybrid_v2/models/mamba_core.py` | 三频状态缓存、medium/slow reuse、stale-time 编码、官方/回退双路径并存；属于数值与时序语义都敏感的核心 |
| 中高 | `vla_hybrid_v2/models/flow_action_expert.py` | 训练 forward 与采样 sample 是两套路径，Euler/Midpoint/AdaRMSNorm/selective_scan fallback 都可能出现行为不一致 |
| 中高 | `vla_hybrid_v2/models/qwen2vl_backbone.py` | 外部 HF/Qwen2-VL/PEFT 版本强依赖；freeze、LoRA、multi-camera embedding、processor/token format 容易受上游变动影响 |
| 中 | `vla_hybrid_v2/data/collate.py` | `refresh_*_list` 的 batch 转置、None/Tensor 混合填零、视觉张量 padding 都可能引入隐蔽错误 |
| 中 | `libero_hybrid/configs/*.yaml` 与 `configs/*.yaml` | 文档宣称的训练策略与实际 benchmark config 存在差异；例如 generic Stage C 打开 RTC/FASTER，但 LIBERO Stage C 默认关闭，两者需要重点对齐 |
| 中 | `vla_hybrid_v2/experimental/world_model/` | 当前默认关闭，不是主线风险；但如果用户后续想审查“设计完整性”而非“可运行主线”，这一块必须单独拆开评估 |

### D2. 我认为最该优先深挖的 5 个点

1. `forward_train()` 的时间维语义是否自洽。
   重点看 refresh schedule、`refresh_map`、`medium_set`、`action_chunks[t] = actions[t:t+H]`、last-step expert supervision 是否严格对应设计。

2. 数据归一化与 rollout 反归一化是否完全对齐。
   重点看 `compute_*_stats.py`、`build_dataset()`、`HybridVLALiberoPolicy.load_policy_normalizers()`、`control_step_from_obs()`。

3. Stage A/B/C 的冻结、resume、EMA 顺序是否真的符合设计。
   重点看 `configure_trainable_modules()`、cross-stage `load_checkpoint()`、EMA 初始化时机、FSDP wrap 前后参数名一致性。

4. LIBERO adapter 的 schema 假设是否与真实数据完全一致。
   重点看 `problem_info`、`obs` 键名、相机键名、`joint_states + gripper_states` 拼接、val split 逻辑、demo 长度边界。

5. rollout runtime cache 是否严格做到 per-env 隔离且 refresh 行为正确。
   重点看 `RuntimeCache`、`refresh_counter`、`_last_seen_refresh`、`current_chunk`、`prev_chunk_tail`、`action_history`。

## E. 还缺哪些文件/信息会影响后续 code review 的准确性

下面这些内容如果没有，后续 review 结论会明显打折：

| 缺失项 | 为什么重要 |
|---|---|
| 真实训练时产出的 `resolved_config.yaml` | `train_libero.py` 和 wrapper 会覆盖默认 YAML；不看 resolved config，容易把“模板配置”和“真实运行配置”混为一谈 |
| 一份真实 LIBERO HDF5 样本结构 | 代码假设 `data/demo_x/obs/*`、`problem_info`、图像键名和 proprio 维度；没有样本就只能审静态逻辑，无法确认 schema 兼容性 |
| 一份真实通用 HDF5 样本结构 | generic `hdf5_adapter.py` 与 LIBERO schema 不同；如果后续要审 generic pipeline，需要真实文件结构确认 |
| 实际训练/评测命令与环境版本 | `transformers`、`peft`、`mamba_ssm`、`libero`、`robosuite` 的版本会直接影响 backbone、processor 和 Mamba path 行为 |
| 真实 checkpoint 目录样本 | checkpoint 是否真的带 `assets/resolved_config.yaml`、`assets/normalizer_stats/`，会决定推理/恢复链是否可靠 |
| 如果要评估世界模型分支: `vla_hybrid_v2/experimental/world_model/*.py` 的设计意图和计划使用方式 | 当前主路径未启用；没有“预期接入方式”说明，很难判断这些代码是未完成、废弃，还是仅未接线 |
| 若存在 CI/发布要求: `.github/workflows/*` | 这会影响后续对测试可信度、依赖矩阵、回归保护强度的判断 |
| 若存在训练日志/评测日志样本 | 有助于判断代码路径是否真的被跑过，哪些模块只“理论存在”但缺真实验证 |

## 附加观察

### 1. 这个仓库的真实主线不是 README 里写的“所有能力都已落地”

代码里能看到明显分层：

- 主训练/推理/benchmark 路径已经打通。
- 世界模型路径仍是预留和试验性分支。
- `libero_hybrid` 是真实 benchmark 首发路径，很多设计比 README 更保守。

最明显的例子是：

- generic `configs/train/stage_c.yaml` 打开了 `rtc.enable=true` 和 `faster.enable=true`
- 但 `libero_hybrid/configs/train/libero_stage_c.yaml` 默认把两者都关掉，且 `stop_gradient_cond_prefix=true`

这说明“论文式总设计”和“当前 benchmark 生产配置”之间有策略落差，后续 review 不能只按 README 宣称来判。

### 2. 这个项目最值得警惕的是“静默错误”

不是那种一跑就 crash 的错误，而是：

- batch shape 对得上，但监督时间对齐错了
- normalizer 文件能加载，但训练/推理空间不一致
- checkpoint 能恢复，但恢复的是不完整参数
- rollout 能执行，但 runtime state 在多 env 之间串了
- multi-camera tokenizer 能跑，但 camera ordering 与 embedding 对不上

因此后续 code review 需要优先查“语义一致性”和“跨模块契约”，而不只是局部 Python 风格问题。

## 建议的下一轮审查入口顺序

1. `vla_hybrid_v2/models/hybrid_vla_v2.py`
2. `scripts/train_unified.py`
3. `vla_hybrid_v2/data/libero_hdf5_adapter.py`
4. `vla_hybrid_v2/infer/libero_policy.py`
5. `vla_hybrid_v2/utils/checkpointing.py` / `vla_hybrid_v2/utils/ema.py`
6. `vla_hybrid_v2/models/mamba_core.py`
7. `libero_hybrid/scripts/eval_libero_rollout.py`

以上顺序更适合“完整 code review”，因为它先覆盖最可能造成系统级行为错误的链路，再下钻到单模块实现。
