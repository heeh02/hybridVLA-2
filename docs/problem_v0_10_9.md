# HybridVLA v2 v0.10.9+ — 综合问题清单 (架构关联版)

> 汇总日期: 2026-03-28
> 验证日期: 2026-03-28 (基于当前代码状态重新核验每项问题)
> 来源文档:
> 1. `comparsion_between_pi_and_hybridVLA_3_gpt.md` — GPT 代码核验版
> 2. `analysis_v0_10_9_fix_review_v3.md` — 深度结构与效率分析
> 3. `comparsion_between_pi_and_hybridVLA_3_claude.md` — Claude 客观对比报告 v3
> 4. `code_review_for_pi_hybridvla.md` — Claude 详细代码审查

---

## 零、模型架构总览

### 0.1 模块层级 & 参数分布

```
HybridVLAv2 (~9.5B total: 7.5B frozen + 2.0B trainable)
│
├── [Layer 1] Qwen2VLBackboneWrapper                      ~7.6B (7.5B frozen + ~120M LoRA)
│   ├── Qwen2-VL-7B-Instruct (3584d, 28 layers)
│   │   ├── Vision Tower (frozen)
│   │   ├── Text Layers 0-15 (frozen)
│   │   └── Text Layers 16-27 (LoRA r=64, Stage C unfreeze)
│   ├── MultiScaleAdapter: 3×[LN(3584)+Linear(3584→2048)] + gated sum
│   └── CameraPositionEmbedding: Embedding(8, 2048)
│
├── [Layer 2] HierarchicalAttentionGrounder               ~256M trainable
│   ├── 96 Learned Latents: [global(1) + objects(48) + phase(1) + unc(1) + aff(1) + aux(44)]
│   ├── Layers 1-4: 4× GrounderBlock [CrossAttn(16h) + SelfAttn(16h) + FFN(2048→8192)]
│   ├── SlotCompression (at layer 4): CrossAttn 48→24 slots via learned routing queries
│   └── Layers 5-8: 4× GrounderBlock on 72 tokens (after compression)
│   Output: GrounderOutput {global[B,2048], compressed_slots[B,24,2048], phase/unc/aff[B,2048]}
│
├── [Layer 3] TriRateMambaCore                            ~840M trainable
│   ├── StaleTimeEncoding: sinusoidal(2048) + MLP(2048→2048)
│   ├── FastMamba: 20L MambaBlock, d=2048, d_state=128   ~540M    [runs every step, ~50Hz]
│   ├── MediumMamba: 6L MambaBlock, d=2048, d_state=128  ~160M    [runs every 2 steps, ~25Hz]
│   ├── SlowMamba: 10L MambaBlock, d=2048, d_state=256   ~270M    [runs on refresh, ~12.5Hz]
│   ├── CrossAttentionFusion: 2L [fusion_query(learned) × MHA(8h) + FFN(2048→8192)]
│   └── Input per step: 33 tokens = 9 singles + 24 compressed slots
│   Output: TemporalOutput {fused_state, fast/medium/slow_token [B,2048], next_state}
│
├── [Layer 3a] ActionHistoryEncoder                       ~1.7M trainable (v0.10.10 fix)
│   ├── action_proj: Linear(14→256)
│   ├── _MambaStack: 2L MambaBlock, d=256, d_state=64
│   └── out_proj: Linear(256→2048)
│   Input: [B,8,14] (last 8 actions) → Output: [B,2048]
│
├── [Layer 4] Condition Prefix Builder
│   ├── Token assembly: [global(1) + slots(24) + phase(1) + unc(1) + aff(1)
│   │                     + fused(1) + fast(1) + medium(1) + slow(1)] = 32 tokens
│   ├── cond_builder: LN(2048) + Linear(2048→2048) + GELU + Linear(2048→2048)
│   └── core_to_expert: Linear(2048→1536)
│   Output: cond_prefix [B, 32, 1536]
│
├── [Layer 5] FlowActionExpert                            ~384M trainable
│   ├── action_proj: Linear(14→1536) + LearnedPositionalEmbedding(24, 1536)
│   ├── t_embed: SinusoidalTimestepEmbedding(1536) + MLP(1536→6144→1536)
│   ├── 18 Layers (M-M-A × 6 pattern):
│   │   ├── 12× ExpertMambaBlock (d=1536, d_state=96, AdaRMSNorm)
│   │   └── 6× ExpertAttentionBlock (d=1536, 24 heads, cross+self+FFN, AdaRMSNorm)
│   ├── Sample: midpoint ODE solver (2nd-order, default 8 steps)
│   └── Input: noisy_actions[B,24,14] + flow_t[B] + cond_prefix[B,32,1536]
│   Output: velocity[B,24,14] → denoised_action[B,24,14]
│
├── [Layer 6] Prediction Heads                            ~1M trainable
│   ├── FASTDiscreteHead: LN(2048)→768→GELU→[24×192]→[14×512] logits
│   ├── PhaseHead: LN(2048)→1024→GELU→16 classes
│   └── AffordanceHead: LN(2048)→1024→GELU→8 types
│
├── [Layer 7] Loss Functions
│   ├── FlowMatchingLoss: MSE(v_pred, x_1-x_0), logit_normal t schedule
│   ├── DiscreteCELoss: CE with label_smoothing=0.1 on 512-bin discrete
│   ├── V2ConsistencyLoss:
│   │   ├── ContrastiveTemporalLoss: InfoNCE + VICReg variance term
│   │   ├── SlowFastAgreementLoss: MSE(slow, EMA(fast)), bidirectional
│   │   └── ActionConsistencyLoss: MSE(discrete_decoded, continuous) in 14D
│   └── RTC Loss (Stage C): MSE(curr_head, prev_tail) + smoothness
│
└── [Experimental] WorldModel (enable=false)
    └── vla_hybrid_v2/experimental/world_model/ (10 files, ~1,200 lines)
```

### 0.2 三阶段训练方案

| | Stage A | Stage B | Stage C |
|--|---------|---------|---------|
| **目标** | 感知对齐 | 动作生成 | 精细调优 |
| **步数** | 120K | 200K | 80K |
| **Backbone** | LoRA (r=64) | LoRA | LoRA + text L16-27 unfreeze |
| **Grounder** | Trainable | Trainable | Trainable |
| **TriRate Core** | Trainable | Trainable | Trainable |
| **Expert** | **Frozen** | Trainable (cond.detach) | Trainable |
| **EMA** | Off | **Starts** | Continues |
| **RTC** | Off | Off | **Enabled** |
| **FASTER** | Off | Off | **Enabled** |
| **LR** | backbone 2e-5, core 2e-4 | + expert 1e-4 | 同 B |
| **Optimizer** | AdamW fused, betas=(0.9,0.95), wd=0.01 |||
| **Batch** | global 64 = 8 GPU × B2 × accum4 |||
| **Precision** | bfloat16, FSDP FULL_SHARD |||

### 0.3 训练数据流

```
HDF5/LIBERO Dataset
  → WindowSample: actions[T=24,H=24,A=14], proprio[T,14], prev_actions[T,14],
    input_ids[L], refresh_pixel_values_list[R=4], ...
  → vla_collate_fn() → Batch [B, ...]
  → forward_train():
    1. Backbone × R refreshes → [B, N, 2048] features
    2. Grounder × R → GrounderOutput (global + 24 slots + tokens)
    3. Temporal Core × T=24 loop → fused_states[B,T,2048], fast_tokens[B,T,2048]
    4. FAST Head on all T steps → loss_fast (multi-step supervision)
    5. Expert on t=-1 only (Stage B/C) → loss_fm
    6. RTC + FASTER (Stage C) → loss_rtc, loss_faster
    7. Consistency on fused_states → loss_consistency
    8. loss_total = sum(all losses)
```

### 0.4 推理数据流

```
Env Observation
  → semantic_step(): Backbone → Grounder → GrounderOutput (cached)
  → control_step() @ 50Hz:
    1. Temporal Core (always runs, updates recurrent SSM state)
    2. If need_new_chunk: Expert.sample(midpoint ODE, 8 steps)
       → RTC blend with prev_chunk_tail
       → Cache new chunk
    3. Return chunk[chunk_step] → single action [B, 14]
```

---

## 一、总体评估

四份文档的共识结论:

- v0.10.9 已修复所有已知 FSDP/EMA P0 bug，代码层面正确性显著提升。
- **核心差距**: 从未在任何数据上完成过一次端到端训练 — 所有架构"优势"尚属纸上谈兵。
- **已修复大量问题**: 用户对 L-1 ~ L-20 中的多项进行了修复 (见下表验证结果)。
- **综合评分**: GPT 无打分 / Claude-对比 4.0 / Claude-代码审查 B- / 结构分析 6.9 (修复前)。

---

## 二、当前可解决 — 验证状态

> 以下每项均基于当前代码状态重新核验。

### A. 代码正确性 Bug

| ID | 架构层 | 严重性 | 问题 | 状态 | 验证证据 |
|----|--------|--------|------|------|---------|
| **L-1** | Layer 7 → forward_train | **P0** | Optional field contract bug | **FIXED** | `batch.get("phase_labels") is not None` (L542), `batch.get("affordance_labels") is not None` (L554), `batch.get("refresh_input_ids") is not None` (L413) — 全部改为 `.get()` + None 检查 |
| **L-2** | Layer 4 → _build_cond_prefix | **P1** | 静默截断 cond_prefix | **FIXED** | 截断时 `logger.warning(...)` (L278-283)，明确警告 temporal tokens 被丢弃 |
| **L-3** | Layer 1 → CameraPositionEmbedding | **P1** | 所有 batch item 应用相同 cam_emb | **FIXED** | 现在 per-batch-item 循环 `for b in range(B): start_img = b * images_per_batch` (L99-122) |
| **L-4** | 训练循环 → evaluate | **P1** | evaluate 缺 dist.barrier() | **FIXED** | `dist.barrier()` 在 `ema.apply()` 前 (L561) 和 `ema.restore()` 后 (L567) |
| **L-5** | Layer 5+7 → RTC train path | **P1** | RTC train-infer 分布不一致 | **MITIGATED** | 训练 RTC 现在对 cond_prefix 加噪声 `prev_cond = cond_prefix + 0.01 * randn_like(...)` (L634-635)，注释承认 L-5。**根本分布偏移仍在**: 训练用当前帧+噪声, 推理用上一帧真实 cond |
| **L-6** | Layer 5 → control_step | **P1** | FASTER 推理 NotImplementedError | **OPEN** | 仍 `raise NotImplementedError(...)` (L731)，但错误信息改进为引导用户禁用 `infer.faster.enable=False` |
| **L-7** | types.py → ActionHistoryBuffer | **P2** | push() 用 torch.roll | **FIXED (train) / OPEN (infer)** | 训练路径: index-based 环形缓冲 `buffer[:, _write_idx % max_len]` (types.py:119)。推理路径: `control_step` 仍用 `torch.roll` (L839) |
| **L-8** | Layer 7 → forward_train | **P2** | loss_total 用 Python sum() | **FIXED** | 现为 `torch.stack(list(losses.values())).sum()` (L694) |
| **L-9** | Layer 7 → FlowMatchingLoss | **P3** | 未使用 t 参数 | **FIXED** | 添加注释说明 Rectified Flow 下 target velocity 与 t 无关 (flow_matching.py:17) |
| **L-10** | forward_train → _validate_batch | **P3** | 使用 assert | **FIXED** | 全部改为 `raise ValueError(...)` (L304-372) |

### B. 架构/设计问题

| ID | 架构层 | 严重性 | 问题 | 状态 | 验证证据 |
|----|--------|--------|------|------|---------|
| **L-11** | Layer 3a → ActionHistoryEncoder | **P1** | 过参数化 (原 108M) | **FIXED** | d_inner=256, min(num_layers,2) 层 (mamba_core.py:538-543)，降至 ~1.7M 参数 |
| **L-12** | Layer 7 → ContrastiveTemporalLoss | **P2** | Collapse 风险 | **FIXED** | 添加 VICReg-style variance regularization (consistency_loss.py:53-59) |
| **L-13** | Layer 7 → ActionConsistencyLoss | **P2** | 约束过弱 | **FIXED** | 改为 14D action 空间直接 MSE (consistency_loss.py:99) |
| **L-14** | Layer 7 → SlowFastAgreementLoss | **P2** | .detach() 单向 | **FIXED** | 移除 .detach()，双向梯度 (consistency_loss.py:82) |
| **L-15** | Layer 6 → PhaseHead/AffordanceHead + Data | **P2** | 头缺少监督标签 | **PARTIALLY FIXED** | 添加 logger.warning (L149-159) 提醒数据需提供标签。**但头仍默认开启** (HeadsConfig.phase_head=True)，无标签时 loss 为 0、cond_prefix 中的 token 为噪声 |
| **L-16** | Layer 5 → ExpertMambaBlock | **P2** | 未用 official Mamba2 | **FIXED** | 添加 `HAS_MAMBA2_MODULE` 检测 (flow_action_expert.py:27-31) |
| **L-17** | Layer 6 → FASTDiscreteHead | **P3** | 命名误导 | **OPEN** | 仍名 FASTDiscreteHead (discrete_heads.py)，与 pi-0-FAST 机制完全不同 |
| **L-18** | Layer 7 → V2ConsistencyLoss | **P3** | 子权重硬编码 | **FIXED** | 构造函数参数化 `slow_fast_weight`, `action_weight`, `temperature` (consistency_loss.py:108-111) |
| **L-19** | Config → _dict_to_dataclass | **P3** | eval() 解析 annotation | **FIXED** | 改用 `typing.get_type_hints()` (config.py:371-377) |

### C. 死代码/代码质量

| ID | 架构层 | 严重性 | 问题 | 状态 | 验证证据 |
|----|--------|--------|------|------|---------|
| **L-20** | [Experimental] WorldModel | **P2** | 1,200 行死代码 | **FIXED** | 移至 `vla_hybrid_v2/experimental/world_model/` (10 files)，主模型通过 `if wmcfg.enable:` 条件 import (L207-210) |
| **L-21** | scripts/ | **P2** | train_stage_a.py 遗留入口 | **OPEN** | `scripts/train_stage_a.py` 仍存在 (278 行)，运行语义与 train_unified.py 分叉 |
| **L-22** | 全局 | **P3** | 未使用 imports | **MOSTLY FIXED** | 仅剩 mamba_core.py 一处 `noqa: F401` (mamba_ssm import) |
| **L-23** | Layer 7 → forward_train | **P3** | 过长 (~310 行) | **OPEN** | forward_train 仍为单一长方法，cyclomatic complexity ~25 |
| **L-24** | 根目录 | **P3** | 无 CI/CD | **OPEN** | 仍无 `.github/workflows/` 或 `.pre-commit-config.yaml` |

### D. 测试覆盖缺失

| ID | 架构层 | 严重性 | 需补测试 | 状态 |
|----|--------|--------|---------|------|
| **L-25** | Layer 3 → TriRateMambaCore | **P1** | 0 专项测试 (786 行最复杂模块) | **OPEN** |
| **L-26** | Layer 2 → Grounder | **P1** | 0 专项测试 (261 行) | **OPEN** |
| **L-27** | Layer 3 → _MambaStack official path | **P2** | 0 测试 (需 mamba_ssm) | **OPEN** |
| **L-28** | 推理 → control_step × N | **P2** | 0 multi-step rollout 测试 | **OPEN** |
| **L-29** | 训练 → EMA | **P2** | 0 apply/restore round-trip 测试 | **OPEN** |
| **L-30** | Layer 1 → multi-camera | **P2** | 0 multi-camera forward 测试 | **OPEN** |
| **L-31** | Layer 7 → forward_train | **P2** | Optional field=None 测试 | **N/A** (L-1 已修复，但回归测试仍应补) |
| **L-32** | 训练 → FSDP | **P3** | 0 FSDP+training step 测试 | **OPEN** |
| **L-33** | types.py → ActionHistoryBuffer | **P3** | 0 环形行为测试 | **OPEN** |
| **L-34** | 推理 → RTC/FASTER | **P3** | 0 RTC blending/FASTER 测试 | **OPEN** |

---

## 三、HPC3 上机后解决

### E. 训练可行性验证

| ID | 架构层 | 严重性 | 问题 | 说明 |
|----|--------|--------|------|------|
| **H-1** | 全流程 | **P0** | **零验证** | 从未完成过一次训练 |
| **H-2** | 全流程 | **P0** | LIBERO 500 步单卡 smoke test | 确认 loss 下降, 无 NaN, GPU 显存够用 |
| **H-3** | 全流程 | **P0** | LIBERO 5K 步单卡收敛验证 | 确认收敛趋势 |
| **H-4** | 训练+FSDP | **P0** | 2×GPU smoke test (每 stage 100 步) | 验证 loss 非零, EMA 更新, LR 正确, eval 不死锁 |
| **H-5** | 训练+checkpoint | **P1** | 15K 步 Stage A→B 验证 | 确认跨阶段 checkpoint 加载, expert 学习 |
| **H-6** | FSDP scaling | **P1** | 8×H100 全规模验证 | FSDP bugs 已修复, 但从未实际运行 |

### F. 性能基准测试

| ID | 架构层 | 严重性 | 问题 | 说明 | 配置变更影响 |
|----|--------|--------|------|------|------------|
| **H-7** | Layer 3 → _MambaStack | **P0** | Benchmark official vs fallback Mamba | 100 步实测 step/s + GPU 显存 | **mamba_impl 已默认 "fallback"** (config.py:100)，训练不会走 token-by-token loop |
| **H-8** | Layer 3 → official path | **P1** | Token-by-token 循环性能测量 | ~19,536 次 step()/forward; **仅在 mamba_impl="auto" 且 mamba_ssm 安装时触发** | 当前默认 fallback，此问题**仅影响选择 auto 的场景** |
| **H-9** | Layer 3 → official path | **P1** | Activation checkpointing 验证 | official 路径下 checkpoint_wrapper 绕过 step() | **同上，fallback 路径有 activation_checkpoint** (mamba_core.py:472-475) |
| **H-10** | 全模型 | **P1** | GPU 显存峰值测量 | 估算 per-GPU ~22.5-31 GB | fallback path + checkpointing 应节省 2-4 GB |
| **H-11** | 全流程 | **P2** | 训练吞吐量实测 | 估算 per optimizer step: fallback ~6-8s, official ~12s | 默认 fallback 应在较好端 |

### G. 训练效果验证

| ID | 架构层 | 严重性 | 问题 | 说明 |
|----|--------|--------|------|------|
| **H-12** | Layer 7 → ConsistencyLoss | **P1** | Consistency loss 有效性验证 | L-12 已加 VICReg variance term; 需观察 loss curve 确认不 collapse |
| **H-13** | Layer 6 → Phase/Affordance Heads | **P1** | 头在真实数据上的效果 | L-15 部分修复; 确认 LIBERO 数据是否产出标签 |
| **H-14** | Layer 3a → ActionHistoryEncoder | **P2** | Ablation: 1.7M vs 更小 | L-11 已缩减; 验证缩减后是否影响收敛 |
| **H-15** | Layer 3 → TriRateMambaCore | **P2** | 频率分配 20/6/10 验证 | slow 每 ~8 步更新是否过于陈旧 |
| **H-16** | Layer 2 → SlotCompression | **P2** | 48→24 槽压缩效果 | 小数据集上能否学出有意义表示 |

### H. 性能优化 (长期)

| ID | 架构层 | 严重性 | 问题 | 说明 |
|----|--------|--------|------|------|
| **H-17** | Layer 3 | **P1** | 确认 fallback 路径综合效率 | 当前已默认 fallback; benchmark 确认决策正确 |
| **H-18** | Layer 3 → mamba_ssm | **P2** | Fork mamba_ssm 返回 SSM state | 长期最优解: 消除 Python loop 且保留 state |
| **H-19** | Layer 3 | **P2** | 减少 input_seq 长度 L | 线性减少循环次数 |
| **H-20** | 训练循环 | **P3** | TensorBoard/WandB 集成 | 训练可视化 |
| **H-21** | 推理 | **P3** | 推理服务化 (FastAPI/gRPC) | 目前仅 LIBERO benchmark |

---

## 四、修复进度总结

### 统计

| 类别 | 总数 | FIXED | PARTIALLY FIXED | MITIGATED | OPEN |
|------|------|-------|-----------------|-----------|------|
| A. 代码 Bug (L-1~10) | 10 | **8** | 0 | 1 (L-5) | 1 (L-6) |
| B. 架构设计 (L-11~19) | 9 | **7** | 1 (L-15) | 0 | 1 (L-17) |
| C. 死代码/质量 (L-20~24) | 5 | **2** | 1 (L-22) | 0 | 2 (L-21,23,24) |
| D. 测试缺失 (L-25~34) | 10 | **0** | 0 | 0 | 10 |
| **本地合计** | **34** | **17** | **2** | **1** | **14** |

### 仍 OPEN 的关键项

| ID | 问题 | 建议 |
|----|------|------|
| **L-5** | RTC train-infer 分布偏移 (已 mitigated 但根本问题在) | 可接受先上线, 在训练中观察 RTC loss 曲线; 长期需改用前一步 temporal state 生成 prev_chunk |
| **L-6** | FASTER 推理 NotImplementedError | Stage A/B 不受影响; Stage C 推理前须实现或确保 config 禁用 |
| **L-7** (infer) | control_step 仍用 torch.roll | 低影响 (推理 [B,8,14] 小 tensor), 非阻塞 |
| **L-15** | Phase/Affordance 头默认开启但无标签 | 建议 HeadsConfig 默认 phase_head=False, affordance_head=False |
| **L-21** | train_stage_a.py 遗留入口 | 建议删除 |
| **L-25/26** | TriRateMamba/Grounder 0 测试 | **HPC3 前应至少补 smoke test** |

---

## 五、优先级总结

### 第一优先级: 训练前必须完成

```
本地:  L-21 (删除 train_stage_a.py)
       L-15 (Phase/Affordance 头: 默认关闭 或 确认数据有标签)
       L-25/L-26 (TriRateMamba/Grounder 补最小 smoke test)

HPC3:  H-2 (500 步 smoke test)
       H-4 (2×GPU 100 步 smoke test)
       H-7 (fallback 路径 benchmark 确认)
```

### 第二优先级: 训练初期/并行修复

```
本地:  L-6 (FASTER 推理: 实现 或 确保 Stage C 推理 config 禁用)
       L-28 (multi-step rollout 测试)
       L-29 (EMA round-trip 测试)

HPC3:  H-3 (5K 步收敛验证)
       H-5 (15K 步 Stage A→B)
       H-12 (Consistency loss 有效性, 确认 VICReg fix 生效)
       H-13 (Phase/Affordance 在 LIBERO 上的实际情况)
```

### 第三优先级: 稳定后清理

```
本地:  L-23 (forward_train 拆分)
       L-24 (CI/CD)
       L-17 (FAST 头更名)
       剩余测试 (L-27~34)

HPC3:  H-14~16 (ablation 实验)
       H-18 (fork mamba_ssm)
       H-20/21 (监控/服务化)
```

---

## 六、关键数字速览

| 指标 | 当前值 | 修复前 | 参考 |
|------|--------|--------|------|
| 真实训练 run | **0 次** | 0 | >= 1 次 Stage A |
| Mamba 默认路径 | **fallback** (vectorized+checkpoint) | official (token-by-token) | — |
| 三阶段训练总时长 (fallback) | ~28-35 天估算 | ~55-64 天 (official) | ~20-30 天 (OpenPI) |
| ActionHistoryEncoder 参数 | **~1.7M** | 108M | 合理 |
| Consistency loss collapse 防护 | **VICReg variance + MSE** | 无 | 已修复 |
| 本地问题修复率 | **17/34 = 50%** | 0/34 | 100% |
| 死代码 | ~278 行 (train_stage_a.py) | ~1,400 行 (16%) | 0 |
| 测试覆盖 (核心模块) | TriRateMamba 0, Grounder 0 | 同 | 有专项测试 |
| CI/CD | 无 | 无 | pre-commit + GH Actions |
