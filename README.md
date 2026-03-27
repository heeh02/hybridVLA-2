# HybridVLA v2

**Hybrid Vision-Language-Action Model with Tri-Rate Temporal Reasoning for Robotic Manipulation**

HybridVLA v2 is a vision-language-action (VLA) architecture that addresses a fundamental tension in robotic control: reactive low-level motor commands need high-frequency updates (~50 Hz), while semantic understanding of the scene — "which object to pick," "where to place it" — changes slowly (~12.5 Hz). Existing VLA models either process everything at a single rate (wasting compute on redundant semantic re-encoding) or use a dual-rate design with a large frequency gap that loses mid-frequency dynamics.

HybridVLA v2 solves this with a **tri-rate Mamba temporal core** — three parallel state-space model streams operating at 50 Hz, 25 Hz, and 12.5 Hz, fused via cross-attention rather than scalar gating. Combined with a 7B vision-language backbone, a flow-matching action expert with multiplicative timestep conditioning, and a latent world model, the architecture targets 8xH100 80GB SXM clusters (~9B total parameters, ~1.5B trainable).

## Table of Contents

- [Design Philosophy](#design-philosophy)
- [Architecture](#architecture)
  - [Multi-Scale Vision-Language Backbone](#1-multi-scale-vision-language-backbone-qwen2-vl-7b)
  - [Hierarchical Attention Grounder](#2-hierarchical-attention-grounder)
  - [Tri-Rate Mamba Temporal Core](#3-tri-rate-mamba-temporal-core)
  - [Flow-Matching Action Expert](#4-flow-matching-action-expert-with-adarmsnorm)
  - [Hybrid Action Heads](#5-hybrid-action-heads)
  - [Latent World Model](#6-latent-world-model)
- [Training Strategy](#training-strategy)
- [Inference Pipeline](#inference-pipeline)
- [Parameter Budget](#parameter-budget)
- [Project Structure](#project-structure)
- [Hardware Requirements](#hardware-requirements)
- [Status](#status)
- [Collaboration](#collaboration)

## Design Philosophy

The design is driven by three observations:

1. **Temporal multi-scale nature of robotic control.** Compliant force control changes at 50+ Hz; object-relative trajectories evolve at ~25 Hz; task semantics shift at ~10 Hz. A single-rate model either over-computes semantics or under-samples motor signals. HybridVLA v2 assigns each frequency band its own Mamba stream with appropriate state capacity.

2. **Flow matching needs multiplicative conditioning.** At noise level t ~ 0, the denoiser must produce large corrections; at t ~ 1, only fine adjustments. Additive timestep embeddings (`x + t_emb`) shift but cannot rescale feature distributions. AdaRMSNorm's `gate * ((1 + scale) * x_norm + shift)` modulates the dynamic range per noise level — validated by the pi-0 line of work.

3. **Perception and action should train with gradient isolation.** The vision-language backbone encodes scene understanding; the action expert learns denoising dynamics. Allowing flow-matching gradients to backpropagate into the backbone causes catastrophic forgetting of visual representations. HybridVLA v2 uses a three-stage training strategy with explicit `stop_gradient` on the condition prefix during expert training.

## Architecture

```
                    Multi-Camera RGB (wrist, shoulder, overhead)
                              + Language Instruction
                                       |
                          Qwen2-VL-7B Backbone (frozen + LoRA)
                         Multi-scale features [L10, L18, L28]
                           Gated fusion -> [B, N, 2048]
                                       |
                     Hierarchical Attention Grounder (8 layers)
                       96 latents cross-attend to features
                     Layer 4: 48 object slots -> 24 compressed
                                       |
        +--------- Structured output tokens (32 total) ---------+
        | global(1), objects(24), phase(1), uncertainty(1),      |
        | affordance(1), fused(1), fast(1), medium(1), slow(1)   |
        +--------------------------------------------------------+
                                       |
                    +---------Tri-Rate Mamba Core---------+
                    |                  |                   |
             Fast (50Hz)        Medium (25Hz)       Slow (12.5Hz)
              20 layers           6 layers           10 layers
            d_state=128         d_state=128        d_state=256
           every step          every 2 steps      semantic refresh
                    |                  |                   |
                    +------ Cross-Attention Fusion --------+
                           (2 layers, 8 heads)
                    stale-time encoding conditions fusion
                                       |
                              Fused state [B, 2048]
                                       |
                    +------------------+------------------+
                    |                                     |
         Flow Action Expert (18L)                FAST Discrete Head
         M-M-A x 6, d=1536, AdaRMSNorm           512 bins, 14 DoF
         Midpoint ODE, chunk_horizon=24          + Phase/Affordance
                    |                                     |
          Continuous actions [B, 24, 14]       Discrete tokens [B, 24, 14]
                    |                                     |
                    +---- Consistency Loss alignment -----+
```

### 1. Multi-Scale Vision-Language Backbone (Qwen2-VL-7B)

The backbone extracts visual-linguistic features at three depths of the Qwen2-VL-7B transformer (3584d, 28 layers):

- **Layer 10** (early): fine-grained spatial features — edges, textures, grasp-point geometry
- **Layer 18** (mid): intermediate representations — object boundaries, part-whole relations
- **Layer 28** (final): high-level semantics — object identities, spatial relations, instruction grounding

A `MultiScaleAdapter` inspired by FPN projects each layer's output from 3584d to 2048d, then applies **learned per-scale gating** (not a fixed sum) to weight the contribution of each scale based on global context. This means the model can emphasize spatial detail for precision grasping or semantic features for instruction following, depending on the input.

**LoRA** (rank=64, alpha=128) is applied to all 28 layers' attention and MLP projections (q/k/v/o/gate/up/down), injecting ~90M trainable parameters into the 7.6B frozen backbone. This is a deliberate departure from v1's conservative approach of LoRA on only the last 8 layers — full-layer adaptation allows even early visual features to shift toward robotic manipulation distributions.

**Multi-camera** inputs (wrist, shoulder, overhead) are processed independently through the backbone and concatenated before the grounder, providing multi-view spatial reasoning without architectural changes.

### 2. Hierarchical Attention Grounder

A Perceiver-style module that converts variable-length backbone features into a fixed set of structured tokens. Uses 96 learned latent queries organized as:

```
[global(1), object_slots(48), phase(1), uncertainty(1), affordance(1), auxiliary(44)]
```

Processing proceeds in two phases across 8 cross-attention + self-attention layers:

- **Layers 0-3**: All 96 latents cross-attend to backbone features. Each object slot specializes to track a scene element. The auxiliary latents absorb context that doesn't fit the structured slots.
- **Layer 4 (compression)**: A `SlotCompression` module uses 24 learned routing queries to cross-attend to the 48 raw object slots, producing 24 compressed slots. This is a learned attention pooling — the model learns to merge similar objects (a row of cups -> "group of cups"), filter irrelevant background objects, and allocate higher-resolution slots to task-relevant objects.
- **Layers 4-7**: Continue processing with the compressed layout (72 latents total).

The output is a structured set of tokens: `global_token`, `compressed_object_slots` (24), `phase_token`, `uncertainty_token`, and `affordance_token`.

### 3. Tri-Rate Mamba Temporal Core

The central architectural contribution. Three parallel Mamba-2 SSM stacks process the same input tokens at different temporal frequencies:

| Stream | Frequency | Layers | d_state | Update Condition | Purpose |
|--------|-----------|--------|---------|------------------|---------|
| Fast | 50 Hz | 20 | 128 | Every control step | Reactive motor control — force, velocity, contact |
| Medium | 25 Hz | 6 | 128 | Every 2nd step | Mid-frequency dynamics — object displacement, trajectory curvature |
| Slow | 12.5 Hz | 10 | 256 | Semantic refresh only | Task planning — goal states, phase transitions |

**Why tri-rate instead of dual-rate?** In a dual-rate design (50 Hz / 12.5 Hz), the fast stream runs ~4 consecutive steps with no new semantic information. The medium stream (25 Hz) halves this gap, providing fresher context for dynamics that change faster than task semantics but slower than motor commands — such as an object being pushed across a surface.

**Cross-attention fusion** replaces v1's scalar sigmoid gate. A learned fusion query attends to the three stream output tokens, conditioned on a **stale-time encoding** (sinusoidal encoding of steps since last semantic refresh). This enables per-dimension, content-dependent weighting — the model can trust the fast stream for force-sensitive dimensions while relying on the slow stream for position targets, adapting the blend based on how stale the semantic information is.

**Action history encoder**: A 4-layer Mamba stack (d_state=64) processes the last K=8 executed actions into a summary token, providing short-term motor context to the temporal core.

### 4. Flow-Matching Action Expert with AdaRMSNorm

An 18-layer hybrid Mamba/Attention network (1536d) that denoises action chunks via Rectified Flow:

**Layer pattern**: `[Mamba, Mamba, Attention] x 6` — Mamba blocks handle sequential dependencies within the action chunk; cross-attention blocks attend to the 32-token condition prefix from the temporal core.

**AdaRMSNorm** (from pi-0): Every normalization layer in the expert is conditioned on the flow timestep t. Given a condition vector from `t_cond_mlp(timestep_embedding(t))`:
```
scale, shift, gate = linear(cond).chunk(3)
output = sigmoid(gate) * ((1 + scale) * RMSNorm(x) + shift)
```
The gate bias is initialized to +2 (sigmoid ~ 0.88) to prevent activation collapse through 18 residual layers. This multiplicative conditioning is critical — additive embeddings can only translate features, while AdaRMSNorm rescales the entire feature distribution at each noise level.

**Midpoint ODE solver** (2nd-order Runge-Kutta) for inference:
```
v1 = f(x_t, t)
v2 = f(x_t + 0.5*dt*v1, t + 0.5*dt)
x_{t+1} = x_t + dt * v2
```
8 midpoint steps achieve the accuracy of ~16 Euler steps with the same number of forward passes (8 * 2 = 16 * 1), but with O(dt^3) local truncation error vs O(dt^2).

**Condition prefix** (32 tokens fed to expert cross-attention):
```
global(1) + compressed_objects(24) + phase(1) + uncertainty(1) + affordance(1)
+ fused_state(1) + fast_token(1) + medium_token(1) + slow_token(1) = 32
```

### 5. Hybrid Action Heads

Two complementary action representations with a consistency loss:

- **Flow head** (continuous): Generates smooth 24-step action chunks (14 DoF) via flow matching. Precise but requires iterative denoising.
- **FAST head** (discrete): Tokenizes each action dimension into 512 bins, enabling single-pass autoregressive inference. Fast but quantized.

A `V2ConsistencyLoss` ensures the two heads agree, combining:
- **ContrastiveTemporalLoss**: InfoNCE on consecutive fused states — learns meaningful temporal structure rather than just penalizing magnitude changes.
- **SlowFastAgreementLoss**: Constrains the slow stream output to approximate an exponential moving average of recent fast outputs, preventing stream divergence.
- **ActionConsistencyLoss**: Projects discrete and continuous action predictions into a shared embedding space and maximizes cosine similarity.

Additional heads:
- **Phase head** (16 classes): Predicts the current manipulation phase (approach, grasp, lift, transport, place, etc.)
- **Affordance head**: Predicts object affordance types relevant to the current task

### 6. Latent World Model

An optional DreamerV3-inspired imagination engine that predicts future latent states, enabling the agent to evaluate action consequences before execution:

- **StochasticStateModule**: Categorical latent variables (48 categories x 48 classes) with separate prior/posterior encoding, enabling multi-modal state predictions
- **ImaginationMamba**: 8-layer Mamba stack (d_state=128) for latent dynamics transitions with explicit state persistence across imagination steps
- **ObjectPhysicsEngine**: GNN-based (6 layers) object interaction modeling with intrinsic property prediction and interaction weight estimation
- **NoiseAugmentation**: Schedule-aware noise injection during imagination rollouts to prevent compounding errors
- **CNNWorldDecoder**: Optional visual prediction for auxiliary supervision
- **LatentSubgoalPlanner**: Proposes intermediate latent goals for long-horizon tasks

The imagination engine performs 32-step rollouts, collecting full loss data (reward/value logits, prior distributions, physics outputs, visual predictions) for end-to-end training via `WorldModelLoss` with free-bits KL and categorical cross-entropy.

## Training Strategy

Three-stage progressive training with gradient isolation:

| Stage | Steps | LR | Trainable Components | Key Design Choice |
|-------|-------|-----|---------------------|-------------------|
| **A** | 120K | 2e-4 | Backbone LoRA + Grounder + Tri-Rate Core + Discrete Heads | Expert frozen. Perception learns scene structure without denoising interference. |
| **B** | 200K | 1e-4 | + Flow Expert + EMA starts | `cond_prefix.detach()` — expert trains with frozen condition representations, preventing flow-matching gradients from destabilizing the backbone. |
| **C** | 80K | 3e-5 | All components + RTC + FASTER | End-to-end fine-tuning with receding-horizon temporal consistency (RTC) and near/far step scheduling (FASTER). Lower LR preserves learned representations. |

**Global batch size = 64** (2 per GPU x 8 GPUs x 4 gradient accumulation steps). **EMA** with decay ramping from 0.999 to 0.9999 over 20K steps, starting at Stage B.

## Inference Pipeline

The runtime operates at two frequencies:

- **Semantic loop (~12.5 Hz)**: Runs the backbone + grounder to produce a `GrounderOutput`. Triggered by a refresh schedule or scene change detection.
- **Control loop (50 Hz)**: Runs the temporal core (fast stream every step, medium every 2, slow reuses cached token) and generates action chunks via the flow expert.

**Action chunk caching**: A 24-step chunk is generated once and executed over 8 steps (execution horizon). A new chunk is generated only when: (a) the cache is exhausted, (b) a semantic refresh occurs (invalidating the current plan), or (c) the cache is uninitialized. This avoids redundant expert forward passes during steady-state execution.

## Parameter Budget

| Component | Parameters | Trainable | Notes |
|-----------|-----------|-----------|-------|
| Qwen2-VL-7B Backbone | 7,600M | 90M (LoRA) | rank=64, all 28 layers |
| Attention Grounder | ~200M | 200M | 96 latents, 8 layers, 2048d |
| Fast Mamba (20L) | ~330M | 330M | d_state=128 |
| Medium Mamba (6L) | ~100M | 100M | d_state=128 |
| Slow Mamba (10L) | ~200M | 200M | d_state=256 |
| Cross-Attention Fusion | ~35M | 35M | 2 layers, 8 heads |
| Action History Encoder | ~65M | 65M | 4L Mamba, d_state=64 |
| Flow Action Expert | ~350M | 350M | 18L, 1536d, M-M-A x 6 |
| Discrete Heads | ~12M | 12M | FAST 512-bin + Phase + Affordance |
| Projections / Embeddings | ~60M | 60M | Core-to-expert, stale-time, etc. |
| **Total** | **~9.0B** | **~1.5B** | |

Per-GPU memory (FSDP full-shard, bs=2): ~50-55 GB / 80 GB, leaving ~25 GB headroom.

## Project Structure

```
hybridVLA_2/
  configs/
    model/                     # Architecture configs (YAML)
    train/                     # Per-stage training configs (stage_a/b/c.yaml)
  scripts/
    train_stage_a.py           # Training entry point
    train_smoke_test.py        # Config + forward pass validation
    compute_stats.py           # Dataset normalization statistics
  vla_hybrid_v2/
    __init__.py                # Package root (v2.0.0)
    config.py                  # Structured dataclass configs with YAML loader
    types.py                   # Core types: GrounderOutput, TriRateTemporalState, etc.
    models/
      hybrid_vla_v2.py         # Top-level model: forward_train + control_step
      qwen2vl_backbone.py      # 7B backbone + MultiScaleAdapter + multi-camera
      attention_grounder.py    # Hierarchical 96-latent grounder + SlotCompression
      mamba_core.py            # Tri-Rate Mamba (Fast/Medium/Slow) + CrossAttentionFusion
      flow_action_expert.py    # 18L AdaRMSNorm expert + Euler/Midpoint samplers
      discrete_heads.py        # FAST 512-bin, Phase 16-class, Affordance heads
    losses/
      flow_matching.py         # Rectified Flow MSE loss + logit-normal timestep sampling
      discrete_loss.py         # Cross-entropy with label smoothing
      consistency_loss.py      # Contrastive temporal + slow-fast agreement + action consistency
    world_model/
      imagination_engine.py    # 32-step imagination rollout orchestrator
      imagination_mamba.py     # 8L Mamba for latent dynamics
      stochastic_state.py      # Categorical latent variables (prior/posterior)
      object_physics.py        # GNN-based object interaction modeling
      noise_augmentation.py    # Schedule-aware noise injection
      visual_decoder.py        # CNN decoder for visual predictions
      subgoal_planner.py       # Latent subgoal proposal
      world_model_heads.py     # Reward/value/done prediction heads
      world_model_loss.py      # Combined world model loss
    ops/
      selective_scan.py        # JIT-compiled SSM scan + CUDA dispatch
    data/
      schema.py                # Episode schema + validation
      base_adapter.py          # Abstract dataset adapter
      hdf5_adapter.py          # HDF5 episode loading
      normalizer.py            # Per-field running statistics
      collate.py               # Multi-camera batch collation
      dummy.py                 # Synthetic data for smoke tests
    infer/                     # Runtime inference loop (WIP)
    utils/
      checkpointing.py         # Save/load with stage awareness
      distributed.py           # FSDP helpers
      ema.py                   # Exponential moving average with decay ramp
  docs/                        # Design documents and iteration history (v0.1 - v0.10.2)
```

## Hardware Requirements

- **Training**: 8x NVIDIA H100 80GB SXM (FSDP full-shard, activation checkpointing, bf16)
- **Inference**: Single H100 or A100 (bf16), ~50 Hz control loop with chunk caching

## Status

The model architecture (v0.10.2) has been through 10+ iterations of cross-audits (Claude + GPT) and is considered mature. Current development priorities:

- Data pipeline integration with real robot datasets (LIBERO, Calvin, DROID)
- Evaluation framework and benchmark suite
- Runtime inference optimization and `torch.compile` integration
- World model training loop

## Collaboration

This project is under active development. If you are interested in the architecture, have access to robot hardware for evaluation, or would like to contribute, please reach out:

- **GitHub Issues**: Open an issue in this repository
- **Email**: [jhe000@connect.hkust-gz.edu.cn](mailto:jhe000@connect.hkust-gz.edu.cn)

Contributions, discussions, and feedback are welcome.

## License

This project is currently unlicensed. Please contact the author before using any part of this codebase.

---

# HybridVLA v2 (中文)

**混合视觉-语言-动作模型：基于三频时序推理的机器人操作**

HybridVLA v2 是一个视觉-语言-动作 (VLA) 架构，旨在解决机器人控制中的一个根本矛盾：底层运动控制需要高频更新（~50 Hz），而场景的语义理解——"抓哪个物体""放到哪里"——变化缓慢（~12.5 Hz）。现有 VLA 模型要么以单一频率处理所有信息（在冗余的语义重编码上浪费算力），要么使用双频设计但频率间隔过大，丢失了中频动态。

HybridVLA v2 通过**三频 Mamba 时序核心**解决这一问题——三条并行的状态空间模型流分别以 50 Hz、25 Hz 和 12.5 Hz 运行，并通过交叉注意力而非标量门控进行融合。结合 7B 视觉语言骨干、乘法时间步条件化的 Flow Matching 动作专家，以及潜在世界模型，整体架构面向 8×H100 80GB SXM 集群（总参数约 9B，可训练约 1.5B）。

## 目录

- [设计思想](#设计思想)
- [架构详解](#架构详解)
  - [多尺度视觉语言骨干](#1-多尺度视觉语言骨干-qwen2-vl-7b)
  - [层次注意力 Grounder](#2-层次注意力-grounder)
  - [三频 Mamba 时序核心](#3-三频-mamba-时序核心)
  - [Flow Matching 动作专家](#4-adarmsnorm-flow-matching-动作专家)
  - [混合动作头](#5-混合动作头)
  - [潜在世界模型](#6-潜在世界模型)
- [训练策略](#训练策略)
- [推理管线](#推理管线)
- [参数预算](#参数预算)
- [项目结构](#项目结构-1)

## 设计思想

设计基于三个核心观察：

1. **机器人控制具有时间多尺度特性。** 柔顺力控制在 50+ Hz 变化；物体相对轨迹在 ~25 Hz 演变；任务语义在 ~10 Hz 切换。单一频率的模型要么过度计算语义信息，要么欠采样运动信号。HybridVLA v2 为每个频段分配独立的 Mamba 流，配备适当的状态容量。

2. **Flow Matching 需要乘法条件化。** 在噪声水平 t ~ 0 时，去噪器需要大幅度修正；在 t ~ 1 时，只需微调。加法时间步嵌入（`x + t_emb`）只能平移特征分布，无法改变其动态范围。AdaRMSNorm 的 `gate * ((1 + scale) * x_norm + shift)` 在每个噪声水平对特征分布进行乘法调制——这一点已被 pi-0 系列工作验证。

3. **感知与动作应当梯度隔离。** 视觉语言骨干编码场景理解；动作专家学习去噪动力学。如果允许 Flow Matching 梯度回传到骨干网络，会导致视觉表征的灾难性遗忘。HybridVLA v2 采用三阶段训练策略，在专家训练期间对条件前缀执行显式 `stop_gradient`。

## 架构详解

```
                    多相机 RGB（腕部、肩部、俯视）
                         + 语言指令
                              |
                     Qwen2-VL-7B 骨干（冻结 + LoRA）
                    多尺度特征 [L10, L18, L28]
                      门控融合 -> [B, N, 2048]
                              |
                层次注意力 Grounder（8 层）
                  96 个隐变量交叉注意力到特征
                第 4 层：48 物体槽位 -> 24 压缩槽位
                              |
       +-------- 结构化输出 token（共 32 个）--------+
       | global(1), objects(24), phase(1),            |
       | uncertainty(1), affordance(1), fused(1),     |
       | fast(1), medium(1), slow(1)                  |
       +----------------------------------------------+
                              |
                 +----三频 Mamba 核心----+
                 |          |           |
           快速(50Hz)  中速(25Hz)  慢速(12.5Hz)
            20 层        6 层        10 层
          d_state=128  d_state=128  d_state=256
           每步运行    每 2 步      语义刷新时
                 |          |           |
                 +-- 交叉注意力融合 -----+
                    (2 层, 8 头)
               陈旧时间编码调节融合权重
                         |
                   融合状态 [B, 2048]
                         |
               +---------+---------+
               |                   |
     Flow 动作专家 (18L)      FAST 离散头
     M-M-A x 6, d=1536       512 bins, 14 DoF
     AdaRMSNorm, 中点ODE     + 阶段/Affordance
               |                   |
     连续动作 [B, 24, 14]    离散 token [B, 24, 14]
               |                   |
               +-- 一致性损失对齐 --+
```

### 1. 多尺度视觉语言骨干 (Qwen2-VL-7B)

骨干从 Qwen2-VL-7B 变换器（3584d，28 层）的三个深度提取视觉-语言特征：

- **第 10 层**（早期）：细粒度空间特征——边缘、纹理、抓取点几何
- **第 18 层**（中期）：中间表征——物体边界、部分-整体关系
- **第 28 层**（末期）：高层语义——物体身份、空间关系、指令落地

`MultiScaleAdapter` 受 FPN 启发，将每层输出从 3584d 投影到 2048d，然后施加**学习的逐尺度门控**（而非固定求和），根据全局上下文加权各尺度的贡献。这意味着模型可以在精确抓取时强调空间细节，在指令执行时侧重语义特征。

**LoRA**（rank=64, alpha=128）应用于全部 28 层的注意力和 MLP 投影（q/k/v/o/gate/up/down），在 7.6B 冻结骨干中注入约 90M 可训练参数。与 v1 仅在最后 8 层使用 LoRA 的保守策略不同，全层适应使早期视觉特征也能向机器人操作数据分布偏移。

**多相机**输入（腕部、肩部、俯视）独立通过骨干处理并在 Grounder 前拼接。

### 2. 层次注意力 Grounder

Perceiver 风格的模块，将变长骨干特征转换为固定数量的结构化 token。使用 96 个学习的隐变量查询，组织为：

```
[全局(1), 物体槽位(48), 阶段(1), 不确定性(1), affordance(1), 辅助(44)]
```

处理分两阶段，贯穿 8 层交叉注意力 + 自注意力：

- **第 0-3 层**：全部 96 个隐变量交叉注意力到骨干特征。每个物体槽位专门化以跟踪场景中的一个元素。
- **第 4 层（压缩）**：`SlotCompression` 模块用 24 个学习的路由查询交叉注意力到 48 个原始物体槽位，产生 24 个压缩槽位。这是一种学习的注意力池化——模型学会合并相似物体（一排杯子 -> "一组杯子"）、过滤无关背景物体、并为任务相关物体分配更高分辨率的槽位。
- **第 4-7 层**：以压缩后的布局（共 72 个隐变量）继续处理。

### 3. 三频 Mamba 时序核心

核心架构贡献。三条并行的 Mamba-2 SSM 栈以不同时间频率处理相同的输入 token：

| 流 | 频率 | 层数 | d_state | 更新条件 | 职责 |
|---|------|------|---------|---------|------|
| 快速 | 50 Hz | 20 | 128 | 每个控制步 | 反应式运动控制——力、速度、接触 |
| 中速 | 25 Hz | 6 | 128 | 每 2 步 | 中频动态——物体位移、轨迹曲率 |
| 慢速 | 12.5 Hz | 10 | 256 | 仅语义刷新时 | 任务规划——目标状态、阶段转换 |

**为何三频而非双频？** 在双频设计（50 Hz / 12.5 Hz）中，快速流连续运行约 4 步而没有新的语义信息注入。中速流（25 Hz）将这一间隔减半，为那些变化速度介于运动指令和任务语义之间的动态（如物体被推过桌面）提供更新鲜的上下文。

**交叉注意力融合**取代了 v1 的标量 sigmoid 门控。一个学习的融合查询注意力到三个流的输出 token，并以**陈旧时间编码**（自上次语义刷新以来的步数的正弦编码）为条件。这实现了逐维度、依赖内容的加权——模型可以在力敏感维度信任快速流，在位置目标上依赖慢速流，并根据语义信息的"陈旧程度"自适应调整融合比例。

**动作历史编码器**：4 层 Mamba 栈（d_state=64）处理最近 K=8 个已执行动作为一个摘要 token，为时序核心提供短期运动上下文。

### 4. AdaRMSNorm Flow Matching 动作专家

18 层混合 Mamba/Attention 网络（1536d），通过 Rectified Flow 对动作块去噪：

**层模式**：`[Mamba, Mamba, Attention] x 6` —— Mamba 块处理动作块内的序列依赖；交叉注意力块注意力到来自时序核心的 32-token 条件前缀。

**AdaRMSNorm**（源自 pi-0）：专家中的每个归一化层都以 flow 时间步 t 为条件：
```
scale, shift, gate = linear(cond).chunk(3)
output = sigmoid(gate) * ((1 + scale) * RMSNorm(x) + shift)
```
门控偏置初始化为 +2（sigmoid ~ 0.88），防止通过 18 个残差层的激活坍缩。乘法条件化至关重要——加法嵌入只能平移特征，而 AdaRMSNorm 在每个噪声水平重新缩放整个特征分布的动态范围。

**中点 ODE 求解器**（2 阶 Runge-Kutta）用于推理：8 步中点法达到约 16 步 Euler 法的精度，前向传播次数相同（8×2 = 16×1），但局部截断误差为 O(dt^3) vs O(dt^2)。

### 5. 混合动作头

两种互补的动作表示，通过一致性损失保持一致：

- **Flow 头**（连续）：通过 Flow Matching 生成平滑的 24 步动作块（14 DoF）。精确但需迭代去噪。
- **FAST 头**（离散）：将每个动作维度 token 化为 512 bins，实现单次自回归推理。快速但有量化误差。

`V2ConsistencyLoss` 包含三项：
- **对比时序损失**：对连续融合状态施加 InfoNCE——学习有意义的时序结构。
- **慢-快一致性损失**：约束慢速流输出近似近期快速流输出的指数移动平均，防止流间发散。
- **动作一致性损失**：将离散和连续动作预测投影到共享嵌入空间，最大化余弦相似度。

### 6. 潜在世界模型

可选的 DreamerV3 风格想象引擎，预测未来潜在状态，使智能体在执行前评估动作后果：

- **随机状态模块**：类别潜变量（48 类别 × 48 类），分离的先验/后验编码
- **想象 Mamba**：8 层 Mamba（d_state=128）进行潜在动力学转移，步间显式状态持久化
- **物体物理引擎**：基于 GNN 的物体交互建模（6 层），预测内在属性和交互权重
- **噪声增强**：调度感知的噪声注入，防止想象 rollout 中的误差累积
- **视觉解码器**：CNN 解码器用于辅助视觉监督
- **子目标规划器**：为长时任务提议中间潜在目标

想象引擎执行 32 步 rollout，收集完整的损失数据用于端到端训练。

## 训练策略

三阶段渐进训练，带梯度隔离：

| 阶段 | 步数 | 学习率 | 可训练组件 | 关键设计选择 |
|------|------|--------|-----------|------------|
| **A** | 120K | 2e-4 | 骨干 LoRA + Grounder + 三频核心 + 离散头 | 专家冻结。感知模块学习场景结构，不受去噪干扰。 |
| **B** | 200K | 1e-4 | + Flow 专家 + EMA 启动 | `cond_prefix.detach()` —— 专家以冻结的条件表征训练，防止 flow matching 梯度破坏骨干稳定性。 |
| **C** | 80K | 3e-5 | 全部组件 + RTC + FASTER | 端到端微调。更低学习率保护已学表征。 |

**全局 batch size = 64**（每 GPU 2 × 8 GPU × 4 梯度累积）。**EMA** 衰减从 0.999 渐进到 0.9999，经过 20K 步，从 Stage B 开始。

## 推理管线

运行时以两个频率运行：

- **语义循环（~12.5 Hz）**：运行骨干 + Grounder 产生 `GrounderOutput`。
- **控制循环（50 Hz）**：运行时序核心（快速流每步、中速流每 2 步、慢速流复用缓存 token）并通过 Flow 专家生成动作块。

**动作块缓存**：一个 24 步的动作块生成一次，执行 8 步（执行视野）。仅在缓存耗尽、语义刷新发生或缓存未初始化时才生成新块，避免稳态执行中的冗余专家前向传播。

## 参数预算

| 组件 | 参数量 | 可训练 | 说明 |
|------|--------|--------|------|
| Qwen2-VL-7B 骨干 | 7,600M | 90M (LoRA) | rank=64，全部 28 层 |
| 注意力 Grounder | ~200M | 200M | 96 隐变量，8 层，2048d |
| 快速 Mamba (20L) | ~330M | 330M | d_state=128 |
| 中速 Mamba (6L) | ~100M | 100M | d_state=128 |
| 慢速 Mamba (10L) | ~200M | 200M | d_state=256 |
| 交叉注意力融合 | ~35M | 35M | 2 层，8 头 |
| 动作历史编码器 | ~65M | 65M | 4L Mamba, d_state=64 |
| Flow 动作专家 | ~350M | 350M | 18L, 1536d, M-M-A x 6 |
| 离散头 | ~12M | 12M | FAST 512-bin + 阶段 + Affordance |
| 投影 / 嵌入 | ~60M | 60M | Core-to-expert, 陈旧时间编码等 |
| **总计** | **~9.0B** | **~1.5B** | |

每 GPU 显存（FSDP 全分片，bs=2）：约 50-55 GB / 80 GB，剩余约 25 GB 余量。

## 合作

本项目正在积极开发中。如果你对该架构感兴趣、拥有可用于评估的机器人硬件、或希望参与贡献，请通过以下方式联系：

- **GitHub Issues**：在本仓库提 Issue
- **邮箱**：[jhe000@connect.hkust-gz.edu.cn](mailto:jhe000@connect.hkust-gz.edu.cn)

欢迎贡献代码、参与讨论和提供反馈。

## 许可证

本项目目前未设定许可证。使用本代码库的任何部分前，请先联系作者。
