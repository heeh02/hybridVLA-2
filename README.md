# HybridVLA v2

**A Hybrid Vision-Language-Action Model with Tri-Rate Temporal Reasoning for Robotic Manipulation**

HybridVLA v2 is an aggressive architectural upgrade over v1, designed to fully exploit 8xH100 80GB SXM clusters. It combines a 7B vision-language backbone with a novel tri-rate Mamba temporal core and a flow-matching action expert, targeting high-precision robotic manipulation across diverse environments.

## Key Design Ideas

### 1. Strong Vision-Language Backbone (Qwen2-VL-7B)

We use **Qwen2-VL-7B** as the vision-language backbone (3.5x larger than v1's 2B), with multi-scale feature extraction from layers [10, 18, 28] to capture both fine-grained spatial details (edges, grasp points) and high-level semantics. LoRA (rank-64) is applied across all layers for efficient domain adaptation.

### 2. Attention Grounder with Hierarchical Compression

A Perceiver-style grounder with **96 latents** and **8 layers** converts raw visual tokens into compact object-centric representations. Hierarchical compression at layer 4 reduces 96 latents to 24 compressed slots, balancing representational richness with computational efficiency.

### 3. Tri-Rate Mamba Temporal Core

The core architectural innovation: three parallel Mamba SSM streams operating at different temporal frequencies:

- **Fast stream** (50 Hz, 20 layers): Reactive low-level control
- **Medium stream** (25 Hz, 6 layers): Bridging frequency gap between fast and slow
- **Slow stream** (12.5 Hz, 10 layers): Long-horizon planning with d_state=256

Streams are fused via **cross-attention** (2 layers, 8 heads) instead of simple scalar gates, enabling fine-grained per-dimension temporal reasoning.

### 4. Flow-Matching Action Expert

An **18-layer, 1536d** action expert with **AdaRMSNorm** timestep conditioning generates continuous action chunks (horizon=24) via flow matching. Uses **midpoint ODE solver** for higher-quality action generation with fewer denoising steps.

### 5. Hybrid Action Representation

Dual action heads for complementary strengths:
- **Flow head**: Continuous actions via flow matching for smooth, precise trajectories
- **FAST head**: Discrete tokenized actions (512 bins) for fast autoregressive inference

### 6. World Model with Imagination Engine

A latent world model predicts future visual states, enabling the agent to "imagine" action consequences before execution. Includes stochastic state modeling, noise augmentation, and a subgoal planner for long-horizon tasks.

### 7. Multi-Camera Support

Native support for **3 cameras** (wrist, shoulder, overhead), providing multi-view spatial reasoning critical for complex manipulation tasks.

## Architecture Overview

```
Multi-Camera Images + Language Instruction
              |
    Qwen2-VL-7B Backbone (multi-scale features)
              |
    Attention Grounder (96 -> 24 compressed slots)
              |
    Tri-Rate Mamba Temporal Core
    [Fast 50Hz | Medium 25Hz | Slow 12.5Hz]
              |  (cross-attention fusion)
              |
     +--------+--------+
     |                  |
Flow Action Expert   FAST Head (discrete)
  (18L, 1536d)       (512 bins)
     |                  |
Continuous Actions  Discrete Actions
```

## Project Structure

```
hybridVLA_2/
  configs/           # Model and training configurations
  docs/              # Design documents and iteration analysis
  scripts/           # Training and evaluation scripts
  vla_hybrid_v2/     # Core library
    models/          # Backbone, grounder, expert, Mamba core
    losses/          # Flow matching, discrete, consistency losses
    data/            # Data pipeline (HDF5 adapter, normalizer, collate)
    world_model/     # Imagination engine, subgoal planner
    ops/             # Custom ops (selective scan)
    utils/           # Checkpointing, distributed, EMA
```

## Hardware Requirements

- **Training**: 8x H100 80GB SXM (FSDP full-shard, ~55 GB/GPU)
- **Inference**: Single H100 or A100 (bf16)

## Status

This project is under active development. The model architecture (v0.10.2) has been through multiple rounds of cross-audits and is considered mature. Current focus areas:

- Data pipeline integration with real robot datasets
- Evaluation framework
- Runtime inference optimization

## Collaboration

If you are interested in this project or would like to collaborate, feel free to reach out:

- **GitHub Issues**: Open an issue in this repository
- **Email**: [jhe000@connect.hkust-gz.edu.cn](mailto:jhe000@connect.hkust-gz.edu.cn)

Contributions, discussions, and feedback are all welcome!

## License

This project is currently unlicensed. Please contact the author before using any part of this codebase.

---

# HybridVLA v2 (中文)

**混合视觉-语言-动作模型：基于三频时序推理的机器人操作**

HybridVLA v2 是对 v1 的大幅架构升级，旨在充分利用 8×H100 80GB SXM 集群的算力。它将 7B 视觉语言骨干网络与创新的三频 Mamba 时序核心和 Flow Matching 动作专家相结合，面向多样化环境中的高精度机器人操作。

## 核心设计思想

### 1. 强大的视觉语言骨干网络 (Qwen2-VL-7B)

采用 **Qwen2-VL-7B** 作为视觉语言骨干（参数量为 v1 的 3.5 倍），从第 [10, 18, 28] 层提取多尺度特征，既捕获细粒度空间细节（边缘、抓取点），也捕获高层语义。全层应用 LoRA（rank-64）实现高效的领域自适应。

### 2. 层次压缩注意力 Grounder

Perceiver 风格的 Grounder 使用 **96 个隐变量**和 **8 层**网络，将原始视觉 token 转换为紧凑的以物体为中心的表征。在第 4 层进行层次压缩，将 96 个隐变量压缩为 24 个，在表征丰富度与计算效率之间取得平衡。

### 3. 三频 Mamba 时序核心

核心架构创新——三条并行的 Mamba SSM 流在不同时间频率上运行：

- **快速流**（50 Hz，20 层）：反应式底层控制
- **中速流**（25 Hz，6 层）：弥合快慢流之间的频率鸿沟
- **慢速流**（12.5 Hz，10 层）：长时规划，d_state=256

各流通过**交叉注意力**（2 层，8 头）融合，而非简单的标量门控，实现逐维度的细粒度时序推理。

### 4. Flow Matching 动作专家

**18 层、1536 维**的动作专家，采用 **AdaRMSNorm** 时间步条件化，通过 Flow Matching 生成连续动作块（horizon=24）。使用**中点 ODE 求解器**，以更少的去噪步数获得更高质量的动作生成。

### 5. 混合动作表示

双动作头，优势互补：
- **Flow 头**：通过 Flow Matching 生成连续动作，轨迹平滑精确
- **FAST 头**：离散 token 化动作（512 bins），快速自回归推理

### 6. 世界模型与想象引擎

潜在世界模型预测未来视觉状态，使智能体在执行前能"想象"动作后果。包括随机状态建模、噪声增强和子目标规划器，支持长时任务规划。

### 7. 多相机支持

原生支持 **3 个相机**（腕部、肩部、俯视），提供多视角空间推理，对复杂操作任务至关重要。

## 架构总览

```
多相机图像 + 语言指令
         |
  Qwen2-VL-7B 骨干网络（多尺度特征）
         |
  注意力 Grounder（96 → 24 压缩槽位）
         |
  三频 Mamba 时序核心
  [快速 50Hz | 中速 25Hz | 慢速 12.5Hz]
         |  （交叉注意力融合）
         |
    +----+----+
    |              |
Flow 动作专家   FAST 头（离散）
 (18L, 1536d)   (512 bins)
    |              |
 连续动作       离散动作
```

## 项目结构

```
hybridVLA_2/
  configs/           # 模型与训练配置
  docs/              # 设计文档与迭代分析
  scripts/           # 训练与评估脚本
  vla_hybrid_v2/     # 核心库
    models/          # 骨干网络、Grounder、专家、Mamba 核心
    losses/          # Flow Matching、离散、一致性损失
    data/            # 数据管线（HDF5 适配器、归一化器、整理器）
    world_model/     # 想象引擎、子目标规划器
    ops/             # 自定义算子（选择性扫描）
    utils/           # 检查点、分布式、EMA
```

## 硬件需求

- **训练**：8× H100 80GB SXM（FSDP 全分片，约 55 GB/GPU）
- **推理**：单张 H100 或 A100（bf16）

## 项目状态

本项目正在积极开发中。模型架构（v0.10.2）已经过多轮交叉审计，趋于成熟。当前重点方向：

- 真实机器人数据集的数据管线集成
- 评估框架搭建
- 运行时推理优化

## 合作

如果您对本项目感兴趣或希望合作，欢迎联系：

- **GitHub Issues**：在本仓库提 Issue
- **邮箱**：[jhe000@connect.hkust-gz.edu.cn](mailto:jhe000@connect.hkust-gz.edu.cn)

欢迎贡献代码、参与讨论和提供反馈！

## 许可证

本项目目前未设定许可证。使用本代码库的任何部分前，请先联系作者。
