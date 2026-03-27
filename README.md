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
