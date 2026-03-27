# HybridVLA v2 Correctness Analysis — Post v0.7 Fix

This analysis is based on the code state after all `recovery_v0_7.md` fixes were applied.

---

## v0.7 Fix Verification

### Fix #1 & #2: MambaBlock Official Path Residual + Pre-Norm -- VERIFIED CORRECT

`mamba_core.py:236-247` (`_forward_official`) and `mamba_core.py:227-232` (`_step_official`) now both follow the pattern:

```
out = x + Mamba2(LayerNorm(x))
```

This matches the canonical usage in the official `mamba_ssm` codebase (`Block` class) and is consistent with the fallback path. The `self.norm = nn.LayerNorm(d_model)` is correctly created in both branches of `__init__`.

### Fix #3 & #4: Qwen2VL Freeze & LoRA Path -- VERIFIED CORRECT

`qwen2vl_backbone.py:110-134` now navigates through `getattr(self.model, "model", self.model)` and checks for both `.visual` and `.language_model` paths. Compatible with old and new transformers versions. The `_apply_lora` at line 142-146 uses the same navigation logic for layer count.

### Fix #5: Conv State Truncation -- VERIFIED CORRECT

`mamba_core.py:273-286` now takes `x_conv[:, :, (d_conv-1):(d_conv-1)+L]` when `conv_state` is prepended. This correctly skips the zero-padded region and extracts the L causally-correct output positions.

### Fix #6: init_states() Shape -- VERIFIED CORRECT

`mamba_core.py:340-384` creates Mamba-2-shaped states `(B, nheads, headdim, d_state)` for the official path and Mamba-1-shaped `(B, d_inner, d_state)` for fallback.

### Fix #7: Dead sigma_min -- VERIFIED CORRECT

`flow_matching.py` no longer has `sigma_min` parameter. The `interpolate()` static method signature is clean.

---

## Remaining Issues (Post v0.7) -- All Resolved in v0.7.1

### Issue 1: Stale Docstring in MambaBlock.step() -- FIXED (v0.7.1)

Docstring updated to document both Mamba-2 and Mamba-1 state shapes.

### Issue 2: Architecture Mismatch Remains By Design (KNOWN)

The fallback (Mamba-1) and official (Mamba-2) paths implement fundamentally different SSM architectures. This is acknowledged in the codebase and `recovery_v0_7.md`. Training and inference must use the same backend.

### Issue 3: Sinusoidal Embedding Ordering -- FIXED (v0.7.1)

`StaleTimeEncoding` and `SinusoidalTimestepEmbedding` now both use **[cos, sin]** order.

---

## Full Module-by-Module Status (v0.7)

### Qwen2-VL Backbone (`qwen2vl_backbone.py`) -- ALL CORRECT

| Component | Status | Notes |
|-----------|--------|-------|
| Model loading | CORRECT | `Qwen2VLForConditionalGeneration`, hidden_size=3584, 28 layers |
| Token IDs | CORRECT | IMAGE=151655, VIDEO=151656 |
| Vision tower freeze | CORRECT (v0.7) | Navigates `composite_model.visual` |
| Text layer freeze | CORRECT (v0.7) | Handles both `model.layers` and `model.language_model.layers` |
| LoRA application | CORRECT (v0.7) | Correct layer count via navigated text_model |
| Multi-scale extraction | CORRECT | `min()` clamping, layers [10, 18, 28] |
| MultiScaleAdapter | CORRECT | FPN-style gated fusion |

### Mamba Core (`mamba_core.py`) -- ALL CRITICAL FIXED

| Component | Status | Notes |
|-----------|--------|-------|
| `_forward_official` | CORRECT (v0.7) | `x + Mamba2(LN(x))` |
| `_step_official` | CORRECT (v0.7) | `x + Mamba2.step(LN(x))` |
| `_forward_fallback` | CORRECT (v0.7) | Conv state truncation fixed |
| `init_states()` | CORRECT (v0.7) | Backend-aware shapes |
| `TriRateMambaCore` | CORRECT | Tri-rate scheduling, cross-attention fusion |
| `CrossAttentionFusion` | CORRECT | 2-layer cross-attention with stale-time |
| `ActionHistoryEncoder` | CORRECT | 4-layer Mamba, last-token summary |
| `StaleTimeEncoding` | CORRECT | Sinusoidal + MLP |

### Selective Scan (`ops/selective_scan.py`) -- CORRECT

| Component | Status | Notes |
|-----------|--------|-------|
| `ssm_scan` (JIT) | CORRECT | Standard pre-discretized SSM recurrence |
| CUDA imports | CORRECT | Graceful fallback |

### Flow Action Expert (`flow_action_expert.py`) -- ALL CORRECT

| Component | Status | Notes |
|-----------|--------|-------|
| `ExpertMambaBlock` | CORRECT | Own pre-norm (AdaRMSNorm) + residual; unaffected by v0.7 |
| `selective_scan_fn` usage | CORRECT | Tensor layouts match official spec |
| `AdaRMSNorm` | CORRECT | pi-0.5 style `sigmoid(gate) * (RMSNorm(x) * (1+scale) + shift)` |
| `ExpertAttentionBlock` | CORRECT | Cross-attn + self-attn + FFN with AdaRMSNorm |
| Euler solver | CORRECT | Standard explicit Euler |
| Midpoint solver | CORRECT | Standard RK2 midpoint |

### Attention Grounder (`attention_grounder.py`) -- ALL CORRECT

| Component | Status | Notes |
|-----------|--------|-------|
| Perceiver architecture | CORRECT | 96 latents, 8 layers |
| Hierarchical compression | CORRECT | 48 -> 24 slots at layer 4 |
| Slot layout | CORRECT | [global, objects, phase, unc, aff, aux] |
| Cross/Self attention | CORRECT | F.scaled_dot_product_attention |

### Discrete Heads (`discrete_heads.py`) -- ALL CORRECT

| Component | Status | Notes |
|-----------|--------|-------|
| FASTDiscreteHead | CORRECT | 512-bin, factorized prediction |
| Discretization/undiscretization | CORRECT | Exact inverse mapping |
| PhaseHead / AffordanceHead | CORRECT | Simple MLP classifiers |

### Flow Matching Loss (`losses/flow_matching.py`) -- ALL CORRECT (v0.7)

| Component | Status | Notes |
|-----------|--------|-------|
| Velocity target | CORRECT | `x_1 - x_0` (Rectified Flow) |
| Logit-Normal sampling | CORRECT | `sigmoid(randn)` |
| Interpolation | CORRECT (v0.7) | `sigma_min` removed |

### Consistency Loss (`losses/consistency_loss.py`) -- ALL CORRECT

| Component | Status | Notes |
|-----------|--------|-------|
| ContrastiveTemporalLoss | CORRECT | InfoNCE on consecutive states |
| SlowFastAgreementLoss | CORRECT | EMA of fast vs slow |
| ActionConsistencyLoss | CORRECT | Projected cosine similarity |

### World Model -- ALL CORRECT

| Component | Status | Notes |
|-----------|--------|-------|
| StochasticStateModule | CORRECT | DreamerV3-style 48x48 categorical |
| ImaginationMamba | CORRECT | Uses `MambaBlock.step()` (now with v0.7 residual fix) |
| ObjectPhysicsEngine | CORRECT | 6-layer attention GNN |
| NoiseAugmentation | CORRECT | GameNGen-style linear schedule |
| WorldModelHeads | CORRECT | Symlog two-hot regression |
| KL Loss | CORRECT | Per-category free-bits (v0.4 fix) |

### Full Model Assembly (`hybrid_vla_v2.py`) -- ALL CORRECT

| Component | Status | Notes |
|-----------|--------|-------|
| Cond prefix (32 tokens) | CORRECT | 1+24+1+1+1+1+1+1+1 = 32 |
| Core-to-Expert projection | CORRECT | 2048d -> 1536d |
| Stage gating (A/B/C) | CORRECT | Progressive unfreezing |
| Training loop | CORRECT | Temporal iteration with refresh scheduling |
| Inference path | CORRECT | semantic_step + control_step |

---

## Summary

| Severity | Count (v0.6) | Count (v0.7) | Count (v0.7.1) | Status |
|----------|-------------|-------------|---------------|--------|
| Critical | 2 | **0** | **0** | All fixed |
| High | 2 | **0** | **0** | All fixed |
| Medium | 2 | **0** | **0** | All fixed |
| Low | 1 | **0** | **0** | Fixed |
| Cosmetic | 2 | 2 | **0** | Fixed in v0.7.1 |
| By Design | 0 | 1 | **1** | Mamba-1/Mamba-2 architecture mismatch (documented) |

**All issues resolved.** The only remaining item is the Mamba-1 vs Mamba-2 fallback architecture difference, which is by design.

---

## v0.7 正确性分析（中文版）

本分析基于 `recovery_v0_7.md` 中所有修复已应用后的代码状态。

---

## v0.7 修复验证

### 修复 #1 & #2: MambaBlock 官方路径残差 + 前置归一化 -- 验证正确

`mamba_core.py:236-247`（`_forward_official`）和 `mamba_core.py:227-232`（`_step_official`）现在均遵循：

```
out = x + Mamba2(LayerNorm(x))
```

与官方 `mamba_ssm` 代码库中 `Block` 类的规范用法一致，且与回退路径保持一致。`self.norm = nn.LayerNorm(d_model)` 在 `__init__` 的两个分支中均正确创建。

### 修复 #3 & #4: Qwen2VL 冻结与 LoRA 路径 -- 验证正确

`qwen2vl_backbone.py:110-134` 现通过 `getattr(self.model, "model", self.model)` 导航，并检查 `.visual` 和 `.language_model` 两种路径。兼容新旧版 transformers。`_apply_lora`（第 142-146 行）使用相同的导航逻辑获取层数。

### 修复 #5: Conv State 截断 -- 验证正确

`mamba_core.py:273-286` 当 `conv_state` 被拼接时，现在取 `x_conv[:, :, (d_conv-1):(d_conv-1)+L]`。正确跳过零填充区域，提取 L 个因果正确的输出位置。

### 修复 #6: init_states() 形状 -- 验证正确

`mamba_core.py:340-384` 为官方路径创建 Mamba-2 形状的状态 `(B, nheads, headdim, d_state)`，为回退路径创建 Mamba-1 形状 `(B, d_inner, d_state)`。

### 修复 #7: 无用的 sigma_min -- 验证正确

`flow_matching.py` 已移除 `sigma_min` 参数。`interpolate()` 静态方法签名简洁。

---

## 剩余问题（v0.7 之后）-- 全部在 v0.7.1 中解决

### 问题 1: MambaBlock.step() 文档字符串过时 -- 已修复 (v0.7.1)

文档已更新，现在同时记录 Mamba-2 和 Mamba-1 两种状态形状。

### 问题 2: 架构不匹配仍按设计保留（已知）

回退（Mamba-1）与官方（Mamba-2）路径实现的是根本不同的 SSM 架构。已在代码库和 `recovery_v0_7.md` 中说明。训练和推理必须使用相同后端。

### 问题 3: 正弦嵌入顺序 -- 已修复 (v0.7.1)

`StaleTimeEncoding` 和 `SinusoidalTimestepEmbedding` 现在均使用 **[cos, sin]** 顺序。

---

## 各模块完整状态（v0.7）

### Qwen2-VL Backbone (`qwen2vl_backbone.py`) -- 全部正确

| 组件 | 状态 | 说明 |
|------|------|------|
| 模型加载 | 正确 | `Qwen2VLForConditionalGeneration`, hidden_size=3584, 28层 |
| Token ID | 正确 | IMAGE=151655, VIDEO=151656 |
| 视觉塔冻结 | 正确 (v0.7) | 通过 `composite_model.visual` 导航 |
| 文本层冻结 | 正确 (v0.7) | 兼容 `model.layers` 和 `model.language_model.layers` |
| LoRA 应用 | 正确 (v0.7) | 通过导航后的 text_model 获取正确层数 |
| 多尺度特征提取 | 正确 | `min()` 截断, layers [10, 18, 28] |
| MultiScaleAdapter | 正确 | FPN 风格门控融合 |

### Mamba 核心 (`mamba_core.py`) -- 所有关键问题已修复

| 组件 | 状态 | 说明 |
|------|------|------|
| `_forward_official` | 正确 (v0.7) | `x + Mamba2(LN(x))` |
| `_step_official` | 正确 (v0.7) | `x + Mamba2.step(LN(x))` |
| `_forward_fallback` | 正确 (v0.7) | Conv state 截断已修复 |
| `init_states()` | 正确 (v0.7) | 感知后端的状态形状 |
| `TriRateMambaCore` | 正确 | 三速率调度、交叉注意力融合 |
| `CrossAttentionFusion` | 正确 | 2层交叉注意力 + 过时时间条件 |
| `ActionHistoryEncoder` | 正确 | 4层 Mamba，取最后 token |
| `StaleTimeEncoding` | 正确 | 正弦编码 + MLP |

### 选择性扫描 (`ops/selective_scan.py`) -- 正确

### Flow Action Expert (`flow_action_expert.py`) -- 全部正确

| 组件 | 状态 | 说明 |
|------|------|------|
| `ExpertMambaBlock` | 正确 | 自带 AdaRMSNorm + 残差；不受 v0.7 影响 |
| `selective_scan_fn` 调用 | 正确 | 张量布局符合官方规范 |
| `AdaRMSNorm` | 正确 | pi-0.5 风格自适应归一化 |
| ODE 求解器 | 正确 | Euler + Midpoint |

### Attention Grounder (`attention_grounder.py`) -- 全部正确

### 离散头 (`discrete_heads.py`) -- 全部正确

### Flow Matching 损失 (`losses/flow_matching.py`) -- 全部正确 (v0.7)

### 一致性损失 (`losses/consistency_loss.py`) -- 全部正确

### World Model -- 全部正确

| 组件 | 状态 | 说明 |
|------|------|------|
| StochasticStateModule | 正确 | DreamerV3 风格 48x48 类别分布 |
| ImaginationMamba | 正确 | 使用 `MambaBlock.step()`（v0.7 残差修复已生效） |
| ObjectPhysicsEngine | 正确 | 6层注意力 GNN |
| NoiseAugmentation | 正确 | GameNGen 风格线性噪声调度 |
| WorldModelHeads | 正确 | Symlog two-hot 回归 |
| KL Loss | 正确 | 每类别 free-bits（v0.4 修复） |

### 完整模型组装 (`hybrid_vla_v2.py`) -- 全部正确

| 组件 | 状态 | 说明 |
|------|------|------|
| 条件前缀 (32 tokens) | 正确 | 1+24+1+1+1+1+1+1+1 = 32 |
| 核心到专家投影 | 正确 | 2048d -> 1536d |
| 阶段门控 (A/B/C) | 正确 | 渐进式解冻 |
| 训练循环 | 正确 | 含刷新调度的时间迭代 |
| 推理路径 | 正确 | semantic_step + control_step |

---

## 总结

| 严重性 | v0.6 数量 | v0.7 数量 | v0.7.1 数量 | 状态 |
|--------|----------|----------|------------|------|
| 严重 | 2 | **0** | **0** | 全部修复 |
| 高危 | 2 | **0** | **0** | 全部修复 |
| 中危 | 2 | **0** | **0** | 全部修复 |
| 低危 | 1 | **0** | **0** | 已修复 |
| 外观问题 | 2 | 2 | **0** | v0.7.1 修复 |
| 按设计保留 | 0 | 1 | **1** | Mamba-1/Mamba-2 架构不匹配（已文档化） |

**所有问题均已解决。** 唯一剩余项为 Mamba-1 与 Mamba-2 回退架构差异，属于设计决策。
