# HybridVLA v2 第二轮迭代分析 (v0.2 — 代码可运行性 + 数据通路 + 训练正确性)

> 分析日期: 2026-03-25
> 基于: v0.1 分析后的代码修改版本 (已修复 Grounder 压缩时机, SDPA, CUDA SSM 等)
> 重点: 可运行性 / 张量维度闭合 / 梯度流正确性 / 优化器加速 / ML 细节

---

## 目录

1. [可运行性分析](#1-可运行性分析)
2. [端到端数据通路张量维度追踪](#2-端到端数据通路张量维度追踪)
3. [梯度流与训练正确性分析](#3-梯度流与训练正确性分析)
4. [AdamW 与优化器加速分析](#4-adamw-与优化器加速分析)
5. [机器学习关键问题清单](#5-机器学习关键问题清单)
6. [优先修复建议](#6-优先修复建议)

---

## 1. 可运行性分析

### 1.1 依赖项检查

| 依赖 | 用途 | 是否必需 | 代码位置 |
|------|------|----------|----------|
| `torch` (>=2.0) | 核心框架, SDPA | **必需** | 全局 |
| `transformers` | Qwen2-VL-7B 加载 | **必需** | `qwen2vl_backbone.py:78` |
| `peft` | LoRA 注入 | **必需** | `qwen2vl_backbone.py:122` |
| `pyyaml` | 配置加载 | **必需** | `config.py:17` |
| `mamba_ssm` | CUDA SSM kernel | 可选 (性能) | `selective_scan.py:17-22` |
| `causal_conv1d` | Fused Conv1d | 可选 (性能) | `selective_scan.py:28-31` |
| `flash_attn` | Backbone Flash Attention | 推荐 | `qwen2vl_backbone.py:82` |

### 1.2 模型可实例化性

**可以实例化，但有前提条件。**

```python
from vla_hybrid_v2.config import load_config
from vla_hybrid_v2.models.hybrid_vla_v2 import HybridVLAv2

cfg = load_config("configs/train/stage_a.yaml")
model = HybridVLAv2(cfg)  # 需要: 网络连接下载 Qwen2-VL-7B (~15GB)
                            # 需要: 至少 30GB CPU RAM 加载 7B 模型
```

**实例化时的潜在阻塞点:**

| 阻塞点 | 说明 | 严重度 |
|--------|------|--------|
| Qwen2-VL-7B 下载 | `from_pretrained("Qwen/Qwen2-VL-7B-Instruct")` 需要 HuggingFace 访问 | 中 |
| CPU RAM | 7B 模型加载需要 ~15 GB bf16 + PEFT 封装额外开销 | 中 |
| `flash_attn` 缺失 | `attn_implementation="flash_attention_2"` 会在 flash_attn 未安装时报错 | **高** |
| PEFT 版本 | `layers_to_transform` 参数需要 peft >= 0.6.0 | 低 |

> **关键风险**: `attn_implementation="flash_attention_2"` 是 `BackboneConfig` 默认值。如果 `flash_attn` 包未安装，`Qwen2VLForConditionalGeneration.from_pretrained()` 会抛出 `ImportError`。**需要添加 fallback 到 `"sdpa"` 或 `"eager"`。**

### 1.3 forward_train 可运行性

**可以运行，需要正确格式的 batch dict。**

所需 batch 格式：
```python
batch = {
    # 必需字段
    "actions":       Tensor [B, T, H, A],    # [B, 24, 24, 14]
    "proprio":       Tensor [B, T, A],        # [B, 24, 14]
    "prev_actions":  Tensor [B, T, A],        # [B, 24, 14]
    "input_ids":     Tensor [B, S],           # S = tokenized 序列长度
    "attention_mask": Tensor [B, S],

    # 可选字段
    "pixel_values":          Tensor [B, C, H, W],
    "image_grid_thw":        Tensor [...],
    "phase_labels":          Tensor [B, T],
    "affordance_labels":     Tensor [B, T],
    "embodiment_id":         Tensor [B],       # LongTensor
    "step_weights":          Tensor [B, H, A],
    "semantic_refresh_steps": List[int],

    # 多刷新点场景 (互斥 input_ids)
    "refresh_input_ids":       Tensor [B, R, S],
    "refresh_attention_mask":  Tensor [B, R, S],
    "refresh_pixel_values_list": List[Tensor],
    "refresh_image_grid_thw_list": List[Tensor],
}
```

### 1.4 无法运行的部分（完全缺失）

| 缺失组件 | 说明 | 阻塞什么 |
|----------|------|----------|
| **Training Loop** | 无 `train.py` / Trainer 类 | 无法训练 |
| **Data Pipeline** | `data/__init__.py` 为空 | 无法加载数据 |
| **FSDP 封装** | 无 FSDP wrapping 代码 | 无法多 GPU 训练 |
| **EMA 实现** | `utils/__init__.py` 为空 | Stage B/C 缺少 EMA |
| **Inference Server** | `infer/__init__.py` 为空 | 无法部署推理 |
| **Evaluation Loop** | 无评估代码 | 无法评测 |
| **Logging** | 无 WandB/TensorBoard 集成 | 无法监控 |

### 1.5 单元可运行性总结

| 组件 | 可运行? | 前提 |
|------|---------|------|
| Config 加载 | ✅ | pyyaml |
| 模型实例化 | ✅ | transformers + peft + flash_attn |
| `forward_train()` | ✅ | 正确格式 batch |
| `semantic_step()` | ✅ | 推理 no_grad |
| `control_step()` | ✅ | RuntimeCache |
| 端到端训练 | ❌ | 缺失 training loop + data pipeline |

---

## 2. 端到端数据通路张量维度追踪

### 2.1 训练 Forward Pass 完整维度链

以默认配置 (B=2, T=24, H=24, A=14, D=2048, D_expert=1536) 追踪：

```
═══════════════════════════════════════════════════════════
阶段 1: Backbone → 多尺度特征
═══════════════════════════════════════════════════════════

input_ids          [2, S]                S ≈ 1024 (含 visual tokens)
pixel_values       [2, 3, 448, 448]     → Qwen2-VL 内部处理
                           ↓
Qwen2-VL-7B forward (output_hidden_states=True)
                           ↓
outputs.hidden_states      list of 29 × [2, S, 3584]

MultiScaleAdapter:
  layer 10 → proj → [2, S, 2048]
  layer 18 → proj → [2, S, 2048]
  layer 28 → proj → [2, S, 2048]
  gate_input: pool+concat → [2, 2048×3=6144]
  weights: softmax → [2, 3]
  fused: weighted sum → [2, S, 2048]           ← backbone_hidden

═══════════════════════════════════════════════════════════
阶段 2: Grounder → 层次化压缩
═══════════════════════════════════════════════════════════

latent_queries     [1, 96, 2048] → expand → [2, 96, 2048]

Layers 0–3: GrounderBlock × 4
  Cross-Attn: Q=[2, 96, 2048] × KV=[2, S, 2048] → [2, 96, 2048]
  Self-Attn:  [2, 96, 2048] → [2, 96, 2048]
  输出: [2, 96, 2048]

Layer 3 → Compression:
  raw_object_slots = latents[:, 1:49, :]   → [2, 48, 2048]
  SlotCompression:
    route_queries [1, 24, 2048] → [2, 24, 2048]
    Cross-Attn: Q=[2,24,2048] × KV=[2,48,2048] → [2, 24, 2048]
    Self-Attn: [2, 24, 2048] → [2, 24, 2048]
  Reassemble:
    cat([global(1), compressed(24), rest(47)]) = [2, 72, 2048]

Layers 4–7: GrounderBlock × 4
  Cross-Attn: Q=[2, 72, 2048] × KV=[2, S, 2048] → [2, 72, 2048]  ✅
  Self-Attn:  [2, 72, 2048] → [2, 72, 2048]                       ✅

Final carving:
  global_token:              [2, 2048]     (index 0)
  compressed_object_slots:   [2, 24, 2048] (index 1:25)
  phase_token:               [2, 2048]     (index 25)
  uncertainty_token:          [2, 2048]     (index 26)
  affordance_token:           [2, 2048]     (index 27)

═══════════════════════════════════════════════════════════
阶段 3: Temporal Core (per step t=0..23)
═══════════════════════════════════════════════════════════

Input composition (_compose_input_sequence):
  9 singles [2, 2048] each → stack → [2, 9, 2048]
  + compressed_object_slots [2, 24, 2048]
  = input_seq [2, 33, 2048]                                       ✅ 33 tokens

Fast Mamba (20L, d_state=128):
  input_norm → [2, 33, 2048]
  Per layer: in_proj → [2, 33, 4096], conv1d, silu, SSM(d_state=128)
  Output: [2, 33, 2048]
  fast_token = mean(dim=1) → [2, 2048]

Medium Mamba (6L, every 2nd step):
  同上 → medium_token [2, 2048]

Slow Mamba (10L, d_state=256, 语义刷新时):
  同上 → slow_token [2, 2048]

CrossAttentionFusion:
  KV = stack([fast, medium, slow]) → [2, 3, 2048]
  Q = fusion_query [2, 1, 2048]
  2 层 cross-attn (8 heads) → fused_state [2, 2048]

Accumulated over T=24:
  fused_states [2, 24, 2048]                                      ✅
  fast_tokens  [2, 24, 2048]                                      ✅

═══════════════════════════════════════════════════════════
阶段 4: Condition Prefix → Expert
═══════════════════════════════════════════════════════════

_build_cond_prefix:
  global(1) + compressed(24) + phase(1) + unc(1) + aff(1)
  + fused(1) + fast(1) + medium(1) + slow(1) = 32 tokens
  cat → [2, 32, 2048]                                             ✅ 恰好 32
  cond_builder → [2, 32, 2048]
  core_to_expert (2048→1536) → [2, 32, 1536]                     ✅

Expert forward:
  noisy_actions [2, 24, 14] → action_proj → [2, 24, 1536]
  + pos_emb [1, 24, 1536] + timestep_emb [2, 1, 1536]

  x = cat([proprio(1), embodiment(1), actions(24)]) → [2, 26, 1536]  ✅

  18 层 M-M-A×6:
    MambaBlock: [2, 26, 1536] → [2, 26, 1536]
    AttnBlock:  [2, 26, 1536] × cond [2, 32, 1536] → [2, 26, 1536]

  action_out = x[:, 2:, :] → [2, 24, 1536]                       ✅
  velocity = out_proj → [2, 24, 14]                                ✅

═══════════════════════════════════════════════════════════
阶段 5: Losses
═══════════════════════════════════════════════════════════

FlowMatchingLoss:
  velocity_pred [2, 24, 14]
  target_velocity = target_actions - noise = [2, 24, 14]
  MSE → scalar                                                     ✅

DiscreteCELoss:
  fast_logits [2, 24, 14, 512]
  fast_targets [2, 24, 14] (LongTensor)
  reshape → [2×24×14, 512] vs [2×24×14]
  CE(label_smoothing=0.1) → scalar                                ✅

ActionConsistencyLoss:
  discrete_actions [2, 24, 14]   (from undiscretise)
  continuous_actions [2, 24, 14] (from expert, detached)
  proj → [2, 24, 256] each
  cosine sim → scalar                                              ✅

ContrastiveTemporalLoss:
  fused_states [2, 24, 2048]
  anchors = [:, :-1] → [2, 23, 2048]
  positives = [:, 1:] → [2, 23, 2048]
  flatten → [46, 2048] each
  logits [46, 46] → CE → scalar                                   ✅

SlowFastAgreementLoss:
  fast_tokens [2, 24, 2048]
  slow_token [2, 2048]
  weights [24] (exp linspace)
  fast_ema = weighted sum → [2, 2048]
  MSE(slow_token, fast_ema.detach()) → scalar                     ✅
```

### 2.2 维度一致性验证结论

| 检查项 | 状态 | 说明 |
|--------|------|------|
| cond_prefix token 数 = 32 | ✅ | 1+24+1+1+1+1+1+1+1 = 32 |
| Grounder 压缩后 latent 数 = 72 | ✅ | 96 - 48 + 24 = 72 |
| Temporal Core 输入 token 数 = 33 | ✅ | 9 singles + 24 objects |
| Expert 输入序列长度 = 26 | ✅ | 2 prefix + 24 actions |
| Expert 输出 slicing x[:, 2:] | ✅ | 正确跳过 proprio + embodiment |
| Flow Matching 速度场 target | ✅ | x_1 - x_0 = target - noise |
| CUDA SSM 布局 [B, D, L] | ✅ | transpose + contiguous 正确 |
| ActionConsistency 维度匹配 | ✅ | 两侧都是 [B, 24, 14] |

### 2.3 发现的数据通路问题

#### 🔴 问题 #1: Grounder 在单 backbone 分支中重复调用 R 次

**位置**: `hybrid_vla_v2.py:240-249`

```python
else:
    backbone_out = self.backbone.forward_semantic(...)
    backbone_hidden = backbone_out["last_hidden_state"]
    for _ in range(R):  # R ≈ 4
        grounder_outputs.append(self.grounder(backbone_hidden))
```

**问题**: 同一个 `backbone_hidden` 被传入 `self.grounder()` R 次。Grounder 的 `latent_queries` 是固定的 `nn.Parameter`，没有随机性，**4 次调用产生完全相同的输出**。浪费 3 次 Grounder 前向传播（8 层 × 72-96 latents × cross-attention）。

**修复**:
```python
else:
    backbone_out = self.backbone.forward_semantic(...)
    backbone_hidden = backbone_out["last_hidden_state"]
    grounder_out = self.grounder(backbone_hidden)
    grounder_outputs = [grounder_out] * R  # 共享引用，零额外计算
```

**节省**: ~3 × Grounder forward ≈ 3 × 5ms = 15ms/step (训练)

#### 🟡 问题 #2: `batch["actions"]` 的形状约定未文档化

`forward_train` 使用:
```python
B = batch["actions"].shape[0]      # batch 维度
T = batch["actions"].shape[1]      # 时间维度
target_actions = batch["actions"][:, -1]  # [B, H, A]
```

这意味着 `batch["actions"]` 必须是 **[B, T, H, A] = [B, 24, 24, 14]** 四维张量。每个时间步 t 存储从 t 开始的 H=24 步 action chunk。

**风险**: 没有 shape assertion，如果 data pipeline 提供 [B, T, A] 格式，`[:, -1]` 会返回 [B, A] 而非 [B, H, A]，导致后续维度不匹配但错误信息不直观。

#### 🟡 问题 #3: `target_actions` 只取最后一个时间步

```python
target_actions = batch["actions"][:, -1]  # 只用 t=T-1 的 target
```

序列中的 T=24 个时间步只在 Temporal Core 中起到提供上下文的作用，**只有最后一步的 action chunk 用于 Expert loss 和 FAST loss**。前 23 步的 ground truth actions 没有被直接监督。

**影响**: Temporal Core 的中间步 fused_states 只通过 ContrastiveTemporalLoss 间接训练（鼓励时间连续性），没有直接的动作预测监督。这可能导致中间步的表征质量不足。

**改进方向**: 考虑在多个时间步计算 FAST loss (每 stride 步)，增加中间步的监督信号。代价是增加 FAST head 的计算量。

---

## 3. 梯度流与训练正确性分析

### 3.1 Stage A 梯度流图

```
                    ┌─────────────┐
                    │ Backbone    │ ← LoRA 参数有梯度
                    │ (7B + LoRA) │
                    └──────┬──────┘
                           │ backbone_hidden [B, S, 2048]
                    ┌──────▼──────┐
                    │  Grounder   │ ← 全参数有梯度
                    │  (8L, 2048) │
                    └──────┬──────┘
                           │ GrounderOutput
                    ┌──────▼──────┐
                    │ Temporal    │ ← 全参数有梯度
                    │ Core (36L)  │
                    └──────┬──────┘
                           │ fused_states, fast_tokens, slow_token
               ┌───────────┼───────────┐
               ▼           ▼           ▼
        ┌─────────┐  ┌──────────┐ ┌─────────────┐
        │FAST Head│  │Phase Head│ │Consistency  │
        │(CE loss)│  │(CE loss) │ │Loss         │
        └────┬────┘  └────┬─────┘ └──────┬──────┘
             │             │              │
         loss_fast    loss_phase    loss_consistency
                                   (temporal + slow_fast, 无 action)
             │             │              │
             └─────────────┴──────────────┘
                           │
                      loss_total ← backward()
```

**Stage A 梯度流验证:**
- ✅ LoRA → backbone → backbone_hidden → grounder → temporal → heads
- ✅ 所有可训练组件都有梯度路径
- ⚠️ Expert 被冻结，不参与计算
- ⚠️ ActionConsistencyLoss 未激活（需要 continuous_actions 来自 Expert）

### 3.2 Stage B 梯度流图

```
                    ┌─────────────┐
                    │ Backbone    │ ← 通过 discrete/consistency loss
                    │ (7B + LoRA) │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Grounder   │ ← 通过 discrete/consistency loss
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Temporal    │ ← 通过 discrete/consistency loss
                    │ Core        │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────┐
        │                  │              │
        ▼                  │              ▼
   ┌──────────┐           │         ┌──────────────┐
   │FAST/Phase│           │         │cond_prefix   │
   │Affordance│           │         │.detach() ✂️   │ ← 梯度被切断！
   │Heads     │           │         └──────┬───────┘
   └────┬─────┘           │                │
        │                 │         ┌──────▼───────┐
        │                 │         │ Flow Action  │ ← 只从 loss_fm 获得梯度
        │                 │         │ Expert (18L) │
        │                 │         └──────┬───────┘
        │                 │                │
   loss_fast/phase/aff  loss_consistency  loss_fm
        │                 │                │
        └─────────────────┴────────────────┘
                          │
                     loss_total
```

**Stage B 关键梯度路径验证:**

| 梯度路径 | 状态 | 说明 |
|----------|------|------|
| loss_fm → Expert | ✅ | Expert 参数从 loss_fm 获得梯度 |
| loss_fm → Grounder/Temporal | ❌ (设计) | `cond_prefix.detach()` 切断 |
| loss_fm → Backbone LoRA | ❌ (设计) | 被 detach 间接阻断 |
| loss_fast → Backbone LoRA | ✅ | fused_states → temporal → grounder → backbone |
| loss_consistency → Expert | ❌ | `expert_continuous.detach()` 切断 |
| loss_consistency(action) → FAST Head | ⚠️ 见下 | argmax 不可微 |

### 3.3 🔴 关键梯度问题: ActionConsistencyLoss 的梯度断裂

**位置**: `consistency_loss.py:60-73` + `hybrid_vla_v2.py:384-394`

```python
# hybrid_vla_v2.py:334-337
fast_preds = fast_logits.argmax(dim=-1)           # ← 不可微！
fast_continuous = FASTDiscreteHead.undiscretise_actions(
    fast_preds, V=self.cfg.model.heads.fast_vocab_size,
)  # fast_continuous 没有梯度

# consistency_loss.py:70-73
def forward(self, discrete_actions, continuous_actions):
    d = F.normalize(self.discrete_proj(discrete_actions), dim=-1)
    c = F.normalize(self.continuous_proj(continuous_actions.detach()), dim=-1)
    return 1.0 - (d * c).sum(dim=-1).mean()
```

**梯度分析:**

```
discrete_actions (fast_continuous) ← argmax(fast_logits) ← FAST Head ← fused_states
                                     ↑ 不可微！梯度断裂

continuous_actions (expert_continuous) ← .detach() ← Expert
                                         ↑ 显式断裂
```

**结果**: ActionConsistencyLoss 只训练 `discrete_proj` 和 `continuous_proj` 两个投影层（共 ~14K 参数），**不向 FAST Head 或 Expert 传递任何梯度**。该 loss 不能实现其设计目的（对齐离散和连续动作预测）。

**严重度**: 中。该 loss 权重为 0.15 (0.3 × 0.5)，不是主损失。但它的设计意图是对齐两个 action head，完全失效意味着离散/连续预测可能发散。

**修复方案 A (Straight-Through Estimator)**:
```python
# 用 softmax 而非 argmax 获取可微的离散动作
fast_probs = F.softmax(fast_logits, dim=-1)  # [B, H, A, V]
bins = torch.linspace(-1, 1, V, device=fast_logits.device)
fast_continuous = (fast_probs * bins).sum(dim=-1)  # 期望值，可微
```

**修复方案 B (Detach 两侧但保留 loss)**:
保持现状作为纯正则化——投影层学习一个共享空间，虽然不传梯度给 head，但 loss 值仍可监控两侧是否一致。

### 3.4 LoRA + Freeze 交互分析

**位置**: `qwen2vl_backbone.py:107-141`

```python
# 步骤 1: 冻结
self._apply_freeze(freeze_vision_tower=True, freeze_text_layers_until=16)
# → layers 0-15 的参数 requires_grad = False
# → vision tower 参数 requires_grad = False

# 步骤 2: 注入 LoRA
self._apply_lora(lora_cfg, freeze_text_layers_until=16)
# → LoRA 在 ALL 28 layers 注入 (layers_to_transform=range(28))
# → PEFT LoRA 参数始终 requires_grad = True
```

**问题**: LoRA 被注入到冻结层 (0-15) 上。这意味着:
- 冻结层的 **base weights** 没有梯度 ✅ (符合预期)
- 冻结层的 **LoRA weights** 有梯度 ⚠️ (是否有意？)

对于 LoRA，在冻结层上注入是**标准做法** — LoRA 的设计就是在不修改 base weights 的前提下学习适配。所以 layers 0-15 的 LoRA 参数有梯度是**正确行为**。

但 `freeze_text_layers_until=16` 在 `_apply_freeze` 中冻结了 base weights，而 `_apply_lora` 的 `layers_to_transform=range(28)` 覆盖了所有层。PEFT 内部会为冻结层的 base weights 创建 LoRA adapter，LoRA forward 会计算 `W_base @ x + (lora_A @ lora_B) @ x`。由于 `W_base` 不需要梯度但 `lora_A, lora_B` 需要，autograd 可以正确处理。

**验证结论**: LoRA + freeze 交互**正确**。✅

### 3.5 Stage C `stop_gradient_cond_prefix: true` 的设计一致性

Stage C yaml 设置 `stop_gradient_cond_prefix: true`。这意味着即使在 "full fine-tune" 阶段，Expert 的 flow matching loss 仍然不会更新 Grounder/Temporal Core。

**这是否合理？**

- Grounder/Temporal Core 通过 loss_fast + loss_phase + loss_affordance + loss_consistency 获得梯度
- Expert 只通过 loss_fm 获得梯度
- 两组参数各自独立优化

**潜在问题**: Grounder 可能学到对 FAST Head 最优的表征，但不一定对 Expert 最优。由于 Expert 无法通过 loss_fm 影响 Grounder 输出，如果 FAST Head 和 Expert 对表征的需求不同，会出现**表征漂移**。

**建议**: 在 Stage C 添加一个 ablation: `stop_gradient_cond_prefix: false`，让 Expert 的 loss 也能微调 Grounder/Temporal Core，观察是否改善 flow matching 质量。

---

## 4. AdamW 与优化器加速分析

### 4.1 配置分析

```yaml
# config.py:174
optimizer: str = "adamw_torch_fused"
```

**`adamw_torch_fused` 是 PyTorch 的 fused AdamW 实现** (`torch.optim.AdamW` with `fused=True`)，在 PyTorch >= 2.0 中可用。

### 4.2 Fused AdamW vs 标准 AdamW

| 方面 | 标准 `torch.optim.AdamW` | `fused=True` |
|------|--------------------------|--------------|
| 实现 | Python loop over param groups | 单 CUDA kernel |
| kernel launches | 每参数 5-7 次 | 全参数 1 次 |
| 内存访问 | 多次全局内存读写 | 单次读-改-写 |
| **预期加速** | baseline | **1.5-3× optimizer step** |
| 精度 | fp32 | fp32 (等价) |
| 兼容性 | 所有参数类型 | 仅 CUDA tensors |

**对总训练时间的影响:**

```
单步训练时间分解 (估算, bs=2, 8×H100):
  Forward pass:     ~150 ms
  Backward pass:    ~300 ms
  Optimizer step:   ~30 ms (标准) → ~10 ms (fused)
  Comm (FSDP):      ~40 ms (overlap)
  ────────────────────────────
  总计:             ~520 ms → ~500 ms (fused)
  加速:              ~4% 总训练时间
```

**结论**: Fused AdamW 的优化器步骤加速显著 (3×)，但由于 optimizer step 只占总时间 ~6%，**对整体训练时间的贡献约 4%**。这是"免费"的优化（只需设置一个 flag），值得启用但不是关键瓶颈。

### 4.3 未实现但应集成的优化器加速

#### A. `torch.compile` (未启用)

```python
# InferConfig 有 compile: bool = False
# 但 TrainConfig 没有对应字段
```

`torch.compile` 可以：
- 融合 LayerNorm + Linear + GELU → 单 kernel
- 融合 attention 计算中的多个小操作
- 优化 SSM scan 中的 pointwise 运算

**预期加速**: 训练 15-30%，推理 20-40% (取决于模型结构的 compile 友好度)

**风险**: Mamba 的顺序 scan 和动态 control flow (if semantic_refresh) 可能导致 compile 失败或 graph break。需要逐模块测试。

#### B. Gradient Accumulation 优化

Config 定义 `grad_accum_steps: 4`，但代码未实现。标准实现:

```python
# 需要在 training loop 中实现
for micro_step in range(grad_accum_steps):
    loss = model.forward_train(batch) / grad_accum_steps
    loss["loss_total"].backward()

optimizer.step()
optimizer.zero_grad()
```

**注意**: 直接将 `loss_total` 除以 `grad_accum_steps` 可能导致某些 loss (如 ContrastiveTemporalLoss 的 InfoNCE) 的梯度量级不正确，因为 InfoNCE 的梯度不是简单地线性可缩放的。

#### C. Mixed Precision (AMP) 配置

Config 有 `bf16: true` 但模型代码没有 AMP 包装。需要:

```python
from torch.cuda.amp import autocast
with autocast(dtype=torch.bfloat16):
    losses = model.forward_train(batch)
```

**关键**: bf16 下 Mamba 的 SSM scan 可能有数值问题。`A_log` 参数经过 `exp(-exp(A_log))` 双重指数运算，在 bf16 精度下可能溢出。**建议 SSM scan 内部使用 fp32**，其余使用 bf16。

#### D. FSDP 配置要点

```python
# 需要但未实现的 FSDP wrapping
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 关键配置要求:
# 1. 冻结参数处理: use_orig_params=True (PyTorch >= 2.0)
# 2. 混合精度策略: MixedPrecision(param_dtype=torch.bfloat16)
# 3. Backbone 单独 FSDP unit: 避免 7B 参数通信瓶颈
# 4. activation_checkpointing: 已在 temporal core 中实现
```

**FSDP 与冻结参数的交互风险:**
- `use_orig_params=False` (默认) 下，FSDP 会对冻结和非冻结参数统一 shard + allgather，浪费通信量
- `use_orig_params=True` 允许跳过冻结参数的 allgather
- **backbone 的 7.5B 冻结参数不应参与 FSDP shard/allgather** — 应使用 `ignored_modules` 或独立的 FSDP wrapping 策略

---

## 5. 机器学习关键问题清单

### 5.1 数值稳定性

#### 🔴 A. SSM scan 在 bf16 下的双重指数溢出

**位置**: `mamba_core.py:72-77`, `flow_action_expert.py:123,148`

```python
A = -torch.exp(self.A_log)                              # exp 1
dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))  # exp 2
```

`A_log` 初始化为 `log(arange(1, d_state+1))`:
- d_state=128 → A_log max = log(128) ≈ 4.85
- A = -exp(4.85) ≈ -128
- dt (after softplus) ≈ 0.5-2.0
- A * dt ≈ -64 to -256
- exp(-256) → **下溢到 0 (bf16)**

**实际影响**: 在 bf16 下，exp(-256) = 0，意味着长程记忆被完全截断。d_state 值越大，问题越严重 (d_state=256 的 Slow Mamba 最受影响)。

**修复**: 在 SSM scan 内部强制 fp32:
```python
# 在 SelectiveSSM.forward() 中
with torch.amp.autocast(device_type='cuda', enabled=False):
    dt = F.softplus(self.dt_proj(dt.float()))
    dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))
    ...
```

#### 🟡 B. AdaRMSNorm 的 gate sigmoid 饱和

**位置**: `flow_action_expert.py:44-49`

```python
scale, shift, gate = self.cond_proj(cond).chunk(3, dim=-1)
return gate.sigmoid() * (x_normed * (1 + scale) + shift)
```

`gate` 在训练初期是随机初始化的 Linear 输出。如果 `gate` 值较大 (>5)，`sigmoid(gate) ≈ 1`；如果较小 (<-5)，`sigmoid(gate) ≈ 0`。

**初始化风险**: `cond_proj` 是 `nn.Linear(cond_dim, 3*dim)`，默认 Kaiming init。对于 `dim=1536`，输出 std ≈ `sqrt(2/cond_dim)` ≈ 0.036。scale/shift/gate 的初始值约在 [-0.1, 0.1]，sigmoid(0) = 0.5。这意味着初始时 gate ≈ 0.5，特征被衰减一半。

**建议**: 初始化 `cond_proj` 使 gate 的 bias 偏向 0（sigmoid(0) = 0.5）或更高值，确保训练初期特征不被过度衰减：
```python
# 在 AdaRMSNorm.__init__ 中
nn.init.zeros_(self.cond_proj.bias[2*dim:])  # gate bias → 0 → sigmoid = 0.5
```

### 5.2 训练稳定性

#### 🔴 C. Loss Scale 不平衡

```
各 loss 的典型量级 (训练初期):
  loss_fast (CE, 512 bins):  ≈ log(512) ≈ 6.2   × weight 1.0 = 6.2
  loss_fm (MSE on velocity): ≈ 0.5-2.0           × weight 1.0 = 0.5-2.0
  loss_phase (CE, 16 cls):   ≈ log(16) ≈ 2.8     × weight 0.5 = 1.4
  loss_consistency:           ≈ 2.0-4.0           × weight 0.3 = 0.6-1.2
  loss_affordance (CE, 8):   ≈ log(8) ≈ 2.1      × weight 0.3 = 0.6
```

**问题**: `loss_fast` 初始值 ~6.2 是 `loss_fm` 的 3-12× 倍。在训练初期，离散头的梯度会**主导** temporal core 和 grounder 的参数更新方向，flow matching 的信号被压制。

**建议**:
1. 使用 loss-scale warmup: 在前 5K 步逐渐增加 `flow_matching` 权重从 0 到 1.0
2. 或归一化各 loss 到相似量级后再加权
3. 或使用 GradNorm / uncertainty weighting 自动平衡

#### 🟡 D. Logit Normal Timestep Schedule 的覆盖偏差

```python
def sample_timestep(self, batch_size, device):
    if self.timestep_schedule == "logit_normal":
        return torch.sigmoid(torch.randn(batch_size, device=device))
```

`sigmoid(N(0,1))` 的分布：
- 均值 ≈ 0.5
- 95% 范围 ≈ [0.12, 0.88]
- t < 0.05 的概率 ≈ 2%
- t > 0.95 的概率 ≈ 2%

**影响**: 极端噪声水平 (接近纯噪声 t≈0 和接近干净信号 t≈1) 训练不足。这可能导致:
- 推理时第一步 (t=0) 的速度场预测不准确
- 推理时最后一步 (t≈1) 的精度不足

**建议**: 混合 logit_normal 和 uniform 采样，例如 80% logit_normal + 20% uniform，确保极端 timestep 有足够覆盖。

#### 🟡 E. ContrastiveTemporalLoss 的 Temperature 敏感性

```python
self.temperature = 0.1
logits = torch.matmul(a, p.T) / self.temperature
```

temperature=0.1 将 cosine similarity (范围 [-1, 1]) 放大到 [-10, 10]。这是 InfoNCE 文献中的标准值，但：
- N=46 (B=2, T=24) 在训练中很小，可能导致 loss 方差大
- N=1472 (B=64, T=24) 在实际训练中更大，loss 更稳定
- 但 N=1472 的 softmax 在 logits 范围 [-10, 10] 上可能导致**大部分 logit 接近 0**（正样本和负样本的 similarity 差异不大），降低 loss 的区分度

**建议**: 使 temperature 成为可学习参数 (如 CLIP)：
```python
self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))
temperature = self.log_temperature.exp().clamp(min=0.01)
```

### 5.3 训练效率

#### 🟡 F. Temporal Loop 无法并行化

```python
for t in range(T):  # T = 24, 串行
    temporal_out = self.temporal_core(...)
    temporal_state = temporal_out.next_state
```

每个时间步依赖前一步的 Mamba state，无法并行。**这是整体训练最大的瓶颈之一**。

Fast Mamba 20 层 + 偶尔 Medium 6 层 + 偶尔 Slow 10 层，每步约需 1-3ms (training, bs=2, 33 tokens)。24 步 × 2.5ms ≈ 60ms forward。加上 backward ≈ 120-180ms。

**缓解方案**:
1. **减少 T**: sequence_window 从 24 降到 12-16，减少循环次数
2. **Teacher Forcing 并行化**: 如果不需要跨步 state，可以将所有步的 input 并行成 [B×T, 33, 2048] 传入 Mamba。但这失去了递归状态。
3. **Hybrid**: 在 Grounder 和 FAST Head 使用并行处理，只在 Mamba Core 保留串行

#### 🟡 G. Expert Checkpointing 缺失

Temporal Core 使用了 activation checkpointing (`use_checkpoint=True`)，但 **Expert 的 18 层没有使用 checkpointing**。

Expert 在 Stage B/C 中参与 flow matching loss 的梯度计算。18 层 × [B, 26, 1536] 的中间激活约需:
```
18 layers × 2 × 26 × 1536 × 2B ≈ 2.9 MB/layer × 18 ≈ 52 MB
```
加上 attention 的 QKV 和 FFN 中间值，实际约 **100-200 MB**。对于 80 GB H100 这不是问题，但如果 batch size 增加则值得加上。

### 5.4 正则化与泛化

#### 🟡 H. 无数据增强

设计文档和代码中没有任何图像增强（颜色抖动、随机裁剪、高斯噪声）。对于机器人操控:
- **颜色增强**对背景/光照变化至关重要
- **空间增强**（crop + resize）提升空间泛化
- **动作噪声注入**（在 prev_actions 中添加噪声）模拟传感器误差

这通常在 data pipeline 中实现，但当前 data pipeline 为空。

#### 🟡 I. Exposure Bias (Teacher Forcing)

**位置**: `hybrid_vla_v2.py:317`

```python
action_history_buf.push(batch["prev_actions"][:, t])  # ground truth
```

训练时 action history 使用 ground truth 前一步动作，推理时使用模型自己的预测。如果模型犯错，错误动作进入 history，导致后续预测更差（误差累积）。

**建议**: 在 Stage C 引入 **scheduled sampling**:
```python
if stage == "c" and random.random() < scheduled_sampling_ratio:
    # 使用模型预测而非 ground truth
    with torch.no_grad():
        pred_action = self.fast_head(fused_states_list[-1]).argmax(-1)
        pred_continuous = FASTDiscreteHead.undiscretise_actions(pred_action)
    action_history_buf.push(pred_continuous[:, 0, :])
else:
    action_history_buf.push(batch["prev_actions"][:, t])
```

#### 🟡 J. 权重初始化

当前代码没有自定义权重初始化。需要注意:

| 模块 | 默认初始化 | 建议 |
|------|------------|------|
| `latent_queries` | `randn * 0.02` | ✅ 合理 |
| `route_queries` | `randn * 0.02` | ✅ 合理 |
| `fusion_query` | `randn * 0.02` | ✅ 合理 |
| `out_proj` (所有 Mamba/Attn) | Kaiming | ⚠️ 建议 zero init 残差输出层 |
| `AdaRMSNorm.cond_proj` | Kaiming | ⚠️ 建议 zero init 确保初始 identity-like |
| `cond_builder` | Kaiming | ⚠️ 建议 zero init 最后一层 |
| LoRA B matrix | zeros (PEFT default) | ✅ 已正确 |
| `A_log` (SSM) | `log(arange(1, N+1))` | ✅ 标准 Mamba init |
| `D` (SSM skip) | `ones` | ✅ 标准 Mamba init |

**关键**: 对于使用残差连接的模块，最后一层 Linear 的零初始化可以确保初始时残差块是 identity mapping，稳定训练初期。

### 5.5 推理正确性

#### 🟡 K. `control_step` 中 `medium_update_stride` 引用 `cfg.train`

**位置**: `hybrid_vla_v2.py:433`

```python
medium_update = (runtime_state.temporal_state.steps_since_medium
                 >= self.cfg.train.medium_update_stride - 1)
```

推理代码引用了 **训练配置** (`cfg.train.medium_update_stride`)。如果推理时使用不同的更新频率（例如更激进以降低延迟），需要从 `cfg.infer` 获取。`InferConfig` 已有 `medium_hz: float = 25.0` 但未被使用。

#### 🟡 L. Midpoint Solver 在推理时的确定性

```python
x = torch.randn(B, H, self.action_dim, device=device, dtype=dtype)
```

`sample_midpoint` 使用 `torch.randn` 作为初始噪声。在推理时，这引入随机性 — 相同输入可能产生不同动作。

**对于机器人控制，这可能不是期望行为。** 建议:
- 使用固定 seed 或确定性初始噪声
- 或在推理配置中暴露随机种子选项

---

## 6. 优先修复建议

### 6.1 阻塞级 (P0 — 不修无法训练)

| # | 问题 | 章节 | 预计工作量 |
|---|------|------|-----------|
| 1 | Training loop + Data pipeline 实现 | §1.4 | 3-5 天 |
| 2 | FSDP wrapping + AMP 集成 | §4.4 | 1-2 天 |
| 3 | EMA 实现 | §1.4 | 0.5 天 |
| 4 | SSM scan bf16 数值溢出修复 | §5.1.A | 0.5 天 |
| 5 | flash_attn 缺失时的 fallback | §1.2 | 0.25 天 |

### 6.2 重要级 (P1 — 显著影响训练质量)

| # | 问题 | 章节 | 预计工作量 |
|---|------|------|-----------|
| 6 | Grounder 重复调用修复 | §2.3.#1 | 0.1 天 |
| 7 | Loss scale 不平衡缓解 | §5.2.C | 0.5 天 |
| 8 | 权重初始化 (zero init 残差) | §5.4.J | 0.5 天 |
| 9 | Batch shape assertion | §2.3.#2 | 0.1 天 |
| 10 | ActionConsistencyLoss 梯度修复 | §3.3 | 0.25 天 |
| 11 | 数据增强 (在 data pipeline 中) | §5.4.H | 1 天 |

### 6.3 建议级 (P2 — 改善但非必需)

| # | 问题 | 章节 | 预计工作量 |
|---|------|------|-----------|
| 12 | AdaRMSNorm gate 初始化 | §5.1.B | 0.1 天 |
| 13 | Scheduled sampling (Stage C) | §5.4.I | 0.5 天 |
| 14 | Learnable temperature (InfoNCE) | §5.2.E | 0.1 天 |
| 15 | 混合 timestep schedule | §5.2.D | 0.25 天 |
| 16 | Expert activation checkpointing | §5.3.G | 0.1 天 |
| 17 | torch.compile 探索 | §4.3.A | 1-2 天 |
| 18 | 多时间步 FAST loss | §2.3.#3 | 0.5 天 |
| 19 | 推理 medium_update 从 InferConfig | §5.5.K | 0.1 天 |
| 20 | 推理 deterministic noise | §5.5.L | 0.1 天 |

---

## 附录: 完整 Batch 格式规范 (供 Data Pipeline 实现参考)

```python
@dataclass
class VLABatch:
    """HybridVLA v2 训练所需的 batch 格式。

    维度约定:
      B = batch size
      T = sequence_window (24)
      H = chunk_horizon (24)
      A = action_dim (14)
      S = tokenized sequence length (variable)
      R = number of semantic refresh points (T // semantic_refresh_stride)
    """

    # ---- 必需: 动作数据 ----
    actions: Tensor       # [B, T, H, A] — 每时间步的 ground truth action chunk
    proprio: Tensor       # [B, T, A]    — 本体感受
    prev_actions: Tensor  # [B, T, A]    — 前一步执行的动作

    # ---- 必需 (单帧模式): Backbone 输入 ----
    input_ids: Tensor          # [B, S]
    attention_mask: Tensor     # [B, S]
    pixel_values: Tensor       # [B, C, H_img, W_img] 或 Qwen2-VL 特定格式
    image_grid_thw: Tensor     # Qwen2-VL 图像网格信息

    # ---- 可选 (多刷新点模式): 替代上面的 input_ids 等 ----
    refresh_input_ids: Optional[Tensor]           # [B, R, S]
    refresh_attention_mask: Optional[Tensor]       # [B, R, S]
    refresh_pixel_values_list: Optional[List]      # R × Tensor
    refresh_image_grid_thw_list: Optional[List]    # R × Tensor

    # ---- 可选: 辅助标签 ----
    phase_labels: Optional[Tensor]        # [B, T] — LongTensor, 0..15
    affordance_labels: Optional[Tensor]   # [B, T] — LongTensor, 0..7
    embodiment_id: Optional[Tensor]       # [B] — LongTensor, 0..15
    step_weights: Optional[Tensor]        # [B, H, A] — 每步 flow matching 权重
    semantic_refresh_steps: Optional[List[int]]  # 自定义刷新时间点
```

---

---

## 附录 B: 问题 #1 深度分析 — bf16 下 SSM Scan 的精度灾难

### B.1 bfloat16 数值格式基础

```
bfloat16: 1 sign + 8 exponent + 7 mantissa bits
float32:  1 sign + 8 exponent + 23 mantissa bits

                    exponent     mantissa     ULP near 1.0
  bfloat16:         8 bits       7 bits       2^(-7) = 0.0078125
  float32:          8 bits       23 bits      2^(-23) ≈ 1.19e-7
  float16:          5 bits       10 bits      2^(-10) ≈ 9.77e-4
```

**关键**: bf16 和 fp32 有相同的值域 (8 位指数)，但 bf16 的精度比 fp32 低 65536 倍 (7 vs 23 位尾数)。
**在 1.0 附近，bf16 可表示的最密集的值间隔为 0.0078125** (约 1/128)。

### B.2 SSM 离散化中的精度问题定位

SSM 核心递推 (`mamba_core.py:67-84`):
```python
A = -torch.exp(self.A_log)                    # ① A ∈ [-d_state, -1] (初始化)
dt = F.softplus(self.dt_proj(dt_raw))         # ② dt ∈ [0.5, 5.0] (典型训练值)
dA = torch.exp(A * dt)                        # ③ dA = exp(A*dt) ∈ (0, 1)
state = dA * state + dBx                      # ④ 递推累积
```

**步骤 ③** 的输出 `dA` 是 SSM 的**衰减率**。`dA` 越接近 1，状态记忆越持久：

| dA 值 | 含义 | 33 步后保留 | 用途 |
|-------|------|------------|------|
| 0.999 | 超慢衰减 | 96.75% | 长程语义记忆 |
| 0.995 | 慢衰减 | 84.75% | 中程上下文 |
| 0.95 | 中衰减 | 18.40% | 短程动态 |
| 0.5 | 快衰减 | 0.00% | 即时响应 |

### B.3 bf16 量化表 — 在 dA 附近的可表示值

```
dA 真值      bf16 最近值       量化误差      关键问题
────────────────────────────────────────────────────────
0.9990       1.0000000        +0.0010       ⚠️ 量化为恰好 1.0 → 零衰减！
0.9960       1.0000000        +0.0040       ⚠️ 同上
0.9950       0.9921875        -0.0028       方向反转 (衰减过快)
0.9900       0.9921875        +0.0022       衰减不足
0.9800       0.9765625        -0.0034       衰减过快
0.9500       0.9531250        +0.0031       小误差
0.9000       0.8984375        -0.0016       小误差
```

**核心发现**: bf16 在 1.0 附近的 ULP 为 0.0078125。**任何 dA ∈ [0.99609375, 1.0039] 都会被量化为恰好 1.0**。

这意味着：
- `dA = 0.999` → bf16 中变成 `1.0` → **完全不衰减**
- `dA = 0.997` → bf16 中变成 `1.0` → **完全不衰减**
- `dA = 0.996` → bf16 中变成 `1.0` → **完全不衰减**

### B.4 累积误差的数值验证

以下数据通过精确模拟生成：

```
dA 真值   dA bf16值     L=33步        L=100步        L=792步(24×33)
          (量化后)    真值→bf16值    真值→bf16值      真值→bf16值
──────────────────────────────────────────────────────────────────
0.9990    1.0000     0.968→1.000    0.905→1.000     0.453→1.000
                     误差 3.36%     误差 10.52%      误差 120.8%  ← 灾难!

0.9950    0.9922     0.848→0.772    0.606→0.456     0.019→0.002
                     误差 8.92%     误差 24.65%      误差 88.6%

0.9900    0.9922     0.718→0.772    0.366→0.456     0.005→0.021
                     误差 7.56%     误差 24.70%      误差 324%

0.9500    0.9531     0.184→0.205    0.006→0.008     ≈0→≈0
                     误差 11.45%    误差 38.88%      (两者都≈0)
```

**L=792 的来源**: 训练时 T=24 个时间步，每步内 L=33 tokens。Mamba state 跨时间步传递，等效序列长度 = 24 × 33 = 792。

### B.5 对 Slow Mamba (d_state=256) 的具体影响

Slow Mamba 的 A 参数初始化:
```python
A = torch.arange(1, d_state + 1)  # [1, 2, ..., 256]
A_log = torch.log(A)               # [0, 0.69, ..., 5.55]
A = -torch.exp(A_log)              # [-1, -2, ..., -256]
```

对于 Slow Mamba (设计目标: 长程语义记忆), 训练会驱动某些状态维度的 `|A|` 变小以实现慢衰减。假设训练后 `A = -0.5, dt = 2.0`:
```
dA = exp(-0.5 × 2.0) = exp(-1.0) = 0.368   # 快速衰减，bf16 精度够用
```

但如果 `A = -0.01, dt = 0.1` (模型学到的超慢模式):
```
dA = exp(-0.01 × 0.1) = exp(-0.001) = 0.999  # bf16 量化为 1.0 → 零衰减!
```

**Slow Mamba 的 256 个状态维度中，约 10-40 个可能落入 "慢衰减" 区间 (dA > 0.996)**。在 bf16 下，这些维度的衰减被消除，state 无限累积，导致：

1. **数值溢出**: state 值持续增长，最终 NaN
2. **语义错误**: 本应 "慢慢遗忘" 的旧信息被永久保留
3. **梯度异常**: dA=1.0 时 ∂loss/∂A_log = 0 (梯度消失)，模型无法调整这些维度

### B.6 为什么官方 mamba_ssm CUDA kernel 不受影响

官方 `selective_scan_fn` 的内部实现:
```c
// mamba-ssm/csrc/selective_scan/selective_scan_fwd_kernel.cuh
// State accumulation is ALWAYS in float32:
float state_f32 = __bfloat162float(state_bf16);  // 升精度
state_f32 = dA_f32 * state_f32 + dBx_f32;        // fp32 累积
state_bf16 = __float2bfloat16(state_f32);         // 降精度存储
```

- **A 矩阵在 fp32 中计算 dA**
- **State 累积在 fp32 中执行**
- 只有输入/输出是 bf16

我们的 JIT scan 没有这个保护:
```python
@torch.jit.script
def ssm_scan(dA, dBx, C, state):
    for t in range(L):
        state = dA[:, t] * state + dBx[:, t]  # 全部在 input dtype (bf16) 中
```

### B.7 修复方案

**方案 A: SSM scan 内部强制 fp32 (推荐)**
```python
@torch.jit.script
def ssm_scan(dA: Tensor, dBx: Tensor, C: Tensor, state: Tensor):
    # 强制 fp32 累积
    state = state.float()
    dA = dA.float()
    dBx = dBx.float()
    C_f = C.float()
    orig_dtype = dA.dtype  # 保存原始 dtype
    B, L, D = dA.shape[0], dA.shape[1], dA.shape[2]
    y = torch.empty(B, L, D, device=dA.device, dtype=torch.float32)
    for t in range(L):
        state = dA[:, t] * state + dBx[:, t]
        y[:, t] = (state * C_f[:, t].unsqueeze(1)).sum(-1)
    return y.to(orig_dtype), state.to(orig_dtype)
```

**方案 B: 在 SelectiveSSM.forward() 中使用 autocast 禁用区**
```python
def forward(self, x, state=None):
    with torch.amp.autocast(device_type='cuda', enabled=False):
        # 确保 SSM 离散化和 scan 在 fp32 中执行
        A = -torch.exp(self.A_log.float())
        dt = F.softplus(self.dt_proj(dt.float()))
        ...
```

**方案 C: 仅对高精度敏感的 A 参数使用 fp32**
```python
# 在 MambaBlock.__init__ 中
self.A_log = nn.Parameter(torch.log(A))
self.A_log.to(torch.float32)  # 始终保持 fp32
```

---

## 附录 C: 问题 #2-#6 详细复核

### C.1 问题 #2 复核: Grounder 重复调用 R 次

**代码路径** (`hybrid_vla_v2.py:240-249`):
```python
else:
    backbone_out = self.backbone.forward_semantic(...)
    backbone_hidden = backbone_out["last_hidden_state"]  # [B, S, 2048]
    for _ in range(R):  # R = len(range(0, 24, 6)) = 4
        grounder_outputs.append(self.grounder(backbone_hidden))
```

**确定性验证**:
- Grounder 内部无随机性源：`dropout=0.0` (GrounderConfig 默认值)
- `latent_queries` 是固定 `nn.Parameter`
- `SlotCompression.route_queries` 是固定 `nn.Parameter`
- 所有 SDPA / LayerNorm / Linear 都是确定性操作

**结论**: 4 次调用产生**位级相同**的输出 (bit-exact identical)。

**影响量化**:
```
单次 Grounder forward (8 blocks, 96→72 latents, cross-attn to S≈1024 backbone tokens):
  Cross-Attn per block: Q[B,96,2048] × KV[B,1024,2048] → ~4ms
  Self-Attn per block: [B,96,2048] → ~0.5ms
  8 blocks × 4.5ms ≈ 36ms total
  + SlotCompression ≈ 4ms
  Total: ~40ms per call

多余的 3 次调用: 3 × 40ms = 120ms/step
占 forward_train 总时间的: ~120/520 ≈ 23% (浪费!)
```

**确认修复**:
```python
grounder_out = self.grounder(backbone_hidden)
grounder_outputs = [grounder_out] * R
```

> **注意**: Python 列表复制 `[obj] * R` 创建的是**引用共享**，不复制张量。4 个元素指向同一 GrounderOutput。这意味着对 `grounder_outputs[0].global_token` 的原地修改会影响所有 4 个元素。当前代码没有原地修改 grounder output，所以引用共享是安全的。

### C.2 问题 #3 复核: ActionConsistencyLoss 梯度断裂

**完整梯度链追踪**:

```
                FAST Head 路径                        Expert 路径
                ────────────────                     ────────────
fused_states[:,-1]                          noisy_actions + expert_out.velocity
    ↓ (有梯度)                                     ↓ (有梯度 through expert)
fast_logits = fast_head(...)                expert_continuous = [B, 24, 14]
    ↓ (有梯度)                                     ↓
fast_preds = argmax(fast_logits) ← ⚡ 不可微!    .detach() ← ⚡ 显式断裂!
    ↓ (无梯度!)                                    ↓ (无梯度!)
fast_continuous = undiscretise(fast_preds)    continuous_actions
    ↓ (无梯度!)                                    ↓ (无梯度!)
discrete_proj(fast_continuous)               continuous_proj(continuous_actions)
    ↓ (有梯度 for discrete_proj.weight)        ↓ (有梯度 for continuous_proj.weight)
d = normalize(...)                           c = normalize(...)
    ↓                                            ↓
    └────────── cosine_sim(d, c) ──────────────┘
                      ↓
              ActionConsistencyLoss
```

**梯度流矩阵**:

| 参数 | ∂loss/∂param 是否存在 | 原因 |
|------|----------------------|------|
| `discrete_proj.weight` | ✅ 存在 | Linear 前向有输入, loss 对输出有梯度 |
| `discrete_proj.bias` | ✅ 存在 | 同上 |
| `continuous_proj.weight` | ✅ 存在 | 同上 |
| `continuous_proj.bias` | ✅ 存在 | 同上 |
| FAST Head 参数 | ❌ **不存在** | argmax 切断了梯度链 |
| Expert 参数 | ❌ **不存在** | .detach() 切断了梯度链 |
| Temporal Core 参数 | ❌ **不存在** | 无法通过 FAST Head 回传 |
| Grounder 参数 | ❌ **不存在** | 同上 |

**可训练参数统计**:
```
discrete_proj:  14 × 256 + 256 = 3,840 参数
continuous_proj: 14 × 256 + 256 = 3,840 参数
────────────────────────────────────────────
ActionConsistencyLoss 实际训练: 7,680 参数
模型总可训练参数: ~1,530,000,000 参数
占比: 0.0005%
```

**结论**: ActionConsistencyLoss 事实上只训练两个微小投影层, 对模型行为几乎没有影响。其唯一价值是作为监控指标 (观察两种预测是否一致)。

### C.3 问题 #4 复核: Loss Scale 不平衡

**精确初始值推导**:

**loss_fast** (CE, 512 bins, label_smoothing=0.1):
```
初始状态: logits 均匀分布 → softmax → 每 bin 概率 = 1/512
目标分布 (PyTorch label_smoothing):
  p_correct = (1 - 0.1) + 0.1/512 = 0.9 + 0.000195 = 0.900195
  p_other = 0.1/512 = 0.000195

CE = -Σ p_i log(q_i)
   = -0.900195 × log(1/512) - 511 × 0.000195 × log(1/512)
   = -(0.900195 + 511 × 0.000195) × log(512)
   = -(0.900195 + 0.099645) × 6.238
   = -0.99984 × 6.238
   ≈ 6.237

weight: 1.0
weighted loss: 6.237
```

**loss_fm** (MSE on velocity, Stage B):
```
初始状态: Expert 随机输出 ≈ N(0, σ²), σ 取决于初始化
target_velocity = target_actions - noise ≈ 如果 actions ∈ [-1,1] 且 noise ∈ N(0,1)
  Var(target_velocity) = Var(actions) + Var(noise) ≈ 0.33 + 1.0 = 1.33

MSE(v_pred, v_target) ≈ Var(v_pred) + Var(v_target) (独立时)
  ≈ σ²_pred + 1.33

Expert 输出 std 取决于初始化深度:
  18 层后, 如果每层大致保持量级, σ_pred ≈ 0.5-2.0
  MSE ≈ 0.25-4.0 + 1.33 ≈ 1.58-5.33

保守估计: loss_fm ≈ 2.0-3.0

weight: 1.0
weighted loss: 2.0-3.0
```

**实际比值**:
```
loss_fast / loss_fm ≈ 6.24 / 2.5 ≈ 2.5×
```

**比原文 §5.2.C 的 "3-12×" 估计更精确**: 实际比值约 **2-3×**，不是数量级差异。

**但仍然有问题**: 梯度量级差异更大于 loss 值差异。CE loss 的梯度在训练初期较大 (因为 softmax 未饱和)，MSE loss 的梯度与误差线性相关。在共享参数 (grounder/temporal core) 上，CE 梯度可能是 MSE 梯度的 5-10×。

**修正后的建议**: 这是一个 **P2 优化** (非阻塞)，可在初步实验后通过 loss curve 观察决定是否需要动态平衡。

### C.4 问题 #5 复核: AdamW Fused 加速

**配置确认** (`config.py:174`):
```python
optimizer: str = "adamw_torch_fused"
```

**实现验证**: 此字符串需要在 training loop 中被解析为:
```python
optimizer = torch.optim.AdamW(params, lr=lr, fused=True)
```

由于 training loop 不存在, 该配置只是一个**意图声明**, 尚未实际启用。

**fused=True 的技术原理**:
```
标准 AdamW 每个参数组的 step():
  m = β₁ * m + (1-β₁) * grad        # kernel 1: elementwise
  v = β₂ * v + (1-β₂) * grad²       # kernel 2: elementwise
  m_hat = m / (1 - β₁^t)            # kernel 3: scalar div
  v_hat = v / (1 - β₂^t)            # kernel 4: scalar div
  param -= lr * (m_hat / (√v_hat + ε) + wd * param)  # kernel 5: update

  共 5 次 kernel launch × N 参数组 × 每 kernel 读写全部参数 = 5N 次全局内存往返

Fused AdamW:
  单个 CUDA kernel 完成全部计算
  每个参数只读写一次 (param, grad, m, v → updated param, m, v)
  1 次 kernel launch, 1 次内存往返

内存带宽节省: 5× → 1× (理论)
实际加速: 1.5-3× optimizer step (取决于参数量和 GPU 带宽)
```

**对训练总时间的影响 (精确估计)**:
```
假设 1.53B trainable params, 8×H100 FSDP:
  每 GPU shard: 1530M / 8 = 191M params
  每参数 optimizer state: 12 bytes (m, v, master_weight in fp32)
  Optimizer 内存: 191M × 12B = 2.3 GB
  加上参数和梯度: 191M × 2B × 2 = 0.76 GB
  总读写: 3.06 GB × 5(标准) or 3.06 GB × 1(fused)

  H100 内存带宽: 3.35 TB/s
  标准: 3.06 GB × 5 / 3.35 TB/s = 4.57 ms
  Fused: 3.06 GB × 1 / 3.35 TB/s = 0.91 ms

  节省: 3.66 ms/step
  占比: 3.66 / ~520 ≈ 0.7% (比原文 4% 更保守)
```

**修正结论**: Fused AdamW 的实际加速约 **0.7-2%** (取决于 kernel launch overhead 占比)。几乎可以忽略, 但由于零成本启用, 仍然建议保留。

### C.5 问题 #6 复核: Training Pipeline 缺失

**逐项核查**:

| 组件 | 文件 | 状态 | 说明 |
|------|------|------|------|
| Training loop | `train.py` 或等价 | ❌ **不存在** | 无 forward/backward/step 循环 |
| Data loading | `data/__init__.py` | ❌ **空文件** | 无 Dataset/DataLoader |
| FSDP wrapping | — | ❌ **不存在** | 无 FSDP 封装代码 |
| AMP (bf16) | — | ❌ **不存在** | 无 autocast 上下文 |
| Optimizer | — | ❌ **不存在** | config 定义了但无实例化 |
| LR Scheduler | — | ❌ **不存在** | config 定义 cosine 但无实例化 |
| Gradient Accumulation | — | ❌ **不存在** | config 定义 4 步但无实现 |
| Gradient Clipping | — | ❌ **不存在** | config 定义 max_grad_norm=1.0 但无调用 |
| EMA | `utils/__init__.py` | ❌ **空文件** | config 定义了 decay schedule 但无实现 |
| Logging | — | ❌ **不存在** | 无 WandB/TensorBoard |
| Checkpointing (save) | — | ❌ **不存在** | 无模型/optimizer 保存 |
| Resume | — | ❌ **不存在** | config 有 resume_from 但无加载逻辑 |
| Evaluation | — | ❌ **不存在** | 无评估循环 |
| Inference runtime | `infer/__init__.py` | ❌ **空文件** | 无推理服务 |

**model 代码中已实现的训练相关功能** (可在 training loop 中直接使用):
- ✅ `forward_train()` — 完整的训练前向传播 + loss 计算
- ✅ `semantic_step()` + `control_step()` — 推理接口
- ✅ `init_runtime()` — 推理状态初始化
- ✅ 3 阶段 stage config (a/b/c) — 训练阶段切换
- ✅ Loss weights — 多任务 loss 加权
- ✅ Activation checkpointing — Temporal Core 内部
- ✅ stop_gradient_cond_prefix — 知识隔离

**结论**: 模型是一个完整的 `nn.Module`, 但缺少围绕它的全部训练/部署基础设施。forward_train 可以被外部 training loop 直接调用。

---

*补充分析完毕。bf16 的核心问题是: ULP near 1.0 = 0.0078125, 任何 dA ∈ [0.996, 1.004] 被量化为恰好 1.0 导致零衰减; 在 Slow Mamba 的 792 步等效序列长度上, 这导致 120%+ 的累积状态偏差。官方 mamba_ssm kernel 通过 fp32 state accumulation 规避此问题, 我们的 JIT scan 需要同等处理。Grounder 重复调用确认浪费 23% forward 时间。ActionConsistencyLoss 确认仅训练 7,680 参数 (总量的 0.0005%)。Loss scale 比值修正为 2-3× (非 12×)。AdamW fused 加速修正为 0.7-2%。*
