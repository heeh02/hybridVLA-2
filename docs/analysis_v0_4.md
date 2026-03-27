# HybridVLA v2 第四轮分析 (v0.4 — 世界模型集成审计)

> 分析日期: 2026-03-25
> 输入: `docs/recovery_v0_3.md` + `world_model/` 全部 9 个 Python 文件 + VLA 代码现状
> 方法: 逐模块张量追踪 → 跨模块接口审计 → loss 梯度流分析

---

## 1. 新增代码结构总览

```
vla_hybrid_v2/                          (VLA 核心, 2,660 行)
├── models/
│   ├── hybrid_vla_v2.py               主模型组装
│   ├── mamba_core.py                  Tri-Rate Mamba (含 MambaBlock)
│   ├── flow_action_expert.py          18L Flow Expert
│   ├── attention_grounder.py          层次化 Grounder
│   ├── qwen2vl_backbone.py            7B Backbone
│   └── discrete_heads.py             FAST/Phase/Affordance
├── losses/                            VLA 损失函数
├── ops/selective_scan.py              SSM 底层
├── config.py + types.py

vla_hybrid_v2/world_model/              (世界模型, 1,084 行) ★ 新增
├── stochastic_state.py        [97]    48×48 categorical + ST gradient
├── imagination_mamba.py      [113]    8L Mamba-2 转移网络
├── object_physics.py         [152]    6L GNN 物理引擎
├── world_model_heads.py      [115]    Reward/Value/Done + SymlogTwoHot
├── noise_augmentation.py      [79]    GameNGen 噪声增强
├── visual_decoder.py          [89]    112×112 CNN 解码器
├── subgoal_planner.py         [40]    潜在子目标规划
├── world_model_loss.py       [174]    KL + Physics + 综合损失
└── imagination_engine.py     [224]    32 步想象展开引擎

总计: 3,744 行 Python
```

### 模块依赖图

```
                          ┌─────────────────────┐
                          │  hybrid_vla_v2.py    │ ← 无 WM 导入 (断裂!)
                          │  (VLA 主模型)         │
                          └──────────┬───────────┘
                                     │ 使用
                          ┌──────────▼───────────┐
                          │  mamba_core.py        │
                          │  MambaBlock           │◄──────────────────────┐
                          └──────────────────────┘                       │ 复用
                                                                         │
┌────────────────────────────────────────────────────────────────────────┤
│ world_model/                                                           │
│  imagination_engine.py ─┬─► imagination_mamba.py ──► MambaBlock ───────┘
│                         ├─► stochastic_state.py
│                         ├─► noise_augmentation.py
│                         ├─► object_physics.py
│                         └─► world_model_heads.py
│
│  world_model_loss.py ────► world_model_heads.py (SymlogTwoHot)
│  visual_decoder.py       (独立, 未被 engine 引用)
│  subgoal_planner.py      (独立, 未被 engine 引用)
└────────────────────────────────────────────────────────────────────────
```

---

## 2. 逐模块审计

### 2.1 StochasticStateModule ✅ 设计合理

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 48×48 categorical 维度 | ✅ | z_dim = 2304, z_full = z_det(2048) + z_proj(z_sto)(2048) = 4096 |
| Straight-through gradient | ✅ | `sample + probs - probs.detach()` 正确实现 |
| 1% uniform mix | ✅ | `(1-0.01)*softmax + 0.01/48` 防止坍塌 |
| Posterior/Prior 分离 | ✅ | posterior 接收 `[z_det, obs]`; prior 仅接收 `z_det` |
| z_proj 投影维度 | ✅ | `Linear(2304, 2048)` 将离散码映射回 d_model |

### 2.2 ImaginationMamba ⚠️ 状态传递存在问题

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 独立参数 | ✅ | 不与 VLA TriRateMambaCore 共享权重 |
| 8L Mamba-2 + d_state=128 | ✅ | 复用 `MambaBlock` |
| 输入投影维度 | ✅ | `[z_full(4096) + a_emb(2048) + noise(2048)]` = 8192 → proj → 2048 |
| 残差预测 | ✅ | `next_z_det = z_det + delta_z` |
| **单 token 处理 + 状态传递** | 🔴 | 见下文 §3.1 |

### 2.3 ObjectPhysicsEngine ✅ 设计合理, 一个小问题

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 6L GNN + MHA | ✅ | `nn.MultiheadAttention` + FFN + 残差 |
| Intrinsic/Extrinsic 分离 | ✅ | `d_node//2` 维度分别投影 |
| 残差输出 (惯性) | ✅ | `next_slots = object_slots + slot_output(h)` |
| 交互图从 attn weights | ✅ | 最后一层 GNN 的 attention weights |
| `next_intrinsic` 重计算 | ⚠️ | `slot_encoder` 对 `next_slots` 重新编码后取 intrinsic，增加 ~20% 计算 |

### 2.4 WorldModelHeads ✅ 设计合理

| 检查项 | 状态 | 说明 |
|--------|------|------|
| SymlogTwoHot 编码 | ✅ | `symlog/symexp/twohot_encode/loss` 完整实现 |
| 255 bins [-20, 20] | ✅ | `reward_bins_t` 和 `value_bins_t` 注册为 buffer |
| Done head (Bernoulli) | ✅ | 输出 raw logit, 外部 sigmoid |
| decode 方法 | ✅ | `softmax → weighted sum → symexp` |

### 2.5 NoiseAugmentation ✅ 设计合理

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 线性噪声递增 | ✅ | `sigma = max_sigma * step/total` |
| 16 bucket embedding | ✅ | `Embedding(16, 512) → MLP → d_model` |
| 推理时零噪声 | ✅ | 返回原始 z + zero-bucket embedding |
| embed_dim = z_dim//2 = 2048 | ✅ | 与 ImaginationMamba 的 noise_embed 输入匹配 |

### 2.6 CNNWorldDecoder ✅ 设计合理

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 4 级 ConvTranspose2d | ✅ | 7→14→28→56→112 |
| GroupNorm + SiLU | ✅ | 标准做法 |
| LPIPS lazy init | ✅ | 仅在 lpips 包存在时使用 |
| Target 下采样 | ✅ | 224→112 bilinear |

### 2.7 LatentSubgoalPlanner ⚠️ 功能完整但孤立

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 残差 MLP | ✅ | `z_goal = z_full + net(...)` |
| 输入: z_full + phase + language | ✅ | 6144 → 2048 → 1536 → 4096 |
| **未被 ImaginationEngine 引用** | ⚠️ | 定义了但没有调用方 |

### 2.8 WorldModelLoss ⚠️ 多个问题

| 检查项 | 状态 | 说明 |
|--------|------|------|
| KL balancing (α=0.8) | ✅ | DreamerV3 公式正确 |
| **KL free_bits 粒度错误** | 🔴 | 见 §3.3 |
| Latent prediction L2 | ✅ | MSE(z_pred, z_true) |
| Reward symlog two-hot | ✅ | 引用 SymlogTwoHot.loss |
| Done BCE | ✅ | `binary_cross_entropy_with_logits` |
| **slot_pred 与 slot_smoothness 重复** | 🟡 | 见 §3.4 |
| **缺少 visual decoder loss** | 🟡 | 见 §3.5 |

### 2.9 ImaginationEngine ⚠️ 核心集成, 多个问题

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 32 步展开循环 | ✅ | `for t in range(self.horizon)` |
| Gradient checkpoint | ✅ | 每 8 步, `use_reentrant=False` |
| Policy no_grad | ✅ | `with torch.no_grad(): action = policy(z_full)` |
| **ImaginationMamba 状态丢失** | 🔴 | 见 §3.1 |
| **Trajectory 缺少 loss 所需数据** | 🔴 | 见 §3.2 |
| **Physics 输出被丢弃** | 🟡 | 见 §3.6 |

---

## 3. 问题详述

### 3.1 🔴 CRITICAL: ImaginationMamba 跨步 SSM 状态丢失 (v0.3 回归遗传)

**根因**: 与 v0.3 分析中 Tri-Rate Core 的问题完全相同 — `MambaBlock._forward_official` 返回 `(out, None, None)`, SSM state 不在步间传递。

**ImaginationMamba 的具体影响**:

```python
# imagination_mamba.py:102-113
for i, layer in enumerate(self.layers):
    s_i = ssm_states[i] if ssm_states else None  # 第二步起: ssm_states=[None]*8 → s_i=None
    c_i = conv_states[i] if conv_states else None
    h, s, c = layer(h, s_i, c_i)                 # official: (out, None, None)
    new_ssm.append(s)                              # appends None
    new_conv.append(c)                             # appends None

return delta_z, new_ssm or None, new_conv or None
# new_ssm = [None]*8 → truthy → 返回 [None]*8
```

```python
# imagination_engine.py:126-128
delta_z, new_ssm, new_conv = self.dynamics(
    z_noisy, action, noise_emb, ssm_states, conv_states
)
# new_ssm = [None]*8, 传入下一步作为 ssm_states
# 下一步: ssm_states[i] = None → 零状态
```

**影响**: ImaginationMamba 的 8 层 Mamba-2 在 32 步想象展开中，每步都从零 SSM 状态开始。它退化为一个**无记忆的前馈网络**:

```
step 0: delta_z = f(z_full_0, action_0, noise_0)    ← 仅依赖当前输入
step 1: delta_z = f(z_full_1, action_1, noise_1)    ← 不知道 step 0 发生了什么
...
step 31: delta_z = f(z_full_31, action_31, noise_31) ← 完全无历史
```

世界模型的核心价值 — 通过 SSM 递推捕获环境动态 (碰撞后物体继续运动, 夹紧后夹爪保持闭合) — **完全丧失**。

**注意**: Fallback 路径 (`_use_official=False`) 正确传递状态。所以在无 `mamba_ssm` 的环境中此问题不存在。

### 3.2 🔴 CRITICAL: ImaginationTrajectory 缺少 loss 计算所需的全部数据

**WorldModelLoss 需要的输入**:

| 参数 | 来源 | ImaginationTrajectory 中是否有 |
|------|------|-------------------------------|
| `posterior_logits` | StochasticState.encode_posterior | ❌ **训练时未调用 posterior** |
| `prior_logits` | StochasticState.encode_prior | ❌ **rollout 中调用了但未保存** |
| `z_pred` | ImaginationMamba output | ❌ **不在 trajectory** |
| `z_true` | 真实观测编码 | ❌ **来自 VLA, 不在 WM 内部** |
| `reward_logits` | WorldModelHeads.forward | ❌ **decode 后丢失 logits** |
| `reward_target` | 真实环境 reward | ❌ **来自数据** |
| `reward_bins` | WorldModelHeads.reward_bins_t | ✅ 可从 heads 获取 |
| `done_logit` | WorldModelHeads.forward | ❌ **sigmoid 后丢失 logit** |
| `done_target` | 真实环境 done | ❌ **来自数据** |
| `pred_slots` | ObjectPhysicsEngine | ⚠️ 仅 next_slots 被保存 |
| `target_slots` | 真实 grounder 输出 | ❌ **来自 VLA** |
| `intrinsic` | ObjectPhysicsEngine | ❌ **被丢弃** |
| `next_intrinsic` | ObjectPhysicsEngine | ❌ **被丢弃** |
| `interaction_weights` | ObjectPhysicsEngine | ❌ **被丢弃** |

**具体代码问题**:

```python
# imagination_engine.py:136-138 — 解码了 logits, 丢失了原始 logits
reward = self.heads.decode_reward(head_out["reward_logits"])  # scalar, 不可逆
value = self.heads.decode_value(head_out["value_logits"])     # scalar, 不可逆
done = torch.sigmoid(head_out["done_logit"].squeeze(-1))      # sigmoid 丢失 logit

# imagination_engine.py:206-208 — Physics 输出只保留 next_slots
next_slots, _, _, _ = self.physics(slots, action, z_full)
#               ↑ interaction_weights, intrinsic, next_intrinsic 全部丢弃!
```

**结果**: `ImaginationTrajectory` 无法提供 `WorldModelLoss.forward()` 所需的任何关键输入。WM loss 无法计算。

### 3.3 🔴 KL free_bits 粒度错误

**DreamerV3 原始实现**:
```python
# 每个 category 独立限制最小 KL
kl_per_cat = kl_divergence(post_dist, pri_dist)  # [B, n_cat]
kl_clamped = torch.clamp(kl_per_cat, min=free_bits)  # per-category free bits
kl_loss = kl_clamped.sum(-1).mean()  # sum categories, mean batch
```

**当前实现** (`world_model_loss.py:55-60`):
```python
kl = alpha * kl_post_to_pri.sum(-1).mean() + (1-alpha) * kl_pri_to_post.sum(-1).mean()
return torch.clamp(kl, min=self.free_bits)  # ← free_bits 应用于总 KL!
```

**差异**: free_bits=1.0 应该意味着"每个 category 至少贡献 1 nat"。48 个 categories 的总 KL 最小值应为 48 nats。但当前实现对总 KL (已经是 ~48× 单 category KL) 应用 clamp, 阈值 1.0 几乎永远不会被触发。

**修复**:
```python
kl_per_cat = kl_divergence(post_dist, pri_dist)  # [B, n_cat]
kl_clamped = torch.clamp(kl_per_cat, min=self.free_bits)
kl = self.alpha * kl_clamped.sum(-1).mean() + ...
```

### 3.4 🟡 PhysicsLoss: slot_pred 与 slot_smoothness 完全相同

```python
losses["slot_pred"] = F.mse_loss(pred_slots, target_slots)     # = ((p-t)^2).mean()
losses["slot_smoothness"] = (pred_slots - target_slots).pow(2).mean()  # 完全相同
```

两者都是 MSE, 在 WorldModelLoss 中分别乘以 `w_slot` 和没有单独权重 (slot_smoothness 未在 WorldModelLoss 中引用)。实际上 `slot_smoothness` 是死代码 — 它在 PhysicsLoss 内部计算但 WorldModelLoss 没有取用它。

### 3.5 🟡 Visual Decoder 与 Loss 系统断裂

- `CNNWorldDecoder` 和 `WorldDecoderLoss` 定义在 `visual_decoder.py`
- `ImaginationEngine` 不包含 `CNNWorldDecoder`
- `WorldModelLoss` 不包含 `WorldDecoderLoss`
- 没有任何代码调用 visual decoder 或计算 visual loss

视觉重建 (L1 图像级) 是世界模型训练的重要信号, 但完全未集成。

### 3.6 🟡 Physics 输出在 rollout 中被丢弃

```python
# imagination_engine.py:206-208
next_slots, _, _, _ = self.physics(slots, action, z_full)
#           interaction_weights → 丢弃
#           intrinsic → 丢弃
#           next_intrinsic → 丢弃
```

PhysicsLoss 需要 `intrinsic`, `next_intrinsic`, `interaction_weights` 来计算 intrinsic 不变性和交互稀疏性正则。丢弃这些意味着这两个 loss 项在想象展开中无法计算。

### 3.7 🟡 VLA ↔ World Model 零集成

| 集成点 | 需求 | 当前状态 |
|--------|------|---------|
| z_det 提取 | WM 需要 VLA fused_state 作为初始 z_det | ❌ 无接口 |
| Posterior encoding | WM 训练需要 backbone 输出作为 obs_encoding | ❌ 无接口 |
| WM loss 加入训练 | WM loss 应加入 forward_train 的 loss_total | ❌ hybrid_vla_v2.py 无 WM 导入 |
| Actor-critic | imagination rollout 需要 policy (FAST head?) | ❌ 未定义 policy 接口 |
| Subgoal feedback | SubgoalPlanner 输出应影响 Expert 的 cond_prefix | ❌ SubgoalPlanner 未连接 |
| WorldModelConfig | 需要配置 WM 超参数 | ❌ config.py 无 WM 配置 |

---

## 4. 数据流分析: 世界模型训练需要什么

### 4.1 世界模型训练 forward pass (未实现, 预期设计)

```
Step 1: VLA 前向 → 获取表征
  backbone(image) → grounder → compressed_slots [B, 24, 2048]
  temporal_core → fused_state [B, 2048]  ← 这就是 z_det

Step 2: 编码 posterior (有观测)
  obs_encoding = fused_state (或 backbone hidden state)
  z_full_post, post_probs, post_logits = stochastic.encode_posterior(z_det, obs_encoding)

Step 3: 编码 prior (无观测)
  z_full_prior, prior_probs, prior_logits = stochastic.encode_prior(z_det)

Step 4: 一步世界模型前向
  delta_z = imagination_mamba(z_full_post, action, noise_emb)
  z_det_next = z_det + delta_z
  z_full_next_prior = stochastic.encode_prior(z_det_next)

Step 5: 计算 losses
  KL(post || prior)
  MSE(z_det_next, z_det_target)       ← z_det_target 来自下一步 VLA fused_state
  Reward(heads(z_full_post), reward)
  Done(heads(z_full_post), done)
  Visual(decoder(z_full_post), image)
  Physics(physics(slots, action), next_slots)
```

### 4.2 Actor-Critic 想象训练 (未实现, 预期设计)

```
Step 1: 从 VLA 获取初始状态
  z_det_init = VLA.temporal_core 的 fused_state (detached)

Step 2: 32 步想象展开
  trajectory = imagination_engine.rollout(z_det_init, policy)
  其中 policy 应该是 VLA 的 FAST head 或 Expert

Step 3: 计算 actor-critic 目标
  returns = lambda_return(trajectory.rewards, trajectory.values, trajectory.dones)
  actor_loss = -returns.mean()
  critic_loss = symlog_twohot(trajectory.value_logits, returns)

Step 4: 更新
  actor_loss → 更新 policy
  critic_loss → 更新 value head
  WM 参数冻结 (不从 actor-critic loss 更新)
```

---

## 5. 改进方案

### 5.1 P0: 修复 ImaginationMamba 状态传递

与 VLA Core 的修复方案相同, 但更简单 (单 token 处理):

```python
# imagination_mamba.py — 方案 A: 强制 fallback
class ImaginationMamba(nn.Module):
    def __init__(self, ...):
        # 对 ImaginationMamba 禁用官方 Mamba2, 使用 fallback + fp32 scan
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])
        # 强制每层使用 fallback
        for layer in self.layers:
            if hasattr(layer, '_use_official'):
                layer._use_official = False
```

**方案 B (更好): 为 MambaBlock 添加 step() 模式**

在 `mamba_core.py` 的 `MambaBlock` 中添加:
```python
def step(self, x, ssm_state, conv_state):
    """Single-token step with explicit state management."""
    if self._use_official:
        # 使用 Mamba2 的 step() 方法
        out, new_state = self.mamba.step(x, ssm_state_dict)
        return out, new_state.ssm_state, new_state.conv_state
    return self._forward_fallback(x, ssm_state, conv_state)
```

### 5.2 P0: 扩展 ImaginationTrajectory 包含 loss 所需数据

```python
@dataclass
class ImaginationTrajectory:
    states: Tensor              # [B, H, z_dim]
    rewards: Tensor             # [B, H]
    values: Tensor              # [B, H]
    dones: Tensor               # [B, H]
    actions: Tensor             # [B, H, action_dim]
    slots: Optional[Tensor] = None

    # ---- 新增: loss 计算所需 ----
    reward_logits: Optional[Tensor] = None    # [B, H, 255]
    value_logits: Optional[Tensor] = None     # [B, H, 255]
    done_logits: Optional[Tensor] = None      # [B, H, 1]
    prior_logits: Optional[Tensor] = None     # [B, H, 2304]
    intrinsic: Optional[Tensor] = None        # [B, H, S, d_node//2]
    next_intrinsic: Optional[Tensor] = None   # [B, H, S, d_node//2]
    interaction_weights: Optional[Tensor] = None  # [B, H, S, S]
```

同时修改 `_single_step` 返回 raw logits:
```python
def _single_step(self, ...):
    ...
    head_out = self.heads(z_full_next)
    return (z_det_next, z_full_next,
            head_out["reward_logits"],
            head_out["value_logits"],
            head_out["done_logit"],
            prior_logits,               # 新增
            new_ssm, new_conv)
```

### 5.3 P0: 添加 VLA ↔ WM 集成接口

在 `hybrid_vla_v2.py` 中添加:

```python
def get_world_model_state(self, grounder_out, temporal_out):
    """Extract z_det and obs_encoding for world model training."""
    z_det = temporal_out.fused_state              # [B, 2048]
    obs_encoding = temporal_out.fused_state       # 或用 grounder_out.global_token
    slots = grounder_out.compressed_object_slots   # [B, 24, 2048]
    return z_det, obs_encoding, slots
```

### 5.4 P0: 添加 WorldModelConfig

```python
@dataclass
class WorldModelConfig:
    d_model: int = 2048
    z_dim: int = 4096
    n_categories: int = 48
    n_classes: int = 48
    imagination_layers: int = 8
    imagination_d_state: int = 128
    num_slots: int = 24
    gnn_layers: int = 6
    d_node: int = 512
    horizon: int = 32
    checkpoint_every: int = 8
    max_noise_sigma: float = 0.7
    noise_buckets: int = 16
    reward_bins: int = 255
    value_bins: int = 255
    kl_free_bits: float = 1.0
    kl_alpha: float = 0.8
    enable_visual_decoder: bool = True
    enable_subgoal_planner: bool = True
```

### 5.5 P1: 修复 KL free_bits 粒度

```python
class KLLoss(nn.Module):
    def forward(self, posterior_logits, prior_logits, n_categories, n_classes):
        post = posterior_logits.view(-1, n_categories, n_classes)
        pri = prior_logits.view(-1, n_categories, n_classes)
        post_dist = torch.distributions.Categorical(logits=post)
        pri_dist = torch.distributions.Categorical(logits=pri)

        kl_fwd = torch.distributions.kl_divergence(post_dist, pri_dist)  # [B, n_cat]
        kl_rev = torch.distributions.kl_divergence(pri_dist, post_dist)

        # Per-category free bits (DreamerV3 正确做法)
        kl_fwd_clamped = torch.clamp(kl_fwd, min=self.free_bits)
        kl_rev_clamped = torch.clamp(kl_rev, min=self.free_bits)

        kl = (self.alpha * kl_fwd_clamped.sum(-1).mean()
              + (1 - self.alpha) * kl_rev_clamped.sum(-1).mean())
        return kl
```

### 5.6 P1: 集成 Visual Decoder 到 Engine + Loss

在 `ImaginationEngine.__init__` 中添加:
```python
self.visual_decoder = CNNWorldDecoder(z_dim=self.z_dim)
```

在 rollout 或单独的训练 forward 中:
```python
pred_image = self.visual_decoder(z_full)
# 作为额外 loss 信号
```

### 5.7 P2: 移除 PhysicsLoss 中的重复项

删除 `slot_smoothness` (与 `slot_pred` 完全相同):
```python
class PhysicsLoss(nn.Module):
    def forward(self, ...):
        losses = {}
        losses["slot_pred"] = F.mse_loss(pred_slots, target_slots)
        losses["intrinsic_invariance"] = F.mse_loss(intrinsic, next_intrinsic.detach())
        if interaction_weights is not None:
            losses["interaction_sparsity"] = interaction_weights.mean()
        return losses
```

---

## 6. 总结

### 6.1 世界模型代码质量

| 维度 | 评分 | 说明 |
|------|------|------|
| 模块设计 | 8/10 | DreamerV3 + GameNGen + GNN 物理, 方案合理 |
| 代码质量 | 7/10 | 清晰、文档好, 但有重复 loss 和遗漏 |
| 张量维度一致性 | 9/10 | 所有模块维度匹配 (4096 z_full / 2048 d_model) |
| **与 VLA 集成** | **1/10** | 零集成, 无导入, 无配置, 无数据流接口 |
| **与 WM Loss 集成** | **2/10** | Trajectory 缺少 loss 所需数据 |
| **SSM 状态传递** | **2/10** | 继承 v0.3 回归, ImaginationMamba 无记忆 |

### 6.2 优先级排序

| 优先级 | 问题 | 工作量 |
|--------|------|--------|
| **P0** | ImaginationMamba SSM 状态修复 (与 VLA Core 同源) | 1-2 天 |
| **P0** | ImaginationTrajectory 扩展 (保存 logits + physics 输出) | 0.5 天 |
| **P0** | VLA ↔ WM 集成接口 | 1 天 |
| **P0** | WorldModelConfig 添加 | 0.25 天 |
| **P1** | KL free_bits 粒度修复 | 0.25 天 |
| **P1** | Visual Decoder 集成到 Engine + Loss | 0.5 天 |
| **P1** | WM training loop (独立于 VLA, 可先验证) | 2-3 天 |
| **P2** | PhysicsLoss 重复项清理 | 0.1 天 |
| **P2** | SubgoalPlanner 接入 | 0.5 天 |

### 6.3 当前整体状态

```
VLA 模型定义:   85%  (Core Mamba 状态问题)
世界模型定义:   70%  (模块完整, 集成缺失)
VLA ↔ WM 集成:  5%  (仅 MambaBlock 复用)
训练 Pipeline:   0%  (完全缺失)
```

**最关键的单一修复**: 为 `MambaBlock` 添加 `step()` 模式支持显式状态传递。这一个修复同时解决 VLA Tri-Rate Core (v0.3 发现) 和 ImaginationMamba (本轮发现) 两个问题。

---

*分析完毕。世界模型 9 个模块的单模块设计质量良好 (DreamerV3 + GameNGen + GNN 范式), 但存在三个系统级问题: (1) ImaginationMamba 继承了 MambaBlock 的跨步状态丢失回归; (2) ImaginationTrajectory 不返回 loss 计算所需的 raw logits 和 physics 输出; (3) VLA 与 World Model 之间零集成 (无导入, 无配置, 无数据流接口)。所有三个问题的根源可追溯到缺少一个统一的 MambaBlock.step() 接口和一个跨 VLA/WM 的顶层 orchestrator。*
