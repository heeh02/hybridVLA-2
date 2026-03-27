# HybridVLA v2 第三轮分析 (v0.3 — 基于 recovery_v0_2 的修正验证)

> 分析日期: 2026-03-25
> 输入: `docs/recovery_v0_2.md` + 更新后全部代码
> 方法: 逐文件 diff 审计 → 问题对照 → 新回归检测

---

## 1. v0.2 问题修复对照表

| # | v0.2 问题 | 修复状态 | 修复方式 | 验证结论 |
|---|----------|---------|---------|---------|
| 1 | **bf16 SSM 精度灾难** | ✅ **已解决 (官方路径)** | Core Mamba → `mamba_ssm.Mamba2` (fp32 state 内部) ; Expert Mamba → `selective_scan_fn` (fp32 state 内部) | JIT fallback 仍有此问题，但 fallback 仅用于无 CUDA 的 debug 场景，可接受 |
| 2 | **Grounder 重复调用 R 次** | ❌ **未修复** | `hybrid_vla_v2.py:240-249` 未变更 | 仍浪费 ~120ms/step (~23% forward) |
| 3 | **ActionConsistencyLoss 梯度断裂** | ❌ **未修复** | `consistency_loss.py` 未变更 | argmax 仍切断梯度，仅训练 7,680 参数 |
| 4 | **Loss scale 不平衡** | ❌ **未修复** | loss_weights 未调整 | loss_fast/loss_fm ≈ 2.5× 初期 |
| 5 | **Training pipeline 缺失** | ❌ **未修复** | data/infer/utils/ 仍为空 | 无法训练 |
| 6 | **EMA 未实现** | ❌ **未修复** | 无 EMA 代码 | Stage B/C 缺少 EMA |
| 7 | **Multi-camera 未实现** | ❌ **未修复** | 骨干仍处理单组输入 | — |
| 8 | **RTC/FASTER 未实现** | ❌ **未修复** | Stage C 空配置 | — |
| 9 | **flash_attn fallback** | ❌ **未修复** | `attn_implementation` 硬编码 `"flash_attention_2"` | 无 flash_attn 时 backbone 加载崩溃 |
| 10 | **Exposure bias (action history)** | ❌ **未修复** | 训练用 GT, 推理用预测 | — |

**小结**: recovery_v0_2 的核心贡献是集成官方 `mamba_ssm.Mamba2`，解决了 bf16 精度和性能两个关键问题。但 v0.2 报告中的 10 个问题中 **仅 1 个被实质修复**，其余 9 个原样遗留。

---

## 2. 🔴 新引入的关键回归: 跨时间步 SSM 状态丢失

### 2.1 问题定位

**文件**: `mamba_core.py:170-185`

```python
# _forward_official 路径:
def _forward_official(self, x: Tensor) -> Tuple[Tensor, None, None]:
    out = self.mamba(x)     # Mamba2 处理 L=33 tokens
    return out, None, None  # ← 返回 None 状态!
```

**文件**: `mamba_core.py:298-328`

```python
# _MambaStack.forward:
uses_official = self.layers[0]._use_official
if not uses_official:
    if ssm_states is None or conv_states is None:
        ssm_states, conv_states = self.init_states(...)  # 仅 fallback 初始化

# ...
return (x, new_ssm if new_ssm else None, new_conv if new_conv else None)
# 官方路径: new_ssm = [], new_conv = [] → 返回 (x, None, None)
```

**文件**: `mamba_core.py:616-632` (TriRateMambaCore.forward)

```python
next_state = TriRateTemporalState(
    fast_ssm_states=fast_new_ssm,    # None (官方路径)
    fast_conv_states=fast_new_conv,   # None (官方路径)
    ...
)
```

### 2.2 影响分析

训练的 temporal loop (`hybrid_vla_v2.py:273-314`):

```
时间步 t=0:
  state = TriRateTemporalState()  # 初始
  fast_mamba(input, state.fast_ssm_states=None)
  → Mamba2 处理 33 tokens, 输出 (out, None, None)
  → next_state.fast_ssm_states = None

时间步 t=1:
  fast_mamba(input, state.fast_ssm_states=None)  # 又是 None!
  → Mamba2 从零开始处理 33 tokens, 完全不知道 t=0 的信息
  → 等效于独立窗口处理, 丧失全部跨步递推!

时间步 t=2 .. t=23: 同上, 每步独立
```

**对比 fallback 路径**:
```
时间步 t=0: ssm_state = zeros → 处理 33 tokens → ssm_state_new (携带 t=0 信息)
时间步 t=1: ssm_state = ssm_state_new → 处理 33 tokens → ssm_state_new' (携带 t=0+t=1 信息)
时间步 t=23: ssm_state 携带全部 0-22 步的信息
```

### 2.3 严重度评估

**严重度: 🔴 CRITICAL**

Tri-Rate Mamba Core 的核心设计假设是**跨时间步递推**:
- Fast Mamba (50 Hz): 每步更新，携带连续的动作-感知状态
- Slow Mamba (12.5 Hz): 语义刷新时更新，携带长程语义记忆

如果官方 Mamba2 路径不传递状态:
1. **Fast stream 退化为窗口处理器**: 每步只看当前 33 tokens，丧失 "我之前做了什么动作" 的递推记忆
2. **Slow stream 完全无用**: 它只在语义刷新时运行，依赖 SSM 状态在非刷新步保持信息。状态为 None 时，`last_slow_token` 是缓存的，但 SSM 内部状态无法累积
3. **Medium stream 部分受损**: 类似 Fast，但 Medium 还依赖每 2 步的状态传递

**唯一的时间信息来源退化为**: 输入中的 action_history_token + prev_action_token + stale_token。这些提供了**显式**时间信息，但远不如 SSM **隐式**状态丰富。

### 2.4 根因

recovery_v0_2.md 注释:
> "官方 Mamba2 内部管理状态，stack 层检测 `_use_official` 后决定是否传递 cache"

这是一个**错误假设**。`Mamba2.forward()` 管理的是**单次调用内** L=33 tokens 之间的 SSM 递推，**不是跨多次调用**的状态持久化。每次 `self.mamba(x)` 调用都从零状态开始。

`Mamba2` 确实有 `inference_params` 参数支持 step-by-step 推理（类似 KV cache），但 `forward()` 模式不使用它。

### 2.5 修复方案

**方案 A: 回退 Core Mamba 到 fallback + fp32 scan (简单但慢)**

将 `_MambaStack` 中 `uses_official` 强制为 False，配合 fp32 state accumulation。性能退回 JIT 速度。

**方案 B: 使用 Mamba2.step() 模式 (推荐)**

Mamba2 支持 `step()` 方法进行单步推理并显式传递/返回 state：

```python
# 每层维护 inference_params (conv_state + ssm_state)
inference_params = InferenceParams(max_seqlen=33, max_batch_size=B)

# 逐 token 调用 step():
for token_idx in range(33):
    out = self.mamba.step(x[:, token_idx], inference_params)

# 或用 allocate_inference_cache + forward:
self.mamba.allocate_inference_cache(B, max_seqlen=33)
out = self.mamba(x, inference_params=inference_params)
# inference_params 现在包含了最终 SSM state
```

但这需要重构 `_MambaStack` 以管理 `InferenceParams` 而非裸 ssm_state tensors。

**方案 C: 混合模式 — forward() 处理序列 + 手动提取末尾 state (最佳)**

在 `_forward_official` 中使用 Mamba2 的 `forward()` 处理完 33 tokens 后，从 Mamba2 内部提取最终 SSM 状态并返回：

```python
def _forward_official(self, x):
    # 使用带 state 返回的 forward
    out = self.mamba(x)

    # 提取内部 SSM state (如果 Mamba2 支持)
    # Mamba2 的 ssm_state 存储在 self.mamba.ssm_state 或类似属性中
    # 具体取决于 mamba_ssm 版本

    return out, extracted_ssm_state, extracted_conv_state
```

> **注意**: 这依赖于 `mamba_ssm` 的内部 API 稳定性。需要检查具体版本。

---

## 3. 其他新发现

### 3.1 Mamba2 vs Mamba1 架构差异

recovery 使用 `Mamba2` (not `Mamba`)。两者有结构差异:

| 方面 | Mamba1 | Mamba2 |
|------|--------|--------|
| 核心算法 | Selective Scan (线性递推) | Structured State-Space Duality (SSD) |
| 头数 | 无 (single head) | 多头 (nheads = d_model // headdim) |
| 效率 | O(BLD N) | O(BLD + BLN²) — 更快 |
| 参数 | A_log, D, dt_proj | 类似但内部重组 |

设计文档和 config 中的 `d_state=128/256` 参数名来自 Mamba1。Mamba2 的 `d_state` 语义略不同 (与 headdim 耦合)。虽然构造函数接受相同参数，实际行为可能不一致。

**影响**: 参数量可能与设计文档估算有偏差。Mamba2 通常比 Mamba1 参数更少 (因为 SSD 更高效)。

### 3.2 Fallback 路径的 activation_checkpoint 与官方路径不对称

```python
# _MambaStack.forward:
if use_checkpoint and self.training:
    x, s, c = activation_checkpoint(layer, x, s_i, c_i, use_reentrant=False)
```

当 `_use_official=True` 时，`layer(x, None, None)` → `_forward_official(x)` 忽略 s_i, c_i。`activation_checkpoint` 对 `_forward_official` 仍然有效 (对 Mamba2 内部计算做 checkpointing)。但 Mamba2 的 CUDA kernel 可能不兼容 `torch.utils.checkpoint` (重计算时 kernel 行为可能不确定)。

**建议**: 测试 `use_checkpoint=True` + `_use_official=True` 的组合是否产生正确梯度。

### 3.3 `_forward_official` 缺少 input norm

Fallback 路径有 `self.norm(x)` (pre-norm):
```python
def _forward_fallback(self, x, ...):
    residual = x
    x = self.norm(x)  # ← 显式 LayerNorm
    ...
```

官方路径:
```python
def _forward_official(self, x):
    out = self.mamba(x)  # Mamba2 内部有自己的 norm
    ...
```

Mamba2 **内部包含 LayerNorm** (`self.norm = RMSNorm(d_model)`)。所以 norm 是存在的，只是封装在 Mamba2 内部。✅ 无问题。

但注意: Mamba2 用的是 **RMSNorm**，而 fallback 用的是 **LayerNorm**。两者有细微数学差异 (RMSNorm 无 mean-centering)。对于预训练模型这无关紧要 (参数独立)，但如果想在官方/fallback 之间迁移 checkpoint，norm 类型不同会导致不兼容。

---

## 4. 当前代码完整度评分

### 4.1 模块完整度

| 模块 | 代码存在 | 功能正确 | 性能优化 | 评分 |
|------|----------|---------|---------|------|
| Backbone (Qwen2-VL-7B) | ✅ | ✅ | ✅ (flash_attn) | 9/10 |
| MultiScaleAdapter | ✅ | ✅ (gate 已修复) | ✅ | 9/10 |
| Grounder (层次化压缩) | ✅ | ✅ (mid-layer 已修复) | ✅ (SDPA) | 9/10 |
| **Core Mamba (Tri-Rate)** | ✅ | **🔴 跨步状态丢失** | ✅ (Mamba2 CUDA) | **3/10** |
| Expert (18L Flow) | ✅ | ✅ | ✅ (SDPA + CUDA SSM) | 9/10 |
| Discrete Heads | ✅ | ✅ | — | 8/10 |
| Losses | ✅ | ⚠️ (ActionConsistency 梯度断裂) | — | 6/10 |
| Training Pipeline | ❌ | — | — | 0/10 |
| Data Pipeline | ❌ | — | — | 0/10 |
| EMA | ❌ | — | — | 0/10 |
| Inference Runtime | ❌ | — | — | 0/10 |

### 4.2 总体就绪度

```
模型定义完整度:   85%  (核心 Mamba 状态问题降分)
训练就绪度:       10%  (forward_train 可用, 但无 pipeline)
推理就绪度:       30%  (control_step 结构完整, 但 Mamba 状态问题同样影响)
```

---

## 5. 遗留 + 新增问题优先级排序

### P0 — 不修无法正确训练

| # | 问题 | 来源 | 工作量 |
|---|------|------|--------|
| **1** | **Core Mamba 跨步 SSM 状态丢失** | v0.3 新发现 | 1-2 天 |
| 2 | Training loop + Data pipeline | v0.2 遗留 | 3-5 天 |
| 3 | FSDP + AMP 集成 | v0.2 遗留 | 1-2 天 |
| 4 | EMA 实现 | v0.2 遗留 | 0.5 天 |
| 5 | flash_attn fallback | v0.2 遗留 | 0.25 天 |

### P1 — 显著影响训练质量

| # | 问题 | 来源 | 工作量 |
|---|------|------|--------|
| 6 | Grounder 重复调用 (23% waste) | v0.2 遗留 | 0.1 天 |
| 7 | ActionConsistencyLoss 梯度修复 | v0.2 遗留 | 0.25 天 |
| 8 | 权重初始化 (zero init 残差) | v0.2 遗留 | 0.5 天 |
| 9 | Mamba2 + activation_checkpoint 兼容性验证 | v0.3 新发现 | 0.5 天 |
| 10 | Batch shape assertion | v0.2 遗留 | 0.1 天 |

### P2 — 改善但非阻塞

| # | 问题 | 来源 | 工作量 |
|---|------|------|--------|
| 11 | Loss scale 平衡 | v0.2 遗留 | 0.5 天 |
| 12 | Scheduled sampling (exposure bias) | v0.2 遗留 | 0.5 天 |
| 13 | Multi-camera 支持 | v0.1 遗留 | 1-2 天 |
| 14 | RTC / FASTER | v0.1 遗留 | 2-3 天 |
| 15 | Mamba1/Mamba2 参数量审计 | v0.3 新发现 | 0.25 天 |

---

## 6. 总结

### 6.1 recovery_v0_2 的正面贡献

1. **官方 mamba_ssm 集成**: 架构正确、降级完善、导入检测健壮
2. **bf16 精度**: CUDA 路径使用 fp32 state accumulation，问题实质解决
3. **性能**: Mamba2 fused kernel 5-10× 加速 Mamba 层
4. **ActionHistoryEncoder**: 从继承改为组合，更清晰

### 6.2 需要紧急修复

**跨时间步 SSM 状态丢失是当前代码的最严重问题**。它使 Tri-Rate 设计的核心价值 (Fast/Medium/Slow 的递推时间建模) 在 CUDA 路径下完全失效。该问题在 fallback 路径中不存在 (fallback 正确传递 state)，因此:

- **临时解决**: 强制 `_use_official = False` 直到状态管理问题解决
- **长期解决**: 实现 Mamba2 的 `inference_params` 状态管理，或使用 `step()` 模式

### 6.3 距离可训练状态的路径

```
当前位置 ──┬── P0 #1: 修复 Mamba 状态   (1-2 天)
            ├── P0 #2: Training loop      (3-5 天, 可并行)
            ├── P0 #3: FSDP + AMP         (1-2 天, 依赖 #2)
            ├── P0 #4: EMA                 (0.5 天)
            └── P0 #5: flash_attn fallback (0.25 天)
            ↓
首次训练 ─── ~7-10 工作日
```

---

*分析完毕。recovery_v0_2 的 Mamba2 集成解决了 bf16 精度和 CUDA 性能两个关键问题，但引入了一个严重回归: 官方 Mamba2 的 `forward()` 不跨调用保持 SSM 状态，导致 Tri-Rate Mamba Core 的 24 个时间步退化为独立窗口处理，丧失跨步递推能力。这是当前最高优先级需修复的问题。其余 v0.2 报告中 9/10 的问题均未修复。*
