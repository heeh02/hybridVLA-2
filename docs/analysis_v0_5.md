# HybridVLA v2 第五轮分析 — 最终版 (v0.5)

> 分析日期: 2026-03-25
> 输入: `docs/recovery_v0_5.md` + 全部代码 (29 Python files + 1 smoke test, ~4,164 行)
> 核心: 验证修复 → 残留问题 → 训练就绪判定

---

## 1. v0.5 修复验证 (5/5 通过)

### 1.1 VLA Core 跨步状态传递 ✅

`_MambaStack.forward()` (line 375-397) 当 `uses_official=True` 时, 使用逐 token `layer.step()` 循环:

```python
for t in range(x.shape[1]):         # L=33 tokens
    x_t = x[:, t, :]
    for i, layer in enumerate(self.layers):   # 20 layers (Fast)
        x_t, ssm_states_list[i], conv_states_list[i] = layer.step(...)
    out[:, t, :] = x_t
return out, ssm_states_list, conv_states_list  # 真实状态!
```

**数学等价性**: 逐 token 处理与序列并行处理对 Mamba SSM/Conv1d 数学等价 — SSM 是因果递推, Conv1d 是因果卷积, 两种遍历顺序产生相同结果。 ✅

**状态返回类型**: `Tuple[Tensor, List[Tensor], List[Tensor]]` — 两条路径 (official + fallback) 现在返回一致的非 None 状态。 ✅

### 1.2 Grounder 单次调用 ✅

```python
# hybrid_vla_v2.py:297-300
single_grounder_out = self.grounder(backbone_hidden)
for _ in range(R):
    grounder_outputs.append(single_grounder_out)  # 引用共享
```

节省 (R-1) 次 Grounder forward (~120ms)。引用共享安全 — 后续代码只读取 GrounderOutput 字段。 ✅

### 1.3 ActionConsistencyLoss 可微化 ✅

```python
# hybrid_vla_v2.py:391-394
fast_probs = fast_logits.softmax(dim=-1)              # 可微
fast_continuous = (fast_probs * bin_centers).sum(-1)    # 期望值
```

梯度链: `loss_consistency → fast_continuous → fast_probs → fast_logits → fast_head → fused_states → temporal core`。 ✅ argmax 断裂已修复。

### 1.4 Expert cond_dim 修复 ✅

```python
# hybrid_vla_v2.py:113
cond_dim=ecfg.d_model,  # = d_expert = 1536 (not d_core=2048)
```

`_build_cond_prefix` 末尾调用 `core_to_expert(cond)` 已经将 2048→1536。Expert 的 `cond_proj` 收到 [B, 32, 1536], 与 `Linear(1536, 1536)` 匹配。实际因 `cond_dim == d_model`, `cond_proj = nn.Identity()`。 ✅

### 1.5 Smoke Test ✅

`scripts/train_smoke_test.py` (200 行): mock backbone + dummy data + mini config → Stage A 20 步 + Stage B 10 步通过, 无 NaN。 ✅

---

## 2. 残留问题与新发现

### 2.1 🟡 性能: 官方路径丧失序列并行性

**位置**: `_MambaStack.forward()` line 375-397

逐 token `step()` 替代了 Mamba2 的序列并行 `forward()`:

```
Fast Mamba (20L, L=33):  33 × 20 = 660 step() 调用/temporal step
Medium Mamba (6L):       33 × 6  = 198
Slow Mamba (10L):        33 × 10 = 330
─────────────────────────────────────────
合计:                    1,188 step() 调用/temporal step
× T=24 temporal steps:   28,512 step() 调用/training step
```

每个 `step()` 是一次 CUDA kernel launch。在 H100 上, kernel launch overhead ~5μs, 28,512 × 5μs ≈ **143ms 纯 overhead** (占训练步的 ~25%)。

**对比原设计**: Mamba2.forward() 每层处理完整 33-token 序列只需 1 次 kernel launch。20 层 × 3 流 × 24 步 = 1,440 launch, overhead ~7ms。

**缓解方案**: 未来可用 Mamba2 的 `inference_params` 批量处理序列并返回最终 state, 恢复序列并行性。当前方案**功能正确, 性能待优化**。

### 2.2 🟡 官方路径未使用 activation_checkpoint

```python
if uses_official:
    # token-by-token loop — 无 checkpoint
    ...
else:
    # fallback path
    if use_checkpoint and self.training:
        x, s, c = activation_checkpoint(layer, x, s_i, c_i, ...)
```

fallback 路径对每层做 activation checkpoint, 官方路径没有。对于 Fast Mamba 20 层 × 33 tokens, 所有中间激活都保留在显存中。

**影响**: 单步显存增加约 20L × B × 33 × d_inner × 2B = 20 × 2 × 33 × 4096 × 2 ≈ 21 MB (bs=2, bf16)。乘以 T=24 步, 最大 ~500 MB。在 H100-80GB 上可接受, 但 batch size 增大时需注意。

### 2.3 🟡 `bin_centers` 每次 forward 重新创建

```python
# hybrid_vla_v2.py:392
bin_centers = torch.linspace(-1, 1, V, device=device)  # 每次 forward 创建
```

应注册为 buffer 避免重复分配:
```python
# __init__ 中:
self.register_buffer("_bin_centers", torch.linspace(-1, 1, V))
```

微小但累积的 GPU 内存分配开销。

### 2.4 🟢 `init_states()` 与官方路径状态形状不一致 (仅影响外部调用)

`_MambaStack.init_states()` 创建 `[B, d_inner, d_state]` 形状的状态, 但 `_step_official()` 分配 `[B, nheads, headdim, d_state]` 形状。当前代码流中两者不会混用 (初始状态始终为 None, 由 step() 内部分配), 但 `init_states()` 对官方路径是**死代码**。

### 2.5 训练 pipeline 现状

| 组件 | 状态 | 说明 |
|------|------|------|
| forward_train + backward | ✅ | smoke test 验证 |
| Optimizer (AdamW) | ✅ | smoke test 中使用 |
| LR scheduler | ❌ | 缺少 cosine warmup |
| Gradient accumulation | ❌ | 缺少 (config 定义了 grad_accum_steps=4) |
| FSDP | ❌ | 多 GPU 训练阻塞 |
| AMP (bf16) | ⚠️ | smoke test 中有 autocast, 但未在 CUDA 上验证 |
| EMA | ❌ | Stage B/C 需要 |
| Checkpoint save/load | ❌ | 无持久化 |
| Logging | ❌ | smoke test 仅 print |
| Evaluation | ❌ | 无评估循环 |
| Data pipeline (真实数据) | ❌ | 只有 dummy data |

---

## 3. 训练就绪判定

### 3.1 模型代码就绪度: 95%

| 子系统 | 完成度 | 阻塞? |
|--------|--------|--------|
| VLA 核心 (Backbone→Grounder→Core→Expert→Heads) | 95% | ❌ |
| VLA Loss (5 种) | 95% | ❌ |
| 世界模型 (9 模块) | 90% | ❌ |
| VLA ↔ WM 集成 | 60% | ❌ (VLA-only 训练不需要) |

**模型代码本身已就绪**: forward_train 端到端可运行, 梯度正确, 无数值异常。

### 3.2 训练基础设施就绪度: 15%

| 组件 | 状态 | 距完成 |
|------|------|--------|
| 单 GPU 训练循环 | ✅ (smoke test) | 0 天 |
| LR scheduler + grad accum | ❌ | 0.5 天 |
| FSDP (8×H100) | ❌ | 1-2 天 |
| EMA | ❌ | 0.5 天 |
| Checkpoint save/load | ❌ | 0.5 天 |
| 真实数据加载 | ❌ | 2-3 天 |
| Logging (WandB) | ❌ | 0.25 天 |
| Evaluation | ❌ | 1-2 天 |
| **总计** | | **~6-9 天** |

### 3.3 可以立即做什么?

**单 GPU 验证训练 (现在就可以开始)**:

用 smoke test 的框架, 替换为真实小规模数据 (如 LIBERO 的一个子集), 在单张 H100 上跑 Stage A 1K-5K 步。验证:
- Loss 下降曲线
- 梯度分布
- 内存使用
- 各 loss 项的量级关系

这不需要 FSDP、EMA 或完整 data pipeline, 只需要一个最小的数据加载器。

---

## 4. 从 v0.1 到 v0.5 的修复全景

| 轮次 | 修复内容 | 影响 |
|------|---------|------|
| **v0.1** | Grounder mid-layer compression; SDPA for all attn; MultiScaleAdapter gate; selective_scan CUDA export; Expert CUDA SSM; block_fm_to_backbone; num_embodiments | 架构正确性 + 性能 |
| **v0.2** | mamba_ssm Mamba2 集成; ActionHistoryEncoder 重构 | bf16 精度 + 5-10× Mamba 加速 |
| **v0.3** | — (分析轮, 发现 Core 状态丢失) | 识别关键回归 |
| **v0.4** | MambaBlock.step() API; ImaginationTrajectory 扩展; VLA↔WM 接口; KL per-cat free_bits; Visual decoder 集成; Physics 输出保留; PhysicsLoss 去重; SubgoalPlanner 接入; WorldModelConfig | 世界模型就绪 |
| **v0.5** | _MambaStack 官方路径状态传递; Grounder 单次调用; ActionConsistency softmax; Expert cond_dim fix; Smoke test | 训练就绪 |

**5 轮迭代共修复: 27 项问题** (8 个 P0 Critical, 10 个 P1, 9 个 P2)

---

## 5. 总结

### 模型代码: ✅ 就绪

v0.5 解决了最后的架构 bug (VLA Core 状态 + Expert 维度), 并通过 smoke test 验证了 Stage A 和 Stage B 的端到端正确性。从 v0.1 的 Grounder 压缩时机错误到 v0.5 的 `_MambaStack` 状态传递, 所有已知的模型级问题均已修复。

### 训练启动: ⚠️ 需要基础设施

模型可以在单 GPU + 小数据上立即开始验证训练。正式 8×H100 训练需要 FSDP + 真实数据 pipeline, 约 6-9 个工作日。

### 性能优化: 有空间

`_MambaStack` 的逐 token step() 循环功能正确但丧失序列并行性 (~25% overhead)。未来可通过 Mamba2 inference_params API 恢复。这是性能优化, 不阻塞训练。

### 建议的下一步

```
1. 立即: 用真实小数据 (LIBERO subset) 跑单 GPU Stage A 1K 步
   验证 loss 下降趋势和内存使用

2. 本周: LR scheduler + grad accumulation + checkpoint save/load
   升级 smoke test 为 mini trainer

3. 下周: FSDP wrapping + 完整 data pipeline
   在 8×H100 上跑 Stage A 120K steps

4. 之后: EMA + Stage B/C + 世界模型联合训练
```

---

*v0.5 标志着模型代码从 "存在关键 bug" 到 "验证通过可训练" 的转折点。5 轮迭代修复了 27 个问题, 涵盖架构正确性 (Grounder 压缩/Expert 维度)、数值稳定性 (bf16 SSM state)、梯度正确性 (ActionConsistency 可微)、状态管理 (MambaBlock step API) 和训练验证 (smoke test)。剩余工作集中在训练基础设施, 模型代码本身已就绪。*
