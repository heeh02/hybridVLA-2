# Mamba Performance Fix — v0.10.10
# Mamba 性能修复 — v0.10.10

> **Date**: 2026-03-28
> **日期**: 2026-03-28
> **Fixes**: Token-by-token kernel launch explosion + activation checkpointing failure on official Mamba2 path
> **修复内容**: 官方 Mamba2 路径上的逐 token 内核启动爆炸 + 激活检查点失效
> **Files changed**: `mamba_core.py`, `selective_scan.py`, `config.py`, `hybrid_vla_v2.py`
> **修改文件**: `mamba_core.py`, `selective_scan.py`, `config.py`, `hybrid_vla_v2.py`

---

## 1. Problem Statement
## 1. 问题描述

Two performance issues that together make multi-GPU training infeasible on target hardware:
两个性能问题叠加后，使得在目标硬件上进行多 GPU 训练变得不可行：

### 1.1 Token-by-token kernel launch explosion
### 1.1 逐 token 内核启动爆炸

**Location**: `_MambaStack.forward()` (mamba_core.py, pre-fix lines 432-454)
**位置**: `_MambaStack.forward()`（mamba_core.py，修复前第 432-454 行）

When the official `mamba_ssm.Mamba2` package is installed, `MambaBlock._forward_official()` uses fused CUDA kernels but **discards SSM/conv states** (returns `(out, None, None)`). To preserve cross-temporal-step states, a v0.5 fix introduced a **token-by-token Python loop**:
当安装官方 `mamba_ssm.Mamba2` 包时，`MambaBlock._forward_official()` 会使用融合 CUDA 内核，但会**丢弃 SSM/conv 状态**（返回 `(out, None, None)`）。为了保留跨时间步状态，v0.5 的修复引入了**逐 token 的 Python 循环**：

```python
# Pre-fix: O(T * L * N) kernel launches
# 修复前：O(T * L * N) 次内核启动
for t in range(x.shape[1]):           # L tokens per temporal step (~9-12)
    # 每个时间步有 L 个 token（约 9-12）
    for i, layer in enumerate(self.layers):  # N layers per stream
        # 每个流有 N 层
        x_t = layer.step(x_t, ...)    # Individual CUDA kernel launch
        # 单次 CUDA 内核启动
```

**Kernel launch count per `forward_train`** (T=24, L~9):
**每次 `forward_train` 的内核启动次数**（T=24，L~9）：

| Stream | Formula | Calls |
| 流 | 公式 | 次数 |
|--------|---------|------:|
| Fast (20L) | 24 x 9 x 20 | 4,320 |
| Fast（20 层） | 24 x 9 x 20 | 4,320 |
| Medium (6L) | 12 x 9 x 6 | 648 |
| Medium（6 层） | 12 x 9 x 6 | 648 |
| Slow (10L) | 4 x 9 x 10 | 360 |
| Slow（10 层） | 4 x 9 x 10 | 360 |
| ActionHistory (4L) | 24 x 8 x 4 | 768 |
| ActionHistory（4 层） | 24 x 8 x 4 | 768 |
| **Total** | | **6,096** |
| **总计** | | **6,096** |

With backward pass recompute: **~12,000+ kernel dispatches per optimization step**.
若考虑反向传播重计算：**每个优化步约有 12,000+ 次内核调度**。

Each dispatch incurs Python-to-CUDA launch latency (~5-15us), making the bottleneck **launch overhead**, not compute.
每次调度都会产生 Python 到 CUDA 的启动延迟（约 5-15us），使瓶颈变成**启动开销**而非计算本身。

### 1.2 Activation checkpointing failure
### 1.2 激活检查点失效

**Location**: `_MambaStack.forward()` (mamba_core.py, pre-fix lines 464-467)
**位置**: `_MambaStack.forward()`（mamba_core.py，修复前第 464-467 行）

Activation checkpointing only exists in the **fallback** branch:
激活检查点仅存在于 **fallback** 分支中：

```python
# Fallback path — checkpointing works
# Fallback 路径：检查点可用
if use_checkpoint and self.training:
    x, s, c = activation_checkpoint(layer, x, s_i, c_i, ...)

# Official path — checkpointing completely bypassed
# Official 路径：检查点被完全绕过
for t in range(x.shape[1]):
    for i, layer in enumerate(self.layers):
        x_t = layer.step(x_t, ...)  # Not wrapped by checkpoint
        # 未被 checkpoint 包裹
```

The `use_checkpoint` parameter is **ignored** when `uses_official=True`. Additionally, FSDP-level activation checkpointing wraps `MambaBlock.forward()` but the official path calls `layer.step()` directly, bypassing the wrapper.
当 `uses_official=True` 时，`use_checkpoint` 参数会被**忽略**。此外，FSDP 级别的激活检查点包裹的是 `MambaBlock.forward()`，但官方路径直接调用 `layer.step()`，从而绕过该包裹。

**Consequence**: All intermediate activations from 6,096+ step calls are retained in memory. Estimated ~3.4 GB/GPU additional memory on 8xH100.
**后果**：来自 6,096+ 次 step 调用的全部中间激活都被保留在内存中。在 8xH100 上估计额外占用约 3.4 GB/GPU。

---

## 2. Root Cause
## 2. 根因

Both problems share a single root cause:
这两个问题共享同一个根因：

> **Official `Mamba2.forward()` returns `(out, None, None)` — SSM/conv states are discarded by the fused CUDA kernel.**
> **官方 `Mamba2.forward()` 返回 `(out, None, None)`，即 SSM/conv 状态被融合 CUDA 内核丢弃。**

The v0.5 token-by-token `step()` workaround solved the state persistence problem but introduced kernel launch explosion and made checkpointing unreachable.
v0.5 的逐 token `step()` 变通方案解决了状态持久化问题，但引入了内核启动爆炸，并使检查点路径不可达。

### Why the fallback path doesn't have this problem
### 为什么 fallback 路径没有这个问题

The fallback `MambaBlock._forward_fallback()` implements the SSM scan in pure PyTorch with explicit state management. It:
fallback 的 `MambaBlock._forward_fallback()` 使用纯 PyTorch 实现 SSM 扫描，并进行显式状态管理。它：
1. Processes all L tokens per layer in one vectorized call (projection -> conv -> scan -> output)
1. 每层通过一次向量化调用处理全部 L 个 token（projection -> conv -> scan -> output）
2. Returns SSM/conv states correctly
2. 正确返回 SSM/conv 状态
3. Is compatible with `activation_checkpoint()` wrapping
3. 与 `activation_checkpoint()` 包裹兼容

---

## 3. Fix Strategy
## 3. 修复策略

### 3.1 Force fallback path for training (Tier 1)
### 3.1 训练时强制使用 fallback 路径（Tier 1）

**Approach**: Add `force_fallback: bool` parameter to `MambaBlock.__init__()`. When True, always create pure-PyTorch fallback parameters regardless of `mamba_ssm` availability.
**方案**：在 `MambaBlock.__init__()` 中新增 `force_fallback: bool` 参数。当其为 True 时，无论 `mamba_ssm` 是否可用，都始终创建纯 PyTorch 的 fallback 参数。

**Config**: `TemporalCoreConfig.mamba_impl: str = "fallback"` (default). Set to `"auto"` to use official Mamba2 when available.
**配置**：`TemporalCoreConfig.mamba_impl: str = "fallback"`（默认）。设为 `"auto"` 时在可用情况下使用官方 Mamba2。

**Propagation chain**:
**传递链路**：
```
TemporalCoreConfig.mamba_impl
  -> TriRateMambaCore(mamba_impl=...)
    -> FastMamba/MediumMamba/SlowMamba(force_fallback=...)
      -> _MambaStack(force_fallback=...)
        -> MambaBlock(force_fallback=...)
  -> ActionHistoryEncoder(force_fallback=...)
```

**Kernel count after fix**:
**修复后的内核计数**：

| Stream | Formula | Calls |
| 流 | 公式 | 次数 |
|--------|---------|------:|
| Fast (20L) | 24 x 20 | 480 |
| Fast（20 层） | 24 x 20 | 480 |
| Medium (6L) | 12 x 6 | 72 |
| Medium（6 层） | 12 x 6 | 72 |
| Slow (10L) | 4 x 10 | 40 |
| Slow（10 层） | 4 x 10 | 40 |
| ActionHistory (4L) | 24 x 4 | 96 |
| ActionHistory（4 层） | 24 x 4 | 96 |
| **Total** | | **688** |
| **总计** | | **688** |

**Reduction: 6,096 -> 688 calls (8.9x fewer)**. Each call now processes the full L-token sequence vectorized.
**下降幅度：6,096 -> 688 次调用（减少 8.9 倍）**。每次调用现在都以向量化方式处理完整的 L-token 序列。

### 3.2 torch.compile acceleration (Tier 2 — user-applied)
### 3.2 torch.compile 加速（Tier 2，用户自行启用）

The selective scan remains JIT-compiled (`@torch.jit.script`) for broad compatibility. Users can apply `torch.compile` at the model or block level for additional speed on CUDA-capable systems:
为了兼容性，selective scan 仍保持 JIT 编译（`@torch.jit.script`）。在支持 CUDA 的系统上，用户可在模型或模块级应用 `torch.compile` 获得额外加速：

```python
# In training script, after model creation:
# 在训练脚本中，模型创建后：
model = torch.compile(model)  # Fuses per-iteration ops, ~1.5-2x on top of JIT
# 融合每轮迭代操作，在 JIT 基础上约再提升 1.5-2x
```

This fuses per-iteration matrix operations within the scan and reduces Python dispatch overhead. Not applied by default because `torch.compile` can be slow on first invocation and has backend requirements (Triton/inductor).
它会融合 scan 内部的逐迭代矩阵操作并降低 Python 调度开销。未默认启用是因为 `torch.compile` 首次调用可能较慢，且依赖后端（Triton/inductor）。

### 3.3 ActionHistoryEncoder stateless fast-path (Bonus)
### 3.3 ActionHistoryEncoder 无状态快速路径（附加优化）

`ActionHistoryEncoder.encode()` doesn't need cross-step states (passes None, discards returned states). Added `stateless=True` parameter to `_MambaStack.forward()`:
`ActionHistoryEncoder.encode()` 不需要跨步状态（传入 None，并丢弃返回状态）。因此为 `_MambaStack.forward()` 新增 `stateless=True` 参数：

- When `stateless=True` AND using official path: uses fused `_forward_official()` per-layer (no step loop)
- 当 `stateless=True` 且使用官方路径时：每层使用融合的 `_forward_official()`（无 step 循环）
- When using fallback path: `stateless` has no effect (fallback already vectorized)
- 使用 fallback 路径时：`stateless` 无影响（fallback 本身已向量化）

This ensures ActionHistoryEncoder is fast regardless of `mamba_impl` setting.
这保证了无论 `mamba_impl` 如何设置，ActionHistoryEncoder 都能保持较快。

---

## 4. Changes Summary
## 4. 变更总结

### `vla_hybrid_v2/config.py`
### `vla_hybrid_v2/config.py`
- Added `mamba_impl: str = "fallback"` to `TemporalCoreConfig`
- 在 `TemporalCoreConfig` 中新增 `mamba_impl: str = "fallback"`

### `vla_hybrid_v2/models/mamba_core.py`
### `vla_hybrid_v2/models/mamba_core.py`
- `MambaBlock.__init__()`: Added `force_fallback` parameter; uses `HAS_MAMBA_SSM and not force_fallback` to decide path
- `MambaBlock.__init__()`：新增 `force_fallback` 参数；使用 `HAS_MAMBA_SSM and not force_fallback` 判定路径
- `_MambaStack.__init__()`: Added `force_fallback`, passes to `MambaBlock`
- `_MambaStack.__init__()`：新增 `force_fallback`，并传递给 `MambaBlock`
- `_MambaStack.forward()`: Added `stateless` parameter; restructured with three branches (official-stateless / official-stateful / fallback)
- `_MambaStack.forward()`：新增 `stateless` 参数；重构为三个分支（official-stateless / official-stateful / fallback）
- `FastMamba/MediumMamba/SlowMamba.__init__()`: Added `force_fallback`
- `FastMamba/MediumMamba/SlowMamba.__init__()`：新增 `force_fallback`
- `ActionHistoryEncoder.__init__()`: Added `force_fallback`
- `ActionHistoryEncoder.__init__()`：新增 `force_fallback`
- `ActionHistoryEncoder.encode()`: Passes `stateless=True`
- `ActionHistoryEncoder.encode()`：传入 `stateless=True`
- `TriRateMambaCore.__init__()`: Added `mamba_impl` parameter, converts to `force_fallback`
- `TriRateMambaCore.__init__()`：新增 `mamba_impl` 参数，并转换为 `force_fallback`

### `vla_hybrid_v2/models/hybrid_vla_v2.py`
### `vla_hybrid_v2/models/hybrid_vla_v2.py`
- `HybridVLAv2.__init__()`: Passes `mamba_impl` from config to `TriRateMambaCore`; passes `force_fallback` to `ActionHistoryEncoder`
- `HybridVLAv2.__init__()`：将配置中的 `mamba_impl` 传给 `TriRateMambaCore`；将 `force_fallback` 传给 `ActionHistoryEncoder`

### `vla_hybrid_v2/ops/selective_scan.py`
### `vla_hybrid_v2/ops/selective_scan.py`
- Retained `@torch.jit.script`; added docstring noting `torch.compile` as opt-in Tier 2 acceleration
- 保留 `@torch.jit.script`；新增文档字符串，说明 `torch.compile` 作为可选 Tier 2 加速

---

## 5. Performance Impact
## 5. 性能影响

| Metric | Before (official step loop) | After (fallback) |
| 指标 | 修复前（官方 step 循环） | 修复后（fallback） |
|--------|:--------------------------:|:--------------------------:|
| Kernel launches / forward_train | ~6,096 | ~688 |
| 每次 forward_train 的内核启动数 | ~6,096 | ~688 |
| Activation checkpointing | **Bypassed** | **Working** |
| 激活检查点 | **被绕过** | **可用** |
| Memory overhead (est.) | +3.4 GB/GPU | Checkpointed |
| 内存额外开销（估算） | +3.4 GB/GPU | 已检查点化 |
| Estimated training speed | Baseline | ~3-5x faster (+ optional torch.compile) |
| 训练速度估计 | 基线 | ~3-5 倍提升（可选叠加 torch.compile） |
| Config default | N/A (hardcoded) | `mamba_impl: "fallback"` |
| 默认配置 | 不适用（硬编码） | `mamba_impl: "fallback"` |

### Compatibility
### 兼容性

| Scenario | `mamba_impl="fallback"` (default) | `mamba_impl="auto"` |
| 场景 | `mamba_impl="fallback"`（默认） | `mamba_impl="auto"` |
|----------|:-:|:-:|
| Training correctness | Vectorized, checkpointed | Step loop (slow, no checkpointing) |
| 训练正确性 | 向量化并可检查点 | Step 循环（慢且无检查点） |
| Inference correctness | Fallback step() | Official step() |
| 推理正确性 | Fallback step() | Official step() |
| Existing checkpoints (official) | **Incompatible** (different param names) | Compatible |
| 现有检查点（官方） | **不兼容**（参数名不同） | 兼容 |
| Existing checkpoints (fallback) | Compatible | **Incompatible** |
| 现有检查点（fallback） | 兼容 | **不兼容** |
| New training runs | **Recommended** | Only if fused kernel needed |
| 新训练任务 | **推荐** | 仅在需要融合内核时使用 |

---

## 6. Verification
## 6. 验证

- All 49 existing tests pass with `mamba_impl="fallback"` (the new default)
- 使用 `mamba_impl="fallback"`（新默认值）时，现有 49 个测试全部通过
- The fallback path has been validated since v0.2 of the codebase
- fallback 路径自代码库 v0.2 起已被验证
- Activation checkpointing path is exercised by `test_forward_train.py` tests
- 激活检查点路径由 `test_forward_train.py` 中的测试覆盖
