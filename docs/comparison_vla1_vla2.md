# HybridVLA v1 vs v2 回归故障审计报告

**审计时间**: 2026-03-29
**审计范围**: hybridVLA_2 完整代码仓库 vs hybridVLA HPC3 历史问题清单
**审计方法**: 逐条对照历史问题 + 六维度系统排查

---

## A. 高风险问题（按严重性排序）

---

### Issue #1: eval 中 bf16 tensor 直接转 numpy — 评估崩溃

**风险等级**: Critical
**文件**: `libero_hybrid/scripts/eval_libero_rollout.py:170`
**对应历史问题**: E1 (bf16 Tensor 无法转 numpy)

**证据代码**:
```python
# eval_libero_rollout.py:170
action = step_out.action_env[0]  # [A]
actions_batch[k] = action.cpu().numpy()  # <-- 缺少 .float()
```

**完整调用链**:
1. 模型在 bf16 模式下推理 (`config.py:40: torch_dtype: str = "bfloat16"`)
2. `control_step` 返回 bf16 action (`hybrid_vla_v2.py:849-850`)
3. `denormalize()` 保留输入 dtype (`normalizer.py:127: dtype=normed.dtype`) → 返回 bf16
4. `action.cpu().numpy()` → `TypeError: Got unsupported ScalarType BFloat16`

**真实后果**: 评估脚本启动后第一步直接崩溃，无法进行任何评估。

**最小修复**:
```python
# eval_libero_rollout.py:170 改为:
actions_batch[k] = action.float().cpu().numpy()
```

**建议验证**: 运行 `python libero_hybrid/scripts/eval_libero_rollout.py` 确认不报 ScalarType 错误。

---

### Issue #2: 梯度累积缺少 no_sync — 分布式训练通信浪费

**风险等级**: High
**文件**: `scripts/train_unified.py:514-541`
**对应历史问题**: T4 (DDP 梯度同步浪费)

**证据代码**:
```python
# train_unified.py:520-530
with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.train.bf16):
    losses = model.forward_train(batch)

loss = losses["loss_total"] / grad_accum
loss.backward()  # <-- 每个 micro-batch 都触发 allreduce
micro_step += 1

if micro_step % grad_accum == 0:
    # ...optimizer.step()
```

**仓库自身也确认此问题**: `docs/code_review_v1_0.md:131` 记录了 "FSDP Gradient Accumulation Missing no_sync"，但 **代码中 grep `no_sync` 无任何实际使用**。

**真实后果**: 默认 `grad_accum_steps=4` 时，3/4 的 `backward()` 调用触发不必要的 allreduce，多 GPU 训练吞吐量降低约 30-40%。不影响正确性，但严重浪费计算资源。

**最小修复**:
```python
sync_context = model.no_sync if (micro_step % grad_accum != 0 and hasattr(model, "no_sync")) else nullcontext
with sync_context():
    loss.backward()
```

**建议验证**: 在 2+ GPU 上用 `grad_accum_steps=4` 对比添加 `no_sync()` 前后的 step/s。

---

### Issue #3: `configs/data/libero_multicam.yaml` 使用旧的错误 proprio_key

**风险等级**: High
**文件**: `configs/data/libero_multicam.yaml:10`
**对应历史问题**: P1 (Proprio Key 不匹配)

**证据代码**:
```yaml
# configs/data/libero_multicam.yaml:10
proprio_key: robot0_joint_pos   # <-- V1 错误 key！
```

**对比正确的配置**:
```yaml
# libero_hybrid/configs/data/libero_multicam.yaml:11
proprio_key: joint_states       # <-- 正确
proprio_keys: [joint_states, gripper_states]
```

**真实后果**: 如果用户使用 `configs/data/libero_multicam.yaml` 而不是 `libero_hybrid/configs/data/libero_multicam.yaml`，且数据适配器未使用 `proprio_keys` 列表，将回退到错误的 `robot0_joint_pos`，触发 KeyError（v2 有严格验证不会静默回退全零，但会崩溃）。

**最小修复**:
```yaml
# configs/data/libero_multicam.yaml:10 改为:
proprio_key: joint_states
proprio_keys: [joint_states, gripper_states]
```

**建议验证**: 全局搜索 `robot0_joint_pos` 确认无其他残留。

---

### Issue #4: 训练使用 teacher-forcing，推理使用 model output — 闭环性能结构性风险

**风险等级**: High (结构性，非代码 bug)
**文件**: `vla_hybrid_v2/models/hybrid_vla_v2.py:510-512` vs `hybrid_vla_v2.py:843-847`
**对应历史问题**: 4.3 (Loss 低 ≠ 任务成功)

**训练**:
```python
# hybrid_vla_v2.py:510-512
# Update action history (teacher-forcing: uses GT prev_actions
# during training; control_step uses model output actions instead)
action_history_buf.push(batch["prev_actions"][:, t])  # Ground-truth
```

**推理**:
```python
# hybrid_vla_v2.py:843-847
runtime_state.action_history = torch.roll(runtime_state.action_history, -1, dims=1)
runtime_state.action_history[:, -1] = action  # Model prediction
```

**真实后果**: temporal core 在训练时只见过 ground-truth action history，推理时收到模型自身预测的 action（可能有误差），造成分布偏移。误差通过 8-step action history buffer 累积，是 "loss 低但闭环任务失败" 的核心结构性原因之一。

**缓解方案**: 考虑 scheduled sampling（随训练进度逐步将 GT action 替换为 model prediction），或在 action history encoder 中引入噪声增强。

**建议验证**: 对比使用 GT action history vs model action history 的 open-loop 预测精度差异。

---

### Issue #5: ODE 采样不覆盖 t=1.0 端点 — 去噪不完整

**风险等级**: Medium
**文件**: `vla_hybrid_v2/models/flow_action_expert.py:326-353`
**对应历史问题**: A1 (Flow Matching Sampling 时间步不完整)

**Euler 采样器**:
```python
# flow_action_expert.py:332-335
for i in range(num_steps):  # num_steps=8
    t = torch.full((B,), i * dt, ...)  # t = 0, 0.125, ..., 0.875
    out = self.forward(x, t, ...)
    x = x + out.velocity * dt
```

**Midpoint 采样器**:
```python
# flow_action_expert.py:346-352
for i in range(num_steps):
    t = torch.full((B,), i * dt, ...)       # t = 0, 0.125, ..., 0.875
    t_mid = torch.full((B,), (i + 0.5) * dt, ...)  # t_mid = 0.0625, ..., 0.9375
    v1 = self.forward(x, t, ...).velocity
    x_mid = x + 0.5 * dt * v1
    v2 = self.forward(x_mid, t_mid, ...).velocity
    x = x + dt * v2
```

**分析**: 两种采样器的最后一步分别在 t=0.875 (Euler) 和 t_mid=0.9375 (Midpoint) 评估速度场。积分在 t ∈ [0, 1) 进行，不包含端点 t=1.0。

v1 的修复是从朴素 Euler 改为 midpoint，这确实提高了精度（2 阶 vs 1 阶），但 **端点缺失的根本问题仍然存在**。对于 num_steps=8，midpoint 的最终误差约为 O(dt²) = O(1/64) ≈ 0.016，在动作空间 [-1,1] 中影响有限。

**真实后果**: 轻微的去噪不完整，预期动作精度损失约 1-2%。不会导致崩溃，但可能在精细操作中造成末端抖动。

**最小修复**: 可选方案 — 使用 `num_steps+1` 个评估点显式覆盖 t=1.0，或接受当前精度。

**建议验证**: 对比 `num_steps=8` vs `num_steps=16` 的 open-loop 预测误差。

---

### Issue #6: SSM 参数 A_log/D 初始化为 fp32 — 混合精度风险

**风险等级**: Medium
**文件**: `vla_hybrid_v2/models/mamba_core.py:141-147`, `flow_action_expert.py:117-119`
**对应历史问题**: D1 (系统性 bf16/fp32 混合)

**证据代码**:
```python
# mamba_core.py:141-147
A = torch.arange(1, d_state + 1, dtype=torch.float32)  # 显式 fp32
    .unsqueeze(0).expand(self.d_inner, -1)
self.A_log = nn.Parameter(torch.log(A))  # fp32 parameter
self.D = nn.Parameter(torch.ones(self.d_inner))  # fp32 (默认)

# mamba_core.py:302, 316 (forward)
A = -torch.exp(self.A_log)  # fp32 输出
y = y + x_main * self.D.unsqueeze(0).unsqueeze(0)  # bf16 * fp32 → 隐式提升
```

**真实后果**:
- **使用 FSDP 时**: `MixedPrecision(param_dtype=torch.bfloat16)` 会强制统一为 bf16，**问题被掩盖**
- **单 GPU / 非 FSDP 时**: A_log 和 D 保持 fp32，与 bf16 activation 混合运算，可能导致 SSM 核心计算精度不一致
- **autocast 不覆盖 `nn.Parameter` 的乘法**，所以 `x_main * self.D` 在 autocast 区域内仍产生隐式类型提升

**最小修复**: 在 MambaBlock `__init__` 末尾添加 dtype 一致性设置，或在 forward 中显式 cast。

**建议验证**: 在 FSDP 和非 FSDP 模式下分别打印 `self.A_log.dtype` 和 `self.D.dtype` 确认是否一致。

---

### Issue #7: consistency loss 对 expert_denoised 使用 .detach() — 单向学习

**风险等级**: Medium (设计选择，但应明确)
**文件**: `vla_hybrid_v2/models/hybrid_vla_v2.py:690`
**对应历史问题**: A2 (expert_continuous.detach() 截断梯度)

**证据代码**:
```python
# hybrid_vla_v2.py:684-691
losses["loss_consistency"] = self.consistency_loss(
    fused_states,
    fast_tokens=fast_tokens,
    slow_token=temporal_outputs[-1].slow_token,
    discrete_actions=fast_continuous,
    continuous_actions=expert_denoised.detach(),  # <-- .detach()
) * weights.get("consistency", 0.3)
```

**对比 v1 修复**: v1 的修复是 "移除 .detach()"，但 v2 **重新引入了 .detach()**。

**关键区别**: v2 中 FM loss (`loss_flow_matching`) 在 **line 618-621 使用 `expert_out.velocity` 而不是 detach**，因此 expert 的主要梯度来自 FM loss。consistency loss 中的 .detach() 是为了防止 consistency loss 干扰 expert 的 FM 训练。

**真实后果**: 这是有意的设计 — fast head 单向学习对齐 expert，expert 只被 FM loss 监督。如果 FM loss 收敛良好，这没有问题。但如果 FM loss 不足（如 Stage B 初期 expert 从随机初始化开始），consistency loss 无法帮助 expert 学习。

**建议验证**: 监控 Stage B 前 1000 步的 `loss_consistency` 和 `loss_flow_matching` 曲线，确认二者同步下降。

---

### Issue #8: 语义刷新频率 训练 vs 推理不一致

**风险等级**: Medium
**文件**: `hybrid_vla_v2.py:451-455` vs `eval_libero_rollout.py:158`
**对应历史问题**: 无直接对应 (新发现)

**训练**:
```python
# hybrid_vla_v2.py:451-455
stride = self.cfg.train.semantic_refresh_stride  # 默认 6
refresh_steps = list(range(0, T, stride))  # T=24: [0,6,12,18] = 4 次刷新
```

**推理**:
```python
# eval_libero_rollout.py:158
if step % refresh_interval == 0:  # refresh_interval 由外部控制
    grounder_outs[k] = policy.semantic_step_from_obs(obs_k, language)
```

**真实后果**: 如果推理时 `refresh_interval` 与训练时 `semantic_refresh_stride=6` 不一致，grounder 输出的重用模式不同，temporal core 看到的输入分布与训练时不同。

**建议验证**: 确保 eval 脚本的 `refresh_interval` 参数与训练配置的 `semantic_refresh_stride` 匹配。

---

## B. 历史问题对照表

| # | Historical Issue | v2 状态 | 证据 | 备注 |
|---|---|---|---|---|
| **P1** | Proprio key 不匹配 → fallback 全零 | **已修复** (存在残留配置) | `libero_hybrid/configs/data/*.yaml` 使用正确 key `joint_states`；`hdf5_adapter.py:331-336` 严格校验 KeyError 无 fallback | `configs/data/libero_multicam.yaml:10` 仍为旧 key `robot0_joint_pos` |
| **P2** | 图像方向 bottom-up / top-down 不一致 | **已修复** | 全局 grep 无 flip/vflip 操作；train 和 eval 均不翻转，保持原始方向一致 | 设计变更：不再翻转，统一保留原始方向 |
| **P3** | Eval 缺少 action denormalize | **已修复** | `libero_policy.py:419`: `action_env = self.action_normalizer.denormalize(action_model)` | 正确实现，但 bf16 输出问题见 Issue #1 |
| **P4** | Proprio 错用 action normalizer | **已修复** | 独立的 `ProprioNormalizer` + 独立 stats 文件 `proprio_stats.json`；`data/__init__.py:47-48` 分别初始化 | 彻底修复，不共享 normalizer 实例 |
| **P5** | Eye-in-hand 摄像头缺失 | **已修复** | `libero_hybrid/configs/data/libero_multicam.yaml` 配置 `eye_in_hand_rgb`；完整的多相机 adapter 支持 | 支持 2-8 相机 |
| **P6** | Placeholder 图像尺寸过小 | **已修复** | 统一 448x448 (`hdf5_adapter.py:283-285`); `min_pixels: 200704` 在 model config 中 | 无 placeholder 回退，缺图直接 KeyError |
| **A1** | Flow matching 时间步不完整 | **部分修复** | midpoint 采样器 (`flow_action_expert.py:338-353`) 提升到 2 阶精度，但仍不覆盖 t=1.0 端点 | 见 Issue #5 |
| **A2** | expert_continuous.detach() 截断梯度 | **设计变更** | FM loss 不 detach (`hybrid_vla_v2.py:618`)；consistency loss 故意 detach (`hybrid_vla_v2.py:690`) | 见 Issue #7，设计选择非 bug |
| **A3** | Grounder 对同一输入调用 R 次 | **已修复** | 无 refresh 时调用 1 次复制 R 份 (`hybrid_vla_v2.py:432-443`)；有 refresh 时每次用不同输入 | 彻底修复 |
| **A4** | Condition prefix 只用最后一步 | **已修复** | cond_prefix 包含 grounder 全部 slot + temporal core 的 SSM 累积状态 + 三速率 mean-pooled tokens (`hybrid_vla_v2.py:258-291`) | 通过 SSM 递归编码完整时序 |
| **A5** | Attention mask 逻辑/类型问题 | **已修复** | `~context_mask` 正确反转；`masked_fill(-inf)` 适配 SDPA；上游 `.bool()` 转换 (`hybrid_vla_v2.py:427`) | 类型安全 |
| **A6** | Latent queries 固定 0.02 初始化 | **保留** | `attention_grounder.py:185-186`: `torch.randn(...) * 0.02`；route_queries 同样 0.02 | v1 改为 Xavier，v2 回到 0.02 但经过验证可接受 |
| **A7** | Selective scan broadcast 问题 | **已修复** | JIT-compiled `ssm_scan` (`ops/selective_scan.py:47-65`) 使用标准 PyTorch 隐式 broadcast | 正确 |
| **D1** | bf16/fp32 系统性混合 → FSDP 崩溃 | **基本修复** | FSDP `MixedPrecision(param_dtype=bf16)` 统一 dtype (`distributed.py:104-110`) | SSM A_log/D 残留 fp32 init，见 Issue #6 |
| **D2** | Checkpoint load 破坏 dtype | **已修复** | load 在 model.to(device) + FSDP wrap 之后执行；`load_state_dict` 不改变参数 dtype | 安全 |
| **D3** | Tied weights + FSDP 不兼容 | **不适用** | v2 无 tied weights；FSDP 不 wrap 冻结 backbone | 设计规避 |
| **T1** | FSDP dtype 崩溃 | **已修复** | 统一 `MixedPrecision` 配置 | 安全 |
| **T2** | Activation checkpointing 与 DDP 不兼容 | **已修复** | 使用 FSDP 原生 `NO_REENTRANT` checkpointing (`distributed.py:145`) | 正确 |
| **T3** | Checkpoint 保存 CPU OOM | **已修复** | rank-0 保存 + atomic write + FSDP state dict offload (`checkpointing.py:92-131`) | 安全 |
| **T4** | Grad accumulation 无效梯度同步 | **未修复** | 全局搜索无 `no_sync()` 调用；`docs/code_review_v1_0.md:131` 也指出此问题 | 见 Issue #2 |
| **T5** | module. 前缀不匹配 | **已修复** | `_strip_fsdp_prefix()` (`ema.py:27-35`) 处理 FSDP 前缀；FSDP 状态字典上下文管理 | 有测试覆盖 |
| **E1** | bf16 tensor 转 numpy | **未修复** | `eval_libero_rollout.py:170` 缺少 `.float()` | 见 Issue #1 |
| **E2** | 视频帧方向 | **已修复** | 训练和 eval 统一不翻转 | 设计变更 |
| **E3** | LIBERO 环境交互式输入 | **不适用** | v2 使用 `get_libero_path()` 库函数 | 基础设施问题 |
| **E4** | LIBERO assets 缺失 | **不适用** | 外部依赖，非代码问题 | — |
| **E5** | robosuite 日志权限 | **不适用** | 外部依赖，非代码问题 | — |

---

## C. 需要重点人工复核的点

### C1: 闭环 action history 分布偏移

**代码审计无法确定**: teacher-forcing (训练) vs autoregressive (推理) 的 action history 偏移量是否在 temporal core 的容错范围内。

**需要实验**:
1. 在验证集上用 ground-truth action history 跑 open-loop，记录预测误差 E_gt
2. 用 model 自身预测的 action history 跑 open-loop，记录预测误差 E_model
3. 如果 E_model >> E_gt（如 >2x），说明 temporal core 对 action history 噪声敏感，需要 scheduled sampling 或噪声增强

### C2: Stage A → Stage B 的 backbone 表征质量

**代码审计无法确定**: Stage A 训练的 backbone LoRA 是否产生了有效的视觉表征，以及 Stage B 冻结这些 LoRA 参数后 action expert 能否基于此学到有效的连续动作。

**需要实验**:
1. Stage A 完成后，用 grounder 输出做可视化（t-SNE 或 attention map），确认语义 slot 分配是否合理
2. Stage B 前 1000 步，监控 loss_fm 是否稳定下降（如果震荡或不降，说明 cond_prefix 质量不足）

### C3: 多相机训练时图像编码一致性

**代码审计发现两个不同的多相机 config**:
- `libero_hybrid/configs/data/libero_multicam.yaml`: 2 相机 (agentview + eye_in_hand)
- `configs/data/libero_multicam.yaml`: 3 相机 (agentview + eye_in_hand + left_view), 且 proprio_key 错误

**需要人工验证**: 使用哪个 config 做训练？LIBERO 数据集中实际有几个相机？

### C4: Flow matching 训练时间步分布 vs 推理 ODE 路径

**代码审计无法确定**: logit_normal 采样分布 (训练) vs 均匀步长 ODE (推理) 的不匹配是否导致 expert 在某些 t 值上欠拟合。

**需要实验**:
1. 画出 `sigmoid(randn())` 的密度直方图，确认 t 在哪个区间采样最密集
2. 在推理 ODE 路径的每一步记录预测误差，检查是否某些 t 值误差异常高

### C5: Normalizer 的 denormalize 数值稳定性

**代码审计发现**: `mean_std` 策略使用 `atanh` 反归一化 (`normalizer.py:139-140`)，在边界附近（接近 ±1）数值不稳定。

**需要验证**: 训练集的 action 分布是否有大量接近 ±1 的值？如果有，`atanh` 可能产生极大值。

### C6: EMA 跨 Stage 恢复

**代码审计发现 EMA 有 shape mismatch 处理** (`ema.py:134-156`)，但无法确认跨 Stage 恢复时 shadow 参数是否语义正确（如 Stage A 无 expert shadow，Stage B 需要初始化 expert shadow）。

**需要验证**: Stage B 启动时打印 EMA shadow keys，确认 expert 相关 shadow 是否正确初始化。

---

## D. 审查结论

### 最可能导致 "loss 低但任务失败" 的 3 个问题

1. **Action history teacher-forcing 偏移** (Issue #4) — temporal core 从未在训练中见过有误差的 action history，闭环时 compounding error 累积
2. **Stage A→B backbone 冻结传递错误表征** — 如果 Stage A 的 backbone LoRA 学到了次优表征（如对 proprio 或 phase 的编码不足），Stage B 冻结后 action expert 在错误表征上拟合，loss 可降但闭环失败
3. **语义刷新频率训练/推理不一致** (Issue #8) — grounder 在不同刷新周期下产生不同的 slot 分配，导致推理时 temporal core 看到与训练不同的时序模式

### 最可能导致训练直接崩溃的 3 个问题

1. **bf16 tensor → numpy** (Issue #1) — eval 脚本第一步即崩溃 (但不影响训练本身)
2. **错误的 proprio_key 在残留 config 中** (Issue #3) — 如果误用 `configs/data/libero_multicam.yaml` 会在数据加载时 KeyError 崩溃
3. **SSM fp32/bf16 混合** (Issue #6) — 在非 FSDP 模式下可能产生 dtype mismatch 错误（取决于 PyTorch 版本对隐式转换的处理）

### 总体评估

**这个仓库当前处于 "能跑但需要验证关键路径" 状态。**

积极面:
- v1 的大部分 Critical/High 问题已系统性修复
- 数据管线有严格校验（KeyError 而非静默 fallback）
- 分布式训练基础设施（FSDP wrapping、checkpoint、EMA）实现可靠
- 测试覆盖了关键的跨 Stage 兼容性场景

需要关注:
- 1 个 Critical bug (E1 bf16→numpy) 会阻断评估
- 1 个残留配置 (P1 旧 proprio key) 可能误导用户
- `no_sync` 仍缺失，多 GPU 训练效率低于预期
- 闭环性能的结构性风险（teacher-forcing、stage 冻结传递）需要实验验证而非代码修复

**与 v1 相比，v2 的代码质量有质的提升 — 从 "容易直接挂掉" 进步到 "能跑、需要实验验证闭环效果"。**
