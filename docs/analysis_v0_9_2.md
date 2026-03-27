# HybridVLA v2 — v0.9.2 Cross-Audit Report

综合 `optimize_v0_9_1.md`（用户修复报告）与 GPT 逐文件审计报告，对代码做严格交叉验证。

---

## Part 1: v0.9.1 Fix Verification (9/9)

逐项对照源码，确认修复是否正确落地。

| # | Fix | Status | Code Evidence |
|---|-----|--------|---------------|
| F1 | Denoising `x_t+(1-t)*v` | **PASS** | `hybrid_vla_v2.py:491` — `noisy_actions + (1.0 - flow_t[:, None, None]) * expert_out.velocity` |
| F2 | proprio_dim 解耦 | **PASS** | `config.py:179` — `proprio_dim: int = 14`; `hybrid_vla_v2.py:139` — `nn.Linear(mcfg.proprio_dim, d_core)`; `train_smoke_test.py:31-32` — `A=7, P=9` |
| F3 | Grounder mask 3/3 | **PASS** | 训练多刷新路径 `:334-336`、单观测路径 `:348`、推理路径 `:521-524` 均传入 `attention_mask=` |
| F4 | res_scale no-decay | **PASS** | `train_stage_a.py:142` — `no_decay_keywords = {"bias", "res_scale", "LayerNorm.weight", ...}`; 两组 param group |
| F5 | refresh_counter 替代 id() | **PASS** | `types.py:94-95` — `refresh_counter: int = 0` + `_last_seen_refresh: int = -1`; `hybrid_vla_v2.py:543` — 比较计数器 |
| F6 | ControlStepOutput | **PASS** | `types.py:70-81` — 含 `action: Tensor [B,A]`; `hybrid_vla_v2.py:629-634` — 返回 `ControlStepOutput(action=action, ...)` |
| F7 | 推理配置解耦 | **PASS** | `hybrid_vla_v2.py:559-561` — `medium_stride = max(1, round(self.cfg.infer.control_hz / self.cfg.infer.medium_hz))` |
| F8 | _validate_batch | **PASS** | `hybrid_vla_v2.py:259-290` — 检查 5 个必需键 + actions/proprio/prev_actions 维度; `:297` 调用 |
| F9 | 移除冗余 detach | **PASS** | `consistency_loss.py:72` — 无 `.detach()`; `hybrid_vla_v2.py:498` — 保留调用侧 `.detach()` |

**F9 设计合理性说明**：调用侧保留 `.detach()` 是正确的。梯度屏障是模型层决策（"一致性损失不反传到 expert"），不应写死在损失模块内。这使得 `ActionConsistencyLoss` 在其他需要梯度流的场景下可复用。

**结论**：9 项修复全部正确落地，无遗漏或错误实现。

---

## Part 2: GPT 审计报告交叉校验

GPT 报告覆盖了 config.py / types.py / hybrid_vla_v2.py / qwen2vl_backbone.py / attention_grounder.py / mamba_core.py / flow_action_expert.py / discrete_heads.py / losses/* / utils/* / scripts/* / world_model/* 的逐文件审计。以下逐条验证其发现的准确性。

### GPT 结论准确且已被 v0.9.1 修复的（6 项）

| GPT 发现 | 对应 Fix | 验证 |
|----------|---------|------|
| proprio_proj 使用 action_dim | F2 | 已修，`mcfg.proprio_dim` |
| Grounder mask 未传入 | F3 | 已修，3 个调用点 |
| control_step 不返回当前 action | F6 | 已修，`ControlStepOutput` |
| 推理 medium_update 引用 train config | F7 | 已修，从 Hz 计算 |
| denoising 公式 `x_t + v` | F1 | 已修，`(1-t)*v` |
| batch 无验证 | F8 | 已修，`_validate_batch()` |

### GPT 结论准确且仍为 Open 的（5 项）

#### G1. 真实数据管线缺失 — **CONFIRMED, P0**

GPT："`data/` 空壳，`DataConfig` 字段无消费"。

代码验证：`vla_hybrid_v2/data/__init__.py` 仅含 docstring。`train_stage_a.py` 使用 DummyVLADataset。`DataConfig` 的 `format`/`paths`/`data_dir`/`camera_keys` 均未被任何代码读取。

**状态**：v0.9.1 未改，是项目最大阻断项。

#### G2. RTC/FASTER/step_weights 仅为配置 — **CONFIRMED, P2**

GPT："config.py 和 stage_c.yaml 声明 rtc/faster，但训练端只读 `batch.get("step_weights")`，推理端仅 chunk cache"。

代码验证：
- `flow_matching.py:19-20` — `step_weights` 消费逻辑存在
- `hybrid_vla_v2.py:482` — `step_weights = batch.get("step_weights", None)` — 永远为 None（无生成器）
- `rtc.enable` / `faster.enable` 无任何代码检查

**状态**：Stage C 功能未实现，但不阻塞 Stage A/B。

#### G3. 多相机配置但未实现 — **CONFIRMED, P3**

GPT："`forward_semantic()` 是单路 `pixel_values`，无相机循环/融合"。

代码验证：`qwen2vl_backbone.py:6` docstring 声称多相机，但 `:181-213` 只接受单个 `pixel_values`。`config.py:52-58` 的 `MultiCameraConfig` 未被消费。

**状态**：GPT 建议"先标注未实现，或实现最小接口"。这是正确的方向。

#### G4. World Model 未接入训练 — **CONFIRMED, P2**

GPT："`imagination_engine` 和 `world_model_loss_fn` 在 `forward_train()` 中从未调用"。

代码验证：`hybrid_vla_v2.py:175-200` 初始化了 world model 组件，但 `forward_train()` 的 ~210 行中无任何 `imagination_engine` 或 `world_model_loss_fn` 调用。`get_world_model_state()` 方法存在但无调用点。

**状态**：Dead code，初始化有显存开销但无训练信号。

#### G5. activation checkpointing 仅覆盖 fallback path — **CONFIRMED, LOW**

GPT 与代码一致：`mamba_core.py` 的 `_MambaStack.forward()` 中，official Mamba2 路径走 token-by-token `step()` 循环，完全忽略 `use_checkpoint` 参数。只有 fallback path 使用 `activation_checkpoint()`。

**状态**：架构约束，非 bug。Official path 的 step() 本身不支持 checkpoint。

---

### GPT 结论准确且为新发现的（3 项）

#### G6. `_dict_to_dataclass` 静默丢弃未知 key — **NEW, MEDIUM**

```python
# config.py:334-336
for k, v in data.items():
    if k not in field_types:
        continue  # ← 静默跳过，无警告
```

**影响**：YAML 配置中的拼写错误（如 `leraning_rate` 代替 `learning_rate`）会被无声忽略，使用默认值。用户以为改了配置，实际没有生效。这是典型的"debugging 黑洞"。

**修复建议**：增加 `strict` 模式或至少 `logger.warning(f"Unknown config key: {k}")`:
```python
if k not in field_types:
    import warnings
    warnings.warn(f"Unknown config key '{k}' in {cls.__name__}, ignored")
    continue
```

**Priority**: P1 — 一旦接入真实训练，配置错误将难以排查。

#### G7. AffordanceHead docstring 与实现不匹配 — **NEW, LOW**

```python
# discrete_heads.py:61-64 — docstring 声称输出 spatial map + categorical
"""Outputs a spatial affordance map (where to act) and a categorical
affordance type (how to act)."""

# discrete_heads.py:75-76 — 实际只有 categorical
def forward(self, affordance_token: Tensor) -> Tensor:
    return self.type_head(affordance_token)  # 仅分类 logits
```

**影响**：Docstring 暗示有空间 affordance map 能力，实际只有分类输出。不会导致运行错误，但会误导协作者。

**修复**：将 docstring 改为 `"Predicts categorical affordance type from the affordance token."`

**Priority**: P3 — 文档准确性。

#### G8. `torch.load(weights_only=True)` 无旧版 PyTorch 兼容 — **NEW, LOW**

`checkpointing.py:107,116,120,124` 全部使用 `weights_only=True`。此参数需要 PyTorch >= 2.1。repo 未声明最低 PyTorch 版本。

**影响**：旧环境运行 `load_checkpoint` 会报 `TypeError: unexpected keyword argument 'weights_only'`。

**Priority**: P3 — 添加 `requirements.txt` 中的版本约束即可。

---

### GPT 结论需要修正或补充的（4 项）

#### C1. label_smoothing "未暴露到 config" — **部分正确**

GPT 称 "label smoothing 固定为 0.1，未暴露到 config"。

**实际情况**：
- `DiscreteCELoss.__init__` 接受 `label_smoothing` 参数 → 类本身可配置 ✓
- `hybrid_vla_v2.py:171` — `self.discrete_loss = DiscreteCELoss(label_smoothing=0.1)` → 模型类硬编码 0.1 ✗
- `config.py` — 无 `label_smoothing` 字段 → 配置不可控 ✗

**结论**：GPT 的判断方向正确（训练时不可从 config 控制），但表述不够精确（类本身支持参数化）。实际修复应在 `HeadsConfig` 增加 `label_smoothing: float = 0.1` 并在模型初始化时读取。

**Priority**: P3 — 功能不影响正确性。

#### C2. "Temporal core 与 cond_prefix token 不匹配" — **GPT 的隐含担忧，但非 bug**

GPT 审计一个子代理标注了 temporal_core 输入序列 (33 tokens: global+phase+unc+aff+proprio+prev_action+stale+emb+action_history+24 slots) 与 cond_prefix (32 tokens: global+24 slots+phase+unc+aff+fused+fast+medium+slow) 的组成不同，标记为 "CRITICAL inconsistency"。

**这不是 bug**。这是两个不同阶段的不同功能：
- **Temporal core 输入**：原始观测 tokens → 经过 tri-rate Mamba 处理 → 输出 fused/fast/medium/slow
- **Cond prefix**：从 grounder 输出 + temporal core 输出组装 → 条件化 action expert

cond_prefix 使用 temporal core 的**输出**（fused_state, fast_token 等），而非其输入。两者 token 集合不同是**设计正确**的。

#### C3. "ContrastiveTemporalLoss O(N^2)" — **已知，GPT 分析正确但无新信息**

N = B*(T-1)。对 B=64, T=24: N=1472，矩阵 1472×1472 ≈ 8.7MB (fp32)。当前规模可接受。GPT 建议增加负样本采样/分块开关，作为 P2 优化合理。

#### C4. "output_hidden_states=True 内存开销" — **正确观察，非必须修复**

`qwen2vl_backbone.py:185` 的 `output_hidden_states=True` 会保留所有 28 层 hidden states。对 7B 模型 + 2048 token 序列，每层约 `2048 * 3584 * 2 ≈ 14MB` (bf16)，28 层共 ~400MB。

这是多尺度特征提取所必需的（取 layer 10/18/28），除非改用 hook 选层。当前为合理的工程权衡。

---

## Part 3: 本次审计发现的额外问题

### N1. `DiscreteCELoss` 硬编码 label_smoothing — **NEW, P3**

```python
# hybrid_vla_v2.py:171
self.discrete_loss = DiscreteCELoss(label_smoothing=0.1)
```

值未从 config 读取。`HeadsConfig` 中无此字段。需在小数据集/不同任务时可调。

### N2. `_fast_bin_centers` 与 `discretise_actions` 的范围耦合 — **NEW, P2**

```python
# hybrid_vla_v2.py:165
self.register_buffer("_fast_bin_centers", torch.linspace(-1, 1, V))

# discrete_heads.py:38-40
def discretise_actions(actions, lo=-1.0, hi=1.0, V=512):
    actions = actions.clamp(lo, hi)
    return ((actions - lo) / (hi - lo) * (V - 1)).long()
```

两处的 `[-1, 1]` 范围是隐式绑定的。如果将来改变 action 归一化范围（如 `[0, 1]`），需要同时修改两处，但没有共享常量或配置来强制一致。

**修复建议**：在 `HeadsConfig` 增加 `action_range: Tuple[float, float] = (-1.0, 1.0)`，两处都从 config 读取。

### N3. 训练循环 action_history 推入 `prev_actions[:, t]` 的语义 — **值得文档化**

```python
# hybrid_vla_v2.py:418
action_history_buf.push(batch["prev_actions"][:, t])
```

在每个时间步 t，action_history 编码的是 **t 之前的动作历史**（push 在 temporal_core 之后）。但推入的是 `prev_actions[:, t]` 而非当前步的模型输出动作。这意味着训练时使用 ground-truth 历史，推理时使用模型自身输出。

这是 VLA 领域的标准 teacher-forcing 模式，设计正确。但应在代码中注明，以免与 `control_step()` 中的自回归更新（line 623-626 推入模型输出 action）产生混淆。

### N4. `cond_builder` 输出维度假设 — **MINOR**

```python
# hybrid_vla_v2.py:144-146
self.cond_builder = nn.Sequential(
    nn.LayerNorm(d_core),
    nn.Linear(d_core, d_core),
    nn.GELU(),
)
```

`cond_builder` 输出维度为 `d_core`，然后 `core_to_expert` (`:154`) 将其投影到 `d_expert`。这意味着 `_build_cond_prefix` 的 pad/truncate (`:246-251`) 操作在 `d_core` 空间进行，零 padding 的语义是"无信息的 d_core 向量"。投影到 `d_expert` 后零向量变为 `core_to_expert.bias`（如果有 bias）而非严格零。

**影响**：仅当 token 数量不等于 `cond_tokens=32` 时触发（当前恰好 32，不触发）。属于边界条件防护，不影响正常运行。

---

## Part 4: Remaining Open Issues (Consolidated)

将所有来源（v0.9.1 遗留 + GPT 新发现 + 本次审计补充）的 open issues 整合为一张表。

### P0 — 阻塞真实训练

| # | Issue | Source | Status |
|---|-------|--------|--------|
| R1 | 真实数据管线 (`vla_hybrid_v2/data/` 空壳) | User 10-point #5 / GPT | **未实现** |

### P1 — 影响工程质量

| # | Issue | Source | Status |
|---|-------|--------|--------|
| R2 | `_dict_to_dataclass` 静默跳过未知 key | GPT (G6) | **新发现** |
| R3 | Stage B/C 训练脚本不存在 | GPT | **已知未实现** |
| R4 | Evaluation loop / metrics 不存在 | v0.9.1 报告 | **已知未实现** |

### P2 — 功能完整性

| # | Issue | Source | Status |
|---|-------|--------|--------|
| R5 | RTC/FASTER/step_weights 仅为配置 | User #7 / GPT (G2) | **已知未实现** |
| R6 | World model 未接入训练 | GPT (G4) | **已知，by design** |
| R7 | `_fast_bin_centers` 与 `discretise_actions` 范围隐式耦合 | 本次 (N2) | **新发现** |
| R8 | 多相机前向路径未实现 | User #9 / GPT (G3) | **已知** |

### P3 — 文档/代码卫生

| # | Issue | Source | Status |
|---|-------|--------|--------|
| R9 | AffordanceHead docstring 声称 spatial map | GPT (G7) | **新发现** |
| R10 | label_smoothing 模型初始化硬编码 0.1 | GPT/本次 (C1/N1) | **新发现** |
| R11 | `torch.load weights_only` 无旧版兼容 | GPT (G8) | **新发现** |
| R12 | FASTDiscreteHead 名称可能误导 | GPT | **观察** |
| R13 | activation checkpoint 仅覆盖 fallback Mamba | GPT (G5) | **已知，架构约束** |

---

## Part 5: Updated Scoring

### 修复验证后评分（v0.9.2 审计口径）

v0.9.1 的 9 项修复全部通过验证。在此基础上，纳入 GPT 审计和本次交叉检查发现的新问题，给出更精确的评分。

| Dimension | v0.9.1 (报告值) | v0.9.2 (审计校准) | Delta | Justification |
|-----------|-----------------|-------------------|-------|---------------|
| Design coherence | 8.5 | **8.5** | — | F2/F6/F7 确认有效；config silent skip (R2) 和 label_smoothing 硬编码 (R10) 为小瑕疵 |
| Correctness | 9.5 | **9.5** | — | F1/F3/F5 全部正确落地，无新 correctness bug |
| Completeness | 5.5 | **5.0** | -0.5 | 数据管线仍空，Stage B/C 脚本缺失，evaluation 不存在；GPT 审计让缺口更显性化 |
| Training stability | 9.0 | **9.0** | — | F4(res_scale no-decay) + F1(正确 consistency target) 验证有效 |
| Scalability | 7.0 | **7.0** | — | 无变化 |
| Performance | 6.0 | **6.0** | — | 无变化 |
| Production readiness | 6.5 | **6.0** | -0.5 | config silent skip 是生产环境隐患；Stage B/C 不存在 |
| **Weighted avg** | **7.5** | **7.3** | -0.2 | |

**解读**：v0.9.1 报告的 7.5 分略偏乐观。经 GPT 审计交叉验证后，completeness 和 production readiness 各扣 0.5，反映 config 静默丢弃和基础设施缺口。核心模型层面（correctness, stability, design）评分确认无误。

### 下一步收益预估

| Action | Effort | Score Impact |
|--------|--------|-------------|
| R1: 最小真实数据 loader | 5 days | Completeness +1.5, Production +1.0 |
| R2: config unknown key warning | 0.5 day | Production +0.5 |
| R3: Stage B 训练脚本 | 2 days | Completeness +0.5 |
| R4: 基础 evaluation loop | 2 days | Production +1.0 |
| **合计** | ~10 days | **≈ 8.8** (+1.5) |

---

## Part 6: GPT 审计报告质量评估

GPT 的逐文件审计报告整体质量较高，以下为具体评估：

### 优势

1. **覆盖面广**：逐文件审计覆盖了 config / types / 所有 models / losses / utils / scripts / world_model，比之前的审计报告更系统
2. **接口契约表**：将每个函数的输入/输出形状和 config 依赖整理成表格，对后续开发者非常有价值
3. **"未指定/不明确"标注**：明确区分了"代码实现"与"配置声明"之间的差距，避免假设
4. **识别 `_dict_to_dataclass` 静默丢弃**（G6）：这是之前所有审计轮次都未发现的实质性工程问题
5. **数值风险关注**：指出 bf16 ODE 积分精度、output_hidden_states 内存等实际部署问题

### 需要修正的

| GPT 结论 | 实际情况 |
|----------|---------|
| "label_smoothing 固定为 0.1" | 类可配置，模型初始化硬编码。方向正确，表述不精确 |
| "temporal core vs cond_prefix token 不匹配"（隐含于接口表） | 这是 input→output 的正常转换，非 inconsistency |
| "FASTDiscreteHead 命名误导（非频域）" | "FAST" 在 pi-0 论文中指 Factorized Action Space Tokenization，IS per-dim binning。名称无误 |

### 未覆盖的

GPT 审计未涉及：
- v0.9 引入的 `res_scale` 与 weight_decay 交互（F4 修复内容）
- `id()` 语义刷新检测的脆弱性（F5 修复内容）
- 去噪公式的数学推导验证（F1 修复内容）

这些在 `analysis_v0_9_1.md` 中已覆盖，两份报告互补。

---

## 中文总结

### 核心结论

1. **v0.9.1 的 9 项修复全部正确落地**，代码与报告描述一致，无遗漏。

2. **GPT 审计报告质量良好**，覆盖面超过之前所有审计轮次。其中 3 项为新发现（G6 config 静默丢弃、G7 AffordanceHead docstring、G8 torch.load 兼容性），1 项需要修正（label_smoothing 表述不精确），1 项为误判（temporal core vs cond_prefix 并非 inconsistency）。

3. **当前最大阻断项仍然是真实数据管线**（R1），这一点 GPT 报告和所有之前的审计一致。

4. **新增的工程风险是 config 静默丢弃未知 key**（R2），建议优先修复——一行 `warnings.warn` 即可防止大量 debugging 时间。

5. **校准后评分 7.3/10**（v0.9.1 报告为 7.5），差异来自 completeness 和 production readiness 的 -0.5 调整。核心模型质量（correctness 9.5, stability 9.0, design 8.5）确认无误。

### 推荐的下一步

```
R2: config unknown key warning     — 1 行改动，即时防错
  ↓
R1: 最小真实数据 loader            — 项目最大阻断项
  ↓
R3: Stage B 训练脚本               — 启用 expert 训练
  ↓
R4: evaluation loop                — 能够衡量训练效果
  ↓
R7: action range 配置化            — 消除隐式耦合
  ↓
R5: step_weights 最小实现          — 启用 Stage C
```

完成 R1-R4 后，预计评分可达 **8.8/10**，进入"可运行实验平台"阶段。
