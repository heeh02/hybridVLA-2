# HybridVLA v2 — v0.9.1 Code Audit Report
# HybridVLA v2 — v0.9.1 代码审计报告

Based on user's 10-point analysis, this document verifies each claim against source code, assesses severity, and identifies additional issues.
基于用户提出的 10 点分析，本文档逐条对照源码进行核验，评估严重性，并补充额外问题。

---

## Verification Status Summary
## 核验状态总览

| # | Issue | User Claim | Code Verification | Severity |
|---|-------|-----------|-------------------|----------|
| 1 | proprio_dim coupling | **CONFIRMED** | `proprio_proj` uses `ecfg.action_dim` | **HIGH** |
| 2 | control_step return semantics | **CONFIRMED** | Returns full chunk, not current action | **MEDIUM** |
| 3 | Grounder attention_mask | **CONFIRMED** | All 3 call sites omit mask | **HIGH** |
| 4 | Batch schema validation | **CONFIRMED** | Zero assertions, zero validation | **MEDIUM** |
| 5 | Real data pipeline | **CONFIRMED** | `vla_hybrid_v2/data/` is empty | **HIGH** |
| 6 | Train/Infer config coupling | **CONFIRMED** | `control_step` reads `cfg.train.*` | **MEDIUM** |
| 7 | step_weights/RTC/FASTER | **CONFIRMED** | Config-only, no implementation | **MEDIUM** |
| 8 | Denoising formula | **CONFIRMED — BUG** | `x_t + v` should be `x_t + (1-t)*v` | **HIGH** |
| 9 | Multi-camera fake support | **CONFIRMED** | Config + docstring only, no code | **LOW** |
| 10 | God-class refactoring | **VALID CONCERN** | 580 lines, 5+ responsibilities | **LOW** (future debt) |

中文对照（表格摘要）：
1）proprio_dim 与 action_dim 被错误耦合（高）。
2）control_step 返回整块 chunk 而非当前动作（中）。
3）Grounder 的 attention_mask 三处调用均未传（高）。
4）batch 无结构校验（中）。
5）真实数据管线为空（高）。
6）推理流程读取训练配置（中）。
7）step_weights/RTC/FASTER 仅有配置无实现（中）。
8）去噪公式存在数学错误（高）。
9）多相机支持为“伪支持”（低）。
10）God-class 为有效技术债（低）。

---

## Detailed Verification
## 详细核验

### Issue 1: proprio_dim Coupling to action_dim — CONFIRMED HIGH
### 问题 1：proprio_dim 与 action_dim 耦合 —— 已确认（高）

**Code Evidence:**
**代码证据：**

```python
# hybrid_vla_v2.py:137-138
self.proprio_proj = nn.Linear(ecfg.action_dim, d_core)
self.prev_action_proj = nn.Linear(ecfg.action_dim, d_core)
```

```python
# config.py:97-110 — ActionExpertConfig
action_dim: int = 14
# NO proprio_dim field exists anywhere in config
```

```python
# train_smoke_test.py:121
"proprio": torch.randn(T, A),  # A = action_dim
```

**Assessment**: User's analysis is 100% correct.
**评估**：用户该项分析 100% 正确。
In real robotics:
在真实机器人系统中：
- **Franka Panda**: proprio = 23 (7 joint pos + 7 joint vel + 7 joint torque + 1 gripper + 1 gripper vel), action = 7 (delta joint pos)
- **Franka Panda**：proprio=23（7 关节位置 + 7 关节速度 + 7 关节力矩 + 1 夹爪 + 1 夹爪速度），action=7（关节位置增量）
- **ALOHA**: proprio = 14 (joint pos), action = 14 (matches by coincidence)
- **ALOHA**：proprio=14（关节位置），action=14（仅是巧合一致）
- **Mobile Manipulators**: proprio = 20+ (base vel + arm + hand), action = 7-8
- **移动机械臂平台**：proprio 常为 20+（底盘速度 + 手臂 + 手部），action 多为 7-8

Hardcoding `proprio_dim == action_dim` will silently produce wrong projections on most platforms.
将 `proprio_dim == action_dim` 硬编码会在多数平台上静默产生错误投影。

**Fix Locations**:
**建议修改位置：**
- `config.py`: Add `proprio_dim: int = 14` to `ModelConfig` (not `ActionExpertConfig`, since proprio is a model-level input, not expert-specific)
- `config.py`：在 `ModelConfig` 中新增 `proprio_dim: int = 14`（而不是放在 `ActionExpertConfig`，因为 proprio 属于模型输入层级）
- `hybrid_vla_v2.py:137`: `self.proprio_proj = nn.Linear(mcfg.proprio_dim, d_core)`
- `hybrid_vla_v2.py:137`：改为 `self.proprio_proj = nn.Linear(mcfg.proprio_dim, d_core)`
- `hybrid_vla_v2.py:138`: `self.prev_action_proj = nn.Linear(ecfg.action_dim, d_core)` (keep — this IS action dim)
- `hybrid_vla_v2.py:138`：`self.prev_action_proj = nn.Linear(ecfg.action_dim, d_core)` 保持不变（该项确实是动作维度）
- Dummy dataset: Use `cfg.model.proprio_dim` for proprio shape
- Dummy 数据集：`proprio` 形状应使用 `cfg.model.proprio_dim`

**Priority**: P0 — blocks real-robot integration.
**优先级**：P0 —— 会直接阻塞真实机器人集成。

---

### Issue 2: control_step() Return Semantics — CONFIRMED MEDIUM
### 问题 2：control_step() 返回语义 —— 已确认（中）

**Code Evidence:**
**代码证据：**

```python
# hybrid_vla_v2.py:561-574
action = runtime_state.current_chunk[:, runtime_state.chunk_step]  # [B, A] — COMPUTED
runtime_state.chunk_step += 1

# ... but returns the FULL CHUNK:
return ActionExpertOutput(
    velocity=torch.zeros_like(runtime_state.current_chunk),  # zeros, meaningless
    denoised_action=runtime_state.current_chunk,              # [B, H, A] full chunk
)
```

```python
# types.py:62-66
@dataclass
class ActionExpertOutput:
    velocity: Tensor                        # [B, H, A]
    denoised_action: Optional[Tensor] = None
```

**Assessment**: The current step action `[B, A]` is extracted then discarded.
**评估**：当前步动作 `[B, A]` 被提取出来后又被丢弃。
The caller receives the full chunk and must track step index externally.
调用方拿到的是整块 chunk，只能在外部自行维护步索引。
This is an API anti-pattern because callee already knows the answer.
这是典型 API 反模式，因为被调方已掌握当前步信息却未直接返回。

Additionally, `velocity=torch.zeros_like(...)` at inference is semantically meaningless and wastes memory.
此外，推理阶段返回 `velocity=torch.zeros_like(...)` 语义上无效且浪费内存。

**Recommended Fix**:
**建议修复：**

```python
# types.py — new inference output type
@dataclass
class ControlStepOutput:
    action: Tensor                     # [B, A] — the ONE action to execute NOW
    chunk: Optional[Tensor] = None     # [B, H, A] — full chunk for debugging
    chunk_step: int = 0
    semantic_refresh: bool = False
```

**Priority**: P1 — functional but misleading API.
**优先级**：P1 —— 功能可用，但接口语义误导。

---

### Issue 3: Grounder attention_mask Not Connected — CONFIRMED HIGH
### 问题 3：Grounder 的 attention_mask 未接入 —— 已确认（高）

**Code Evidence:**
**代码证据：**

Grounder **accepts** mask:
Grounder **支持接收** mask：
```python
# attention_grounder.py:201-202
def forward(self, backbone_hidden: Tensor,
            attention_mask: Optional[Tensor] = None) -> GrounderOutput:
```

Mask is **available** from backbone:
Mask 在 backbone 中 **已计算可用**：
```python
# qwen2vl_backbone.py:205-212
vision_mask = (input_ids == self.IMAGE_TOKEN_ID) | (input_ids == self.VIDEO_TOKEN_ID)
text_mask = attention_mask.bool() & ~vision_mask
return {
    "last_hidden_state": fused,
    "vision_mask": vision_mask,    # COMPUTED but NEVER USED downstream
    "text_mask": text_mask,        # COMPUTED but NEVER USED downstream
}
```

But **all 3 call sites** omit it:
但 **3 个调用点** 都没有传入：
```python
# hybrid_vla_v2.py:292 (forward_train, multi-refresh)
grounder_outputs.append(self.grounder(backbone_out["last_hidden_state"]))

# hybrid_vla_v2.py:302 (forward_train, single)
single_grounder_out = self.grounder(backbone_hidden)

# hybrid_vla_v2.py:472 (semantic_step, inference)
return self.grounder(backbone_out["last_hidden_state"])
```

**Impact Analysis**:
**影响分析：**
The grounder uses cross-attention where latent queries attend to backbone tokens.
Grounder 使用 cross-attention，让潜变量查询 backbone token。
Without a mask, padding and irrelevant tokens pollute attention scores.
若无 mask，padding 与无关 token 会污染注意力分数。
For Qwen2-VL sequence length 1024-2048, padding can be 20%-50%.
对 Qwen2-VL 的 1024-2048 序列，padding 可达 20%-50%。
This causes representation contamination and slower convergence.
这会造成表征污染并拖慢收敛。

**Fix**:
**修复建议：**
```python
# hybrid_vla_v2.py — all 3 call sites
attn_mask = backbone_out.get("text_mask", batch["attention_mask"].bool())
self.grounder(backbone_out["last_hidden_state"], attention_mask=attn_mask)
```

**Decision Point**: Prefer `attention_mask.bool()` to keep both vision and text (mask only padding).
**决策点**：更推荐 `attention_mask.bool()`，保留视觉与文本 token，仅屏蔽 padding。

**Priority**: P0 — silent training quality degradation.
**优先级**：P0 —— 会造成静默训练质量下降。

---

### Issue 4: Batch Schema Validation — CONFIRMED MEDIUM
### 问题 4：Batch 结构校验缺失 —— 已确认（中）

**Code Evidence:**
**代码证据：**

`forward_train()` pulls from batch with zero validation.
`forward_train()` 从 batch 取字段时没有任何校验。

```python
batch["actions"]      # assumed [B, T, H, A]
batch["proprio"]      # assumed [B, T, P]
batch["prev_actions"] # assumed [B, T, A]
batch["input_ids"]    # assumed [B, L]
batch["attention_mask"]  # assumed [B, L]
```

Optional keys are also consumed without schema enforcement.
可选字段同样被直接使用，没有结构约束。

```python
batch.get("step_weights", None)
batch.get("phase_labels", None)
batch.get("affordance_labels", None)
batch.get("semantic_refresh_steps", None)
batch.get("embodiment_id", None)
```

The `vla_hybrid_v2/data/` directory contains only an empty `__init__.py`.
`vla_hybrid_v2/data/` 目录只有空的 `__init__.py`。

**Assessment**: This is the classic “model-forward works, data integration explodes” pattern.
**评估**：这是典型“模型前向能跑，数据接入就爆炸”的模式。

**Recommendation**: Add a lightweight `validate_batch()` at `forward_train()` entry.
**建议**：在 `forward_train()` 入口加入轻量 `validate_batch()`。
No need for full TypedDict refactor; runtime assertions are enough initially.
初期不必做完整 TypedDict 重构，运行时断言即可。

**Priority**: P1 — should be fixed with Issue 1 and data pipeline.
**优先级**：P1 —— 建议与问题 1 及数据管线同步修。

---

### Issue 5: Real Data Pipeline — CONFIRMED HIGH
### 问题 5：真实数据管线缺失 —— 已确认（高）

**Code Evidence:**
**代码证据：**

```python
# vla_hybrid_v2/data/__init__.py
"""HybridVLA v2 data pipeline."""
# (empty)
```

```python
# config.py:283-306 — DataConfig
format: Optional[str] = None
paths: List[str] = field(default_factory=list)
data_dir: Optional[str] = None
...
camera_keys: List[str] = field(default_factory=lambda: [
    "agentview_rgb", "wrist_rgb", "overhead_rgb",
])
```

None of these DataConfig fields are consumed anywhere.
这些 DataConfig 字段在代码中均未被消费。
The only dataset is `DummyVLADataset` in `scripts/train_smoke_test.py`.
目前唯一数据集是 `scripts/train_smoke_test.py` 里的 `DummyVLADataset`。

**Assessment**: This is the largest gap between a model skeleton and a trainable system.
**评估**：这是“模型骨架”与“可训练系统”之间最大的缺口。
Without this, other optimizations cannot be validated on real data.
若不补齐，其他优化都无法在真实数据上验证。

**Priority**: P0 — project-level blocker.
**优先级**：P0 —— 项目级阻塞项。

---

### Issue 6: Train/Infer Config Coupling — CONFIRMED MEDIUM
### 问题 6：训练/推理配置耦合 —— 已确认（中）

**Code Evidence:**
**代码证据：**

```python
# hybrid_vla_v2.py:504-505 (inside control_step)
medium_update = (runtime_state.temporal_state.steps_since_medium
                 >= self.cfg.train.medium_update_stride - 1)
```

Inference logic reads a training config field.
推理逻辑读取了训练配置字段。

```python
# config.py:268-280
class InferConfig:
    control_hz: float = 50.0
    semantic_hz: float = 12.5
    medium_hz: float = 25.0
    execution_horizon: int = 8
```

These infer-time fields exist but are mostly unused.
这些推理配置字段虽然存在，却基本未被使用。

**Assessment**: Current behavior is accidentally correct only when train/infer settings coincide.
**评估**：当前行为只是在训练/推理参数碰巧一致时“意外正确”。
If deployment frequency changes, behavior diverges silently.
一旦部署频率变化，行为会静默偏离预期。

**Fix**:
**修复建议：**
Add infer-specific stride fields, or derive strides from Hz values.
在 `InferConfig` 增加推理步幅字段，或由 Hz 自动换算。

```python
medium_stride = max(1, round(self.cfg.infer.control_hz / self.cfg.infer.medium_hz))
```

**Priority**: P1 — correctness risk in deployment.
**优先级**：P1 —— 部署正确性风险。

---

### Issue 7: step_weights/RTC/FASTER — CONFIRMED MEDIUM
### 问题 7：step_weights/RTC/FASTER 仅配置未实现 —— 已确认（中）

**Code Evidence:**
**代码证据：**

```python
# flow_matching.py:16-21
def forward(self, velocity_pred, x_0, x_1, t, step_weights=None):
    ...
    if step_weights is not None:
        loss = loss * step_weights.unsqueeze(-1)
```

```python
# hybrid_vla_v2.py:435
step_weights = batch.get("step_weights", None)  # ALWAYS None
```

```yaml
# configs/train/stage_c.yaml:29-38
rtc:
  enable: true
  execution_horizon: 8
  overlap_ratio: 0.333
faster:
  enable: true
  near_ratio: 0.3
  near_steps: 2
  far_steps: 8
```

Neither `rtc.enable` nor `faster.enable` is consumed in training loop.
训练循环中并未消费 `rtc.enable` 或 `faster.enable`。

**Assessment**: This is “config nouns without code”.
**评估**：这属于“配置名词存在，但实现缺席”。
The loss path supports step weights structurally, but generation logic is missing.
损失函数路径已支持 step weights，但权重生成逻辑完全缺失。

**Recommendation**:
**建议：**
1. Remove these config knobs temporarily with explicit TODO notes, or
1）暂时移除这类配置并显式标注 TODO，或
2. Implement a minimal step-weight schedule (e.g., exponential decay).
2）先实现最小可用权重调度（例如指数衰减）。

**Priority**: P2 — blocks Stage C quality goals.
**优先级**：P2 —— 会阻塞 Stage C 目标。

---

### Issue 8: Denoising Formula Bug — CONFIRMED HIGH (BUG)
### 问题 8：去噪公式错误 —— 已确认（高，真实 Bug）

**Code Evidence:**
**代码证据：**

Flow interpolation (correct):
流插值（正确）：
```python
# flow_matching.py:29-32
x_t = (1 - t) * x_0 + t * x_1
```

Loss target (correct):
损失目标（正确）：
```python
# flow_matching.py:17
target_velocity = x_1 - x_0
```

Denoising in `forward_train` (BUG):
`forward_train` 中去噪公式（错误）：
```python
# hybrid_vla_v2.py:443
expert_continuous = noisy_actions + expert_out.velocity
```

**Mathematical Proof of Bug:**
**错误的数学证明：**
Given `x_t = (1-t)*x_0 + t*x_1` and `v = x_1 - x_0`,
已知 `x_t = (1-t)*x_0 + t*x_1`，且 `v = x_1 - x_0`，
we derive `x_1 = x_t + (1-t)*v`.
可推得 `x_1 = x_t + (1-t)*v`。
So the current `x_t + v` is incorrect.
因此当前实现 `x_t + v` 是错误的。

**Error magnitude**: error equals `t*v`.
**误差幅度**：误差项为 `t*v`。
With `t` near 0.5, typical error is about `0.5 * ||v||`.
当 `t` 聚集在 0.5 附近时，典型误差约为 `0.5 * ||v||`。

**Fix**:
**修复代码：**
```python
expert_continuous = noisy_actions + (1.0 - flow_t[:, None, None]) * expert_out.velocity
```

**Priority**: P0 — direct training-target bias.
**优先级**：P0 —— 直接导致训练目标偏置。

---

### Issue 9: Multi-Camera Fake Support — CONFIRMED LOW
### 问题 9：多相机“伪支持” —— 已确认（低）

**Code Evidence:**
**代码证据：**

Docstring claims multi-camera support.
文档字符串声称支持多相机。
Config defines multi-camera fields.
配置中也定义了多相机字段。
But `forward_semantic()` takes a single `pixel_values` tensor with no camera loop/fusion.
但 `forward_semantic()` 实际只接收单个 `pixel_values`，没有相机循环与融合逻辑。

**Assessment**: Config and docs are ahead of implementation.
**评估**：配置和文档领先于实现。

**Recommendation**: Mark as TODO in docs/config, avoid rushed implementation.
**建议**：在文档/配置中明确标记 TODO，不建议仓促实现。

**Priority**: P3 — documentation honesty and expectation management.
**优先级**：P3 —— 主要是文档诚实性与预期管理问题。

---

### Issue 10: God-Class Refactoring — VALID BUT LOW PRIORITY
### 问题 10：God-Class 重构 —— 关注点合理但优先级低

**Code Evidence**: `hybrid_vla_v2.py` is ~580 lines with multiple responsibilities.
**代码证据**：`hybrid_vla_v2.py` 约 580 行，承担多个职责。

**Assessment**: This size is still acceptable for current model-assembly stage.
**评估**：在当前模型组装阶段，这个体量仍可接受。
Premature abstraction may increase churn later.
过早抽象反而可能提高后续重构成本。

**Priority**: P3 — revisit when complexity truly blocks iteration.
**优先级**：P3 —— 复杂度真正阻碍迭代时再处理。

---

## Additional Issues Found (Not in User's 10 Points)
## 额外发现（不在原 10 点内）

### A1. res_scale Weight Decay Risk (STILL OPEN)
### A1. `res_scale` 权重衰减风险（仍未解决）

```python
# mamba_core.py:107
self.res_scale = nn.Parameter(torch.ones(1))
```

`res_scale` will be decayed by AdamW by default.
`res_scale` 默认会被 AdamW 权重衰减。
Long training can drive it toward 0 and suppress residual branches.
长程训练可能将其推向 0，进而抑制残差分支。

**Fix**: Exclude `res_scale` from `weight_decay` param group.
**修复建议**：在优化器参数分组中排除 `res_scale` 的权重衰减。

```python
no_decay = [n for n, _ in model.named_parameters() if "res_scale" in n or "bias" in n]
```

**Priority**: P1 — stability risk at scale.
**优先级**：P1 —— 大规模训练稳定性风险。

### A2. `id()` Semantic Refresh Detection is Fragile
### A2. 使用 `id()` 判断语义刷新不可靠

```python
# hybrid_vla_v2.py:490
semantic_refresh = (id(semantic_summary) != runtime_state.last_semantic_id)
```

`id()` is memory-address-based and can fail under object reuse/caching patterns.
`id()` 基于内存地址，遇到对象复用/缓存时会出现误判。

**Fix**: Use a monotonic counter or content hash.
**修复建议**：改为单调计数器或内容哈希。

```python
# In RuntimeCache: refresh_counter: int = 0
runtime_state.refresh_counter += 1
```

**Priority**: P1 — subtle inference correctness issue.
**优先级**：P1 —— 细微但真实的推理正确性风险。

### A3. Double `.detach()` in ActionConsistencyLoss
### A3. ActionConsistencyLoss 中存在双重 `.detach()`

`expert_continuous` is detached twice.
`expert_continuous` 被重复 `detach` 两次。
This is harmless but indicates unclear gradient-barrier ownership.
虽然无害，但反映梯度阻断职责边界不清晰。

**Priority**: P3 — code hygiene.
**优先级**：P3 —— 代码整洁性问题。

### A4. `fast_continuous` Variable Used but Not Shown
### A4. `fast_continuous` 变量被使用但未在审计片段中展示来源

Need to verify construction path of `fast_continuous`.
需要补查 `fast_continuous` 的构造路径是否完整正确。

---

## Priority Summary
## 优先级汇总

### P0 — Must Fix Before Training on Real Data
### P0 —— 真实数据训练前必须修复

| # | Issue | Risk |
|---|-------|------|
| 8 | Denoising formula `x_t + v` → `x_t + (1-t)*v` | Math bug, biases consistency target |
| 1 | proprio_dim decoupling | Blocks real robot data |
| 3 | Grounder attention_mask | Silent training quality loss |
| 5 | Real data pipeline | Project blocks entirely |

中文对照：P0 包含去噪公式、proprio 解耦、grounder mask、真实数据管线，均为关键阻塞项。

### P1 — Must Fix Before Serious Training
### P1 —— 进入严肃训练前应修复

| # | Issue | Risk |
|---|-------|------|
| A1 | res_scale weight_decay exclusion | Training instability at scale |
| A2 | `id()` semantic refresh fragility | Inference correctness |
| 2 | control_step return type | Robot integration API |
| 4 | Batch schema validation | Silent shape errors |
| 6 | Train/Infer config decoupling | Wrong inference behavior |

中文对照：P1 主要涉及稳定性、接口语义、训练/推理解耦与数据契约。

### P2 — Before Stage C
### P2 —— Stage C 前修复

| # | Issue | Risk |
|---|-------|------|
| 7 | step_weights/RTC/FASTER | Stage C non-functional |

中文对照：P2 聚焦 Stage C 功能闭环。

### P3 — Future Cleanup
### P3 —— 后续清理项

| # | Issue | Risk |
|---|-------|------|
| 9 | Multi-camera doc cleanup | Misleading documentation |
| 10 | God-class refactoring | Future maintainability |
| A3 | Double detach cleanup | Code hygiene |

中文对照：P3 侧重文档一致性与工程可维护性。

---

## Updated Scoring (v0.9.1)
## 更新评分（v0.9.1）

The scoring below reflects the codebase as-is.
以下评分反映当前代码库“修复前”的真实状态。

| Dimension | v0.9 | Adjusted | Delta | Justification |
|-----------|------|----------|-------|---------------|
| Design coherence | 8.5 | **7.5** | -1.0 | proprio coupling, fake multi-cam, train/infer coupling |
| Correctness | 9.5 | **8.5** | -1.0 | Denoising formula bug (Issue 8), id() fragility (A2) |
| Completeness | 5.5 | **4.5** | -1.0 | No data pipeline, no batch contract, grounder mask missing |
| Training stability | 9.0 | **8.0** | -1.0 | res_scale decay risk, biased consistency target |
| Scalability | 7.0 | **7.0** | — | No new scalability findings |
| Performance | 6.0 | **6.0** | — | Chunk caching remains valid |
| Production readiness | 5.5 | **4.5** | -1.0 | API semantics unclear, no real data, no evaluation |
| **Weighted avg** | **7.3** | **6.5** | -0.8 | |

**Note**: Lower score means broader audit coverage, not code regression.
**说明**：分数下降代表审计覆盖更完整，并不代表代码回退。

### Projected Score After P0+P1 Fixes
### 完成 P0+P1 后的预估分

| Dimension | Current | After Fixes | Delta |
|-----------|---------|-------------|-------|
| Design coherence | 7.5 | **8.5** | +1.0 |
| Correctness | 8.5 | **9.5** | +1.0 |
| Completeness | 4.5 | **6.5** | +2.0 |
| Training stability | 8.0 | **9.0** | +1.0 |
| Production readiness | 4.5 | **6.5** | +2.0 |
| **Weighted avg** | **6.5** | **7.8** | +1.3 |

中文对照：若完成 P0+P1，综合分预计可由 6.5 提升到 7.8。

---

## Recommended Fix Order
## 推荐修复顺序

```
Issue 8 (denoising formula)     — 1 line change, instant correctness gain
  ↓
Issue 3 (grounder mask)         — 3 call sites, 5 minutes
  ↓
Issue 1 (proprio_dim)           — config + 2 projections + dummy data
  ↓
Issue A1 (res_scale no_decay)   — optimizer param group change
  ↓
Issue A2 (refresh counter)      — replace id() with counter
  ↓
Issue 2 (control_step output)   — new dataclass + return change
  ↓
Issue 6 (infer config)          — add stride fields, rewire control_step
  ↓
Issue 4 (batch validation)      — validate_batch() function
  ↓
Issue 5 (data pipeline)         — minimal real loader
  ↓
Issue 7 (step_weights)          — implement or remove
```

中文对照（顺序说明）：
先修数学正确性（Issue 8）与关键掩码链路（Issue 3），再处理输入维度与稳定性（Issue 1/A1/A2），随后统一接口与配置（Issue 2/6/4），最后补齐数据管线与 Stage C 策略（Issue 5/7）。

---

## 中文摘要

### 核心发现

用户提出的 10 个问题经代码逐一验证，**全部确认存在**。其中 4 个为 P0 级别：

1. **去噪公式 Bug**（Issue 8）：`expert_continuous = x_t + v` 应为 `x_t + (1-t)*v`。这是数学错误，不是设计偏好。流匹配的线性插值 `x_t = (1-t)*x_0 + t*x_1` 下，从 `x_t` 恢复 `x_1` 需要乘以 `(1-t)` 因子。当前公式使一致性损失的监督目标系统性偏移，影响离散头与连续头的协同训练。

2. **proprio_dim 耦合**（Issue 1）：`proprio_proj` 使用 `ecfg.action_dim` 而非独立的 `proprio_dim`。真实机器人的本体感觉维度几乎从不等于动作维度（如 Franka: proprio=23, action=7）。

3. **Grounder 注意力掩码缺失**（Issue 3）：`qwen2vl_backbone.py` 计算了 `vision_mask` 和 `text_mask` 并返回，但 `hybrid_vla_v2.py` 的 3 个 grounder 调用点全部未传递。填充 token 悄悄污染所有 grounder 表征。

4. **真实数据管线**（Issue 5）：`vla_hybrid_v2/data/` 为空壳，`DataConfig` 的字段无代码消费。

### 额外发现

- **A1**: `res_scale` 作为 `nn.Parameter` 参与 AdamW 权重衰减，长期训练会被推向 0
- **A2**: `id()` 检测语义刷新不可靠（内存地址复用、同一对象修改等场景）
- **A3**: `expert_continuous` 被双重 `.detach()`，冗余但无害

### 修复优先级

**先修 1 行**：去噪公式（Issue 8），即时正确性收益。
**再修 3 行**：grounder mask（Issue 3），即时训练质量收益。
**然后修输入契约**：proprio_dim + batch schema（Issue 1 + 4）。
**最后搭管线**：真实数据 loader（Issue 5）。

### 评分调整

v0.9 评分过于乐观（7.3），修正后为 **6.5**（审计面更完整）。完成 P0+P1 修复后预计达 **7.8**。降分不代表代码退步，而是审计覆盖面扩大后暴露了此前未检查的跨模块问题。
