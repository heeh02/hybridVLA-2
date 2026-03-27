# HybridVLA v2 — v0.9.1 Code Audit Report

Based on user's 10-point analysis, this document verifies each claim against source code, assesses severity, and identifies additional issues.

---

## Verification Status Summary

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

---

## Detailed Verification

### Issue 1: proprio_dim Coupling to action_dim — CONFIRMED HIGH

**Code Evidence:**

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

**Assessment**: User's analysis is 100% correct. In real robotics:
- **Franka Panda**: proprio = 23 (7 joint pos + 7 joint vel + 7 joint torque + 1 gripper + 1 gripper vel), action = 7 (delta joint pos)
- **ALOHA**: proprio = 14 (joint pos), action = 14 (matches by coincidence)
- **Mobile Manipulators**: proprio = 20+ (base vel + arm + hand), action = 7-8

Hardcoding `proprio_dim == action_dim` will silently produce wrong projections on most platforms.

**Fix Locations**:
- `config.py`: Add `proprio_dim: int = 14` to `ModelConfig` (not `ActionExpertConfig`, since proprio is a model-level input, not expert-specific)
- `hybrid_vla_v2.py:137`: `self.proprio_proj = nn.Linear(mcfg.proprio_dim, d_core)`
- `hybrid_vla_v2.py:138`: `self.prev_action_proj = nn.Linear(ecfg.action_dim, d_core)` (keep — this IS action dim)
- Dummy dataset: Use `cfg.model.proprio_dim` for proprio shape

**Priority**: P0 — blocks real-robot integration.

---

### Issue 2: control_step() Return Semantics — CONFIRMED MEDIUM

**Code Evidence:**

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

**Assessment**: The current step action `[B, A]` is extracted (line 561) then **discarded** — only used for history update. The caller receives the full chunk and must independently track which step to index. This is a classic API anti-pattern: the callee has the information, but the caller must reconstruct it.

Additionally, `velocity=torch.zeros_like(...)` at inference is semantically meaningless — it wastes memory creating a zero tensor of shape `[B, H, A]`.

**Recommended Fix**:

```python
# types.py — new inference output type
@dataclass
class ControlStepOutput:
    action: Tensor                     # [B, A] — the ONE action to execute NOW
    chunk: Optional[Tensor] = None     # [B, H, A] — full chunk for debugging
    chunk_step: int = 0
    semantic_refresh: bool = False
```

**Priority**: P1 — functional but misleading API. Must fix before real robot integration.

---

### Issue 3: Grounder attention_mask Not Connected — CONFIRMED HIGH

**Code Evidence:**

Grounder **accepts** mask:
```python
# attention_grounder.py:201-202
def forward(self, backbone_hidden: Tensor,
            attention_mask: Optional[Tensor] = None) -> GrounderOutput:
```

Mask is **available** from backbone:
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
```python
# hybrid_vla_v2.py:292 (forward_train, multi-refresh)
grounder_outputs.append(self.grounder(backbone_out["last_hidden_state"]))

# hybrid_vla_v2.py:302 (forward_train, single)
single_grounder_out = self.grounder(backbone_hidden)

# hybrid_vla_v2.py:472 (semantic_step, inference)
return self.grounder(backbone_out["last_hidden_state"])
```

**Impact Analysis**:

The grounder uses cross-attention (CrossAttentionLayer in attention_grounder.py) where latent queries attend to backbone tokens. Without a mask:
- **Padding tokens** (attention_mask=0) contribute to attention scores
- **Vision tokens** participate equally with text tokens (no selective attention)
- The 44 auxiliary tokens + 24 object slots + 24 compressed slots + 4 special tokens all attend to noise

For Qwen2-VL with typical sequence lengths of 1024-2048 tokens, padding can be 20-50% of the sequence. This means the grounder's learned representations are contaminated with padding features.

**Why this doesn't crash but still hurts**: Padding tokens tend to have near-zero hidden states (post-norm), so the attention weights on them are small but non-zero. The effect is a systematic bias toward the zero vector, which manifests as:
- Slower convergence of grounder training
- Noisier object slot representations
- Reduced phase/affordance classification accuracy

**Fix**:
```python
# hybrid_vla_v2.py — all 3 call sites
attn_mask = backbone_out.get("text_mask", batch["attention_mask"].bool())
self.grounder(backbone_out["last_hidden_state"], attention_mask=attn_mask)
```

**Decision Point**: Whether to pass `text_mask` (text-only tokens) or `attention_mask` (all non-padding) depends on whether vision tokens should participate in grounding. For a VLA where visual grounding is critical, passing `attention_mask.bool()` (include vision) is more appropriate. `text_mask` would be wrong — it would exclude the very visual features the grounder needs.

Recommended: `attention_mask.bool()` (mask out padding, keep both text and vision).

**Priority**: P0 — silent training quality degradation.

---

### Issue 4: Batch Schema Validation — CONFIRMED MEDIUM

**Code Evidence:**

`forward_train()` (lines 257-459) pulls from batch with **zero validation**:
```python
batch["actions"]      # assumed [B, T, H, A]
batch["proprio"]      # assumed [B, T, P] but actually [B, T, A] via dummy
batch["prev_actions"] # assumed [B, T, A]
batch["input_ids"]    # assumed [B, L]
batch["attention_mask"]  # assumed [B, L]
```

Optional keys use `.get()`:
```python
batch.get("step_weights", None)
batch.get("phase_labels", None)
batch.get("affordance_labels", None)
batch.get("semantic_refresh_steps", None)
batch.get("embodiment_id", None)
```

The `vla_hybrid_v2/data/` directory contains only an empty `__init__.py`.

**Assessment**: User is correct that this is a "model-forward works, data integration explodes" pattern. The **implicit batch contract** is:

| Key | Required | Expected Shape | Actual Constraint |
|-----|----------|---------------|-------------------|
| `actions` | Yes | `[B, T, H, A]` | Must match `chunk_horizon` and `action_dim` |
| `proprio` | Yes | `[B, T, P]` | Currently assumed `P == A` (Issue 1) |
| `prev_actions` | Yes | `[B, T, A]` | Must match `action_dim` |
| `input_ids` | Yes | `[B, L]` | Token IDs |
| `attention_mask` | Yes | `[B, L]` | Binary mask |
| `pixel_values` | Conditional | `[B, ...]` | Required if vision is used |
| `image_grid_thw` | Conditional | `[B, N, 3]` | Required with pixel_values |
| `phase_labels` | Optional | `[B, T]` | Integer class labels |
| `affordance_labels` | Optional | `[B, T]` | Float or int |
| `step_weights` | Optional | `[B, H]` | Per-horizon-step weights |
| `embodiment_id` | Optional | `[B]` | Long integer |
| `semantic_refresh_steps` | Optional | `List[int]` | Indices within [0, T) |

**Recommendation**: A `validate_batch()` function at `forward_train()` entry. Not a full TypedDict refactor — just runtime assertions that fail fast with clear messages.

**Priority**: P1 — should happen alongside Issue 1 and real data pipeline.

---

### Issue 5: Real Data Pipeline — CONFIRMED HIGH

**Code Evidence:**

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

The only dataset is `DummyVLADataset` in `scripts/train_smoke_test.py`.

**Assessment**: User is correct. This is the single largest gap preventing the project from being a "trainable system" vs. "model skeleton". Without addressing this, all other optimizations (res_scale, chunk caching, etc.) are untestable on real data.

**Priority**: P0 — the entire project blocks on this.

---

### Issue 6: Train/Infer Config Coupling — CONFIRMED MEDIUM

**Code Evidence:**

```python
# hybrid_vla_v2.py:504-505 (inside control_step)
medium_update = (runtime_state.temporal_state.steps_since_medium
                 >= self.cfg.train.medium_update_stride - 1)
                 #  ^^^^^^^^^^^^^^^ reads TRAINING config during INFERENCE
```

Meanwhile `InferConfig` defines but never uses:
```python
# config.py:268-280
class InferConfig:
    control_hz: float = 50.0     # UNUSED
    semantic_hz: float = 12.5    # UNUSED
    medium_hz: float = 25.0      # UNUSED
    execution_horizon: int = 8   # USED (line 486)
```

**Assessment**: The `medium_hz = 25.0` in InferConfig implies a stride of `control_hz / medium_hz = 50/25 = 2`, which happens to match `train.medium_update_stride = 2`. So **currently** the behavior is accidentally correct. But if a deployment changes control_hz to 100 Hz (keeping medium_hz at 25), the stride should become 4, yet the code would still use stride=2 from train config.

**Fix**: Add `medium_update_stride` and `semantic_refresh_stride` to `InferConfig`, or compute them from the Hz fields:

```python
# In control_step:
medium_stride = max(1, round(self.cfg.infer.control_hz / self.cfg.infer.medium_hz))
```

**Priority**: P1 — correctness risk when deployment config diverges from training.

---

### Issue 7: step_weights/RTC/FASTER — CONFIRMED MEDIUM

**Code Evidence:**

```python
# flow_matching.py:16-21 — step_weights consumer
def forward(self, velocity_pred, x_0, x_1, t, step_weights=None):
    ...
    if step_weights is not None:
        loss = loss * step_weights.unsqueeze(-1)  # expects [B, H] -> [B, H, 1]
```

```python
# hybrid_vla_v2.py:435 — step_weights source
step_weights = batch.get("step_weights", None)  # ALWAYS None (no generator)
```

```yaml
# configs/train/stage_c.yaml:29-38 — config exists
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

Neither `rtc.enable` nor `faster.enable` is checked anywhere in the training loop. The configuration is dead.

**Assessment**: The user's characterization of "config nouns without code" is accurate. The step_weights mechanism is structurally present (loss accepts them, forward_train passes them) but the **generation logic** is missing entirely. RTC and FASTER are pure config placeholders.

**Recommendation**: Either:
1. **Remove** the config entries with a `# TODO: Stage C` comment (honest)
2. **Implement** at least a basic step_weight generator (e.g., exponential decay `w_h = gamma^(H-h)`) as a minimal starting point

**Priority**: P2 — blocks Stage C but Stage A/B work without it.

---

### Issue 8: Denoising Formula Bug — CONFIRMED HIGH (BUG)

**Code Evidence:**

Flow interpolation (correct):
```python
# flow_matching.py:29-32
x_t = (1 - t) * x_0 + t * x_1   # x_0=noise, x_1=target
```

Loss target (correct):
```python
# flow_matching.py:17
target_velocity = x_1 - x_0      # v = x_1 - x_0
```

Denoising in forward_train (**BUG**):
```python
# hybrid_vla_v2.py:443
expert_continuous = noisy_actions + expert_out.velocity
#                   ^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^
#                   x_t             v_pred (estimates x_1 - x_0)
```

**Mathematical Proof of Bug:**

Given:
- `x_t = (1-t)*x_0 + t*x_1`
- `v = x_1 - x_0`

Solving for `x_1`:
```
x_t = (1-t)*x_0 + t*x_1
x_t = (1-t)*(x_1 - v) + t*x_1     [substitute x_0 = x_1 - v]
x_t = x_1 - (1-t)*v
x_1 = x_t + (1-t)*v                [CORRECT formula]
```

The code computes `x_t + v` instead of `x_t + (1-t)*v`.

**Error magnitude**: `x_t + v - x_1 = x_t + v - (x_t + (1-t)*v) = t*v`. The error is proportional to `t * v`. With logit-normal scheduling concentrating `t` around 0.5, the typical error is `~0.5 * ||v||`.

**Impact**: This biased `expert_continuous` is fed into `ActionConsistencyLoss` (line 444-450), which aligns discrete predictions against a wrong continuous target. The discrete head learns to match `x_t + v` instead of the true denoised action `x_1`. This creates a systematic alignment error between the two action heads.

**Fix**:
```python
# hybrid_vla_v2.py:443
expert_continuous = noisy_actions + (1.0 - flow_t[:, None, None]) * expert_out.velocity
```

**Verification**: At `t=0`: `x_0 + 1*v = x_0 + (x_1-x_0) = x_1` (correct). At `t=1`: `x_1 + 0*v = x_1` (correct). At `t=0.5`: `x_0.5 + 0.5*v` (correct — halfway point plus half the velocity recovers x_1).

**Priority**: P0 — mathematical bug affecting training quality.

---

### Issue 9: Multi-Camera Fake Support — CONFIRMED LOW

**Code Evidence:**

Docstring claims:
```python
# qwen2vl_backbone.py:6
# - Multi-camera support: processes each camera independently
```

Config exists:
```python
# config.py:52-58
class MultiCameraConfig:
    enable: bool = True
    num_cameras: int = 3
    camera_names: List[str] = field(default_factory=lambda: ["wrist", "shoulder", "overhead"])
```

But `forward_semantic()` takes single `pixel_values` tensor — no camera loop, no per-camera encoding, no camera embedding, no fusion.

**Assessment**: User is correct that this is "config ahead of code". The docstring and config create a false impression of capability.

**Recommendation**: Mark as `# TODO: not implemented` in config and docstring. Do NOT implement now — it's a significant architectural change that shouldn't be rushed.

**Priority**: P3 — documentation/config cleanup, not blocking.

---

### Issue 10: God-Class Refactoring — VALID BUT LOW PRIORITY

**Code Evidence**: `hybrid_vla_v2.py` is 580 lines with:
- `__init__` (module assembly)
- `_build_cond_prefix` (tensor manipulation)
- `forward_train` (~200 lines, training forward)
- `semantic_step` (backbone + grounder)
- `control_step` (~100 lines, inference)
- `init_runtime` (state initialization)

**Assessment**: 580 lines for a model assembly class is **not unreasonable** at this stage. Models like LLaVA's `LlavaLlamaForCausalLM` or RT-2's model class are similarly structured. The refactoring becomes valuable when:
- World model training is integrated
- Stage B/C have divergent forward paths
- Online rollout with environment feedback is added

Premature refactoring would create abstraction boundaries that may need to change.

**Priority**: P3 — revisit when complexity actually impedes development.

---

## Additional Issues Found (Not in User's 10 Points)

### A1. res_scale Weight Decay Risk (from v0.9 analysis, STILL OPEN)

```python
# mamba_core.py:107
self.res_scale = nn.Parameter(torch.ones(1))
```

`nn.Parameter` participates in AdamW weight decay by default. With `weight_decay=0.01`, a 1M-step training will push `res_scale` toward 0, effectively silencing residual branches in later layers.

**Fix**: Exclude `res_scale` from weight_decay group in optimizer construction:

```python
no_decay = [n for n, _ in model.named_parameters() if "res_scale" in n or "bias" in n]
```

**Priority**: P1 — affects training stability at scale.

### A2. `id()` Semantic Refresh Detection is Fragile

```python
# hybrid_vla_v2.py:490
semantic_refresh = (id(semantic_summary) != runtime_state.last_semantic_id)
```

Python's `id()` returns the memory address of an object. This has failure modes:
- If the caller passes the same tensor object (e.g., from a cached variable), no refresh is detected even if contents changed (e.g., in-place modification)
- If CPython reuses a memory address for a new tensor (common after GC), a different tensor could have the same `id()`, causing missed refresh

**Fix**: Use a monotonic counter or content hash instead:
```python
# In RuntimeCache: refresh_counter: int = 0
runtime_state.refresh_counter += 1  # caller increments when new observation arrives
```

**Priority**: P1 — subtle inference-time correctness risk.

### A3. Double `.detach()` in ActionConsistencyLoss

```python
# hybrid_vla_v2.py:449
continuous_actions=expert_continuous.detach(),

# consistency_loss.py:72
c = F.normalize(self.continuous_proj(continuous_actions.detach()), dim=-1)
```

`expert_continuous` is detached twice. Harmless but indicates unclear ownership of the gradient barrier. Should be in ONE place.

**Priority**: P3 — code hygiene.

### A4. `fast_continuous` Variable Used but Not Shown

```python
# hybrid_vla_v2.py:448
discrete_actions=fast_continuous,
```

Need to verify `fast_continuous` is properly constructed. It appears to be the fast-stream discrete action prediction, but its construction wasn't in the lines I audited.

---

## Priority Summary

### P0 — Must Fix Before Training on Real Data

| # | Issue | Risk |
|---|-------|------|
| 8 | Denoising formula `x_t + v` → `x_t + (1-t)*v` | Math bug, biases consistency target |
| 1 | proprio_dim decoupling | Blocks real robot data |
| 3 | Grounder attention_mask | Silent training quality loss |
| 5 | Real data pipeline | Project blocks entirely |

### P1 — Must Fix Before Serious Training

| # | Issue | Risk |
|---|-------|------|
| A1 | res_scale weight_decay exclusion | Training instability at scale |
| A2 | `id()` semantic refresh fragility | Inference correctness |
| 2 | control_step return type | Robot integration API |
| 4 | Batch schema validation | Silent shape errors |
| 6 | Train/Infer config decoupling | Wrong inference behavior |

### P2 — Before Stage C

| # | Issue | Risk |
|---|-------|------|
| 7 | step_weights/RTC/FASTER | Stage C non-functional |

### P3 — Future Cleanup

| # | Issue | Risk |
|---|-------|------|
| 9 | Multi-camera doc cleanup | Misleading documentation |
| 10 | God-class refactoring | Future maintainability |
| A3 | Double detach cleanup | Code hygiene |

---

## Updated Scoring (v0.9.1)

The scoring below reflects the codebase **as-is** (before fixing the issues identified here).

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

**Note**: This lower score doesn't mean v0.9 is worse than v0.7.2. It means the v0.9 scoring was too optimistic by not examining these cross-cutting concerns. The actual code quality is approximately the same — the scoring now reflects a more complete audit surface.

### Projected Score After P0+P1 Fixes

| Dimension | Current | After Fixes | Delta |
|-----------|---------|-------------|-------|
| Design coherence | 7.5 | **8.5** | +1.0 |
| Correctness | 8.5 | **9.5** | +1.0 |
| Completeness | 4.5 | **6.5** | +2.0 |
| Training stability | 8.0 | **9.0** | +1.0 |
| Production readiness | 4.5 | **6.5** | +2.0 |
| **Weighted avg** | **6.5** | **7.8** | +1.3 |

---

## Recommended Fix Order

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
