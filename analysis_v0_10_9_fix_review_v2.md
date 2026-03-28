# v0.10.9 修复二次验证 — Corrected Code Review

> **Date**: 2026-03-28
> **背景**: `analysis_v0_10_9_fix_review.md` 结论偏乐观，GPT 复核指出 EMA/FSDP 名称匹配和 per-module LR 仍然失效。本次严格复审。
> **标准**: 不做 "应该可以工作" 的推理。代码路径必须无歧义地正确。

---

## 0. 上一版 review 的错误

`analysis_v0_10_9_fix_review.md` 中以下结论是**错误的**：

| 上一版结论 | 实际情况 | 原因 |
|-----------|---------|------|
| "EMA/FSDP 已修复 ✅" | **❌ 未修复** | `use_orig_params=True` 不消除 `_fsdp_wrapped_module.` 前缀；shadow key 与 FSDP 参数名不匹配 |
| "Per-module LR scaling 正确 ✅" | **❌ 仍然失效** | 同一前缀问题使 `name.startswith("backbone")` 永远为 False |
| "评分 8.7" | **过高** | 两个 P1 未解决 |

错误根因：我假设 `use_orig_params=True` 使 `model.named_parameters()` 返回无前缀的原始名称。GPT 本地复现了名称不匹配。在 FSDP 顶层包装下，`named_parameters()` 返回的名称始终带有 `_fsdp_wrapped_module.` 前缀，不论 `use_orig_params` 设置。

---

## 1. 逐项严格复审

### 1.1 GPT-P1 [P0]: FSDP evaluate() 死锁 — **✅ 已修复，正确**

`train_unified.py:552-567`:
- 所有 rank 参与 evaluate ✅
- `dist.all_reduce(AVG)` 正确聚合 ✅
- 日志限 rank 0 ✅
- EMA apply/restore 包裹 evaluate ✅

**判定: CLOSED。**

---

### 1.2 GPT-P3 [P2]: Per-module gnorm timing — **✅ 已修复，正确**

`train_unified.py:525-532`:
- `_log_per_module_grad_norm` 在 L529（zero_grad 在 L532 之后）✅
- `next_step = global_step + 1` 正确处理 step 偏移 ✅

**判定: CLOSED。**

---

### 1.3 Claude-1 [P0-P1]: EMA/FSDP save/load — **❌ 未修复**

**问题核心**: FSDP 参数名前缀不匹配。

```
EMA shadow 初始化 (FSDP 前):
  shadow["backbone.multi_scale_adapter.gate.0.weight"] = tensor(...)
  shadow["grounder.blocks.0.cross_attn.norm_q.weight"] = tensor(...)
  shadow["action_expert.layers.0.norm.weight"] = tensor(...)

FSDP wrapping 后 model.named_parameters():
  "_fsdp_wrapped_module.backbone.multi_scale_adapter.gate.0.weight"
  "_fsdp_wrapped_module.grounder.blocks.0.cross_attn.norm_q.weight"
  "_fsdp_wrapped_module.action_expert.layers.0.norm.weight"
```

`ema.py:82-83`:
```python
for name, param in model.named_parameters():
    if param.requires_grad and name in self.shadow:  # ← ALWAYS FALSE
```

`name` = `"_fsdp_wrapped_module.backbone.xxx"`
`self.shadow` keys = `"backbone.xxx"`
→ **交集为空。EMA update/apply/restore 在多卡下是空操作。**

**影响**:
- `ema.update()` — 不更新任何 shadow 参数 → EMA 权重永远停留在初始值
- `ema.apply()` — 不替换任何参数 → eval 使用 base model（不是 EMA）
- `ema.restore()` — backup 为空 → 无操作
- `ema.state_dict()` — 保存的 shadow 是初始值（或上一次 resume 的值）
- 推理 `from_checkpoint()` 加载 ema.pt → 应用的是从未真正更新的 shadow → **推理权重错误**

**严重度: P1**（Stage B+ 多卡。Stage A EMA 禁用不受影响。单卡无 FSDP 不受影响。）

**判定: NOT FIXED。需要名称映射或前缀剥离。**

---

### 1.4 新发现 [P1]: Per-module LR scaling 在 FSDP 下失效

`train_unified.py:396-408`:
```python
for name, param in model.named_parameters():     # model 已经是 FSDP 包装后的
    if not param.requires_grad:
        continue
    if name.startswith("backbone"):               # ← FSDP 下 name 以 "_fsdp_wrapped_module." 开头
        group = "backbone"                         #    永远不进入此分支
        lr_scale = cfg.train.backbone_lr_scale     #    backbone_lr_scale = 0.1
    elif name.startswith("action_expert"):         # ← 同理，永远不进入
        group = "expert"
        lr_scale = cfg.train.expert_lr_scale       #    expert_lr_scale = 0.5
    else:
        group = "core"                             # ← 所有参数进入此分支
        lr_scale = 1.0                             #    所有参数使用 1.0× LR
```

**实际效果（8×H100）**:
- backbone LoRA 应该 0.1× LR → 实际 1.0× LR（**10× 过高**）
- action expert 应该 0.5× LR → 实际 1.0× LR（**2× 过高**）
- 所有参数使用同一 LR → **per-module LR 策略完全失效**

**影响**: backbone LoRA 以 10× 预期 LR 训练会导致：
- Stage A: backbone 表征被快速破坏，grounder/core 训练不稳定
- Stage B: expert 学习过快，与 core 的协调失调
- 训练可能不 crash，但收敛到差的局部最优

**严重度: P1** — 不崩溃但训练质量严重受损。

**注意**: 此问题在 v0.10.5 引入 per-module LR 时就存在，但因 v0.10.9 review 聚焦于 EMA 和 eval 而被遗漏。**单卡训练不受影响**（无 FSDP）。

**判定: NOT FIXED（也从未被正确发现）。**

---

### 1.5 GPT-P2 [P1]: 推理 EMA 加载 — **⚠️ 部分有效，有条件**

`libero_policy.py:235-246`:
```python
ema_state = torch.load(ema_path, map_location="cpu", weights_only=True)
shadow = ema_state["shadow"]
for name, param in model.named_parameters():     # 推理时 model 未经 FSDP
    if name in shadow:
        param.data.copy_(shadow[name])
```

推理时 `model = HybridVLAv2(cfg)` 未经 FSDP → 参数名无前缀 → **与 shadow key 匹配**。

**但**: shadow 内容是否正确？
- 单卡训练: EMA 正常更新 → shadow 有效 → 推理正确 ✅
- 多卡训练: EMA update 是空操作（§1.3）→ shadow 停留在初始值 → **推理加载的是从未更新的权重** ❌

**判定: 代码本身正确，但依赖上游 EMA 训练正确性。多卡训练出的 ema.pt 无效。**

---

### 1.6 新发现 [P2]: EMA shadow 初始化早于 cross-stage resume

`train_unified.py` 执行顺序:
```
L370: model.to(device)
L374: ema = EMAModel(model, ...)       ← shadow 克隆当前（未 resume）权重
L386: model = wrap_fsdp(model, ...)
L390: optimizer = ...
L431: load_checkpoint(resume_from, model)  ← model 权重更新为上一 stage
L446: auto_resume(output_dir, ..., ema=ema)  ← 仅同 stage resume 能修复 ema
```

**场景**: Fresh Stage B run (resume_from = Stage A checkpoint)
1. `ema.shadow` = 随机初始化/预训练权重（Qwen2-VL）
2. `load_checkpoint` 将 model 更新为 Stage A 训练后的权重
3. `auto_resume` 找不到同 stage checkpoint → 不加载 ema.pt
4. `ema.shadow` 仍然是步骤 1 的旧权重

**影响**: EMA 的初始 shadow 与 model 不同步。前 ~1000 步（`1/(1-0.999)` ≈ 1000 步 EMA 半衰期）EMA 权重受旧 shadow 污染。

**严重度: P2** — EMA 最终会收敛到正确值，但前 ~1000 步的 shadow 包含噪声。对于 200K 步的 Stage B 影响有限。

**判定: NOT FIXED。**

---

### 1.7 GPT-P1b [P1]: Normalizer stats 覆盖 — **⚠️ 仅 warning，未阻止**

`libero_policy.py:82-90` 新增了 warning，但 `_candidate_stats_dirs()`:112-113 仍将 `cfg.data.normalizer_stats_dir` 放在最高优先级。

**判定: 部分缓解（可见性提升），未根治。** P2（降级——有 warning 后用户可主动规避）。

---

### 1.8 Claude-5 [P2]: Action clipping — **⚠️ 有效但语义不精确**

`libero_policy.py:409-410`:
```python
lo, hi = self.cfg.model.heads.action_range    # 模型空间范围，不是环境空间范围
action_env = action_env.clamp(lo, hi)
```

`heads.action_range` = `(-1.0, 1.0)` — 这是 FAST 离散化目标范围（模型空间）。
LIBERO env 动作范围也恰好是 `[-1, 1]`。
但语义上 `heads.action_range` 是模型归一化范围，不是环境动作限制。

**判定: 对 LIBERO 功能正确，语义不精确。P3。**

---

### 1.9 Claude-6 [P3]: Hardcoded 448×448 — **未修改，P3**

**判定: 可接受。**

---

## 2. 完整问题状态表

| ID | 问题 | 原始严重度 | 当前状态 | 判定 |
|----|------|:---------:|:--------:|:----:|
| GPT-P1 | FSDP evaluate 死锁 | P0 | **✅ CLOSED** | 正确修复 |
| GPT-P3 | Gnorm timing | P2 | **✅ CLOSED** | 正确修复 |
| Claude-4 | Eval EMA apply/restore | P2 | **✅ CLOSED** | 包含在 GPT-P1 修复中 |
| Claude-3 | Val DistributedSampler | P2 | **✅ CLOSED** | 正确修复 |
| **Claude-1** | **EMA/FSDP 名称不匹配** | **P0-P1** | **❌ NOT FIXED** | shadow key 与 FSDP 参数名前缀不匹配 |
| **NEW** | **Per-module LR FSDP 失效** | **P1** | **❌ NOT FIXED** | startswith 检查被 FSDP 前缀破坏 |
| **NEW** | **EMA shadow stale cross-stage** | **P2** | **❌ NOT FIXED** | EMA 在 resume_from 之前初始化 |
| GPT-P2 | 推理 EMA 加载 | P1 | **⚠️ 代码正确，依赖上游** | 多卡 ema.pt 内容无效 |
| GPT-P1b | Normalizer 覆盖 | P1→P2 | **⚠️ 部分缓解** | 有 warning 但未阻止 |
| Claude-5 | Action clipping | P2→P3 | **⚠️ 功能正确，语义不精确** | LIBERO 有效 |
| Claude-6 | Hardcoded 448 | P3 | 未修改 | 可接受 |

**总结: 4 项 CLOSED，3 项 NOT FIXED（含 2 个 P1），4 项部分/条件性解决。**

---

## 3. FSDP 前缀问题深度分析

这是此次 review 最核心的问题。**FSDP 顶层包装使 `named_parameters()` 返回带 `_fsdp_wrapped_module.` 前缀的名称**。这个单一根因同时破坏了三个系统：

### 3.1 影响范围

| 系统 | 代码位置 | 使用方式 | 是否受影响 |
|------|---------|---------|:---------:|
| EMA update | `ema.py:82-83` | `name in self.shadow` | **是** |
| EMA apply | `ema.py:89-90` | `name in self.shadow` | **是** |
| EMA restore | `ema.py:97-98` | `name in self.backup` | **是** |
| Optimizer LR groups | `train_unified.py:400-408` | `name.startswith("backbone")` | **是** |
| Per-module gnorm | `train_unified.py:280` | `getattr(model, mod_name)` | **部分**（FSDP `__getattr__` 可能转发）|
| Checkpoint load | `checkpointing.py:135-142` | FSDP state_dict_type context | **否**（正确处理）|
| Inference EMA load | `libero_policy.py:242-244` | 无 FSDP 包装 | **否** |

### 3.2 根因确认

```python
# 在 wrap_fsdp 之前:
for name, _ in model.named_parameters():
    print(name)
# → "backbone.multi_scale_adapter.gate.0.weight"
# → "grounder.blocks.0.cross_attn.norm_q.weight"

# 在 wrap_fsdp 之后 (use_orig_params=True):
model = FSDP(model, use_orig_params=True, ...)
for name, _ in model.named_parameters():
    print(name)
# → "_fsdp_wrapped_module.backbone.multi_scale_adapter.gate.0.weight"
# → "_fsdp_wrapped_module.grounder.blocks.0.cross_attn.norm_q.weight"
```

`use_orig_params=True` 保留了参数对象的同一性（不 flatten），但不改变模块层次结构中 FSDP wrapper 引入的前缀。

### 3.3 修复方向

**方案 A（最小侵入）**: 在 EMA 和 optimizer 代码中剥离 FSDP 前缀

```python
# 工具函数
_FSDP_PREFIX = "_fsdp_wrapped_module."
def _strip_fsdp_prefix(name: str) -> str:
    """Strip all FSDP wrapper prefixes from parameter name."""
    while _FSDP_PREFIX in name:
        name = name.replace(_FSDP_PREFIX, "")
    return name
```

EMA update 修改:
```python
def update(self, model, step):
    with _maybe_summon_full_params(model, writeback=False):
        for name, param in model.named_parameters():
            clean_name = _strip_fsdp_prefix(name)
            if param.requires_grad and clean_name in self.shadow:
                self.shadow[clean_name].lerp_(param.data, 1.0 - decay)
```

Optimizer grouping 修改:
```python
for name, param in model.named_parameters():
    clean = _strip_fsdp_prefix(name)
    if clean.startswith("backbone"):
        group = "backbone"
    elif clean.startswith("action_expert"):
        group = "expert"
    else:
        group = "core"
```

**方案 B（更稳健）**: EMA 内部维护名称映射 dict

```python
def __init__(self, model, ...):
    self.shadow = {}
    self._name_map = {}  # fsdp_name → shadow_name
    for name, param in model.named_parameters():
        if param.requires_grad:
            self.shadow[name] = param.data.clone()

def build_name_map(self, model):
    """Call after FSDP wrapping to build name mapping."""
    for fsdp_name, _ in model.named_parameters():
        clean = _strip_fsdp_prefix(fsdp_name)
        if clean in self.shadow:
            self._name_map[fsdp_name] = clean
```

**推荐**: 方案 A。更简单，不需要额外调用。

---

## 4. EMA cross-stage 初始化问题

### 当前执行顺序
```
EMA init → FSDP wrap → optimizer → cross-stage load → auto_resume
```

### 正确执行顺序
```
model.to(device) → cross-stage load → EMA init → FSDP wrap → optimizer → auto_resume
```

**修复**: 将 `resume_from` 加载移到 EMA 初始化之前。

```python
# 1. Cross-stage resume FIRST
model = model.to(device)
if cfg.train.resume_from:
    load_checkpoint(_resume_path, model, strict=False)

# 2. EMA from resumed model
ema = EMAModel(model, ...) if cfg.model.ema.enable else None

# 3. FSDP wrapping
if cfg.train.fsdp and get_world_size() > 1:
    model = wrap_fsdp(model, ...)

# 4. Optimizer (with FSDP-prefix-aware grouping)
...

# 5. Auto-resume (same-stage, loads optimizer/scheduler/ema)
start_step, start_epoch = auto_resume(...)
```

---

## 5. 更正评分

| # | 维度 | 上一版 (错误) | 本次更正 | 理由 |
|---|------|:-----------:|:--------:|------|
| 1 | 设计一致性 | 8.5 | **8.5** | 无变化 |
| 2 | 正确性 | 9.5 | **8.0** | EMA/FSDP 未修复 + LR 分组失效 |
| 3 | 完备性 | 9.0 | **9.0** | 无变化 |
| 4 | 训练稳定性 | 9.5 | **7.5** | 死锁修复 ✅，但 LR 分组失效使训练质量不可预测 |
| 5 | 可扩展性 | 7.5 | **7.5** | 无变化 |
| 6 | 性能设计 | 6.5 | **6.5** | 无变化 |
| 7 | 生产就绪度 | 9.0 | **7.5** | 多卡 EMA 无效 + LR 失效 |
| 8 | 代码质量 | 9.0 | **8.0** | EMA FSDP 方案架构合理但实现有缺陷 |
| 9 | 文档 | 6.0 | **6.0** | 无变化 |
| 10 | 测试 | 7.0 | **6.5** | 无 FSDP 多卡名称匹配测试 |
| | **加权均分** | **~~8.7~~** | **7.7** | |

---

## 6. 训练就绪度（更正）

| 场景 | 判定 | 阻塞因素 |
|------|:----:|---------|
| **单卡 Stage A** | **✅ 可以** | 无 FSDP，所有功能正常 |
| **单卡 Stage B/C** | **✅ 可以** | EMA 正常，LR 正常 |
| **8×H100 Stage A** | **⚠️ 可以但 LR 失效** | 死锁已修复；backbone 0.1× LR 未生效 → 所有参数 1.0× LR |
| **8×H100 Stage B** | **❌ 有风险** | LR 失效 + EMA 无效（空操作）+ EMA shadow 可能 stale |
| **8×H100 Stage C** | **❌ 有风险** | 同 Stage B |
| **LIBERO 推理 (单卡训练后)** | **✅ 可以** | EMA 正确加载 |
| **LIBERO 推理 (多卡训练后)** | **⚠️ EMA 无效** | ema.pt 内容为初始权重 |

---

## 7. 修复优先级（更新）

| 优先 | 修复 | 工作量 | 解决问题 | 效果 |
|:----:|------|:------:|---------|------|
| **1** | **FSDP 名称前缀剥离** | ~15 行 | EMA 名称匹配 + LR 分组 | 解除 8×H100 最大风险 |
| **2** | **Cross-stage resume 前移** | ~10 行 | EMA shadow stale | 确保 Stage B/C EMA 正确初始化 |
| 3 | Normalizer 候选优先级翻转 | ~3 行 | checkpoint-first reproducibility | 安全默认值 |
| | **总计** | **~28 行** | 3 项 P1-P2 | 解除所有多卡训练阻塞 |

---

## 8. 中文摘要

### 上一版 review 错误更正

`analysis_v0_10_9_fix_review.md` 将 EMA/FSDP 标记为 "✅ 已修复" 是错误的。核心原因：`use_orig_params=True` 不消除 FSDP 的 `_fsdp_wrapped_module.` 参数名前缀。这导致：

1. **EMA 在多卡下是空操作** — shadow key (`backbone.xxx`) 与 FSDP 参数名 (`_fsdp_wrapped_module.backbone.xxx`) 不匹配 → update/apply/restore 匹配不到任何参数
2. **Per-module LR 在多卡下失效** — `name.startswith("backbone")` 在 FSDP 前缀下永远为 False → 所有参数使用 1.0× LR

### 实际修复状态

- **已解决**: FSDP eval 死锁、gnorm timing、val DistributedSampler、eval EMA wrap（4 项）
- **未解决**: FSDP 名称前缀问题（影响 EMA + LR）、EMA cross-stage stale、normalizer 优先级（3 项）

### 评分

**7.7/10**（非上一版的 8.7）。单卡训练就绪。8×H100 多卡训练需要先修复 FSDP 名称前缀问题（~15 行）。
