# HybridVLA v2 — v0.10.9 Analysis (Post-v0.10.8 Code Review)

> **Date**: 2026-03-28
> **Scope**: (1) v0.10.8 变更验证 (2) GPT 交叉审查验证 (3) 独立发现 (4) 综合评分
> **标准**: "8×H100 FSDP 训练会不会死锁" + "推理质量是否正确" + "工程成熟度"
> **基线**: v0.10.7 summary (8.3/10, 9,870 行, 28 测试, LIBERO 全管线)
> **审查来源**: GPT 自动化审查 (4 项发现) + Claude 独立审查 (6 项发现)

---

## 1. v0.10.8 变更验证

v0.10.8 以推理闭环为核心（`docs/optimize_v0_10_8.md`），逐项验证：

| # | 变更 | 文件 | 验证结果 |
|---|------|------|:--------:|
| 1 | 统一 LIBERO 推理策略 | `vla_hybrid_v2/infer/libero_policy.py` (393 行, 新增) | **✅ PASS** |
| 2 | RTC infer schema 修复 | `config.py:266-269` (`RTCInferConfig.overlap_ratio = 0.333`) | **✅ PASS** |
| 3 | FASTER infer fail-fast | `hybrid_vla_v2.py:690-695` (`NotImplementedError`) + `config.py:274-275` (default=False) | **✅ PASS** |
| 4 | LIBERO rollout 接入统一策略 | `eval_libero_rollout.py:63-73` (使用 `HybridVLALiberoPolicy.from_checkpoint()`) | **✅ PASS** |
| 5 | Checkpoint 资产打包 | `checkpointing.py:73-87` (asset_paths 复制) + `train_unified.py` (checkpoint_assets dict) | **✅ PASS** |
| 6 | 新增 4 个测试文件 | `test_infer_policy.py`, `test_control_step.py`, `test_checkpoint_assets.py`, `test_eval_config_resolution.py` | **✅ PASS** |
| 7 | `infer/__init__.py` 导出策略 | 从 1 行空 stub → 20 行, 导出 6 个公共符号 | **✅ PASS** |

**结论**: v0.10.8 全部 7 项变更验证通过。推理模块从空壳升级为 393 行功能完整的策略包装器，checkpoint 从权重优先升级为推理优先（自包含 config + normalizer stats）。

**v0.10.8 工作质量: 8/10** — 扎实的工程，关键遗漏在 EMA 推理加载（见 §2 GPT-P2）。

---

## 2. GPT 交叉审查验证

逐项验证 GPT 的 4 项发现，给出独立确认证据链。

### GPT-P1: FSDP evaluate() 死锁 — **确认，重新分级为 P0**

**证据链**:

```python
# train_unified.py:547-549 — 仅 rank 0 进入 evaluate
if (val_loader is not None
        and global_step % cfg.train.eval_interval == 0
        and is_main_process()):           # ← rank 0 only
    metrics = evaluate(model, val_loader, device, cfg)
```

```python
# train_unified.py:328 — evaluate 内部调用 forward_train
losses = model.forward_train(batch)       # ← FSDP all-gather 需要所有 rank 参与
```

```python
# distributed.py:106-116 — FSDP 使用 FULL_SHARD 策略
model = FSDP(model, ..., sharding_strategy=ShardingStrategy.FULL_SHARD, ...)
```

**死锁机制**: FSDP FULL_SHARD 在 forward 时执行 all-gather（集合通信），需要所有 rank 同步参与。Rank 0 独自调用 `model.forward_train()` → 进入 all-gather → 等待其他 rank。Ranks 1-7 在训练循环中继续 → 到达下一个 `loss.backward()` 或 `save_checkpoint()` 的 barrier → 等待 rank 0。**双向死锁**。

**影响**: 8×H100 训练在 `eval_interval`（默认 2000 步，约训练 40 分钟后）**必然死锁**。单卡训练不受影响（无 FSDP wrapping）。

**严重度: P0** — 阻塞多卡训练。

---

### GPT-P1b: 外部 config 可静默覆盖 normalizer — **确认，P1**

**证据链**:

```python
# libero_policy.py:96-123 — _candidate_stats_dirs() 优先级
def _candidate_stats_dirs(cfg, checkpoint_path):
    candidates = []
    if cfg.data.normalizer_stats_dir:                   # ← 最高优先级
        candidates.append(Path(cfg.data.normalizer_stats_dir))
    candidates.extend([
        ckpt / "assets" / "normalizer_stats",           # ← 第二优先级（checkpoint 自带）
        ckpt.parent / "normalizer_stats",
        ...
    ])
```

```python
# libero_policy.py:56-93 — resolve_policy_config 校验范围
# 校验: multi_camera.enable (hard error)
# 校验: proprio_dim (warning)
# 未校验: normalizer_stats_dir ← 遗漏
```

**场景**: 用户传入 `--config` 指向另一个实验的 config，其 `normalizer_stats_dir` 指向不同 stats → 推理使用错误的归一化参数 → benchmark 结果不可信。

**缓解**: checkpoint 资产打包（v0.10.8）将 stats 复制到 `checkpoint/assets/`，第二优先级可正确发现。仅当用户显式指定 stats_dir 时才会出问题。

**严重度: P1** — 影响推理正确性，但需用户主动错误操作。

---

### GPT-P2: 推理未加载 EMA 权重 — **确认，重新分级为 P1**

**证据链**:

```python
# libero_policy.py:222-225 — from_checkpoint 仅加载 model.pt
cfg, _resolved = resolve_policy_config(checkpoint_path, config_path)
model = HybridVLAv2(cfg)
load_checkpoint(checkpoint_path, model, strict=False)    # ← 无 ema= 参数
model = model.to(device).eval()
```

```python
# checkpointing.py:157-160 — EMA 加载逻辑存在但未被调用
if ema and (ckpt_dir / "ema.pt").exists():
    ema.load_state_dict(
        torch.load(ckpt_dir / "ema.pt", map_location=map_location, weights_only=True)
    )
```

```python
# checkpointing.py:66-67 — 训练时 EMA 被保存
if ema is not None:
    torch.save(ema.state_dict(), tmp_dir / "ema.pt")
```

**影响**: `model.pt` 包含基础模型权重，不是 EMA 平滑后的权重。EMA 在机器人策略训练中通常提升 5-15% 性能。推理系统性地使用次优权重。

**严重度: P1** — 推理质量系统性降低。

---

### GPT-P3: Per-module 梯度范数在 zero_grad 之后记录 — **确认，P2**

**证据链**:

```python
# train_unified.py:519-542 — 时序关系
grad_norm = clip_grad_norm_fsdp(model, cfg.train.max_grad_norm)  # L520: 梯度存在
optimizer.step()                                                   # L521: 更新参数
scheduler.step()                                                   # L522
optimizer.zero_grad(set_to_none=True)                             # L523: 删除所有梯度 ← HERE
if ema is not None:
    ema.update(model, global_step)                                # L524-525
global_step += 1                                                   # L526

# ... logging block ...
if global_step % (cfg.train.log_interval * 5) == 0:
    _log_per_module_grad_norm(model)                              # L541-542: p.grad 已经是 None
```

```python
# train_unified.py:280-285 — grad norm 函数检查 p.grad is not None
for p in mod.parameters():
    if p.requires_grad and p.grad is not None:  # ← 始终 False（zero_grad 后）
        sq_sum += p.grad.detach().norm(2).item() ** 2
        count += 1
if count > 0:                                   # ← 始终 False
    parts.append(...)
```

**影响**: Per-module 梯度范数日志始终为空。v0.10.5 加入此功能用于验证 Stage B 梯度隔离——但从未真正工作过。全局 `grad_norm`（L520 记录、L538 输出）仍然正确。

**严重度: P2** — 监控失效，不影响训练。

---

## 3. 独立发现

### Claude-1: EMA save/load 在 FSDP 下语义错误 — P0-P1

**证据链**:

```python
# train_unified.py:369-374 — FSDP 包装发生在这里
model = model.to(device)                                          # L369
if cfg.train.fsdp and get_world_size() > 1:
    model = wrap_fsdp(model, mixed_precision=cfg.train.bf16, ...) # L371-372

# train_unified.py:416-424 — EMA 在 FSDP 之后初始化
if cfg.model.ema.enable:
    ema = EMAModel(model, ...)                                    # L420: model 已经是 FSDP 包装后的
```

```python
# ema.py:40-42 — EMA 初始化克隆参数
for name, param in model.named_parameters():     # FSDP 下返回 sharded/flat 参数
    if param.requires_grad:
        self.shadow[name] = param.data.clone()    # 克隆的是 rank-local shard（仅 1/8）
```

```python
# checkpointing.py:52-55 — 仅 rank 0 保存
if not is_main_process():
    if dist.is_initialized():
        dist.barrier()
    return None

# checkpointing.py:67 — 保存 rank 0 的 EMA 状态
torch.save(ema.state_dict(), tmp_dir / "ema.pt")  # 仅含 rank 0 的 shard
```

**问题**: 在 FSDP FULL_SHARD 下，每个 rank 持有参数的 1/N 分片。`ema.state_dict()` 保存的是 rank 0 的局部分片。Resume 时所有 rank 加载同一个分片——其他 rank 的 EMA 状态被覆盖为 rank 0 的分片。

**影响**:
- Stage A（EMA 通常禁用）: 不受影响
- Stage B+（EMA 启用）: checkpoint resume 后 EMA 状态损坏，后续 EMA 更新基于错误基线
- 严重程度取决于 PyTorch 版本和 FSDP 参数名映射方式（`use_orig_params` 设置）

**严重度: P0-P1** — Stage B+ 多卡训练 EMA 功能不可靠。

---

### Claude-2: FSDP 下 per-module 梯度属性解析失败 — P2

**证据链**:

```python
# train_unified.py:268-270 — 通过 getattr 获取子模块
_GRAD_MODULES = [
    "backbone", "grounder", "temporal_core", "action_history_encoder",
    "action_expert", "fast_head", "phase_head", "affordance_head", "cond_builder",
]

# train_unified.py:280 — 在 FSDP-wrapped 模型上调用
mod = getattr(model, mod_name, None)
```

FSDP 包装后，`model` 是 `FullyShardedDataParallel` 实例。`getattr(model, "backbone")` 依赖 FSDP 的 `__getattr__` 转发到 `_fsdp_wrapped_module.backbone`。即使解析成功，子模块的参数是 flat/sharded 形式，不对应原始 per-layer 结构。

叠加 GPT-P3（timing bug），此函数双重失效。

**严重度: P2** — 监控功能在多卡下完全无效。

---

### Claude-3: Val DataLoader 缺少 DistributedSampler — P2

**证据链**:

```python
# train_unified.py:460-463 — 训练 loader 正确使用 DistributedSampler
sampler = (
    torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    if get_world_size() > 1 else None
)

# train_unified.py:471-483 — val loader 无 DistributedSampler
val_loader = DataLoader(
    val_dataset, batch_size=cfg.train.per_device_batch_size,
    shuffle=False, num_workers=1, pin_memory=True,
    drop_last=False, collate_fn=val_collate_fn,
)
```

当前被 GPT-P1 死锁掩盖（evaluate 永远不会成功执行）。修复死锁后，所有 rank 会迭代完整 val 数据集——正确但低效。需要 `DistributedSampler` + `dist.all_reduce()` 实现正确的分布式验证。

**严重度: P2** — 与 GPT-P1 耦合。

---

### Claude-4: 验证未使用 EMA 权重 — P2

**证据链**:

```python
# train_unified.py:546-552 — eval 块
if (val_loader is not None
        and global_step % cfg.train.eval_interval == 0
        and is_main_process()):
    metrics = evaluate(model, val_loader, device, cfg)  # 使用 base model 权重
    # ← 缺少 ema.apply(model) / ema.restore(model)
```

```python
# ema.py:57-68 — apply/restore 专为此场景设计
def apply(self, model):
    """Temporarily replace model weights with EMA weights for eval."""
    ...
def restore(self, model):
    """Restore original weights after eval."""
    ...
```

**影响**: 验证指标始终反映 base model 而非 EMA model。如果 EMA 是部署目标，val loss 曲线具有误导性。

**严重度: P2** — 不影响训练，但验证指标不准确。

---

### Claude-5: Rollout 缺少 action clipping — P2

**证据链**:

```python
# libero_policy.py:384-385 — 去归一化后直接返回
action_model = control.action.detach()
action_env = self.action_normalizer.denormalize(action_model)  # 无 clipping
```

```python
# eval_libero_rollout.py:169-172 — 动作直接送入环境
action = step_out.action_env[0]
actions_batch[k] = action.cpu().numpy()
...
obs_batch, reward_batch, done_batch, info_batch = env.step(actions_batch)
```

LIBERO/robosuite 通常期望动作在 `[-1, 1]` 范围内。如果模型输出偏离训练分布的值，去归一化后可能超出此范围。

**严重度: P2** — 可能导致成功率下降或偶发物理模拟异常。

---

### Claude-6: 推理 hardcoded 448×448 — P3

```python
# libero_policy.py:158
return img.resize((448, 448), Image.BILINEAR).convert("RGB")
```

与训练一致（`hdf5_adapter.py` 和 `libero_hdf5_adapter.py` 均使用 448×448），但不可移植。

**严重度: P3** — 当前正确，未来换 backbone 时脆弱。

---

## 4. 完整问题清单

| 优先级 | ID | 问题 | 位置 | 阻塞训练? |
|:------:|-----|------|------|:---------:|
| **P0** | GPT-P1 | FSDP evaluate() 死锁 | `train_unified.py:547-549` | **是（多卡）** |
| **P0-P1** | Claude-1 | EMA save/load FSDP 语义错误 | `train_unified.py:416-424`, `ema.py:40-42` | Stage B+ 多卡 |
| **P1** | GPT-P2 | 推理未加载 EMA 权重 | `libero_policy.py:224` | 推理质量 |
| **P1** | GPT-P1b | Config normalizer 静默覆盖 | `libero_policy.py:96-123` | 推理正确性 |
| P2 | GPT-P3 | Gnorm 记录在 zero_grad 之后 | `train_unified.py:523,541-542` | 否（监控） |
| P2 | Claude-2 | FSDP gnorm 属性解析失败 | `train_unified.py:275-291` | 否（监控） |
| P2 | Claude-3 | Val DataLoader 无 DistributedSampler | `train_unified.py:471-483` | 否（被 P0 掩盖） |
| P2 | Claude-4 | 验证未用 EMA 权重 | `train_unified.py:546-552` | 否（指标误导） |
| P2 | Claude-5 | Rollout 无 action clipping | `libero_policy.py:385` | 否（推理健壮性） |
| P3 | Claude-6 | Hardcoded 448×448 | `libero_policy.py:158` | 否 |

**统计: 2×P0, 2×P1, 5×P2, 1×P3**

---

## 5. 优先修复指南

### Fix 1 (P0): FSDP evaluate 死锁 — ~15 行

**当前代码** (`train_unified.py:547-552`):
```python
if (val_loader is not None
        and global_step % cfg.train.eval_interval == 0
        and is_main_process()):
    metrics = evaluate(model, val_loader, device, cfg)
```

**修复方案**: 所有 rank 参与 evaluate，rank 0 负责日志输出。

```python
# 修改 1: 移除 is_main_process() 门控
if (val_loader is not None
        and global_step % cfg.train.eval_interval == 0):
    metrics = evaluate(model, val_loader, device, cfg)
    # 修改 2: 所有 rank reduce 指标
    if dist.is_initialized():
        for k in list(metrics.keys()):
            t = torch.tensor(metrics[k], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            metrics[k] = t.item()
    if is_main_process():
        parts = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info("Eval step %d | %s", global_step, parts)
```

同时修改 val_loader 创建（修复 Claude-3）：
```python
val_sampler = (
    torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    if get_world_size() > 1 else None
)
val_loader = DataLoader(
    val_dataset, batch_size=cfg.train.per_device_batch_size,
    sampler=val_sampler, shuffle=False, num_workers=1, ...
)
```

---

### Fix 2 (P0-P1): EMA/FSDP — ~25 行

**方案 A（推荐）**: 在 FSDP wrapping 之前初始化 EMA

```python
# train_unified.py — 调整顺序
model = model.to(device)

# EMA 在 FSDP 之前初始化（使用原始参数名和完整参数）
ema = None
if cfg.model.ema.enable:
    from vla_hybrid_v2.utils.ema import EMAModel
    ema = EMAModel(model, ...)

# FSDP wrapping
if cfg.train.fsdp and get_world_size() > 1:
    model = wrap_fsdp(model, ...)
```

EMA `update()` 需要改为使用 FSDP `summon_full_params` 上下文：
```python
def update(self, model, step):
    decay = self._get_decay(step)
    with FSDP.summon_full_params(model, writeback=False):
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.shadow[name].lerp_(param.data, 1.0 - decay)
```

或者 **方案 B**: 使用 FSDP state_dict_type 上下文保存/加载 EMA（更复杂但不改变初始化顺序）。

---

### Fix 3 (P1): 推理加载 EMA — ~10 行

```python
# libero_policy.py:222-225 — from_checkpoint 修改
cfg, _resolved = resolve_policy_config(checkpoint_path, config_path)
model = HybridVLAv2(cfg)
load_checkpoint(checkpoint_path, model, strict=False)

# 新增: 尝试加载 EMA 权重
ckpt_dir = resolve_checkpoint_dir(checkpoint_path)
ema_path = ckpt_dir / "ema.pt"
if ema_path.exists():
    from vla_hybrid_v2.utils.ema import EMAModel
    ema = EMAModel(model, ...)
    ema.load_state_dict(torch.load(ema_path, map_location="cpu", weights_only=True))
    ema.apply(model)
    logger.info("Applied EMA weights from %s", ema_path)

model = model.to(device).eval()
```

---

### Fix 4 (P1): Normalizer 覆盖保护 — ~5 行

```python
# libero_policy.py:63-82 — resolve_policy_config 中增加校验
if cfg.data.normalizer_stats_dir and resolved_cfg:
    if resolved_cfg.data.normalizer_stats_dir != cfg.data.normalizer_stats_dir:
        logger.warning(
            "Config mismatch: normalizer_stats_dir differs. "
            "--config=%s, resolved=%s. Using --config value.",
            cfg.data.normalizer_stats_dir, resolved_cfg.data.normalizer_stats_dir,
        )
```

---

### Fix 5 (P2): Gnorm 记录时序 — ~3 行

将 `_log_per_module_grad_norm(model)` 移到 `optimizer.zero_grad()` 之前：

```python
# train_unified.py — 修改顺序
grad_norm = clip_grad_norm_fsdp(model, cfg.train.max_grad_norm)
# V5: per-module grad norm (移到 zero_grad 之前)
if is_main_process() and global_step % (cfg.train.log_interval * 5) == 0:
    _log_per_module_grad_norm(model)
optimizer.step()
scheduler.step()
optimizer.zero_grad(set_to_none=True)
```

注意：需要将 `global_step` 计算也前移，或使用 `(global_step + 1)` 条件判断。

---

### Fix 6 (P2): Action clipping — ~2 行

```python
# libero_policy.py:385 — 增加 clipping
action_env = self.action_normalizer.denormalize(action_model)
lo, hi = self.cfg.model.heads.action_range
action_env = action_env.clamp(lo, hi)
```

---

## 6. 评分

| # | 维度 | v0.10.7 | v0.10.9 (修复前) | Δ | 理由 |
|---|------|:-------:|:----------------:|:-:|------|
| 1 | 设计一致性 | 8.5 | **8.5** | 0 | 推理策略包装器遵循现有模式 |
| 2 | 正确性 | 9.5 | **8.0** | **-1.5** | P0 死锁 + P0-P1 EMA/FSDP 腐败；推理归一化修复部分抵消 |
| 3 | 完备性 | 9.0 | **9.0** | 0 | 推理策略填补最后一个重大空白；checkpoint 资产提升自包含性 |
| 4 | 训练稳定性 | 9.0 | **8.0** | **-1.0** | eval_interval 死锁终止多卡训练；EMA resume 腐败 |
| 5 | 可扩展性 | 7.5 | **7.5** | 0 | 无变化 |
| 6 | 性能设计 | 6.5 | **6.5** | 0 | 无变化 |
| 7 | 生产就绪度 | 8.5 | **8.0** | **-0.5** | 推理策略是进步；但 EMA 未加载 + 无 clipping 是回退 |
| 8 | 代码质量 | 8.5 | **8.5** | 0 | 新代码质量好；测试结构合理 |
| 9 | 文档 | 5.5 | **6.0** | +0.5 | `optimize_v0_10_8.md` 详尽；README 更新 |
| 10 | 测试 | 6.0 | **7.0** | **+1.0** | 4 新测试文件覆盖推理策略、control_step、checkpoint 资产、config 解析 |
| | **加权均分** | **8.3** | **7.8** | **-0.5** | P0 发现拖低分数 |

### 评分说明

- **下降原因**: 2 个 P0 问题（FSDP 死锁 + EMA/FSDP）此前未被发现，它们并非 v0.10.8 引入的新问题，而是存在已久的结构性缺陷，此次审查首次暴露
- **v0.10.8 自身质量**: 8/10 — 推理策略包装器设计优良，checkpoint 资产打包解决了真实的可移植性问题
- **修复后预期**: ~8.8-9.0（Fix 1-2 约 40 行即可解决两个 P0）

---

## 7. 训练就绪度

| 场景 | 判定 | 阻塞因素 |
|------|:----:|---------|
| **单卡 Stage A** | **✅ 可以开始** | 无 FSDP，无死锁风险 |
| **8×H100 Stage A** | **❌ 阻塞** | GPT-P1: FSDP evaluate 死锁（每 2000 步） |
| **8×H100 Stage B+** | **❌ 阻塞** | GPT-P1 + Claude-1: 死锁 + EMA/FSDP 腐败 |
| **LIBERO 推理** | **⚠️ 可用但次优** | GPT-P2: 使用 base model 而非 EMA 权重；Claude-5: 无 action clipping |

### 临时绕行方案

如果需要立即启动 8×H100 训练，最快的绕行方案：

```python
# train_unified.py — 临时禁用 eval（2 行改动）
# 将 eval_interval 设为 > max_steps，跳过 evaluate 调用
# 或在 config 中设置 eval_interval: 999999
```

这不解决问题，但避免死锁。EMA 问题需要在 Stage B 之前修复。

---

## 8. 与 GPT 评分对比

| 维度 | GPT 评分 | Claude 评分 | 差异原因 |
|------|:--------:|:----------:|---------|
| 综合 | 7.8 | **7.8** | 一致 |
| 研究方向 | 8.8 | — | Claude 不评此维度 |
| 架构设计 | 8.3 | 8.5 (设计一致性) | 基本一致 |
| 工程实现 | 7.2 | 8.0 (正确性) | GPT 更严格；Claude 注意到 P0 拉低但 v0.10.8 本身质量好 |
| 训练推理闭环 | 6.9 | 8.0 (训练稳定性) | GPT 识别了闭环缺陷；Claude 聚焦具体技术问题 |
| 测试质量 | 7.1 | 7.0 (测试) | 一致 |
| 文档可复现 | 8.5 | 6.0 (文档) | GPT 涵盖了 README/optimize 记录；Claude 基线更严 |

**共识**: 综合 7.8/10 是双方独立评估的交汇点。主要分歧在维度粒度和评分框架。

---

## 9. 中文摘要

### v0.10.8 变更
7 项变更全部验证通过。核心进步：推理模块从空壳升级为 393 行功能完整的 LIBERO 策略包装器（归一化闭环、config 自动发现、checkpoint 资产打包）。

### GPT 交叉审查
4 项发现全部确认。最严重的是 **FSDP evaluate 死锁**（P0）——8×H100 训练会在首次验证时死锁。GPT 的 EMA 推理问题从 P2 提升为 P1。

### 独立发现
6 项新问题。最严重的是 **EMA/FSDP save/load 语义错误**（P0-P1）——Stage B+ 多卡 EMA 在 checkpoint resume 后损坏。

### 完整清单
**2×P0, 2×P1, 5×P2, 1×P3**。两个 P0 均为多卡特有，单卡不受影响。

### 评分
**7.8/10**（较 v0.10.7 下降 0.5 分，源于发现的 P0 问题）。v0.10.8 本身工作质量 8/10。修复后预期恢复至 ~8.8-9.0。

### 修复优先级

| 优先 | 修复 | 工作量 | 解决问题 |
|:----:|------|:------:|---------|
| 1 | FSDP eval 死锁 | ~15 行 | GPT-P1 + Claude-3 |
| 2 | EMA/FSDP | ~25 行 | Claude-1 + Claude-4 |
| 3 | EMA 推理加载 | ~10 行 | GPT-P2 |
| 4 | Normalizer 校验 | ~5 行 | GPT-P1b |
| 5 | Gnorm 时序 | ~3 行 | GPT-P3 |
| 6 | Action clipping | ~2 行 | Claude-5 |
| | **总计** | **~60 行** | **10 项中 9 项** |

**结论**: 约 60 行修复即可解决全部 P0-P2 问题，将评分从 7.8 提升至 ~8.8-9.0。Fix 1（死锁）是 8×H100 训练的硬性前提，应最先完成。
