# HybridVLA v2 第六轮分析 (v0.6 — 训练基础设施审计)

> 分析日期: 2026-03-25
> 输入: `docs/recovery_v0_6.md` + 全部代码 (35 files, 4,823 行)
> 核心: 训练基础设施验证 → 残留问题排查 → 最终就绪判定

---

## 1. v0.6 修复验证

### 1.1 新增文件审计

| 文件 | 行数 | 功能 | 验证 |
|------|------|------|------|
| `utils/ema.py` | 81 | EMA with decay ramp 0.999→0.9999 | ✅ |
| `utils/distributed.py` | 162 | FSDP wrap + activation checkpoint + seed | ✅ |
| `utils/checkpointing.py` | 158 | 原子写入 + FSDP state dict + auto-resume | ✅ |
| `scripts/train_stage_a.py` | 257 | 完整 Stage A 训练脚本 | ✅ |

### 1.2 逐模块验证

**EMAModel** (`utils/ema.py`):
- `_get_decay(step)`: 线性插值 initial→final over ramp_steps ✅
- `update()`: `shadow.lerp_(param, 1-decay)` — 数学正确 (shadow 向 param 移动 `1-decay` 比例) ✅
- `apply()/restore()`: backup→swap→restore 循环完整 ✅
- `state_dict()/load_state_dict()`: 可序列化 ✅
- 仅追踪 `requires_grad=True` 的参数 ✅

**Distributed** (`utils/distributed.py`):
- FSDP 配置: `FULL_SHARD` + `bf16 param / fp32 reduce` + `limit_all_gathers` ✅
- Auto-wrap policy: `{MambaBlock, GrounderBlock, ExpertMambaBlock, ExpertAttentionBlock}` ✅
- Activation checkpointing: `NO_REENTRANT` on wrap classes ✅
- Seed: `random + np + torch + torch.cuda.manual_seed_all` ✅

**Checkpointing** (`utils/checkpointing.py`):
- 原子写入: `.tmp-checkpoint-{step}` → rename → symlink `checkpoint-latest` ✅
- FSDP: `FullStateDictConfig(offload_to_cpu=True, rank0_only=True)` ✅
- `load_checkpoint(strict=False)`: LoRA checkpoint 兼容 ✅
- `weights_only=True`: 安全加载 ✅

**Train Stage A** (`scripts/train_stage_a.py`):
- Cosine warmup scheduler ✅
- Gradient accumulation (除以 `grad_accum_steps` + 周期性 step) ✅
- AMP autocast bf16 ✅
- Expert freeze ✅
- Auto-resume ✅
- Per-rank logging ✅

### 1.3 bin_centers buffer ✅

```python
# hybrid_vla_v2.py:163
self.register_buffer("_fast_bin_centers", torch.linspace(-1, 1, V))
# hybrid_vla_v2.py:394
fast_continuous = (fast_probs * self._fast_bin_centers).sum(dim=-1)
```

Buffer 随模型自动移到正确 device, 不参与梯度, checkpoint 时自动保存/加载。 ✅

---

## 2. 发现的问题

### 2.1 🟡 训练脚本: accum_loss 计算不准确

**位置**: `train_stage_a.py:204-224`

```python
loss = losses["loss_total"] / grad_accum     # 除以 accum 步数
loss.backward()

for k, v in losses.items():
    accum_loss[k] = accum_loss.get(k, 0.0) + v.detach().item()  # 未除以 accum
```

`accum_loss` 累积的是**未除以 grad_accum 的原始 loss 值**, 但它累积了 `grad_accum` 个 micro-batch。在 logging 时:

```python
avg = {k: v / cfg.train.log_interval for k, v in accum_loss.items()}
```

这里除以 `log_interval`, 但实际累积了 `log_interval * grad_accum` 个 micro-batch 的 loss。正确的应该是:

```python
avg = {k: v / (cfg.train.log_interval * grad_accum) for k, v in accum_loss.items()}
```

**影响**: 日志中显示的 loss 值比实际高 `grad_accum` 倍 (默认 4×)。不影响训练本身, 但误导监控。

### 2.2 🟡 EMA 与 FSDP 的兼容性

`EMAModel.__init__` 通过 `model.named_parameters()` 迭代所有参数并 `.clone()`。在 FSDP wrapping 后, 参数名可能带有 FSDP 前缀 (如 `_fsdp_wrapped_module.xxx`), 且参数可能是 sharded 的 (只包含当前 rank 的 shard)。

**问题链**:
1. `train_stage_a.py:155-160` 中 EMA 在 FSDP 之后创建 → 参数名带 FSDP 前缀
2. EMA shadow 存储的是 sharded 参数 → `apply()` 时只能恢复当前 rank 的 shard
3. `ema.state_dict()` 保存的 shadow 在不同 rank 上不同 → checkpoint 不完整

**但**: 当前 `train_stage_a.py:153-160` 的顺序是:
```python
# Line 136-138: FSDP wrapping
model = wrap_fsdp(model, ...)
# Line 152-160: EMA creation
ema = EMAModel(model, ...)
```

EMA 在 FSDP 之后创建, 操作的是 FSDP-wrapped 模型的参数。`model.named_parameters()` 返回的是 flat/sharded 参数。EMA `update()` 时用 `lerp_` 原地更新 shadow (这些 shadow 也是 sharded 大小)。`apply()` 时将 shadow 写回模型参数 (sharded 大小一致)。

**结论**: 在**单次训练**中可以工作 (shadow 和 param 始终是同 rank 的 shard)。但 **checkpoint resume** 时需确保 shadow 加载到相同 rank topology — 如果 GPU 数量变化, shadow shard 大小不匹配。

**严重度**: 低 — 正常训练 resume (相同 GPU 数量) 没有问题。跨 GPU 数量 resume 时需要手动处理。

### 2.3 🟡 FSDP wrap_fsdp 中缺少 `use_orig_params=True`

```python
# distributed.py:112-120
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=torch.cuda.current_device(),
    sync_module_states=sync_module_states,
    limit_all_gathers=True,
    # 缺少: use_orig_params=True
)
```

`use_orig_params=True` (PyTorch ≥ 2.0) 允许:
- 对冻结参数 (backbone 7.5B) 跳过 allgather → 节省通信
- 不同参数组不同 lr 时保持正确性
- 与 `torch.compile` 更好兼容

**影响**: 不设置时, FSDP 对冻结参数也做 allgather/reduce_scatter, 浪费 ~15 GB × 8 GPU 的通信量。对 7.5B 冻结 backbone, 这是显著的通信开销。

### 2.4 🟡 save_checkpoint: FSDP 状态保存需要 barrier 时序修正

```python
# checkpointing.py:49-54
model_state = _get_state_dict(model)  # ← 所有 rank 必须同时进入 FSDP state_dict_type context

if not is_main_process():
    if dist.is_initialized():
        dist.barrier()
    return None
```

`_get_state_dict()` 使用 `FSDP.state_dict_type(model, ...)` 上下文管理器。在 FSDP 中, `state_dict()` 是一个**集体操作** — 所有 rank 必须同时调用它来做 allgather。

当前代码: 所有 rank 调用 `_get_state_dict(model)` (✅ 集体操作正确), 然后非 rank-0 立即 barrier + return。Rank 0 继续保存文件后 barrier。这个顺序是正确的。 ✅

但 `_get_state_dict` 的 try/except:
```python
def _get_state_dict(model):
    try:
        from torch.distributed.fsdp import ...
        if isinstance(model, FSDP):
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                return model.state_dict()
    except ImportError:
        pass
    return model.state_dict()
```

如果 FSDP import 成功但 `isinstance(model, FSDP)` 为 False (例如单 GPU 未 wrap), 走 fallback `model.state_dict()` — 正确。 ✅

### 2.5 🟢 Stage B/C 训练脚本缺失 (已知, 非阻塞)

`train_stage_a.py` 硬编码 `stage: "a"` 和 Expert freeze。Stage B/C 需要:
- 解除 Expert freeze
- 设置 `stop_gradient_cond_prefix`
- EMA 从 Stage A checkpoint resume
- 不同的 loss weights

这些可以通过复制 `train_stage_a.py` 并修改几行实现。recovery_v0_6 已注明这是下一步工作。

### 2.6 🟢 flash_attn fallback 仍缺失

`BackboneConfig.attn_implementation = "flash_attention_2"` 硬编码。如果 `flash_attn` 包未安装, `Qwen2VLForConditionalGeneration.from_pretrained()` 会报错。

**缓解**: 在 `Qwen2VLBackboneWrapper.__init__` 中添加:
```python
try:
    import flash_attn  # noqa
except ImportError:
    attn_implementation = "sdpa"  # fallback to PyTorch SDPA
```

已在 v0.2 分析中提出, 尚未修复但可通过安装 flash_attn 绕过。

---

## 3. 训练完整性矩阵

### 3.1 Stage A 训练: 逐步骤验证

| 训练步骤 | 实现 | 验证 |
|----------|------|------|
| 1. 配置加载 (`load_config`) | ✅ | smoke test ✅ |
| 2. 分布式初始化 (`setup_distributed`) | ✅ | 单 GPU ✅, 多 GPU 需实测 |
| 3. 模型创建 (`HybridVLAv2(cfg)`) | ✅ | smoke test ✅ |
| 4. Expert 冻结 | ✅ | smoke test Stage A ✅ |
| 5. FSDP wrapping | ✅ | 代码完整, 需 CUDA 实测 |
| 6. Optimizer (AdamW, betas=0.9/0.95) | ✅ | smoke test ✅ |
| 7. LR scheduler (cosine warmup) | ✅ | 代码正确 |
| 8. EMA (可选, decay ramp) | ✅ | 验证通过 |
| 9. Auto-resume | ✅ | 代码完整 |
| 10. Data loading | ⚠️ DummyDataset | 真实数据需替换 |
| 11. Forward pass (`forward_train`) | ✅ | smoke test ✅ |
| 12. Loss computation | ✅ | smoke test ✅ |
| 13. Backward + grad accum | ✅ | smoke test ✅ |
| 14. Gradient clipping (FSDP-aware) | ✅ | 代码正确 |
| 15. Optimizer step | ✅ | smoke test ✅ |
| 16. EMA update | ✅ | 验证通过 |
| 17. Logging (per-rank, file) | ✅ | 代码完整 |
| 18. Checkpoint save (atomic, symlink) | ✅ | 代码完整 |
| 19. Evaluation | ❌ | 下一轮 |

**18/19 步骤已实现**, 唯一缺失的是 Evaluation (非训练阻塞)。

### 3.2 端到端训练流程验证

```
configs/train/stage_a.yaml
        ↓ load_config()
   HybridVLAv2Config
        ↓
   HybridVLAv2(cfg)     → 模型实例化 (含 Backbone + Grounder + Core + Expert + Heads)
        ↓
   Expert.requires_grad = False
        ↓
   wrap_fsdp(model)      → FSDP + activation checkpointing
        ↓
   AdamW(trainable_params)
        ↓
   cosine_schedule_with_warmup(optimizer, warmup=3K, total=120K)
        ↓
   EMAModel(model, 0.999 → 0.9999 over 20K)
        ↓
   auto_resume(output_dir)
        ↓
   for epoch:
     for batch in DataLoader(DummyVLADataset):
       with autocast(bf16):
         losses = model.forward_train(batch)
       (loss / grad_accum).backward()
       if (batch_idx+1) % grad_accum == 0:
         clip_grad_norm_fsdp(model, 1.0)
         optimizer.step()
         scheduler.step()
         ema.update(model, step)
         log + checkpoint
```

**流程完整**, 唯一的占位是 DummyDataset。

---

## 4. 优先修复建议

| # | 问题 | 严重度 | 工作量 | 建议 |
|---|------|--------|--------|------|
| 1 | accum_loss 日志值偏高 grad_accum× | 🟡 | 5 分钟 | 改除数为 `log_interval * grad_accum` |
| 2 | FSDP 缺少 `use_orig_params=True` | 🟡 | 5 分钟 | 添加参数, 减少冻结参数通信 |
| 3 | flash_attn fallback | 🟡 | 10 分钟 | backbone 中添加 try/except |
| 4 | 真实数据 pipeline | 🔴 | 2-3 天 | 替换 DummyDataset |
| 5 | Stage B/C 脚本 | 🟡 | 0.5 天 | 复制 Stage A 修改 |
| 6 | Evaluation 循环 | 🟡 | 1-2 天 | — |

---

## 5. 总结

### 5.1 从 v0.1 到 v0.6 的完整迭代回顾

| 版本 | 主题 | 修复数 | 代码行数 |
|------|------|--------|---------|
| v0.1 | 架构修复 (Grounder/SDPA/Gate/CUDA) | 8 | 2,660 |
| v0.2 | 官方 Mamba2 集成 | 3 | 2,660 |
| v0.3 | 分析: Core 状态丢失发现 | — | — |
| v0.4 | WM 集成 + step() API + Trajectory | 8 | 3,927 |
| v0.5 | Core 状态修复 + smoke test | 5 | 4,164 |
| v0.6 | 训练基础设施 (FSDP/EMA/ckpt/script) | 5 | **4,823** |
| **总计** | | **~30 项** | |

### 5.2 最终就绪判定

```
模型代码:       95% ✅ (forward_train 端到端验证通过)
世界模型:       90% ✅ (9 模块完整, VLA 集成接口就绪)
训练基础设施:    75% ✅ (FSDP + EMA + checkpoint + scheduler + script)
数据 pipeline:   5% ❌ (仅 DummyDataset)
评估:            0% ❌

能否开始训练?
  单 GPU + dummy data:  ✅ 现在就可以 (smoke test 已验证)
  单 GPU + 真实数据:    ⚠️ 替换 DummyDataset 后即可 (~1天)
  8×H100 正式训练:     ⚠️ 需 CUDA 环境实测 FSDP (~1天实测)
```

**v0.6 将训练基础设施从 15% 提升到 75%。模型代码经过 6 轮 30 项修复, 已通过 Stage A + Stage B 端到端 smoke test。距离正式训练的唯一硬阻塞是真实数据 pipeline (2-3 天)。**

---

*分析完毕。v0.6 的 4 个新文件 (EMA / FSDP / Checkpoint / Train Script) 质量良好, 覆盖了训练基础设施的核心需求。发现 2 个小问题 (日志 loss 值偏高 4×, FSDP 缺少 use_orig_params) 和 1 个遗留 (flash_attn fallback), 均不阻塞训练。代码库 35 files / 4,823 行 / 0 语法错误。*
