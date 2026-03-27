# HybridVLA v2 — v0.9.3 Infrastructure Audit Report

从"通用接入数据集/benchmark"视角，结合 GPT 逐文件审计与代码交叉验证，对当前工程的数据层、训练循环、评估循环、推理适配进行全面审计。

---

## Part 1: GPT 分析交叉验证

### GPT 9 项诊断逐条验证

| # | GPT 诊断 | 代码验证 | 准确性 | 补充说明 |
|---|---------|---------|--------|---------|
| 1 | `data/` 空壳 | `data/__init__.py` 仅含 docstring，无任何类/函数 | **100% 准确** | — |
| 2 | 训练脚本内嵌数据逻辑 | `train_stage_a.py:84-105` 定义 DummyVLADataset，`:202-214` 直接构建 DataLoader | **100% 准确** | `num_workers=2, pin_memory=True, drop_last=True`，无自定义 collate_fn |
| 3 | batch 验证不够完整 | `_validate_batch()` 检查 5 必需 key + actions/proprio/prev_actions 维度 | **准确，但有进展** | v0.9.1 已添加基础验证（GPT 审计时尚无），但仍缺 T 一致性、refresh 键、dtype、值域 |
| 4 | refresh batch 未工程化 | `refresh_pixel_values_list` 为 Python list，按 `[r]` 索引 | **100% 准确** | 模型层直接消费 list 结构，无协议抽象 |
| 5 | 无归一化层 | 全 repo 仅 `consistency_loss.py` 有 `F.normalize`（L2 向量归一化，非数据归一化） | **100% 准确** | 无 ActionNormalizer / ProprioNormalizer / stats 存储 |
| 6 | `infer/` 空壳 | `infer/__init__.py` 仅含 docstring | **100% 准确** | 无 PolicyWrapper / EnvAdapter / action postprocess |
| 7 | 无 validation/eval hook | `TrainConfig.eval_interval=2000` 存在但**从未被任何代码读取** | **100% 准确** | 训练循环无 eval 分支，无 validation dataloader |
| 8 | HybridVLAv2 职责过重 | 580→~650 行，含 batch 验证 + refresh 逻辑 + temporal rollout + loss 汇总 + 在线控制 | **准确，程度合理** | 当前阶段可接受，但数据层/runtime 层独立后应自然减薄 |
| 9 | 仓库噪音 | 无 `.gitignore`；`.DS_Store`(10KB)；`.venv/` 存在；7 个 `__pycache__/`；30 个 docs 文件 | **100% 准确** | 无 `requirements.txt`、无 `pyproject.toml`、`tests/` 空目录 |

**结论**：GPT 的 9 项基础设施诊断全部准确，无误判。

---

### GPT 目录重构方案评估

GPT 提出了一套 ~40 个新文件的五层架构（models / data / train / runtime / eval）。

**合理之处**：
- 分层原则正确：模型层不应承载数据/benchmark 语义
- `data/adapters/` + `data/normalizers.py` + `data/schema.py` 的核心三件套是必需的
- `runtime/policy.py` 将推理逻辑从模型类中解耦是正确方向
- `eval/` 与 `train/` 平行独立是工程最佳实践

**过度之处**（当前阶段不应全做）：
- `src/` 目录迁移：破坏所有 import path，收益仅为包管理规范。研究阶段不值得
- 4 个 dataset adapter（libero / robomimic / rlds / custom_hdf5）：应按需实现，不应预先创建空壳
- `data/transforms/{image,language,action,proprio}.py` 拆分过细：初始阶段 1 个 `transforms.py` 足够
- `train/hooks.py`、`train/schedulers.py`：现有逻辑量不支撑独立文件
- `eval/benchmark_base.py` + `eval/libero_eval.py`：在无真实 eval 需求前是过早抽象

**推荐的分阶段方案**：

Phase 1（本次 v0.9.3 范围）：最小可用数据层
```
vla_hybrid_v2/data/
├── __init__.py
├── schema.py          # WindowSample / TrainBatch dataclass
├── normalizer.py      # ActionNormalizer + ProprioNormalizer + stats IO
├── base_adapter.py    # BaseDatasetAdapter abstract class
├── hdf5_adapter.py    # 最小真实数据 loader（HDF5 episodes）
├── collate.py         # vla_collate_fn（处理 refresh 组装）
└── dummy.py           # DummyVLADataset 从 train_stage_a.py 迁入
```

Phase 2（接入首个 benchmark 时）：runtime + eval
```
vla_hybrid_v2/runtime/
├── __init__.py
├── policy.py          # PolicyWrapper（封装 semantic_step + control_step）
└── postprocess.py     # action denormalize + clamp

vla_hybrid_v2/eval/
├── __init__.py
├── metrics.py         # success_rate / MSE / 轨迹统计
└── offline_eval.py    # 离线 eval loop
```

Phase 3（多 benchmark 扩展时）：adapter 注册 + benchmark 适配
```
vla_hybrid_v2/data/adapters/
├── libero.py
├── robomimic.py
└── rlds.py

vla_hybrid_v2/eval/
└── libero_eval.py     # 在线 rollout
```

---

## Part 2: 代码深度审计 — 数据接入相关问题

### D1. DataConfig 字段全部为死代码 — **P0**

```python
# config.py:286-304
@dataclass
class DataConfig:
    format: Optional[str] = None        # ← 无消费
    paths: List[str] = field(...)       # ← 无消费
    data_dir: Optional[str] = None      # ← 无消费
    dataset_name: Optional[str] = None  # ← 无消费
    split: str = "train"                # ← 无消费
    image_key: str = "agentview_rgb"    # ← 无消费
    proprio_key: str = "robot0_joint_pos" # ← 无消费
    action_key: str = "actions"         # ← 无消费
    language_key: str = "language_instruction" # ← 无消费
    camera_keys: List[str] = field(...) # ← 无消费
```

这 10+ 个字段表明设计者已经考虑了通用数据接入，但**零行代码**读取这些字段。

**风险**：协作者看到 `DataConfig.proprio_key="robot0_joint_pos"` 会以为系统已适配 robomimic 数据格式。实际上需要从零实现。

### D2. DummyVLADataset 的形状与真实数据不兼容 — **P0**

```python
# train_stage_a.py:96-105
def __getitem__(self, idx):
    return {
        "input_ids": torch.randint(0, 32000, (self.L,)),  # L=128 固定
        "attention_mask": torch.ones(self.L, dtype=torch.long),  # 全 1
        "actions": torch.randn(self.T, self.H, self.A),   # random normal
        "proprio": torch.randn(self.T, self.P),
        "prev_actions": torch.randn(self.T, self.A),
        "phase_labels": torch.randint(0, 16, (self.T,)),
        "embodiment_id": torch.tensor(0, dtype=torch.long),
    }
```

问题清单：

| 字段 | Dummy 值 | 真实数据预期 | 差异 |
|------|---------|-------------|------|
| `input_ids` | random int [0, 32000] | Qwen2-VL tokenizer 输出 + 视觉占位符 | token 分布完全不同 |
| `attention_mask` | 全 1 | 含 padding 零区域 | 无法测试 mask 路径 |
| `actions` | `randn` ∈ (-inf, inf) | 归一化后 [-1, 1] | 值域不匹配 |
| `proprio` | `randn` ∈ (-inf, inf) | 归一化后 [-1, 1] 或 raw joint pos | 值域不匹配 |
| `pixel_values` | **缺失** | Qwen2-VL processor 输出 | backbone 始终走 text-only |
| `image_grid_thw` | **缺失** | Qwen2-VL processor 输出 | 视觉特征始终为空 |
| `refresh_*` | **缺失** | 多帧 semantic refresh 数据 | 只测试 single-obs 路径 |
| `affordance_labels` | **缺失** | 按需提供 | 永不触发 affordance loss |
| `step_weights` | **缺失** | 按 horizon 的权重 | FM loss 无加权 |

**核心风险**：DummyVLADataset 缺少 `pixel_values`，意味着 backbone 的视觉通路**从未被测试**。所有 smoke test 都在走 text-only 前向。

### D3. Qwen2-VL Processor 初始化但从未调用 — **P1**

```python
# qwen2vl_backbone.py:168-170 (推测)
self.processor = AutoProcessor.from_pretrained(...)
```

Processor 负责：
- 图像 resize + normalize → `pixel_values`
- 文本 tokenize → `input_ids` + `attention_mask`
- 视觉-文本混合 → `image_grid_thw`

但训练脚本和 dummy dataset 直接生成 `input_ids`，跳过了 processor。这意味着：
1. 数据管线必须在模型外部调用 processor（正确做法）
2. 但当前没有任何代码调用它
3. 当接入真实数据时，必须知道 processor 的输出格式才能正确组装 batch

### D4. refresh batch 结构对 collation 不友好 — **P1**

当前模型期望的 refresh 结构：
```python
batch["refresh_input_ids"]         # [B, R, seq_len] — 可 collate
batch["refresh_attention_mask"]    # [B, R, seq_len] — 可 collate
batch["refresh_pixel_values_list"] # List[Optional[Tensor]] len=R — 不可直接 collate
batch["refresh_image_grid_thw_list"] # List[Optional[Tensor]] len=R — 不可直接 collate
```

后两个是 Python list，每个元素的 shape 可能不同（不同 refresh 帧的图像 patch 数不同）。PyTorch 默认 collate 会报错。

**解决方向**：
- 方案 A：统一 pad 到 max patch count，添加 pixel_mask
- 方案 B：collate_fn 中保持 list-of-tensors 结构
- 方案 C：改用 nested tensor（PyTorch 2.1+）

无论哪种，都需要自定义 `collate_fn`。当前没有。

### D5. `_validate_batch` 遗漏的关键检查 — **P1**

已验证：
- 5 个必需 key 存在性 ✓
- actions 维度 [B,T,H,A]，H/A 匹配 config ✓
- proprio 维度 [B,T,P]，P 匹配 config ✓
- prev_actions 维度 [B,T,A]，A 匹配 config ✓

未验证：

| 检查项 | 风险 | 发生条件 |
|--------|------|---------|
| `proprio.shape[1] == actions.shape[1]` (T 一致性) | 静默 shape 错误 | 不同 window 长度 |
| `input_ids.shape == attention_mask.shape` | CUDA crash | tokenizer 输出不对齐 |
| `refresh_input_ids.shape[1] == R` 与 refresh schedule 一致 | 索引越界 | 数据组装错误 |
| `pixel_values` 存在时 `image_grid_thw` 也必须存在 | backbone crash | 部分视觉数据 |
| `actions` 值域在 `action_range` 内 | discretise 溢出 | 未归一化数据 |
| `embodiment_id` 在 `[0, num_embodiments)` 范围 | embedding 索引越界 | 多 embodiment 数据 |

### D6. 无 action/proprio 归一化基础设施 — **P0**

这是通用接入中**最隐蔽的坑**。

不同数据源的 action 语义完全不同：

| Dataset | Action 语义 | 值域 | Proprio 语义 | 值域 |
|---------|-----------|------|-------------|------|
| LIBERO | delta EE pose + gripper | [-0.05, 0.05] + {0,1} | joint pos + gripper | [-π, π] + [0, 1] |
| robomimic | delta EE pose | [-1, 1] (已归一化) | joint pos + vel | 各异 |
| RT-1/RLDS | discrete bins | [0, 255] | 无标准化 | 各异 |
| Bridge v2 | delta EE + gripper | [-0.03, 0.03] | EE pose + gripper | 各异 |

模型内部假设 `action_range = (-1, 1)`。必须有归一化层做桥接。

**最小必需实现**：
```python
class ActionNormalizer:
    def fit(self, actions: np.ndarray) -> None: ...      # 计算 mean/std 或 min/max
    def normalize(self, actions: Tensor) -> Tensor: ...   # raw → [-1, 1]
    def denormalize(self, actions: Tensor) -> Tensor: ... # [-1, 1] → raw
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

### D7. 无 Stage B/C 训练脚本 — **P1**

`scripts/` 仅含：
- `train_stage_a.py` — Stage A
- `train_smoke_test.py` — 冒烟测试

Stage B/C 在 config 中有完整定义（`stage_b.yaml`, `stage_c.yaml`），模型 `forward_train()` 有 `stage != "a"` 分支。但无训练脚本消费这些配置。

**影响**：即使数据管线就绪，也只能训练 Stage A（backbone LoRA + grounder + temporal core）。Expert 训练 (Stage B) 和 full fine-tune (Stage C) 无法进行。

**修复方向**：训练循环应统一为 1 个 `train.py`，通过 `--stage a/b/c` 参数切换。Stage A/B/C 的差异（冻结策略、loss 权重、EMA 启用）已在 config 中定义，脚本只需根据 stage 选择 config。

### D8. `eval_interval` 定义但未使用 — **P1**

```python
# config.py:247
eval_interval: int = 2000

# train_stage_a.py — 训练循环 (lines 220-258)
# 搜索 "eval" — 无结果
```

配置声明了 2000 步评估一次，但训练循环中：
- 无 validation dataloader
- 无 eval 函数调用
- 无 metric 日志
- 无 checkpoint selection（best vs latest）

### D9. configs/data/ 和 configs/infer/ 为空目录 — **P2**

```
configs/
├── model/v2_qwen2vl_7b_trirate_expert18.yaml  ✓
├── train/stage_a.yaml                          ✓
├── train/stage_b.yaml                          ✓
├── train/stage_c.yaml                          ✓
├── data/                                       ← 空
└── infer/                                      ← 空
```

data config 和 infer config 的默认值全部硬编码在 Python dataclass 中。无法通过 YAML 覆盖 data/infer 相关设置。

---

## Part 3: 代码深度审计 — 仓库工程质量

### E1. 无 `.gitignore` — **P0**

仓库根目录无 `.gitignore`。当前存在：
- `.DS_Store` (10KB)
- `.venv/` (完整 Python 虚拟环境)
- 7 个 `__pycache__/` 目录

如果初始化 git 并 push，这些全部会被提交。

**最小 .gitignore**：
```
__pycache__/
*.pyc
.venv/
.DS_Store
*.egg-info/
dist/
build/
outputs/
wandb/
*.pt
*.bin
```

### E2. 无 `requirements.txt` 或 `pyproject.toml` — **P1**

依赖关系完全隐式。从代码 import 推断的关键依赖：
- `torch >= 2.1` (for `weights_only=True` in torch.load)
- `transformers` (Qwen2-VL)
- `peft` (LoRA)
- `mamba_ssm` (可选，official Mamba2)
- `causal_conv1d` (可选)
- `pyyaml`
- `einops`（如果使用）

无版本锁定意味着不同环境可能产生不同行为。

### E3. `tests/` 空目录 — **P1**

```
tests/
└── (empty)
```

唯一的测试是 `scripts/train_smoke_test.py`（手动运行的集成测试）。无 pytest 单元测试。

### E4. 30 个 docs 文件堆积 — **P3**

`docs/` 包含 30 个 markdown 文件，覆盖 v0.1 到 v0.9.2 的分析/修复/设计文档。其中：
- 22 个为 analysis/audit 文档（多数已过时）
- 6 个为 recovery 文档
- 2 个为设计/rescore 文档

**建议**：保留 `docs/design.md`（最新架构说明）+ `docs/changelog.md`（变更日志），其余归档到 `docs/archive/`。

---

## Part 4: 当前 Batch 协议完整定义

基于代码审计，整理出 `forward_train()` 的完整 batch 协议。这是未来数据层的**输出规格书**。

### 必需字段

| Key | Shape | Dtype | 语义 | 来源 |
|-----|-------|-------|------|------|
| `actions` | `[B, T, H, A]` | float32 | 归一化后动作序列 (H=chunk_horizon) | 数据集 |
| `proprio` | `[B, T, P]` | float32 | 归一化后本体感觉 | 数据集 |
| `prev_actions` | `[B, T, A]` | float32 | 上一步动作 | 数据集 (shift) |
| `input_ids` | `[B, L]` | int64 | Qwen2-VL token IDs (含视觉占位符) | Processor |
| `attention_mask` | `[B, L]` | int64/bool | 非 padding 位置为 1 | Processor |

### 条件必需（视觉输入，backbone 非 text-only 时）

| Key | Shape | Dtype | 语义 |
|-----|-------|-------|------|
| `pixel_values` | `[B, N_patches, patch_dim]` | float32/bf16 | Qwen2-VL 视觉特征 |
| `image_grid_thw` | `[B, N_images, 3]` | int64 | 每张图的 T/H/W grid |

### 条件必需（多帧 semantic refresh 时）

| Key | Shape | Dtype | 语义 |
|-----|-------|-------|------|
| `refresh_input_ids` | `[B, R, L]` | int64 | 每个 refresh 帧的 token IDs |
| `refresh_attention_mask` | `[B, R, L]` | int64/bool | 每个 refresh 帧的 mask |
| `refresh_pixel_values_list` | `List[Tensor]` len=R | float32 | 每个 refresh 帧的视觉特征（shape 可变） |
| `refresh_image_grid_thw_list` | `List[Tensor]` len=R | int64 | 每个 refresh 帧的 grid |

### 可选字段

| Key | Shape | Dtype | 语义 | 默认行为 |
|-----|-------|-------|------|---------|
| `semantic_refresh_steps` | `List[int]` | — | 显式 refresh 时间步 | 从 `semantic_refresh_stride` 生成 |
| `embodiment_id` | `[B]` | int64 | 机器人 embodiment ID | `torch.zeros(B, dtype=long)` |
| `phase_labels` | `[B, T]` | int64 | 任务阶段标签 | 不计算 phase loss |
| `affordance_labels` | `[B, T]` | int64 | affordance 类别 | 不计算 affordance loss |
| `step_weights` | `[B, H]` | float32 | 每 horizon 步的 FM loss 权重 | 均匀权重 |

**关键约束**：
- `actions.shape[1] == proprio.shape[1] == prev_actions.shape[1]` (T 必须一致)
- `actions` 值域必须在 `action_range` 内（默认 [-1, 1]）
- `input_ids` 和 `attention_mask` 形状必须一致
- 若提供 `pixel_values`，必须同时提供 `image_grid_thw`

---

## Part 5: 推荐修复清单

### P0 — 阻塞真实数据训练

| # | Issue | 具体修复 | 估计工作量 |
|---|-------|---------|-----------|
| 1 | `.gitignore` 缺失 | 创建标准 Python/ML `.gitignore` | 10 min |
| 2 | 样本协议 `schema.py` | 定义 `WindowSample` / `TrainBatch` dataclass + 文档 | 0.5 day |
| 3 | 归一化层 `normalizer.py` | `ActionNormalizer` + `ProprioNormalizer` (fit/norm/denorm/save/load) | 1 day |
| 4 | 基础 adapter `base_adapter.py` | `BaseDatasetAdapter(Dataset)` 抽象基类 + `DummyAdapter` | 0.5 day |
| 5 | 最小真实 loader `hdf5_adapter.py` | 从 HDF5 episode 读取 + window 切片 + 归一化 | 2 days |
| 6 | Collate 函数 `collate.py` | 处理 refresh 帧组装 + variable-length pixel_values | 1 day |
| 7 | 训练脚本重构 | `train_stage_a.py` 调用 `build_dataset(cfg)` 而非内嵌 Dummy | 0.5 day |
| 8 | `requirements.txt` | 锁定核心依赖版本 | 10 min |

### P1 — 阻塞完整训练循环

| # | Issue | 具体修复 | 估计工作量 |
|---|-------|---------|-----------|
| 9 | 统一训练脚本 `train.py` | 合并 Stage A/B/C 为一个入口，`--stage` 参数切换 | 1 day |
| 10 | `_validate_batch` 补全 | 增加 T 一致性、input/mask 一致性、值域检查 | 0.5 day |
| 11 | Eval 最小框架 | `eval/offline_eval.py` — 在 validation set 上计算 action MSE | 1 day |
| 12 | `runtime/policy.py` | PolicyWrapper 封装 semantic_step + control_step + cache lifecycle | 1 day |

### P2 — 工程质量

| # | Issue | 具体修复 | 估计工作量 |
|---|-------|---------|-----------|
| 13 | 基础 pytest | `test_schema.py` + `test_normalizer.py` + `test_policy.py` | 1 day |
| 14 | data/infer config YAML | `configs/data/default.yaml` + `configs/infer/default.yaml` | 0.5 day |
| 15 | docs 归档 | 保留 design + changelog，其余移入 `docs/archive/` | 0.5 day |

---

## Part 6: 5 个关键接口定义

### Interface 1: WindowSample — 数据层输出的单条样本

```python
@dataclass
class WindowSample:
    """One training window from an episode.

    The data adapter produces this; the collate function
    stacks B samples into a TrainBatch.
    """
    # Required
    actions: Tensor          # [T, H, A] — normalized to action_range
    proprio: Tensor          # [T, P] — normalized
    prev_actions: Tensor     # [T, A] — normalized
    input_ids: Tensor        # [L] — tokenized text (+ image placeholders)
    attention_mask: Tensor   # [L]

    # Vision (None if text-only)
    pixel_values: Optional[Tensor] = None       # [N_patches, patch_dim]
    image_grid_thw: Optional[Tensor] = None     # [N_images, 3]

    # Refresh frames (None if single-observation)
    refresh_input_ids: Optional[Tensor] = None        # [R, L]
    refresh_attention_mask: Optional[Tensor] = None    # [R, L]
    refresh_pixel_values: Optional[List[Tensor]] = None  # R tensors, variable shape
    refresh_image_grid_thw: Optional[List[Tensor]] = None

    # Optional labels
    phase_labels: Optional[Tensor] = None        # [T]
    affordance_labels: Optional[Tensor] = None   # [T]
    embodiment_id: int = 0
    step_weights: Optional[Tensor] = None        # [H]
```

### Interface 2: ActionNormalizer — 动作空间桥接

```python
class ActionNormalizer:
    """Maps raw actions ↔ model canonical range.

    Fits statistics from dataset, then normalizes/denormalizes.
    Statistics are saved/loaded alongside checkpoints.
    """
    def __init__(self, strategy: str = "min_max",
                 target_range: Tuple[float, float] = (-1.0, 1.0)):
        ...

    def fit(self, raw_actions: np.ndarray) -> None:
        """Compute statistics from dataset. Call once before training."""
        ...

    def normalize(self, raw: Tensor) -> Tensor:
        """raw space → model space [-1, 1]"""
        ...

    def denormalize(self, normed: Tensor) -> Tensor:
        """model space [-1, 1] → raw space (for env execution)"""
        ...

    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

### Interface 3: BaseDatasetAdapter — 数据集抽象

```python
class BaseDatasetAdapter(Dataset):
    """Abstract base for all dataset adapters.

    Each adapter reads a specific format (HDF5, RLDS, etc.)
    and outputs WindowSample in the model's canonical format.
    """
    def __init__(self, cfg: DataConfig,
                 action_normalizer: ActionNormalizer,
                 proprio_normalizer: ActionNormalizer,
                 processor: AutoProcessor,  # Qwen2-VL processor
                 split: str = "train"):
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> WindowSample: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    @abstractmethod
    def episode_lengths(self) -> List[int]:
        """For window sampling."""
        ...
```

### Interface 4: PolicyWrapper — 推理封装

```python
class PolicyWrapper:
    """Wraps HybridVLAv2 for deployment.

    Manages RuntimeCache lifecycle, refresh scheduling,
    action denormalization, and observation preprocessing.
    """
    def __init__(self, model: HybridVLAv2,
                 action_normalizer: ActionNormalizer,
                 processor: AutoProcessor,
                 cfg: InferConfig):
        ...

    def reset(self) -> None:
        """Call at episode start. Clears all state."""
        ...

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Single-step policy interface.

        obs: raw observation from environment
        returns: raw action (denormalized) for env.step()
        """
        # 1. Preprocess obs → model input
        # 2. Decide if semantic refresh needed
        # 3. Call semantic_step() if needed, increment refresh_counter
        # 4. Call control_step()
        # 5. Denormalize action
        ...
```

### Interface 5: vla_collate_fn — 批处理组装

```python
def vla_collate_fn(samples: List[WindowSample]) -> Dict[str, Any]:
    """Custom collate that handles variable-length vision tensors.

    Stacks fixed-shape tensors normally.
    For refresh_pixel_values (variable patch count per frame),
    either pads to max length or preserves as list.

    Returns dict matching forward_train() batch protocol.
    """
    ...
```

---

## Part 7: 文件迁移路径

### 当前文件 → 目标位置

| 当前文件 | 操作 | 目标位置 |
|---------|------|---------|
| `vla_hybrid_v2/models/*` | **保留** | 不变 |
| `vla_hybrid_v2/losses/*` | **保留** | 不变 |
| `vla_hybrid_v2/utils/*` | **保留** | 不变 |
| `vla_hybrid_v2/ops/*` | **保留** | 不变 |
| `vla_hybrid_v2/types.py` | **保留** | 不变 |
| `vla_hybrid_v2/config.py` | **保留** | 不变 |
| `vla_hybrid_v2/data/__init__.py` | **扩展** | 添加 6 个文件到 `data/` |
| `vla_hybrid_v2/infer/__init__.py` | **扩展** | Phase 2: 添加 `runtime/` 目录 |
| `vla_hybrid_v2/world_model/*` | **保留** | 不变（不在本次范围） |
| `scripts/train_stage_a.py` | **重构** | DummyVLADataset → `data/dummy.py`；训练循环保留但改用 `build_dataset()` |
| `scripts/train_smoke_test.py` | **保留** | DummyVLADataset 改为 import from `data/dummy.py` |
| `configs/data/` | **填充** | 添加 `default.yaml` |
| `configs/infer/` | **填充** | 添加 `default.yaml` |
| `tests/` | **填充** | 添加基础 pytest 文件 |
| `.DS_Store` | **删除** | — |
| `.venv/` | **保留但 gitignore** | — |
| `__pycache__/` ×7 | **删除 + gitignore** | — |
| `docs/` (30 files) | **归档** | 保留 2-3 个，其余 → `docs/archive/` |

### 不应移动的（GPT 方案修正）

| GPT 建议 | 不采纳原因 |
|----------|-----------|
| `vla_hybrid_v2/` → `src/vla_hybrid_v2/` | 破坏所有 import，收益仅为 pyproject 规范 |
| 创建 `data/transforms/{image,language,action,proprio}.py` | 过度拆分，1 个文件足够 |
| 创建 `train/hooks.py`, `train/schedulers.py` | 当前逻辑量不支撑独立文件 |
| 预创建 `adapters/libero.py` 等空壳 | 应按需实现 |

---

## Part 8: Updated Scoring

从"通用数据接入就绪度"视角，单独给出基础设施评分。

| Dimension | Score | Justification |
|-----------|-------|---------------|
| 模型层成熟度 | **8.5/10** | 架构完整，correctness 高，v0.9.1/2 修复到位 |
| 数据层就绪度 | **1.0/10** | 空壳目录，零实现，零归一化，零真实数据测试 |
| 训练循环完整度 | **4.0/10** | Stage A 可运行（dummy），无 Stage B/C 脚本，无 eval，无 validation |
| 推理/部署就绪度 | **3.0/10** | control_step API 清晰（v0.9.1），但无 PolicyWrapper/denorm/env 适配 |
| 评估就绪度 | **0.5/10** | 无 eval 目录，无 metrics，无 rollout |
| 工程规范度 | **3.0/10** | 无 .gitignore/requirements/pyproject/tests，30 个 docs 堆积 |
| **综合基础设施** | **3.3/10** | |
| **综合（含模型层）** | **5.5/10** | 模型层拉高，基础设施拖低 |

### 预计提升路径

| 阶段 | 内容 | Score 预估 |
|------|------|-----------|
| 当前 | v0.9.2 | 5.5 |
| Phase 1 完成 | data/ 最小实现 + .gitignore + requirements + 训练重构 | **7.0** |
| Phase 2 完成 | runtime/ + eval/ + pytest | **8.0** |
| Phase 3 完成 | 多 benchmark 适配 + Stage B/C | **9.0** |

---

## 中文总结

### 核心判断

GPT 的基础设施诊断 **9/9 全部准确**。当前项目的主要矛盾已从"模型正确性"转移到"数据/训练/评估基础设施缺失"：

1. **数据层 1.0/10**：`data/` 空壳，`DataConfig` 的 10+ 字段全为死代码。无归一化层，无真实数据 loader，无 collate 函数。DummyVLADataset 甚至缺少 `pixel_values`，意味着视觉通路从未被测试。

2. **训练循环 4.0/10**：Stage A 可在 dummy 上跑通，但无 validation split、无 eval hook（`eval_interval=2000` 定义但从未使用）、无 Stage B/C 脚本。

3. **推理/部署 3.0/10**：`control_step()` API 在 v0.9.1 后已清晰（`ControlStepOutput` 返回当前 action），但无 PolicyWrapper、无 action denormalization、无 env 适配层。

4. **工程规范 3.0/10**：无 `.gitignore`（.venv 和 __pycache__ 会被提交）、无 `requirements.txt`、`tests/` 空目录。

### GPT 方案的采纳与修正

- **采纳**：五层分离原则（models/data/train/runtime/eval）、核心三件套（schema + normalizer + adapter）、`collate_fn` 自定义、统一训练入口
- **修正**：不做 `src/` 迁移、不预创建 4 个 benchmark adapter 空壳、不过度拆分 transforms、分 3 个 Phase 实施而非一次性重构

### 最优先的 3 件事

1. **创建 `.gitignore` + `requirements.txt`** — 10 分钟，消除仓库卫生债务
2. **实现 `data/schema.py` + `data/normalizer.py`** — 定义数据协议和归一化桥梁
3. **实现 `data/hdf5_adapter.py` + `data/collate.py`** — 最小可用的真实数据接入

完成这 4 个文件后，项目从"能 forward 一次的模型原型"变成"能接入真实数据训练的实验平台"。
