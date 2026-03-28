# HybridVLA_2 工程总评估

更新时间：2026-03-27  
审计方式：阅读仓库主干代码、配置与脚本，并进行了最小运行核验（Stage A/B/C smoke test、multi-camera smoke test、在线 `semantic_step/control_step` 路径测试）。

## 一句话结论

这是一个**架构主干已经成型、离线训练骨架已经存在、但真实数据实验闭环和评测闭环尚未完成**的研究型 VLA 工程。  
如果按“是否具备完整的核心模型实现”来衡量，我会给它 **70-75%**；如果按“是否能直接、稳定地进入真实机器人数据实验”来衡量，大约 **55-60%**；如果按“完整项目闭环（真实数据、benchmark、在线推理系统、世界模型、测试）”来衡量，大约 **35-40%**。

## 核心判断

这不是空架子仓库。`vla_hybrid_v2/models/`、`scripts/train_unified.py`、`vla_hybrid_v2/data/hdf5_adapter.py` 这些文件说明主模型、训练主循环、HDF5 数据适配器都已经写到可运行状态，而不是只有设计文档。

但它也还不是“开箱即训”的完整工程。最关键的几个证据是：

1. 默认 stage 配置并没有接入真实数据配置。`configs/train/stage_a.yaml`、`stage_b.yaml`、`stage_c.yaml` 只在 `defaults` 中引入了模型配置，没有引入 `configs/data/...`；而 `build_dataset()` 在 `cfg.data.format is None` 时会回落到 `DummyVLADataset`。对应代码见：
   - `configs/train/stage_a.yaml:4-45`
   - `configs/train/stage_b.yaml:3-38`
   - `configs/train/stage_c.yaml:3-49`
   - `vla_hybrid_v2/config.py:367-380`
   - `vla_hybrid_v2/data/__init__.py:29-82`

2. Stage C 配置中声明了 RTC/FASTER，但当前训练代码里没有真正使用这两套机制。`stage_c.yaml` 中 `rtc.enable=true`、`faster.enable=true`，但全仓库搜索后，除了配置定义和 `infer.execution_horizon` 之外，没有训练逻辑消费这些字段。也就是说，**当前 Stage C 更像是“低学习率继续联合训练”，而不是 README 叙述中的 RTC/FASTER fully implemented**。对应证据：
   - `configs/train/stage_c.yaml:29-39`
   - `vla_hybrid_v2/config.py:248-285`
   - `vla_hybrid_v2/models/hybrid_vla_v2.py:613`

3. 世界模型支线已经有模块，但还没有真正接到训练闭环里。主模型只是在初始化时可选创建 `ImaginationEngine` 和 `WorldModelLoss`，并提供 `get_world_model_state()`；`forward_train()` 和 `scripts/train_unified.py` 并没有把 world model rollout 或 loss 接进去。对应代码：
   - `vla_hybrid_v2/models/hybrid_vla_v2.py:181-226`
   - `vla_hybrid_v2/models/hybrid_vla_v2.py:345-580`
   - `vla_hybrid_v2/world_model/imagination_engine.py:58-236`
   - `vla_hybrid_v2/world_model/world_model_loss.py:88-184`

4. 在线推理只有模型内方法，没有完整 runtime/system 层。`HybridVLAv2` 内部已有 `semantic_step()`、`control_step()`、`init_runtime()`，但 `vla_hybrid_v2/infer/` 基本还是空壳。对应代码：
   - `vla_hybrid_v2/models/hybrid_vla_v2.py:587-711`
   - `vla_hybrid_v2/infer/__init__.py:1`

## 一、架构分析与架构合理性

### 1. Backbone + Multi-scale Adapter

骨干采用 `Qwen2-VL-7B-Instruct`，并从多层 hidden states 抽特征后通过 `MultiScaleAdapter` 投影到统一 2048 维，再进入 grounder。这是一个**合理的工程折中**：

- 它没有直接把 7B backbone 的最后一层特征硬送给动作模块，而是显式保留 early/mid/late 三个尺度，适合操作场景中的“几何细节 + 目标语义”混合需求。
- LoRA 打在全层，而不是只打最后几层，说明作者想让视觉-语言 backbone 更深地向 manipulation domain 适配。

对应实现：
- `vla_hybrid_v2/models/qwen2vl_backbone.py:24-53`
- `vla_hybrid_v2/models/qwen2vl_backbone.py:126-236`

合理性评价：

- **优点**：设计逻辑清晰，和机器人 manipulation 的 feature demand 是一致的。
- **风险**：这是一个非常重的 backbone 依赖，且真实集成路径并没有通过真实 Qwen2-VL 权重的 smoke test 验证。当前 smoke test 使用的是 mock backbone，而不是实际 HF 模型。对应证据：
  - `scripts/train_smoke_test.py:6-11`
  - `scripts/train_smoke_test.py:97-113`

结论：**架构设计合理，代码实现也真实存在，但实际大模型集成可靠性仍需真实环境验证。**

### 2. Hierarchical Attention Grounder

Grounder 是本项目架构里非常合理的一层“结构化桥接层”。它把 variable-length backbone token 压成固定结构输出：

- `global_token`
- `compressed_object_slots`
- `phase_token`
- `uncertainty_token`
- `affordance_token`

并在中间层做 `48 -> 24` 的 slot compression。对应代码：
- `vla_hybrid_v2/models/attention_grounder.py:128-150`
- `vla_hybrid_v2/models/attention_grounder.py:157-260`

合理性评价：

- **这是一个正确的系统分层**。VLA 如果直接把大 backbone token 喂给控制模块，时序模块负担会过大；Grounder 的存在让“感知结构化”成为独立阶段，工程上是对的。
- **当前问题不在设计，而在监督完整性**。`phase_token` 与 `affordance_token` 只有在 batch 里存在标签时才有显式监督；`uncertainty_token` 当前没有专门 loss。对应代码：
  - `vla_hybrid_v2/models/hybrid_vla_v2.py:507-529`
  - `vla_hybrid_v2/models/hybrid_vla_v2.py:232-260`

结论：**Grounder 是该项目最成熟、最合理的结构之一，但其中一部分“结构化 token 语义”目前仍偏弱监督。**

### 3. Tri-Rate Mamba Core

这是整个项目最有辨识度的架构贡献。`TriRateMambaCore` 明确分成 fast/medium/slow 三条流，并支持跨 step 的显式 state 持久化。对应代码：

- `vla_hybrid_v2/models/mamba_core.py:601-760`
- `vla_hybrid_v2/models/mamba_core.py:328-454`

合理性评价：

- **从机器人控制视角，这个架构是成立的。** 高频控制、对象级动态、中低频任务语义，本来就不应当被一个单频时序模块粗暴建模。
- 代码层面也不是伪实现，Mamba state 的 carry-over、medium/slow 的稀疏更新、stale-time encoding 都落地了。
- 这部分实现质量明显高于很多“只在 README 里讲 multi-rate”的项目。

但这里也有两个工程风险：

1. `fast_token / medium_token / slow_token` 当前是对输出序列做 `mean(dim=1)` 得到的。这种 readout 很简单，但也意味着所有结构 token、object slot、proprio token 被平均到一起，信息瓶颈比较粗。对应代码：
   - `vla_hybrid_v2/models/mamba_core.py:712-758`

2. 三频设计显著提高了系统复杂度，但目前仓库里还没有配套的 ablation / benchmark 框架，因此“为什么 tri-rate 明显优于 dual-rate/single-rate”在代码层无法被验证，只能从设计逻辑上认为合理。

结论：**Tri-rate core 是本仓库最强的研究性亮点之一，逻辑与实现都成立，但经验验证层明显滞后于设计复杂度。**

### 4. Flow Action Expert

`FlowActionExpert` 不是占位代码，是真正可训练的 18 层混合 Mamba/Attention denoiser，并实现了 AdaRMSNorm、Euler/Midpoint sampling。对应代码：

- `vla_hybrid_v2/models/flow_action_expert.py:31-53`
- `vla_hybrid_v2/models/flow_action_expert.py:94-161`
- `vla_hybrid_v2/models/flow_action_expert.py:168-343`

合理性评价：

- AdaRMSNorm + flow timestep conditioning 是合理的，尤其对于 action denoising 这类在不同噪声尺度下动态范围差异很大的任务。
- Stage B 的 `cond_prefix.detach()` 策略也合理，避免 flow matching 梯度直接扰乱 backbone/grounder。

对应代码：
- `vla_hybrid_v2/models/hybrid_vla_v2.py:531-572`

结论：**动作专家是完整实现，不是“计划中”。这一块已经能进训练。**

### 5. World Model

world model 分支的模块量很大，但当前完成度不能高估。

优点：

- `ImaginationEngine`、`StochasticStateModule`、`ObjectPhysicsEngine`、`WorldModelLoss` 都存在，且写得不像草稿。
- 从对象中心物理 + DreamerV3 离散 latent 的组合上看，设计方向是清晰的。

问题：

- 训练主循环没有接入。
- `visual_decoder.py` 文件头自己就写了 “L2: Placeholder for frozen pretrained diffusion decoder”。对应代码：
  - `vla_hybrid_v2/world_model/visual_decoder.py:1-6`

结论：**world model 是一条“实现了很多模块、但尚未形成训练产品”的支线，当前更适合定义为 scaffold / branch，而不是主线功能。**

## 二、数据管线分析

### 已实现部分

数据层并非空白，且已经具备最基本真实数据训练能力：

- `Normalizer` / `compute_stats.py` 负责动作与 proprio 统计与归一化；
- `HDF5DatasetAdapter` 支持窗口切片、`T + H - 1` future action chunk、图像读取、Qwen2-VL processor tokenization；
- `vla_collate_fn` 处理 refresh 帧与 vision tensor。

对应代码：
- `vla_hybrid_v2/data/normalizer.py:1-178`
- `scripts/compute_stats.py:1-186`
- `vla_hybrid_v2/data/hdf5_adapter.py:315-457`
- `vla_hybrid_v2/data/collate.py:1-113`

这说明项目已经跨过“只有模型，没有数据入口”的阶段。

### 当前主要问题

#### 1. 默认配置不接真实数据

这是当前完成度里最影响实际使用的一点。

实测加载结果：

```text
configs/train/stage_a.yaml data.format= None data_dir= None multi_camera= False
configs/train/stage_b.yaml data.format= None data_dir= None multi_camera= False
configs/train/stage_c.yaml data.format= None data_dir= None multi_camera= False
```

也就是说，按 README 里最直观的 `python -m scripts.train_unified --config configs/train/stage_a.yaml` 方式启动时，默认不会走真实 HDF5，而会落回 dummy 数据路径。

对应代码：
- `vla_hybrid_v2/data/__init__.py:42-76`
- `vla_hybrid_v2/config.py:367-380`
- `configs/train/stage_a.yaml:4-45`
- `configs/data/libero_multicam.yaml:4-20`

这意味着：

- 工程已经支持真实数据；
- 但默认工作流还没有把“真实数据配置”纳入一级入口。

#### 2. 真实 HDF5 adapter 还没有把多任务监督做全

`HDF5DatasetAdapter.__getitem__()` 返回的 sample 字段包括：

- `input_ids`
- `attention_mask`
- `actions`
- `proprio`
- `prev_actions`
- `embodiment_id`
- 可选的视觉字段和 refresh 字段

但它**没有**产出：

- `phase_labels`
- `affordance_labels`
- `step_weights`
- `semantic_refresh_steps`

对应代码：
- `vla_hybrid_v2/data/hdf5_adapter.py:392-455`

而训练前向里，这些监督只有在 batch 里存在时才启用：
- `vla_hybrid_v2/models/hybrid_vla_v2.py:507-529`
- `vla_hybrid_v2/models/hybrid_vla_v2.py:555-559`

结果就是：

- 在 dummy 数据下，phase/affordance 可以被随机标签驱动；
- 在真实 HDF5 路径下，这些头默认可能根本不训练。

这会直接削弱 README 中“结构化 token + 多头监督”的真实落地程度。

#### 3. 数据格式支持仍然单一

目前 `build_dataset()` 只显式支持 `hdf5` 和 `dummy`。没有 RLDS、robomimic、DROID、Calvin registry，也没有 manifest/mixture 机制。对应代码：
- `vla_hybrid_v2/data/__init__.py:29-82`

这不影响当前作为单数据源研究工程推进，但它意味着：

- 项目还没有进入“大规模异构机器人数据工程化”阶段；
- README 中提到的多数据集训练目标，目前还主要停留在未来计划层。

### 数据管线完成度结论

我会给数据管线 **60-65%**。

理由是：

- 已经有真实 HDF5 读取、归一化、窗口切片、视觉 tokenization；
- 但默认配置没接上真实数据，监督字段还不完整，格式支持和 dataset orchestration 也偏薄。

## 三、训练系统分析

### 已经成熟的部分

训练主循环是当前仓库里第二成熟的部分。

`scripts/train_unified.py` 已经具备：

- A/B/C 统一入口；
- 显式 stage gate；
- FSDP 包装；
- per-module optimizer group；
- checkpoint / auto-resume；
- validation loss 评估；
- EMA；
- distributed seed / rank / logging。

对应代码：
- `scripts/train_unified.py:87-231`
- `scripts/train_unified.py:314-536`

这说明项目不只是“模型能 forward”，而是**已经有研究训练工程的主干**。

### Stage A / B 的完成度

Stage A/B 在当前代码里是实打实存在的：

- Stage A：冻结 expert，只训练 backbone LoRA + grounder + temporal core + heads；
- Stage B：引入 expert，并通过 `cond_prefix.detach()` 做梯度隔离。

这些逻辑都在代码中直接成立：
- `scripts/train_unified.py:87-157`
- `vla_hybrid_v2/models/hybrid_vla_v2.py:531-572`

并且我实际运行了最小 smoke test，结果通过：

```text
python -m scripts.train_smoke_test --steps 2 --stage a
-> Smoke test PASSED — no NaN, no crash.

python -m scripts.train_smoke_test --steps 2 --stage b
-> Stage B assertions PASSED: loss_fm present, expert params updated.
```

### Stage C 的真实状态

Stage C 是当前项目“文档完成度高于代码完成度”的最明显部分。

`configs/train/stage_c.yaml` 中宣称：

- RTC enable
- FASTER enable

但代码层面：

- `train_unified.py` 没有任何 RTC/FASTER 调度逻辑；
- `HybridVLAv2.forward_train()` 也没有相关分支；
- `rg` 全仓库搜索，除了 config dataclass 和 `infer.execution_horizon` 外，没有训练路径消费这些字段。

因此当前 Stage C 的真实语义更接近：

- 在 Stage B 基础上继续联合训练；
- 学习率更低；
- 配置里保留 RTC/FASTER 参数，但尚未真正落地。

这会影响“完成程度”判断，因为 README 对 Stage C 的表述明显更前。

### 评估系统现状

`evaluate()` 当前只是离线平均 loss 的聚合，没有任务成功率、trajectory rollout、benchmark harness。对应代码：
- `scripts/train_unified.py:268-307`

README 自己也承认 evaluation framework 还是开发优先项：
- `README.md:284-292`

这意味着：

- 训练 loop 是成熟的；
- 但 experiment loop 还不成熟。

### 训练系统完成度结论

我会给训练系统 **75-80%**。

理由是：

- 统一训练入口、分阶段训练、断点续训、分布式、eval-loss 都已经具备；
- 但 Stage C 的高级机制未真正实现，benchmark/evaluation 仍然缺席。

## 四、推理与运行时分析

### 已有能力

模型内部已经提供：

- `semantic_step()`
- `control_step()`
- `init_runtime()`

并实现了：

- tri-rate recurrent state 持久化；
- action chunk caching；
- semantic refresh 触发 chunk 失效重采样。

对应代码：
- `vla_hybrid_v2/models/hybrid_vla_v2.py:587-711`

我也做了最小 runtime 核验，结果如下：

```text
ok torch.Size([2, 7]) torch.Size([2, 4, 7]) True False 2
```

这说明：

- 第一次 `control_step()` 能生成 chunk；
- 第二次在未 refresh 情况下能复用 chunk；
- runtime cache 逻辑是通的。

### 缺失部分

但项目还没有完整的 runtime / deployment layer：

- `vla_hybrid_v2/infer/` 只有一个空的 `__init__.py`；
- 没有 robot env wrapper；
- 没有 sensor bridge；
- 没有 latency benchmark；
- 没有 policy server / rollout runner。

所以目前的在线推理能力更准确地说是：

- **模型内部 API 已有**
- **系统级推理工程还没展开**

### 推理完成度结论

我会给在线推理 **45-50%**。

## 五、测试与可验证性

### 当前已有验证

我实际跑过以下核验：

```text
python -m scripts.train_smoke_test --steps 2 --stage a
python -m scripts.train_smoke_test --steps 2 --stage b
python -m scripts.train_smoke_test --steps 1 --stage c
python -m scripts.train_smoke_test --steps 2 --multi-camera
```

全部通过，无 NaN、无 crash。

### 但测试体系仍然很薄

1. `tests/` 目录目前没有实际测试文件。
2. smoke test 使用的是 mock backbone，而不是实际 `Qwen2-VL + PEFT + processor + mamba_ssm` 全路径。对应代码：
   - `scripts/train_smoke_test.py:6-11`
   - `scripts/train_smoke_test.py:97-113`
3. `requirements.txt` 里 `mamba_ssm` / `causal_conv1d` 仍被标成 optional，但从目标硬件和模型规模看，它们在真实训练中几乎不是“可选优化”，而是“现实前提”。对应文件：
   - `requirements.txt:9-15`

所以当前测试更多证明：

- 模型胶水逻辑是通的；

还不能证明：

- 实际大模型依赖栈稳定；
- 真正 8xH100 路径稳定；
- 真实数据训练能复现实验结果。

## 六、项目完成度分项评分

| 维度 | 评分 | 判断 |
|---|---:|---|
| 核心架构实现 | 80/100 | Backbone/Grounder/Tri-rate/Expert 主干都已落地 |
| 架构合理性 | 78/100 | 思路清晰，分层正确，但复杂度显著超前于验证闭环 |
| 数据管线 | 63/100 | 已有 HDF5 真路径，但默认配置未接入、监督字段不全 |
| 训练基础设施 | 78/100 | unified training、FSDP、EMA、resume 都已具备 |
| Stage C 完整度 | 45/100 | 配置前瞻，RTC/FASTER 代码未真正落地 |
| 在线推理系统 | 48/100 | 模型 API 存在，但 infer/runtime 层基本未展开 |
| 世界模型支线 | 28/100 | 模块很多，但未形成训练闭环 |
| 评测/benchmark | 25/100 | 只有离线 loss eval，没有任务成功率基准 |
| 测试体系 | 35/100 | 有 smoke test，但缺 unit/integration/regression tests |

## 七、我对“当前完成程度”的最终判断

### 作为“研究架构原型”

我认为它已经**相当接近完成**。  
核心创新点不是 PPT，而是已经写成可运行的 PyTorch 模块，并且 smoke test 证明主路径能 forward/backward。

### 作为“真实数据离线训练工程”

它已经**跨过 50% 的关键门槛**，但还没有完全进入“稳定做实验”的状态。  
最关键的短板是：

- 默认配置没有接真实数据；
- 真实数据 batch 没有把 phase/affordance/step_weights/semantic_refresh 这些高级监督字段接起来；
- Stage C 的高级训练机制还没有真正落地。

### 作为“完整 VLA 项目”

它仍然**明显未完成**。  
完整 VLA 项目至少还应有：

- benchmark/eval 套件；
- online rollout/runtime 层；
- 可复现的 real-data config 入口；
- 更完整的 regression tests；
- world model 是否进入主线的明确收敛策略。

## 八、最值得优先补的 6 件事

1. 把真实数据配置接入一级训练入口。最少应让 `configs/train/stage_*.yaml` 能直接合并 `configs/data/...`，否则 README 的默认启动方式会误导用户进入 dummy 路径。
2. 真正实现或暂时删除 Stage C 的 RTC/FASTER 叙事。现在最危险的不是“没实现”，而是“配置里看起来实现了”。
3. 为 HDF5 adapter 增补 phase/affordance/step_weights/semantic_refresh_steps 生成或映射逻辑，否则结构化头的大部分监督在真实数据上是空转的。
4. 建立最小 benchmark harness。哪怕先只有 offline rollout loss + 简单 imitation success proxy，也比现在只看平均 loss 强。
5. 增加真实依赖栈 integration test。至少要覆盖 `transformers + peft + actual processor + real Qwen2VLBackboneWrapper` 初始化，而不是只测 mock backbone。
6. 明确 world model 的路线。如果短期不做，就把它正式降为 experimental branch；如果要做，就需要单独的 Stage D 训练计划和数据接口。

## 最终结论

这个仓库的**核心价值已经成立**：它不是“只有想法”，而是“已经有比较完整的模型主干和训练框架”的研究工程。

但它当前更准确的定位应当是：

**一个完成了主干建模、具备离线训练雏形、正在从 architecture-ready 向 experiment-ready 过渡的 VLA 研究仓库。**

它距离“可被认真用于首轮真实数据实验”已经不远；但距离“完整、可验证、可比较、可部署的系统”还有一段明确且不小的工程差距。
