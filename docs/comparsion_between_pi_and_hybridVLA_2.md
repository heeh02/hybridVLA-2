# HybridVLA v2 vs OpenPI 深度对比分析报告（更新版）

> 分析日期: 2026-03-28  
> HybridVLA 仓库: `/Users/geminihe/hybridVLA_2`  
> OpenPI 参照仓库: `/Users/geminihe/code/openpi/openpi`  
> 分析方法: 静态代码审读 + `pytest -q` 实测 + 最小化 `control_step()` 复现

---

## 1. 核心结论

这轮修改之后，HybridVLA 已经明显不是上一版报告里的“Stage C 配了但没接、推理链路几乎空壳”的状态了。当前仓库已经具备：

- 三阶段训练主路径
- Stage C 的 RTC / FASTER 训练损失
- `semantic_step()` / `control_step()` 在线接口
- LIBERO 数据、训练、rollout 评估闭环
- `42` 个通过的测试

但如果和 OpenPI 直接比较，当前最主要的问题已经从“训练代码不完整”转移成了两类更硬的工程问题：

1. **训练空间和推理空间没有完全对齐**
2. **部署/推理层抽象还没有达到 OpenPI 的成熟度**

这意味着 HybridVLA 现在更像一个“已经能跑训练、也有闭环评估骨架的研究型系统”，而 OpenPI 仍然是“可训练、可部署、可复用的产品级研究代码库”。

一句话判断：

- 如果目标是 **研究新架构、做 ablation、探索时序建模**，HybridVLA 已经有不少地方比 OpenPI 更激进、更有想法。
- 如果目标是 **稳定复现、真实机器人接入、训练/推理一致性、部署成熟度**，OpenPI 仍然明显领先。

---

## 2. 相比上一轮，HybridVLA 已经补齐了什么

这一轮代码相对上一版对比报告，确实有实质进展：

| 项目 | 之前 | 当前状态 | 评价 |
|------|------|----------|------|
| Stage C 训练 | 配置存在，主干未真正接入 | `forward_train()` 已接入 RTC / FASTER | 属于明确补齐 |
| 在线控制接口 | `infer/` 几乎空 | `semantic_step()` / `control_step()` 已实现 | 骨架已成型 |
| LIBERO rollout | 之前闭环不完整 | `eval_libero_rollout.py` 已具备 per-env state、refresh、prev_action | 明显进步 |
| 测试 | 旧报告认为几乎没有 | 当前 `pytest -q` 实测 `42 passed` | 基础质量显著提升 |
| LIBERO 工程化 | 基础训练配置为主 | 现在有 `train_libero.py` / `compute_libero_stats.py` / `validate_libero_hdf5.py` / `resolved_config.yaml` | 可用性提升明显 |

这说明你的修改不是“文档级修补”，而是真正把几个关键闭环接起来了。

---

## 3. 和 OpenPI 相比，当前还存在什么问题

下面按严重性排序。前四项我认为属于当前最需要优先处理的差距。

### P0. 训练和推理的归一化空间不一致

这是目前最严重的问题。

证据：

- 训练侧 `vla_hybrid_v2/data/libero_hdf5_adapter.py` 在构建样本时，明确对 `actions` 和 `proprio` 做了 normalizer 变换。
- 评估侧 `libero_hybrid/scripts/eval_libero_rollout.py` 直接把原始 `proprio` 喂给 `control_step()`，并把 `control_step().action` 直接送进 `env.step()`。
- 当前 rollout 路径里没有看到对应的 `proprio normalize` 和 `action denormalize`。

这和 OpenPI 的策略接口差异非常大。OpenPI 在 `policy_config.create_trained_policy()` 里把：

- `Normalize(...)`
- `Unnormalize(...)`

统一放进 policy pipeline，因此训练和推理默认走的是同一套数值空间。

对 HybridVLA 的影响：

- 如果训练 stats 不是恒等映射，当前 rollout 不是在训练时定义的输入/输出空间上运行。
- 即使模型本身没问题，闭环成功率也可能被系统性拉低或完全失真。
- 这会直接削弱“LIBERO closed-loop ready”的结论可信度。

我的判断：

- 这是一个 **比缺少 server/client 还更优先** 的问题。
- 在修掉这个问题之前，任何 rollout 数字都应该谨慎解释。

### P0. 默认 `control_step()` 路径当前就可能报错

这是一个已经复现的在线推理 bug。

证据：

- `vla_hybrid_v2/config.py` 的 `RTCInferConfig` 没有 `overlap_ratio` 字段。
- `vla_hybrid_v2/models/hybrid_vla_v2.py` 的 `control_step()` 却会访问 `rtc_cfg.overlap_ratio`。
- 我用最小模型直接调用 `control_step()`，已经实测复现：
  - `AttributeError: 'RTCInferConfig' object has no attribute 'overlap_ratio'`

更严重的是：

- `infer.rtc.enable` 默认是 `True`
- 所以这不是边缘路径，而是默认在线路径

这说明当前测试虽然都过了，但没有覆盖真正的 online control 主路径。

### P0. FASTER 推理配置存在，但推理逻辑并未真正实现

当前状态是：

- 训练侧 Stage C 已经接入 `train.faster`
- 推理侧 `FASTERInferConfig` 存在
- 但 `control_step()` 中没有真正使用 `cfg.infer.faster`

这带来的问题是：

- 训练和推理目标不完全一致
- 配置层暴露出“看起来可用”的能力，但运行时没有落实
- 文档与代码容易继续漂移

和 OpenPI 相比，这类“配置存在但路径不闭合”的情况仍然偏多。OpenPI 的 policy pipeline 通常是“能力暴露即有明确入口”，不会留下这么多半接线状态。

### P0. 推理链路仍然存在 fail-open，而不是 fail-fast

当前 HybridVLA 的若干关键路径在依赖缺失时会“继续跑”，而不是“立即失败”。

具体表现：

- `eval_libero_rollout.py` 如果 `AutoProcessor` 加载失败，只会 `warning`，随后 `obs_to_semantic_input()` 返回全零 token。
- `obs_to_proprio()` 如果环境里没拿到目标 key，会回退成全零 proprio。
- HDF5 / LIBERO adapter 在 `processor is None` 时也会构造占位 token。

这类设计的风险是：

- 代码表面“没崩”，但实际上模型已经退化到异常输入空间
- 实验结果会变成 silent failure，最难排查

OpenPI 在这方面明显更严格：

- 没有 norm stats 会直接报错
- policy 构造会显式加载资产和归一化信息
- 训练/推理的 transform contract 更统一

所以从工程可靠性上看，OpenPI 依然更强。

### P1. 缺少统一的 `Policy` 抽象，训练/推理预处理仍然分裂

OpenPI 的一个关键优势不是模型本身，而是接口设计：

- `libero_policy.py` 定义环境到模型、模型到环境的输入输出映射
- `policy_config.py` 把 transforms、normalization、checkpoint、metadata 统一封装成 policy
- 同一套 policy 能服务训练后的本地推理、远程推理、client/server

HybridVLA 当前不是这样：

- 训练侧 LIBERO 输入逻辑在 `vla_hybrid_v2/data/libero_hdf5_adapter.py`
- 评估侧输入逻辑在 `libero_hybrid/scripts/eval_libero_rollout.py`
- 归一化、tokenization、camera 映射、proprio 拼接并没有进入统一 policy abstraction

这会带来持续风险：

- 一侧修了，另一侧容易忘
- 训练/评估 drift 难避免
- 未来接真实机器人时还要再复制第三套逻辑

从架构层面看，HybridVLA 在“模型模块化”上比 OpenPI 更细，但在“系统接口统一”上比 OpenPI 更弱。

### P1. `infer/` 仍然几乎是空包，部署栈和运行时抽象还没成体系

当前仓库虽然已经有 `control_step()`，但 `vla_hybrid_v2/infer/__init__.py` 仍然只有一句模块说明。

和 OpenPI 相比，HybridVLA 还缺：

- 通用 policy wrapper
- 远程推理 server
- websocket client
- action chunk broker
- 机器人 runtime / agent / subscriber 抽象
- Docker/compose 的部署样例

这意味着：

- 当前仓库可以“研究模型”
- 但还不适合“把模型接进一个长期运行的机器人系统”

OpenPI 在这方面明显成熟得多。

### P1. Checkpoint 资产管理不如 OpenPI 完整

OpenPI 的 checkpoint 不只是权重，还会把 norm stats 等资产一起复制到 checkpoint assets 中，policy 加载时直接读取，训练/推理一致性更强。

HybridVLA 当前 checkpoint 主要保存：

- model
- optimizer
- scheduler
- ema
- meta

但没有把 normalizer stats、policy metadata 等随 checkpoint 一起封装。结果就是：

- checkpoint 本身不足以独立恢复真实推理语义
- rollout 逻辑必须额外知道 stats 路径
- 更容易出现“权重对了，输入输出空间错了”

这正是 OpenPI 已经系统化解决、而 HybridVLA 仍明显落后的地方。

### P1. 测试数量提升了，但关键覆盖面仍然不如 OpenPI

当前 HybridVLA 的进步值得肯定：

- `pytest -q` 实测 `42 passed`

但测试面仍然集中在：

- `forward_train`
- loss
- normalizer
- config resolution
- expert ODE

缺的关键测试包括：

- `control_step()` 在线推理主路径
- 推理时 normalization / denormalization
- RTC infer / FASTER infer
- processor failure 的 fail-fast 行为
- HDF5 / LIBERO adapter 的真实样本契约
- checkpoint 资产恢复
- distributed / FSDP / resume 回归

OpenPI 的测试虽然不一定每个都重，但覆盖面横跨：

- model
- lora
- tokenizer
- transforms
- data_loader
- train resume
- policy
- client-side utilities

所以当前 HybridVLA 的测试结论更适合作为“训练 sanity”，还不足以作为“部署稳定性”证明。

### P1. 文档、配置、代码仍存在漂移

当前仓库还有几个明显的漂移信号：

- README 仍写着“RTC/FASTER loss implementation not yet wired”，但代码其实已经接入
- README 仍把仓库描述为“LIBERO benchmark closed-loop ready”，但当前默认 `control_step()` 仍可直接报错
- `InferConfig` 中存在一些尚未真正落地或未统一收敛的字段

这种漂移的影响不是表面上的“文档没更新”，而是：

- 用户会误判当前系统真实成熟度
- 后续维护会越来越依赖口头上下文
- 新加入的人更难分辨哪些能力是“真 ready”，哪些是“设计意图”

### P1. 世界模型仍是脚手架，不是已闭环能力

世界模型目录已经不小，模块拆分也比较完整，但当前状态仍然是：

- `world_model.enable = false`
- 主训练路径里没有把 imagination / world-model loss 真正并入
- 没有测试、没有验证、没有 benchmark 结果

这说明它目前更像“未来研究方向储备”，不是当前对比 OpenPI 的现实竞争力。

### P2. 泛化与真实机器人证据仍然远少于 OpenPI

这是代码之外，但对仓库定位非常关键。

OpenPI 的优势是：

- 有预训练基座
- 有真实机器人微调 checkpoint
- 有 ALOHA / DROID / LIBERO / UR5 等完整例子
- 有 remote inference 与 deployment 文档

HybridVLA 当前更像：

- 研究型架构 + LIBERO oriented 工程化
- 还没有形成“模型资产 + 部署资产 + 平台资产”的完整体系

所以从“仓库即产品”角度，OpenPI 仍然代差明显。

### P2. 默认数据入口仍允许回退到 dummy dataset

`vla_hybrid_v2/data/__init__.py` 里，`cfg.data.format is None` 会回退到 `DummyVLADataset`。

虽然你已经用 `libero_hybrid/scripts/train_libero.py` 专门绕开了这个风险，但全仓库层面这个设计仍然意味着：

- 对新用户来说，配置写错时有可能“训练跑起来了”，但其实没在用真实数据
- 这类 silent fallback 在研究仓库里会制造很多假进展

OpenPI 在这方面更倾向于显式配置和 fail-fast。

---

## 4. HybridVLA 相对 OpenPI 的真实优势

上面的问题很多，但 HybridVLA 并不是“全面落后”。它有几项优势是明确存在的。

### 优势 1. 时序建模明显比 OpenPI 更深入

OpenPI 的核心强项是大规模数据和成熟训练/部署，但在“显式时序结构”上它相对保守。

HybridVLA 的优势在于：

- tri-rate temporal core
- 显式 recurrent state
- stale-time encoding
- action history encoder
- semantic refresh / medium update / control update 三种频率解耦

这套设计至少在研究层面比 OpenPI 更有“控制系统意识”。

换句话说：

- OpenPI 更像“强大的单步 VLA”
- HybridVLA 更像“把 VLA 往状态化控制器推进”

这不代表它一定更强，但研究上更有独立价值。

### 优势 2. 中间表征更结构化、更可解释

HybridVLA 的 grounder 不是简单把 backbone 特征直接扔给动作专家，而是显式构造：

- global token
- object slots
- compressed slots
- phase token
- uncertainty token
- affordance token

相对 OpenPI，这种设计的优势是：

- 更方便做可解释性分析
- 更适合后续接世界模型、subgoal、affordance 研究
- 更容易做模块级 ablation

代价是系统复杂度更高，但这确实是 HybridVLA 的独到点。

### 优势 3. 对原始 LIBERO HDF5 更友好

OpenPI 对 LIBERO 的正式路径更偏向：

- 先转换成 LeRobot
- 再走统一 data/policy pipeline

HybridVLA 的优势是：

- 直接读 official LIBERO / robomimic 风格 HDF5
- 有 suite 解析、stats 计算、结构验证、rollout 脚本
- 对“只想快速在 LIBERO 上试新架构”的研究者更直接

所以在“原始 LIBERO 实验便利性”上，HybridVLA 其实比 OpenPI 更顺手。

### 优势 4. 纯 PyTorch 的研究可改性更强

这点非常实际。

OpenPI 虽然现在也有 PyTorch 支持，但其 README 明确写了 PyTorch 路径当前仍缺：

- mixed precision training
- FSDP
- LoRA training
- EMA

HybridVLA 当前 PyTorch 路径已经具备：

- FSDP 包装
- LoRA backbone 适配
- EMA
- 显式 stage gate
- 单脚本三阶段训练

因此如果比较的是“纯 PyTorch 研究工作流”，HybridVLA 其实比 OpenPI 的 PyTorch 分支更激进，也更适合做结构改造。

需要强调的是：

- 这只是说 **PyTorch 研究可改性**
- 不代表其整体工程成熟度超过 OpenPI

### 优势 5. 模块边界更清晰，适合做学术 ablation

HybridVLA 当前模块划分很细：

- backbone
- grounder
- temporal core
- action history
- expert
- discrete heads
- world model scaffold

这种设计的好处是：

- 更容易替换单个子模块
- 更适合做论文级 ablation
- 对“想知道某个设计到底有没有贡献”的研究工作更友好

OpenPI 的系统接口更成熟，但从“研究改装台”的角度，HybridVLA 更像一个开放实验平台。

---

## 5. 综合判断

如果按维度打一个定性判断：

| 维度 | 结论 |
|------|------|
| 架构创新性 | HybridVLA 更强 |
| 时序建模深度 | HybridVLA 更强 |
| 中间表征可解释性 | HybridVLA 更强 |
| 原始 LIBERO 接入便利性 | HybridVLA 更强 |
| PyTorch-only 研究改造性 | HybridVLA 更强 |
| 训练/推理一致性 | OpenPI 明显更强 |
| Policy 抽象与部署栈 | OpenPI 明显更强 |
| Checkpoint 资产完整性 | OpenPI 明显更强 |
| 测试覆盖面与 CI | OpenPI 更强 |
| 真实机器人经验与可复用资产 | OpenPI 代差领先 |

最重要的现实判断是：

- **HybridVLA 当前已经不是“不完整原型”，而是“有研究价值、训练闭环基本成型、但推理与部署层还没有收口”的系统。**
- **它和 OpenPI 的差距，已经不主要在模型想法，而主要在 system glue。**

这其实是好消息，因为 system glue 通常比重写整套模型更容易补。

---

## 6. 我建议你下一步优先做什么

如果目标是把 HybridVLA 真正推进到“能和 OpenPI 在工程上正面对比”的阶段，我建议优先级如下：

1. 修正推理归一化闭环  
把 `proprio normalize`、`action denormalize`、stats 加载正式接入 inference / rollout / future policy wrapper。

2. 修掉 RTC infer schema bug，并补 `control_step()` 测试  
至少覆盖 chunk 生成、RTC、refresh、prev_action、runtime cache。

3. 要么真正实现 FASTER infer，要么暂时移除 infer 侧暴露配置  
避免训练/推理不一致和伪能力暴露。

4. 抽一个统一 `Policy` 层  
把 tokenization、camera mapping、proprio mapping、normalization、checkpoint assets 放到一个统一入口。

5. 把 checkpoint 从“权重快照”升级成“可推理资产包”  
至少包含 resolved config、normalizer stats、必要 metadata。

6. 建最小 CI  
哪怕只有 `ruff + pytest + 一个 control_step smoke test`，收益都会很高。

---

## 7. 最终结论

与 OpenPI 相比，HybridVLA 当前**最大的短板不是模型结构，而是训练/推理一致性、推理接口抽象、资产封装与部署成熟度**。  
与 OpenPI 相比，HybridVLA 当前**最大的优势是时序建模、结构化接地、PyTorch 研究可改性，以及直接面向原始 LIBERO HDF5 的实验便利性**。

所以最准确的定位应该是：

- **OpenPI 是成熟的 VLA 系统基线**
- **HybridVLA 是更有研究野心、但仍需把系统接口打磨完整的下一代研究型仓库**

只要把 inference normalization、policy abstraction、checkpoint assets 这几项补齐，HybridVLA 和 OpenPI 的差距会显著缩小；否则它会长期停留在“架构很强，但系统不够闭合”的状态。
