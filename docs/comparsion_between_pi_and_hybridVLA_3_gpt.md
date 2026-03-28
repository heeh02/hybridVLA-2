# HybridVLA v2 vs OpenPI 对比分析报告 v3 (GPT 代码核验版)

> 审核日期: 2026-03-28
> 审核方式: 本地代码审阅 + `pytest -q` + 最小复现脚本
> 审核范围: 仅重新核验 HybridVLA 仓库本身。OpenPI 侧事实沿用 `comparsion_between_pi_and_hybridVLA_3_claude.md` 的上下文, 本报告未重新联网核对 OpenPI 源码或实验数据。

---

## 1. 一句话结论

更准确的结论是: **HybridVLA v2 不是“完全没验证过的空架子”, 而是“已经有 mock/mini-config 级别回归, 但真实训练配置、官方 Mamba 路径、多卡/FSDP、真实 LIBERO 闭环仍有明显缺口和若干阻塞问题”的研究型代码库。**

---

## 2. 这次实际核验到了什么

- `pytest -q` 在当前环境通过 `49/49`, 耗时 `18.66s`。
- 当前环境 `vla_hybrid_v2.models.mamba_core.HAS_MAMBA_SSM == False`, 所以测试覆盖的是 fallback Mamba 路径, 不是官方 `mamba_ssm` 路径。
- 测试框架使用 CPU 小尺寸配置和 mock backbone, 见 `tests/conftest.py:33-146`。
- 仓库已配置 `pytest` 和 `ruff`, 见 `pyproject.toml:6-29`。
- 仓库中未看到检入的 CI workflow 或 pre-commit 配置。

这意味着:

- 不能再把仓库描述成“完全没跑过任何端到端路径”。
- 也不能把当前测试绿灯误读成“真实训练配置已经被证明可用”。

---

## 3. Claude 文档中关键问题的核验结果总表

| 议题 | GPT 核验 | 结论 |
|------|----------|------|
| FASTER inference 未实现 | 已核实 | 存在 |
| RTC 训练/推理不一致 | 已核实 | 存在 |
| 官方 Mamba 路径逐 token loop + checkpoint 缺口 | 已核实 | 存在 |
| 真实 HDF5/LIBERO 路径缺少 phase/affordance labels | 已核实 | 存在 |
| world model 是未接入主链路的死分支 | 已核实 | 存在 |
| `train_stage_a.py` 只是重复代码 | 已核实并上调风险 | 不只是重复, 还是会绕开统一脚本修复的遗留入口 |
| “零验证” | 需修正 | 不成立为绝对表述; 更准确是“真实配置验证证据不足” |
| “无 linter” | 需修正 | 不成立, `ruff` 已配置 |
| `config.py` 的 `eval()` 是安全漏洞 | 需修正 | 不宜定性为安全漏洞, 更准确是可维护性/反射实现问题 |

---

## 4. 已确认存在的问题

### 4.1 FASTER 推理仍然未实现

证据:

- `vla_hybrid_v2/models/hybrid_vla_v2.py:689-695` 在 `cfg.infer.faster.enable=True` 时直接 `raise NotImplementedError(...)`。
- `tests/test_control_step.py:38-50` 明确把这个异常当成当前预期行为。

影响:

- Stage C 训练可以开启 FASTER, 但推理端没有对等实现。
- 这意味着“训练时使用的 near/far 调度”无法原样带到 rollout。

建议 solution:

1. 要么真正实现与训练对应的 FASTER 推理调度。
2. 要么在 Stage C 配置和 README 中明确把 FASTER 标为 train-only experiment, 并保证推理配置强制关闭。
3. 增加一个配置一致性检查: 若 checkpoint 训练时开启 FASTER, 但推理请求也开启相应模式, 则必须命中已实现路径, 否则在加载阶段就报错。

### 4.2 RTC 存在明确的 train/infer 分布不一致

证据:

- 训练时, RTC 的 `prev_chunk` 用的是当前 `cond_prefix` 采样出来的块, 见 `vla_hybrid_v2/models/hybrid_vla_v2.py:593-607`。
- 推理时, overlap 混合用的是上一个 chunk 的尾部 `runtime_state.prev_chunk_tail`, 它来自前一次 control chunk, 见 `vla_hybrid_v2/models/hybrid_vla_v2.py:772-788`。

这两者不是一回事:

- 训练 RTC: “同一条件前缀下, 当前块头部去贴合伪造的前序块尾部”。
- 推理 RTC: “上一时刻真实生成出的块尾部, 去和本时刻新块头部做拼接”。

影响:

- RTC loss 优化的分布与真实 rollout 分布并不一致。
- 训练可能学到的是“同一观测条件下的自一致性”, 而不是“跨 chunk、跨控制步的边界一致性”。

建议 solution:

1. 训练 RTC 时不要用当前 `cond_prefix` 伪造 previous chunk; 应使用前一个时间步或前一个 execution horizon 对应的 temporal state / proprio / prev_action 条件来生成。
2. 如果 batch 无法提供前一个 chunk 的条件, 先下线 RTC loss, 直到数据和训练接口补齐。
3. 为 RTC 单独加一个 regression test: 构造相邻两个 control chunk, 验证训练使用的 previous condition 与推理缓存逻辑一致。

### 4.3 官方 Mamba 路径存在明显的性能和显存风险, 且 checkpoint 覆盖不到主路径

证据:

- `_MambaStack.forward()` 在官方路径上走的是逐 token、逐层 `layer.step()`, 见 `vla_hybrid_v2/models/mamba_core.py:432-454`。
- `use_checkpoint` 只在 fallback 路径分支使用, 见 `vla_hybrid_v2/models/mamba_core.py:460-468`。
- `_MambaStack` 的官方路径没有任何 `use_checkpoint` 分支。
- FSDP 侧虽然对 `MambaBlock` 做了 checkpoint wrapper, 见 `vla_hybrid_v2/utils/distributed.py:123-149`, 但官方路径实际调用的是 `layer.step()` 而不是 `forward()`, 因此这层 checkpoint 很可能也覆盖不到最重的序列扫描路径。
- 默认配置下 `sequence_window=24`, 见 `vla_hybrid_v2/config.py:224-226`。
- `TriRateMambaCore` 的输入序列长度是 `9 + compressed_slots = 33`, 见 `vla_hybrid_v2/models/mamba_core.py:660-674`。
- 仅按默认配置粗算, 单个 `forward_train()` 至少会触发:
  - Fast: `24 * 33 * 20 = 15,840` 次 `layer.step()`
  - Medium: `12 * 33 * 6 = 2,376` 次
  - Slow: `4 * 33 * 10 = 1,320` 次
  - ActionHistoryEncoder: `24 * 8 * 4 = 768` 次
  - 合计约 `20,304` 次 Python-level `layer.step()` 调用

影响:

- 这不是“可能慢一点”, 而是训练速度、kernel launch overhead、激活内存曲线都可能和设计预期严重偏离。
- 当前测试环境又没有安装 `mamba_ssm`, 所以最关键的官方路径实际上没有被自动化验证。

建议 solution:

1. 先把官方 Mamba 路径当作单独的手工验证目标, 而不是默认可信路径。
2. 增加一个 `@pytest.mark.manual` GPU 测试, 至少覆盖:
   - `HAS_MAMBA_SSM=True`
   - 单次 `forward_train/backward`
   - `cfg.train.checkpointing=True`
   - 记录显存峰值和 step time
3. 如果官方路径不能在显存/速度上闭合, 就应暂时默认 fallback 路径或提供显式开关, 避免 README 把“官方 Mamba 推荐训练”写成事实默认值。
4. 从实现上, 优先考虑“训练阶段 fused sequence path + 仅推理阶段显式 state step path”的双路径设计, 不要把 offline training 也绑在 token-by-token `step()` 上。

### 4.4 真实 HDF5 / LIBERO 数据路径没有给 phase / affordance 头提供监督

证据:

- `hdf5_adapter.py` 和 `libero_hdf5_adapter.py` 构造 sample 时只放入 `input_ids / attention_mask / actions / proprio / prev_actions / embodiment_id / refresh_* / num_cameras`, 见:
  - `vla_hybrid_v2/data/hdf5_adapter.py:401-464`
  - `vla_hybrid_v2/data/libero_hdf5_adapter.py:468-524`
- 这两条真实数据路径都没有生成 `phase_labels` 或 `affordance_labels`。
- 但模型默认会创建 `phase_head` 和 `affordance_head`, 见 `vla_hybrid_v2/models/hybrid_vla_v2.py:121-136`。
- 对应 loss 只有在 batch 里存在这些 key 时才计算, 见 `vla_hybrid_v2/models/hybrid_vla_v2.py:508-530`。

影响:

- 默认配置下, phase/affordance 头在真实数据路径上大概率没有直接监督。
- 这些头不是“学得差”, 而是“根本可能没在训练目标里出现”。
- 更糟的是, 相关 token 仍然被放进 cond_prefix, 容易让人误以为这些结构 token 有可靠语义。

建议 solution:

1. 如果数据集没有这类标注, 默认关闭 `phase_head` 和 `affordance_head`。
2. 如果这些头是核心设计, 就必须在 adapter 中显式补标签映射或弱监督生成逻辑。
3. 添加启动期断言: 当 `phase_head=True` 或 `affordance_head=True` 时, 训练数据必须声明是否提供相应标签, 否则日志里给出明确 warning。

### 4.5 world model 目前属于未接入主闭环的死分支

证据:

- 模型只在初始化时可选创建 `imagination_engine` 和 `world_model_loss_fn`, 见 `vla_hybrid_v2/models/hybrid_vla_v2.py:182-207`。
- 提供了 `get_world_model_state()` 接口, 见 `vla_hybrid_v2/models/hybrid_vla_v2.py:213-227`。
- 但代码搜索结果表明, 这些对象没有被 `forward_train()`、`train_unified.py`、推理路径或测试真正调用。
- `vla_hybrid_v2/world_model/*.py` 总计约 `1129` 行。

影响:

- 这是明确的维护负担, 而不是已经接入的实验分支。
- 任何围绕 world model 的“架构优势”目前都还不能算进训练系统能力。

建议 solution:

1. 若短期不用, 迁移到 `experimental/` 或直接移出主路径。
2. 若要保留, 就至少补一个最小闭环:
   - 构造 world model 输入
   - 计算 world model loss
   - 在 `forward_train()` 里 gated 接入
   - 增加最小单测

### 4.6 `train_stage_a.py` 不是单纯的重复代码, 而是危险的遗留入口

证据:

- README 已把它标为 legacy, 见 `README.md:268-269`。
- 但 `scripts/train_stage_a.py:1-278` 仍是可直接执行的训练入口。
- 相比 `scripts/train_unified.py:462-589`, 它缺少:
  - 统一 stage gate 与 sanity check
  - validation/evaluate
  - resolved config 资产打包
  - per-module LR groups
  - per-module grad norm logging
  - 修正后的 loss averaging 逻辑
- 因此它不是“写法旧一点”, 而是“运行语义已经和主入口分叉”。

影响:

- 用户只要误用旧入口, 就会绕开统一脚本里后来补上的一系列修复和可观测性。
- 审计报告如果只写“278 行重复代码”, 低估了它的实际风险。

建议 solution:

1. 最好删除 `train_stage_a.py`。
2. 如果保留, 至少把它改成一个只做参数解析后直接调用 `train_unified.train(cfg)` 的薄包装。
3. README 和 docs 里只保留统一入口, 不再给 legacy 脚本提供使用示例。

---

## 5. 新增发现: 数据协议与模型消费逻辑不一致

这是 Claude 文档没有指出、但我认为实际优先级不低的问题。

### 5.1 可选字段允许为 `None`, 但模型按“key 是否存在”分支, 导致直接崩溃

证据:

- `WindowSample` 文档明确把 refresh 和 label 字段定义为 Optional, 见 `vla_hybrid_v2/data/schema.py:33-48`。
- `vla_collate_fn()` 也会把整列 `None` 折叠成 `batch[key] = None`, 见 `vla_hybrid_v2/data/collate.py:59-62`。
- 但 `forward_train()` 对 refresh / phase / affordance 的判断是:
  - `if "refresh_input_ids" in batch: ...`
  - `if self.phase_head is not None and "phase_labels" in batch: ...`
  - `if self.affordance_head is not None and "affordance_labels" in batch: ...`
  见 `vla_hybrid_v2/models/hybrid_vla_v2.py:380-392` 和 `508-530`。
- 我用最小复现脚本验证了两种情况都会直接报:
  - `refresh_input_ids=None` -> `TypeError: 'NoneType' object is not subscriptable`
  - `phase_labels=None` -> `TypeError: 'NoneType' object is not subscriptable`

影响:

- 当前代码实际上要求“缺失可选字段时必须省略 key”, 而不是使用 `None`。
- 这与 schema/collate 的语义不一致, 属于明确的 batch contract bug。
- 这类 bug 很隐蔽, 因为当前真实 adapter 恰好大多选择“省略 key”, 所以没有被现有测试撞出来。

建议 solution:

1. 所有可选字段统一改成“取值判空”, 不要再用“只看 key 是否存在”:
   - `refresh_input_ids = batch.get("refresh_input_ids")`
   - `if refresh_input_ids is not None: ...`
   - `phase_labels = batch.get("phase_labels")`
   - `if phase_labels is not None: ...`
2. `_validate_batch()` 也应同步做 Optional-aware 校验。
3. 增加三组单测:
   - `refresh_input_ids=None`
   - `phase_labels=None`
   - `affordance_labels=None`

---

## 6. 需要修正或收窄的原报告表述

### 6.1 “零验证”这个说法太绝对

更准确的表述:

- 有自动化验证, 但主要是 mock/mini-config 级别。
- 没有证据证明真实训练配置已经闭环。

证据:

- `tests/conftest.py:33-146` 使用小尺寸配置 + mock backbone。
- `tests/test_forward_train.py:26-133` 覆盖的是 mini batch 的 forward/backward。
- `pytest -q` 当前为 `49/49` 通过。
- 当前环境 `HAS_MAMBA_SSM=False`, 所以最重的官方路径未被覆盖。

因此应把原来的“零验证”改成:

**“存在基础级自动化验证, 但缺少真实配置、官方 Mamba、多卡 FSDP 和真实数据路径的强证据。”**

### 6.2 “无 linter”不成立, 但“无可见 CI workflow”成立

证据:

- `pyproject.toml:14-29` 已配置 `ruff` 和 `ruff format`。
- `pyproject.toml:6-12` 已配置 `pytest`。
- 仓库里未看到 `.github/workflows/*` 或 `.pre-commit-config.yaml`。

因此更准确的说法是:

**“仓库已有本地 lint/test 配置, 但未看到检入的自动执行入口。”**

### 6.3 `config.py` 的 `eval()` 不应直接定性为安全漏洞

证据:

- `vla_hybrid_v2/config.py:364-384` 中的 `eval(ft, ...)` 是用来解析 dataclass annotation string, 不是执行 YAML 里的值。

这仍然值得改:

- 反射方式脆弱
- 可读性差
- 对静态分析不友好

但更准确的定性是:

**“实现方式不优雅, 建议用 `typing.get_type_hints()` 替换”, 而不是“配置注入安全洞”。**

---

## 7. 设计风险: 我同意需要实验, 但不把它们写成代码 bug

以下几项我认为应该保留为“高风险设计假设”, 而不是“已确认代码错误”:

- `ActionHistoryEncoder` 用 4 层 Mamba 编 8 个动作, 设计明显偏重, 但是否“过度参数化到无意义”仍需要 ablation 才能定论。
- `ContrastiveTemporalLoss / SlowFastAgreementLoss / ActionConsistencyLoss` 的理论收益存在疑问, 见 `vla_hybrid_v2/losses/consistency_loss.py:16-95`, 但这属于研究设计风险, 不是实现 bug。
- “FAST 头命名是否误导”是命名/论文表达问题, 不是代码正确性问题。

更专业的表述应是:

**“这些模块增加了复杂度, 但当前缺少实验证据证明其收益大于训练成本。”**

---

## 8. 结合 OpenPI 的更稳妥结论

如果只结合现有仓库代码而不额外重跑实验, 我认为与 OpenPI 的真实差距应该这样描述:

- HybridVLA 的主要问题不是“代码完全不能跑”, 而是“真实主路径没有被足够强地验证到可以支撑复杂架构主张”。
- OpenPI 的主要优势不是某个单独模块, 而是更早闭合了训练、推理、评估这一整条主链路。
- HybridVLA 当前最像“强架构假设 + 中等强度本地回归 + 缺真实配置证据”的研究原型, 而不是“完全没工程化”的草稿, 也不是“可直接与 OpenPI 同等级对比”的成熟系统。

---

## 9. 我建议的优先级

### 第一优先级

1. 修掉 Optional field contract bug。
2. 决定 FASTER 是真的要做推理闭环, 还是先从 Stage C 移出。
3. 修 RTC 的 train/infer 条件不一致。
4. 对官方 Mamba 路径做一次真正的手工 benchmark:
   - 单次 forward/backward
   - 峰值显存
   - step/sec
   - checkpoint 开/关对比

### 第二优先级

5. 对真实 HDF5/LIBERO 路径补 phase/affordance 标签逻辑, 或者默认关闭这些头。
6. 删除或薄包装 `train_stage_a.py`。
7. 为以下路径补自动化测试:
   - `refresh_input_ids` 分支
   - Optional field = None
   - `HAS_MAMBA_SSM=True` 的 manual test
   - FSDP multi-GPU manual test
   - 至少一个真实 adapter 的 smoke batch

### 第三优先级

8. 处理 world model:
   - 要么接进主训练闭环
   - 要么从主路径移出
9. 用 `typing.get_type_hints()` 替换 `config.py` 里的 `eval()`
10. 补 CI / pre-commit, 让现有 `ruff + pytest` 真正自动执行

---

## 10. 最终判断

如果问题是“Claude 文档里列的问题到底存不存在”, 答案是:

- 大部分主问题是存在的, 而且其中几项是明确的阻塞项。
- 但原文里有几条结论说得过头了, 尤其是“零验证”“无 linter”“eval 安全漏洞”。
- 同时, 本次又发现了一个此前没写出来的真实 bug: Optional field 的 `None` 语义和 model 端消费逻辑不一致。

因此我对这份仓库的更准确评价是:

**它已经不是空壳, 但距离用代码证据支撑其复杂架构主张, 还差若干关键闭环。**
