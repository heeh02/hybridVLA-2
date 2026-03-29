# hybridVLA1 -> hybridVLA2 定向回归审计

本报告基于 `summary_claude.md` 的历史故障库，对 `hybridVLA_2` 的数据、模型、dtype/混合精度、分布式、评估与训练-评估闭环做定向回归排查。结论不是代码风格建议，而是只关注会影响训练正确性、收敛、闭环控制、分布式稳定性、评估可信度的问题。

## A. 高风险问题（按严重性排序）

### 1. FSDP 仍是默认主路径，但模型没有完成 dtype 统一，也没有看到对 Qwen2-VL tied weights 的额外保护

1. 问题标题  
   FSDP 默认启用，但 HybridVLA2 仍保留 backbone `bf16` + 自定义模块 `fp32` 的混合参数体系。
2. 风险等级  
   Critical
3. 在仓库中的具体文件和函数  
   `vla_hybrid_v2/models/qwen2vl_backbone.py:145-190`  
   `vla_hybrid_v2/config.py:235-237`  
   `configs/train/stage_b.yaml:22-27`  
   `configs/train/stage_c.yaml:22-27`  
   `libero_hybrid/configs/train/libero_stage_b.yaml:21-26`  
   `libero_hybrid/configs/train/libero_stage_c.yaml:21-25`  
   `scripts/train_unified.py:400-403`  
   `vla_hybrid_v2/utils/distributed.py:85-125`
4. 证据代码片段或逻辑描述  
   `Qwen2VLForConditionalGeneration.from_pretrained(..., torch_dtype=self._dtype)` 在 backbone 侧显式使用 `bf16`。  
   同一个 `Qwen2VLBackboneWrapper` 内部新建的 `MultiScaleAdapter` / `CameraPositionEmbedding`，以及 `HybridVLAv2` 里的 grounder / temporal / expert / head，都是 PyTorch 默认 `fp32` 初始化。  
   训练配置仍普遍是：

   ```python
   bf16: true
   fsdp: true
   checkpointing: true
   ```

   训练入口仍直接：

   ```python
   if cfg.train.fsdp and get_world_size() > 1:
       model = wrap_fsdp(model, mixed_precision=cfg.train.bf16, ...)
   ```

   但仓库里没有看到类似“在 `__init__` 末尾统一非-backbone 子模块到 `bf16`”或“load checkpoint 后再统一 cast”的逻辑。
5. 它对应的是哪一个 historical issue  
   `bf16 / fp32 混合导致 FSDP 崩溃或数值不一致`  
   `checkpoint load/save 破坏 dtype 统一`  
   `tied weights 与 FSDP / cast 冲突`
6. 为什么它会在训练或评估中造成真实后果  
   这是 HPC3 历史根因的原形重现。当前 intended 路径仍是多卡 FSDP，而不是当时最终绕过问题的 DDP。Qwen2-VL backbone 依然是 tied-weight 大模型，自定义模块仍默认 `fp32`。这会把 wrap-time flatten、resume、mixed-precision all-gather、甚至某些 matmul 路径重新带回不稳定区间。  
   即使能勉强跑起来，也会导致“训练时依赖 autocast 才不报错，分布式包裹/恢复时又重新出错”的脆弱状态。
7. 最小修复方案  
   两条路选一条，不建议继续半修半不修：
   1. 对 HPC3 / LIBERO 路径回退到 DDP，并把配置默认值从 `fsdp: true` 改成明确的 DDP 路径。
   2. 如果坚持 FSDP，则必须在模型构造后、checkpoint load 后做一次确定性的 dtype 统一，并补上 real-backbone + real-checkpoint 的 FSDP wrap/resume 冒烟测试。
8. 我建议优先验证的实验或单元测试  
   `torchrun --nproc_per_node=2` 跑一次真实 `libero_stage_b` 的单 step dry-run，覆盖：
   `HybridVLAv2(cfg)` -> cross-stage `load_checkpoint()` -> `wrap_fsdp()` -> 1 次 forward/backward。  
   这比 CPU/mock 单元测试更接近 HPC3 真实失败面。

### 2. 推理/评估链路没有 autocast，且 eval 仍直接把动作 tensor 转成 numpy

1. 问题标题  
   训练依赖 `autocast` 才能跑通，但 inference/eval 侧没有对应保护；同时 rollout 仍有 `tensor.cpu().numpy()` 直转。
2. 风险等级  
   Critical
3. 在仓库中的具体文件和函数  
   `scripts/train_unified.py:520-521`  
   `vla_hybrid_v2/infer/libero_policy.py:233-250`  
   `vla_hybrid_v2/models/qwen2vl_backbone.py:263-287`  
   `vla_hybrid_v2/infer/libero_policy.py:418-421`  
   `libero_hybrid/scripts/eval_libero_rollout.py:157-170`
4. 证据代码片段或逻辑描述  
   训练侧明确包在 autocast 里：

   ```python
   with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.train.bf16):
       losses = model.forward_train(batch)
   ```

   但推理侧只有：

   ```python
   model = model.to(device).eval()
   ```

   `semantic_step()` / `control_step()` 都没有 `autocast`。  
   同时 rollout 仍有：

   ```python
   actions_batch[k] = action.cpu().numpy()
   ```

   我还做了一个本地最小实验：`nn.Linear(fp32)` 接 `bf16` 输入，在没有 autocast 时直接报 `RuntimeError: mat1 and mat2 must have the same dtype`。这与当前推理路径的混合 dtype 条件一致。
5. 它对应的是哪一个 historical issue  
   `bf16 / fp32 混合导致 FSDP 崩溃或数值不一致`  
   `eval 时 bf16 tensor 直接转 numpy`
6. 为什么它会在训练或评估中造成真实后果  
   如果 checkpoint/模块最终以 `bf16` 输出动作，`cpu().numpy()` 会再次触发历史的 `BFloat16` numpy 转换失败。  
   更严重的是，即便还没走到 numpy，推理图本身就可能先在 backbone `bf16` -> 自定义 `fp32` 线性层处直接炸掉。也就是说，当前评估代码并不是“数值有偏差”，而是“可能第一步就不能稳定执行”。
7. 最小修复方案  
   在 `semantic_step()` / `control_step()` 或 `HybridVLALiberoPolicy` 外层统一加推理 autocast。  
   同时把 rollout 里的动作输出改成：

   ```python
   actions_batch[k] = action.float().cpu().numpy()
   ```

   如果最终决定统一全模型 `bf16`，则这里的 `.float()` 仍然建议保留。
8. 我建议优先验证的实验或单元测试  
   用真实 checkpoint 在 GPU 上做 1 次 `semantic_step_from_obs()` + 1 次 `control_step_from_obs()` 冒烟测试，并强制覆盖：
   `action.dtype`、`action_env.dtype`、`action.cpu().numpy()`。

### 3. consistency loss 仍然把 expert 分支 detach；LIBERO Stage C 仍默认 stop-gradient 条件前缀

1. 问题标题  
   `expert_denoised` 仍在 consistency loss 调用点被 detach；而 LIBERO Stage C 仍然默认 `stop_gradient_cond_prefix: true`。
2. 风险等级  
   High
3. 在仓库中的具体文件和函数  
   `vla_hybrid_v2/models/hybrid_vla_v2.py:573-577`  
   `vla_hybrid_v2/models/hybrid_vla_v2.py:684-691`  
   `libero_hybrid/configs/train/libero_stage_b.yaml:21-26`  
   `libero_hybrid/configs/train/libero_stage_c.yaml:21-25`  
   `configs/train/stage_b.yaml:22-27`
4. 证据代码片段或逻辑描述  
   当前实现仍然有：

   ```python
   if (self.cfg.train.stop_gradient_cond_prefix
           or self.cfg.train.block_fm_to_backbone):
       cond_prefix = cond_prefix.detach()
   ...
   losses["loss_consistency"] = self.consistency_loss(
       ...,
       continuous_actions=expert_denoised.detach(),
   )
   ```

   同时，LIBERO Stage C 配置默认仍是：

   ```yaml
   stop_gradient_cond_prefix: true
   ```

   这意味着即使进入 Stage C，FM 分支默认也不会把表征修正回 backbone/temporal/cond-builder。
5. 它对应的是哪一个 historical issue  
   `expert_continuous.detach() 截断梯度`  
   `Stage A / Stage B / Stage C 的 freeze-unfreeze 策略是否可能固化错误表征`
6. 为什么它会在训练或评估中造成真实后果  
   这是“loss 低但任务失败”的典型结构性来源。  
   FAST head 与 expert 的一致性约束仍是单向的，expert 学不到与 FAST 对齐；而在 LIBERO 路径上，Stage C 仍默认把 expert 条件前缀梯度切断，无法用 expert/FM 反向修正前面学坏的视觉-时序表征。  
   如果 Stage A/B 的表征有偏差，当前 Stage C 默认并不能真正闭环修复。
7. 最小修复方案  
   删除 `continuous_actions=expert_denoised.detach()` 的 detach。  
   把 `libero_stage_c.yaml` 的 `stop_gradient_cond_prefix` 默认改为 `false`。  
   如果仍想保留“知识隔离”实验，请放到单独 ablation config，而不是默认生产配置。
8. 我建议优先验证的实验或单元测试  
   增加一个梯度单测，断言 Stage C 下 `loss_total.backward()` 之后：
   `action_expert`、`cond_builder`、`core_to_expert`、backbone LoRA 至少部分参数梯度非零。  
   然后做 A/B 对照：`detach on/off` 的 closed-loop success rate。

### 4. 多相机路径没有真正闭环：基础 multicam 配置错误，loader 会静默退化为单相机，eval 也未显式请求 wrist camera

1. 问题标题  
   eye-in-hand 分支在 generic 配置、数据读取和 rollout 环境三端都没有形成严格闭环。
2. 风险等级  
   High
3. 在仓库中的具体文件和函数  
   `configs/data/libero_multicam.yaml:4-20`  
   `libero_hybrid/scripts/train_libero.py:28-54`  
   `vla_hybrid_v2/data/libero_hdf5_adapter.py:232-245`  
   `vla_hybrid_v2/data/libero_hdf5_adapter.py:295-340`  
   `vla_hybrid_v2/data/libero_hdf5_adapter.py:423-486`  
   `vla_hybrid_v2/models/hybrid_vla_v2.py:410-415`  
   `vla_hybrid_v2/infer/libero_policy.py:215-219`  
   `vla_hybrid_v2/infer/libero_policy.py:299-305`  
   `libero_hybrid/scripts/eval_libero_rollout.py:117-123`
4. 证据代码片段或逻辑描述  
   generic multicam 配置仍写成：

   ```yaml
   camera_keys:
     - agentview_rgb
     - robot0_eye_in_hand_rgb
     - robot0_agentview_left_rgb
   proprio_key: robot0_joint_pos
   num_cameras: 3
   ```

   但 LIBERO 正常路径实际要靠 `train_libero.py` 再覆盖成 2 相机、`eye_in_hand_rgb`、`joint_states`。  
   更关键的是，`LiberoHDF5DatasetAdapter._collect_demo_refs()` 只校验 proprio，不校验 camera key；而 `_process_text_multi_image()` 对缺失相机是“静默丢弃并回退到单图路径”，`sample["num_cameras"]` 直接取 `len(valid_images)`。  
   最后 `forward_train()` 还只读 batch 第一条样本的 `num_cameras`：

   ```python
   num_cameras = int(nc[0].item())
   ```

   推理端则反过来要求 `multi_camera.enable=True` 时必须恰好 2 相机，否则直接 `NotImplementedError`；eval 端又没有在 `OffScreenRenderEnv` 里显式设置 wrist camera。
5. 它对应的是哪一个 historical issue  
   `eye-in-hand 摄像头缺失`  
   `多相机输入遗漏或命名不一致`
6. 为什么它会在训练或评估中造成真实后果  
   这会产生两类坏结果：
   1. 训练“看起来是 multicam”，实际部分甚至全部样本 silently 退化为 single-cam。
   2. 训练出的 multicam checkpoint 可能在 rollout 时根本拿不到 `robot0_eye_in_hand_image`，导致直接报错或 token layout 与训练不一致。  
   这类问题最容易把精细操作失败伪装成“模型能力不够”。
7. 最小修复方案  
   multicam 训练必须 fail-fast：所有 `camera_keys` 缺任意一个就拒绝该 demo/该 job。  
   删除或修正仓库顶层 `configs/data/libero_multicam.yaml` 的陈旧 key。  
   在 eval 创建环境时显式指定所需相机集合，并增加一个 rollout 前断言，确认 `obs` 真的含 `robot0_eye_in_hand_image`。  
   禁止同一 batch 中混入不同 `num_cameras`。
8. 我建议优先验证的实验或单元测试  
   1. 数据集扫描：统计每个 demo 是否齐备 `agentview_rgb` 和 `eye_in_hand_rgb`。  
   2. multicam batch 单测：只要样本 `num_cameras` 不一致，就必须抛错而不是默认取第一条。  
   3. rollout smoke test：`multi_camera.enable=True` 时第一步就断言 env obs 含 `robot0_eye_in_hand_image`。

### 5. 训练与闭环评估的时序接口并不一致：semantic refresh 训练每 6 步，评估每 4 步；action history 训练比推理慢一拍

1. 问题标题  
   temporal core 的 refresh cadence 和 action-history 语义，在 train vs eval 两端并不一致。
2. 风险等级  
   High
3. 在仓库中的具体文件和函数  
   `vla_hybrid_v2/config.py:228-230`  
   `vla_hybrid_v2/config.py:291-295`  
   `vla_hybrid_v2/models/hybrid_vla_v2.py:391-397`  
   `vla_hybrid_v2/models/hybrid_vla_v2.py:457-512`  
   `vla_hybrid_v2/models/hybrid_vla_v2.py:758-779`  
   `libero_hybrid/scripts/eval_libero_rollout.py:107-110`  
   `libero_hybrid/scripts/eval_libero_rollout.py:157-160`
4. 证据代码片段或逻辑描述  
   训练默认：

   ```python
   semantic_refresh_stride = 6
   medium_update_stride = 2
   ```

   推理默认：

   ```python
   control_hz = 50.0
   semantic_hz = 12.5   # -> refresh every 4 steps
   medium_hz = 25.0     # -> medium every 2 steps
   ```

   我用配置实际算了一次：`train refresh = 6`，`infer refresh = 4`，只有 medium stride 是一致的。  
   另外，训练时 history buffer 在每个 step 末尾 push 的是 `prev_actions[:, t]`；推理时末尾 push 的是刚执行的 `action`。这会让训练端的长历史永远比推理端少最新的一步动作。
5. 它对应的是哪一个 historical issue  
   `训练-评估闭环一致性`  
   `condition prefix 只取最后一步导致时序上下文丢失` 的同类变体
6. 为什么它会在训练或评估中造成真实后果  
   temporal core 明确依赖 `stale_token`、`semantic_refresh`、`action_history_token`。  
   当训练时序统计和闭环 rollout 时序统计不同，模型 offline 学到的是一套 refresh/history 分布，online 却在另一套分布上递推。这非常容易表现为：open-loop loss 很低，但 closed-loop 一上环境就发散或恢复能力很差。
7. 最小修复方案  
   让 infer 侧 refresh cadence 显式对齐 train config，而不是靠 `50 / 12.5` 这类独立默认值。  
   同时把训练 action-history 的更新语义改成与推理一致，至少要保证“history 包含最近实际执行的动作”。
8. 我建议优先验证的实验或单元测试  
   1. 单测：给一段 toy action 序列，比较 train-side 与 infer-side 生成的 `prev_action_token` + `action_history_token` 是否逐步一致。  
   2. 闭环 ablation：只改 refresh cadence 对齐，观察 success rate 是否显著变化。  
   3. 再做一次 action-history 语义对齐 ablation。

### 6. FSDP checkpoint_wrapper 与 Mamba 内部 activation checkpoint 同时启用，且 grad accumulation 没有 `no_sync()`

1. 问题标题  
   当前多卡训练把两套 checkpointing 机制叠在一起，同时在梯度累积时仍默认每个 micro-step 同步。
2. 风险等级  
   High
3. 在仓库中的具体文件和函数  
   `vla_hybrid_v2/utils/distributed.py:85-149`  
   `vla_hybrid_v2/models/mamba_core.py:468-477`  
   `scripts/train_unified.py:400-403`  
   `scripts/train_unified.py:520-540`
4. 证据代码片段或逻辑描述  
   FSDP 包装时，如果 `checkpointing: true`：

   ```python
   model = wrap_fsdp(..., use_activation_checkpointing=True)
   ```

   这会用 `checkpoint_wrapper` 包住 `MambaBlock` / `GrounderBlock` / `Expert*Block`。  
   但 fallback Mamba 路径内部又会在每层再做一次：

   ```python
   x, s, c = activation_checkpoint(layer, x, s_i, c_i, use_reentrant=False)
   ```

   同时训练循环里没有任何 `no_sync()`，`rg "no_sync("` 在训练代码中是空结果。
5. 它对应的是哪一个 historical issue  
   `activation checkpointing 与并行策略不兼容`  
   `grad accumulation 中无效梯度同步`
6. 为什么它会在训练或评估中造成真实后果  
   双重 checkpointing 不是简单“多耗点时间”，它可能重新引入历史上 checkpoint metadata / recompute 路径不一致的问题，尤其是在真实 FSDP + 大模型 + 恢复训练时。  
   而没有 `no_sync()` 会让梯度累积的每个 micro-batch 都发生跨卡同步，极大放大通信负担，掩盖甚至放大 checkpointing/FSDP 的稳定性问题。
7. 最小修复方案  
   只保留一套 checkpointing：  
   对 fallback Mamba，优先保留内部 `activation_checkpoint`，关闭 FSDP `checkpoint_wrapper` 对这些块的包装。  
   在 world size > 1 且非累积末步时，显式使用 `model.no_sync()`。
8. 我建议优先验证的实验或单元测试  
   1. 2-GPU stage B dry-run，分别测试 `checkpointing on/off`。  
   2. profiler 统计 grad_accum=4 时 allreduce 次数，确认非末步不再同步。  
   3. resume 后再做 1 step，确认 nested checkpointing 不引发 recompute metadata 错误。

## B. 历史问题对照表

| historical issue | hybridVLA2 状态 | 证据 | 备注 |
|---|---|---|---|
| proprio key 不匹配导致 fallback 全零 | 已修复 | `libero_hybrid/scripts/train_libero.py:28-54` 会改成 `joint_states + gripper_states`；`vla_hybrid_v2/data/libero_hdf5_adapter.py:232-245,401-412` 现在是 skip/raise，不再 silent zero | 推荐 LIBERO 路径已修；但 `vla_hybrid_v2/config.py:326` 和 `configs/data/libero_multicam.yaml:10` 仍保留危险默认值 |
| 图像方向 bottom-up / top-down 不一致 | 疑似存在 | 训练 `libero_hdf5_adapter._read_image()` 与评估 `_make_pil_image()` 都没有 flip；仓库内也没有 convention 断言 | 代码两端没有显式 mismatch，但也没有任何方向校验；需看真实 HDF5 与 env |
| eval 缺少 action denormalize | 已修复 | `vla_hybrid_v2/infer/libero_policy.py:418-421` | 仍建议把输出 `.float()` 后再转 numpy |
| proprio 错用 action normalizer | 已修复 | `build_dataset()` 分别加载 action/proprio stats；`libero_policy.py:406-407` 用的是 `proprio_normalizer` | 新风险不在“错用 normalizer”，而在 stale config |
| eye-in-hand 摄像头缺失 | 存在 | `configs/data/libero_multicam.yaml:6-10` 仍是错误 key；`libero_hdf5_adapter.py:232-245` 不校验 camera key；`eval_libero_rollout.py:117-123` 未显式请求 wrist camera | multicam 闭环没有真正收口 |
| placeholder 图像尺寸不满足视觉模型要求 | 已修复 | `libero_hdf5_adapter.py:301-308,346-349` 和 `libero_policy.py:162-169` 全部统一 resize 到 `448x448` | 未见 224x224 placeholder 路径 |
| flow matching sampling 时间步不完整 | 已修复 | `vla_hybrid_v2/models/flow_action_expert.py:339-353` 使用 midpoint `t_mid=(i+0.5)*dt` | 该项已经闭环 |
| expert_continuous.detach() 截断梯度 | 存在 | `vla_hybrid_v2/models/hybrid_vla_v2.py:684-691` 仍有 `continuous_actions=expert_denoised.detach()` | 与历史问题同型，且是调用点上的真实 detach |
| grounder 对同一输入重复调用导致 dropout 非确定性 | 已修复 | `vla_hybrid_v2/models/hybrid_vla_v2.py:431-443` 对单次 grounder 输出复用，不再重复 forward | 只有 refresh path 才重新算 grounder，符合设计 |
| condition prefix 只取最后一步导致时序上下文丢失 | 疑似存在 | `vla_hybrid_v2/models/hybrid_vla_v2.py:573-577` 仍只把 `temporal_outputs[-1]` 送入 expert | v2 recurrent core 缓解了问题，但 expert 仍只看最后时刻摘要 |
| attention mask 逻辑/类型问题 | 已修复 | `hybrid_vla_v2.py:427,440,717` 都显式 `.bool()` 后再传 grounder | 未见历史里的 mask bool 漏洞 |
| latent queries 初始化不合理 | 存在 | `vla_hybrid_v2/models/attention_grounder.py:138-140,185-187` 仍是固定 `* 0.02` | 风险低于 v1，但确实没有沿用“更合理初始化”的修复思路 |
| selective scan / broadcast 语义问题 | 已修复 | `vla_hybrid_v2/ops/selective_scan.py:56-63` 当前广播表达与期望 einsum 语义一致 | 未见历史里的错误 broadcast |
| bf16 / fp32 混合导致 FSDP 崩溃或数值不一致 | 存在 | `qwen2vl_backbone.py:160-189` backbone bf16 + adapter fp32；`distributed.py:104-121` 仍默认 FSDP mixed precision | 当前 intended 多卡路径仍踩在历史雷区上 |
| checkpoint load/save 破坏 dtype 统一 | 疑似存在 | `checkpointing.py:164-205` 只 load，不做任何 post-load dtype re-cast | 目前更糟的是根本没看到 dtype 统一流程 |
| tied weights 与 FSDP / cast 冲突 | 疑似存在 | `qwen2vl_backbone.py:160-178` 仍基于 `Qwen2VLForConditionalGeneration`；`distributed.py:112-121` 没有任何 tied-weight 例外逻辑 | 需要真实多卡 wrap 验证 |
| activation checkpointing 与并行策略不兼容 | 存在 | `distributed.py:123-149` + `mamba_core.py:472-475` | 现在是 FSDP checkpoint_wrapper 与层内 activation_checkpoint 双开 |
| checkpoint 保存导致 CPU OOM | 疑似存在 | `checkpointing.py:82-88` 仍先 materialize full state/optim state，再做主进程分支 | 比历史“8 rank 都 save”好，但大模型 FSDP full state 仍重 |
| grad accumulation 中无效梯度同步 | 存在 | `scripts/train_unified.py:514-540` 训练循环没有 `no_sync()`；`rg "no_sync("` 为空 | 正确性影响较小，但 HPC3 吞吐会明显受损 |
| DDP 保存的 module. 前缀导致加载失败 | 疑似存在 | `checkpointing.py:164-205` 没有任何 `module.` strip 逻辑 | 如果复用老 DDP checkpoint，这个坑会重新出现 |
| eval 时 bf16 tensor 直接转 numpy | 疑似存在 | `libero_hybrid/scripts/eval_libero_rollout.py:169-170` 仍是 `action.cpu().numpy()` | 如果动作最终是 bf16，会直接重现历史报错 |
| 训练和评估图像翻转语义不一致 | 疑似存在 | 训练与评估都没有显式 flip 处理，也没有一致性断言 | 静态上看不到“训练翻/评估不翻”，但也看不到任何防回归保护 |
| 环境初始化、assets、日志路径等基础设施问题 | 疑似存在 | `eval_libero_rollout.py:112-133` 直接依赖 `get_libero_path("bddl_files"/"init_states")`；仓库内未见 `.libero/config.yaml`、assets bootstrap、robosuite log path 修复逻辑 | 这类问题仍强依赖用户外部环境手工准备 |

## C. 需要重点人工复核的点

1. 图像方向约定  
   代码里没有任何 vertical flip 或 convention assert，必须拿真实 HDF5 帧和真实 env 帧做逐帧可视化，确认训练与评估看到的不是上下颠倒的同一场景。
2. 真实 Qwen2-VL + GPU 推理 dtype  
   当前推理代码没有 autocast。必须用真实 checkpoint 跑一遍 `semantic_step_from_obs()`，确认不会在 bf16/ fp32 线性层边界报 dtype mismatch，也确认 `action.cpu().numpy()` 不会炸。
3. FSDP wrap / resume / checkpointing  
   现有测试几乎都是 CPU/mock，不覆盖真实 Qwen2-VL + FSDP + checkpoint load。这个组合必须在 2 张卡上做最小 dry-run。
4. multicam env 返回值  
   rollout 环境是否真的返回 `robot0_eye_in_hand_image`，当前静态代码无法确认。需要在 `OffScreenRenderEnv` 初始化后打印一次 obs keys。
5. 训练-评估时序对齐  
   `semantic_refresh_stride=6` vs `refresh every 4 steps` 是静态可见的，但它对 success rate 的敏感度只能通过 ablation 判断。建议至少比较两组：`infer refresh=4` 与 `infer refresh=6`。
6. 旧 checkpoint 兼容性  
   如果你打算复用 hybridVLA / HPC3 旧 DDP checkpoint，必须先做一次 `load_checkpoint()` 冒烟测试，确认不会因为 `module.` 前缀导致基本没加载到参数。

## D. 审查结论

### 哪 3 个问题最可能再次导致“loss 低但任务失败”

1. `expert_denoised.detach()` + LIBERO Stage C 默认 `stop_gradient_cond_prefix: true`  
   这会让 Stage C 仍然无法用 expert/FM 分支修复前面学坏的表征。
2. multicam / eye-in-hand 路径没有闭环  
   训练可能 silently 退化成 single-cam，评估却按 multicam 假设运行，或者根本拿不到 wrist camera。
3. 训练-评估时序接口不一致  
   semantic refresh cadence 与 action-history 语义不一致，会直接放大 open-loop 与 closed-loop 的统计偏差。

### 哪 3 个问题最可能导致训练直接崩溃

1. FSDP 默认启用在 mixed-dtype + Qwen2-VL tied-weight 体系上  
   这是与 HPC3 历史最相似、也最危险的崩溃源。
2. 推理/评估链路没有 autocast，mixed dtype 线性层边界可能直接报错  
   这会让你以为“模型坏了”，实际上是 eval pipeline 先挂了。
3. FSDP checkpoint_wrapper 与内部 activation checkpoint 双开  
   真实多卡 + resume 路径下最可能重新引出 recompute / metadata 类错误。

### 这个仓库当前更像是“能跑但结果不可信”，还是“容易直接挂掉”，还是“总体健康”

如果按 repo 设计的 intended 用法，也就是 Qwen2-VL + LIBERO + 多卡 FSDP + Stage B/C 来看，它更像是：

**容易直接挂掉**。

原因不是单一 bug，而是分布式主路径本身仍站在历史故障根因上。  
但即便你先绕开这些崩溃点，让它“能跑起来”，当前仓库也还明显存在 **“能跑但结果不可信”** 的闭环风险，尤其是：
`expert consistency detach`、`multicam 路径未闭环`、`train/eval temporal interface drift`。

换句话说：  
当前状态不是“总体健康”，而是“多卡主路径容易挂；绕开后结果仍未可信”。
