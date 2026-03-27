# HybridVLA v2 Backbone Architecture (v0.6)

> 生成日期: 2026-03-26
> 基于: `analysis_v0_6.md` + `vla_hybrid_v2/models/` 全部代码 (7 files)
> 总参数量: ~9.9B (7.6B frozen backbone + ~2.3B trainable)

---

## 0. 全局总览图

```
                         ┌─────────────────────────────────────────────────────────────────────┐
                         │                    HybridVLAv2  (~9.9B params)                      │
                         └─────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────┐
   │  input_ids, attention_mask,  │
   │  pixel_values, image_grid    │
   └──────────────┬───────────────┘
                  │
                  ▼
  ╔══════════════════════════════════╗
  ║  1. Qwen2-VL-7B Backbone        ║  7.6B params (frozen + LoRA r=64)
  ║     Multi-Scale [L10, L18, L28] ║  3584d → 2048d
  ╚══════════════╤═══════════════════╝
                 │ [B, N, 2048]
                 ▼
  ╔══════════════════════════════════╗
  ║  2. Hierarchical Attention      ║  96 latents → 72 latents (after compression)
  ║     Grounder (8 layers)         ║  8 × GrounderBlock + SlotCompression
  ╚══════════════╤═══════════════════╝
                 │ GrounderOutput (6 fields, all 2048d)
                 │
    ┌────────────┼─────────────────────────────────────┐
    │            │                                     │
    │            ▼                                     ▼
    │   ┌──────────────────┐                ┌─────────────────────┐
    │   │  proprio_proj    │                │  StaleTimeEncoding   │
    │   │  prev_action_proj│                │  embodiment_emb      │
    │   │  (14→2048)       │                │  ActionHistoryEncoder│
    │   └────────┬─────────┘                └──────────┬──────────┘
    │            │                                     │
    │            └────────────────┬─────────────────────┘
    │                            │
    │                            ▼  33 tokens [B, 33, 2048]
    │            ╔══════════════════════════════════╗
    │            ║  3. Tri-Rate Mamba Core          ║
    │            ║  ┌─────────┬──────────┬────────┐ ║
    │            ║  │Fast 20L │Medium 6L │Slow 10L│ ║
    │            ║  │d_s=128  │d_s=128   │d_s=256 │ ║
    │            ║  │every stp│every 2nd │refresh │ ║
    │            ║  └────┬────┴────┬─────┴───┬────┘ ║
    │            ║       └────┬────┘         │      ║
    │            ║       CrossAttentionFusion        ║
    │            ╚══════════════╤════════════════════╝
    │                          │ TemporalOutput (fused + fast/med/slow tokens, 2048d)
    │                          │
    ├──────────────────────────┤
    │                          │
    ▼                          ▼
  ┌────────────┐    ┌──────────────────────┐
  │ 5. Discrete│    │ 4. Cond Prefix       │
  │    Heads   │    │    Builder           │
  │            │    │ 32 tokens [B,32,2048]│
  │ FAST 512   │    │     ↓                │
  │ Phase 16   │    │ core_to_expert       │
  │ Afford 8   │    │ 2048 → 1536          │
  └────────────┘    └──────────┬───────────┘
                               │ [B, 32, 1536]
                               ▼
                    ╔═══════════════════════════╗
                    ║  6. Flow Action Expert    ║  Stage B/C only
                    ║     18L (M-M-A × 6)      ║
                    ║     d=1536, heads=24      ║
                    ║     AdaRMSNorm + ODE      ║
                    ╚═══════════╤═══════════════╝
                                │
                                ▼
                    ┌───────────────────────┐
                    │ velocity [B, 24, 14]  │
                    │ → denoised actions    │
                    └───────────────────────┘
```

---

## 1. Qwen2-VL-7B Backbone (`qwen2vl_backbone.py`)

### 1.1 结构

```
                        Qwen2VLBackboneWrapper
 ┌──────────────────────────────────────────────────────────┐
 │                                                          │
 │  ┌─────────────────────────────────────────────────┐     │
 │  │    Qwen2-VL-7B-Instruct  (3584d, 28 layers)    │     │
 │  │                                                  │     │
 │  │  ┌────────────────────────────────────────────┐  │     │
 │  │  │  Vision Tower (ViT)                        │  │     │
 │  │  │  ● frozen=True                             │  │     │
 │  │  │  ● pixel_values → visual tokens            │  │     │
 │  │  └─────────────────────┬──────────────────────┘  │     │
 │  │                        │                          │     │
 │  │  ┌─────────────────────▼──────────────────────┐  │     │
 │  │  │  Text Embedding (embed_tokens)             │  │     │
 │  │  │  ● frozen=True                             │  │     │
 │  │  │  input_ids → text embeddings               │  │     │
 │  │  │  visual tokens 插入 (IMAGE/VIDEO token位置) │  │     │
 │  │  └─────────────────────┬──────────────────────┘  │     │
 │  │                        │                          │     │
 │  │  ┌─────────────────────▼──────────────────────┐  │     │
 │  │  │  Transformer Layer  0-15  (frozen)         │  │     │
 │  │  │  + LoRA rank=64 on all layers              │  │     │
 │  │  │    target: q/k/v/o_proj + gate/up/down     │  │     │
 │  │  │                                            │  │     │
 │  │  │  ── Layer 10 ──► hidden_states[10] ────────┼──┼──► scale_0
 │  │  └─────────────────────┬──────────────────────┘  │     │
 │  │                        │                          │     │
 │  │  ┌─────────────────────▼──────────────────────┐  │     │
 │  │  │  Transformer Layer 16-27  (trainable)      │  │     │
 │  │  │  + LoRA rank=64                            │  │     │
 │  │  │                                            │  │     │
 │  │  │  ── Layer 18 ──► hidden_states[18] ────────┼──┼──► scale_1
 │  │  │  ── Layer 28 ──► hidden_states[28] ────────┼──┼──► scale_2
 │  │  └────────────────────────────────────────────┘  │     │
 │  └──────────────────────────────────────────────────┘     │
 │                                                           │
 │  ┌───────────────────────────────────────────────────┐    │
 │  │  MultiScaleAdapter                                │    │
 │  │                                                    │    │
 │  │  scale_0 ──► LN + Linear(3584→2048) ──► proj_0   │    │
 │  │  scale_1 ──► LN + Linear(3584→2048) ──► proj_1   │    │
 │  │  scale_2 ──► LN + Linear(3584→2048) ──► proj_2   │    │
 │  │                                                    │    │
 │  │  gate_input = cat(proj_0.mean, proj_1.mean,       │    │
 │  │                    proj_2.mean)  [B, 6144]         │    │
 │  │       ↓                                            │    │
 │  │  Linear(6144→3) + Softmax → weights [B, 3]       │    │
 │  │       ↓                                            │    │
 │  │  output = Σ(proj_i × weight_i)  [B, N, 2048]     │    │
 │  └───────────────────────────────────────────────────┘    │
 │                                                           │
 │  Output:                                                  │
 │  ├── last_hidden_state: [B, N, 2048]  (fused features)   │
 │  ├── vision_mask: [B, N]                                  │
 │  └── text_mask: [B, N]                                    │
 └───────────────────────────────────────────────────────────┘
```

### 1.2 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | Qwen2-VL-7B-Instruct | 3584d hidden, 28 layers |
| 精度 | bfloat16 | |
| Attention | flash_attention_2 | 硬编码, 缺 fallback |
| LoRA rank | 64 | alpha=128, dropout=0.05 |
| LoRA targets | q/k/v/o/gate/up/down_proj | 所有 28 层 |
| Vision Tower | frozen | 不参与训练 |
| Text layers 0-15 | frozen | embed_tokens 也 frozen |
| Text layers 16-27 | trainable | 通过 LoRA 微调 |
| Multi-scale 抽取 | [10, 18, 28] | 早期(空间) + 中期 + 晚期(语义) |
| 输出维度 | 2048 | 从 3584 投影 |

### 1.3 MultiScaleAdapter 数据流细节

```
Layer 10 output: [B, N, 3584] ── early features (spatial detail)
Layer 18 output: [B, N, 3584] ── mid features (intermediate)
Layer 28 output: [B, N, 3584] ── late features (semantic)
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
 LN+Linear     LN+Linear     LN+Linear
 3584→2048     3584→2048      3584→2048
      │             │             │
      ▼             ▼             ▼
   proj_0        proj_1        proj_2
 [B,N,2048]   [B,N,2048]    [B,N,2048]
      │             │             │
      │  ┌──────────┼─────────────┘
      │  │   mean pool (spatial dim)
      ▼  ▼          ▼
 [B,2048] [B,2048] [B,2048]
      │       │       │
      └───────┼───────┘
              ▼
     cat → [B, 6144]
              │
    Linear(6144→3) + Softmax
              │
              ▼
     weights [B, 3]  (每个 sample 不同的 scale 权重)
              │
              ▼
     stacked [B, N, 2048, 3] × weights[:, None, None, :]
              │
          sum(dim=-1)
              │
              ▼
       [B, N, 2048]  ← backbone 最终输出
```

---

## 2. Hierarchical Attention Grounder (`attention_grounder.py`)

### 2.1 完整结构图

```
          backbone_hidden [B, N, 2048]
                   │
 ┌─────────────────▼─────────────────────────────────────────────────┐
 │                                                                    │
 │  Learned Latent Queries  [1, 96, 2048]  (nn.Parameter)           │
 │                                                                    │
 │  Slot Layout (初始):                                              │
 │  ┌──────┬──────────────────┬─────┬─────┬─────┬──────────────────┐ │
 │  │global│  48 object slots │phase│unc  │aff  │  44 auxiliary    │ │
 │  │ (1)  │     (48)         │ (1) │ (1) │ (1) │     (44)        │ │
 │  └──────┴──────────────────┴─────┴─────┴─────┴──────────────────┘ │
 │  idx: 0    1..48             49    50    51     52..95            │
 │  ────────────────────────────── 96 latents ────────────────────── │
 │                                                                    │
 │  ╔═══════════════════════════════════════════╗                    │
 │  ║  GrounderBlock × 4  (layers 0-3)         ║                    │
 │  ║  ┌─────────────────────────────────────┐  ║                    │
 │  ║  │ CrossAttentionLayer                 │  ║                    │
 │  ║  │   Q: latents [B,96,2048]            │  ║                    │
 │  ║  │   K,V: backbone_hidden [B,N,2048]   │  ║                    │
 │  ║  │   16 heads, head_dim=128            │  ║                    │
 │  ║  │   ┌──────────────────────┐          │  ║                    │
 │  ║  │   │ LN(Q) → q_proj      │          │  ║                    │
 │  ║  │   │ LN(KV) → k_proj     │          │  ║                    │
 │  ║  │   │         → v_proj     │          │  ║                    │
 │  ║  │   │ SDPA (Flash/MemEff)  │          │  ║                    │
 │  ║  │   │ out_proj + residual  │          │  ║                    │
 │  ║  │   │ + FFN (LN→4×→GELU→1×)│         │  ║                    │
 │  ║  │   └──────────────────────┘          │  ║                    │
 │  ║  ├─────────────────────────────────────┤  ║                    │
 │  ║  │ SelfAttentionLayer                  │  ║                    │
 │  ║  │   Q,K,V: latents [B,96,2048]       │  ║                    │
 │  ║  │   16 heads, head_dim=128            │  ║                    │
 │  ║  │   ┌──────────────────────┐          │  ║                    │
 │  ║  │   │ LN → QKV_proj (3×D) │          │  ║                    │
 │  ║  │   │ SDPA (Flash/MemEff)  │          │  ║                    │
 │  ║  │   │ out_proj + residual  │          │  ║                    │
 │  ║  │   │ + FFN (LN→4×→GELU→1×)│         │  ║                    │
 │  ║  │   └──────────────────────┘          │  ║                    │
 │  ║  └─────────────────────────────────────┘  ║                    │
 │  ╚═══════════════════╤═══════════════════════╝                    │
 │                      │                                             │
 │                      │  latents [B, 96, 2048]                     │
 │                      │                                             │
 │  ╔═══════════════════▼═══════════════════════╗                    │
 │  ║  SlotCompression (between layer 3→4)      ║                    │
 │  ║                                            ║                    │
 │  ║  Extract: latents[:, 1:49, :]              ║                    │
 │  ║           → raw_object_slots [B, 48, 2048] ║                    │
 │  ║                                            ║                    │
 │  ║  route_queries [1, 24, 2048] (nn.Param)   ║                    │
 │  ║       │                                    ║                    │
 │  ║       ▼                                    ║                    │
 │  ║  CrossAttentionLayer                       ║                    │
 │  ║    Q: route_queries [B, 24, 2048]          ║                    │
 │  ║    K,V: raw_slots   [B, 48, 2048]          ║                    │
 │  ║       │                                    ║                    │
 │  ║       ▼                                    ║                    │
 │  ║  SelfAttentionLayer                        ║                    │
 │  ║    → compressed [B, 24, 2048]              ║                    │
 │  ║                                            ║                    │
 │  ║  Reassemble latents:                       ║                    │
 │  ║  ┌──────┬─────────────────┬─────┬────┬────┬──────────────┐    ║
 │  ║  │global│ 24 compressed   │phase│unc│aff │ 44 auxiliary  │    ║
 │  ║  │ (1)  │    (24)         │ (1) │(1)│(1) │    (44)       │    ║
 │  ║  └──────┴─────────────────┴─────┴────┴────┴──────────────┘    ║
 │  ║  ─────────────────────── 72 latents ──────────────────────    ║
 │  ╚═══════════════════╤═══════════════════════╝                    │
 │                      │                                             │
 │  ╔═══════════════════▼═══════════════════════╗                    │
 │  ║  GrounderBlock × 4  (layers 4-7)         ║                    │
 │  ║  (same structure as layers 0-3)           ║                    │
 │  ║  CrossAttn + SelfAttn on 72 latents      ║                    │
 │  ╚═══════════════════╤═══════════════════════╝                    │
 │                      │                                             │
 │                      ▼                                             │
 │              LayerNorm(2048)                                      │
 │                      │                                             │
 │  ┌───────────────────▼───────────────────────────────┐            │
 │  │              GrounderOutput                        │            │
 │  │                                                    │            │
 │  │  Post-compression layout:                          │            │
 │  │  [global(1) | compressed(24) | phase(1) | unc(1) | aff(1) | aux(44)]
 │  │   idx: 0       1..24           25         26       27       28..71
 │  │                                                    │            │
 │  │  ● global_token          = latents[:, 0]    [B, 2048]         │
 │  │  ● compressed_obj_slots  = latents[:, 1:25] [B, 24, 2048]    │
 │  │  ● phase_token           = latents[:, 25]   [B, 2048]         │
 │  │  ● uncertainty_token     = latents[:, 26]   [B, 2048]         │
 │  │  ● affordance_token      = latents[:, 27]   [B, 2048]         │
 │  │  ● object_slots (raw)    = saved before compression [B,48,2048]│
 │  └───────────────────────────────────────────────────┘            │
 └───────────────────────────────────────────────────────────────────┘
```

### 2.2 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_size | 2048 | |
| num_latents | 96 | 初始 latent 数 |
| num_object_slots | 48 | 压缩前 |
| compressed_slots | 24 | 压缩后 |
| num_layers | 8 | 4 pre-compression + 4 post |
| num_heads | 16 | head_dim = 128 |
| mlp_ratio | 4.0 | FFN hidden = 8192 |
| compression_layer | 4 | 在第 4 层后压缩 |

### 2.3 GrounderBlock 内部连接

```
输入: latents [B, L, 2048],  context [B, N, 2048]

   ═══ CrossAttentionLayer ═══
   │
   │  q = q_proj(LN(latents))              [B, L, 16, 128] → [B, 16, L, 128]
   │  k = k_proj(LN(context))              [B, N, 16, 128] → [B, 16, N, 128]
   │  v = v_proj(LN(context))              [B, N, 16, 128] → [B, 16, N, 128]
   │  attn = SDPA(q, k, v)                 [B, 16, L, 128] → [B, L, 2048]
   │  latents = latents + dropout(out_proj(attn))              ← residual
   │  latents = latents + FFN(latents)                          ← FFN residual
   │     FFN: LN → Linear(2048→8192) → GELU → dropout → Linear(8192→2048)
   │
   ═══ SelfAttentionLayer ═══
   │
   │  qkv = QKV_proj(LN(latents))          [B, L, 3, 16, 128]
   │  q, k, v = unbind                     each [B, 16, L, 128]
   │  attn = SDPA(q, k, v)                 [B, 16, L, 128] → [B, L, 2048]
   │  latents = latents + dropout(out_proj(attn))              ← residual
   │  latents = latents + FFN(latents)                          ← FFN residual
   │
输出: latents [B, L, 2048]
```

---

## 3. Tri-Rate Mamba Core (`mamba_core.py`)

### 3.1 输入组装

```
  ┌────────────────────────────────────────────────────────────────────┐
  │  _compose_input_sequence                                          │
  │                                                                    │
  │  9 个 single tokens  [B, 2048] each:                              │
  │  ┌──────────┬──────┬──────┬──────┬────────┬──────────┬─────┐      │
  │  │global_tok│phase │unc  │aff   │proprio │prev_act  │stale│      │
  │  │    (1)   │ (1)  │ (1) │ (1)  │  (1)   │   (1)    │ (1) │      │
  │  ├──────────┴──────┴──────┴──────┴────────┴──────────┴─────┤      │
  │  │embodiment│action_history│                                │      │
  │  │   (1)    │     (1)      │                                │      │
  │  └──────────┴──────────────┘                                │      │
  │       stack → [B, 9, 2048]                                  │      │
  │                    +                                        │      │
  │  compressed_object_slots [B, 24, 2048]                      │      │
  │                    ↓                                        │      │
  │  cat → input_seq [B, 33, 2048]                              │      │
  └────────────────────────────────────────────────────────────────────┘
```

### 3.2 Token 来源追踪

| Token | 来源 | 维度变换 |
|-------|------|---------|
| `global_token` | Grounder output | 直接 2048 |
| `phase_token` | Grounder output | 直接 2048 |
| `uncertainty_token` | Grounder output | 直接 2048 |
| `affordance_token` | Grounder output | 直接 2048 |
| `object_slots` | Grounder compressed | [B, 24, 2048] |
| `proprio_token` | `proprio_proj(proprio)` | Linear(14→2048) |
| `prev_action_token` | `prev_action_proj(prev_act)` | Linear(14→2048) |
| `stale_token` | `StaleTimeEncoding(steps)` | sinusoidal → MLP → 2048 |
| `embodiment_token` | `Embedding(16, 2048)` | lookup → 2048 |
| `action_history_token` | `ActionHistoryEncoder` | 4-layer Mamba → 2048 |

### 3.3 三流并行处理

```
                            input_seq [B, 33, 2048]
                                     │
                 ┌───────────────────┬┴───────────────────┐
                 │                   │                     │
          fast_input_norm     medium_input_norm     slow_input_norm
          LN(2048)            LN(2048)              LN(2048)
                 │                   │                     │
                 ▼                   ▼                     ▼
  ╔══════════════════════╗ ╔═════════════════════╗ ╔══════════════════════╗
  ║   FastMamba          ║ ║  MediumMamba        ║ ║   SlowMamba          ║
  ║   20 × MambaBlock    ║ ║  6 × MambaBlock    ║ ║   10 × MambaBlock   ║
  ║                      ║ ║                     ║ ║                      ║
  ║   d_model  = 2048    ║ ║  d_model  = 2048   ║ ║   d_model  = 2048   ║
  ║   d_state  = 128     ║ ║  d_state  = 128    ║ ║   d_state  = 256    ║
  ║   d_conv   = 4       ║ ║  d_conv   = 4      ║ ║   d_conv   = 4      ║
  ║   expand   = 2       ║ ║  expand   = 2      ║ ║   expand   = 2      ║
  ║   d_inner  = 4096    ║ ║  d_inner  = 4096   ║ ║   d_inner  = 4096   ║
  ║                      ║ ║                     ║ ║                      ║
  ║   更新: EVERY step   ║ ║  更新: every 2 stp ║ ║   更新: refresh only║
  ║   频率: ~50 Hz       ║ ║  频率: ~25 Hz      ║ ║   频率: ~12.5 Hz   ║
  ╚══════════╤═══════════╝ ╚══════════╤══════════╝ ╚══════════╤═══════════╝
             │                        │                       │
        out [B,33,2048]          out [B,33,2048]         out [B,33,2048]
             │                        │                       │
        mean(dim=1)              mean(dim=1)              mean(dim=1)
             │                        │                       │
             ▼                        ▼                       ▼
      fast_token [B,2048]    medium_token [B,2048]    slow_token [B,2048]
             │                        │                       │
             └────────────┬───────────┘                       │
                          │    ┌───────────────────────────────┘
                          │    │
                          ▼    ▼
              ╔════════════════════════════════╗
              ║   CrossAttentionFusion         ║
              ║   (详见 3.5)                    ║
              ╚══════════════╤═════════════════╝
                             │
                             ▼
                      fused_state [B, 2048]
```

### 3.4 MambaBlock 内部连接 (单层)

```
  ════════════════════════════════════════════════════════
   MambaBlock (pre-norm residual pattern)
  ════════════════════════════════════════════════════════

   x [B, L, 2048]
   │
   ├──────────────────────────────────── (residual)
   │                                         │
   ▼                                         │
  LN(2048)                                   │
   │                                         │
   ▼                                         │
  in_proj: Linear(2048 → 8192, no bias)      │
   │                                         │
   split                                     │
   ├──► x_main [B, L, 4096]                 │
   │        │                                │
   │        ▼                                │
   │   Conv1d(4096, 4096, k=4,              │
   │         groups=4096, padding=3)         │
   │        │                                │
   │        ▼                                │
   │     SiLU                                │
   │        │                                │
   │        ▼                                │
   │   x_proj: Linear(4096 → dt_rank+256)   │
   │        │                                │
   │        split                            │
   │        ├── dt  [B,L,dt_rank=128]        │
   │        ├── B_ssm [B,L,d_state=128]      │
   │        └── C_ssm [B,L,d_state=128]      │
   │                                         │
   │   dt_proj: Linear(128→4096) + softplus  │
   │        │                                │
   │        ▼                                │
   │   ╔═══════════════════════════╗         │
   │   ║  SSM Scan                 ║         │
   │   ║  A = -exp(A_log)          ║         │
   │   ║  dA = exp(A * dt)         ║         │
   │   ║  dBx = dt * B * x_main   ║         │
   │   ║  h[t] = dA*h[t-1] + dBx  ║         │
   │   ║  y[t] = C * h[t]         ║         │
   │   ╚══════════╤════════════════╝         │
   │              │                           │
   │              ▼                           │
   │   y = y + x_main * D                    │
   │              │                           │
   └──► z [B, L, 4096]                       │
        │    │                                │
        │    ▼                                │
        │  SiLU(z)                            │
        │    │                                │
        ▼    ▼                                │
      y = y * SiLU(z)      (gating)          │
        │                                     │
        ▼                                     │
   out_proj: Linear(4096→2048, no bias)       │
        │                                     │
        ▼                                     ▼
      output = out_proj(y) + x          (+ residual)
        │
        ▼
   [B, L, 2048]
```

### 3.5 CrossAttentionFusion 内部连接

```
  fast_token [B, 2048]   medium_token [B, 2048]   slow_token [B, 2048]
       │                        │                        │
       └────────────────────────┼────────────────────────┘
                                │
                       stack → kv [B, 3, 2048]
                                │
                                + stale_proj(stale_token).unsqueeze(1)
                                │               (stale 时间信息注入)
                                ▼
                          kv [B, 3, 2048]

  fusion_query [1, 1, 2048]  (nn.Parameter, learned)
       │
       expand → q [B, 1, 2048]

  ╔════ Fusion Layer × 2 ════╗
  ║                           ║
  ║  q_norm = LN(q)           ║
  ║  kv_norm = LN(kv)         ║
  ║                           ║
  ║  cross_attn:              ║
  ║    MHA(q_norm, kv_norm,   ║
  ║        kv_norm)           ║
  ║    8 heads, head_dim=256  ║
  ║                           ║
  ║  q = q + attn_out         ║    ← residual
  ║                           ║
  ║  ffn:                     ║
  ║    LN → Linear(2048→8192) ║
  ║    → GELU                 ║
  ║    → Linear(8192→2048)    ║
  ║                           ║
  ║  q = q + ffn(q)           ║    ← residual
  ║                           ║
  ╚═══════════════════════════╝
       │
       ▼
  output_norm = LN(q.squeeze(1))
       │
       ▼
  fused_state [B, 2048]
```

### 3.6 ActionHistoryEncoder

```
  action_history [B, K=8, A=14]
       │
       ▼
  action_proj: Linear(14→2048)
       │
       ▼
  [B, 8, 2048]
       │
       ▼
  _MambaStack (4 layers, d_state=64, d_conv=4, expand=2)
       │
       ▼
  out [B, 8, 2048]
       │
  out[:, -1, :]   (取最后一个 token 作为 summary)
       │
       ▼
  action_history_token [B, 2048]
```

### 3.7 StaleTimeEncoding

```
  steps_since_refresh [B]  (integer, clamp to [0, 256])
       │
       ▼
  sinusoidal encoding:
    half = 1024
    freqs = exp(-log(10000) * arange(1024) / 1024)
    args = steps[:, None] * freqs[None, :]
    emb = cat(sin(args), cos(args))  → [B, 2048]
       │
       ▼
  MLP: Linear(2048→2048) → SiLU → Linear(2048→2048)
       │
       ▼
  stale_token [B, 2048]
```

### 3.8 Tri-Rate 更新时序

```
  时间 t:    0   1   2   3   4   5   6   7   8   9  10  11  12 ...
           ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
  Fast     │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │ ■ │  每步
           ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
  Medium   │ ■ │   │ ■ │   │ ■ │   │ ■ │   │ ■ │   │ ■ │   │ ■ │  stride=2
           ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
  Slow     │ ■ │   │   │   │   │   │ ■ │   │   │   │   │   │ ■ │  stride=6
           └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
            ▲                       ▲                       ▲
         refresh_0              refresh_1              refresh_2
    (semantic_refresh_stride=6 → backbone+grounder 重新运行)

  不更新时: 重用上一次的 medium_token / slow_token
           SSM states 保持不变 (不 step)
```

---

## 4. Condition Prefix Builder (`hybrid_vla_v2.py:224-251`)

### 4.1 组装与投影

```
  ┌─── From Grounder ─────────────────────────────────────┐
  │  global_token.unsqueeze(1)           [B, 1, 2048]     │
  │  compressed_object_slots             [B, 24, 2048]    │
  │  phase_token.unsqueeze(1)            [B, 1, 2048]     │
  │  uncertainty_token.unsqueeze(1)      [B, 1, 2048]     │
  │  affordance_token.unsqueeze(1)       [B, 1, 2048]     │
  └───────────────────────────────────────────────────────┘
  ┌─── From Temporal Core ────────────────────────────────┐
  │  fused_state.unsqueeze(1)            [B, 1, 2048]     │
  │  fast_token.unsqueeze(1)             [B, 1, 2048]     │
  │  medium_token.unsqueeze(1)           [B, 1, 2048]     │
  │  slow_token.unsqueeze(1)             [B, 1, 2048]     │
  └───────────────────────────────────────────────────────┘
                           │
                   cat(dim=1)
                           │
                           ▼
               cond [B, 32, 2048]
          (1+24+1+1+1 + 1+1+1+1 = 32 tokens)
                           │
                           ▼
                ┌────────────────────┐
                │   cond_builder     │
                │   LN(2048)         │
                │   Linear(2048→2048)│
                │   GELU             │
                │   Linear(2048→2048)│
                └─────────┬──────────┘
                          │
                          ▼
               cond [B, 32, 2048]
                          │
                          ▼
             core_to_expert: Linear(2048→1536)
                          │
                          ▼
               cond [B, 32, 1536]  ← 输入 FlowActionExpert
```

### 4.2 Stage 门控

```
  Stage A:  Expert 冻结, cond_prefix 不构建
            仅训练: backbone LoRA + grounder + tri-rate core + discrete heads

  Stage B:  Expert 参与, cond_prefix.detach()
            stop_gradient_cond_prefix=True → 梯度不回传到 backbone/grounder/core

  Stage C:  全部微调 (RTC/FASTER)
            梯度端到端流动
```

---

## 5. Flow Action Expert (`flow_action_expert.py`)

### 5.1 完整结构图

```
  ┌─── Inputs ──────────────────────────────────────────────────────────┐
  │                                                                      │
  │  noisy_actions [B, H=24, A=14]     flow_t [B]   cond_prefix [B,32,1536]│
  │       │                              │                │              │
  │       ▼                              ▼                ▼              │
  │  action_proj                   timestep_emb      cond_proj          │
  │  Linear(14→1536)               Sinusoidal+MLP    Linear(2048→1536)  │
  │       │                        → [B, 1536]       or Identity        │
  │       + action_pos_emb(24)           │                │              │
  │       │   Learned [1,24,1536]        │                │              │
  │       │                              │                │              │
  │       + t_emb.unsqueeze(1)     ┌─────┤                │              │
  │       │                        │     │                │              │
  │       ▼                        │     ▼                │              │
  │  action_tokens [B,24,1536]     │  t_cond_mlp          │              │
  │                                │  Linear→SiLU→Linear   │              │
  │  proprio_proj(proprio)         │     │                │              │
  │  → [B, 1, 1536]               │     ▼                │              │
  │                                │  t_cond [B, 1536]    │              │
  │  embodiment_proj(emb)          │  (for AdaRMSNorm)    │              │
  │  → [B, 1, 1536]               │                      │              │
  │       │                        │                      │              │
  │       ▼                        │                      │              │
  │  x = cat([proprio, emb,        │                      │              │
  │           action_tokens])      │                      │              │
  │     [B, 26, 1536]              │                      │              │
  │                                │                      │              │
  └────────────────────────────────┼──────────────────────┼──────────────┘
                                   │                      │
         ┌─────────────────────────┘                      │
         │                                                │
         ▼                                                ▼
  ╔══════════════════════════════════════════════════════════════════╗
  ║                   18 Layers (M-M-A × 6)                         ║
  ║                                                                  ║
  ║  Layer  0: ExpertMambaBlock   (x, t_cond)                       ║
  ║  Layer  1: ExpertMambaBlock   (x, t_cond)                       ║
  ║  Layer  2: ExpertAttentionBlock(x, cond_prefix, t_cond)         ║
  ║  Layer  3: ExpertMambaBlock   (x, t_cond)                       ║
  ║  Layer  4: ExpertMambaBlock   (x, t_cond)                       ║
  ║  Layer  5: ExpertAttentionBlock(x, cond_prefix, t_cond)         ║
  ║  Layer  6: ExpertMambaBlock   (x, t_cond)                       ║
  ║  Layer  7: ExpertMambaBlock   (x, t_cond)                       ║
  ║  Layer  8: ExpertAttentionBlock(x, cond_prefix, t_cond)         ║
  ║  Layer  9: ExpertMambaBlock   (x, t_cond)                       ║
  ║  Layer 10: ExpertMambaBlock   (x, t_cond)                       ║
  ║  Layer 11: ExpertAttentionBlock(x, cond_prefix, t_cond)         ║
  ║  Layer 12: ExpertMambaBlock   (x, t_cond)                       ║
  ║  Layer 13: ExpertMambaBlock   (x, t_cond)                       ║
  ║  Layer 14: ExpertAttentionBlock(x, cond_prefix, t_cond)         ║
  ║  Layer 15: ExpertMambaBlock   (x, t_cond)                       ║
  ║  Layer 16: ExpertMambaBlock   (x, t_cond)                       ║
  ║  Layer 17: ExpertAttentionBlock(x, cond_prefix, t_cond)         ║
  ╚══════════════════════════╤═══════════════════════════════════════╝
                             │
                     x [B, 26, 1536]
                             │
                    x[:, 2:, :]    (去掉 proprio + embodiment prefix)
                             │
                             ▼
                     action_out [B, 24, 1536]
                             │
                     LN(1536) → Linear(1536→14)
                             │
                             ▼
                     velocity [B, 24, 14]
```

### 5.2 ExpertMambaBlock 内部连接

```
  x [B, L, 1536],  t_cond [B, L, 1536]
  │
  ├────────────────────────────────── (residual)
  │                                       │
  ▼                                       │
  AdaRMSNorm(x, t_cond):                 │
    RMS = sqrt(mean(x^2) + eps)           │
    x_norm = x / RMS                      │
    scale, shift, gate = cond_proj(t_cond).chunk(3)   (Linear(1536→4608))
    out = sigmoid(gate) * (x_norm * (1+scale) + shift)
  │                                       │
  ▼                                       │
  in_proj: Linear(1536→6144, no bias)     │
  split → x_main [B,L,3072] + z [B,L,3072]
  │                                       │
  x_main → Conv1d(3072, k=4, groups=3072) │
         → SiLU                           │
         → x_proj → (dt, B_ssm, C_ssm)   │
         → SSM scan                       │
         → y + x_main * D                 │
         → y * SiLU(z)                    │
         → out_proj: Linear(3072→1536)    │
  │                                       │
  └───────────────────────────────────────┘
                  + residual
                  │
                  ▼
          [B, L, 1536]
```

### 5.3 ExpertAttentionBlock 内部连接

```
  x [B, L=26, D=1536],  cond [B, C=32, D=1536],  t_cond [B, L, D=1536]
  │
  ═══ Cross-Attention ═══
  │
  │  Q = cross_q(AdaRMSNorm(x, t_cond))     [B, L, 24, 64] → [B, 24, L, 64]
  │  K = cross_k(LN(cond))                  [B, C, 24, 64] → [B, 24, C, 64]
  │  V = cross_v(LN(cond))                  [B, C, 24, 64] → [B, 24, C, 64]
  │
  │  SDPA(Q, K, V)                           [B, 24, L, 64] → [B, L, 1536]
  │  x = x + dropout(cross_out(attn))                        ← residual
  │
  ═══ Self-Attention ═══
  │
  │  h = AdaRMSNorm(x, t_cond)
  │  QKV = self_qkv(h) → split             each [B, 24, L, 64]
  │  SDPA(Q, K, V)                          [B, 24, L, 64] → [B, L, 1536]
  │  x = x + dropout(self_out(attn))                         ← residual
  │
  ═══ FFN ═══
  │
  │  x = x + FFN(AdaRMSNorm(x, t_cond))
  │       FFN: Linear(1536→6144) → GELU → dropout → Linear(6144→1536)
  │
  └──► [B, L, 1536]
```

### 5.4 ODE 推理 (Midpoint Solver)

```
  x_0 ~ N(0, I)  [B, 24, 14]     (纯噪声初始化)
       │
  for i in range(num_steps=8):
       │
       t_i = i / 8
       t_mid = (i + 0.5) / 8
       dt = 1/8
       │
       v_1 = Expert.forward(x, t_i, cond)        ← 第一次前向
       x_mid = x + 0.5 * dt * v_1                 ← 半步 Euler
       v_2 = Expert.forward(x_mid, t_mid, cond)   ← 第二次前向 (midpoint)
       x = x + dt * v_2                           ← 用 midpoint 速度全步更新
       │
  x_final [B, 24, 14]  ← denoised action chunk
```

---

## 6. Discrete Heads (`discrete_heads.py`)

### 6.1 FASTDiscreteHead

```
  fused_state [B, 2048]
       │
       ▼
  encoder: LN(2048) → Linear(2048→768) → GELU
       │
       ▼
  [B, 768]
       │
       ▼
  step_proj: Linear(768 → 24 × 192 = 4608)
       │
       ▼
  reshape → [B×24, 192]
       │
       ▼
  dim_head: LN(192) → Linear(192 → 14 × 512 = 7168)
       │
       ▼
  reshape → [B, 24, 14, 512]  ← logits
                                  H=24 steps, A=14 dims, V=512 bins

  训练: CE loss with label_smoothing=0.1
  推理: softmax → bin_centers 加权和 → continuous actions [B, 24, 14]
```

### 6.2 PhaseHead

```
  phase_token [B, 2048]
       │
       ▼
  LN(2048) → Linear(2048→1024) → GELU → Linear(1024→16)
       │
       ▼
  logits [B, 16]  ← 16 phase classes
```

### 6.3 AffordanceHead

```
  affordance_token [B, 2048]
       │
       ▼
  LN(2048) → Linear(2048→1024) → GELU → Linear(1024→8)
       │
       ▼
  logits [B, 8]  ← 8 affordance types
```

---

## 7. 端到端维度追踪表

从输入到输出, 每一步的 tensor shape 变化:

| 步骤 | 操作 | 输入 shape | 输出 shape |
|------|------|-----------|-----------|
| 1 | Qwen2-VL forward | input_ids [B,N], pixels [...] | hidden_states [28+1 × B,N,3584] |
| 2 | Multi-scale extract | [L10, L18, L28] × [B,N,3584] | 3 × [B,N,3584] |
| 3 | MultiScaleAdapter proj | 3 × [B,N,3584] | 3 × [B,N,2048] |
| 4 | Gated fusion | 3 × [B,N,2048] | [B,N,2048] |
| 5 | Grounder init | backbone [B,N,2048] | latents [B,96,2048] |
| 6 | Grounder L0-3 | latents × context | latents [B,96,2048] |
| 7 | SlotCompression | raw_slots [B,48,2048] | compressed [B,24,2048] |
| 8 | Latent reassembly | 96 latents | 72 latents [B,72,2048] |
| 9 | Grounder L4-7 | latents [B,72,2048] × context | latents [B,72,2048] |
| 10 | Grounder output | latents [B,72,2048] | 6 fields (2048d each) |
| 11 | Token composition | 9 singles + 24 slots | input_seq [B,33,2048] |
| 12 | Fast Mamba (20L) | [B,33,2048] | fast_token [B,2048] |
| 13 | Medium Mamba (6L) | [B,33,2048] | medium_token [B,2048] |
| 14 | Slow Mamba (10L) | [B,33,2048] | slow_token [B,2048] |
| 15 | CrossAttentionFusion | 3 tokens + stale | fused_state [B,2048] |
| 16 | Cond prefix build | 9 tokens (32 total) | cond [B,32,2048] |
| 17 | core_to_expert | [B,32,2048] | [B,32,1536] |
| 18 | Action proj | noisy [B,24,14] | tokens [B,24,1536] |
| 19 | Expert input | [proprio,emb,actions] | x [B,26,1536] |
| 20 | Expert 18L (M-M-A×6) | x + cond + t_cond | x [B,26,1536] |
| 21 | Expert output | x[:, 2:] → LN → proj | velocity [B,24,14] |
| 22 | FAST head | fused_state [B,2048] | logits [B,24,14,512] |
| 23 | Phase head | phase_token [B,2048] | logits [B,16] |
| 24 | Affordance head | aff_token [B,2048] | logits [B,8] |

---

## 8. 参数量估算

### 8.1 各模块参数量

| 模块 | 参数量 | 训练状态 |
|------|--------|---------|
| **Qwen2-VL-7B** | ~7.6B | Frozen (LoRA ~80M trainable) |
| - Vision Tower | ~600M | Frozen |
| - Text Layers 0-15 | ~3.5B | Frozen (LoRA active) |
| - Text Layers 16-27 | ~3.0B | LoRA + unfrozen |
| - embed_tokens | ~500M | Frozen |
| **MultiScaleAdapter** | ~22M | Trainable |
| **Grounder (8L)** | ~340M | Trainable |
| - 8 × GrounderBlock | ~320M | |
| - SlotCompression | ~20M | |
| **Tri-Rate Mamba Core** | ~1.2B | Trainable |
| - FastMamba (20L) | ~670M | |
| - MediumMamba (6L) | ~200M | |
| - SlowMamba (10L) | ~340M | |
| - CrossAttentionFusion | ~67M | |
| - StaleTimeEncoding | ~8M | |
| - ActionHistoryEncoder | ~135M | |
| **Projections** | ~20M | Trainable |
| - proprio/prev_action/emb | | |
| - core_to_expert/cond_builder | | |
| **Flow Action Expert (18L)** | ~570M | Stage A: Frozen |
| - 12 × ExpertMambaBlock | ~300M | |
| - 6 × ExpertAttentionBlock | ~250M | |
| - Embeddings + output | ~20M | |
| **Discrete Heads** | ~15M | Trainable |
| - FASTDiscreteHead | ~12M | |
| - PhaseHead | ~1.5M | |
| - AffordanceHead | ~1.5M | |
| **总计** | ~9.9B | ~2.3B trainable (Stage A) |

### 8.2 单层参数量明细

**GrounderBlock (1 layer @ 2048d)**:
- CrossAttn: Q/K/V/O proj = 4 × 2048² = 16.8M, FFN = 2 × 2048×8192 = 33.6M → **~50M**
- SelfAttn: QKV proj = 3 × 2048² = 12.6M, O proj = 4.2M, FFN = 33.6M → **~50M**
- Total per block: **~100M** (by treating LN + biases as negligible)

**MambaBlock (1 layer @ 2048d, d_state=128, expand=2)**:
- in_proj: 2048 × 8192 = 16.8M
- conv1d: 4096 × 4 = 16K (depthwise)
- x_proj: 4096 × 384 = 1.6M
- dt_proj: 128 × 4096 = 0.5M
- A_log + D: 4096 × 128 + 4096 = 0.5M
- out_proj: 4096 × 2048 = 8.4M
- Total per block: **~28M**

**ExpertMambaBlock (1 layer @ 1536d, d_state=96, expand=2)**:
- AdaRMSNorm cond_proj: 1536 × 4608 = 7.1M
- in_proj: 1536 × 6144 = 9.4M
- Other SSM params: ~6M
- out_proj: 3072 × 1536 = 4.7M
- Total per block: **~27M**

**ExpertAttentionBlock (1 layer @ 1536d, 24 heads)**:
- Cross-Attn: Q/K/V/O + AdaRMSNorm + LN = ~18M
- Self-Attn: QKV/O + AdaRMSNorm = ~16M
- FFN: 2 × 1536 × 6144 + AdaRMSNorm = ~26M
- Total per block: **~60M**

---

## 9. 训练阶段数据流对比

### 9.1 Stage A (backbone LoRA + grounder + core + heads)

```
  Image + Text
       │
       ▼
  [Backbone LoRA] ──► [Grounder] ──► [Tri-Rate Core] ──► [Discrete Heads]
       │                                    │
       │                                    ├──► loss_fast (CE, w=1.0)
       │                                    ├──► loss_phase (CE, w=0.5)
       │                                    ├──► loss_affordance (CE, w=0.3)
       │                                    └──► loss_consistency (contrastive, w=0.3)
       │
  Expert: FROZEN (不参与前向/反向)
```

### 9.2 Stage B (adds expert, cond detached)

```
  Image + Text
       │
       ▼
  [Backbone LoRA] ──► [Grounder] ──► [Tri-Rate Core] ──► [Discrete Heads]
       │                    │               │
       │                    │               │ (上面同 Stage A losses)
       │                    └───────┬───────┘
       │                            │
       │                     cond_prefix.detach()    ← 梯度截断!
       │                            │
       │                            ▼
       │                    [Flow Action Expert]
       │                            │
       │                            ├──► loss_fm (flow matching, w=1.0)
       │                            └──► loss_consistency (+ action agreement)
       │
  梯度流: Expert 梯度不回传到 backbone/grounder/core
```

### 9.3 Stage C (full fine-tune + RTC/FASTER)

```
  Image + Text
       │
       ▼
  [Backbone LoRA] ──► [Grounder] ──► [Tri-Rate Core] ──► [Discrete Heads]
       │                    │               │
       │                    └───────┬───────┘
       │                            │
       │                     cond_prefix (梯度流通!)
       │                            │
       │                            ▼
       │                    [Flow Action Expert]
       │                            │
       │                    所有 losses + RTC + FASTER
       │
  梯度流: 端到端, Expert → Core → Grounder → Backbone LoRA
```

---

## 10. Loss 全景

```
                          ┌─────────────────────────────────────────────────┐
                          │              loss_total = Σ all losses          │
                          └─────────────────────────────────────────────────┘
                                           │
          ┌───────────────┬───────────────┬┼───────────────┬───────────────┐
          │               │               │                │               │
          ▼               ▼               ▼                ▼               ▼
   ┌─────────────┐ ┌────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
   │ loss_fast   │ │ loss_phase │ │loss_affordance│ │loss_fm      │ │loss_consistency│
   │ CE + smooth │ │ ordinal CE │ │    CE        │ │flow matching│ │contrastive    │
   │ w=1.0       │ │ w=0.5      │ │ w=0.3        │ │ w=1.0       │ │ w=0.3         │
   │ Stage A+    │ │ Stage A+   │ │ Stage A+     │ │ Stage B+    │ │ Stage A+      │
   └─────────────┘ └────────────┘ └──────────────┘ └─────────────┘ └──────────────┘
         ↑               ↑              ↑                ↑              ↑
    FAST head       Phase head    Affordance head   Expert output   fast_tokens
    (fused_state)  (phase_token) (affordance_token) (velocity)     + slow_token
                                                                   + fused_states
```

---

## 11. 推理时序图

```
  ┌──── 时间线 (50 Hz control loop) ──────────────────────────────────────────┐
  │                                                                            │
  │  t=0 (semantic refresh)                                                   │
  │  ┌─────────────────────────────────────────────────────────────────────┐   │
  │  │ 1. semantic_step():                                                 │   │
  │  │    Backbone(image+text) → Grounder → GrounderOutput                 │   │
  │  │    (最重, ~50ms on H100)                                            │   │
  │  └─────────────────────────────────────────────────────────────────────┘   │
  │                                                                            │
  │  t=0..5 (每个 control step, 20ms budget)                                  │
  │  ┌─────────────────────────────────────────────────────────────────────┐   │
  │  │ 2. control_step():                                                  │   │
  │  │    proprio/prev_action/stale/history → Token composition            │   │
  │  │    → TriRateMambaCore (fast always, medium/slow conditional)        │   │
  │  │    → _build_cond_prefix                                             │   │
  │  │    → FlowActionExpert.sample(midpoint, 8 steps)                     │   │
  │  │    → denoised_action [B, 24, 14]                                    │   │
  │  │    (~15ms on H100)                                                   │   │
  │  └─────────────────────────────────────────────────────────────────────┘   │
  │                                                                            │
  │  t=6 (semantic refresh again)                                             │
  │  ┌─────────────────────────────────────────────────────────────────────┐   │
  │  │ 新 backbone+grounder forward                                        │   │
  │  │ slow_mamba 更新                                                     │   │
  │  │ steps_since_refresh 归零                                            │   │
  │  └─────────────────────────────────────────────────────────────────────┘   │
  │                                                                            │
  └────────────────────────────────────────────────────────────────────────────┘

  频率分配:
  ┌──────────────────┬────────────┬────────────────────┐
  │  Component        │  Frequency │  运算量             │
  ├──────────────────┼────────────┼────────────────────┤
  │  Backbone+Grounder│  12.5 Hz  │  最重 (7.6B forward)│
  │  Slow Mamba       │  12.5 Hz  │  中 (10L Mamba)    │
  │  Medium Mamba     │  25.0 Hz  │  轻 (6L Mamba)     │
  │  Fast Mamba       │  50.0 Hz  │  中 (20L Mamba)    │
  │  Expert (8-step)  │  50.0 Hz  │  重 (18L × 8)      │
  └──────────────────┴────────────┴────────────────────┘
```

---

## 12. 关键设计决策总结

| 设计 | 选择 | 理由 |
|------|------|------|
| Backbone | Qwen2-VL-7B | 多模态, 原生视觉理解 |
| Multi-scale | [L10, L18, L28] + gated fusion | FPN 思想: 早期空间 + 晚期语义 |
| LoRA | rank=64, all 28 layers | 比 v1 (rank=32, last 8) 更充分 |
| Grounder latents | 96 → 72 (after compression) | 丰富表征 + 层级压缩去冗余 |
| Slot compression | 48 → 24 | 减少 core 输入 token 数 (33 vs 57) |
| Tri-rate | Fast/Medium/Slow | 匹配不同时间尺度: 反应/规划/语义 |
| Mamba vs Transformer | Core 用 Mamba, Expert 用 hybrid | Core: 长序列状态 (SSM); Expert: 精确生成 (Attn) |
| Cross-Attn fusion | 替代 v1 的 scalar gate | 内容相关的 per-dimension 融合 |
| AdaRMSNorm | Expert 中替代标准 LN | flow timestep 调制, 类 DiT/Pi0.5 |
| ODE solver | Midpoint (2nd order) | 同等 NFE 下精度翻倍 |
| Action expert dim | 1536 (< core 2048) | 降低 expert 计算量, 通过 projection 桥接 |
| cond_tokens | 32 | 24 object + 5 grounder + 3 temporal = 32 |
| FAST head | 512 bins, factorized | 粗粒度快速推理, 与 expert 互补 |

---

*文档生成完毕。覆盖 backbone 全部 7 个模型文件的逐层连接分析。*
