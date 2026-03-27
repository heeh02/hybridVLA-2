# HybridVLA v2 Documentation Index

## Architecture

| Document | Description |
|----------|-------------|
| [hybridvla_v2_design.md](hybridvla_v2_design.md) | v2 architecture design (Qwen2-VL-7B + Tri-Rate Mamba + Flow Expert) |
| [backbone_v0_6.md](backbone_v0_6.md) | Backbone design notes (Qwen2-VL-7B multi-scale, LoRA, multi-camera) |

## Version Analysis (Design Iterations)

| Document | Version | Focus |
|----------|---------|-------|
| [analysis_v0_1.md](analysis_v0_1.md) | v0.1 | Initial architecture analysis |
| [analysis_v0_2.md](analysis_v0_2.md) | v0.2 | Official mamba_ssm integration |
| [analysis_v0_3.md](analysis_v0_3.md) | v0.3 | World model + stochastic state |
| [analysis_v0_4.md](analysis_v0_4.md) | v0.4 | Imagination engine fixes |
| [analysis_v0_5.md](analysis_v0_5.md) | v0.5 | Mamba state persistence fix |
| [analysis_v0_6.md](analysis_v0_6.md) | v0.6 | Buffer allocation optimization |
| [analysis_v0_6_1.md](analysis_v0_6_1.md) | v0.6.1 | Detailed architecture review |

## Recovery Logs (Bug Fixes)

| Document | Version | Fixes |
|----------|---------|-------|
| [recovery_v0_2.md](recovery_v0_2.md) | v0.2 | mamba_ssm integration issues |
| [recovery_v0_3.md](recovery_v0_3.md) | v0.3 | World model loss + state bugs |
| [recovery_v0_4.md](recovery_v0_4.md) | v0.4 | ImaginationMamba state persistence |
| [recovery_v0_5.md](recovery_v0_5.md) | v0.5 | Official Mamba2 state capture |
| [recovery_v0_6.md](recovery_v0_6.md) | v0.6 | Buffer pre-allocation |
| [recovery_v0_7.md](recovery_v0_7.md) | v0.7 | Mamba2 residual, Qwen2VL freeze, conv state |

## Correctness Audits

| Document | Version | Result |
|----------|---------|--------|
| [accuracy_v0_6.md](accuracy_v0_6.md) | v0.6 | Found 2 critical + 2 high + 2 medium bugs |
| [accuracy_v0_7.md](accuracy_v0_7.md) | v0.7 | All code bugs resolved; 0 remaining issues |
| [final_analysis.md](final_analysis.md) | v0.7.1 | Full architecture, training, precision, inference audit |
| [fixed_final_analysis.md](fixed_final_analysis.md) | v0.7.2 | Cross-ref with expert1 review. 5 code fixes applied. |
| [optimize_v0_9.md](optimize_v0_9.md) | v0.9 | Rescore-driven optimization: res_scale, chunk cache, remove double-LN. |
| [optimize_v0_9_1.md](optimize_v0_9_1.md) | v0.9.1 | 9 fixes: denoising formula, proprio_dim, grounder mask, res_scale decay, refresh counter, API cleanup. |
| [optimize_v0_9_2.md](optimize_v0_9_2.md) | v0.9.2 | 5 fixes: config key warning, label_smoothing/action_range configurable, docstring, teacher-forcing doc. |
| [optimize_v0_9_3.md](optimize_v0_9_3.md) | v0.9.3 | Infrastructure: data pipeline (schema, normalizer, HDF5 adapter, collate), .gitignore, requirements.txt, enhanced validation. |
| [analysis_v0_10.md](analysis_v0_10.md) | v0.10 | Full codebase structure audit: model layer PASS, 9 data layer issues found. |
| [optimize_v0_10.md](optimize_v0_10.md) | v0.10 | Data layer fixes: schema name mismatch, short-episode bug, inheritance, HDF5 validation, collate None, normalizer warnings, compute_stats script. |
| [analysis_v0_10_1.md](analysis_v0_10_1.md) | v0.10.1 | Claude x GPT cross-audit: chunk supervision bug (P0-3), stats coupling, 5 new findings. |
| [optimize_v0_10_1.md](optimize_v0_10_1.md) | v0.10.1 | Critical: action chunk supervision fix (T+H-1 read), stats path decoupling, batch device transfer, affordance labels, normalizer warnings. |
| [optimize_v0_10_2.md](optimize_v0_10_2.md) | v0.10.2 | 5 fixes: num_affordance_types configurable, _to_device out of loop, step_weights validation, smoke test doc, remove redundant .to(device). |
| [analysis_v0_10_3.md](analysis_v0_10_3.md) | v0.10.3 | Claude x GPT deep cross-audit: single-step supervision (P1-C), processor disconnected (P0-A), no vision (P0-B). Score reconciliation 6.8/10. |
| **[optimize_v0_10_3.md](optimize_v0_10_3.md)** | **v0.10.3** | **5 critical fixes: connect Processor, HDF5 image reading, multi-step supervision, unified training script (Stage A/B/C), eval loop.** |
