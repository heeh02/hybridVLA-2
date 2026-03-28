# Changelog

All notable changes to the HybridVLA v2 project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-29

### Added
- EMA/FSDP gap test suite (`test_ema_fsdp_gaps.py`)
- Checkpoint validation utilities
- v1.0 code review and architecture analysis documents

### Changed
- Improved MambaCore memory management and selective scan logic
- Enhanced consistency loss with proper gradient handling
- Refactored HybridVLA v2 forward pass and action expert integration
- Simplified `train_stage_a.py`, removed redundant boilerplate
- Improved EMA state dict handling for FSDP compatibility
- Updated training config with new parameters

### Moved
- World model code (~1,200 lines) relocated to `vla_hybrid_v2/experimental/world_model/`
  (backward-compatible re-exports preserved)

### Fixed
- Selective scan edge cases
- Libero inference policy imports
- Test imports for control step and losses

## [0.10.9] - 2026-03-28

### Added
- Inference policy for LIBERO evaluation
- Documentation reorganization

### Fixed
- FSDP gradient sync issues
- EMA/checkpoint loading improvements

## [0.10.7] - 2026-03-27

### Added
- LIBERO benchmark integration
- Full test suite (forward train, expert, losses, normalizer, checkpoint, inference)
- Image augmentation pipeline

## [0.10.5] - 2026-03-26

### Changed
- Config parameter fixes and improvements
- Collate function improvements
- HDF5 adapter updates
- Unified training script enhancements
- Smoke test updates

## [0.10.3] - 2026-03-25

### Added
- Qwen2-VL processor connection
- HDF5 image reading support
- Multi-step supervision
- Unified training script (`train_unified.py`)

## [0.1.0] - 2026-03-23

### Added
- Initial project structure
- Qwen2-VL backbone with LoRA
- Flow matching action expert
- Mamba-based action chunking
- Discrete action heads
- Three-stage training pipeline design
- README (English and Chinese)

[1.0.0]: https://github.com/heeh02/hybridVLA-2/compare/v0.10.9...v1.0.0
[0.10.9]: https://github.com/heeh02/hybridVLA-2/compare/v0.10.7...v0.10.9
[0.10.7]: https://github.com/heeh02/hybridVLA-2/compare/v0.10.5...v0.10.7
[0.10.5]: https://github.com/heeh02/hybridVLA-2/compare/v0.10.3...v0.10.5
[0.10.3]: https://github.com/heeh02/hybridVLA-2/compare/v0.1.0...v0.10.3
[0.1.0]: https://github.com/heeh02/hybridVLA-2/releases/tag/v0.1.0
