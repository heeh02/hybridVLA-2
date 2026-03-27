# LIBERO Hybrid Audit Changelog

> **Date**: 2026-03-28
> **Scope**: Align libero_hybrid with official LIBERO benchmark semantics, not just HDF5 format
> **References**: Official repo cloned to /tmp/libero_official (github.com/Lifelong-Robot-Learning/LIBERO)
> **Standard**: "can it reproduce official benchmark success rates, not just report offline loss"

---

## Round 1 Fixes (data format alignment)

### 1. P0: action_dim / proprio_dim Mismatch (14 vs 7)

Model defaults to `action_dim=14`, `proprio_dim=14`. LIBERO Franka Panda has 7-dim actions. Configs now override both.

**Files**: `configs/data/libero_singlecam.yaml`, `libero_multicam.yaml`, `scripts/train_libero.py`

### 2. P1: Image Augmentation Missing from LIBERO Adapter

`libero_hdf5_adapter.py` had no augmentation; `transforms.py` crop was 224 not 448.

**Files**: `vla_hybrid_v2/data/libero_hdf5_adapter.py`, `vla_hybrid_v2/data/transforms.py`

### 3. P2: Stage B/C Missing Explicit Intervals

Added `log_interval`, `eval_interval`, `save_interval` to Stage B/C configs.

**Files**: `configs/train/libero_stage_b.yaml`, `libero_stage_c.yaml`

---

## Round 2 Fixes (benchmark semantic alignment)

### 4. P0: proprio_dim was 7 but official spec is 9

**Problem**: Official LIBERO obs spec (`default.yaml:27`) defines `low_dim: ["gripper_states", "joint_states"]`. The dataset creation script (`create_dataset.py:240-243`) writes both fields. Official policies use `gripper_states_dim=2` + `joint_states_dim=7` = 9-dim proprio. Our config only used `joint_states` (7-dim), missing 2-dim gripper state.

**Fix**:
- Added `proprio_keys: List[str]` field to `DataConfig` in `config.py`
- LIBERO configs now set `proprio_keys: [joint_states, gripper_states]` and `proprio_dim: 9`
- `libero_hdf5_adapter.py __getitem__`: when `proprio_keys` is set, reads and concatenates all keys along last dim
- `compute_libero_stats.py`: computes stats over concatenated proprio
- `validate_libero_hdf5.py`: validates all proprio keys exist
- `train_libero.py _apply_variant()`: sets `proprio_dim=9` and `proprio_keys`

**Files changed**:
- `vla_hybrid_v2/config.py` (added `proprio_keys` field)
- `vla_hybrid_v2/data/libero_hdf5_adapter.py` (concat proprio reading)
- `libero_hybrid/configs/data/libero_singlecam.yaml`
- `libero_hybrid/configs/data/libero_multicam.yaml`
- `libero_hybrid/scripts/train_libero.py`
- `libero_hybrid/scripts/compute_libero_stats.py`
- `libero_hybrid/scripts/validate_libero_hdf5.py`

---

### 5. P0: libero_100 falsely listed as supported suite

**Problem**: `utils.py` listed `libero_100` in `LIBERO_SUITES`. But the official `benchmark/__init__.py:56-62` `libero_suites` list does NOT include `libero_100`. The `LIBERO_100` class exists (line 214-219) but `_make_benchmark()` calls `task_maps["libero_100"]` which was never populated, raising `KeyError`.

**Fix**: Removed `libero_100` from `LIBERO_SUITES` with comment explaining why.

**File**: `libero_hybrid/utils.py`

---

### 6. P0: No benchmark rollout evaluation

**Problem**: The entire evaluation path was offline loss only (`train_unified.py evaluate()` line 268). Official LIBERO benchmark requires:
- BDDL task environment setup
- Fixed initial states from `.pruned_init` files
- SubprocVectorEnv with 20 parallel rollouts
- 600 max steps per trial
- Success rate = fraction of completed tasks
- Official task orders (21 permutations, `benchmark/__init__.py:83-105`)

Without this, no claimed benchmark result is reproducible.

**Fix**: Created `libero_hybrid/scripts/eval_libero_rollout.py` implementing the full official protocol:
- Uses `libero.libero.benchmark.get_benchmark()` for task lookup
- Loads fixed init states via `get_task_init_states()`
- Creates `SubprocVectorEnv` with `OffScreenRenderEnv` per task
- Runs rollouts with HybridVLA `semantic_step` + `control_step`
- Maps env obs keys to HDF5 obs keys (e.g., `agentview_image` -> `agentview_rgb`)
- Reports per-task and average success rate
- Supports `--all-tasks` for full suite evaluation
- Outputs JSON compatible with downstream analysis

**Usage**:
```bash
python -m libero_hybrid.scripts.eval_libero_rollout \
    --checkpoint outputs/libero_hybrid/libero_spatial/stage_c/checkpoint-latest \
    --config libero_hybrid/configs/train/libero_stage_c.yaml \
    --suite libero_spatial \
    --all-tasks \
    --output results/libero_spatial_eval.json
```

**File**: `libero_hybrid/scripts/eval_libero_rollout.py` (new)

---

### 7. P1: validate_libero_hdf5.py defaults didn't match training config

**Problem**: `chunk_horizon` default was 24 but training config uses 16. `--proprio-key` only validated `joint_states`, not `gripper_states`. Running validation with README defaults would be stricter than actual training on min-length, and miss gripper_states validation.

**Fix**:
- Changed `--chunk-horizon` default from 24 to 16
- Added `--proprio-keys` argument (default: `joint_states,gripper_states`)
- `_check_demo()` now validates all proprio keys, not just one
- Output now shows which proprio keys were checked

**File**: `libero_hybrid/scripts/validate_libero_hdf5.py`

---

## Round 3 Fixes (rollout correctness)

### 8. P0: Rollout state contamination — 20 envs shared one RuntimeCache

**Problem**: `evaluate_task()` created a single `RuntimeCache(batch_size=1)` and reused it for all 20 env iterations in the inner loop. `control_step()` mutates `temporal_state`, `current_chunk`, `chunk_step`, `action_history`, and `_last_seen_refresh` in-place (hybrid_vla_v2.py:745-794). Env k's state leaked into env k+1. On non-refresh steps the `grounder_out` local variable was also stale from the previous env.

**Fix**: Created per-env lists: `runtimes[k]`, `grounder_outs[k]`, `prev_actions[k]`. Each env gets its own `RuntimeCache` and `GrounderOutput`, eliminating all cross-contamination.

**File**: `libero_hybrid/scripts/eval_libero_rollout.py`

---

### 9. P1: prev_action hardcoded to zeros

**Problem**: Every `control_step()` call passed `prev_action = torch.zeros(...)`. The model projects `prev_action` into `prev_action_token` (hybrid_vla_v2.py:701) which feeds the temporal core. Zero prev_action means the temporal core never sees the actual action history through the explicit token path, diverging from training conditions.

**Fix**: Maintain `prev_actions[k]` per env, initialized to zeros, updated to the last emitted action after each `control_step()`.

**File**: `libero_hybrid/scripts/eval_libero_rollout.py`

---

### 10. P1: Multi-camera checkpoints evaluated as single-camera

**Problem**: `obs_to_model_input()` only processed `agentview_image`, ignoring `robot0_eye_in_hand_image`. `semantic_step()` was called with default `num_cameras=1`. A multi-camera checkpoint would be evaluated with the wrong observation path and token layout.

**Fix**: New `obs_to_semantic_input()` reads config `multi_camera.enable`. When True, processes both agentview and eye_in_hand images via `apply_chat_template`, and passes `num_cameras` to `semantic_step()`. Single-camera path unchanged.

**File**: `libero_hybrid/scripts/eval_libero_rollout.py`

---

### 11. Doc: README still showed libero_100 in directory layout

**Problem**: `README.md` line 46 listed `libero_100/` in the example directory tree, inconsistent with the code removing it from LIBERO_SUITES.

**Fix**: Replaced `libero_100/` with `libero_90/` + `libero_10/` to match official suites.

**File**: `libero_hybrid/README.md`

---

## Key Dimension Mapping (Updated)

| LIBERO Official | HDF5 Key | Env Obs Key | Config Field | Dim |
|-----------------|----------|-------------|-------------|:---:|
| Joint positions | `obs/joint_states` | `robot0_joint_pos` | `proprio_keys[0]` | 7 |
| Gripper position | `obs/gripper_states` | `robot0_gripper_qpos` | `proprio_keys[1]` | 2 |
| **Total proprio** | concat | concat | `proprio_dim` | **9** |
| Actions | `actions` | N/A | `action_expert.action_dim` | 7 |
| Agent camera | `obs/agentview_rgb` | `agentview_image` | `image_key` | HxWx3 |
| Wrist camera | `obs/eye_in_hand_rgb` | `robot0_eye_in_hand_image` | `camera_keys[1]` | HxWx3 |

## Data Flow (Updated)
```
LIBERO HDF5 (task_demo.hdf5)
  data/demo_X/actions [T, 7]
  data/demo_X/obs/joint_states [T, 7]
  data/demo_X/obs/gripper_states [T, 2]    <-- NEW: was missing
  data/demo_X/obs/agentview_rgb [T, H, W, 3]
  data/demo_X/obs/eye_in_hand_rgb [T, H, W, 3]
  data/attrs/problem_info -> JSON -> language_instruction
        |
LiberoHDF5DatasetAdapter
  proprio = concat(joint_states, gripper_states) -> [T, 9]   <-- NEW
  augmentation (train only): crop 448, rotate, color jitter
  normalize actions/proprio via pre-computed stats
  slice into windows (T=24, chunk_H=16)
  tokenize via Qwen2-VL processor
        |
Model (action_dim=7, proprio_dim=9, chunk_horizon=16)
        |
eval_libero_rollout.py
  SubprocVectorEnv x 20 with BDDL + init_states
  Per-env: RuntimeCache[k], grounder_out[k], prev_action[k]   <-- Round 3 fix
  semantic_step(num_cameras) at refresh interval               <-- multi-cam aware
  control_step(proprio, prev_action=last_action[k])            <-- proper prev_action
  success_rate = sum(dones) / 20
```

## Official Benchmark Suites (Corrected)

| Suite | # Tasks | Notes |
|-------|:-------:|-------|
| `libero_spatial` | 10 | Spatial variations |
| `libero_object` | 10 | Object variations |
| `libero_goal` | 10 | Goal variations |
| `libero_10` | 10 | Multi-scene complex |
| `libero_90` | 90 | Extended (no task_order) |
| ~~`libero_100`~~ | ~~N/A~~ | **REMOVED**: not in official `task_maps`, `LIBERO_100` class raises `KeyError` |
