# LIBERO Hybrid

This directory isolates the first-pass LIBERO experiment workflow from the
main HybridVLA research tree.

Current scope:

- Single-camera LIBERO baseline configs
- Official two-camera LIBERO config
- HDF5 validation utility
- Normalizer-stat computation wrapper
- Stage A/B/C training wrapper with explicit real-data overrides
- LIBERO suite-aware path resolution that matches the official layout

## Official Reference

Official LIBERO repository:

- [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

Official dataset workflow from the upstream project:

- `python benchmark_scripts/download_libero_datasets.py`
- datasets are stored under the LIBERO datasets root and organized by suite

The important implication for this project is that the data is expected to look
like:

```text
<datasets_root>/
  libero_spatial/
    <task_name>_demo.hdf5
      data/
        problem_info
        demo_0/
          actions
          obs/
            agentview_rgb
            eye_in_hand_rgb
            joint_states
        demo_1/
  libero_object/
    <task_name>_demo.hdf5
  libero_goal/
    <task_name>_demo.hdf5
  libero_90/
    <task_name>_demo.hdf5
  libero_10/
    <task_name>_demo.hdf5
```

So the wrappers here accept a dataset root plus a `--suite`, and then resolve
the actual HDF5 directory to `<datasets_root>/<suite>`. Official LIBERO is not
"one episode per file"; each task file contains many `demo_x` trajectories.

## Recommended Workflow

1. Validate the official task HDF5 files:

```bash
python -m libero_hybrid.scripts.validate_libero_hdf5 \
  --data-dir /path/to/LIBERO/datasets \
  --suite libero_spatial
```

2. Compute normalization statistics:

```bash
python -m libero_hybrid.scripts.compute_libero_stats \
  --data-dir /path/to/LIBERO/datasets \
  --suite libero_spatial \
  --output-root outputs/libero_hybrid
```

3. Dry-run the resolved Stage A config:

```bash
python -m libero_hybrid.scripts.train_libero \
  --stage a \
  --data-dir /path/to/LIBERO/datasets \
  --suite libero_spatial \
  --output-root outputs/libero_hybrid \
  --dry-run
```

4. Start Stage A/B/C:

```bash
python -m libero_hybrid.scripts.train_libero \
  --stage a \
  --data-dir /path/to/LIBERO/datasets \
  --suite libero_spatial \
  --output-root outputs/libero_hybrid

python -m libero_hybrid.scripts.train_libero \
  --stage b \
  --data-dir /path/to/LIBERO/datasets \
  --suite libero_spatial \
  --output-root outputs/libero_hybrid

python -m libero_hybrid.scripts.train_libero \
  --stage c \
  --data-dir /path/to/LIBERO/datasets \
  --suite libero_spatial \
  --output-root outputs/libero_hybrid
```

## Design Choices

- The baseline defaults to single-camera LIBERO.
- The wrappers follow the official LIBERO suite layout and `data/demo_x/obs/*`
  structure from upstream.
- The training wrapper uses `libero_hdf5`, not the generic flat `hdf5` loader.
- `phase` and `affordance` losses are disabled in the baseline configs until
  real labels or a weak-labeling pipeline are added.
- Stage C is configured as an honest low-LR joint fine-tune. RTC/FASTER are
  intentionally disabled here until the implementation is real.
