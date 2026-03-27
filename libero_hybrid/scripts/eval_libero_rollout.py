"""LIBERO benchmark rollout evaluation for HybridVLA.

Performs success-rate evaluation using the official LIBERO benchmark protocol:
- BDDL task definitions
- Fixed initial states from .pruned_init files
- SubprocVectorEnv for parallel rollouts
- 20 trials per task, 600 max steps per trial

Config resolution (in priority order):
1. If --config is passed, use it (with mismatch warning against resolved_config.yaml)
2. Otherwise, auto-discover resolved_config.yaml from the checkpoint's parent directory
3. Fail if neither is found

Requires:
    pip install libero robosuite

Usage:
    # Best practice: omit --config, let the script find resolved_config.yaml
    python -m libero_hybrid.scripts.eval_libero_rollout \
        --checkpoint outputs/libero_hybrid/libero_spatial/stage_c/checkpoint-latest \
        --suite libero_spatial \
        --all-tasks

    # Explicit config (will warn if it mismatches resolved_config.yaml)
    python -m libero_hybrid.scripts.eval_libero_rollout \
        --checkpoint outputs/libero_hybrid/libero_spatial/stage_c/checkpoint-latest \
        --suite libero_spatial \
        --task-id 0 \
        --config path/to/resolved_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# LIBERO env obs key -> HDF5 obs key mapping
_ENV_IMAGE_MAP = {
    "agentview_image": "agentview_rgb",
    "robot0_eye_in_hand_image": "eye_in_hand_rgb",
}
_HDF5_PROPRIO_TO_ENV = {
    "joint_states": "robot0_joint_pos",
    "gripper_states": "robot0_gripper_qpos",
}


def _import_libero():
    """Lazy import of LIBERO benchmark components."""
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

    return get_benchmark, OffScreenRenderEnv, SubprocVectorEnv


def _find_resolved_config(checkpoint_path: str) -> Optional[Path]:
    """Search for resolved_config.yaml near the checkpoint directory.

    train_libero.py saves resolved_config.yaml in the stage output dir,
    which is the parent (or grandparent for checkpoint-latest symlinks)
    of the checkpoint directory.
    """
    ckpt = Path(checkpoint_path)
    if ckpt.is_symlink():
        ckpt = ckpt.resolve()
    # checkpoint-STEP/ is inside the stage output dir
    for parent in [ckpt.parent, ckpt.parent.parent]:
        candidate = parent / "resolved_config.yaml"
        if candidate.exists():
            return candidate
    return None


def load_hybridvla_policy(
    checkpoint_path: str,
    config_path: Optional[str],
    device: str = "cuda:0",
):
    """Load a trained HybridVLA model for inference.

    Config resolution:
    - If config_path is given, use it but warn if resolved_config.yaml disagrees
      on multi_camera.enable.
    - If config_path is None, auto-discover resolved_config.yaml from the
      checkpoint directory.  Fail if not found.

    Returns (model, cfg, processor).
    """
    from vla_hybrid_v2.config import load_config
    from vla_hybrid_v2.utils.checkpointing import load_checkpoint

    resolved = _find_resolved_config(checkpoint_path)

    if config_path is not None:
        cfg = load_config(config_path)
        # Mismatch detection: compare with resolved_config.yaml if it exists
        if resolved is not None:
            resolved_cfg = load_config(str(resolved))
            if resolved_cfg.model.multi_camera.enable != cfg.model.multi_camera.enable:
                raise RuntimeError(
                    f"Config mismatch: --config has multi_camera.enable="
                    f"{cfg.model.multi_camera.enable} but the checkpoint was "
                    f"trained with multi_camera.enable="
                    f"{resolved_cfg.model.multi_camera.enable} "
                    f"(from {resolved}).\n"
                    f"Use the resolved config or omit --config to auto-discover it."
                )
            if resolved_cfg.model.proprio_dim != cfg.model.proprio_dim:
                logger.warning(
                    "Config mismatch: --config proprio_dim=%d but resolved=%d. "
                    "Using --config value. This may cause shape errors.",
                    cfg.model.proprio_dim, resolved_cfg.model.proprio_dim,
                )
    elif resolved is not None:
        logger.info("Auto-discovered config: %s", resolved)
        cfg = load_config(str(resolved))
    else:
        raise FileNotFoundError(
            f"No --config provided and no resolved_config.yaml found near "
            f"{checkpoint_path}. Either pass --config explicitly or ensure "
            f"the checkpoint was created by train_libero.py (which saves "
            f"resolved_config.yaml)."
        )

    from vla_hybrid_v2.models.hybrid_vla_v2 import HybridVLAv2

    model = HybridVLAv2(cfg)
    load_checkpoint(checkpoint_path, model, strict=False)
    model = model.to(device).eval()

    processor = None
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(cfg.model.backbone.name)
    except Exception as e:
        logger.warning("Could not load processor: %s", e)

    return model, cfg, processor


# ------------------------------------------------------------------
# Observation conversion
# ------------------------------------------------------------------

def _extract_env_obs_single(obs, k: int) -> dict:
    """Extract observation dict for env index k from vectorized obs."""
    if isinstance(obs, list):
        return obs[k]
    return {key: obs[key][k] for key in obs}


def _make_pil_image(img_np):
    """Convert HWC uint8 numpy array to 448x448 RGB PIL Image."""
    from PIL import Image
    if img_np is None or img_np.ndim != 3:
        return None
    img = Image.fromarray(img_np.astype(np.uint8))
    return img.resize((448, 448), Image.BILINEAR).convert("RGB")


def obs_to_semantic_input(
    obs_single: dict,
    language: str,
    cfg,
    processor,
    device: str,
    multi_camera: bool,
) -> dict:
    """Convert single-env observation to semantic_step input tensors.

    Handles both single-camera and multi-camera modes.
    """
    agentview = _make_pil_image(obs_single.get("agentview_image"))
    eye_in_hand = _make_pil_image(obs_single.get("robot0_eye_in_hand_image"))

    if processor is None:
        L = cfg.data.max_text_length
        return {
            "input_ids": torch.zeros(1, L, dtype=torch.long, device=device),
            "attention_mask": torch.ones(1, L, dtype=torch.long, device=device),
        }

    # Fail-fast: multicam mode requires both cameras — never silently degrade
    if multi_camera and eye_in_hand is None:
        raise RuntimeError(
            "multi_camera.enable=True but robot0_eye_in_hand_image is missing "
            "from env observation. Check that the LIBERO env is configured to "
            "render eye_in_hand camera."
        )

    if multi_camera and agentview is not None and eye_in_hand is not None:
        # Multi-camera: use apply_chat_template for proper multi-image tokens
        content = [{"type": "image"}, {"type": "image"},
                   {"type": "text", "text": language}]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        tok = processor(
            text=[text], images=[agentview, eye_in_hand],
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=cfg.data.max_text_length,
        )
    elif agentview is not None:
        tok = processor(
            text=language, images=agentview,
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=cfg.data.max_text_length,
        )
    else:
        tok = processor(
            text=language, return_tensors="pt",
            padding="max_length", truncation=True,
            max_length=cfg.data.max_text_length,
        )

    result = {
        "input_ids": tok["input_ids"].to(device),
        "attention_mask": tok["attention_mask"].to(device),
    }
    if "pixel_values" in tok:
        result["pixel_values"] = tok["pixel_values"].to(device)
    if "image_grid_thw" in tok:
        result["image_grid_thw"] = tok["image_grid_thw"].to(device)
    return result


def obs_to_proprio(
    obs_single: dict,
    proprio_keys: List[str],
    device: str,
) -> torch.Tensor:
    """Extract and concatenate proprio from env observation. Returns [1, P]."""
    parts = []
    for pk in proprio_keys:
        env_key = _HDF5_PROPRIO_TO_ENV.get(pk, pk)
        if env_key in obs_single:
            parts.append(obs_single[env_key].astype(np.float32))
    if parts:
        proprio = np.concatenate(parts)
    else:
        proprio = np.zeros(9, dtype=np.float32)
    return torch.from_numpy(proprio).unsqueeze(0).to(device)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_task(
    model,
    cfg,
    processor,
    benchmark,
    task_id: int,
    device: str = "cuda:0",
    n_eval: int = 20,
    max_steps: int = 600,
) -> Dict[str, float]:
    """Evaluate a single LIBERO task using official benchmark protocol.

    Each of the n_eval environments gets its own RuntimeCache so there is
    no cross-contamination of temporal state, action history, or chunk cache.
    """
    _, OffScreenRenderEnv, SubprocVectorEnv = _import_libero()
    from libero.libero import get_libero_path

    task = benchmark.get_task(task_id)
    language = task.language
    action_dim = cfg.model.action_expert.action_dim
    proprio_keys = cfg.data.proprio_keys if cfg.data.proprio_keys else [cfg.data.proprio_key]
    multi_camera = cfg.model.multi_camera.enable
    num_cameras = cfg.model.multi_camera.num_cameras if multi_camera else 1
    refresh_interval = max(1, int(cfg.infer.control_hz / cfg.infer.semantic_hz))

    # ---- Set up environments ----
    bddl_file_path = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env_args = {
        "bddl_file_name": bddl_file_path,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    env = SubprocVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(n_eval)]
    )
    env.seed(0)

    # Fixed initial states (official benchmark protocol)
    init_states_path = os.path.join(
        get_libero_path("init_states"),
        task.problem_folder,
        task.init_states_file,
    )
    init_states = torch.load(init_states_path, weights_only=True)
    indices = np.arange(n_eval) % init_states.shape[0]
    obs = env.set_init_state(init_states[indices])

    # Warm up physics (official protocol: 5 zero-action steps)
    for _ in range(5):
        env.step(np.zeros((n_eval, 7)))

    # ---- Per-env state: one RuntimeCache, one grounder_out, one prev_action per env ----
    runtimes = [model.init_runtime(batch_size=1, device=device) for _ in range(n_eval)]
    grounder_outs = [None] * n_eval
    prev_actions = [torch.zeros(1, action_dim, device=device) for _ in range(n_eval)]
    dones = [False] * n_eval
    total_steps = 0

    with torch.no_grad():
        for step in range(max_steps):
            actions_batch = np.zeros((n_eval, action_dim))

            for k in range(n_eval):
                if dones[k]:
                    continue

                obs_k = _extract_env_obs_single(obs, k)

                # Semantic step at refresh interval
                if step % refresh_interval == 0:
                    sem_input = obs_to_semantic_input(
                        obs_k, language, cfg, processor, device, multi_camera,
                    )
                    grounder_outs[k] = model.semantic_step(
                        input_ids=sem_input.get("input_ids"),
                        attention_mask=sem_input.get("attention_mask"),
                        pixel_values=sem_input.get("pixel_values"),
                        image_grid_thw=sem_input.get("image_grid_thw"),
                        num_cameras=num_cameras,
                    )
                    runtimes[k].refresh_counter += 1

                # Control step with correct per-env state
                proprio = obs_to_proprio(obs_k, proprio_keys, device)
                control_out = model.control_step(
                    proprio=proprio,
                    prev_action=prev_actions[k],
                    semantic_summary=grounder_outs[k],
                    runtime_state=runtimes[k],
                )

                action = control_out.action[0]  # [A]
                actions_batch[k] = action.cpu().numpy()
                prev_actions[k] = action.unsqueeze(0)  # [1, A] for next step

            obs, reward, done, info = env.step(actions_batch)
            total_steps += 1

            for k in range(n_eval):
                dones[k] = dones[k] or done[k]
            if all(dones):
                break

    num_success = sum(int(d) for d in dones)
    env.close()

    return {
        "task_id": task_id,
        "task_name": task.name,
        "language": language,
        "success_rate": num_success / n_eval,
        "num_success": num_success,
        "n_eval": n_eval,
        "total_steps": total_steps,
        "multi_camera": multi_camera,
    }


def main():
    parser = argparse.ArgumentParser(description="LIBERO Benchmark Rollout Evaluation")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML. If omitted, auto-discovers resolved_config.yaml "
                             "from checkpoint directory.")
    parser.add_argument("--suite", default="libero_spatial", type=str,
                        choices=["libero_spatial", "libero_object", "libero_goal",
                                 "libero_10", "libero_90"])
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--all-tasks", action="store_true",
                        help="Evaluate all tasks in the suite")
    parser.add_argument("--task-order", type=int, default=0,
                        help="Official task order index (0-20)")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="Number of rollout trials per task (official: 20)")
    parser.add_argument("--max-steps", type=int, default=600,
                        help="Max steps per rollout (official: 600)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load benchmark
    get_benchmark, _, _ = _import_libero()
    suite_to_class = {
        "libero_spatial": "LIBERO_SPATIAL",
        "libero_object": "LIBERO_OBJECT",
        "libero_goal": "LIBERO_GOAL",
        "libero_10": "LIBERO_10",
        "libero_90": "LIBERO_90",
    }
    benchmark = get_benchmark(suite_to_class[args.suite])(args.task_order)
    n_tasks = benchmark.get_num_tasks()
    logger.info("Suite: %s, %d tasks, task_order=%d", args.suite, n_tasks, args.task_order)

    # Load model
    model, cfg, processor = load_hybridvla_policy(
        args.checkpoint, args.config, args.device,
    )
    logger.info("Model loaded from %s", args.checkpoint)
    logger.info("  action_dim=%d  proprio_dim=%d  multi_camera=%s",
                cfg.model.action_expert.action_dim, cfg.model.proprio_dim,
                cfg.model.multi_camera.enable)

    # Evaluate
    task_ids = list(range(n_tasks)) if args.all_tasks else [args.task_id]
    results = []

    for tid in task_ids:
        logger.info("Evaluating task %d/%d: %s", tid + 1, n_tasks,
                     benchmark.get_task(tid).name)
        t0 = time.monotonic()
        result = evaluate_task(
            model, cfg, processor, benchmark, tid,
            device=args.device, n_eval=args.n_eval, max_steps=args.max_steps,
        )
        elapsed = time.monotonic() - t0
        result["eval_time_s"] = elapsed
        results.append(result)
        logger.info(
            "  Task %d: success_rate=%.1f%% (%d/%d) in %.1fs",
            tid, result["success_rate"] * 100,
            result["num_success"], result["n_eval"], elapsed,
        )

    # Aggregate
    avg_sr = np.mean([r["success_rate"] for r in results])
    logger.info("Suite %s average success rate: %.1f%%", args.suite, avg_sr * 100)

    summary = {
        "suite": args.suite,
        "task_order": args.task_order,
        "checkpoint": args.checkpoint,
        "n_eval": args.n_eval,
        "max_steps": args.max_steps,
        "average_success_rate": float(avg_sr),
        "per_task": results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Results saved to %s", out_path)
    else:
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
