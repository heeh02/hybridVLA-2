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

from vla_hybrid_v2.infer import HybridVLALiberoPolicy, find_resolved_config

logger = logging.getLogger(__name__)


def _import_libero():
    """Lazy import of LIBERO benchmark components."""
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

    return get_benchmark, OffScreenRenderEnv, SubprocVectorEnv


def _find_resolved_config(checkpoint_path: str) -> Optional[Path]:
    """Compatibility shim for tests and callers importing from this module."""
    return find_resolved_config(checkpoint_path)


def load_hybridvla_policy(
    checkpoint_path: str,
    config_path: Optional[str],
    device: str = "cuda:0",
):
    """Load a trained HybridVLA policy for LIBERO closed-loop inference."""
    return HybridVLALiberoPolicy.from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )


def _extract_env_obs_single(obs, k: int) -> dict:
    """Extract observation dict for env index k from vectorized obs."""
    if isinstance(obs, list):
        return obs[k]
    return {key: obs[key][k] for key in obs}


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_task(
    policy,
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

    cfg = policy.cfg
    task = benchmark.get_task(task_id)
    language = task.language
    action_dim = cfg.model.action_expert.action_dim
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
    runtimes = [policy.init_runtime(batch_size=1) for _ in range(n_eval)]
    grounder_outs = [None] * n_eval
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
                    grounder_outs[k] = policy.semantic_step_from_obs(obs_k, language)
                    runtimes[k].runtime_cache.refresh_counter += 1

                # Control step with correct per-env state
                step_out = policy.control_step_from_obs(
                    obs_single=obs_k,
                    runtime_state=runtimes[k],
                    semantic_summary=grounder_outs[k],
                )

                action = step_out.action_env[0]  # [A]
                actions_batch[k] = action.cpu().numpy()

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
    policy = load_hybridvla_policy(
        args.checkpoint, args.config, args.device,
    )
    cfg = policy.cfg
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
            policy, benchmark, tid,
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
