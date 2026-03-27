"""Validate official LIBERO task HDF5 files before training."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import h5py

from libero_hybrid.utils import (
    extract_libero_language,
    resolve_libero_suite_dir,
    sorted_libero_demo_keys,
)


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _check_demo(
    file_path: Path,
    demo_key: str,
    action_key: str,
    proprio_keys: List[str],
    image_keys: Iterable[str],
    min_len: int,
) -> tuple[bool, Counter]:
    issues: Counter = Counter()
    ok = True

    try:
        with h5py.File(file_path, "r") as f:
            demo_grp = f["data"][demo_key]
            if action_key not in demo_grp:
                issues["missing_action_key"] += 1
                ok = False
            if "obs" not in demo_grp:
                issues["missing_obs_group"] += 1
                return False, issues

            obs_grp = demo_grp["obs"]
            for pk in proprio_keys:
                if pk not in obs_grp:
                    issues[f"missing_proprio:{pk}"] += 1
                    ok = False

            if ok:
                demo_len = demo_grp[action_key].shape[0]
                if demo_len < min_len:
                    issues["too_short"] += 1
                    ok = False

            for key in image_keys:
                if key not in obs_grp:
                    issues[f"missing_image:{key}"] += 1
                    ok = False
    except OSError:
        issues["open_failed"] += 1
        ok = False

    return ok, issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LIBERO HDF5 task files")
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--suite", default="libero_spatial", type=str)
    parser.add_argument("--action-key", default="actions", type=str)
    parser.add_argument("--proprio-key", default="joint_states", type=str)
    parser.add_argument("--proprio-keys", default="joint_states,gripper_states", type=str,
                        help="Comma-separated proprio keys to validate (official: joint_states,gripper_states)")
    parser.add_argument("--image-keys", default="agentview_rgb", type=str)
    parser.add_argument("--window", default=24, type=int)
    parser.add_argument("--chunk-horizon", default=16, type=int,
                        help="Must match training config chunk_horizon (default: 16 for LIBERO)")
    parser.add_argument("--limit", default=None, type=int)
    args = parser.parse_args()

    data_dir = resolve_libero_suite_dir(args.data_dir, args.suite)
    image_keys = _split_csv(args.image_keys)
    proprio_keys = _split_csv(args.proprio_keys)
    task_paths = sorted(data_dir.glob("*.hdf5"))
    if args.limit is not None:
        task_paths = task_paths[: args.limit]

    if not task_paths:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")

    min_len = args.window + args.chunk_horizon - 1
    bad: Counter = Counter()
    valid = 0
    total_demos = 0

    for file_path in task_paths:
        try:
            with h5py.File(file_path, "r") as f:
                if "data" not in f:
                    bad["missing_data_group"] += 1
                    continue
                data_grp = f["data"]
                if not extract_libero_language(data_grp):
                    bad["missing_language"] += 1

                demo_keys = sorted_libero_demo_keys(data_grp)
                if not demo_keys:
                    bad["missing_demos"] += 1
                    continue

                for demo_key in demo_keys:
                    total_demos += 1
                    ok, issues = _check_demo(
                        file_path=file_path,
                        demo_key=demo_key,
                        action_key=args.action_key,
                        proprio_keys=proprio_keys,
                        image_keys=image_keys,
                        min_len=min_len,
                    )
                    if ok:
                        valid += 1
                    else:
                        bad.update(issues)
        except OSError:
            bad["open_failed"] += 1

    print(f"Suite: {args.suite}")
    print(f"Data dir: {data_dir}")
    print(f"Task files checked: {len(task_paths)}")
    print(f"Demos checked: {total_demos}")
    print(f"Valid demos: {valid}")
    print(f"Invalid demos: {total_demos - valid}")
    print(f"Required minimum length: {min_len} (window={args.window} + chunk_horizon={args.chunk_horizon} - 1)")
    print(f"Image keys checked: {image_keys}")
    print(f"Proprio keys checked: {proprio_keys}")
    if bad:
        print("Issue summary:")
        for key, value in sorted(bad.items()):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
