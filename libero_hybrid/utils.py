"""Helpers for aligning local workflows with official LIBERO layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

# Official suites from libero.benchmark.__init__.libero_suites (line 56-62).
# libero_100 is intentionally EXCLUDED: LIBERO_100 class exists but
# libero_suite_task_map has no "libero_100" key, so _make_benchmark() raises KeyError.
LIBERO_SUITES = {
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
}


def resolve_libero_suite_dir(data_path: str, suite: str) -> Path:
    root = Path(data_path).expanduser().resolve()
    if suite not in LIBERO_SUITES:
        raise ValueError(f"Unknown LIBERO suite: {suite}")

    if root.name == suite:
        return root

    candidate = root / suite
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve suite directory for '{suite}'. "
        f"Checked '{root}' and '{candidate}'."
    )


def suite_output_root(output_root: str, suite: str) -> Path:
    return Path(output_root).expanduser().resolve() / suite


def sorted_libero_demo_keys(data_group) -> List[str]:
    demo_keys = [key for key in data_group.keys() if key.startswith("demo_")]

    def _sort_key(name: str) -> tuple[int, str]:
        suffix = name.split("demo_", 1)[-1]
        return (int(suffix), name) if suffix.isdigit() else (10**9, name)

    return sorted(demo_keys, key=_sort_key)


def parse_libero_problem_info(data_group) -> Dict[str, Any]:
    raw = data_group.attrs.get("problem_info")
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(raw)
    except Exception:
        return {}


def extract_libero_language(data_group) -> str:
    problem_info = parse_libero_problem_info(data_group)
    language = problem_info.get("language_instruction")
    if isinstance(language, list):
        language = "".join(language)
    if isinstance(language, str):
        return language.strip().strip('"')
    return ""
