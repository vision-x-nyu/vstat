"""Scan result JSON trees and build per-model × task × duration accuracy matrices.

Input / output documented in scripts/render_data_v2_benchmark_md.py (CLI wrapper).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from data_v2_benchmark_constants import (
    DURATIONS,
    LMMS_RUN,
    RUN_TS,
    STRETCH_RE,
    TASK_IDS_BY_SUFFIX,
)


def path_has_stretch(p: Path) -> bool:
    return STRETCH_RE.search(str(p)) is not None


RESULTS_FILE_TS = re.compile(r"^(\d{8}_\d{6})_results\.json$")


def run_sort_key(p: Path) -> str:
    parent = p.parent.name
    if RUN_TS.match(parent):
        return parent
    m = LMMS_RUN.search(parent)
    if m:
        return m.group(1)
    m = RESULTS_FILE_TS.match(p.name)
    if m:
        return m.group(1)
    return f"{parent}\0{p.name}"


def parse_duration(p: Path) -> str | None:
    s = str(p)
    for d in DURATIONS:
        if f"_v2_{d}" in s or f"eval_v2_{d}" in s:
            return d
    return None


def model_key_from_path(root: Path, p: Path) -> str | None:
    rel = p.relative_to(root)
    assert len(rel.parts) >= 1
    return rel.parts[0]


def is_results_filename(name: str) -> bool:
    if name == "results.json":
        return True
    return re.match(r"^\d{8}_\d{6}_results\.json$", name) is not None


def iter_result_files(root: Path, include_stretch: bool):
    for p in root.rglob("*results.json"):
        if not is_results_filename(p.name):
            continue
        if not include_stretch and path_has_stretch(p):
            continue
        yield p


def load_task_scores(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    out: dict[str, float] = {}
    for key, payload in results.items():
        if not isinstance(payload, dict):
            continue
        raw = payload.get("accuracy,none")
        if not isinstance(raw, (int, float)):
            continue
        for tid in TASK_IDS_BY_SUFFIX:
            if key.endswith(tid):
                out[tid] = float(raw)
                break
    return out


def build_matrix(
    root: Path,
    include_stretch: bool,
) -> dict[str, dict[str, dict[str, float]]]:
    best: dict[tuple[str, str, str], tuple[str, float]] = {}
    for p in iter_result_files(root, include_stretch):
        mk = model_key_from_path(root, p)
        dur = parse_duration(p)
        assert mk is not None
        assert dur is not None
        sk = run_sort_key(p)
        for tid, acc in load_task_scores(p).items():
            slot = (mk, dur, tid)
            if slot not in best or sk > best[slot][0]:
                best[slot] = (sk, acc)
    matrix: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (model_key, duration, tid), (_sk, acc) in best.items():
        matrix[model_key][tid][duration] = acc
    return matrix


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{round(float(value) * 100.0, 1):.1f}"


def random_chance_mcq(json_key: str) -> str:
    if json_key in ("shell_game", "shell_game_rotate"):
        return f"{round(100.0 / 3.0, 1):.1f}"
    return f"{25.0:.1f}"
