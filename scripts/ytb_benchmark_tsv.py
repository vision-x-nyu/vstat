"""Tab-separated ytb benchmark table with per-task ACC / MCQ / NUM columns.

Layout:
  Row 1 (header):     | | | AVG | | task1 | | | task2 | | | ...
  Row 2 (sub-header): | Directory Name | ACC | NUM | ACC | MCQ | NUM | ACC | MCQ | NUM | ...
  Data rows:          Group | Model | ACC | NUM | ACC | MCQ | NUM | ACC | MCQ | NUM | ...

AVG spans 2 columns (ACC, NUM); each task spans 3 columns (ACC, MCQ, NUM).
AVG values are means of the corresponding per-task columns over tasks where defined.

Per-task metrics are read directly from the aggregated results.json:
  ACC = `accuracy,none`   (all docs)
  NUM = `mra,none`        (mean relative accuracy, numerical docs only)
  MCQ = `accuracy,none`   if the task is pure-MCQ (no `mra,none` present), else blank
        (mixed tasks don't expose per-type accuracy in results.json)

Input tree (mirrors results/longvid-reasoning-eval_ytb/ layout):
  <root>/<model_key>/<model_name>/<YYYYMMDD_HHMMSS>/results.json + samples_*.jsonl
The latest run (lexicographically greatest timestamp dir) is picked per model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

TASK_PREFIX = "longvid-reasoning-eval_ytb_"
RUN_TS = re.compile(r"^\d{8}_\d{6}$")
EM = ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_root(arg: Path) -> Path:
    p = arg.expanduser()
    if p.is_absolute():
        out = p.resolve()
    else:
        cwd_guess = (Path.cwd() / p).resolve()
        repo_guess = (_repo_root() / p).resolve()
        out = cwd_guess if cwd_guess.is_dir() else repo_guess
    assert out.is_dir(), f"not a directory: {out}"
    print(out)
    return out


def _run_sort_key(results_json: Path) -> str:
    parent = results_json.parent.name
    return parent if RUN_TS.match(parent) else parent


def _discover_latest_runs(root: Path) -> dict[str, Path]:
    """model_key -> run directory containing results.json + samples_*.jsonl."""
    best: dict[str, tuple[str, Path]] = {}
    for p in root.rglob("results.json"):
        rel = p.relative_to(root)
        if not rel.parts:
            continue
        mk = rel.parts[0]
        key = _run_sort_key(p)
        run_dir = p.parent
        if mk not in best or key > best[mk][0]:
            best[mk] = (key, run_dir)
    return {mk: rd for mk, (_, rd) in best.items()}


def _tasks_from_results(results_json: Path) -> list[str]:
    data = json.loads(results_json.read_text(encoding="utf-8"))
    tasks: list[str] = []
    for key, payload in data.get("results", {}).items():
        if not key.startswith(TASK_PREFIX):
            continue
        if key == TASK_PREFIX.rstrip("_"):
            continue
        if not isinstance(payload, dict) or "accuracy,none" not in payload:
            continue
        tasks.append(key[len(TASK_PREFIX):])
    return sorted(tasks)


def _as_float(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def _compute_task_metrics(results: dict, task: str) -> tuple[float | None, float | None, float | None]:
    payload = results.get(f"{TASK_PREFIX}{task}")
    if not isinstance(payload, dict):
        return (None, None, None)
    acc = _as_float(payload.get("accuracy,none"))
    num = _as_float(payload.get("mra,none"))
    # mcq = acc if num is None else None
    return (acc, None, num)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return EM
    return str(float(value) * 100.0)


def _row(cells: list[str]) -> str:
    return "\t".join(cells)


def render_tsv(root: Path, group_name: str) -> str:
    runs = _discover_latest_runs(root)
    
    assert runs, f"no results.json found under {root}"

    # Canonical task order: union of tasks observed across all runs.
    all_tasks: set[str] = set()
    for rd in runs.values():
        all_tasks.update(_tasks_from_results(rd / "results.json"))
    tasks = sorted(all_tasks)

    # h1: list[str] = ["", "", "AVG", ""]
    # h2: list[str] = ["", "Directory Name", "ACC", "NUM"]
    # for t in tasks:
    #     h1.extend([t, "", ""])
    #     h2.extend(["ACC", "MCQ", "NUM"])

    # lines: list[str] = [_row(h1), _row(h2)]
    lines: list[str] = []

    for mk in sorted(runs):
        run_dir = runs[mk]
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8")).get(
            "results", {}
        )
        per_task: list[tuple[float | None, float | None, float | None]] = [
            _compute_task_metrics(results, t) for t in tasks
        ]
        acc_vals = [a for a, _, _ in per_task if a is not None]
        num_vals = [n for _, _, n in per_task if n is not None]
        avg_acc = sum(acc_vals) / len(acc_vals) if acc_vals else None
        avg_num = sum(num_vals) / len(num_vals) if num_vals else None

        cells: list[str] = [group_name, mk, _fmt_pct(avg_acc), _fmt_pct(avg_num)]
        for acc, mcq, num in per_task:
            cells.extend([_fmt_pct(acc), _fmt_pct(mcq), _fmt_pct(num)])
        lines.append(_row(cells))

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render ytb benchmark TSV with ACC/MCQ/NUM columns per task"
    )
    parser.add_argument(
        "--ytb-root",
        type=Path,
        default=_repo_root() / "results" / "longvid-reasoning-eval_ytb",
        help="Directory containing per-model result trees",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="TSV output path (default: <ytb-root>/benchmark_tables.tsv)",
    )
    args = parser.parse_args()

    root = _resolve_root(args.ytb_root)
    if 'stretch' in root.name:
        group_name = "w stretch"
    else:
        group_name = "w/o stretch"
    out = args.output if args.output is not None else root / "benchmark_tables.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_tsv(root, group_name=group_name), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
