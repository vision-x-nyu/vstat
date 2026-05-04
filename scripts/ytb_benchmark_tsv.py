"""Tab-separated ytb benchmark table with per-task ACC / MCQ / NUM columns.

Layout:
  Row 1 (header):     | | | AVG | | | task1 | | | task2 | | | ...
  Row 2 (sub-header): | Directory Name | ACC | SCORE | NUM | ACC | MCQ | NUM | ACC | MCQ | NUM | ...
  Data rows:          Group | Model | ACC | SCORE | NUM | ACC | MCQ | NUM | ACC | MCQ | NUM | ...

AVG spans 3 columns (ACC, SCORE, NUM); each task spans 3 columns (ACC, MCQ, NUM).
AVG ACC is computed over individual QA entries from samples_*.jsonl.
AVG SCORE averages MCQ correctness (1/0) and numeric MRA over individual QA entries.
AVG NUM is computed over per-entry MRA scores for numeric entries.

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


def _is_run_dir(p: Path) -> bool:
    return (p / "results.json").is_file()


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


def _task_from_samples_path(path: Path) -> str | None:
    prefix = f"samples_{TASK_PREFIX}"
    if not path.name.startswith(prefix) or path.suffix != ".jsonl":
        return None
    return path.stem[len(prefix):]


def _is_mcq_sample(sample: dict) -> bool:
    return "mra" not in sample


def _compute_entry_metrics(run_dir: Path, tasks: set[str]) -> tuple[float | None, float | None, float | None]:
    correct = 0
    total = 0
    score_sum = 0.0
    score_total = 0
    mra_sum = 0.0
    mra_total = 0
    for path in sorted(run_dir.glob(f"samples_{TASK_PREFIX}*.jsonl")):
        task = _task_from_samples_path(path)
        if task not in tasks:
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                acc = sample.get("accuracy")
                if not isinstance(acc, dict) or "is_correct" not in acc:
                    continue
                total += 1
                is_correct = bool(acc["is_correct"])
                correct += is_correct
                if _is_mcq_sample(sample):
                    score_total += 1
                    score_sum += float(is_correct)

                mra = sample.get("mra")
                if isinstance(mra, dict):
                    mra_score = _as_float(mra.get("mra_score"))
                    if mra_score is not None:
                        mra_total += 1
                        mra_sum += mra_score
                        score_total += 1
                        score_sum += mra_score

    avg_acc = correct / total if total else None
    avg_score = score_sum / score_total if score_total else None
    avg_num = mra_sum / mra_total if mra_total else None
    return avg_acc, avg_score, avg_num


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return EM
    return str(float(value) * 100.0)


def _row(cells: list[str]) -> str:
    return "\t".join(cells)


def render_tsv(root: Path, group_name: str) -> str:
    if _is_run_dir(root):
        # Single run directory: <ytb-root>/<model_key>/<model_name>/<timestamp>
        mk = root.parent.parent.name if root.parent.parent != root.parent else root.name
        runs: dict[str, Path] = {mk: root}
    else:
        runs = _discover_latest_runs(root)

    assert runs, f"no results.json found under {root}"

    # Canonical task order: union of tasks observed across all runs.
    all_tasks: set[str] = set()
    for rd in runs.values():
        all_tasks.update(_tasks_from_results(rd / "results.json"))
    tasks = sorted(all_tasks)

    # h1: list[str] = ["", "", "AVG", "", ""]
    # h2: list[str] = ["", "Directory Name", "ACC", "SCORE", "NUM"]
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
        avg_acc, avg_mcq, avg_num = _compute_entry_metrics(run_dir, set(tasks))

        cells: list[str] = [group_name, mk, _fmt_pct(avg_acc), _fmt_pct(avg_mcq), _fmt_pct(avg_num)]
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
