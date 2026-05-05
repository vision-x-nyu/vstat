#!/usr/bin/env python3
"""Compute the seven state breakdown scores from an lmms-eval sample log.

The score definition matches Willis's `per_state_acc_reparse.py`:
numeric answers are reparsed from `filtered_resps` and scored with MRA,
while MCQ answers contribute stored `accuracy.is_correct` as 0/1.  The
reported score is the mean of that mixed value within each state bucket.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_CANDIDATES = (
    REPO_ROOT / "sheets" / "vpi_full.csv",
    Path("/nas2/willis/longvid-reasoning-eval/sheets/vpi_full.csv"),
)

CSV_HEADER_ROWS = 3
COL_QUESTION = 0
COL_VIDEO = 1
COL_TASK = 2
COL_STATE1 = 4
COL_STATE2 = 5
COL_EXCLUDE = 10

TASK_STATE_OVERRIDES = {
    "shell_game_rotate": ("Location", "Atmoic"),
    "shuffle_puzzle": ("Location", "Atmoic"),
    "tilt_v2": ("Location", "Atmoic"),
    "funnel_ball": ("Location", "Dict"),
    "tighten_untighten": ("Count", "Atmoic"),
    "make_coffee": ("Attribute", "Set"),
    "dice": ("Count", "Atmoic"),
    "block_counting": ("Count", "Atmoic"),
}

MCQ_FORCED_TASKS = {"tilt_v2"}

STATE1_ORDER = (("Count", "Count"), ("Location", "Location"), ("Attribute", "Attribute"))
STATE2_ORDER = (("Atmoic", "Atomic"), ("Sequence", "Sequence"), ("Set", "Set"), ("Dict", "Dict"))

MCQ_TAIL_RE = re.compile(r"\s*\n\s*\(A\)", re.IGNORECASE)
QUESTION_RE = re.compile(r"Question:\s*(.+?)(?:\n\n|\Z)", re.DOTALL)
REDACTED_RE = re.compile(r"_redacted(?=\.mp4$)")
INTEGER_RE = re.compile(r"-?\d+")


def default_csv_path() -> Path:
    for path in DEFAULT_CSV_CANDIDATES:
        if path.exists():
            return path
    candidates = ", ".join(str(p) for p in DEFAULT_CSV_CANDIDATES)
    raise FileNotFoundError(f"Could not find vpi_full.csv. Tried: {candidates}")


def norm_path(video_path: str) -> str:
    return REDACTED_RE.sub("", video_path.strip())


def norm_question(text: str) -> str:
    text = (text or "").strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    text = text.replace("\\'", "'").replace('\\"', '"')
    match = MCQ_TAIL_RE.search(text)
    if match:
        text = text[: match.start()].rstrip()
    return text


def extract_question(input_text: str) -> str:
    match = QUESTION_RE.search(input_text or "")
    if not match:
        return ""
    stem = match.group(1).strip()
    mcq_match = MCQ_TAIL_RE.search(stem)
    if mcq_match:
        stem = stem[: mcq_match.start()].rstrip()
    return stem


def load_csv_states(csv_path: Path):
    states: dict[tuple[str, str], tuple[str, str]] = {}
    by_path: dict[str, list[str]] = defaultdict(list)
    excluded = 0

    with csv_path.open(newline="") as fp:
        reader = csv.reader(fp)
        for row_index, row in enumerate(reader):
            if row_index < CSV_HEADER_ROWS or len(row) <= COL_EXCLUDE:
                continue
            if row[COL_EXCLUDE].strip().upper() == "TRUE":
                excluded += 1
                continue

            video_path = norm_path(row[COL_VIDEO])
            question = norm_question(row[COL_QUESTION])
            if not video_path or not question:
                continue

            task = row[COL_TASK].strip()
            state1 = row[COL_STATE1].strip()
            state2 = row[COL_STATE2].strip()
            if task in TASK_STATE_OVERRIDES:
                state1, state2 = TASK_STATE_OVERRIDES[task]

            states[(video_path, question)] = (state1, state2)
            by_path[video_path].append(question)

    return states, dict(by_path), excluded


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def closest_question(by_path: dict[str, list[str]], video_path: str, question: str):
    candidates = by_path.get(video_path)
    if not candidates:
        return None

    q_norm = question.rstrip("?. ")
    prefix_hits = []
    for candidate in candidates:
        c_norm = candidate.rstrip("?. ")
        if c_norm.startswith(q_norm) or q_norm.startswith(c_norm):
            prefix_hits.append((candidate, abs(len(candidate) - len(question)), "prefix"))
    if prefix_hits:
        return min(prefix_hits, key=lambda item: item[1])

    best = min(candidates, key=lambda candidate: levenshtein(candidate, question))
    return best, levenshtein(best, question), "edit"


def sample_files(log_path: Path, recursive: bool) -> list[Path]:
    if log_path.is_file():
        return [log_path]

    direct = sorted(log_path.glob("samples_*.jsonl"))
    if direct or not recursive:
        return direct

    return sorted(log_path.rglob("samples_*.jsonl"))


def iter_jsonl(paths: list[Path]):
    for path in paths:
        with path.open() as fp:
            for line_num, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield path, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_num}: invalid JSON: {exc}") from exc


def task_from_samples_path(path: Path) -> str:
    name = path.name
    if name.startswith("samples_"):
        name = name[len("samples_") :]
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    for prefix in (
        "longvid-reasoning-eval_ytb_stretch_0p2s_",
        "longvid-reasoning-eval_ytb_",
        "longvid-reasoning-eval_recorded_",
    ):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def is_numeric_target(target: Any) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", str(target).strip()))


def target_to_int(target: Any) -> int:
    text = str(target).strip()
    value = float(text) if "." in text else int(text)
    if int(value) != value:
        raise ValueError(f"numeric target is not an integer: {target!r}")
    return int(value)


def extract_last_integer(text: Any) -> int | None:
    matches = [int(match) for match in INTEGER_RE.findall(str(text))]
    return matches[-1] if matches else None


def mra_score(pred: int, target: int) -> float:
    if target == 0:
        return 1.0 if pred == 0 else 0.0
    num_pts = (0.95 - 0.5) / 0.05 + 2
    thresholds = np.linspace(0.5, 0.95, int(num_pts))
    accuracy = abs(pred - target) / abs(target) <= 1 - thresholds
    return float(accuracy.mean())


def empty_bucket() -> dict[str, float]:
    return {"n": 0, "n_num": 0, "is_correct_sum": 0.0, "score_sum": 0.0, "num_sum": 0.0}


def add_to_bucket(bucket: dict[str, float], is_correct: bool, score: float, numeric_score: float | None):
    bucket["n"] += 1
    bucket["is_correct_sum"] += int(is_correct)
    bucket["score_sum"] += score
    if numeric_score is not None:
        bucket["n_num"] += 1
        bucket["num_sum"] += numeric_score


def bucket_metrics(bucket: dict[str, float]):
    n = bucket["n"]
    n_num = bucket["n_num"]
    if n == 0:
        return None
    return {
        "n": int(n),
        "acc": 100.0 * bucket["is_correct_sum"] / n,
        "score": 100.0 * bucket["score_sum"] / n,
        "num": 100.0 * bucket["num_sum"] / n_num if n_num else None,
        "n_num": int(n_num),
    }


def compute_scores(args):
    csv_path = Path(args.csv) if args.csv else default_csv_path()
    states, by_path, excluded = load_csv_states(csv_path)
    files = sample_files(Path(args.log), recursive=args.recursive)
    if not files:
        raise FileNotFoundError(f"No sample JSONL files found under {args.log}")

    overall = empty_bucket()
    by_state1: dict[str, dict[str, float]] = defaultdict(empty_bucket)
    by_state2: dict[str, dict[str, float]] = defaultdict(empty_bucket)
    unmatched = []
    fuzzy_matches = []
    seen = matched = 0

    for path, entry in iter_jsonl(files):
        seen += 1
        task = task_from_samples_path(path)
        video_path = norm_path(str(entry.get("video_path", "")).replace("_stretch_0p2s", ""))
        question = norm_question(extract_question(str(entry.get("input", ""))))

        if task in TASK_STATE_OVERRIDES:
            state1, state2 = TASK_STATE_OVERRIDES[task]
        else:
            key = (video_path, question)
            if key not in states and args.fuzzy:
                best = closest_question(by_path, video_path, question)
                if best is not None and (args.fuzzy_max_dist is None or best[1] <= args.fuzzy_max_dist):
                    csv_question, dist, kind = best
                    key = (video_path, csv_question)
                    fuzzy_matches.append((kind, dist, video_path, question, csv_question))
            if key not in states:
                unmatched.append((str(path), video_path, question))
                continue
            state1, state2 = states[key]

        if not state1 or not state2:
            unmatched.append((str(path), video_path, question))
            continue
        matched += 1

        numeric = is_numeric_target(entry.get("target")) and task not in MCQ_FORCED_TASKS
        if numeric:
            target = target_to_int(entry.get("target"))
            pred = extract_last_integer(entry.get("filtered_resps", ""))
            if pred is None:
                is_correct = False
                score = 0.0
            else:
                is_correct = pred == target
                score = mra_score(pred, target)
            numeric_score = score
        else:
            is_correct = bool(entry.get("accuracy", {}).get("is_correct"))
            score = 1.0 if is_correct else 0.0
            numeric_score = None

        for bucket in (overall, by_state1[state1], by_state2[state2]):
            add_to_bucket(bucket, is_correct, score, numeric_score)

    return {
        "csv": str(csv_path),
        "files": [str(path) for path in files],
        "excluded_csv_rows": excluded,
        "seen": seen,
        "matched": matched,
        "unmatched": unmatched,
        "fuzzy_matches": fuzzy_matches,
        "overall": bucket_metrics(overall),
        "state_element": {label: bucket_metrics(by_state1[key]) for key, label in STATE1_ORDER},
        "state_type": {label: bucket_metrics(by_state2[key]) for key, label in STATE2_ORDER},
    }


def fmt_score(metrics):
    if metrics is None:
        return "-"
    return f"{metrics['score']:.1f}"


def print_pretty(result):
    print(f"CSV: {result['csv']}")
    print(f"sample files: {len(result['files'])}")
    print(
        f"jsonl entries: {result['seen']}, matched: {result['matched']}, "
        f"fuzzy: {len(result['fuzzy_matches'])}, unmatched: {len(result['unmatched'])}"
    )
    print()
    print(f"{'bucket':<16} {'n':>5} {'score':>7}")
    print("-" * 30)
    overall = result["overall"]
    print(f"{'Avg':<16} {overall['n'] if overall else 0:>5} {fmt_score(overall):>7}")
    for label, metrics in result["state_element"].items():
        print(f"{label:<16} {metrics['n'] if metrics else 0:>5} {fmt_score(metrics):>7}")
    for label, metrics in result["state_type"].items():
        print(f"{label:<16} {metrics['n'] if metrics else 0:>5} {fmt_score(metrics):>7}")


def print_tsv(result):
    labels = ["Avg", "Count", "Location", "Attribute", "Atomic", "Sequence", "Set", "Dict"]
    metrics = [result["overall"]]
    metrics.extend(result["state_element"][label] for label in labels[1:4])
    metrics.extend(result["state_type"][label] for label in labels[4:])
    print("\t".join(labels))
    print("\t".join(fmt_score(metric) for metric in metrics))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="A sample JSONL file, or a directory containing samples_*.jsonl.")
    parser.add_argument("--csv", default=None, help="State-label CSV. Defaults to sheets/vpi_full.csv if present, then Willis's copy.")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Only read samples_*.jsonl directly inside the input directory.")
    parser.add_argument("--no-fuzzy", dest="fuzzy", action="store_false", help="Disable fuzzy question matching against the CSV.")
    parser.add_argument("--fuzzy-max-dist", type=int, default=None, help="Maximum edit distance accepted for fuzzy matches.")
    parser.add_argument("--format", choices=("pretty", "tsv", "json"), default="pretty")
    parser.add_argument("--fail-on-unmatched", action="store_true", help="Exit nonzero if any log entries cannot be matched to a state label.")
    parser.set_defaults(recursive=True, fuzzy=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = compute_scores(args)

    if args.format == "json":
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        print()
    elif args.format == "tsv":
        print_tsv(result)
    else:
        print_pretty(result)

    if result["unmatched"]:
        print(f"\nwarning: {len(result['unmatched'])} entries were unmatched", file=sys.stderr)
        if args.fail_on_unmatched:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
