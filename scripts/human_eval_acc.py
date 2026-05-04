"""Aggregate human-eval submissions into accuracy / MRA / MAE per task.

Mirrors the parsing + scoring logic in
`lmms_eval/tasks/longvid-reasoning-eval_ytb/utils.py` so the human numbers are
directly comparable to the model's lmms-eval scores.

Submissions live at /nas2/benchmarks/vpi_eval_site/submissions_v2.jsonl and
look like:
    {"taskId": "ytb:boxing/0001_pt1", "dataset": "ytb", "kind": "numeric",
     "answer": "7", "choiceIndex": null, "correct": true, ...}

Ground truth is loaded from the same merged_qa JSON the eval site reads.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

QA_PATH = Path("/nas2/benchmarks/vpi/real/merged_qa/real_filtered_redact.json")

INTEGER_PATTERN = re.compile(r"-?\d+")
MCQ_LETTER_PATTERN = re.compile(r"\b([A-D])\b")
TASK_ID_RE = re.compile(r"^ytb:(?P<task>[^/]+)/(?P<vid>.+)$")


def _extract_mcq_letter(text):
    matches = MCQ_LETTER_PATTERN.findall(str(text).upper())
    return matches[-1] if matches else None


def _extract_last_integer(text):
    matches = [int(m) for m in INTEGER_PATTERN.findall(str(text))]
    return matches[-1] if matches else None


def _normalize_text(s):
    return str(s).strip().lower()


def _build_truth(qa_path):
    """Return {(source_task, video_id): truth_dict} for the ytb dataset."""
    with open(qa_path) as fh:
        doc = json.load(fh)
    truth = {}
    for source_task, items in doc["data"].items():
        for raw in items:
            choices = raw.get("choices")
            is_mcq = isinstance(choices, list) and len(choices) > 0
            answer_text = str(raw["answer"]).strip()
            target_value = None
            if not is_mcq:
                try:
                    target_value = int(answer_text)
                    answer_text = str(target_value)
                except ValueError:
                    target_value = None
            truth[(source_task, str(raw["video_id"]))] = {
                "source_task": source_task,
                "is_mcq": is_mcq,
                "choices": choices,
                "answer_text": answer_text,
                "answer_index": raw.get("answer_index"),
                "target_value": target_value,
            }
    return truth


def _score_submission(sub, truth_entry):
    """Return (is_correct, mra, mae) for one submission. mra/mae are None when
    the task isn't numerical / has no integer ground truth."""
    is_mcq = truth_entry["is_mcq"]
    answer_text = truth_entry["answer_text"]
    target_value = truth_entry["target_value"]

    if is_mcq:
        ci = sub.get("choiceIndex")
        ai = truth_entry.get("answer_index")
        is_correct = isinstance(ci, int) and isinstance(ai, int) and ci == ai
        return is_correct, None, None

    raw_answer = sub.get("answer", "")
    if target_value is not None:
        parsed = _extract_last_integer(raw_answer)
        if parsed is None:
            return False, 0.0, None
        is_correct = parsed == target_value
        if target_value == 0:
            mra = 1.0 if parsed == 0 else 0.0
        else:
            mra = max(0.0, 1.0 - abs(parsed - target_value) / abs(target_value))
        mae = abs(parsed - target_value)
        return is_correct, mra, mae

    parsed = _normalize_text(raw_answer)
    is_correct = parsed == _normalize_text(answer_text)
    return is_correct, None, None


def _iter_ytb_submissions(submission_path):
    with open(submission_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("dataset") != "ytb":
                continue
            m = TASK_ID_RE.match(rec.get("taskId", ""))
            if not m:
                continue
            yield rec, m.group("task"), m.group("vid")


def aggregate(submission_path, task_filter=None):
    truth = _build_truth(QA_PATH)
    per_task = defaultdict(lambda: {"acc": [], "mra": [], "mae": []})
    missing = 0
    for sub, source_task, video_id in _iter_ytb_submissions(submission_path):
        if task_filter and source_task != task_filter:
            continue
        truth_entry = truth.get((source_task, video_id))
        if truth_entry is None:
            missing += 1
            continue
        is_correct, mra, mae = _score_submission(sub, truth_entry)
        per_task[source_task]["acc"].append(1.0 if is_correct else 0.0)
        if mra is not None:
            per_task[source_task]["mra"].append(mra)
        if mae is not None:
            per_task[source_task]["mae"].append(mae)
    return per_task, missing


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _fmt(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission_path", help="Path to submissions_v2.jsonl")
    ap.add_argument("--task", "-t", default=None,
                    help="Restrict to a single source_task (e.g. boxing)")
    args = ap.parse_args()

    per_task, missing = aggregate(args.submission_path, task_filter=args.task)

    if not per_task:
        print("No matching ytb submissions found.")
        if missing:
            print(f"({missing} submissions had no matching ground truth)")
        return

    header = f"{'task':<28} {'n':>4} {'acc':>8} {'n_num':>6} {'mra':>8} {'mae':>8}"
    print(header)
    print("-" * len(header))
    overall = {"acc": [], "mra": [], "mae": []}
    for task in sorted(per_task):
        rec = per_task[task]
        print(
            f"{task:<28} {len(rec['acc']):>4d} {_fmt(_mean(rec['acc'])):>8} "
            f"{len(rec['mra']):>6d} {_fmt(_mean(rec['mra'])):>8} "
            f"{_fmt(_mean(rec['mae']), digits=3):>8}"
        )
        overall["acc"].extend(rec["acc"])
        overall["mra"].extend(rec["mra"])
        overall["mae"].extend(rec["mae"])

    print("-" * len(header))
    print(
        f"{'OVERALL':<28} {len(overall['acc']):>4d} {_fmt(_mean(overall['acc'])):>8} "
        f"{len(overall['mra']):>6d} {_fmt(_mean(overall['mra'])):>8} "
        f"{_fmt(_mean(overall['mae']), digits=3):>8}"
    )
    if missing:
        print(f"\nNote: {missing} submissions had no matching QA ground truth (skipped).")


if __name__ == "__main__":
    main()
