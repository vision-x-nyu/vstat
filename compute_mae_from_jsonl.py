"""Compute MAE from existing eval JSONL files.

Usage:
    python compute_mae_from_jsonl.py [results_dir]

Scans all JSONL files under results_dir (default: results/), extracts
predictions and ground truth for numerical tasks, and computes MAE.
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

INTEGER_PATTERN = re.compile(r"\b\d+\b")

# Tasks that are numerical (not MCQ)
NUM_TASKS = {
    "block_counting", "hidden_dice_roll", "make_coffee",
    "rhythm_game", "tighten_untighten",
}

# Tasks where target is a letter (MCQ) — skip these
MCQ_TASKS = {
    "memory_sliding_puzzle", "opaque", "shell_game",
    "shell_game_rotate", "tilt_box",
}


def extract_last_integer(text):
    matches = [int(m) for m in INTEGER_PATTERN.findall(str(text))]
    return matches[-1] if matches else None


def parse_task_name(filename):
    """Extract task name from JSONL filename."""
    # Format: YYYYMMDD_HHMMSS_samples_longvid-reasoning-eval_Xsec_stretch_0p2s_TASKNAME.jsonl
    for task in sorted(NUM_TASKS | MCQ_TASKS, key=len, reverse=True):
        if f"_{task}.jsonl" in filename:
            return task
    return None


def process_jsonl(filepath, task_name):
    """Process a single JSONL file and return (predictions, targets, mae_scores)."""
    results = []
    with open(filepath) as f:
        for line in f:
            entry = json.loads(line)
            target = entry.get("target", "")
            prediction = entry.get("filtered_resps", "")

            if task_name in MCQ_TASKS:
                continue  # Skip MCQ tasks

            # Parse prediction
            pred_val = extract_last_integer(prediction)

            # Parse target
            try:
                gt_val = int(target)
            except (ValueError, TypeError):
                continue

            mae = abs(pred_val - gt_val) if pred_val is not None else None
            results.append({
                "doc_id": entry.get("doc_id"),
                "target": gt_val,
                "predicted": pred_val,
                "mae": mae,
            })
    return results


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

    # Find all JSONL files
    jsonl_files = sorted(Path(results_dir).rglob("*.jsonl"))

    # Group by model/duration/task
    summary = defaultdict(list)

    for filepath in jsonl_files:
        task_name = parse_task_name(filepath.name)
        if task_name is None or task_name in MCQ_TASKS:
            continue

        # Extract context from path
        parts = str(filepath).split("/")
        # Find duration and model info
        rel_path = str(filepath.relative_to(results_dir))

        results = process_jsonl(filepath, task_name)
        if not results:
            continue

        valid = [r for r in results if r["mae"] is not None]
        if not valid:
            continue

        mae = sum(r["mae"] for r in valid) / len(valid)
        accuracy = sum(1 for r in valid if r["mae"] == 0) / len(valid)

        # Also compute MRA
        mra_scores = []
        for r in valid:
            gt = r["target"]
            pred = r["predicted"]
            if gt == 0:
                mra_scores.append(1.0 if pred == 0 else 0.0)
            else:
                mra_scores.append(max(0.0, 1.0 - abs(pred - gt) / abs(gt)))
        mra = sum(mra_scores) / len(mra_scores)

        parse_failures = sum(1 for r in results if r["predicted"] is None)

        print(f"\n{'='*60}")
        print(f"File: {rel_path}")
        print(f"Task: {task_name} | Samples: {len(valid)} | Parse failures: {parse_failures}")
        print(f"  MAE:      {mae:.2f}")
        print(f"  MRA:      {mra:.1%}")
        print(f"  Accuracy: {accuracy:.1%}")

        # Show per-sample details if small enough
        if len(valid) <= 50:
            wrong = [r for r in valid if r["mae"] > 0]
            if wrong:
                print(f"  Wrong predictions ({len(wrong)}/{len(valid)}):")
                for r in sorted(wrong, key=lambda x: -x["mae"])[:10]:
                    print(f"    doc_id={r['doc_id']}: pred={r['predicted']}, gt={r['target']}, mae={r['mae']}")


if __name__ == "__main__":
    main()
