"""Compute chance-level baselines for longvid-reasoning-eval tasks.

Usage:
    python random_chance.py /nas2/longvideo_eval/blender/data/merged_qa/1min.json \
        --output /nas2/longvideo_eval/longvid-reasoning-eval/results/1min/chance_levels.json

Input spec:
    merged_qa.json with structure:
        {"dataset_name": ..., "data": {"1min": {"task_name": [{"answer": ..., "choices": ...}, ...]}}}

Output spec:
    JSON dict with per-task chance scores:
        {"random": {"task_name": score_or_null, ...}, "frequency": {"task_name": score, ...}}
    Random chance for numeric tasks is null (displayed as "-").
    Frequency chance for numeric tasks is the MRA of the optimal constant prediction.
    Random chance for MCQ tasks is 1/num_choices.
    Frequency chance for MCQ tasks is the fraction of the most common answer.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

BENCH_KEY = "1min"
MCQ_TASKS = frozenset({"shell_game_rotate", "memory_sliding_puzzle"})


def mean_relative_accuracy(prediction, target, start=0.5, end=0.95, interval=0.05):
    steps = int(round((end - start) / interval)) + 1
    thresholds = [start + step * interval for step in range(steps)]
    relative_error = abs(prediction - target) / target
    return sum(relative_error <= 1 - threshold for threshold in thresholds) / len(thresholds)


def extract_numeric_targets(items, task_name):
    targets = []
    for item in items:
        ans = item["answer"]
        if task_name == "ring_toss_counting_physics":
            targets.append(float(ans["success"]))
        else:
            targets.append(float(ans))
    return np.array(targets)


def find_optimal_constant_mra(targets):
    """Find the constant prediction that maximizes average MRA across all targets."""
    ci_values = np.arange(0.5, 0.95 + 0.05, 0.05)
    critical_preds = []
    for t in targets:
        for ci in ci_values:
            critical_preds.extend([t * ci, t * (2 - ci)])
    critical_preds = np.unique(critical_preds)

    best_pred, best_score = None, -np.inf
    for pred in critical_preds:
        avg = np.mean([mean_relative_accuracy(pred, t) for t in targets])
        if avg > best_score:
            best_score = avg
            best_pred = pred
    return best_pred, float(best_score)


def compute_chance_levels(qa_path):
    data = json.loads(Path(qa_path).read_text())
    tasks_data = data["data"][BENCH_KEY]

    random_scores = {}
    frequency_scores = {}

    for task_name, items in tasks_data.items():
        if task_name in MCQ_TASKS:
            num_choices = len(items[0]["choices"])
            random_scores[task_name] = 1.0 / num_choices
            answer_counts = Counter(item["answer"] for item in items)
            frequency_scores[task_name] = max(answer_counts.values()) / len(items)
        else:
            random_scores[task_name] = None
            targets = extract_numeric_targets(items, task_name)
            _, best_mra = find_optimal_constant_mra(targets)
            frequency_scores[task_name] = best_mra

    return {"random": random_scores, "frequency": frequency_scores}


def main():
    parser = argparse.ArgumentParser(description="Compute chance-level baselines")
    parser.add_argument("qa_path", help="Path to merged_qa.json")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    results = compute_chance_levels(args.qa_path)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"\nWritten to {out}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
