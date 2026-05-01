"""Compute chance-level baselines for longvid-reasoning-eval_ytb tasks.

Usage:
    python random_chance_ytb.py /nas2/benchmarks/vpi/ytb-vids/merged_qa/merged_adv_qa.json \
        --output results/ytb/chance_levels.json

Input spec:
    merged_adv_qa.json with structure:
        {"dataset_name": ..., "data": {"<task_name>": [{"answer": ..., "choices": ...}, ...]}}
    Unlike the recorded benchmark, there is no duration sub-key; tasks sit
    directly under "data". Items within a single task may be MCQ
    (choices is not None, with varying num_choices), numeric integer, or
    free-form text (e.g. "Forehand") — matching the per-doc dispatch in
    lmms_eval/tasks/longvid-reasoning-eval_ytb/utils.py.

Output spec:
    Per-task, per-subset baselines. Each subset's score is normalized by the
    size of that subset (not the whole task), so e.g. mcq_random_accuracy
    reflects chance-level on the MCQ items alone:
        {
          "<task_name>": {
            "n_items": int,
            "n_mcq": int,
            "n_numeric": int,
            "n_text": int,
            "mcq_random_accuracy":      float|null,  // mean(1/num_choices) over MCQ items
            "mcq_frequency_accuracy":   float|null,  // most-common letter / n_mcq
            "numeric_frequency_accuracy": float|null, // most-common int / n_numeric
            "text_frequency_accuracy":  float|null,  // most-common text / n_text
            "random_mra":    null,
            "frequency_mra": float|null              // optimal constant MRA over numeric items
          }, ...
        }
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def mean_relative_accuracy(prediction, target, start=0.5, end=0.95, interval=0.05):
    # Match utils.py:_compute_mra semantics for zero-valued ground truth:
    # exact-equality → 1.0 across all thresholds, otherwise → 0.0.
    if target == 0:
        return 1.0 if prediction == 0 else 0.0
    steps = int(round((end - start) / interval)) + 1
    thresholds = [start + step * interval for step in range(steps)]
    relative_error = abs(prediction - target) / abs(target)
    return sum(relative_error <= 1 - threshold for threshold in thresholds) / len(thresholds)


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


def _classify(item):
    """Return ('mcq', letter_answer, num_choices) | ('numeric', int) | ('text', str)."""
    if item.get("choices") is not None:
        return "mcq", str(item["answer"]).strip(), len(item["choices"])
    text = str(item["answer"]).strip()
    try:
        return "numeric", int(text), None
    except ValueError:
        return "text", text.lower(), None


def compute_task_chance(items):
    n = len(items)
    mcq_answers, mcq_choice_sizes = [], []
    numeric_targets = []
    text_answers = []

    for item in items:
        kind, ans, num_choices = _classify(item)
        if kind == "mcq":
            mcq_answers.append(ans)
            mcq_choice_sizes.append(num_choices)
        elif kind == "numeric":
            numeric_targets.append(ans)
        else:
            text_answers.append(ans)

    # Per-subset baselines: each score is normalized by the size of its own
    # subset, not the whole task. This keeps the MCQ/numeric/text baselines
    # comparable across tasks with different type mixes.
    if mcq_answers:
        mcq_random = sum(1.0 / k for k in mcq_choice_sizes) / len(mcq_answers)
        mcq_freq = max(Counter(mcq_answers).values()) / len(mcq_answers)
    else:
        mcq_random = None
        mcq_freq = None

    if numeric_targets:
        numeric_freq = max(Counter(numeric_targets).values()) / len(numeric_targets)
        _, freq_mra = find_optimal_constant_mra(np.array(numeric_targets, dtype=float))
    else:
        numeric_freq = None
        freq_mra = None

    text_freq = (
        max(Counter(text_answers).values()) / len(text_answers) if text_answers else None
    )

    return {
        "n_items": n,
        "n_mcq": len(mcq_answers),
        "n_numeric": len(numeric_targets),
        "n_text": len(text_answers),
        "mcq_random_accuracy": mcq_random,
        "mcq_frequency_accuracy": mcq_freq,
        "numeric_frequency_accuracy": numeric_freq,
        "text_frequency_accuracy": text_freq,
        "random_mra": None,
        "frequency_mra": freq_mra,
    }


def compute_chance_levels(qa_path):
    data = json.loads(Path(qa_path).read_text())
    tasks_data = data["data"]
    return {task_name: compute_task_chance(items) for task_name, items in tasks_data.items()}


def main():
    parser = argparse.ArgumentParser(description="Compute chance-level baselines for ytb benchmark")
    parser.add_argument("qa_path", help="Path to merged_adv_qa.json")
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
