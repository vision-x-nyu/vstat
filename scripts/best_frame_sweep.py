"""Find the best max_frames setting for each model using per_state_accuracy.py.

Iterates over all (model, max_frames) combinations in the ytb/open_source results
directory, runs the configured per_state_accuracy.py on each, and reports the best
overall SCORE (mra_with_mcq) for each model along with which max_frames was picked.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results/ytb/open_source")
PER_STATE_SCRIPT = os.environ.get(
    "PER_STATE_SCRIPT",
    str(Path(__file__).with_name("per_state_accuracy.py")),
)
PYTHON = os.environ.get("PYTHON", sys.executable)


def find_samples_dir(model_dir: Path) -> str | None:
    """Walk into model_dir to find the directory containing samples_*.jsonl."""
    for root, dirs, files in os.walk(model_dir):
        if any(f.startswith("samples_") and f.endswith(".jsonl") for f in files):
            return root
    return None


def parse_overall_score(output: str) -> dict | None:
    """Extract overall metrics from per_state_accuracy.py stdout."""
    in_overall = False
    for line in output.splitlines():
        if "=== overall ===" in line:
            in_overall = True
            continue
        if in_overall and line.strip().startswith("overall"):
            parts = line.split()
            # key, n, is_correct%, mra_w/mcq, mra_num, n_num
            # e.g.: overall   1680  72.32%       0.7145       0.6821    420
            try:
                n = int(parts[1])
                acc_str = parts[2].replace("%", "")
                acc = float(acc_str)
                score = float(parts[3])
                return {"n": n, "acc": acc, "score": score}
            except (IndexError, ValueError):
                return None
        if in_overall and line.startswith("===") and "overall" not in line:
            break
    return None


def run_per_state(samples_dir: str, fuzzy: bool = True) -> dict | None:
    """Run per_state_accuracy.py and return parsed overall metrics."""
    cmd = [PYTHON, PER_STATE_SCRIPT, "--results-dir", samples_dir]
    if fuzzy:
        cmd.append("--fuzzy")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return parse_overall_score(result.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default=RESULTS_ROOT)
    ap.add_argument("--metric", default="score", choices=["acc", "score"],
                    help="Metric to pick best by (default: score = mra_with_mcq)")
    ap.add_argument("--no-fuzzy", action="store_true",
                    help="Disable fuzzy question matching")
    args = ap.parse_args()

    root = Path(args.results_root)
    frame_dirs = sorted(root.glob("max_frames_*"))
    if not frame_dirs:
        print(f"No max_frames_* dirs found in {root}", file=sys.stderr)
        sys.exit(1)

    # Collect all models across all frame settings
    all_models = set()
    for fd in frame_dirs:
        for m in fd.iterdir():
            if m.is_dir():
                all_models.add(m.name)
    all_models = sorted(all_models)

    # For each model, evaluate all frame settings
    results = {}  # model -> list of (max_frames, metrics)
    for model in all_models:
        results[model] = []
        for fd in frame_dirs:
            frames = int(fd.name.split("_")[-1])
            model_dir = fd / model
            if not model_dir.exists():
                continue
            samples_dir = find_samples_dir(model_dir)
            if not samples_dir:
                print(f"  [SKIP] {model} @ {frames}f: no samples found", file=sys.stderr)
                continue
            metrics = run_per_state(samples_dir, fuzzy=not args.no_fuzzy)
            if metrics:
                results[model].append((frames, metrics))
                print(f"  {model} @ {frames}f: acc={metrics['acc']:.2f}% score={metrics['score']:.4f} (n={metrics['n']})",
                      file=sys.stderr)
            else:
                print(f"  [FAIL] {model} @ {frames}f: script returned no results", file=sys.stderr)

    # Pick best for each model
    print("\n" + "=" * 80)
    print(f"{'Model':<30} {'Best Frames':>12} {'ACC':>10} {'SCORE':>10} {'N':>6}   All Frames Tested")
    print("-" * 80)
    for model in all_models:
        entries = results[model]
        if not entries:
            print(f"{model:<30} {'N/A':>12}")
            continue
        best = max(entries, key=lambda x: x[1][args.metric])
        all_frames_str = ", ".join(
            f"{f}f={'*' if f == best[0] else ''}{e[args.metric]:.4f}"
            for f, e in sorted(entries, key=lambda x: x[0])
        )
        print(f"{model:<30} {best[0]:>10}f  {best[1]['acc']:>9.2f}% {best[1]['score']:>9.4f} {best[1]['n']:>6}   {all_frames_str}")


if __name__ == "__main__":
    main()
