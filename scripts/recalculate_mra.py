"""Recalculate MRA scores for all ytb open_source results using Willis's per_state_accuracy.py.

For each (model, max_frames) combination, picks the most complete run (most JSONL entries),
runs per_state_accuracy.py, and saves per-model TSV + a combined summary.
"""

import os
import subprocess
import sys
from pathlib import Path

RESULTS_ROOT = Path("/nas2/longvideo_eval/longvid-reasoning-eval/results/ytb/open_source")
OUTPUT_ROOT = Path("/nas2/longvideo_eval/longvid-reasoning-eval/results/ytb/open_source_recalculate")
PER_STATE_SCRIPT = "/nas2/willis/longvid-reasoning-eval/scripts/per_state_accuracy.py"
PYTHON = "/nas2/edwin/miniconda/envs/lmms_eval/bin/python"


def count_jsonl_entries(d: Path) -> int:
    count = 0
    for f in d.glob("samples_*.jsonl"):
        with open(f) as fp:
            count += sum(1 for line in fp if line.strip())
    return count


def find_best_run(model_dir: Path) -> Path | None:
    """Find the run directory with the most JSONL entries (most complete run)."""
    candidates = []
    for root, dirs, files in os.walk(model_dir):
        if any(f.startswith("samples_") and f.endswith(".jsonl") for f in files):
            candidates.append(Path(root))
    if not candidates:
        return None
    return max(candidates, key=count_jsonl_entries)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined_tsv = OUTPUT_ROOT / "all_results.tsv"

    # Clear combined TSV
    if combined_tsv.exists():
        combined_tsv.unlink()

    frame_dirs = sorted(RESULTS_ROOT.glob("max_frames_*"))
    if not frame_dirs:
        print("No max_frames_* dirs found", file=sys.stderr)
        sys.exit(1)

    # Collect all (max_frames, model) pairs
    runs = []
    for fd in frame_dirs:
        frames = fd.name.split("_")[-1]
        for model_dir in sorted(fd.iterdir()):
            if not model_dir.is_dir():
                continue
            best_run = find_best_run(model_dir)
            if best_run:
                n = count_jsonl_entries(best_run)
                runs.append((frames, model_dir.name, best_run, n))

    print(f"Found {len(runs)} (model, max_frames) combinations\n")

    # Run per_state_accuracy.py on each
    all_outputs = []
    for frames, model, samples_dir, n in runs:
        print(f"--- {model} @ {frames}f (n={n}) ---")

        out_dir = OUTPUT_ROOT / f"max_frames_{frames}" / model
        out_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = out_dir / "per_state.tsv"
        txt_path = out_dir / "full_output.txt"

        cmd = [
            PYTHON, PER_STATE_SCRIPT,
            "--results-dir", str(samples_dir),
            "--fuzzy",
            "--tsv-out", str(tsv_path),
            "--tsv-group", f"max_frames_{frames}",
            "--tsv-model", model,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Save full output
        with open(txt_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

        # Append to combined TSV
        if tsv_path.exists():
            with open(tsv_path) as f:
                lines = f.readlines()
            with open(combined_tsv, "a") as f:
                if combined_tsv.stat().st_size == 0 and len(lines) >= 2:
                    # Write header on first entry
                    f.write(lines[0])
                    f.write(lines[1])
                if len(lines) >= 3:
                    f.write(lines[2])

        # Parse and display key metrics
        overall_line = None
        for line in result.stdout.splitlines():
            if line.strip().startswith("overall"):
                overall_line = line
                break
        if overall_line:
            print(f"  {overall_line.strip()}")
            all_outputs.append((frames, model, overall_line.strip()))
        else:
            print(f"  [NO RESULTS]")
            if result.returncode != 0:
                print(f"  stderr: {result.stderr[:200]}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Model':<25} {'Frames':>6} {'N':>5} {'ACC':>10} {'SCORE':>10} {'MRA_NUM':>10}")
    print("-" * 100)
    for frames, model, line in all_outputs:
        parts = line.split()
        try:
            n = parts[1]
            acc = parts[2]
            score = parts[3]
            mra_num = parts[4] if len(parts) > 4 else "n/a"
            print(f"{model:<25} {frames:>6} {n:>5} {acc:>10} {score:>10} {mra_num:>10}")
        except (IndexError, ValueError):
            print(f"{model:<25} {frames:>6} {line}")

    print(f"\nResults saved to: {OUTPUT_ROOT}")
    print(f"Combined TSV: {combined_tsv}")


if __name__ == "__main__":
    main()
