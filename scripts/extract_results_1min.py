"""Render 1-minute eval results as a paper-style summary table.

Usage:
    python scripts/extract_results_1min.py
    python scripts/extract_results_1min.py /Users/sihyun/Desktop/Research/projects/NYU/codes/longvid-reasoning-eval/results/1min/model_defaults

Input spec:
    The results root contains per-model subdirectories with the newest
    `*_results.json` files for the 1-minute benchmark. Existing summary files
    named `*_longvid_reasoning_eval_1min_summary.md` are refreshed in place.

Output spec:
    Prints markdown with an embedded HTML table to stdout and writes the
    refreshed summary markdown file or files inside the selected results root.
"""

import sys
from pathlib import Path

from extract_results_1min_table import DEFAULT_RESULTS_ROOT, build_markdown, resolve_output_paths


def resolve_paths():
    results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS_ROOT
    assert results_root.exists(), f"Missing results root: {results_root}"
    assert results_root.is_dir(), f"Results root is not a directory: {results_root}"
    return results_root, resolve_output_paths(results_root)


def main():
    results_root, output_paths = resolve_paths()
    markdown = build_markdown(results_root)
    print(markdown)
    for output_path in output_paths:
        output_path.write_text(markdown + "\n", encoding="utf-8")
        print(f"Written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
