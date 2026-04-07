"""Aggregate lmms-eval v2 JSON under results/data_v2 into benchmark markdown tables.

Usage:
    python scripts/render_data_v2_benchmark_md.py \\
        --data-v2-root results/data_v2 \\
        --output results/data_v2/benchmark_tables.md

    Relative --data-v2-root is resolved against the current working directory first;
    if that path is not a directory, the same relative path under this repo
    (parent of scripts/) is tried so runs from a workspace root still work.

Input spec:
    Recursive *results.json files:
        - .../results.json (lmms-eval layout)
        - .../YYYYMMDD_HHMMSS_results.json (HF-style layout)
    Paths containing the substring "stretch" are skipped unless --include-stretch.
    Each file: top-level "results" dict with keys like
    longvid-reasoning-eval_v2_20sec_hidden_dice_roll and metric accuracy,none (float).

Output spec:
    Markdown file with two tables (numerical then MCQ), fixed column order per constants;
    accuracies as percent with one decimal (e.g. 54.0, 33.3). Missing cells use em dash.
    Also writes benchmark_tables.tsv (same directory as the .md unless --tsv-output is set):
    tab-separated, two blocks (numerical then MCQ) separated by a blank row—paste into Sheets/Excel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from data_v2_benchmark_render import render_markdown
from data_v2_benchmark_tsv import render_tsv


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_data_v2_root(arg: Path) -> Path:
    p = arg.expanduser()
    if p.is_absolute():
        out = p.resolve()
        assert out.is_dir(), f"not a directory: {out}"
        return out
    cwd_guess = (Path.cwd() / p).resolve()
    repo_guess = (_repo_root() / p).resolve()
    if cwd_guess.is_dir():
        return cwd_guess
    if repo_guess.is_dir():
        return repo_guess
    assert False, (
        "not a directory: tried "
        f"{cwd_guess} (cwd-relative) and {repo_guess} (repo-relative to {_repo_root()})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render data_v2 benchmark markdown tables")
    parser.add_argument(
        "--data-v2-root",
        type=Path,
        default=_repo_root() / "results" / "data_v2",
        help="Directory containing per-model result trees (relative paths: try cwd, then repo root)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path (default: <data-v2-root>/benchmark_tables.md)",
    )
    parser.add_argument(
        "--include-stretch",
        action="store_true",
        help="Include result trees whose path contains 'stretch'",
    )
    parser.add_argument(
        "--tsv-output",
        type=Path,
        default=None,
        help="Tab-separated file for spreadsheet paste (default: <md-dir>/benchmark_tables.tsv)",
    )
    parser.add_argument(
        "--no-tsv",
        action="store_true",
        help="Do not write the TSV file",
    )
    args = parser.parse_args()
    root = _resolve_data_v2_root(args.data_v2_root)
    out = args.output if args.output is not None else root / "benchmark_tables.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = render_markdown(root, include_stretch=args.include_stretch)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")
    if not args.no_tsv:
        tsv_path = args.tsv_output if args.tsv_output is not None else out.with_suffix(".tsv")
        tsv_text = render_tsv(root, include_stretch=args.include_stretch)
        tsv_path.write_text(tsv_text, encoding="utf-8")
        print(f"Wrote {tsv_path}")


if __name__ == "__main__":
    main()
