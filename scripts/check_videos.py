"""Verify every video referenced by the VSTAT QA JSON exists on disk before eval.

Prevents the silent multi-rank hang where one accelerate worker hits an
`assert os.path.exists(...)` in `doc_to_visual` and the other ranks block
forever on the next NCCL collective.

Usage:
    python scripts/check_videos.py
    python scripts/check_videos.py --qa data/vstat/vstat_qa_clean.json
    VSTAT_VIDEO_ROOT=/some/path python scripts/check_videos.py

Exits non-zero (1) if any referenced video is missing or empty.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QA = REPO_ROOT / "data" / "vstat" / "vstat_qa_clean.json"
DEFAULT_VIDEO_ROOT = REPO_ROOT / "data" / "vstat"


def load_docs(qa_path: Path):
    with qa_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    data = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    if isinstance(data, dict):
        for task, docs in data.items():
            for doc in docs:
                yield task, doc
    else:
        for doc in data:
            yield doc.get("source_task", "?"), doc


def resolve(video_path: str, root: Path) -> Path:
    p = Path(video_path).expanduser()
    return p if p.is_absolute() else root / p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=Path(os.environ.get("VSTAT_QA_PATH", DEFAULT_QA)))
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path(os.environ.get("VSTAT_VIDEO_ROOT", DEFAULT_VIDEO_ROOT)),
    )
    parser.add_argument("--max-print", type=int, default=50, help="cap on missing paths to print")
    args = parser.parse_args()

    if not args.qa.exists():
        print(f"[check_videos] QA file not found: {args.qa}", file=sys.stderr)
        return 2

    print(f"[check_videos] qa={args.qa}")
    print(f"[check_videos] video_root={args.video_root}")

    seen: set[Path] = set()
    missing: list[tuple[str, str, Path]] = []  # (task, video_id, abs_path)
    empty: list[tuple[str, str, Path]] = []
    n_docs = 0
    n_unique = 0

    for task, doc in load_docs(args.qa):
        n_docs += 1
        rel = doc.get("video_path")
        if not rel:
            missing.append((task, doc.get("video_id", "?"), Path("<no video_path>")))
            continue
        abs_path = resolve(rel, args.video_root)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        n_unique += 1
        if not abs_path.exists():
            missing.append((task, doc.get("video_id", "?"), abs_path))
        else:
            try:
                if abs_path.stat().st_size == 0:
                    empty.append((task, doc.get("video_id", "?"), abs_path))
            except OSError as e:
                missing.append((task, doc.get("video_id", "?"), abs_path))
                print(f"[check_videos] stat failed for {abs_path}: {e}", file=sys.stderr)

    print(f"[check_videos] docs={n_docs}  unique_videos={n_unique}  missing={len(missing)}  empty={len(empty)}")

    if missing:
        print(f"\n[check_videos] MISSING ({len(missing)}):")
        for task, vid, path in missing[: args.max_print]:
            print(f"  {task:>30s}  {vid:<24s}  {path}")
        if len(missing) > args.max_print:
            print(f"  ... and {len(missing) - args.max_print} more")

    if empty:
        print(f"\n[check_videos] EMPTY ({len(empty)}):")
        for task, vid, path in empty[: args.max_print]:
            print(f"  {task:>30s}  {vid:<24s}  {path}")
        if len(empty) > args.max_print:
            print(f"  ... and {len(empty) - args.max_print} more")

    return 0 if not (missing or empty) else 1


if __name__ == "__main__":
    sys.exit(main())
