"""Update merged_filtered_redact_qa.json with question/answer/choices from
boxing_volleyball_sequence_options.json, matching by (category, video_id).

Only `question`, `answer`, `answer_index`, and `choices` are overwritten on the
target entry; all other fields are preserved.
"""

import argparse
import json
from pathlib import Path

DEFAULT_SOURCE = Path(
    # "/nas2/benchmarks/vpi/real/merged_qa/boxing_volleyball_sequence_options.json"
    "/nas2/benchmarks/vpi/real/merged_qa/graffiti_options.json"
)
DEFAULT_TARGET = Path(
    "/nas2/benchmarks/vpi/real/merged_qa/vpi_qa_redact.json"
)

UPDATE_FIELDS = ("question", "answer", "answer_index", "choices")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path; defaults to overwriting --target.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing.",
    )
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    target = json.loads(args.target.read_text())

    updated = 0
    missing: list[tuple[str, str]] = []

    for category, src_entries in source["data"].items():
        src_by_id = {e["video_id"]: e for e in src_entries}
        tgt_entries = target["data"].get(category, [])
        tgt_by_id = {e["video_id"]: e for e in tgt_entries}

        for vid, src in src_by_id.items():
            tgt = tgt_by_id.get(vid)
            if tgt is None:
                missing.append((category, vid))
                continue
            for field in UPDATE_FIELDS:
                tgt[field] = src.get(field)
            updated += 1

    print(f"Updated {updated} entries.")
    if missing:
        print(f"Missing {len(missing)} entries in target:")
        for cat, vid in missing:
            print(f"  {cat}/{vid}")

    if args.dry_run:
        print("Dry run; not writing.")
        return

    out_path = args.output or args.target
    out_path.write_text(json.dumps(target, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
