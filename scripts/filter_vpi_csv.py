#!/usr/bin/env python3
"""Filter a VPI CSV to keep only rows whose last column is empty or FALSE."""
import argparse
import csv
from pathlib import Path


def filter_csv(src: Path, dst: Path) -> tuple[int, int]:
    with src.open(newline="") as f:
        rows = list(csv.reader(f))

    kept: list[list[str]] = []
    for row in rows:
        if not row:
            continue
        last = row[-1].strip().upper()
        if last not in ("", "FALSE"):
            continue
        kept.append(row)

    with dst.open("w", newline="") as f:
        csv.writer(f).writerows(kept)

    return len(rows), len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "src",
        nargs="?",
        default="sheets/vpi_redesign_clean_1.csv",
        type=Path,
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    dst = args.output or args.src.with_name(f"{args.src.stem}_filtered.csv")
    total, kept = filter_csv(args.src, dst)
    print(f"Read {total} rows; kept {kept} -> {dst}")


if __name__ == "__main__":
    main()
