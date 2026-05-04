"""Build a filtered merged_qa/all.json for the recorded benchmark.

Same as `build_recorded.py`, but cross-checks each (clip, question) pair
against a CSV review sheet and only keeps entries whose flag column is
empty or "FALSE". Entries marked "TRUE" are dropped; entries absent from
the CSV are dropped with a warning.

Configuration (in priority order):
  1. --base CLI argument
  2. $VPI_RECORDED_BASE environment variable
  3. Default: /nas2/benchmarks/vpi/recorded

Other flags:
  --csv  Path to the review sheet
         (default: <repo>/sheets/vpi_redesign_clean_3_updateQ.csv).
  --out  Output JSON path
         (default: <base>/merged_qa/all_filtered.json).

CSV layout (mirrors build_ytb_filtered.py):
  - Rows 1-3 are header / annotator-comment rows and are skipped.
  - Column 0: question text (compared against the raw question from the
    builder, before any MCQ choice block is appended).
  - Column 1: video_path under /nas2/benchmarks/vpi/recorded/...
  - Column 10: "TRUE" / "FALSE" / "" keep-or-drop flag (clean_3 layout).

Match key: `(<single_or_multi_state>/<task>/<clip>.mp4, normalized question)`.
We strip the recorded-root prefix so a different `--video-root` on either
side still lines up.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import build_recorded as br

DEFAULT_BASE = br.DEFAULT_BASE
DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "sheets", "vpi_redesign_clean_3_updateQ.csv",
)
_CSV_HEADER_ROWS = 3
_CSV_FLAG_COL = 10  # clean_3 column

# Path prefix the review CSV uses for recorded clips. Stripped to
# `<single_state|multi_state>/<task>/<clip>.mp4` for matching.
_RECORDED_PREFIX = "/nas2/benchmarks/vpi/recorded/"

# format_mcq() in build_recorded appends "\n\n(A) ...". Strip from that
# point so we can compare against the raw question stem in the CSV.
_MCQ_TAIL_RE = re.compile(r"\s*\n\s*\(A\)", re.IGNORECASE)


def _norm_q(s):
    s = (s or "").strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    s = s.replace("\\'", "'").replace('\\"', '"')
    m = _MCQ_TAIL_RE.search(s)
    if m:
        s = s[: m.start()].rstrip()
    return s


def load_review_csv(csv_path):
    """Return (review, csv_skipped). review is dict[rel_path] -> [(norm_q, flag)].
    rel_path is the CSV's video_path with the recorded prefix stripped."""
    review = defaultdict(list)
    skipped = 0
    with open(csv_path, newline="") as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if i < _CSV_HEADER_ROWS:
                continue
            if len(row) <= _CSV_FLAG_COL:
                continue
            path = row[1]
            if not path.startswith(_RECORDED_PREFIX):
                skipped += 1
                continue
            rel = path[len(_RECORDED_PREFIX):]
            review[rel].append((_norm_q(row[0]), row[_CSV_FLAG_COL].strip().upper()))
    return dict(review), skipped


def lookup_review(review, rel_path, question):
    candidates = review.get(rel_path)
    if not candidates:
        return None, None, "missing"
    qn = _norm_q(question)
    for q_csv, flag in candidates:
        if q_csv == qn:
            return flag, q_csv, "exact"
    return None, None, "missing"


def closest_csv_question(review, rel_path, question):
    candidates = review.get(rel_path)
    if not candidates:
        return None
    qn = _norm_q(question)
    best = min(candidates, key=lambda c: br.levenshtein(c[0], qn))
    return best[0], br.levenshtein(best[0], qn), best[1]


def _video_path_to_rel(video_root, video_path):
    """Strip <video_root> (or the canonical recorded prefix) so the result
    looks like 'single_state/book/0001.mp4'."""
    p = video_path
    for prefix in (video_root.rstrip("/") + "/", _RECORDED_PREFIX):
        if p.startswith(prefix):
            return p[len(prefix):]
    # Fall back to last 3 components (single_or_multi/task/file.mp4).
    parts = p.split(os.sep)
    return os.path.join(*parts[-3:])


def filter_data(data, review, video_root):
    """Drop entries marked TRUE in the CSV, or absent from the CSV.

    Returns (filtered_data, stats) where stats is a dict of counters and
    diagnostics for the summary at the end of main().
    """
    filtered = {}
    n_kept = 0
    n_dropped_true = 0
    n_dropped_missing = 0
    used_keys = set()      # (rel_path, csv_question) pairs that matched
    missing_keys = []      # (rel_path, build_question) pairs absent from CSV
    fuzzy_log = []         # (dist, rel_path, build_q, csv_q, csv_flag)

    for sub_task, entries in data.items():
        kept = []
        for entry in entries:
            rel = _video_path_to_rel(video_root, entry["video_path"])
            raw_q = _norm_q(entry["question"])
            flag, csv_q, kind = lookup_review(review, rel, raw_q)
            if kind == "missing":
                neighbour = closest_csv_question(review, rel, raw_q)
                if neighbour is not None and neighbour[2] == "TRUE":
                    n_dropped_true += 1
                    continue
                missing_keys.append((rel, raw_q))
                n_dropped_missing += 1
                if neighbour is not None:
                    csv_q_near, dist, csv_flag_near = neighbour
                    fuzzy_log.append((dist, rel, raw_q, csv_q_near, csv_flag_near))
                continue
            used_keys.add((rel, csv_q))
            if flag == "TRUE":
                n_dropped_true += 1
                continue
            kept.append(entry)
            n_kept += 1
        filtered[sub_task] = kept

    return filtered, {
        "n_kept": n_kept,
        "n_dropped_true": n_dropped_true,
        "n_dropped_missing": n_dropped_missing,
        "used_keys": used_keys,
        "missing_keys": missing_keys,
        "fuzzy_log": fuzzy_log,
    }


def _build_unfiltered(base, video_root):
    """Run all build_recorded.* builders and return the raw `data` dict."""
    def lt(rel):
        return br.load_task(base, video_root, rel)

    data = {}
    data.update(br.build_simple_numerical(lt("single_state/book"), "book"))
    data.update(br.build_tilt_box(lt("single_state/tilt_box")))
    data.update(br.build_shell_game(lt("single_state/shell_game")))
    data.update(br.build_simple_mcq(
        lt("multi_state/keyboard"), "keyboard",
        lambda correct, _pool: br.nearest_by_edit(correct, br.english_word_pool(correct), 3),
    ))
    data.update(br.build_simple_mcq(
        lt("multi_state/morse"), "morse",
        lambda correct, _pool: br.nearest_by_edit(correct, br.english_word_pool(correct), 3),
    ))
    data.update(br.build_numberpad(lt("multi_state/numberpad")))
    data.update(br.build_cup_stacking(lt("multi_state/cup_stacking")))
    data.update(br.build_packing_order(lt("multi_state/packing_order")))
    data.update(br.build_showing_card(lt("multi_state/showing_card")))
    return data


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--base",
        default=os.environ.get("VPI_RECORDED_BASE", DEFAULT_BASE),
        help="Root of the recorded benchmark (containing single_state/, "
             "multi_state/). Overrides $VPI_RECORDED_BASE. "
             "Default: %(default)s",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: <base>/merged_qa/all_filtered.json)",
    )
    ap.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Review CSV with the keep/drop flag in column %d. "
             "Default: %%(default)s" % _CSV_FLAG_COL,
    )
    ap.add_argument(
        "--video-root",
        default=None,
        help="Root to prefix video_path with (default: same as --base)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing the output file.",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    base = os.path.abspath(args.base)
    video_root = args.video_root or base
    out_path = args.out or os.path.join(base, "merged_qa", "all_filtered.json")

    if not os.path.isdir(base):
        print(f"ERROR: --base {base} does not exist", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.csv):
        print(f"ERROR: review CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    review, csv_skipped = load_review_csv(args.csv)
    n_review_rows = sum(len(v) for v in review.values())
    print(
        f"loaded {n_review_rows} review rows across {len(review)} clips"
        f" from {args.csv} (skipped {csv_skipped} non-recorded rows)"
    )

    data = _build_unfiltered(base, video_root)
    filtered, stats = filter_data(data, review, video_root)

    if stats["missing_keys"]:
        print(
            f"\nWARNING: {len(stats['missing_keys'])} build (clip,question) "
            f"pair(s) not found in review CSV — these were dropped:",
            file=sys.stderr,
        )
        for rel, q in stats["missing_keys"]:
            print(f"  {rel}  Q: {q!r}", file=sys.stderr)

    all_csv_keys = {(rel, q) for rel, qs in review.items() for q, _ in qs}
    csv_unused = sorted(all_csv_keys - stats["used_keys"])
    if csv_unused:
        print(
            f"\nNOTE: {len(csv_unused)} CSV row(s) did not match any build "
            f"entry (likely renamed/removed clips):",
            file=sys.stderr,
        )
        for rel, q in csv_unused:
            print(f"  {rel}  Q: {q[:120]!r}", file=sys.stderr)

    out = {
        "dataset_name": "longvid-reasoning-eval-recorded",
        "version": "0.1",
        "video_root": video_root,
        "data": filtered,
    }

    if not args.dry_run:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(out, fp, indent=4, ensure_ascii=False)
        print(f"\nwrote {out_path}")

    total = sum(len(v) for v in filtered.values())
    print(
        f"\nfilter: kept={stats['n_kept']}, "
        f"dropped_TRUE={stats['n_dropped_true']}, "
        f"dropped_missing_from_csv={stats['n_dropped_missing']}"
    )
    print(f"sub_tasks: {len(filtered)}, total samples: {total}")
    for k in sorted(filtered):
        print(f"  {k}: {len(filtered[k])}")


if __name__ == "__main__":
    main()
