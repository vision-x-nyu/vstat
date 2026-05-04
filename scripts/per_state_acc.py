"""Compute per-state accuracy for ytb-vids eval results.

Joins per-task `samples_*.jsonl` files in a results directory against the
review CSV (which has `State 1` and `State 2` columns) by `(video_path,
question)`, then reports three metrics broken down by State 1, State 2,
and (State 1, State 2):

  - is_correct:      mean of accuracy.is_correct (0/1)
  - mra_with_mcq:    mean of mra_score for numeric entries; for MCQ
                     entries (mra is null) substitute is_correct (0/1)
  - mra_numeric:     mean of mra_score restricted to numeric entries

Rows whose `Exclude` column is TRUE are skipped.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "sheets", "vpi_redesign_clean_4_updateQ.csv",
)

# Header layout in the redesign CSV:
#   row 0,1: comment/banner rows
#   row 2:   actual column header
#   row 3+:  data
_CSV_HEADER_ROWS = 3
_COL_QUESTION = 0
_COL_VIDEO = 1
_COL_TASK = 2
_COL_STATE1 = 4
_COL_STATE2 = 5
_COL_EXCLUDE = 10

# Optional task-level state overrides. Tasks listed here skip CSV lookup
# entirely during JSONL processing and use the preset (State 1, State 2).
TASK_STATE_OVERRIDES = {
    "shell_game_rotate": ("Location", "Atmoic"),
    "shuffle_puzzle": ("Location", "Atmoic"),
    "tilt_v2": ("Location", "Atmoic"),
    "funnel_ball": ("Location", "Dict"),
    "tighten_untighten": ("Count", "Atmoic"),
    "make_coffee": ("Attribute", "Set"),
    "dice": ("Count", "Atmoic"),
    "block_counting": ("Count", "Atmoic"),
}

_MCQ_TAIL_RE = re.compile(r"\s*\n\s*\(A\)", re.IGNORECASE)
_QUESTION_RE = re.compile(r"Question:\s*(.+?)(?:\n\n|\Z)", re.DOTALL)
_REDACTED_RE = re.compile(r"_redacted(?=\.mp4$)")


def _norm_path(vp: str) -> str:
    return _REDACTED_RE.sub("", vp.strip())


def _norm_q(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    s = s.replace("\\'", "'").replace('\\"', '"')
    m = _MCQ_TAIL_RE.search(s)
    if m:
        s = s[: m.start()].rstrip()
    return s


def extract_question(input_text: str) -> str:
    """Pull the question stem out of the JSONL `input` field."""
    m = _QUESTION_RE.search(input_text)
    if not m:
        return ""
    stem = m.group(1).strip()
    # MCQ inputs include "(A) ..." choices after the stem; strip them.
    m2 = _MCQ_TAIL_RE.search(stem)
    if m2:
        stem = stem[: m2.start()].rstrip()
    return stem


def load_csv_states(csv_path: str):
    """Return (states, by_path, excluded).

    `states` is dict[(video_path, normalized_question)] -> (state1, state2).
    `by_path` is dict[video_path] -> list[normalized_question] (for fuzzy
    fallback matching, in CSV row order).
    Rows with Exclude=TRUE are skipped.
    """
    states = {}
    by_path: dict[str, list[str]] = defaultdict(list)
    excluded = 0
    with open(csv_path, newline="") as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if i < _CSV_HEADER_ROWS:
                continue
            if len(row) <= _COL_EXCLUDE:
                continue
            if row[_COL_EXCLUDE].strip().upper() == "TRUE":
                excluded += 1
                continue
            vp = _norm_path(row[_COL_VIDEO])
            q = _norm_q(row[_COL_QUESTION])
            if not vp or not q:
                continue
            task = row[_COL_TASK].strip()
            s1 = row[_COL_STATE1].strip()
            s2 = row[_COL_STATE2].strip()
            if task in TASK_STATE_OVERRIDES:
                s1, s2 = TASK_STATE_OVERRIDES[task]
            states[(vp, q)] = (s1, s2)
            by_path[vp].append(q)
    return states, dict(by_path), excluded


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[j - 1] + 1, prev[j] + 1,
                            prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def closest_question(by_path, vp, q):
    """Return (best_csv_q, distance, kind) on the same video_path, or None.

    Tries a prefix relation first (CSV question starts with JSONL stem or
    vice versa) — common when the CSV adds a trailing clarification —
    and falls back to nearest by edit distance otherwise.
    """
    candidates = by_path.get(vp)
    if not candidates:
        return None
    qn = q.rstrip("?. ")
    prefix_hits = []
    for c in candidates:
        cn = c.rstrip("?. ")
        if cn.startswith(qn) or qn.startswith(cn):
            prefix_hits.append((c, abs(len(c) - len(q))))
    if prefix_hits:
        c, d = min(prefix_hits, key=lambda x: x[1])
        return c, d, "prefix"
    best = min(candidates, key=lambda c: levenshtein(c, q))
    return best, levenshtein(best, q), "edit"


def iter_jsonl(results_dir: str):
    for f in sorted(glob.glob(os.path.join(results_dir, "samples_*.jsonl"))):
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                yield f, json.loads(line)


def task_from_samples_path(path: str) -> str:
    name = os.path.basename(path)
    if name.startswith("samples_"):
        name = name[len("samples_"):]
    if name.endswith(".jsonl"):
        name = name[:-len(".jsonl")]
    for prefix in (
        "longvid-reasoning-eval_ytb_stretch_0p2s_",
        "longvid-reasoning-eval_ytb_",
        "longvid-reasoning-eval_recorded_",
    ):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def is_numeric_target(target) -> bool:
    s = str(target).strip()
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", s))


def fmt_pct(num, den):
    if den == 0:
        return "    n/a"
    return f"{100.0 * num / den:6.2f}%"


# Canonical state ordering / display names for TSV output.
# Map "lowercased CSV value" -> display label.
TSV_STATE1 = [("count", "Count"), ("location", "Location"), ("attribute", "Attribute")]
TSV_STATE2 = [("atmoic", "Atomic"), ("sequence", "Sequence"),
              ("set", "Set"), ("dict", "Dict")]


def _bucket_metrics(b):
    """Return (acc, score, num) as percent floats or None."""
    n = b["n"]
    n_num = b["n_num"]
    if n == 0:
        return (None, None, None)
    acc = 100.0 * b["is_correct_sum"] / n
    score = 100.0 * b["mra_with_mcq_sum"] / n
    num = 100.0 * b["mra_num_sum"] / n_num if n_num else None
    return (acc, score, num)


def _fmt_cell(v):
    return "" if v is None else f"{v:.4f}"


def write_tsv(tsv_path, group, model, overall, by_s1, by_s2):
    """Write a single-row TSV.

    Columns: Group, Model,
             AVG ACC/SCORE/NUM,
             {Count, Location, Attribute} ACC/SCORE/NUM (State 1),
             {Atomic, Sequence, Set, Dict} ACC/SCORE/NUM (State 2).
    """
    h1 = ["", "", "AVG", "", ""]
    h2 = ["Group", "Model", "ACC", "SCORE", "NUM"]
    for _, label in TSV_STATE1 + TSV_STATE2:
        h1.extend([label, "", ""])
        h2.extend(["ACC", "SCORE", "NUM"])

    cells = [group, model]
    cells.extend(_fmt_cell(v) for v in _bucket_metrics(overall["overall"]))
    for key, _ in TSV_STATE1:
        bucket = next(
            (b for k, b in by_s1.items() if k.lower() == key),
            None,
        )
        if bucket is None:
            cells.extend(["", "", ""])
        else:
            cells.extend(_fmt_cell(v) for v in _bucket_metrics(bucket))
    for key, _ in TSV_STATE2:
        bucket = next(
            (b for k, b in by_s2.items() if k.lower() == key),
            None,
        )
        if bucket is None:
            cells.extend(["", "", ""])
        else:
            cells.extend(_fmt_cell(v) for v in _bucket_metrics(bucket))

    with open(tsv_path, "w") as fp:
        fp.write("\t".join(h1) + "\n")
        fp.write("\t".join(h2) + "\n")
        fp.write("\t".join(cells) + "\n")


def report(title: str, buckets: dict):
    print(f"\n=== {title} ===")
    print(f"{'key':<30} {'n':>5} {'is_correct':>12} {'mra_w/mcq':>12} {'mra_num':>12} {'n_num':>6}")
    for key in sorted(buckets):
        b = buckets[key]
        n = b["n"]
        n_num = b["n_num"]
        ic = fmt_pct(b["is_correct_sum"], n)
        mra_all = b["mra_with_mcq_sum"] / n if n else 0
        mra_num = b["mra_num_sum"] / n_num if n_num else None
        mra_all_s = f"{mra_all:.4f}" if n else "n/a"
        mra_num_s = f"{mra_num:.4f}" if mra_num is not None else "n/a"
        print(f"{str(key):<30} {n:>5} {ic:>12} {mra_all_s:>12} {mra_num_s:>12} {n_num:>6}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--results-dir",
        default="/nas2/willis/longvid-reasoning-eval/results/longvid-reasoning-eval_ytb/"
                "gemini31_pro/gemini-3.1-pro-preview/snapshot_4",
        help="Directory containing samples_*.jsonl files.",
    )
    ap.add_argument("--csv", default=DEFAULT_CSV, help="Review CSV path.")
    ap.add_argument(
        "--unmatched-out",
        default=None,
        help="Optional path to write unmatched (video_path, question) pairs as JSON.",
    )
    ap.add_argument(
        "--fuzzy",
        action="store_true",
        help="Fall back to nearest CSV question on the same video_path "
             "when no exact match is found.",
    )
    ap.add_argument(
        "--fuzzy-max-dist",
        type=int,
        default=None,
        help="Maximum Levenshtein distance to accept a fuzzy match "
             "(default: no cap).",
    )
    ap.add_argument(
        "--tsv-out",
        default=None,
        help="If set, append a per-state TSV row to this file. Header is "
             "written when the file is empty/new.",
    )
    ap.add_argument(
        "--tsv-group",
        default="",
        help="Group label for the TSV row (first column).",
    )
    ap.add_argument(
        "--tsv-model",
        default=None,
        help="Model label for the TSV row (second column). Defaults to "
             "the parent dir name of --results-dir.",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    states, by_path, n_excluded = load_csv_states(args.csv)
    print(f"loaded {len(states)} CSV rows ({n_excluded} excluded)")

    def empty_bucket():
        return {"n": 0, "n_num": 0, "is_correct_sum": 0,
                "mra_with_mcq_sum": 0.0, "mra_num_sum": 0.0}

    by_s1 = defaultdict(empty_bucket)
    by_s2 = defaultdict(empty_bucket)
    by_pair = defaultdict(empty_bucket)
    overall = defaultdict(empty_bucket)

    n_seen = n_matched = n_fuzzy = 0
    unmatched = []
    fuzzy_log = []
    for path, e in iter_jsonl(args.results_dir):
        n_seen += 1
        task = task_from_samples_path(path)
        vp = _norm_path(e.get("video_path", "").replace("_stretch_0p2s", ""))
        q = _norm_q(extract_question(e.get("input", "")))
        if task in TASK_STATE_OVERRIDES:
            s1, s2 = TASK_STATE_OVERRIDES[task]
        else:
            key = (vp, q)
            if key not in states:
                if args.fuzzy:
                    # import pdb; pdb.set_trace()
                    best = closest_question(by_path, vp, q)
                    if best is not None and (
                        args.fuzzy_max_dist is None or best[1] <= args.fuzzy_max_dist
                    ):
                        csv_q, dist, kind = best
                        key = (vp, csv_q)
                        n_fuzzy += 1
                        fuzzy_log.append({
                            "file": os.path.basename(path),
                            "video_path": vp,
                            "kind": kind,
                            "distance": dist,
                            "jsonl_question": q,
                            "csv_question": csv_q,
                        })
                if key not in states:
                    unmatched.append({"file": os.path.basename(path),
                                      "video_path": vp, "question": q})
                    continue
            s1, s2 = states[key]
        # Drop entries that have no State 1 or State 2 label.
        if not s1 or not s2:
            unmatched.append({"file": os.path.basename(path),
                              "video_path": vp, "question": q,
                              "reason": "empty_state"})
            continue
        n_matched += 1

        is_correct = bool(e["accuracy"]["is_correct"])
        numeric = is_numeric_target(e.get("target"))
        mra_obj = e.get("mra")
        if numeric and isinstance(mra_obj, dict) and mra_obj.get("mra_score") is not None:
            mra_num = float(mra_obj["mra_score"])
            mra_all = mra_num
        else:
            # MCQ (or numeric without mra) -> count is_correct as 0/1
            mra_num = None
            mra_all = 1.0 if is_correct else 0.0

        for bucket in (by_s1[s1],
                       by_s2[s2],
                       by_pair[(s1, s2)],
                       overall["overall"]):
            bucket["n"] += 1
            bucket["is_correct_sum"] += int(is_correct)
            bucket["mra_with_mcq_sum"] += mra_all
            if mra_num is not None:
                bucket["n_num"] += 1
                bucket["mra_num_sum"] += mra_num

    print(f"jsonl entries: {n_seen}, matched to CSV: {n_matched} "
          f"(fuzzy: {n_fuzzy}), unmatched: {len(unmatched)}")

    if fuzzy_log:
        # prefix matches first (most reliable), then edit-distance ascending
        fuzzy_log.sort(key=lambda x: (x["kind"] != "prefix", x["distance"]))
        print(f"\n=== fuzzy matches ({len(fuzzy_log)}) ===")
        for r in fuzzy_log:
            print(f"  [{r['kind']}] d={r['distance']:3d}  {r['video_path']}")
            print(f"    jsonl: {r['jsonl_question']!r}")
            print(f"    csv  : {r['csv_question']!r}")

    report("overall", overall)
    report("by State 1", by_s1)
    report("by State 2", by_s2)
    report("by (State 1, State 2)", by_pair)

    if args.unmatched_out and unmatched:
        with open(args.unmatched_out, "w") as fp:
            json.dump(unmatched, fp, indent=2, ensure_ascii=False)
        print(f"\nwrote {len(unmatched)} unmatched entries to {args.unmatched_out}",
              file=sys.stderr)

    if args.tsv_out:
        rd = os.path.normpath(args.results_dir)
        model = args.tsv_model or os.path.basename(os.path.dirname(rd))
        write_tsv(args.tsv_out, args.tsv_group, model, overall, by_s1, by_s2)
        print(f"\nwrote TSV to {args.tsv_out}")


if __name__ == "__main__":
    main()
