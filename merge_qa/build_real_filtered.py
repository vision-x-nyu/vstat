"""Build a filtered merged_qa JSON for the unified real benchmark.

The real benchmark stores both recorded-style tasks and ytb-style tasks under
one flat tree:

    <base>/raw/<task>/*.json
    <base>/processed/<task>/*.mp4

This builder reuses the existing recorded and ytb conversion logic, then
filters entries against the merged review CSV. Rows marked TRUE in the CSV are
dropped. Rows missing from the CSV are dropped unless --force is set.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import build_recorded as br
import build_ytb_filtered as ytb

DEFAULT_BASE = "/nas2/benchmarks/vpi/real"
DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "sheets", "vpi_merge_5.csv",
)

_CSV_HEADER_ROWS = 3
_CSV_FLAG_COL = 10
_REAL_PREFIX = "/nas2/benchmarks/vpi/real/"
_REAL_PROCESSED_PREFIX = "/nas2/benchmarks/vpi/real/processed/"
_PT1_RE = re.compile(r"^(.+)_pt1\.mp4$")

RECORDED_TASKS = {
    "book",
    "tilt_box",
    "shell_game",
    "keyboard",
    "morse",
    "numberpad",
    "cup_stacking",
    "packing_order",
    "showing_card",
}


def _norm_q(s):
    return ytb._norm_q(s)


def _path_to_review_rel(path):
    path = path.replace("\\", "/")
    for prefix in (_REAL_PROCESSED_PREFIX, _REAL_PREFIX):
        if path.startswith(prefix):
            return path[len(prefix):]
    return None


def load_review_csv(csv_path):
    """Return dict[rel_path] -> [(normalized_question, flag)]."""
    review = defaultdict(list)
    skipped = 0
    with open(csv_path, newline="") as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if i < _CSV_HEADER_ROWS:
                continue
            if len(row) <= _CSV_FLAG_COL:
                continue
            rel = _path_to_review_rel(row[1].strip())
            if rel is None:
                skipped += 1
                continue
            review[rel].append((_norm_q(row[0]), row[_CSV_FLAG_COL].strip().upper()))
    return dict(review), skipped


def _video_path_to_rels(video_root, video_path):
    """Return possible review rel paths for a built video_path."""
    p = video_path.replace("\\", "/")
    rels = []
    for prefix in (video_root.rstrip("/") + "/", _REAL_PROCESSED_PREFIX, _REAL_PREFIX):
        if p.startswith(prefix):
            rels.append(p[len(prefix):])
            break
    if not rels:
        parts = p.split("/")
        if len(parts) >= 2:
            rels.append("/".join(parts[-2:]))

    # The CSV uses book/0001.mp4 for recorded-style clips, while processed
    # videos are stored as book/0001_pt1.mp4.
    extra = []
    for rel in rels:
        task, _, name = rel.rpartition("/")
        if task in RECORDED_TASKS:
            m = _PT1_RE.match(name)
            if m:
                extra.append(f"{task}/{m.group(1)}.mp4")
    return rels + [rel for rel in extra if rel not in rels]


def lookup_review(review, rel_paths, question):
    qn = _norm_q(question)
    for rel in rel_paths:
        for q_csv, flag in review.get(rel, []):
            if q_csv == qn:
                return flag, rel, q_csv, "exact"
    return None, rel_paths[0] if rel_paths else "", None, "missing"


def closest_csv_question(review, rel_paths, question):
    qn = _norm_q(question)
    candidates = []
    for rel in rel_paths:
        for q_csv, flag in review.get(rel, []):
            candidates.append((rel, q_csv, flag))
    if not candidates:
        return None
    rel, q_csv, flag = min(candidates, key=lambda c: ytb.levenshtein(c[1], qn))
    return rel, q_csv, ytb.levenshtein(q_csv, qn), flag


def _load_real_task(base, video_root, task):
    items = []
    raw_dir = os.path.join(base, "raw", task)
    processed_dir = os.path.join(video_root, task)
    if not os.path.isdir(raw_dir):
        return items
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".json"):
            continue
        fid = os.path.splitext(name)[0]
        raw_path = os.path.join(raw_dir, name)
        with open(raw_path) as fp:
            data = json.load(fp)
        clip_name = f"{fid}_pt1.mp4"
        video_path = os.path.join(processed_dir, clip_name)
        items.append({"video_id": os.path.splitext(clip_name)[0], "video_path": video_path, "data": data})
    return items


def build_recorded_data(base, video_root):
    def lt(task):
        return _load_real_task(base, video_root, task)

    data = {}
    data.update(br.build_simple_numerical(lt("book"), "book"))
    data.update(br.build_tilt_box(lt("tilt_box")))
    data.update(br.build_shell_game(lt("shell_game")))
    data.update(br.build_simple_mcq(
        lt("keyboard"), "keyboard",
        lambda correct, _pool: br.nearest_by_edit(correct, br.english_word_pool(correct), 3),
    ))
    data.update(br.build_simple_mcq(
        lt("morse"), "morse",
        lambda correct, _pool: br.nearest_by_edit(correct, br.english_word_pool(correct), 3),
    ))
    data.update(br.build_numberpad(lt("numberpad")))
    data.update(br.build_cup_stacking(lt("cup_stacking")))
    data.update(br.build_packing_order(lt("packing_order")))
    data.update(br.build_showing_card(lt("showing_card")))
    return data


def build_ytb_data(base, video_root):
    raw_root = os.path.join(base, "raw")
    proc_root = os.path.join(base, "processed")
    data = defaultdict(list)
    raw_cache = {}

    for task, clip_file, fid, idx in ytb.iter_clips(proc_root, raw_root):
        if task in RECORDED_TASKS or task in ytb.EXCLUDED_TASKS:
            continue
        if task not in raw_cache:
            raw_cache[task] = ytb.load_raw_qa(raw_root, task)
        raw_entry = raw_cache[task].get(fid)
        if raw_entry is None:
            continue
        matched = ytb.match_questions(raw_entry, idx)
        if not matched:
            continue

        video_id_base = clip_file.replace(".mp4", "")
        video_path = os.path.join(video_root, task, clip_file)
        for qi, q in enumerate(matched):
            raw_answer = q.get("answer", "")
            if q.get("question", "") == "What is the number being drawn on the wall?":
                _, norm = ytb.normalize_answer(raw_answer)
                ans_type = "open_text"
            else:
                ans_type, norm = ytb.normalize_answer(raw_answer)
            video_id = video_id_base if len(matched) == 1 else f"{video_id_base}_q{qi}"
            if (task, video_id) in ytb.EXCLUDED_ENTRIES:
                continue
            raw_question = q.get("question", "")
            entry = ytb.build_entry(
                video_id=video_id,
                video_path=video_path,
                question_text=raw_question,
                ans_type=ans_type,
                norm=norm,
                seed_key=f"{task}:{video_id}",
                task_key=task,
                raw_answer=raw_answer,
                raw_entry=raw_entry,
            )
            data[task].append(entry)
    return dict(data)


def filter_data(data, review, video_root, force=False):
    filtered = {}
    stats = {
        "n_kept": 0,
        "n_dropped_true": 0,
        "n_dropped_missing": 0,
        "n_forced_missing": 0,
        "used_keys": set(),
        "missing_keys": [],
        "fuzzy_log": [],
    }

    for task, entries in data.items():
        kept = []
        for entry in entries:
            rels = _video_path_to_rels(video_root, entry["video_path"])
            raw_q = _norm_q(entry["question"])
            flag, rel, csv_q, kind = lookup_review(review, rels, raw_q)
            if kind == "missing":
                neighbour = closest_csv_question(review, rels, raw_q)
                if not force and neighbour is not None and neighbour[3] == "TRUE":
                    stats["n_dropped_true"] += 1
                    continue
                stats["missing_keys"].append((rel, raw_q))
                if force:
                    stats["n_forced_missing"] += 1
                else:
                    stats["n_dropped_missing"] += 1
                if neighbour is not None:
                    near_rel, csv_q_near, dist, csv_flag_near = neighbour
                    stats["fuzzy_log"].append((dist, near_rel, raw_q, csv_q_near, csv_flag_near))
                if not force:
                    continue
            else:
                stats["used_keys"].add((rel, csv_q))
                if flag == "TRUE":
                    stats["n_dropped_true"] += 1
                    continue
            entry["source_task"] = task
            kept.append(entry)
            stats["n_kept"] += 1
        filtered[task] = kept
    return filtered, stats


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--base",
        default=os.environ.get("VPI_REAL_BASE", DEFAULT_BASE),
        help="Root of the real benchmark (containing raw/ and processed/). Default: %(default)s",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: <base>/merged_qa/real_filtered.json)",
    )
    ap.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Merged review CSV with the keep/drop flag in column %d. Default: %%(default)s" % _CSV_FLAG_COL,
    )
    ap.add_argument(
        "--video-root",
        default=None,
        help="Root to prefix video_path with (default: <base>/processed)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Keep build entries that are not found in the review CSV.",
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
    video_root = os.path.abspath(args.video_root) if args.video_root else os.path.join(base, "processed")
    out_path = args.out or os.path.join(base, "merged_qa", "real_filtered.json")

    raw_root = os.path.join(base, "raw")
    proc_root = os.path.join(base, "processed")
    if not os.path.isdir(raw_root) or not os.path.isdir(proc_root):
        print(f"ERROR: expected raw/ and processed/ under {base}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.csv):
        print(f"ERROR: review CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    review, csv_skipped = load_review_csv(args.csv)
    n_review_rows = sum(len(v) for v in review.values())
    print(
        f"loaded {n_review_rows} review rows across {len(review)} clips"
        f" from {args.csv} (skipped {csv_skipped} non-real rows)"
    )

    data = build_recorded_data(base, video_root)
    for task, entries in build_ytb_data(base, video_root).items():
        data[task] = entries

    filtered, stats = filter_data(data, review, video_root, force=args.force)

    if stats["missing_keys"]:
        missing_action = "kept because --force was set" if args.force else "dropped"
        print(
            f"\nWARNING: {len(stats['missing_keys'])} build (clip,question) pair(s) "
            f"not found in review CSV - these were {missing_action}:",
            file=sys.stderr,
        )
        for rel, q in stats["missing_keys"]:
            print(f"  {rel}  Q: {q!r}", file=sys.stderr)

    all_csv_keys = {(rel, q) for rel, qs in review.items() for q, _ in qs}
    csv_unused = sorted(all_csv_keys - stats["used_keys"])
    if csv_unused:
        print(
            f"\nNOTE: {len(csv_unused)} CSV row(s) did not match any build entry:",
            file=sys.stderr,
        )
        for rel, q in csv_unused:
            print(f"  {rel}  Q: {q!r}", file=sys.stderr)

    out = {
        "dataset_name": "vpi-real",
        "version": "0.1",
        "video_root": video_root,
        "data": {k: filtered[k] for k in sorted(filtered)},
    }
    if not args.dry_run:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(out, fp, indent=2, ensure_ascii=False)
        print(f"\nwrote {out_path}")

    total = sum(len(v) for v in filtered.values())
    print(
        f"\nfilter: kept={stats['n_kept']}, "
        f"dropped_TRUE={stats['n_dropped_true']}, "
        f"dropped_missing_from_csv={stats['n_dropped_missing']}, "
        f"forced_missing_from_csv={stats['n_forced_missing']}"
    )
    print(f"tasks: {len(filtered)}, total samples: {total}")
    for k in sorted(filtered):
        n_mcq = sum(1 for e in filtered[k] if e.get("choices"))
        n_num = sum(
            1 for e in filtered[k]
            if not e.get("choices") and re.fullmatch(r"-?\d+(?:\.\d+)?", str(e.get("answer")))
        )
        n_open = len(filtered[k]) - n_mcq - n_num
        print(f"  {k}: {len(filtered[k])} (num={n_num}, mcq={n_mcq}, open={n_open})")


if __name__ == "__main__":
    main()
