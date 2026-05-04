"""Build a unified VPI merged_qa JSON.

Successor to ``build_real_filtered.py``. Combines:

* The legacy recorded build (``book``, ``cup_stacking``, ``packing_order`` ...)
  via ``build_recorded.py`` helpers — unchanged from ``build_real_filtered``.
* The legacy YTB build via ``build_ytb_filtered`` — also unchanged.
* Nine new tasks under ``<base>/raw/`` (block_counting, dice, shell_game_rotate,
  shuffle_puzzle, tighten_untighten, tilt_v2, funnel_ball, hockey, make_coffee).
  These are routed through the YTB-style pipeline so per-task answer-shape
  overrides registered in ``VPI_OVERRIDES`` (below) can convert their custom
  answer formats (rank lists, duration dicts, score strings ...) into MCQ
  ``(correct, distractors)`` pairs.

``build_real_filtered.py`` is kept on disk as legacy reference; this file is
intended to replace it as the build entrypoint.
"""
import argparse
import csv
import json
import os
import random
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
_REDACTED_RE = re.compile(r"_redacted(?=\.mp4$)")

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

# Tasks added after build_real_filtered.py shipped. They live alongside the
# YTB tasks under raw/<task>/<fid>.json + processed/<task>/<fid>_pt1.mp4 and
# are processed by the YTB pipeline so VPI_OVERRIDES below can shape their
# answers into MCQs. The raw JSONs do not carry per-question ``index`` keys,
# so ``_match_questions`` widens the matcher for these tasks.
BLENDER_TASKS = {
    # "block_counting",
    # "dice",
    # "shell_game_rotate",
    # "shuffle_puzzle",
    # "tighten_untighten",
    # "tilt_v2",
    "funnel_ball",
    "hockey",
    # "make_coffee",
}


# --- override hook ---------------------------------------------------------
# Each override takes a context dict and returns ``(correct_choice, [distractors])``
# or ``None`` to defer.
#
# ctx keys:
#   task_key    -- task directory name (e.g. "funnel_ball")
#   raw_answer  -- ORIGINAL answer from raw JSON (may be list/dict/int/str)
#   raw_entry   -- full raw JSON dict for the clip (use for context like
#                  ``raw_entry["ball_ids"]`` or ``raw_entry["game_results"]``)
#   question    -- original question text
#   video_id    -- e.g. "0001_pt1"
#
# Author overrides for any of the new tasks here. Anything left empty falls
# through to the legacy ytb OVERRIDES, and finally to ``build_entry``'s
# default open-text path (entry has no ``choices`` set).
#
# Suggested overrides to author later:
#   _block_counting_override     (2 question variants per video)
#   _funnel_ball_rank_override
#   _funnel_ball_longest_override
#   _funnel_ball_durations_override
#   _funnel_ball_exit_order_override
#   _hockey_winner_override
#   _hockey_final_score_override
#   _hockey_durations_override
#   _shuffle_puzzle_override

def _shell_game_rotate_override(ctx):
    if ctx["task_key"] != "shell_game_rotate":
        return None
    correct = ctx["raw_answer"].strip().rstrip(".").strip()
    pool = ["Left", "Center", "Right"]
    if correct not in pool:
        return None
    distractors = [p for p in pool if p != correct]
    return correct, distractors


_HOCKEY_WINNER_POOL = ["red", "blue", "tie"]
_HOCKEY_SCORE_RE = re.compile(r"\d+-\d+")


def _hockey_longest_game_override(ctx):
    """Hockey 'which game had the longest duration?' → MCQ over Game 1..N.

    Mirrors ``build_blender_hockey.make_game_choices``. ``_match_questions``
    rewrites raw Q6 (the durations-array question) into this form before the
    override runs.
    """
    if ctx["task_key"] != "hockey":
        return None
    if "Which game had the longest duration" not in ctx["question"]:
        return None
    correct = str(ctx["raw_answer"]).strip().rstrip(".").strip()
    games_played = (ctx.get("raw_entry") or {}).get("games_played")
    if not isinstance(games_played, int) or games_played < 2:
        return None
    pool = [f"Game {i + 1}" for i in range(games_played)]
    if correct not in pool:
        return None
    others = [g for g in pool if g != correct]
    if len(pool) <= 4:
        distractors = others
    else:
        rng = random.Random(f"hockey_longest_game:{ctx.get('video_id', '')}:distractors")
        rng.shuffle(others)
        distractors = others[:3]
    return correct, distractors


def _hockey_final_score_override(ctx):
    """Hockey final score 'R-B' → 4-way MCQ.

    Distractors are other ``r-(games_played - r)`` splits picked deterministically
    from the set summing to ``games_played``. Mirrors
    ``build_blender_hockey.make_score_choices``.
    """
    if ctx["task_key"] != "hockey":
        return None
    if "final score" not in ctx["question"].lower():
        return None
    correct = str(ctx["raw_answer"]).strip().rstrip(".").strip()
    if not _HOCKEY_SCORE_RE.fullmatch(correct):
        return None
    games_played = (ctx.get("raw_entry") or {}).get("games_played")
    if not isinstance(games_played, int) or games_played < 0:
        return None

    rng = random.Random(f"hockey_score:{ctx.get('video_id', '')}:distractors")
    pool = [f"{r}-{games_played - r}" for r in range(games_played + 1) if f"{r}-{games_played - r}" != correct]
    rng.shuffle(pool)
    distractors = pool[:3]
    # Pad with synthetic R-B pairs if there aren't enough on-sum distractors
    # (e.g. games_played < 3).
    while len(distractors) < 3:
        extra = f"{rng.randint(0, games_played + 1)}-{rng.randint(0, games_played + 1)}"
        if extra != correct and extra not in distractors:
            distractors.append(extra)
    return correct, distractors


_SHUFFLE_PUZZLE_CELL_RE = re.compile(r"Row\s+(\d+)\s*,\s*Column\s+(\d+)", re.IGNORECASE)


def _shuffle_puzzle_grid_size(raw_entry):
    """Infer NxN grid side from ``final_grid`` keys (``"r,c"`` format)."""
    grid = (raw_entry or {}).get("final_grid")
    if not isinstance(grid, dict):
        return 3  # canonical question wording is "3x3 board"
    max_idx = -1
    for key in grid:
        try:
            r_str, c_str = str(key).split(",")
            max_idx = max(max_idx, int(r_str), int(c_str))
        except ValueError:
            continue
    return max_idx + 1 if max_idx >= 0 else 3


def _shuffle_puzzle_override(ctx):
    """shuffle_puzzle 'Row x, Column y' tile location → 4-way MCQ.

    Distractors are deterministically chosen single-coordinate flips of the
    correct cell (vary the row OR the column, keeping the other fixed) —
    modeled after ``_sequence_flip_distractors`` in ``build_ytb_filtered.py``.
    For a 3x3 board this yields 2 row-flip + 2 column-flip candidates; we
    pick three after a seeded shuffle.
    """
    if ctx["task_key"] != "shuffle_puzzle":
        return None
    correct = str(ctx["raw_answer"]).strip().rstrip(".").strip()
    m = _SHUFFLE_PUZZLE_CELL_RE.match(correct)
    if m is None:
        return None
    r, c = int(m.group(1)), int(m.group(2))

    n = _shuffle_puzzle_grid_size(ctx.get("raw_entry"))
    if not (1 <= r <= n and 1 <= c <= n):
        return None

    flips = [f"Row {r_alt}, Column {c}" for r_alt in range(1, n + 1) if r_alt != r]
    flips += [f"Row {r}, Column {c_alt}" for c_alt in range(1, n + 1) if c_alt != c]
    if not flips:
        return None

    rng = random.Random(f"shuffle_puzzle:{ctx.get('video_id', '')}:distractors")
    rng.shuffle(flips)
    return correct, flips[:3]


# Cap for the number of choices when there are more candidates than slots.
# 1 correct + (cap - 1) distractors. Mirrors build_blender_funnel_ball.MCQ_MAX.
_FUNNEL_BALL_MCQ_MAX = 4


def _funnel_ball_longest_override(ctx):
    """Override for funnel_ball Q1 ('which ball took the longest to fall').

    Ported from ``build_blender_funnel_ball.py``: build a 4-way (or smaller, if
    num_balls < 4) MCQ over ``ball_ids``, with the correct ball-id as the right
    answer and deterministically-shuffled distractors drawn from the remaining
    ball_ids.
    """
    if ctx["task_key"] != "funnel_ball":
        return None
    if "took the longest time to fall" not in ctx["question"]:
        return None

    raw_entry = ctx.get("raw_entry") or {}
    raw_answer = ctx["raw_answer"]
    ball_ids = raw_entry.get("ball_ids")
    if not ball_ids:
        num_balls = raw_entry.get("num_balls")
        if not isinstance(num_balls, int) or num_balls <= 0:
            return None
        ball_ids = [f"ball_{i + 1}" for i in range(num_balls)]

    correct = str(raw_answer).strip()
    if correct not in ball_ids:
        return None

    others = [b for b in ball_ids if b != correct]
    if len(ball_ids) <= _FUNNEL_BALL_MCQ_MAX:
        distractors = others
    else:
        rng = random.Random(f"funnel_ball_longest:{ctx.get('video_id', '')}:distractors")
        rng.shuffle(others)
        distractors = others[: _FUNNEL_BALL_MCQ_MAX - 1]

    return correct, distractors


VPI_OVERRIDES: list = [
    _shell_game_rotate_override,
    _funnel_ball_longest_override,
    _hockey_final_score_override,
    _hockey_longest_game_override,
    _shuffle_puzzle_override,
]


def _vpi_apply_override(ctx):
    for fn in VPI_OVERRIDES:
        r = fn(ctx)
        if r is not None:
            return r
    for fn in _LEGACY_YTB_OVERRIDES:
        r = fn(ctx)
        if r is not None:
            return r
    return None


# Snapshot the legacy YTB override chain at import time and install our
# composed dispatcher. ``ytb.build_entry`` calls ``ytb.apply_override``
# unconditionally, so monkey-patching the module-level binding is the
# cleanest way to inject ``VPI_OVERRIDES`` without forking ``build_entry``.
_LEGACY_YTB_OVERRIDES = list(ytb.OVERRIDES)
ytb.apply_override = _vpi_apply_override


def _norm_q(s):
    return ytb._norm_q(s)


def _path_to_review_rel(path):
    path = path.replace("\\", "/")
    for prefix in (_REAL_PROCESSED_PREFIX, _REAL_PREFIX):
        if path.startswith(prefix):
            return _REDACTED_RE.sub("", path[len(prefix):])
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
    p = _REDACTED_RE.sub("", video_path.replace("\\", "/"))
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


# Per-task question filter applied after the index match. Maps task -> a
# predicate that returns True for questions to keep. Tasks not listed here
# keep all matched questions.
QUESTION_KEEP_FILTERS = {
    # funnel_ball raws ship 4 questions (rank list / longest-ball /
    # duration-dict / exit-order); only the "longest ball" one (Q1) has a
    # registered MCQ override, so drop the others to avoid open-text noise.
    "funnel_ball": lambda q: "took the longest time to fall" in q.get("question", ""),
}


def _hockey_questions_transform(matched, raw_entry):
    """Project raw hockey questions to exactly three derived questions:

      1. Total own goals (synthesized from Q2 + Q3, mirrors the reference's
         ``hockey_own_goal``).
      2. Final score (red-blue) — Q5 kept as-is; ``_hockey_final_score_override``
         turns it into MCQ.
      3. Longest game — Q6 (durations array) rewritten into "Which game had the
         longest duration?" with answer ``Game {idx+1}``;
         ``_hockey_longest_game_override`` turns it into MCQ.

    All other raw questions (red goals, blue goals, per-side own goals, winner)
    are dropped. Single-game clips drop the longest-game question.
    """
    out = []
    red_own_q = next(
        (q for q in matched if "own goals were committed by the red side" in q.get("question", "")),
        None,
    )
    blue_own_q = next(
        (q for q in matched if "own goals were committed by the blue side" in q.get("question", "")),
        None,
    )
    score_q = next(
        (q for q in matched if "final score" in q.get("question", "").lower()),
        None,
    )
    durations_q = next(
        (q for q in matched if "duration of each game" in q.get("question", "")),
        None,
    )

    base_index = (red_own_q or score_q or durations_q or {}).get("index", 1)

    if red_own_q is not None and blue_own_q is not None:
        try:
            total_own = int(red_own_q["answer"]) + int(blue_own_q["answer"])
        except (TypeError, ValueError):
            total_own = None
        if total_own is not None:
            out.append({
                "index": base_index,
                "question": "How many own goals were committed in total across all games?",
                "answer": str(total_own),
            })

    if score_q is not None:
        out.append(score_q)

    if durations_q is not None:
        durations = durations_q.get("answer")
        if isinstance(durations, list) and len(durations) >= 2:
            longest_idx = durations.index(max(durations)) + 1
            rewritten = dict(durations_q)
            rewritten["question"] = "Which game had the longest duration?"
            rewritten["answer"] = f"Game {longest_idx}"
            out.append(rewritten)

    return out


# Per-task whole-list question transformers applied after the index match.
# Each maps task -> (matched_questions, raw_entry) -> new_matched_questions.
# Use this when you need to drop, rewrite, or synthesize questions based on
# multiple raw answers; for simple keep-or-drop predicates use
# ``QUESTION_KEEP_FILTERS`` instead.
QUESTION_TRANSFORMS = {
    "hockey": _hockey_questions_transform,
}


def _match_questions(raw_entry, idx, task):
    """Match clip index → list of question dicts.

    Wider than ``ytb.match_questions``: when the raw entry contains multiple
    questions but none of them carries an ``index`` field (the format used by
    the new VPI tasks), return all questions for ``idx == 1``. Falls back to
    the YTB matcher for everything else. Then applies any per-task filter
    registered in ``QUESTION_KEEP_FILTERS`` and per-task rewrite registered in
    ``QUESTION_TRANSFORMS``.
    """
    questions = raw_entry.get("questions", [])
    if questions and not any("index" in q for q in questions):
        matched = list(questions) if idx == 1 else []
    else:
        matched = ytb.match_questions(raw_entry, idx)
    keep = QUESTION_KEEP_FILTERS.get(task)
    if keep is not None:
        matched = [q for q in matched if keep(q)]
    transform = QUESTION_TRANSFORMS.get(task)
    if transform is not None:
        matched = transform(matched, raw_entry)
    return matched


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
        matched = _match_questions(raw_entry, idx, task)
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
            # Some blender/VPI tasks are still absent from the review CSV; keep
            # them after the task-specific filters/transforms above.
            if task in BLENDER_TASKS:
                entry["source_task"] = task
                kept.append(entry)
                stats["n_kept"] += 1
                continue
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
        help="Root of the VPI benchmark (containing raw/ and processed/). Default: %(default)s",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: <base>/merged_qa/vpi_filtered.json)",
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
    out_path = args.out or os.path.join(base, "merged_qa", "vpi_filtered.json")

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
            if "funnel_ball" in rel or "hockey" in rel:
                continue
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
