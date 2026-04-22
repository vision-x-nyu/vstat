"""Build merged_qa/merged_qa.json for the ytb-vids benchmark.

This is the portable copy of /nas2/benchmarks/vpi/ytb-vids/merged_qa/_build.py.
Paths are parameterized so the script can be run from anywhere.

Configuration (in priority order):
  1. --base CLI argument
  2. $VPI_YTB_BASE environment variable
  3. Default: /nas2/benchmarks/vpi/ytb-vids

Similarly, --out overrides the output path (default: <BASE>/merged_qa/merged_qa.json).

Usage:
    python build_ytb.py
    python build_ytb.py --base /path/to/ytb-vids
    VPI_YTB_BASE=/path/to/ytb-vids python build_ytb.py
    python build_ytb.py --out /tmp/ytb_merged.json

Clip-to-QA mapping:
  - Standard processed clip `<fid>_pt<idx>.mp4` is paired with raw questions
    whose `index == idx` in `raw/<cat>/<task>/<fid>.json`.
  - Fallback: if raw has a single question with no `index` field and only one
    segment, it maps to `_pt1`.
  - Non-standard clips (`_trimmed`, `_fpsN`, `_clip_START-END`, etc.) and
    tasks without raw QA are skipped.

Answer normalization:
  - numerical (int/float) or percent "N%" -> string of the integer/float,
    no choices (blender-style open-ended numerical).
  - yes/no -> 2-way MCQ {Yes, No}.
  - everything else -> open-ended (to be converted to MCQ later).

Entry schema matches blender merged_qa exactly:
  {video_id, video_path, question, answer, answer_index, choices}

Choice order is shuffled deterministically by `(task_key, video_id)` seed,
mirroring the recorded builder.
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict
from functools import lru_cache

DEFAULT_BASE = "/nas2/benchmarks/vpi/ytb-vids"

CLIP_RE = re.compile(r"^(\d+)_pt(\d+)\.mp4$")
PURE_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")

# After reorg, ytb-vids/{raw,processed} are flat at the <task> level.
# The older merged_qa keyed entries by "<category>/<task>" (e.g.
# "state_tracking/lego"). To preserve that downstream convention we carry an
# explicit task -> category map. Tasks not listed here use `CATEGORY_FALLBACK`.
CATEGORY_MAP = {
    # --- preserved from the old 4-level layout ---
    "basketball": "fine_grained",
    "bouldering": "state_tracking",
    "coffee": "fine_grained",
    "cookie_packing": "state_tracking",
    "cooking": "state_tracking",
    "cube": "activity_segmentation",
    "cup_race": "duration_estimation",
    "eating_contest": "activity_segmentation",
    "graffiti": "state_tracking",
    "jump_rope": "activity_segmentation",
    "latte_art": "duration_estimation",
    "lego": "state_tracking",
    "n_back": "state_tracking",
    "soccer": "fine_grained",
    "strand_count": "state_tracking",
    "wii": "state_tracking",
    # --- new tasks after reorg (best-guess categories) ---
    "boxing": "fine_grained",
    "carousel": "fine_grained",
    "circuit": "state_tracking",
    "marching_band": "fine_grained",
    "matryoshka": "state_tracking",
    "order_packing": "state_tracking",
    "shell_game": "state_tracking",
    "street_food": "fine_grained",
}
CATEGORY_FALLBACK = "uncategorized"


# --- edit distance ---------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


# --- MCQ formatting --------------------------------------------------------
def format_mcq(question: str, choices: list) -> str:
    letters = "ABCD"
    body = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
    return f"{question}\n\n{body}"


def build_mcq(question, correct, distractors, seed_key):
    rng = random.Random(seed_key)
    choices = [correct] + list(distractors)
    rng.shuffle(choices)
    idx = choices.index(correct)
    return format_mcq(question, choices), idx, choices


# --- answer normalization --------------------------------------------------
def normalize_answer(ans_raw):
    """Return (type, normalized_value).

    type is one of: 'numerical', 'mcq_yn', 'open_text'.
    """
    s = str(ans_raw).strip().rstrip(".").strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return ("numerical", s)
    pct = PERCENT_RE.fullmatch(s)
    if pct:
        return ("numerical", pct.group(1))
    if s.lower() == "yes":
        return ("mcq_yn", "Yes")
    if s.lower() == "no":
        return ("mcq_yn", "No")
    return ("open_text", s)


# --- raw -> clip matching --------------------------------------------------
def iter_clips(proc_root, raw_root):
    """Yield (cat, task, clip_filename, fid, idx) for every standard clip that
    has a matching raw JSON with questions.

    Layout (post-reorg):
        raw/<task>/<fid>.json
        processed/<task>/<fid>_pt<idx>.mp4

    `cat` is resolved from `CATEGORY_MAP` (falls back to CATEGORY_FALLBACK).
    """
    if not os.path.isdir(proc_root):
        return
    for task in sorted(os.listdir(proc_root)):
        task_dir = f"{proc_root}/{task}"
        if not os.path.isdir(task_dir):
            continue
        raw_task_dir = f"{raw_root}/{task}"
        if not os.path.isdir(raw_task_dir):
            continue
        cat = CATEGORY_MAP.get(task, CATEGORY_FALLBACK)
        for f in sorted(os.listdir(task_dir)):
            if not f.endswith(".mp4"):
                continue
            m = CLIP_RE.match(f)
            if not m:
                continue
            fid = m.group(1)
            idx = int(m.group(2))
            yield cat, task, f, fid, idx


def load_raw_qa(raw_root, cat, task):
    # `cat` is kept in the signature for backwards compatibility but is no
    # longer part of the raw path under the post-reorg flat layout.
    del cat
    out = {}
    d = f"{raw_root}/{task}"
    for rj in sorted(glob.glob(f"{d}/*.json")):
        fid = os.path.basename(rj).replace(".json", "")
        with open(rj) as fp:
            out[fid] = json.load(fp)
    return out


def match_questions(raw_entry, idx):
    """Return list of question dicts matching clip index.

    Handles two cases:
      - questions[i].index == idx (standard)
      - single question with no `index` and only one segment -> maps to pt1
    """
    questions = raw_entry.get("questions", [])
    matched = [q for q in questions if q.get("index") == idx]
    if matched:
        return matched
    if (
        len(questions) == 1
        and questions[0].get("index") is None
        and idx == 1
    ):
        return questions
    return []


# --- entries / tasks to skip entirely --------------------------------------
# Entire tasks to drop (no good MCQ conversion path).
EXCLUDED_TASKS = {
    # Circuit procedural answers (domain-specific, no automatable distractors).
    "circuit",
    "table_tennis",
    "predictive_tennis",
}

# (task_key, video_id) pairs that should be excluded from merged_qa.
EXCLUDED_ENTRIES = {
    # cooking open-text answers we can't cleanly MCQ-ify.
    ("cooking", "0001_pt0"),  # "3 and 3"
    ("cooking", "0003_pt0"),  # "cheese and cilantro"
}


# --- per-task MCQ overrides ------------------------------------------------
# Each override takes a context dict and returns (correct_choice, [distractors])
# or None to defer to other overrides / default handling.
#
# ctx keys:
#   task_key     -- "cat/task"
#   raw_answer   -- original answer string from raw JSON
#   raw_entry    -- full raw JSON dict for this video (has segments, etc.)
#   question     -- original question text


def _cup_race_override(ctx):
    if ctx["task_key"] != "cup_race":
        return None
    text = ctx["raw_answer"].lower()
    if "left" in text and "right" not in text:
        correct = "The person on the left will win."
        other = "The person on the right will win."
    elif "right" in text and "left" not in text:
        correct = "The person on the right will win."
        other = "The person on the left will win."
    else:
        return None
    return correct, [other]


_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth"]

# Per-clip cup pool size for latte_art, based on how many cups are visible in
# that specific clip (NOT simply the number of segments in the raw entry).
# video_id (without .mp4) -> number of cups.
_LATTE_ART_POOL_SIZE = {
    "0001_pt1": 2,
    "0001_pt2": 2,
    "0002_pt1": 3,
    "0002_pt2": 4,
    "0002_pt3": 3,
    "0002_pt4": 2,
    "0003_pt1": 3,
    "0003_pt2": 3,
    "0003_pt3": 3,
    "0003_pt4": 3,
}


def _latte_art_override(ctx):
    """duration_estimation/latte_art: ordinal cup answers -> MCQ.

    Pool size comes from `_LATTE_ART_POOL_SIZE` (manually verified against the
    clips). The correct answer is the ordinal the raw answer mentions;
    distractors are every other ordinal within the pool.
    """
    if ctx["task_key"] != "latte_art":
        return None
    text = ctx["raw_answer"].lower()
    # Handle "the second one" as "second"
    found = None
    for ord_word in _ORDINALS:
        if ord_word in text:
            found = ord_word
            break
    if found is None:
        return None
    n_cups = _LATTE_ART_POOL_SIZE.get(ctx["video_id"])
    if n_cups is None:
        return None
    pool = [f"The {w} cup" for w in _ORDINALS[:n_cups]]
    correct = f"The {found} cup"
    if correct not in pool:
        return None
    distractors = [p for p in pool if p != correct]
    return correct, distractors


_ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@lru_cache(maxsize=1)
def _english_words_by_length():
    """Lowercase English words grouped by length, alphabetic only."""
    from english_words import get_english_words_set
    out = defaultdict(list)
    for w in get_english_words_set(["web2"], lower=True):
        if w.isalpha():
            out[len(w)].append(w)
    return {k: sorted(v) for k, v in out.items()}


def _english_word_pool(correct: str) -> list:
    """Same-length English words, case-matched to `correct`."""
    words = _english_words_by_length().get(len(correct), [])
    if correct.isupper():
        return [w.upper() for w in words]
    return list(words)


_GRAFFITI_ALPHABET_LOOKUP = {
    'A': ['H', 'R', 'V', '4'],   # triangle + crossbar -> H crossbar, R bump+leg, V triangle, 4 diagonals
    'B': ['P', 'R', 'D', '8'],   # two bumps right -> P top-bump, R bump+leg, D one bump, 8 two loops
    'C': ['G', 'O', 'Q', '0'],   # open arc left -> G (C+bar), O circle, Q circle+tail, 0 oval
    'D': ['O', 'B', 'P', '0'],   # one bump right -> O oval, B two bumps, P top bump, 0 oval
    'E': ['F', 'B', 'L', '3'],   # three bars -> F two bars, B closed, L bottom bar, 3 mirrored-E
    'F': ['E', 'P', 'T', '7'],   # two bars open -> E three bars, P closed top, T top bar, 7 top-bar+diagonal
    'G': ['C', 'O', 'Q', '6'],   # C + inner shelf -> C open, O circle, Q tail, 6 loop+tail
    'H': ['A', 'M', 'N', '4'],   # two verticals + crossbar -> A crossbar, M verticals, N diagonal, 4 crossbar
    'I': ['L', 'T', 'J', '1'],   # vertical -> L foot, T cap, J hook, 1 vertical
    'J': ['I', 'L', 'U', '7'],   # hook bottom -> I straight, L foot, U curve, 7 hook
    'K': ['R', 'X', 'Y', '8'],   # diagonals right -> R leg, X crossing, Y fork, 8 symmetric
    'L': ['I', 'T', 'J', '1'],   # vertical + foot -> I no foot, T cap, J hook, 1 vertical
    'M': ['N', 'W', 'H', '3'],   # double-peak -> N one diagonal, W inverted-M, H crossbar, 3 bumps
    'N': ['M', 'H', 'Z', '2'],   # diagonal strut -> M double, H crossbar, Z diagonal, 2 curve+foot
    'O': ['Q', 'C', 'D', '0'],   # closed oval -> Q tail, C open, D bump, 0 zero
    'P': ['F', 'R', 'B', '9'],   # top bump + stem -> F two bars, R bump+leg, B two bumps, 9 loop+tail
    'Q': ['O', 'G', 'C', '9'],   # circle + tail -> O circle, G shelf, C open, 9 loop+tail
    'R': ['P', 'B', 'K', '2'],   # bump + leg -> P no leg, B two bumps, K diagonals, 2 curve+foot
    'S': ['Z', 'C', 'G', '5'],   # S-curve -> Z zigzag, C open arc, G shelf, 5 mirrored-S
    'T': ['I', 'L', 'F', '7'],   # vertical + cap -> I no cap, L foot, F two bars, 7 cap+diagonal
    'U': ['V', 'W', 'J', '0'],   # open-top curve -> V pointed, W double-V, J hook, 0 oval
    'V': ['U', 'W', 'Y', '7'],   # pointed bottom -> U rounded, W double, Y fork, 7 diagonal
    'W': ['M', 'N', 'V', '3'],   # inverted-M -> M normal, N diagonal, V single, 3 bumps
    'X': ['K', 'Y', 'Z', '8'],   # crossing diagonals -> K half-cross, Y fork, Z diagonal, 8 crossings
    'Y': ['V', 'W', 'K', '7'],   # fork -> V no stem, W double, K diagonals, 7 fork-like
    'Z': ['S', 'N', 'E', '2'],   # diagonal bar -> S curve, N diagonal, E bars, 2 diagonal+foot
}

_GRAFFITI_NUMBERS_LOOKUP = {
    '0': ['8', 'O'],   # oval -> 8 double-oval, O letter                                                  
    '1': ['7', 'I'],   # vertical -> 7 vertical+hook, I vertical bar                                             
    '2': ['7', 'Z'],   # curve+diagonal foot -> 7 diagonal, Z same top/bottom bars + diagonal                  
    '3': ['8', 'E'],   # right-side bumps -> 8 full loops, E mirrored-3                                        
    '4': ['9', 'A'],   # loop+descending stroke -> 9 loop+tail, A crossbar+diagonals                             
    '5': ['6', 'S'],   # curve+shelf -> 6 closed loop, S mirrored-5                                              
    '6': ['9', 'G'],   # loop+tail -> 9 mirror-image, G similar open loop+shelf                                  
    '7': ['1', 'T'],   # top bar+diagonal -> 1 simpler vertical, T top bar+stem                                  
    '8': ['3', 'B'],   # two stacked loops -> 3 right bumps, B two right bumps                                   
    '9': ['6', 'P'],   # loop+descending tail -> 6 mirror-image, P loop+stem
}

def _graffiti_adv_override(ctx):
    """state_tracking/graffiti:
      - single letter (A-Z) -> 4-way MCQ with 3 random letters
      - word -> 4-way MCQ via same-length English words + edit distance
      - "X and Y" multi-letter phrases -> 4-way using alphabet-based distractors
    Numeric answers are already normalized to 'numerical' upstream.
    """
    if ctx["task_key"] != "graffiti":
        return None
    # Strip trailing period / whitespace so "R.", "WEKBOY." etc. classify
    # correctly.
    raw = ctx["raw_answer"].strip().rstrip(".").strip()

    # Multi-letter case: "X and Y" (either component may be a digit)
    m = re.fullmatch(r"([A-Za-z0-9])\s+and\s+([A-Za-z0-9])", raw)
    if m:
        l1 = m.group(1).upper()
        l2 = m.group(2).upper()
        correct = f"{l1} and {l2}"
        def _lookalikes(c):
            return _GRAFFITI_NUMBERS_LOOKUP[c] if c.isdigit() else _GRAFFITI_ALPHABET_LOOKUP[c]
        rng = random.Random(f"graffiti_pair:{correct}")
        pick1 = rng.choice(_lookalikes(l1))
        pick2 = rng.choice(_lookalikes(l2))
        pick3 = rng.choice([x for x in _lookalikes(l1) if x != pick1] or _lookalikes(l1))
        distractors = [
            f"{pick1} and {l2}",   # replace first, keep second
            f"{l1} and {pick2}",   # keep first, replace second
            f"{pick3} and {l2}",   # second random lookalike of first
        ]
        return correct, distractors

    # Single letter
    if len(raw) == 1 and raw.isalpha():
        correct = raw.upper()
        rng = random.Random(f"graffiti_letter:{correct}")
        pool = _GRAFFITI_ALPHABET_LOOKUP[correct][:]
        rng.shuffle(pool)
        return correct, pool[:3]

    # Word (alphabetic, length >= 2): replace one letter at a time with its lookalike
    if raw.isalpha() and len(raw) >= 2:
        correct = raw.upper()
        rng = random.Random(f"graffiti_word:{correct}")
        distractors = []
        for i, ch in enumerate(correct):
            if len(distractors) == 3:
                break
            replacement = rng.choice(_GRAFFITI_ALPHABET_LOOKUP[ch][:3])
            distractors.append(correct[:i] + replacement + correct[i + 1:])
        if distractors:
            return correct, distractors

    return None


def _graffiti_random_override(ctx):
    """graffiti:
      - single letter (A-Z) -> 4-way MCQ with 3 random letters
      - word -> 4-way MCQ via same-length English words + edit distance
      - "X and Y" multi-letter phrases -> 4-way using alphabet-based distractors
    Numeric answers are already normalized to 'numerical' upstream.
    """
    if ctx["task_key"] != "graffiti":
        return None
    # Strip trailing period / whitespace so "R.", "WEKBOY." etc. classify
    # correctly.
    raw = ctx["raw_answer"].strip().rstrip(".").strip()

    # Multi-letter case: "X and Y" (either component may be a digit)
    m = re.fullmatch(r"([A-Za-z0-9])\s+and\s+([A-Za-z0-9])", raw)
    if m:
        correct = f"{m.group(1).upper()} and {m.group(2).upper()}"
        # Distractors: pairs of random letters different from correct
        used = {m.group(1).upper(), m.group(2).upper()}
        other = [c for c in _ALPHABET if c not in used]
        rng = random.Random(f"graffiti_pair:{correct}")
        rng.shuffle(other)
        pairs = [f"{other[i]} and {other[i + 1]}" for i in range(0, 6, 2)]
        return correct, pairs[:3]

    # Single letter
    if len(raw) == 1 and raw.isalpha():
        correct = raw.upper()
        others = [c for c in _ALPHABET if c != correct]
        # Deterministic shuffle seeded by the correct letter so the distractor
        # sets are stable across runs.
        rng = random.Random(f"graffiti_letter:{correct}")
        rng.shuffle(others)
        return correct, others[:3]

    # Word (alphabetic, length >= 2)
    if raw.isalpha() and len(raw) >= 2:
        correct = raw.upper()
        pool = _english_word_pool(correct)
        pool = [w for w in pool if w != correct]
        # Prefer edit-distance-closest, fall back to random if pool is empty.
        if pool:
            pool.sort(key=lambda w: (levenshtein(correct, w), w))
            return correct, pool[:3]

    print(ctx["question"], ctx["raw_answer"])

    return None


def _graffiti_override_factory(mode="random"):
    if mode == "random":
        return _graffiti_random_override
    elif mode == "adv":
        return _graffiti_adv_override
    else:
        raise ValueError(f"Invalid mode: {mode}")


def _wii_override(ctx):
    """state_tracking/wii: "What is the name of the last fish caught?"
    Distractors are other Wii Play Fishing fish names (manually curated).
    """
    if ctx["task_key"] != "wii":
        return None
    if "fish" not in ctx["question"].lower():
        return None
    correct = ctx["raw_answer"].strip().rstrip(".").strip()
    pool = ["Touchy Fish", "Mystery Fish", "Nibbler", "Plain Oi' Fish", "Goldfish"]
    # Match case: find the entry that equals correct case-insensitively
    correct_in_pool = next((p for p in pool if p.lower() == correct.lower()), None)
    if correct_in_pool is None:
        return None
    distractors = [p for p in pool if p != correct_in_pool]
    # 4-way MCQ: take 3 distractors
    return correct_in_pool, distractors[:3]


def _cube_override(ctx):
    if ctx["task_key"] != "cube":
        return None
    if "where" not in ctx["question"].lower():
        return None
    correct = ctx["raw_answer"].strip().rstrip(".").strip()
    pool = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    correct_in_pool = next((p for p in pool if p.lower() == correct.lower()), None)
    if correct_in_pool is None:
        return None
    distractors = [p for p in pool if p != correct_in_pool]
    return correct_in_pool, distractors[:3]


def _matryoshka_override(ctx):
    if ctx["task_key"] != "matryoshka":
        return None
    if "what color" not in ctx["question"].lower():
        return None
    correct = ctx["raw_answer"].strip().rstrip(".").strip()
    pool = ['red', 'magenta', 'yellow', 'white']
    correct_in_pool = next((p for p in pool if p.lower() == correct.lower()), None)
    if correct_in_pool is None:
        print(ctx["question"], ctx["raw_answer"])
        return None
    distractors = [p for p in pool if p != correct_in_pool]
    return correct_in_pool, distractors[:3]

_ORDER_PACKING_POOL_SIZE = {
    "0005_pt1": 2,
    "0005_pt2": 2,
    "0005_pt3_q3": 3,
    "0005_pt4_q1": 4,
    "0005_pt5_q1": 3,
    "0005_pt6": 2,
}

def _order_packing_override(ctx):
    if ctx["task_key"] != "order_packing":
        return None
    if "which order" not in ctx["question"].lower():
        return None
    text = ctx["raw_answer"].lower()
    found = None
    for ord_word in _ORDINALS:
        if ord_word in text:
            found = ord_word
            break
    if found is None:
        return None
    n_orders = _ORDER_PACKING_POOL_SIZE.get(ctx["video_id"])
    if n_orders is None:
        print(ctx["video_id"], ctx["question"], ctx["raw_answer"])
        return None
    pool = [f"The {w} one" for w in _ORDINALS[:n_orders]]
    correct = f"The {found} one"
    if correct not in pool:
        return None
    distractors = [p for p in pool if p != correct]
    return correct, distractors


OVERRIDES = [
    _cup_race_override,
    _latte_art_override,
    _graffiti_override_factory("adv"),
    _wii_override,
    _cube_override,
    _matryoshka_override,
    _order_packing_override,
]


def apply_override(ctx):
    for fn in OVERRIDES:
        r = fn(ctx)
        if r is not None:
            return r
    return None


# --- entry builders --------------------------------------------------------
def build_entry(video_id, video_path, question_text, ans_type, norm, seed_key,
                task_key=None, raw_answer=None, raw_entry=None):
    entry = {
        "video_id": video_id,
        "video_path": video_path,
        "question": question_text.strip(),
        "answer": norm,
        "answer_index": None,
        "choices": None,
    }

    # Yes/No MCQ
    if ans_type == "mcq_yn":
        choices = ["Yes", "No"]
        q_text, idx, choices = build_mcq(question_text, norm,
                                         [c for c in choices if c != norm],
                                         seed_key)
        entry["question"] = q_text
        entry["answer"] = "ABCD"[idx]
        entry["answer_index"] = idx
        entry["choices"] = choices
        return entry

    # Task-specific MCQ override (open_text that we know how to convert)
    if ans_type == "open_text" and task_key is not None:
        ctx = {
            "task_key": task_key,
            "raw_answer": raw_answer,
            "raw_entry": raw_entry or {},
            "question": question_text,
            "video_id": video_id,
        }
        override = apply_override(ctx)
        if override is not None:
            correct, distractors = override
            q_text, idx, choices = build_mcq(question_text, correct,
                                             distractors, seed_key)
            entry["question"] = q_text
            entry["answer"] = "ABCD"[idx]
            entry["answer_index"] = idx
            entry["choices"] = choices
            return entry

    return entry


# --- main ------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--base",
        default=os.environ.get("VPI_YTB_BASE", DEFAULT_BASE),
        help="Root of the ytb-vids benchmark (containing raw/ and processed/). "
             "Overrides $VPI_YTB_BASE. Default: %(default)s",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: <base>/merged_qa/merged_qa.json)",
    )
    ap.add_argument(
        "--video-root",
        default=None,
        help="Root to prefix video_path with (default: <base>/processed)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all processed entries to stdout without writing any file.",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    base = os.path.abspath(args.base)
    raw_root = f"{base}/raw"
    proc_root = f"{base}/processed"
    video_root = args.video_root or proc_root
    out_path = args.out or f"{base}/merged_qa/merged_qa.json"

    if not os.path.isdir(raw_root) or not os.path.isdir(proc_root):
        print(f"ERROR: expected raw/ and processed/ under {base}", file=sys.stderr)
        print(f"  raw/       exists: {os.path.isdir(raw_root)}", file=sys.stderr)
        print(f"  processed/ exists: {os.path.isdir(proc_root)}", file=sys.stderr)
        sys.exit(1)

    data = defaultdict(list)

    # Cache raw QA per (cat, task)
    raw_cache = {}

    for task, clip_file, fid, idx in iter_clips(proc_root, raw_root):
        task_key = task
        if task_key in EXCLUDED_TASKS:
            continue
        if task not in raw_cache:
            raw_cache[task] = load_raw_qa(raw_root, task)
        raw_entry = raw_cache[task].get(fid)
        if raw_entry is None:
            continue
        matched = match_questions(raw_entry, idx)
        if not matched:
            continue

        video_id_base = clip_file.replace(".mp4", "")
        video_path = f"{video_root}/{task}/{clip_file}"

        for qi, q in enumerate(matched):
            raw_answer = q.get("answer", "")
            ans_type, norm = normalize_answer(raw_answer)
            # Disambiguate video_id when a clip carries multiple questions.
            video_id = (
                video_id_base if len(matched) == 1 else f"{video_id_base}_q{qi}"
            )
            if (task_key, video_id) in EXCLUDED_ENTRIES:
                continue
            seed_key = f"{task_key}:{video_id}"
            entry = build_entry(
                video_id=video_id,
                video_path=video_path,
                question_text=q.get("question", ""),
                ans_type=ans_type,
                norm=norm,
                seed_key=seed_key,
                task_key=task_key,
                raw_answer=raw_answer,
                raw_entry=raw_entry,
            )
            data[task_key].append(entry)

    out = {
        "dataset_name": "vpi-ytb",
        "version": "0.1",
        "video_root": video_root,
        "data": {k: data[k] for k in sorted(data)},
    }

    if not args.dry_run:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(out, fp, indent=2, ensure_ascii=False)
        print(f"wrote {out_path}")

    # Summary
    total = sum(len(v) for v in data.values())
    print(f"tasks: {len(data)}, total samples: {total}")
    for k in sorted(data):
        n_mcq = sum(1 for e in data[k] if e["choices"])
        n_num = sum(
            1
            for e in data[k]
            if not e["choices"] and re.fullmatch(r"-?\d+(?:\.\d+)?", str(e["answer"]))
        )
        n_open = len(data[k]) - n_mcq - n_num
        print(f"  {k}: {len(data[k])} (num={n_num}, mcq={n_mcq}, open={n_open})")


if __name__ == "__main__":
    main()
