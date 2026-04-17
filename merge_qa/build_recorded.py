"""Build merged_qa/all.json for the recorded benchmark.

Portable copy of /nas2/benchmarks/vpi/recorded/merged_qa/_build.py with the
BASE path parameterized so it can run from anywhere.

Configuration (in priority order):
  1. --base CLI argument
  2. $VPI_RECORDED_BASE environment variable
  3. Default: /nas2/benchmarks/vpi/recorded

Usage:
    python build_recorded.py
    python build_recorded.py --base /path/to/recorded
    VPI_RECORDED_BASE=/path/to/recorded python build_recorded.py
    python build_recorded.py --out /tmp/recorded_all.json

Layout per sub-task entry:
  {video_id, video_path, question, answer[, answer_index, choices]}

Numerical sub-tasks emit an integer-valued answer with no choices.
MCQ sub-tasks emit 3-4 choices with answer_index set accordingly.
Distractor selection is deterministic (seeded by sub_task + video_id) and,
for keyboard/morse/numberpad, uses the 3 nearest-by-edit-distance candidates.
"""
import argparse
import json
import glob
import os
import random
import sys
from functools import lru_cache
from itertools import combinations

from english_words import get_english_words_set

DEFAULT_BASE = "/nas2/benchmarks/vpi/recorded"


# --- English word dictionary (for keyboard/morse distractor pools) ---------
@lru_cache(maxsize=1)
def _english_words_lower():
    # web2 dictionary gives ~234k lowercase words
    return sorted(w for w in get_english_words_set(["web2"], lower=True)
                  if w.isalpha())


def english_word_pool(correct: str) -> list:
    """Return a large word pool matching the case and length of `correct`."""
    target_len = len(correct)
    words = _english_words_lower()
    # Match length (+/- 0 to keep edit distance focused on substitutions).
    same_len = [w for w in words if len(w) == target_len]
    # Match case of the correct answer.
    if correct.isupper():
        return [w.upper() for w in same_len]
    if correct.islower():
        return same_len
    # Mixed case: default to lower, caller can adapt.
    return same_len


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
# Note: the "Please answer with the letter ..." instruction is appended by the
# eval harness (utils.py), not emitted into the canonical question text here.


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


# --- task loaders ----------------------------------------------------------
def load_task(base, video_root, rel_path):
    items = []
    for f in sorted(glob.glob(os.path.join(base, rel_path, "*.json"))):
        with open(f) as fp:
            d = json.load(fp)
        video_id = os.path.splitext(os.path.basename(f))[0]
        video_path = os.path.join(video_root, rel_path, d["video_file"])
        items.append({"video_id": video_id, "video_path": video_path, "data": d})
    return items


# --- distractor strategies -------------------------------------------------
def nearest_by_edit(correct, pool, k=3):
    candidates = [p for p in pool if p != correct]
    candidates.sort(key=lambda x: (levenshtein(correct, x), x))
    return candidates[:k]


def all_digit_pairs():
    # always ascending, "min,max" form
    return [f"{a},{b}" for a, b in combinations(range(10), 2)]


def normalize_pair(ans):
    parts = [p.strip() for p in ans.split(",")]
    nums = sorted(int(p) for p in parts)
    return f"{nums[0]},{nums[1]}"


# --- builders --------------------------------------------------------------
def build_simple_numerical(items, sub_task):
    out = []
    for it in items:
        q = it["data"]["questions"][0]
        out.append({
            "video_id": it["video_id"],
            "video_path": it["video_path"],
            "question": q["question"],
            "answer": str(q["answer"]),
        })
    return {sub_task: out}


def build_simple_mcq(items, sub_task, distractor_fn):
    # distractor_fn(correct, all_answers_for_this_task) -> list of 3
    all_answers = [it["data"]["questions"][0]["answer"] for it in items]
    out = []
    for it in items:
        q = it["data"]["questions"][0]
        correct = q["answer"]
        distractors = distractor_fn(correct, all_answers)
        question, idx, choices = build_mcq(
            q["question"], correct, distractors, f"{sub_task}:{it['video_id']}"
        )
        out.append({
            "video_id": it["video_id"],
            "video_path": it["video_path"],
            "question": question,
            "answer": "ABCD"[idx],
            "answer_index": idx,
            "choices": choices,
        })
    return {sub_task: out}


def build_tilt_box(items):
    # fixed 4-way MCQ over corners {1,2,3,4}
    corners = ["1", "2", "3", "4"]
    out = []
    for it in items:
        q = it["data"]["questions"][0]
        correct = str(q["answer"])
        distractors = [c for c in corners if c != correct]
        question, idx, choices = build_mcq(
            q["question"], correct, distractors, f"tilt_box:{it['video_id']}"
        )
        out.append({
            "video_id": it["video_id"],
            "video_path": it["video_path"],
            "question": question,
            "answer": "ABCD"[idx],
            "answer_index": idx,
            "choices": choices,
        })
    return {"tilt_box": out}


def build_shell_game(items):
    # fixed 3-way MCQ
    out = []
    for it in items:
        q = it["data"]["questions"][0]
        correct = q["answer"]
        choices = ["Left", "Center", "Right"]
        rng = random.Random(f"shell_game:{it['video_id']}")
        rng.shuffle(choices)
        idx = choices.index(correct)
        letters = "ABC"
        body = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
        question = f"{q['question']}\n\n{body}"
        out.append({
            "video_id": it["video_id"],
            "video_path": it["video_path"],
            "question": question,
            "answer": letters[idx],
            "answer_index": idx,
            "choices": choices,
        })
    return {"shell_game": out}


def build_numberpad(items):
    # normalize answers to "min,max"; distractors from all-pairs by edit dist
    pool = all_digit_pairs()
    out = []
    for it in items:
        q = it["data"]["questions"][0]
        correct_raw = q["answer"]
        correct = normalize_pair(correct_raw)
        distractors = [p for p in pool if p != correct]
        distractors.sort(key=lambda x: (levenshtein(correct, x), x))
        distractors = distractors[:3]
        question, idx, choices = build_mcq(
            q["question"], correct, distractors, f"numberpad:{it['video_id']}"
        )
        out.append({
            "video_id": it["video_id"],
            "video_path": it["video_path"],
            "question": question,
            "answer": "ABCD"[idx],
            "answer_index": idx,
            "choices": choices,
        })
    return {"numberpad": out}


def build_cup_stacking(items):
    # 7 sub-tasks: cup_stacking_1st ... cup_stacking_7th
    ordinal = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]
    buckets = {f"cup_stacking_{o}": [] for o in ordinal}
    for it in items:
        order = it["data"].get("order", [])
        for qi, q in enumerate(it["data"]["questions"]):
            correct = q["answer"]
            # distractors: other animals in this video's order, shuffled, pick 3
            candidates = [a for a in order if a != correct]
            rng = random.Random(f"cup_stacking:{it['video_id']}:{qi}")
            rng.shuffle(candidates)
            distractors = candidates[:3]
            question, idx, choices = build_mcq(
                q["question"], correct, distractors, f"cup_stacking_{ordinal[qi]}:{it['video_id']}"
            )
            buckets[f"cup_stacking_{ordinal[qi]}"].append({
                "video_id": it["video_id"],
                "video_path": it["video_path"],
                "question": question,
                "answer": "ABCD"[idx],
                "answer_index": idx,
                "choices": choices,
            })
    return buckets


def build_packing_order(items):
    keys = ["green", "blue", "yellow", "chopsticks"]
    buckets = {f"packing_order_{k}": [] for k in keys}
    for it in items:
        for qi, q in enumerate(it["data"]["questions"]):
            buckets[f"packing_order_{keys[qi]}"].append({
                "video_id": it["video_id"],
                "video_path": it["video_path"],
                "question": q["question"],
                "answer": str(q["answer"]),
            })
    return buckets


def build_showing_card(items):
    count_keys = ["diamond", "heart", "club", "spade"]
    buckets = {f"showing_card_count_{k}": [] for k in count_keys}
    buckets["showing_card_most"] = []
    buckets["showing_card_least"] = []
    suits = ["heart", "diamond", "club", "spade"]
    for it in items:
        qs = it["data"]["questions"]
        for qi in range(4):
            q = qs[qi]
            buckets[f"showing_card_count_{count_keys[qi]}"].append({
                "video_id": it["video_id"],
                "video_path": it["video_path"],
                "question": q["question"],
                "answer": str(q["answer"]),
            })
        # most / least (index 4, 5)
        for qi, sub in [(4, "most"), (5, "least")]:
            q = qs[qi]
            ans = q["answer"]
            if "," in ans:
                # tied case -> keep as open-ended text
                buckets[f"showing_card_{sub}"].append({
                    "video_id": it["video_id"],
                    "video_path": it["video_path"],
                    "question": q["question"],
                    "answer": ans,
                })
            else:
                distractors = [s for s in suits if s != ans]
                question, idx, choices = build_mcq(
                    q["question"], ans, distractors, f"showing_card_{sub}:{it['video_id']}"
                )
                buckets[f"showing_card_{sub}"].append({
                    "video_id": it["video_id"],
                    "video_path": it["video_path"],
                    "question": question,
                    "answer": "ABCD"[idx],
                    "answer_index": idx,
                    "choices": choices,
                })
    return buckets


# --- main ------------------------------------------------------------------
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
        help="Output JSON path (default: <base>/merged_qa/all.json)",
    )
    ap.add_argument(
        "--video-root",
        default=None,
        help="Root to prefix video_path with (default: same as --base)",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    base = os.path.abspath(args.base)
    video_root = args.video_root or base
    out_path = args.out or os.path.join(base, "merged_qa", "all.json")

    if not os.path.isdir(base):
        print(f"ERROR: --base {base} does not exist", file=sys.stderr)
        sys.exit(1)

    def lt(rel):
        return load_task(base, video_root, rel)

    data = {}

    # single-question numerical
    data.update(build_simple_numerical(lt("single_state/book"), "book"))
    data.update(build_tilt_box(lt("single_state/tilt_box")))

    # shell_game (3-way fixed)
    data.update(build_shell_game(lt("single_state/shell_game")))

    # keyboard (MCQ via edit distance over same-length English words)
    data.update(build_simple_mcq(
        lt("single_state/keyboard"), "keyboard",
        lambda correct, _pool: nearest_by_edit(correct, english_word_pool(correct), 3),
    ))

    # morse (MCQ via edit distance over same-length English words, upper-cased)
    data.update(build_simple_mcq(
        lt("multi_state/morse"), "morse",
        lambda correct, _pool: nearest_by_edit(correct, english_word_pool(correct), 3),
    ))

    # numberpad (MCQ via edit distance over all digit pairs)
    data.update(build_numberpad(lt("multi_state/numberpad")))

    # cup_stacking (7 sub-tasks, MCQ from order)
    data.update(build_cup_stacking(lt("multi_state/cup_stacking")))

    # packing_order (4 sub-tasks, numerical)
    data.update(build_packing_order(lt("multi_state/packing_order")))

    # showing_card (4 count sub-tasks numerical + most/least MCQ)
    data.update(build_showing_card(lt("multi_state/showing_card")))

    out = {
        "dataset_name": "longvid-reasoning-eval-recorded",
        "version": "0.1",
        "video_root": video_root,
        "data": data,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=4, ensure_ascii=False)
    # summary
    total = sum(len(v) for v in data.values())
    print(f"wrote {out_path}")
    print(f"sub_tasks: {len(data)}, total samples: {total}")
    for k in sorted(data):
        print(f"  {k}: {len(data[k])}")


if __name__ == "__main__":
    main()
