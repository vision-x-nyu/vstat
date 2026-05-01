"""Build merged_qa entries for blender/multi_state/funnel_ball.

Source layout:
    <BASE>/multi_state/funnel_ball/{5sec,10sec,20sec}/video_NNNNNN.{json,mp4}

Target question (from each raw JSON, `questions[1]`):
    "Balls are indexed left-to-right in the last frame before any release from
    `ball_1` through `ball_N`. Which ball took the longest time to fall
    through the hole after its own release?"

Each clip has `num_balls` in {2, 4, 6} depending on duration, and the answer is
one of `ball_1` ... `ball_N`. We emit a blender-style MCQ entry:

    {
        "video_id": "000000",
        "video_path": "<video_root>/multi_state/funnel_ball/<dur>/video_000000.mp4",
        "question": "<raw question>\n\n(A) ball_2\n(B) ball_1",
        "answer": "B",
        "answer_index": 1,
        "choices": ["ball_2", "ball_1"]
    }

Choices:
  - num_balls <= 4: all balls as choices (2-/3-/4-way MCQ).
  - num_balls > 4: 4-way MCQ = correct + 3 deterministic distractors (seeded
    shuffle of the remaining balls, pick first 3).

Default (merge) mode: insert the new entries under `subtask_key` inside the
existing per-duration merged_qa files under <OUT_DIR>:
    - 5sec.json            (<- 5sec)
    - 10sec.json           (<- 10sec)
    - 20sec.json           (<- 20sec)
These files are rewritten in place (pretty-printed with `indent=2`), preserving
all existing subtasks and adding / overwriting the `funnel_ball_longest` key.

Standalone mode (`--standalone`) instead writes separate files per duration:
    - funnel_ball_5sec.json / funnel_ball_10sec.json / funnel_ball_20sec.json

Configuration (priority: CLI > env > default):
    --base / $VPI_BLENDER_BASE   default /nas2/benchmarks/vpi/blender
    --video-root                 default same as --base
    --out-dir                    default <base>/merged_qa
    --subtask-key                default "funnel_ball_longest"
    --standalone                 write standalone files instead of merging
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

DEFAULT_BASE = "/nas2/benchmarks/vpi/blender"
DURATIONS = [
    "5sec", "10sec", "20sec",
    "5sec_stretch_0p2s", "10sec_stretch_0p2s", "20sec_stretch_0p2s",
]
TARGET_Q_IDX = 1  # "Which ball took the longest time..."
MCQ_MAX = 4        # 4-way MCQ cap


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


def pick_distractors(correct, ball_ids, seed_key, k=3):
    """Return `k` distractor ball_ids (deterministically chosen)."""
    others = [b for b in ball_ids if b != correct]
    if len(others) <= k:
        return others
    rng = random.Random(f"{seed_key}:distractors")
    rng.shuffle(others)
    return others[:k]


def build_entries(base, video_root, duration, subtask_key):
    src_dir = os.path.join(base, "multi_state", "funnel_ball", duration)
    if not os.path.isdir(src_dir):
        print(f"WARN: missing {src_dir}", file=sys.stderr)
        return []
    entries = []
    for f in sorted(os.listdir(src_dir)):
        if not f.endswith(".json"):
            continue
        path = os.path.join(src_dir, f)
        with open(path) as fp:
            raw = json.load(fp)
        video_id = raw.get("video_id") or os.path.splitext(f)[0].replace("video_", "")
        try:
            q = raw["questions"][TARGET_Q_IDX]
        except (KeyError, IndexError):
            print(f"WARN: {path} missing questions[{TARGET_Q_IDX}]", file=sys.stderr)
            continue
        question_text = q["question"]
        correct = q["answer"]
        ball_ids = raw.get("ball_ids") or [f"ball_{i+1}" for i in range(raw["num_balls"])]
        if correct not in ball_ids:
            print(f"WARN: {path} correct {correct!r} not in ball_ids {ball_ids}",
                  file=sys.stderr)
            continue

        seed_key = f"{subtask_key}:{duration}:{video_id}"
        if len(ball_ids) <= MCQ_MAX:
            distractors = [b for b in ball_ids if b != correct]
        else:
            distractors = pick_distractors(correct, ball_ids, seed_key, k=MCQ_MAX - 1)

        q_text, idx, choices = build_mcq(question_text, correct, distractors, seed_key)

        video_file = os.path.splitext(f)[0] + ".mp4"
        video_path = os.path.join(video_root, "multi_state", "funnel_ball", duration,
                                  video_file)

        entries.append({
            "video_id": video_id,
            "video_path": video_path,
            "question": q_text,
            "answer": "ABCD"[idx],
            "answer_index": idx,
            "choices": choices,
        })
    return entries


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--base",
        default=os.environ.get("VPI_BLENDER_BASE", DEFAULT_BASE),
        help="Blender benchmark root (contains multi_state/funnel_ball/). "
             "Overrides $VPI_BLENDER_BASE. Default: %(default)s",
    )
    ap.add_argument(
        "--video-root",
        default=None,
        help="Root to prefix video_path with (default: same as --base).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write per-duration JSONs (default: <base>/merged_qa).",
    )
    ap.add_argument(
        "--subtask-key",
        default="funnel_ball_longest",
        help="Key under 'data' in each output JSON. Default: %(default)s",
    )
    ap.add_argument(
        "--durations",
        nargs="+",
        default=DURATIONS,
        help=f"Which durations to build. Default: {DURATIONS}",
    )
    ap.add_argument(
        "--standalone",
        action="store_true",
        help="Write standalone funnel_ball_<dur>.json files instead of merging "
             "into existing <dur>.json.",
    )
    return ap.parse_args(argv)


# Merge target filename = <duration>.json for all current durations
# (non-stretch: 5sec.json / 10sec.json / 20sec.json; stretch: <dur>_stretch_0p2s.json).
# 20_sec.json was renamed to 20sec.json for consistency; see commit history.
def merge_filename(duration: str) -> str:
    return f"{duration}.json"


def main(argv=None):
    args = parse_args(argv)
    base = os.path.abspath(args.base)
    video_root = args.video_root or base
    out_dir = args.out_dir or os.path.join(base, "merged_qa")

    if not os.path.isdir(base):
        print(f"ERROR: --base {base} does not exist", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    grand_total = 0
    for dur in args.durations:
        entries = build_entries(base, video_root, dur, args.subtask_key)
        dist = defaultdict(int)
        for e in entries:
            dist[len(e["choices"])] += 1
        dist_str = ", ".join(f"{k}-way:{v}" for k, v in sorted(dist.items()))

        if args.standalone:
            out_path = os.path.join(out_dir, f"funnel_ball_{dur}.json")
            payload = {
                "dataset_name": "longvid-reasoning-eval-blender-funnel_ball",
                "version": "0.1",
                "video_root": video_root,
                "data": {args.subtask_key: entries},
            }
            with open(out_path, "w") as fp:
                json.dump(payload, fp, indent=2, ensure_ascii=False)
            print(f"wrote {out_path}  ({len(entries)} entries; {dist_str})")
        else:
            # Merge into existing per-duration merged_qa file.
            target = os.path.join(out_dir, merge_filename(dur))
            if not os.path.isfile(target):
                print(f"ERROR: merge target {target} does not exist; "
                      f"use --standalone to write a new file.", file=sys.stderr)
                sys.exit(2)
            with open(target) as fp:
                doc = json.load(fp)
            doc.setdefault("data", {})
            existed = args.subtask_key in doc["data"]
            doc["data"][args.subtask_key] = entries
            with open(target, "w") as fp:
                json.dump(doc, fp, indent=2, ensure_ascii=False)
            action = "replaced" if existed else "added"
            n_total = sum(len(v) for v in doc["data"].values())
            print(f"merged {target}  ({action} {args.subtask_key!r}: "
                  f"{len(entries)} entries; {dist_str}; "
                  f"file now has {len(doc['data'])} subtasks / {n_total} entries)")
        grand_total += len(entries)

    print(f"total funnel_ball entries: {grand_total}")


if __name__ == "__main__":
    main()
