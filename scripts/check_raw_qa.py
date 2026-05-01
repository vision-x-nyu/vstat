"""Scan raw/<task>/<fid>.json files for mismatches against filenames,
paired MP4s, and processed segments.

Checks:
  1. `local_id` field matches the JSON filename stem
  2. `video_file` matches `<stem>.mp4` and exists alongside the JSON
  3. `questions[]` is non-empty
  4. Mix of None and integer `index` values
  5. Question indices align with `<fid>_pt<idx>.mp4` segments in processed/
     (a single question with `index=None` is treated as pt1, matching
     `build_ytb.match_questions`)
  6. raw MP4s without a paired JSON

Usage:
    python scripts/check_raw_qa.py
    python scripts/check_raw_qa.py --base /path/to/ytb-vids
    VPI_YTB_BASE=/path/to/ytb-vids python scripts/check_raw_qa.py
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

DEFAULT_BASE = "/nas2/benchmarks/vpi/ytb-vids"
CLIP_RE = re.compile(r"^(\d+)_pt(\d+)\.mp4$")
_STRETCH_SUFFIX = "_stretch_0p2s"


def scan(base):
    raw_root = f"{base}/raw"
    proc_root = f"{base}/processed"
    if not os.path.isdir(raw_root):
        print(f"ERROR: {raw_root} not found", file=sys.stderr)
        return 2

    issues = defaultdict(list)

    for task in sorted(os.listdir(raw_root)):
        rdir = f"{raw_root}/{task}"
        if not os.path.isdir(rdir):
            continue
        # Stretched processed dirs share QA with the base task — only check
        # the base task and its corresponding processed/ if it exists.
        pdir = f"{proc_root}/{task}"

        json_files = sorted(f for f in os.listdir(rdir) if f.endswith(".json"))
        mp4_files = set(f for f in os.listdir(rdir) if f.endswith(".mp4"))

        proc_segs = defaultdict(set)
        if os.path.isdir(pdir):
            for f in os.listdir(pdir):
                m = CLIP_RE.match(f)
                if m:
                    proc_segs[m.group(1)].add(int(m.group(2)))

        for jf in json_files:
            stem = jf[:-5]
            path = f"{rdir}/{jf}"
            try:
                data = json.load(open(path))
            except Exception as e:
                issues[task].append(f"{jf}: cannot parse: {e}")
                continue

            local_id = data.get("local_id")
            video_file = data.get("video_file")
            questions = data.get("questions", [])

            if local_id != stem:
                issues[task].append(
                    f"{jf}: local_id={local_id!r} != filename stem {stem!r}"
                )

            if video_file:
                expected_mp4 = f"{stem}.mp4"
                if video_file != expected_mp4:
                    issues[task].append(
                        f"{jf}: video_file={video_file!r} != expected {expected_mp4!r}"
                    )
                if video_file not in mp4_files:
                    issues[task].append(
                        f"{jf}: video_file={video_file!r} not present in raw/{task}/"
                    )

            if not questions:
                issues[task].append(f"{jf}: no questions[]")
            else:
                idxs = [q.get("index") for q in questions]
                none_count = sum(1 for i in idxs if i is None)
                if none_count and any(i is not None for i in idxs):
                    issues[task].append(
                        f"{jf}: mix of None and integer indices in questions ({idxs})"
                    )

                seg_set = proc_segs.get(stem, set())
                # Build the set of segment indices the questions claim to
                # cover. A single question with index=None covers pt1
                # (matches build_ytb.match_questions fallback).
                if len(questions) == 1 and idxs[0] is None:
                    q_idxs = {1}
                else:
                    q_idxs = {i for i in idxs if isinstance(i, int)}

                if seg_set:
                    missing_seg = sorted(i for i in q_idxs if i not in seg_set)
                    unused_seg = sorted(seg_set - q_idxs)
                    if missing_seg:
                        issues[task].append(
                            f"{jf}: question indices with no matching segment: {missing_seg}"
                        )
                    if unused_seg:
                        issues[task].append(
                            f"{jf}: processed segments with no questions: {unused_seg}"
                        )
                elif q_idxs:
                    issues[task].append(
                        f"{jf}: questions reference segments but processed/{task}/"
                        f" has none for fid={stem}"
                    )

        json_stems = {f[:-5] for f in json_files}
        for mp4 in sorted(mp4_files):
            stem = mp4[:-4]
            if stem not in json_stems:
                issues[task].append(f"{mp4}: no paired JSON")

    if not issues:
        print("No mismatches found.")
        return 0

    for task in sorted(issues):
        print(f"\n=== {task} ===")
        for line in issues[task]:
            print(f"  {line}")
    total = sum(len(v) for v in issues.values())
    print(f"\n{total} issue(s) across {len(issues)} task(s).")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--base",
        default=os.environ.get("VPI_YTB_BASE", DEFAULT_BASE),
        help="Root of ytb-vids (containing raw/ and processed/). "
        "Overrides $VPI_YTB_BASE. Default: %(default)s",
    )
    args = ap.parse_args()
    sys.exit(scan(os.path.abspath(args.base)))


if __name__ == "__main__":
    main()
