"""Check whether submissions_v2.jsonl covers every (source_task, video_id) in vpi_qa_redact.json.

A submission's `taskId` looks like `<prefix>:<source_task>/<video_id>` (e.g.
`real:basketball/0001_pt1_q1`, `ytb:boxing/0002_pt1_q0`) or, in the blender case,
`<prefix>:<extra>:<source_task>/<video_id>`. Only the trailing
`<source_task>/<video_id>` is needed for matching against the QA file's
`source_task` key + `video_id` field. Submission entries that do not match any
QA entry are ignored, per request.
"""

import argparse
import json
from collections import defaultdict

DEFAULT_QA = "/nas2/benchmarks/vpi/real/merged_qa/vpi_qa_redact.json"
DEFAULT_SUBMISSIONS = "/nas2/benchmarks/vpi_eval_site/submissions_v2.jsonl"


def parse_task_id(task_id: str) -> tuple[str, str] | None:
    """Return (source_task, video_id) parsed from a submission taskId, or None."""
    if "/" not in task_id:
        return None
    prefix, video_id = task_id.rsplit("/", 1)
    if ":" not in prefix:
        return None
    source_task = prefix.rsplit(":", 1)[1]
    return source_task, video_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa", default=DEFAULT_QA)
    ap.add_argument("--submissions", default=DEFAULT_SUBMISSIONS)
    args = ap.parse_args()

    with open(args.qa) as f:
        qa = json.load(f)

    # Build the set of all (source_task, video_id) pairs the QA file expects.
    expected: set[tuple[str, str]] = set()
    by_task: dict[str, set[str]] = defaultdict(set)
    for source_task, entries in qa["data"].items():
        for entry in entries:
            video_id = entry["video_id"]
            expected.add((source_task, video_id))
            by_task[source_task].add(video_id)

    # Collect every (source_task, video_id) that appears in submissions.
    submitted: set[tuple[str, str]] = set()
    total_submissions = 0
    with open(args.submissions) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_submissions += 1
            entry = json.loads(line)
            parsed = parse_task_id(entry.get("taskId", ""))
            if parsed is not None:
                submitted.add(parsed)

    # Only count submission pairs that match a QA entry.
    matched = submitted & expected
    missing = expected - submitted

    print(f"QA entries:           {len(expected)}")
    print(f"Submission lines:     {total_submissions}")
    print(f"Unique submitted QAs: {len(submitted)}")
    print(f"Matched (in QA):      {len(matched)}")
    print(f"Missing from subs:    {len(missing)}")
    print()

    if not missing:
        print("All QA entries are covered.")
        return

    # Group missing entries by source_task for readability.
    by_missing: dict[str, list[str]] = defaultdict(list)
    for source_task, video_id in missing:
        by_missing[source_task].append(video_id)

    print("Missing entries (source_task / video_id):")
    for source_task in sorted(by_missing):
        vids = sorted(by_missing[source_task])
        print(f"  [{source_task}] {len(vids)}/{len(by_task[source_task])} missing")
        for v in vids:
            print(f"    - {v}")


if __name__ == "__main__":
    main()
