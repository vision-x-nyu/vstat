"""Pretty-print flags.jsonl alongside the corresponding QA pair.

Each line in /nas2/benchmarks/vpi_eval_site/flags.jsonl carries a taskId of
the form `<dataset>:<task>/<video_id>` (with an extra `:<id_tag>` segment for
blender) and a reason / detail. We look up the matching QA item from the
merged QA JSON and print question + answer together with the flag.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_FLAGS = "/nas2/benchmarks/vpi_eval_site/flags.jsonl"
QA_PATH = Path("/nas2/benchmarks/vpi/real/merged_qa/real_filtered_redact.json")

TASK_ID_RE = re.compile(r"^(?P<dataset>[^:]+):(?:(?P<tag>[^:/]+):)?(?P<task>[^/]+)/(?P<vid>.+)$")


def _build_qa_index(qa_path):
    """Return {(task, video_id): qa_item} from the ytb merged QA JSON."""
    with open(qa_path) as fh:
        doc = json.load(fh)
    index = {}
    for task, items in doc["data"].items():
        for raw in items:
            index[(task, str(raw["video_id"]))] = raw
    return index


def _iter_flags(flags_path):
    decoder = json.JSONDecoder()
    with open(flags_path) as fh:
        for line in fh:
            s = line.strip()
            while s:
                try:
                    rec, end = decoder.raw_decode(s)
                except json.JSONDecodeError:
                    break
                yield rec
                s = s[end:].lstrip()


def _wrap(label, text, width=88, indent=4):
    if text is None:
        text = ""
    text = str(text)
    pad = " " * indent
    head = f"{label}: "
    if not text:
        return head
    avail = max(20, width - len(head))
    lines = []
    while len(text) > avail:
        cut = text.rfind(" ", 0, avail)
        if cut <= 0:
            cut = avail
        lines.append(text[:cut])
        text = text[cut:].lstrip()
    lines.append(text)
    out = head + lines[0]
    for ln in lines[1:]:
        out += "\n" + pad + ln
    return out


def _print_entry(i, rec, qa_index):
    tid = rec.get("taskId", "")
    m = TASK_ID_RE.match(tid)
    print(f"=== [{i}] {tid}")
    print(f"  user:   {rec.get('user')}    ts: {rec.get('ts')}")
    print(f"  reason: {rec.get('reason')}")
    if rec.get("detail"):
        print(f"  {_wrap('detail', rec['detail'], indent=10)}")
    if not m or rec.get("dataset") != "ytb":
        print("  (no QA lookup — non-ytb dataset or unparseable taskId)\n")
        return
    qa = qa_index.get((m.group("task"), m.group("vid")))
    if qa is None:
        print(f"  (QA not found for task={m.group('task')} vid={m.group('vid')})\n")
        return
    print(f"  task:   {m.group('task')}    video_id: {m.group('vid')}")
    print(f"  {_wrap('Q', qa.get('question'), indent=5)}")
    choices = qa.get("choices")
    if choices:
        ai = qa.get("answer_index")
        for j, c in enumerate(choices):
            mark = " *" if j == ai else "  "
            print(f"     {mark} ({chr(ord('A') + j)}) {c}")
    print(f"  A:  {qa.get('answer')}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("flags_path", nargs="?", default=DEFAULT_FLAGS,
                    help=f"Path to flags.jsonl (default: {DEFAULT_FLAGS})")
    ap.add_argument("--task", "-t", default=None,
                    help="Restrict to a single source_task (e.g. carousel)")
    ap.add_argument("--reason", "-r", default=None,
                    help="Restrict to a single reason (e.g. unclear_question)")
    ap.add_argument("--summary", action="store_true",
                    help="Append a counts-by-reason / counts-by-task summary")
    args = ap.parse_args()

    qa_index = _build_qa_index(QA_PATH)

    by_reason = defaultdict(int)
    by_task = defaultdict(int)
    n = 0
    for rec in _iter_flags(args.flags_path):
        m = TASK_ID_RE.match(rec.get("taskId", ""))
        task = m.group("task") if m else None
        if args.task and task != args.task:
            continue
        if args.reason and rec.get("reason") != args.reason:
            continue
        n += 1
        _print_entry(n, rec, qa_index)
        by_reason[rec.get("reason")] += 1
        if task:
            by_task[task] += 1

    print(f"Total: {n} flag(s)")
    if args.summary and n:
        print("\nBy reason:")
        for k in sorted(by_reason, key=lambda x: -by_reason[x]):
            print(f"  {by_reason[k]:>4d}  {k}")
        print("\nBy task:")
        for k in sorted(by_task, key=lambda x: -by_task[x]):
            print(f"  {by_task[k]:>4d}  {k}")


if __name__ == "__main__":
    main()
