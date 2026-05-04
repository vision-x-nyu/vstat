"""Pretty-print reasoning traces from samples_*.jsonl file(s).

The raw JSONL packs every field onto one line with escaped newlines,
which makes the reasoning traces unreadable in an editor. This script
renders each sample as a human-readable block with real newlines.

Usage:
    python scripts/inspect_reasoning_trace.py <samples.jsonl-or-dir> [--output OUT_DIR]
                                              [--doc-id N] [--only-wrong]
                                              [--no-reasoning]

Defaults:
    If --output is omitted for a file input, prints to stdout.
    If --output is a directory, writes <OUT_DIR>/<name>.txt where <name> is
    the input stem with "samples" replaced by "traces"
    (e.g. samples_foo.jsonl -> traces_foo.txt).
    If the input is a directory, renders every samples_*.jsonl file and writes
    traces_*.txt files to --output, or to the input directory when --output is
    omitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

SEP = "=" * 80
SUB = "-" * 80


def _fmt_reasoning(reasoning) -> str:
    if reasoning is None:
        return "(no reasoning trace)"
    if isinstance(reasoning, list):
        chunks: list[str] = []
        for i, chunk in enumerate(reasoning):
            if len(reasoning) > 1:
                chunks.append(f"--- reasoning[{i}] ---")
            chunks.append(str(chunk).strip())
        return "\n".join(chunks)
    return str(reasoning).strip()


def _fmt_metric(d, key: str) -> str:
    if not isinstance(d, dict):
        return "—"
    v = d.get(key)
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_sample(d: dict, *, include_reasoning: bool) -> str:
    lines: list[str] = []
    lines.append(SEP)
    lines.append(f"doc_id: {d.get('doc_id')}")
    if d.get("video_path"):
        lines.append(f"video_path: {d.get('video_path')}")
    acc = d.get("accuracy")
    if isinstance(acc, dict) and "is_correct" in acc:
        lines.append(f"result: {'CORRECT' if acc.get('is_correct') else 'INCORRECT'}")
    else:
        lines.append("result: UNKNOWN")
    tc = d.get("token_counts")
    if isinstance(tc, list) and tc:
        tc0 = tc[0] if isinstance(tc[0], dict) else {}
        lines.append(
            f"tokens: input={tc0.get('input_tokens', '?')} "
            f"output={tc0.get('output_tokens', '?')}"
        )
    lines.append(
        f"accuracy={_fmt_metric(d.get('accuracy'), 'is_correct')}  "
        f"mra={_fmt_metric(d.get('mra'), 'mra_score')}  "
        f"mae={_fmt_metric(d.get('mae'), 'mae_score')}"
    )
    lines.append(SUB)
    lines.append("INPUT:")
    lines.append(str(d.get("input", "")).strip())
    lines.append(SUB)
    lines.append(f"TARGET: {d.get('target')}")
    lines.append(SUB)
    lines.append("MODEL RESPONSE:")
    lines.append(str(d.get("filtered_resps", "")).strip())
    if include_reasoning and "reasoning" in d:
        lines.append(SUB)
        lines.append("REASONING:")
        lines.append(_fmt_reasoning(d.get("reasoning")))
    lines.append("")
    return "\n".join(lines)


def iter_samples(path: Path, *, doc_id: int | None, only_wrong: bool):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if doc_id is not None and d.get("doc_id") != doc_id:
                continue
            if only_wrong:
                acc = d.get("accuracy", {})
                if isinstance(acc, dict) and acc.get("is_correct"):
                    continue
            yield d


def output_name_for(path: Path) -> str:
    return path.stem.replace("samples", "traces", 1) + ".txt"


def render_samples_file(path: Path, *, doc_id: int | None, only_wrong: bool, include_reasoning: bool) -> tuple[str, int]:
    blocks: list[str] = []
    for d in iter_samples(path, doc_id=doc_id, only_wrong=only_wrong):
        blocks.append(render_sample(d, include_reasoning=include_reasoning))

    text = "\n".join(blocks) if blocks else "(no matching samples)\n"
    return text, len(blocks)


def write_trace_file(
    samples_path: Path,
    *,
    out_dir: Path,
    doc_id: int | None,
    only_wrong: bool,
    include_reasoning: bool,
) -> None:
    text, sample_count = render_samples_file(
        samples_path,
        doc_id=doc_id,
        only_wrong=only_wrong,
        include_reasoning=include_reasoning,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name_for(samples_path)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path} ({sample_count} sample(s))")


def main() -> None:
    p = argparse.ArgumentParser(description="Pretty-print reasoning traces from samples_*.jsonl file(s)")
    p.add_argument("samples", type=Path, help="Path to a samples_*.jsonl file or directory")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parent directory (default: stdout for file input, input directory for directory input). "
        "File name is derived from the input by replacing 'samples' with 'traces' "
        "and changing the extension to .txt.",
    )
    p.add_argument("--doc-id", type=int, default=None, help="Render only this doc_id")
    p.add_argument("--only-wrong", action="store_true", help="Only render samples where accuracy.is_correct is false")
    p.add_argument("--no-reasoning", action="store_true", help="Omit the reasoning trace field")
    args = p.parse_args()

    if args.samples.is_dir():
        sample_paths = sorted(args.samples.glob("samples_*.jsonl"))
        assert sample_paths, f"no samples_*.jsonl files found in: {args.samples}"
        out_dir = args.output or args.samples
        for sample_path in sample_paths:
            write_trace_file(
                sample_path,
                out_dir=out_dir,
                doc_id=args.doc_id,
                only_wrong=args.only_wrong,
                include_reasoning=not args.no_reasoning,
            )
    else:
        assert args.samples.is_file(), f"not a file or directory: {args.samples}"
        text, sample_count = render_samples_file(
            args.samples,
            doc_id=args.doc_id,
            only_wrong=args.only_wrong,
            include_reasoning=not args.no_reasoning,
        )
        if args.output is None:
            sys.stdout.write(text)
        else:
            out_dir = args.output
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / output_name_for(args.samples)
            out_path.write_text(text, encoding="utf-8")
            print(f"Wrote {out_path} ({sample_count} sample(s))")


if __name__ == "__main__":
    main()
