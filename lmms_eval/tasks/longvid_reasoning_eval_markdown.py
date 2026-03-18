"""Markdown logging helpers for longvid reasoning task reports.

Brief description:
Write per-task markdown reports with aggregate scores and per-sample details.

Usage:
Imported by `longvid_reasoning_eval_utils.py` during metric aggregation.

Input spec:
Aggregated metric payloads containing task metadata, question text, choices,
raw predictions, parsed predictions, and per-sample scores.

Output spec:
Markdown files written to `<output_path>/eval_markdown/<task_id>.md` or to
`./eval_markdown/<task_id>.md` when no output path is provided.
"""

import os
import sys
from pathlib import Path

from loguru import logger as eval_logger


def write_markdown_report(results, metric_name, score, format_options_block):
    output_dir = _resolve_markdown_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = results[0]["task_id"]
    report_path = output_dir / f"{task_id}.md"
    lines = [f"# {task_id}", "", f"- Metric: `{metric_name}`", f"- Aggregate score: `{score:.4f}`", f"- Samples: `{len(results)}`", ""]
    for index, result in enumerate(results, start=1):
        lines.extend([f"## Sample {index}", "", f"- Video ID: `{result['video_id']}`", f"- Source task: `{result['source_task']}`", f"- Target: `{result['target']}`", f"- Parsed prediction: `{result['prediction_parsed']}`", f"- Score: `{result['score']:.4f}`", "- Question:", "", "```text", result["question"], "```"])
        if result["choices"] is not None:
            lines.extend(["- Options:", "", "```text", format_options_block(result["choices"]), "```"])
        lines.extend(["- Raw prediction:", "", "```text", result["prediction_raw"] or "<empty>", "```", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    eval_logger.info(f"Saved markdown eval details to {report_path}")


def _resolve_markdown_dir():
    env_path = os.getenv("LMMS_EVAL_MARKDOWN_DIR")
    if env_path:
        return Path(env_path).expanduser()
    for index, arg in enumerate(sys.argv):
        if arg == "--output_path" and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).expanduser() / "eval_markdown"
        if arg.startswith("--output_path="):
            return Path(arg.split("=", 1)[1]).expanduser() / "eval_markdown"
    return Path.cwd() / "eval_markdown"
