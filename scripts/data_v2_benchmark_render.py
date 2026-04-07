"""Render benchmark markdown matching the reference Google Sheet layout.

Numerical table:
  Row header: Group | Directory Name | Easy | Med | Hard | Chall | (per task: Easy Med Hard Chall)
  Easy=5s, Med=10s, Hard=20s, Chall=1min (not available in data_v2)

MCQ table:
  Row header: Group | Directory Name | (per task: 5sec 10sec 20sec 1min)
"""

from __future__ import annotations

from pathlib import Path

from data_v2_benchmark_constants import (
    DURATIONS,
    MCQ_DUR_LABELS,
    MCQ_TABLE_COLUMNS,
    MODEL_LABEL,
    MODEL_ORDER,
    NUM_DUR_LABELS,
    NUMERIC_DISPLAY,
    NUMERIC_TASKS,
)
from data_v2_benchmark_table import (
    build_matrix,
    fmt_pct,
    random_chance_mcq,
)

EM = "—"


def _numeric_md(matrix: dict) -> list[str]:
    lines: list[str] = []
    lines.append("## Numerical Answer\n")

    h1 = "| | |"
    h2 = "| Group | Directory Name |"
    sep = "| --- | --- |"
    for tid in NUMERIC_TASKS:
        name = NUMERIC_DISPLAY[tid]
        h1 += f" {name} |" + " |" * 3
        for lbl in NUM_DUR_LABELS:
            h2 += f" {lbl} |"
            sep += " --- |"
    lines.extend((h1, sep, h2))

    for mk in MODEL_ORDER:
        if mk not in matrix:
            continue
        row = f"| Uniform | {MODEL_LABEL[mk]} |"
        for tid in NUMERIC_TASKS:
            for d in DURATIONS:
                row += f" {fmt_pct(matrix[mk][tid].get(d))} |"
            row += f" {EM} |"
        lines.append(row)

    return lines


def _mcq_md(matrix: dict) -> list[str]:
    lines: list[str] = []
    lines.append("## Multiple-choice Answer\n")

    h1 = "| | |"
    h2 = "| Group | Directory Name |"
    sep = "| --- | --- |"
    for disp, _jk in MCQ_TABLE_COLUMNS:
        h1 += f" {disp} |" + " |" * 3
        for lbl in MCQ_DUR_LABELS:
            h2 += f" {lbl} |"
            sep += " --- |"
    lines.extend((h1, sep, h2))

    for mk in MODEL_ORDER:
        if mk not in matrix:
            continue
        row = f"| Uniform | {MODEL_LABEL[mk]} |"
        for _disp, jk in MCQ_TABLE_COLUMNS:
            for d in DURATIONS:
                row += f" {fmt_pct(matrix[mk][jk].get(d))} |"
            row += f" {EM} |"
        lines.append(row)

    return lines


def render_markdown(root: Path, include_stretch: bool) -> str:
    matrix = build_matrix(root, include_stretch)
    lines: list[str] = []

    lines.append("# LongVid Reasoning Eval v2 — Open-Source Models\n")
    lines.append(f"Generated from `{root}`.\n")
    lines.append(
        "Easy = 5s, Med = 10s, Hard = 20s, Chall = 1min (not available). "
        f"{EM} = data not yet available.\n\n---\n"
    )

    lines.extend(_numeric_md(matrix))
    lines.append("\n---\n")
    lines.extend(_mcq_md(matrix))
    lines.append("\n---\n")

    lines.append("### Random chance baselines (MCQ)\n")
    lines.append("| Task | Chance (Random) |")
    lines.append("| --- | ---: |")
    for disp, jk in MCQ_TABLE_COLUMNS:
        lines.append(f"| {disp} | {random_chance_mcq(jk)} |")

    return "\n".join(lines) + "\n"
