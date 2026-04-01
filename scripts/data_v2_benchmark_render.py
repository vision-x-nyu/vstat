"""Render benchmark markdown from aggregated v2 matrices."""

from __future__ import annotations

from pathlib import Path

from data_v2_benchmark_constants import (
    DURATIONS,
    MCQ_TABLE_ORDER,
    MODEL_LABEL,
    MODEL_ORDER,
    NUMERIC_TASKS,
)
from data_v2_benchmark_table import (
    build_matrix,
    fmt_pct,
    random_chance_mcq,
    table_header_task_blocks,
)


def render_markdown(root: Path, include_stretch: bool) -> str:
    matrix = build_matrix(root, include_stretch)
    lines: list[str] = []

    title = (
        "LongVid Reasoning Eval v2 (clip lengths)"
        if include_stretch
        else "LongVid Reasoning Eval v2 (non-stretch clip lengths)"
    )
    lines.append(f"# {title}\n")
    if include_stretch:
        lines.append(f"Generated from `{root}` **including** stretch runs.\n")
    else:
        lines.append(
            f"Generated from `{root}`, excluding any path containing `stretch`. "
            "Benchmark YAML defines eight tasks × three clip lengths (5s / 10s / 20s); "
            "there is no 1-minute run and no easy/med/hard challenge split in these JSON files.\n"
        )
    lines.append(
        "- **Numerical table:** columns *Easy / Med / Hard* show accuracies at **5s / 10s / 20s** clips; "
        "*Chall* is not available (—).\n"
    )
    lines.append(
        "- **Multiple-choice table:** *5s / 10s / 20s* filled from data; *1m* is not available (—).\n"
    )
    lines.append(
        "- **Chance (Random)** uses 33.3% for 3-option shell tasks and 25.0% for 4-option tasks "
        "(slide puzzle, tilt box), matching the reference sheet.\n\n---\n"
    )

    lines.append("## Numerical answer (accuracy %)\n")
    h1, sep = table_header_task_blocks(NUMERIC_TASKS, 4)
    lines.extend((h1, sep))
    sub = "| | |"
    for _ in NUMERIC_TASKS:
        sub += " *Easy (5s)* | *Med (10s)* | *Hard (20s)* | *Chall* |"
    lines.append(sub)

    row = "| **Uniform** | Chance level (Random) |"
    for _ in NUMERIC_TASKS:
        row += " — | — | — | — |"
    lines.append(row)
    row = "| **Uniform** | Chance level (Frequency) |"
    for _ in NUMERIC_TASKS:
        row += " — | — | — | — |"
    lines.append(row)

    for mk in MODEL_ORDER:
        if mk not in matrix:
            continue
        row = f"| **Open-source** | {MODEL_LABEL[mk]} |"
        for tid in NUMERIC_TASKS:
            for d in DURATIONS:
                row += f" {fmt_pct(matrix[mk][tid].get(d))} |"
            row += " — |"
        lines.append(row)

    lines.append("\n---\n")
    lines.append("## Multiple-choice answer (accuracy %)\n")
    h1, sep = table_header_task_blocks(MCQ_TABLE_ORDER, 4)
    lines.extend((h1, sep))
    sub = "| | |"
    for _ in MCQ_TABLE_ORDER:
        sub += " *5s* | *10s* | *20s* | *1m* |"
    lines.append(sub)

    row = "| **Uniform** | Chance level (Random) |"
    for tid in MCQ_TABLE_ORDER:
        c = random_chance_mcq(tid)
        row += f" {c} | {c} | {c} | — |"
    lines.append(row)
    row = "| **Uniform** | Chance level (Frequency) |"
    for _ in MCQ_TABLE_ORDER:
        row += " — | — | — | — |"
    lines.append(row)

    for mk in MODEL_ORDER:
        if mk not in matrix:
            continue
        row = f"| **Open-source** | {MODEL_LABEL[mk]} |"
        for tid in MCQ_TABLE_ORDER:
            for d in DURATIONS:
                row += f" {fmt_pct(matrix[mk][tid].get(d))} |"
            row += " — |"
        lines.append(row)

    return "\n".join(lines) + "\n"
