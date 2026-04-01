"""Tab-separated benchmark tables for pasting into spreadsheets."""

from __future__ import annotations

from pathlib import Path

from data_v2_benchmark_constants import (
    DURATIONS,
    MCQ_TABLE_ORDER,
    MODEL_LABEL,
    MODEL_ORDER,
    NUMERIC_TASKS,
    TASK_DISPLAY,
)
from data_v2_benchmark_table import (
    build_matrix,
    fmt_pct,
    random_chance_mcq,
)

DUR_COL = {"5sec": "5s", "10sec": "10s", "20sec": "20s"}


def _row(cells: list[str]) -> str:
    return "\t".join(cells)


def render_tsv(root: Path, include_stretch: bool) -> str:
    matrix = build_matrix(root, include_stretch)
    lines: list[str] = []

    header = ["Group", "Model"]
    for tid in NUMERIC_TASKS:
        name = TASK_DISPLAY[tid]
        header.extend([f"{name} {DUR_COL[d]}" for d in DURATIONS] + [f"{name} Chall"])
    lines.append(_row(header))

    cells = ["Uniform", "Chance level (Random)"]
    for _ in NUMERIC_TASKS:
        cells.extend(["—", "—", "—", "—"])
    lines.append(_row(cells))

    cells = ["Uniform", "Chance level (Frequency)"]
    for _ in NUMERIC_TASKS:
        cells.extend(["—", "—", "—", "—"])
    lines.append(_row(cells))

    for mk in MODEL_ORDER:
        if mk not in matrix:
            continue
        cells = ["Open-source", MODEL_LABEL[mk]]
        for tid in NUMERIC_TASKS:
            for d in DURATIONS:
                cells.append(fmt_pct(matrix[mk][tid].get(d)))
            cells.append("—")
        lines.append(_row(cells))

    lines.append("")

    header = ["Group", "Model"]
    for tid in MCQ_TABLE_ORDER:
        name = TASK_DISPLAY[tid]
        header.extend([f"{name} 5s", f"{name} 10s", f"{name} 20s", f"{name} 1m"])
    lines.append(_row(header))

    cells = ["Uniform", "Chance level (Random)"]
    for tid in MCQ_TABLE_ORDER:
        c = random_chance_mcq(tid)
        cells.extend([c, c, c, "—"])
    lines.append(_row(cells))

    cells = ["Uniform", "Chance level (Frequency)"]
    for _ in MCQ_TABLE_ORDER:
        cells.extend(["—", "—", "—", "—"])
    lines.append(_row(cells))

    for mk in MODEL_ORDER:
        if mk not in matrix:
            continue
        cells = ["Open-source", MODEL_LABEL[mk]]
        for tid in MCQ_TABLE_ORDER:
            for d in DURATIONS:
                cells.append(fmt_pct(matrix[mk][tid].get(d)))
            cells.append("—")
        lines.append(_row(cells))

    return "\n".join(lines) + "\n"
