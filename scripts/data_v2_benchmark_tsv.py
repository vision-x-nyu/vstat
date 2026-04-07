"""Tab-separated benchmark tables matching the reference Google Sheet layout.

Paste directly into the Open-Source Models rows of the spreadsheet.

Numerical block:
  Row 1 (header): empty | empty | block_counting | | | | make_coffee | ...
  Row 2 (sub-header): Directory name | | Easy | Med | Hard | Chall | Easy | ...
  Data rows: Group | Model | values...

MCQ block:
  Same structure but sub-headers are 5sec | 10sec | 20sec | 1min.
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
)

EM = ""


def _row(cells: list[str]) -> str:
    return "\t".join(cells)


def render_tsv(root: Path, include_stretch: bool) -> str:
    matrix = build_matrix(root, include_stretch)
    lines: list[str] = []

    h1 = ["", ""]
    h2 = ["", "Directory Name"]
    for tid in NUMERIC_TASKS:
        name = NUMERIC_DISPLAY[tid]
        h1.extend([name, "", "", ""])
        h2.extend(list(NUM_DUR_LABELS))
    lines.append(_row(h1))
    lines.append(_row(h2))

    for mk in MODEL_ORDER:
        if mk not in matrix:
            continue
        cells = ["Uniform", MODEL_LABEL[mk]]
        for tid in NUMERIC_TASKS:
            for d in DURATIONS:
                cells.append(fmt_pct(matrix[mk][tid].get(d)))
            cells.append(EM)
        lines.append(_row(cells))

    lines.append("")

    h1 = ["", ""]
    h2 = ["", "Directory Name"]
    for disp, _jk in MCQ_TABLE_COLUMNS:
        h1.extend([disp, "", "", ""])
        h2.extend(list(MCQ_DUR_LABELS))
    lines.append(_row(h1))
    lines.append(_row(h2))

    for mk in MODEL_ORDER:
        if mk not in matrix:
            continue
        cells = ["Uniform", MODEL_LABEL[mk]]
        for _disp, jk in MCQ_TABLE_COLUMNS:
            for d in DURATIONS:
                cells.append(fmt_pct(matrix[mk][jk].get(d)))
            cells.append(EM)
        lines.append(_row(cells))

    return "\n".join(lines) + "\n"
