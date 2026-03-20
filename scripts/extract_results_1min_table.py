"""Helpers for rendering 1-minute LongVid summary tables."""

import json
from pathlib import Path

DEFAULT_RESULTS_ROOT = Path("/nas2/longvideo_eval/longvid-reasoning-eval/results/1min/model_defaults")
DEFAULT_CHANCE_PATH = Path("/nas2/longvideo_eval/longvid-reasoning-eval/results/1min/chance_levels.json")
SUMMARY_FILENAME = "longvid_reasoning_eval_1min_summary.md"
TASK_PREFIX = "longvid-reasoning-eval_1min_"
NUMERIC_TASKS = ["block_counting", "make_coffee", "ring_toss_counting_physics", "tighten_untighten", "hidden_dice_roll", "rhythm_game", "tilt_box"]
MCQ_TASKS = ["shell_game_rotate", "memory_sliding_puzzle"]
ALL_TASKS = NUMERIC_TASKS + MCQ_TASKS
TASK_HEADERS = {
    "block_counting": "Block<br>Count", "make_coffee": "Make<br>Coffee", "ring_toss_counting_physics": "Ring<br>Toss",
    "tighten_untighten": "Tighten/<br>Untighten", "hidden_dice_roll": "Dice<br>Roll", "rhythm_game": "Rhythm",
    "tilt_box": "Tilt<br>Box", "shell_game_rotate": "Shell<br>Game", "memory_sliding_puzzle": "Slide<br>Puzzle",
}
OPEN_SOURCE_MODELS = [
    ("internvl3p5_8b", "InternVL3.5-8B"), ("internvl3p5_2b", "InternVL3.5-2B"), ("qwen3vl_8b", "Qwen3-8B"),
    ("qwen3vl_4b", "Qwen3-4B"), ("qwen3vl_2b", "Qwen3-2B"), ("cambrians_7b", "Cambrian-S-7B"),
    ("cambrians_3b", "Cambrian-S-3B"), ("cambrians_1p5b", "Cambrian-S-1.5B"), ("llava_onevision_7b", "LLaVA-OV-7B"),
    ("llava_onevision_0.5b", "LLaVA-OV-0.5B"),
]
API_MODELS = [("gpt5p4", "GPT-5.4"), ("gemini3p1_pro_preview", "Gemini-3.1-Pro"), ("gemini3_flash", "Gemini-3.0-Flash")]
MODEL_SECTIONS = [("Proprietary Models (API)", API_MODELS), ("Open-Source Models", OPEN_SOURCE_MODELS)]
METRIC_COLUMNS = ["avg"] + ALL_TASKS
BEST_COLOR = "#E9E9E9"
SECTION_COLOR = "#F5F5F5"
RANK_COLORS = {1: "#7FD96B", 2: "#B7EEA9", 3: "#E4F7DC"}

def compute_avg(scores, task_names, require_all):
    values = [scores.get(task) for task in task_names]
    if require_all and any(value is None for value in values):
        return None
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None

def build_scores(raw_scores):
    scores = {task: raw_scores.get(task) for task in ALL_TASKS}
    scores["avg"] = compute_avg(scores, ALL_TASKS, require_all=True)
    return scores

def find_latest_results(results_root, model_key):
    model_dir = results_root / model_key
    if not model_dir.exists():
        return None
    results_files = sorted(model_dir.rglob("*_results.json"))
    return results_files[-1] if results_files else None

def extract_scores(results_path):
    results = json.loads(results_path.read_text(encoding="utf-8")).get("results", {})
    raw_scores = {}
    for task in ALL_TASKS:
        metric_key = "accuracy,none" if task in MCQ_TASKS else "MRA:.5:.95:.05,none"
        raw_scores[task] = results.get(TASK_PREFIX + task, {}).get(metric_key)
    return build_scores(raw_scores)

def collect_model_rows(results_root, models):
    rows = []
    for model_key, display_name in models:
        results_path = find_latest_results(results_root, model_key)
        rows.append((model_key, display_name, build_scores({}) if results_path is None else extract_scores(results_path)))
    return rows

def load_chance_rows(results_root):
    chance_path = results_root / "chance_levels.json"
    chance_path = chance_path if chance_path.exists() else DEFAULT_CHANCE_PATH
    if not chance_path.exists():
        return []
    chance_data = json.loads(chance_path.read_text(encoding="utf-8"))
    rows = []
    if "random" in chance_data:
        rows.append(("Chance Level (Random)", build_scores(chance_data["random"])))
    if "frequency" in chance_data:
        rows.append(("Chance Level (Frequency)", build_scores(chance_data["frequency"])))
    return rows

def build_rank_map(rows):
    ranked = sorted([(model_key, scores["avg"]) for model_key, _, scores in rows if scores["avg"] is not None], key=lambda item: item[1], reverse=True)
    rank_map = {}
    last_score = None
    current_rank = 0
    for index, (model_key, score) in enumerate(ranked, start=1):
        if last_score is None or abs(score - last_score) > 1e-12:
            current_rank = index
            last_score = score
        rank_map[model_key] = current_rank
    return rank_map

def build_best_map(rows):
    best_map = {}
    for column in METRIC_COLUMNS:
        values = [scores[column] for _, _, scores in rows if scores[column] is not None]
        if values:
            best_map[column] = max(values)
    return best_map

def format_score(value):
    return "-" if value is None else f"{value * 100:.1f}"

def cell(tag, text, align="center", colspan=None, rowspan=None, bgcolor=None):
    attrs = [f'align="{align}"']
    if colspan is not None:
        attrs.append(f'colspan="{colspan}"')
    if rowspan is not None:
        attrs.append(f'rowspan="{rowspan}"')
    if bgcolor is not None:
        attrs.append(f'bgcolor="{bgcolor}"')
    return f"<{tag} {' '.join(attrs)}>{text}</{tag}>"

def section_header(title):
    return "    <tr>" + cell("td", f"<strong><em>{title}</em></strong>", align="left", colspan=3 + len(ALL_TASKS), bgcolor=SECTION_COLOR) + "</tr>"

def render_score_cell(value, best_value=None):
    text = format_score(value)
    highlight = value is not None and best_value is not None and best_value > 0 and abs(value - best_value) < 1e-12
    if highlight:
        text = f"<strong>{text}</strong>"
    return cell("td", text, align="right", bgcolor=BEST_COLOR if highlight else None)

def render_baseline_row(label, scores):
    cells = [cell("td", label, align="left"), cell("td", "-")]
    cells.extend(render_score_cell(scores[column]) for column in METRIC_COLUMNS)
    return "    <tr>" + "".join(cells) + "</tr>"

def render_model_row(label, scores, rank, best_map):
    cells = [cell("td", label, align="left"), cell("td", "-" if rank is None else str(rank), bgcolor=RANK_COLORS.get(rank))]
    cells.extend(render_score_cell(scores[column], best_map.get(column)) for column in METRIC_COLUMNS)
    return "    <tr>" + "".join(cells) + "</tr>"

def render_table(chance_rows, model_sections):
    lines = [
        "<table>", "  <thead>", "    <tr>", "      " + cell("th", "Method", align="left", rowspan=2),
        "      " + cell("th", "Rank", rowspan=2), "      " + cell("th", "Avg.", rowspan=2),
        "      " + cell("th", "Numerical Answer", colspan=len(NUMERIC_TASKS)),
        "      " + cell("th", "Multiple-Choice Answer", colspan=len(MCQ_TASKS)), "    </tr>", "    <tr>",
    ]
    lines.extend("      " + cell("th", TASK_HEADERS[task]) for task in ALL_TASKS)
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])
    if chance_rows:
        lines.append(section_header("Baseline"))
        lines.extend(render_baseline_row(label, scores) for label, scores in chance_rows)
    for title, rows in model_sections:
        lines.append(section_header(title))
        best_map = build_best_map(rows)
        rank_map = build_rank_map(rows)
        lines.extend(render_model_row(label, scores, rank_map.get(model_key), best_map) for model_key, label, scores in rows)
    lines.extend(["  </tbody>", "</table>"])
    return lines

def build_markdown(results_root):
    chance_rows = load_chance_rows(results_root)
    model_sections = []
    for title, models in MODEL_SECTIONS:
        rows = collect_model_rows(results_root, models)
        if title.startswith("Proprietary") and not any(scores["avg"] is not None for _, _, scores in rows):
            continue
        model_sections.append((title, rows))
    lines = [
        "# LongVid Reasoning Eval - 1min Results", "",
        "Numerical tasks use **MRA** (x100). Multiple-choice tasks use **Accuracy** (x100).",
        "Ranks are computed within each model section from **Avg.**, which is the mean over all task columns and is left blank for incomplete rows.",
        "", *render_table(chance_rows, model_sections), "", "*Auto-generated by `scripts/extract_results_1min.py`.*",
    ]
    return "\n".join(lines)

def resolve_output_paths(results_root):
    default_path = results_root / SUMMARY_FILENAME
    alias_paths = sorted(results_root.glob("*_longvid_reasoning_eval_1min_summary.md"))
    if default_path.exists():
        output_paths = [default_path, *alias_paths]
    elif alias_paths:
        output_paths = alias_paths
    else:
        output_paths = [default_path]
    return list(dict.fromkeys(output_paths))
