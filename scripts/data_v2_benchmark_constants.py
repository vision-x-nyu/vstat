"""Constants for data_v2 benchmark markdown rendering."""

from __future__ import annotations

import re

NUMERIC_TASKS = (
    "hidden_dice_roll",
    "rhythm_game",
    "ring_toss_counting_physics",
    "tighten_untighten",
)
MCQ_TABLE_ORDER = ("shell_game", "shell_game_rotate", "memory_sliding_puzzle", "tilt_box")

TASK_DISPLAY = {
    "hidden_dice_roll": "Dice Roll",
    "rhythm_game": "Rhythm",
    "ring_toss_counting_physics": "Ring toss",
    "tighten_untighten": "tighten/loosen",
    "shell_game": "Shell Game",
    "shell_game_rotate": "Shell Game (rotate)",
    "memory_sliding_puzzle": "Slide Puzzle",
    "tilt_box": "Tilt Box",
}

MODEL_ORDER = (
    "internvl3p5_8b",
    "internvl3p5_2b",
    "qwen3vl_8b",
    "qwen3vl_4b",
    "qwen3vl_2b",
    "cambrians_7b",
    "cambrians_3b",
    "cambrians_1p5b",
    "llava_onevision_7b",
    "llava_onevision_0.5b",
)

MODEL_LABEL = {
    "internvl3p5_8b": "InternVL3.5-8B",
    "internvl3p5_2b": "InternVL3.5-2B",
    "qwen3vl_8b": "Qwen3-VL-8B",
    "qwen3vl_4b": "Qwen3-VL-4B",
    "qwen3vl_2b": "Qwen3-VL-2B",
    "cambrians_7b": "Cambrian-S-7B",
    "cambrians_3b": "Cambrian-S-3B",
    "cambrians_1p5b": "Cambrian-S-1.5B",
    "llava_onevision_7b": "LLaVA-OneVision-7B",
    "llava_onevision_0.5b": "LLaVA-OneVision-0.5B",
}

DURATIONS = ("5sec", "10sec", "20sec")

TASK_IDS_BY_SUFFIX = tuple(sorted(NUMERIC_TASKS + MCQ_TABLE_ORDER, key=len, reverse=True))

STRETCH_RE = re.compile(r"stretch")
RUN_TS = re.compile(r"^\d{8}_\d{6}$")
LMMS_RUN = re.compile(r"(\d{4}_\d{4})")
