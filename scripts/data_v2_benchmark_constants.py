"""Constants for data_v2 benchmark markdown rendering."""

from __future__ import annotations

import re

NUMERIC_TASKS = (
    "block_counting",
    "make_coffee",
    "ring_toss_counting_physics",
    "tighten_untighten",
    "hidden_dice_roll",
    "rhythm_game",
    "sugar_new",
    "air_hockey",
)

MCQ_TABLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("shell_game", "shell_game"),
    ("shell_game_rotate", "shell_game_rotate"),
    ("shuffle_puzzle", "memory_sliding_puzzle"),
    ("tilt", "tilt_box"),
    ("morse", "morse"),
    ("air_hockey", "air_hockey"),
    ("air_hockey", "air_hockey"),
    ("pinwheel", "pinwheel"),
    ("opaque", "opaque"),
)

TASK_DISPLAY = {
    "block_counting": "block_counting",
    "make_coffee": "make_coffee",
    "ring_toss_counting_physics": "ring",
    "tighten_untighten": "tighten_untighten",
    "hidden_dice_roll": "dice",
    "rhythm_game": "rhythm_game",
    "sugar_new": "sugar_new",
    "air_hockey": "air_hockey",
    "shell_game": "shell_game",
    "shell_game_rotate": "shell_game_rotate",
    "memory_sliding_puzzle": "shuffle_puzzle",
    "tilt_box": "tilt",
    "morse": "morse",
    "pinwheel": "pinwheel",
    "opaque": "opaque",
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

_MCQ_JSON_KEYS = tuple({jk for _, jk in MCQ_TABLE_COLUMNS})
TASK_IDS_BY_SUFFIX = tuple(
    sorted(set(NUMERIC_TASKS) | set(_MCQ_JSON_KEYS), key=len, reverse=True)
)

STRETCH_RE = re.compile(r"stretch")
RUN_TS = re.compile(r"^\d{8}_\d{6}$")
LMMS_RUN = re.compile(r"(\d{4}_\d{4})")
