"""Constants for data_v2 benchmark rendering.

Matches the reference Google Sheet layout:
  - Numerical table: Easy(5s) / Med(10s) / Hard(20s) / Chall(1min)
  - MCQ table: 5sec / 10sec / 20sec / 1min
"""

from __future__ import annotations

import re

NUMERIC_TASKS = (
    "block_counting",
    "make_coffee",
    "ring_toss",
    "tighten_untighten",
    "hidden_dice_roll",
    "rhythm_game",
    "sugar_new",
    "hockey_own_goal",
)

NUMERIC_DISPLAY = {
    "block_counting": "Block Count",
    "make_coffee": "Make coffee",
    "ring_toss": "Ring toss",
    "tighten_untighten": "tighten/loosen",
    "hidden_dice_roll": "Dice Roll",
    "rhythm_game": "Rhythm",
    "sugar_new": "Sugar (new)",
    "hockey_own_goal": "Hockey (own goal)",
}

MCQ_TABLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Shell Game", "shell_game"),
    ("Shell Game (rotate)", "shell_game_rotate"),
    ("Slide Puzzle", "memory_sliding_puzzle"),
    ("Tilt Box", "tilt_box"),
    ("Morse Code", "morse"),
    ("Hockey (score)", "hockey_score"),
    ("Hockey (longest)", "hockey_longest_game"),
    ("Pinwheel", "pinwheel"),
    ("Opaque", "opaque"),
)

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
    "llava_onevision_7b": "LLaVA-OV-7B",
    "llava_onevision_0.5b": "LLaVA-OV-0.5B",
}

DURATIONS = ("5sec", "10sec", "20sec")
NUM_DUR_LABELS = ("Easy", "Med", "Hard", "Chall")
MCQ_DUR_LABELS = ("5sec", "10sec", "20sec", "1min")

_MCQ_JSON_KEYS = tuple({jk for _, jk in MCQ_TABLE_COLUMNS})
TASK_IDS_BY_SUFFIX = tuple(
    sorted(set(NUMERIC_TASKS) | set(_MCQ_JSON_KEYS), key=len, reverse=True)
)

STRETCH_RE = re.compile(r"stretch")
RUN_TS = re.compile(r"^\d{8}_\d{6}$")
LMMS_RUN = re.compile(r"(\d{4}_\d{4})")
