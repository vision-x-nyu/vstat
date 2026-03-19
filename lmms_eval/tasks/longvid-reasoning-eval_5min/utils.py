"""Task helpers for longvid-reasoning-eval_5min.

Brief description:
Load the nested 5min Blender QA JSON into lmms_eval subtasks and delegate
prompting and scoring to the shared longvid helpers.

Usage:
Referenced by the YAML task configs in this directory via `!function`.

Input spec:
`/nas2/longvideo_eval/blender/data/5min/merged_qa.json` with
`data -> 5min -> <task_name> -> list[example]`.

Output spec:
Each subtask exposes flat documents with `video_path`, `question`,
`answer_text`, `is_mcq`, `choices`, and either `accuracy` or MRA metrics.
"""

from lmms_eval.tasks.longvid_reasoning_eval_utils import (
    aggregate_accuracy,
    aggregate_mra,
    build_task_dataset,
    doc_to_target,
    doc_to_text,
    doc_to_visual,
    process_results,
)

BENCH_KEY = "5min"


def _build_task_dataset(dataset, source_task):
    return build_task_dataset(dataset, BENCH_KEY, source_task)


def process_block_counting_docs(dataset):
    return _build_task_dataset(dataset, "block_counting")


def process_ring_toss_counting_physics_docs(dataset):
    return _build_task_dataset(dataset, "ring_toss_counting_physics")


def process_hidden_dice_roll_docs(dataset):
    return _build_task_dataset(dataset, "hidden_dice_roll")


def process_rhythm_game_docs(dataset):
    return _build_task_dataset(dataset, "rhythm_game")


def process_shell_game_docs(dataset):
    return _build_task_dataset(dataset, "shell_game")


def process_memory_sliding_puzzle_docs(dataset):
    return _build_task_dataset(dataset, "memory_sliding_puzzle")


def process_tilt_box_docs(dataset):
    return _build_task_dataset(dataset, "tilt_box")
