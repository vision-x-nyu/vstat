"""Task helpers for longvid-reasoning-eval_5sec_0p2s.

Shell game rotate only (5-second stretched videos).
"""

import os
import re

from datasets import Dataset

BENCH_KEY = "5sec"
OPTION_LETTERS = "ABCD"
INTEGER_PATTERN = re.compile(r"\b\d+\b")
MCQ_LETTER_PATTERN = re.compile(r"\b([A-D])\b")
SUCCESS_AFTER_PATTERN = re.compile(r"(?:success|successful)[^0-9]*(\d+)", re.IGNORECASE)
SUCCESS_BEFORE_PATTERN = re.compile(r"(\d+)[^0-9]{0,24}(?:success|successful)", re.IGNORECASE)
FAILURE_AFTER_PATTERN = re.compile(r"(?:failure|unsuccessful)[^0-9]*(\d+)", re.IGNORECASE)
FAILURE_BEFORE_PATTERN = re.compile(r"(\d+)[^0-9]{0,24}(?:failure|unsuccessful)", re.IGNORECASE)
MCQ_NUMBER_PATTERN = re.compile(r"\b([1-4])\b")
MCQ_TASKS = frozenset({
    "shell_game",
    "shell_game_rotate",
    "memory_sliding_puzzle",
    "tilt_box",
    "morse",
    "bulb",
    "opaque",
    "hockey_score",
})
# Tasks where MCQ choices use (1)/(2)/(3)/(4) instead of (A)/(B)/(C)/(D)
MCQ_NUMBER_TASKS = frozenset({"morse"})
TASK_INSTRUCTIONS = {
    "shell_game": "",
    "shell_game_rotate": "",
    "memory_sliding_puzzle": "",
    "tilt_box": "",
    "tighten_untighten": "Return only the final count as a single integer.\nAnswer:",
    "hidden_dice_roll": "Return only the final count as a single integer.\nAnswer:",
    "rhythm_game": "Return only the final count as a single integer.\nAnswer:",
    "morse": "",
    "bulb": "",
    "opaque": "",
    "hockey_score": "",
    "hockey_own_goal": "Return only the final count as a single integer.\nAnswer:",
    "hockey_longest_game": "Return only the game number as a single integer.\nAnswer:",
    "sugar_new": "Return only the final count as a single integer.\nAnswer:",
    "block_counting": "Return only the final count as a single integer.\nAnswer:",
}


def _build_task_dataset(dataset, source_task):
    assert len(dataset) == 1, f"Expected one source row, found {len(dataset)}"
    source_docs = dataset[0]["data"][source_task]
    flat_docs = []
    for doc in source_docs:
        question = doc["question"]
        answer = doc["answer"]
        assert question is not None, f"Missing question for {source_task}"
        assert answer is not None, f"Missing answer for {source_task}"
        is_mcq = source_task in MCQ_TASKS
        flat_doc = {
            "source_task":  source_task,
            "video_id":     str(doc["video_id"]),
            "video_path":   doc["video_path"],
            "question":     question.strip(),
            "is_mcq":       is_mcq,
            "choices":      doc.get("choices"),
            "answer_index": doc.get("answer_index"),
        }
        if is_mcq:
            choices = doc.get("choices")
            assert choices is not None, f"Missing choices for {source_task}/{doc['video_id']}"
            flat_doc["choices"] = choices
            flat_doc["answer_text"] = str(answer).strip()
        else:
            value = int(answer)
            flat_doc["target_value"] = value
            flat_doc["answer_text"] = str(value)
        flat_docs.append(flat_doc)
    return Dataset.from_list(flat_docs)


def _make_processor(source_task):
    def process_docs(dataset):
        return _build_task_dataset(dataset, source_task)
    return process_docs


process_ring_toss_docs = _make_processor("ring_toss")


process_shell_game_docs = _make_processor("shell_game")
process_shell_game_rotate_docs = _make_processor("shell_game_rotate")
process_tighten_untighten_docs = _make_processor("tighten_untighten")
process_memory_sliding_puzzle_docs = _make_processor("memory_sliding_puzzle")
process_make_coffee_docs = _make_processor("make_coffee")
process_hidden_dice_roll_docs = _make_processor("hidden_dice_roll")
process_tilt_box_docs = _make_processor("tilt_box")
process_rhythm_game_docs = _make_processor("rhythm_game")
process_morse_docs = _make_processor("morse")
process_bulb_docs = _make_processor("bulb")
process_opaque_docs = _make_processor("opaque")
process_sugar_new_docs = _make_processor("sugar_new")
process_block_counting_docs = _make_processor("block_counting")
process_hockey_own_goal_docs = _make_processor("hockey_own_goal")
process_hockey_score_docs = _make_processor("hockey_score")


DATA_ROOT      = "/nas2/benchmarks/vpi/blender"
_OLD_DATA_ROOT = "/nas2/longvideo_eval/blender/data_v2"


def doc_to_visual(doc):
    video_path = doc["video_path"]
    if video_path.startswith(_OLD_DATA_ROOT):
        video_path = DATA_ROOT + video_path[len(_OLD_DATA_ROOT):]
    assert os.path.exists(video_path), f"Missing video file: {video_path}"
    return [video_path]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    body = f"Watch the full video carefully before answering.\n\nQuestion: {doc['question']}"
    if doc["is_mcq"]:
        if doc["source_task"] in MCQ_NUMBER_TASKS:
            body += "\n\nPlease answer with the number (1, 2, 3, or 4)."
        else:
            body += "\n\nPlease answer with the letter (A, B, C, or D)."
    else:
        instruction = TASK_INSTRUCTIONS.get(doc["source_task"], "")
        if instruction:
            body += f"\n\n{instruction}"
    return f"{pre_prompt}{body}{post_prompt}"


def doc_to_target(doc):
    return doc["answer_text"]


def _extract_mcq_letter(text):
    matches = MCQ_LETTER_PATTERN.findall(str(text).upper())
    return matches[-1] if matches else None


def _extract_mcq_number(text):
    matches = MCQ_NUMBER_PATTERN.findall(str(text))
    return matches[-1] if matches else None


def _extract_keyword_integer(pattern, text):
    match = pattern.search(str(text))
    if match is None:
        return None
    groups = [g for g in match.groups() if g is not None]
    return int(groups[0]) if groups else None


def _extract_success_failure(text):
    success = _extract_keyword_integer(SUCCESS_AFTER_PATTERN, text)
    if success is None:
        success = _extract_keyword_integer(SUCCESS_BEFORE_PATTERN, text)
    failure = _extract_keyword_integer(FAILURE_AFTER_PATTERN, text)
    if failure is None:
        failure = _extract_keyword_integer(FAILURE_BEFORE_PATTERN, text)
    if success is not None and failure is not None:
        return success, failure
    matches = [int(m) for m in INTEGER_PATTERN.findall(str(text))]
    return tuple(matches[-2:]) if len(matches) >= 2 else None


def _extract_last_integer(text):
    matches = [int(m) for m in INTEGER_PATTERN.findall(str(text))]
    return matches[-1] if matches else None


def _parse_prediction(doc, prediction):
    if doc["is_mcq"]:
        if doc["source_task"] in MCQ_NUMBER_TASKS:
            return _extract_mcq_number(prediction)
        return _extract_mcq_letter(prediction)
    return _extract_last_integer(prediction)


def _is_correct(doc, parsed_prediction):
    if parsed_prediction is None:
        return False
    if doc["is_mcq"]:
        return parsed_prediction == doc["answer_text"]
    return parsed_prediction == doc.get("target_value")


def _compute_mra(doc, parsed_prediction):
    """Mean Relative Accuracy: max(0, 1 - |pred - gt| / gt). Only for numerical tasks."""
    if doc["is_mcq"]:
        return None
    gt = doc.get("target_value")
    if gt is None or parsed_prediction is None:
        return 0.0
    if gt == 0:
        return 1.0 if parsed_prediction == 0 else 0.0
    return max(0.0, 1.0 - abs(parsed_prediction - gt) / abs(gt))


def _compute_mae(doc, parsed_prediction):
    """Mean Absolute Error. Only for numerical tasks."""
    if doc["is_mcq"]:
        return None
    gt = doc.get("target_value")
    if gt is None or parsed_prediction is None:
        return None
    return abs(parsed_prediction - gt)


def process_results(doc, results):
    prediction = str(results[0]).strip() if results else ""
    parsed_prediction = _parse_prediction(doc, prediction)
    result = {"accuracy": {"is_correct": _is_correct(doc, parsed_prediction)}}
    mra = _compute_mra(doc, parsed_prediction)
    if mra is not None:
        result["mra"] = {"mra_score": mra}
    mae = _compute_mae(doc, parsed_prediction)
    if mae is not None:
        result["mae"] = {"mae_score": mae}
    return result


def aggregate_accuracy(results):
    return sum(r["is_correct"] for r in results) / len(results) if results else 0.0


def aggregate_mra(results):
    return sum(r["mra_score"] for r in results) / len(results) if results else 0.0


def aggregate_mae(results):
    return sum(r["mae_score"] for r in results) / len(results) if results else 0.0
