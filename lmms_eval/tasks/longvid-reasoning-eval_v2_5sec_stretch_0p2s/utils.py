"""Task helpers for longvid-reasoning-eval_v2_5sec_stretch_0p2s.

Brief description:
Load the nested VPI (data_v2) Blender QA JSON into lmms_eval subtasks and
score exact-match accuracy. MCQ tasks compare predicted option letters; free-form
tasks parse integers or success / failure pairs.

Usage:
Referenced by the YAML task configs in this directory via `!function`.

Input spec:
`/nas2/longvideo_eval/blender/data_v2/merged_qa/5sec_stretch_0p2s.json` with
`data -> 5sec_stretch_0p2s -> <task_name> -> list[example]`.
Run `export_merged_qa.py --data-dir .../blender/data_v2` to build this file.

Output spec:
Each subtask exposes flat documents with `video_path`, `question`,
`answer_text`, `is_mcq`, `choices`. Metrics return mean `accuracy`.
"""

import os
import re

from datasets import Dataset

BENCH_KEY = "5sec_stretch_0p2s"
OPTION_LETTERS = "ABCD"
INTEGER_PATTERN = re.compile(r"\b\d+\b")
SUCCESS_AFTER_PATTERN = re.compile(r"(?:success|successful)[^0-9]*(\d+)", re.IGNORECASE)
SUCCESS_BEFORE_PATTERN = re.compile(r"(\d+)[^0-9]{0,24}(?:success|successful)", re.IGNORECASE)
FAILURE_AFTER_PATTERN = re.compile(r"(?:failure|unsuccessful)[^0-9]*(\d+)", re.IGNORECASE)
FAILURE_BEFORE_PATTERN = re.compile(r"(\d+)[^0-9]{0,24}(?:failure|unsuccessful)", re.IGNORECASE)
MCQ_LETTER_PATTERN = re.compile(r"\b([A-D])\b")
MCQ_TASKS = frozenset({
    "memory_sliding_puzzle",
    "opaque",
    "shell_game",
    "shell_game_rotate",
    "tilt_box",
})
TASK_INSTRUCTIONS = {
    "hidden_dice_roll": "Return only the final count as a single integer.\nAnswer:",
    "memory_sliding_puzzle": "",
    "ring_toss_counting_physics": "Return only `Success: X, Failure: Y`.\nAnswer:",
    "tilt_box": "",
    "shell_game": "",
    "shell_game_rotate": "",
    "opaque": "",
    "rhythm_game": "Return only the final count as a single integer.\nAnswer:",
    "block_counting": "Return only the final count as a single integer.\nAnswer:",
    "make_coffee": "Return only the final count as a single integer.\nAnswer:",
    "tighten_untighten": "Return only the final count as a single integer.\nAnswer:",
}
TILT_BOX_CHOICES = ["1", "2", "3", "4"]


def _build_task_dataset(dataset, source_task):
    assert len(dataset) == 1, f"Expected one source row, found {len(dataset)}"
    source_docs = dataset[0]["data"][BENCH_KEY][source_task]
    flat_docs = []
    for doc in source_docs:
        question = doc["question"]
        answer = doc["answer"]
        assert question is not None, f"Missing question for {source_task}"
        assert answer is not None, f"Missing answer for {source_task}"
        question_text = question.strip()
        is_mcq = source_task in MCQ_TASKS
        flat_doc = {
            "source_task": source_task,
            "video_id": str(doc["video_id"]),
            "video_path": doc["video_path"],
            "question": question_text,
            "is_mcq": is_mcq,
            "choices": doc.get("choices"),
            "answer_index": doc.get("answer_index"),
        }
        if source_task == "tilt_box":
            choices = list(doc.get("choices") or TILT_BOX_CHOICES)
            flat_doc["choices"] = choices
            flat_doc["question"] = _ensure_mcq_question(question_text, choices)
            flat_doc["answer_text"] = _resolve_tilt_box_answer(answer, doc.get("answer_index"))
        elif is_mcq:
            choices = doc.get("choices")
            assert choices is not None, f"Missing choices for {source_task}/{doc['video_id']}"
            flat_doc["choices"] = choices
            flat_doc["answer_text"] = str(answer).strip()
        elif source_task == "ring_toss_counting_physics":
            success = int(answer["success"])
            failure = int(answer["failure"])
            flat_doc["target_success"] = success
            flat_doc["target_failure"] = failure
            flat_doc["answer_text"] = f"Success: {success}, Failure: {failure}"
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


process_block_counting_docs = _make_processor("block_counting")
process_hidden_dice_roll_docs = _make_processor("hidden_dice_roll")
process_make_coffee_docs = _make_processor("make_coffee")
process_memory_sliding_puzzle_docs = _make_processor("memory_sliding_puzzle")
process_opaque_docs = _make_processor("opaque")
process_rhythm_game_docs = _make_processor("rhythm_game")
process_ring_toss_counting_physics_docs = _make_processor("ring_toss_counting_physics")
process_shell_game_docs = _make_processor("shell_game")
process_shell_game_rotate_docs = _make_processor("shell_game_rotate")
process_tighten_untighten_docs = _make_processor("tighten_untighten")
process_tilt_box_docs = _make_processor("tilt_box")


def doc_to_visual(doc):
    video_path = doc["video_path"]
    assert os.path.exists(video_path), f"Missing video file: {video_path}"
    return [video_path]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    instruction = TASK_INSTRUCTIONS.get(doc["source_task"], "")
    body = f"Watch the full video carefully before answering.\n\nQuestion: {doc['question']}"
    if instruction:
        body += f"\n\n{instruction}"
    return f"{pre_prompt}{body}{post_prompt}"


def doc_to_target(doc):
    return doc["answer_text"]


def _ensure_mcq_question(question, choices):
    question = question.strip()
    if "(A)" in question and "Please answer with the letter" in question:
        return question
    option_lines = [f"({OPTION_LETTERS[index]}) {choice}" for index, choice in enumerate(choices)]
    return question + "\n\n" + "\n".join(option_lines) + "\n\nPlease answer with the letter (A, B, C, or D)."


def _resolve_tilt_box_answer(answer, answer_index):
    answer_text = str(answer).strip().upper()
    if answer_text in OPTION_LETTERS:
        return answer_text
    if answer_index is not None:
        index = int(answer_index)
        assert 0 <= index < len(OPTION_LETTERS), f"Bad answer_index: {answer_index}"
        return OPTION_LETTERS[index]
    corner = int(answer)
    assert 1 <= corner <= len(TILT_BOX_CHOICES), f"Bad tilt_box answer: {answer}"
    return OPTION_LETTERS[corner - 1]


def _extract_last_integer(text):
    matches = [int(m) for m in INTEGER_PATTERN.findall(str(text))]
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


def _extract_mcq_letter(text):
    matches = MCQ_LETTER_PATTERN.findall(str(text).upper())
    return matches[-1] if matches else None


def _parse_prediction(doc, prediction):
    if doc["is_mcq"]:
        return _extract_mcq_letter(prediction)
    if doc["source_task"] == "ring_toss_counting_physics":
        return _extract_success_failure(prediction)
    return _extract_last_integer(prediction)


def _is_correct(doc, parsed_prediction):
    if parsed_prediction is None:
        return False
    if doc["is_mcq"]:
        return parsed_prediction == doc["answer_text"]
    if doc["source_task"] == "ring_toss_counting_physics":
        return parsed_prediction == (doc["target_success"], doc["target_failure"])
    return parsed_prediction == doc.get("target_value")


def process_results(doc, results):
    prediction = str(results[0]).strip() if results else ""
    parsed_prediction = _parse_prediction(doc, prediction)
    return {"accuracy": {"is_correct": _is_correct(doc, parsed_prediction)}}


def aggregate_accuracy(results):
    return sum(r["is_correct"] for r in results) / len(results) if results else 0.0
