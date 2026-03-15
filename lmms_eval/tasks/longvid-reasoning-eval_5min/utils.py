"""Task helpers for `longvid-reasoning-eval_5min`.

Brief description:
Load the nested 5-minute Blender QA JSON into four lmms_eval subtasks and
score normalized exact-match accuracy on free-form model responses.

Usage:
Referenced by the YAML task configs in this directory via `!function`.

Input spec:
`/nas2/longvideo_eval/blender/data/5min/merged_qa.json` loads as one JSON
document with `data -> 5min -> <task_name> -> list[example]`.

Output spec:
Each subtask exposes flat documents with `video_path`, `question`,
`answer_text`, and task-specific target fields. Metrics return mean
`accuracy`, and per-sample metric logs store only `is_correct`.
"""

import os
import re

from datasets import Dataset

BENCH_KEY = "5min"
ROW_COLUMN_PATTERN = re.compile(r"row\s*(\d+)\s*[,;/]?\s*column\s*(\d+)", re.IGNORECASE)
INTEGER_PATTERN = re.compile(r"\b\d+\b")
SUCCESS_AFTER_PATTERN = re.compile(r"(?:success|successful)[^0-9]*(\d+)", re.IGNORECASE)
SUCCESS_BEFORE_PATTERN = re.compile(r"(\d+)[^0-9]{0,24}(?:success|successful)", re.IGNORECASE)
FAILURE_AFTER_PATTERN = re.compile(r"(?:failure|unsuccessful)[^0-9]*(\d+)", re.IGNORECASE)
FAILURE_BEFORE_PATTERN = re.compile(r"(\d+)[^0-9]{0,24}(?:failure|unsuccessful)", re.IGNORECASE)
TASK_INSTRUCTIONS = {
    "hidden_dice_roll": "Return only the final count as a single integer.\nAnswer:",
    "memory_sliding_puzzle": "Return only the final location in the format `Row x, Column y`.\nAnswer:",
    "ring_toss_counting_physics": "Return only `Success: X, Failure: Y`.\nAnswer:",
    "tilt_box": "Return only the final corner number as a single integer from 1 to 4.\nAnswer:",
}


def _build_task_dataset(dataset, source_task):
    assert len(dataset) == 1, f"Expected one source row, found {len(dataset)}"
    source_docs = dataset[0]["data"][BENCH_KEY][source_task]
    flat_docs = []
    for source_doc in source_docs:
        question = source_doc["question"]
        answer = source_doc["answer"]
        assert question is not None, f"Missing question for {source_task}"
        assert answer is not None, f"Missing answer for {source_task}"
        flat_doc = {
            "source_task": source_task,
            "video_id": str(source_doc["video_id"]),
            "video_path": source_doc["video_path"],
            "question": question.strip(),
        }
        if source_task == "ring_toss_counting_physics":
            success = int(answer["success"])
            failure = int(answer["failure"])
            flat_doc["target_success"] = success
            flat_doc["target_failure"] = failure
            flat_doc["answer_text"] = f"Success: {success}, Failure: {failure}"
        elif source_task == "memory_sliding_puzzle":
            match = ROW_COLUMN_PATTERN.search(str(answer))
            assert match is not None, f"Unexpected row/column answer: {answer}"
            row = int(match.group(1))
            column = int(match.group(2))
            flat_doc["target_row"] = row
            flat_doc["target_column"] = column
            flat_doc["answer_text"] = f"Row {row}, Column {column}"
        else:
            value = int(answer)
            flat_doc["target_value"] = value
            flat_doc["answer_text"] = str(value)
        flat_docs.append(flat_doc)
    return Dataset.from_list(flat_docs)


def process_hidden_dice_roll_docs(dataset):
    return _build_task_dataset(dataset, "hidden_dice_roll")


def process_memory_sliding_puzzle_docs(dataset):
    return _build_task_dataset(dataset, "memory_sliding_puzzle")


def process_ring_toss_counting_physics_docs(dataset):
    return _build_task_dataset(dataset, "ring_toss_counting_physics")


def process_tilt_box_docs(dataset):
    return _build_task_dataset(dataset, "tilt_box")


def longvid_reasoning_eval_5min_doc_to_visual(doc):
    video_path = doc["video_path"]
    assert os.path.exists(video_path), f"Missing video file: {video_path}"
    return [video_path]


def longvid_reasoning_eval_5min_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    body = f"Watch the full video carefully before answering.\n\nQuestion: {doc['question']}\n\n{TASK_INSTRUCTIONS[doc['source_task']]}"
    return f"{pre_prompt}{body}{post_prompt}"


def longvid_reasoning_eval_5min_doc_to_target(doc):
    return doc["answer_text"]


def _extract_last_integer(text, allowed_values=None):
    matches = [int(match) for match in INTEGER_PATTERN.findall(str(text))]
    if allowed_values is not None:
        matches = [match for match in matches if match in allowed_values]
    return matches[-1] if matches else None


def _extract_row_column(text):
    match = ROW_COLUMN_PATTERN.search(str(text))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _extract_keyword_integer(pattern, text):
    match = pattern.search(str(text))
    if match is None:
        return None
    groups = [group for group in match.groups() if group is not None]
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
    matches = [int(match) for match in INTEGER_PATTERN.findall(str(text))]
    return tuple(matches[-2:]) if len(matches) >= 2 else None


def _parse_prediction(doc, prediction):
    source_task = doc["source_task"]
    if source_task == "memory_sliding_puzzle":
        return _extract_row_column(prediction)
    if source_task == "ring_toss_counting_physics":
        return _extract_success_failure(prediction)
    if source_task == "tilt_box":
        return _extract_last_integer(prediction, allowed_values={1, 2, 3, 4})
    return _extract_last_integer(prediction)


def _is_correct(doc, parsed_prediction):
    if parsed_prediction is None:
        return False
    if doc["source_task"] == "memory_sliding_puzzle":
        return parsed_prediction == (doc["target_row"], doc["target_column"])
    if doc["source_task"] == "ring_toss_counting_physics":
        return parsed_prediction == (doc["target_success"], doc["target_failure"])
    return parsed_prediction == doc["target_value"]


def longvid_reasoning_eval_5min_process_results(doc, results):
    prediction = str(results[0]).strip() if results else ""
    parsed_prediction = _parse_prediction(doc, prediction)
    return {"accuracy": {"is_correct": _is_correct(doc, parsed_prediction)}}


def longvid_reasoning_eval_5min_aggregate_accuracy(results):
    return sum(result["is_correct"] for result in results) / len(results) if results else 0.0
