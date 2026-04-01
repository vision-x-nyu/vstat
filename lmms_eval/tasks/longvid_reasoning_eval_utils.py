"""Shared helpers for longvid reasoning tasks.

Brief description:
Build flat docs for long video reasoning benchmarks and score numeric tasks
with MRA while scoring MCQ tasks with accuracy.

Usage:
Imported by the duration-specific task utils in `longvid-reasoning-eval_*`.

Input spec:
Nested Blender QA examples with `question`, `answer`, `choices`,
`answer_index`, `video_id`, and `video_path`.

Output spec:
Flat docs with `task_id`, `source_task`, `video_id`, `video_path`, prompt
fields, and either `accuracy` or `MRA:.5:.95:.05` metric payloads.
Aggregated accuracy/MRA means use AGGREGATE_DECIMALS (3) on the [0, 1] scale.
"""

import os
import re

from datasets import Dataset
MRA_METRIC = "MRA:.5:.95:.05"
MCQ_TASKS = frozenset({"memory_sliding_puzzle", "opaque", "shell_game", "shell_game_rotate"})
TASK_INSTRUCTIONS = {
    "hidden_dice_roll": "Return only the final count as a single number.\nAnswer:",
    "memory_sliding_puzzle": "Answer with the option letter only.\nAnswer:",
    "ring_toss_counting_physics": "Return only the number of successful tosses as a single number.\nAnswer:",
    "tilt_box": "Return only the final corner number as a single number from 1 to 4.\nAnswer:",
    "shell_game": "Answer with the option letter only.\nAnswer:",
    "shell_game_rotate": "Answer with the option letter only.\nAnswer:",
    "opaque": "Answer with the option letter only.\nAnswer:",
    "rhythm_game": "Return only the final count as a single number.\nAnswer:",
    "block_counting": "Return only the final count as a single number.\nAnswer:",
    "make_coffee": "Return only the final count as a single number.\nAnswer:",
    "tighten_untighten": "Return only the final count as a single number.\nAnswer:",
}
MCQ_BLOCK_PATTERN = re.compile(r"\n\s*\n\([A-Z]\)\s*")
MCQ_TRAILER_PATTERN = re.compile(r"\n\s*Please answer with the letter.*$", re.IGNORECASE | re.S)
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
SUCCESS_AFTER_PATTERN = re.compile(r"(?:success|successful)[^0-9]*(\d+(?:\.\d+)?)", re.IGNORECASE)
SUCCESS_BEFORE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)[^0-9]{0,24}(?:success|successful)", re.IGNORECASE)
MCQ_LETTER_PATTERN = re.compile(r"\b([A-Z])\b")
OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def build_task_dataset(dataset, bench_key, source_task):
    assert len(dataset) == 1, f"Expected one source row, found {len(dataset)}"
    source_docs = dataset[0]["data"][bench_key][source_task]
    task_id = f"longvid-reasoning-eval_{bench_key}_{source_task}"
    flat_docs = []
    for doc in source_docs:
        question = doc["question"]
        answer = doc["answer"]
        assert question is not None, f"Missing question for {source_task}"
        assert answer is not None, f"Missing answer for {source_task}"
        is_mcq = source_task in MCQ_TASKS
        flat_doc = {
            "task_id": task_id,
            "source_task": source_task,
            "video_id": str(doc["video_id"]),
            "video_path": doc["video_path"],
            "question": question.strip(),
            "is_mcq": is_mcq,
            "choices": doc["choices"],
        }
        if is_mcq:
            flat_doc["answer_text"] = str(answer)
        else:
            value = int(answer["success"]) if source_task == "ring_toss_counting_physics" else float(answer)
            flat_doc["question"] = "How many successful tosses occurred?" if source_task == "ring_toss_counting_physics" else flat_doc["question"]
            flat_doc["target_value"] = float(value)
            flat_doc["answer_text"] = _format_number(float(value))
        flat_docs.append(flat_doc)
    return Dataset.from_list(flat_docs)


def doc_to_visual(doc):
    video_path = doc["video_path"]
    assert os.path.exists(video_path), f"Missing video file: {video_path}"
    return [video_path]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    question = _clean_question(doc)
    sections = [pre_prompt.strip(), "Watch the full video carefully before answering.", f"Question: {question}"]
    if doc["is_mcq"]:
        sections.append(_format_options_block(doc["choices"]))
    sections.append(TASK_INSTRUCTIONS[doc["source_task"]])
    if post_prompt:
        sections.append(post_prompt.strip())
    return "\n\n".join(section for section in sections if section)


def doc_to_target(doc):
    return doc["answer_text"]


def process_results(doc, results):
    prediction_raw = str(results[0]).strip() if results else ""
    prediction_parsed = _parse_prediction(doc, prediction_raw)
    if doc["is_mcq"]:
        metric_name = "accuracy"
        score = 1.0 if prediction_parsed == doc["answer_text"] else 0.0
    else:
        metric_name = MRA_METRIC
        score = 0.0 if prediction_parsed is None else mean_relative_accuracy(prediction_parsed, doc["target_value"])
    return {metric_name: _build_metric_payload(doc, prediction_raw, prediction_parsed, score)}


def aggregate_accuracy(results):
    return _aggregate_metric(results, "accuracy")


def aggregate_mra(results):
    return _aggregate_metric(results, MRA_METRIC)


def mean_relative_accuracy(prediction, target, start=0.5, end=0.95, interval=0.05):
    assert target > 0, f"Expected positive target for MRA, found {target}"
    steps = int(round((end - start) / interval)) + 1
    thresholds = [start + step * interval for step in range(steps)]
    relative_error = abs(prediction - target) / target
    return sum(relative_error <= 1 - threshold for threshold in thresholds) / len(thresholds)


def _clean_question(doc):
    question = doc["question"].strip()
    if not doc["is_mcq"]:
        return question
    question = MCQ_BLOCK_PATTERN.split(question, maxsplit=1)[0].strip()
    return MCQ_TRAILER_PATTERN.sub("", question).strip()


def _format_options_block(choices):
    assert choices is not None, "MCQ task is missing choices"
    assert len(choices) <= len(OPTION_LETTERS), f"Too many choices: {len(choices)}"
    options = [f"({OPTION_LETTERS[index]}) {choice}" for index, choice in enumerate(choices)]
    return "Options:\n" + "\n".join(options)


def _parse_prediction(doc, prediction):
    if doc["is_mcq"]:
        return _extract_mcq_letter(prediction)
    if doc["source_task"] == "ring_toss_counting_physics":
        return _extract_success_count(prediction)
    allowed_values = {1, 2, 3, 4} if doc["source_task"] == "tilt_box" else None
    return _extract_last_number(prediction, allowed_values=allowed_values)


def _extract_last_number(text, allowed_values=None):
    numbers = [float(match) for match in NUMBER_PATTERN.findall(str(text))]
    if allowed_values is not None:
        numbers = [value for value in numbers if value.is_integer() and int(value) in allowed_values]
    return numbers[-1] if numbers else None


def _extract_success_count(text):
    text = str(text)
    for pattern in (SUCCESS_AFTER_PATTERN, SUCCESS_BEFORE_PATTERN):
        match = pattern.search(text)
        if match is not None:
            return float(match.group(1))
    numbers = [float(match) for match in NUMBER_PATTERN.findall(text)]
    return numbers[0] if numbers else None


def _extract_mcq_letter(text):
    matches = MCQ_LETTER_PATTERN.findall(str(text).upper())
    return matches[-1] if matches else None


def _build_metric_payload(doc, prediction_raw, prediction_parsed, score):
    return {
        "score": score,
        "task_id": doc["task_id"],
        "video_id": doc["video_id"],
        "source_task": doc["source_task"],
        "question": _clean_question(doc),
        "choices": doc["choices"],
        "target": doc["answer_text"],
        "prediction_raw": prediction_raw,
        "prediction_parsed": _stringify_prediction(prediction_parsed),
    }


AGGREGATE_DECIMALS = 3


def _aggregate_metric(results, metric_name):
    if not results:
        return 0.0
    mean = sum(result["score"] for result in results) / len(results)
    return round(mean, AGGREGATE_DECIMALS)


def _format_number(value):
    return f"{float(value):g}"


def _stringify_prediction(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return _format_number(value)
    return str(value)
