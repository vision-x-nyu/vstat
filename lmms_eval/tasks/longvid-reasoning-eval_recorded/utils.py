"""Task helpers for longvid-reasoning-eval_recorded.

The recorded benchmark has no duration subfolders; video_paths in the merged QA
JSON already point at the absolute on-disk location, so no prefix rewriting is
needed here. Sub-tasks are a mix of MCQ (3- or 4-way) and numerical (signed or
unsigned integers); a few sub-tasks (showing_card_most/least) mix MCQ and
open-ended text within the same sub-task when the ground-truth answer is tied
across multiple suits. The is_mcq flag is therefore determined per-doc.
"""

import os
import re

from datasets import Dataset

INTEGER_PATTERN = re.compile(r"-?\d+")
MCQ_LETTER_PATTERN = re.compile(r"\b([A-D])\b")

# Per-sub-task answer instruction appended after the question body.
# MCQ sub-tasks already embed "(A) ... (B) ..." in the question text so they
# don't need extra instructions. Numerical sub-tasks get a short prompt.
NUMERICAL_INSTRUCTIONS = {
    "book": "Return only a single integer (positive, negative, or 0).\nAnswer:",
    "packing_order_green": "Return only the final count as a single integer.\nAnswer:",
    "packing_order_blue": "Return only the final count as a single integer.\nAnswer:",
    "packing_order_yellow": "Return only the final count as a single integer.\nAnswer:",
    "packing_order_chopsticks": "Return only the final count as a single integer.\nAnswer:",
    "showing_card_count_diamond": "Return only the final count as a single integer.\nAnswer:",
    "showing_card_count_heart": "Return only the final count as a single integer.\nAnswer:",
    "showing_card_count_club": "Return only the final count as a single integer.\nAnswer:",
    "showing_card_count_spade": "Return only the final count as a single integer.\nAnswer:",
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
        choices = doc.get("choices")
        is_mcq = choices is not None
        flat_doc = {
            "source_task": source_task,
            "video_id":    str(doc["video_id"]),
            "video_path":  doc["video_path"],
            "question":    question.strip(),
            "is_mcq":      is_mcq,
            "choices":     choices,
            "answer_index": doc.get("answer_index"),
        }
        if is_mcq:
            flat_doc["answer_text"] = str(answer).strip()
            flat_doc["target_value"] = None
        else:
            # Open-ended: try int; otherwise fall back to string match (e.g. tied
            # showing_card_most/least answers like "diamond, spade").
            text_answer = str(answer).strip()
            try:
                value = int(text_answer)
                flat_doc["target_value"] = value
                flat_doc["answer_text"] = str(value)
            except ValueError:
                flat_doc["target_value"] = None
                flat_doc["answer_text"] = text_answer
        flat_docs.append(flat_doc)
    return Dataset.from_list(flat_docs)


def _make_processor(source_task):
    def process_docs(dataset):
        return _build_task_dataset(dataset, source_task)
    return process_docs


# --- sub-task processors ---------------------------------------------------
process_book_docs                     = _make_processor("book")
process_keyboard_docs                 = _make_processor("keyboard")
process_shell_game_docs               = _make_processor("shell_game")
process_tilt_box_docs                 = _make_processor("tilt_box")
process_morse_docs                    = _make_processor("morse")
process_numberpad_docs                = _make_processor("numberpad")
process_cup_stacking_1st_docs         = _make_processor("cup_stacking_1st")
process_cup_stacking_2nd_docs         = _make_processor("cup_stacking_2nd")
process_cup_stacking_3rd_docs         = _make_processor("cup_stacking_3rd")
process_cup_stacking_4th_docs         = _make_processor("cup_stacking_4th")
process_cup_stacking_5th_docs         = _make_processor("cup_stacking_5th")
process_cup_stacking_6th_docs         = _make_processor("cup_stacking_6th")
process_cup_stacking_7th_docs         = _make_processor("cup_stacking_7th")
process_packing_order_green_docs      = _make_processor("packing_order_green")
process_packing_order_blue_docs       = _make_processor("packing_order_blue")
process_packing_order_yellow_docs     = _make_processor("packing_order_yellow")
process_packing_order_chopsticks_docs = _make_processor("packing_order_chopsticks")
process_showing_card_count_diamond_docs = _make_processor("showing_card_count_diamond")
process_showing_card_count_heart_docs   = _make_processor("showing_card_count_heart")
process_showing_card_count_club_docs    = _make_processor("showing_card_count_club")
process_showing_card_count_spade_docs   = _make_processor("showing_card_count_spade")
process_showing_card_most_docs          = _make_processor("showing_card_most")
process_showing_card_least_docs         = _make_processor("showing_card_least")


def doc_to_visual(doc):
    video_path = doc["video_path"]
    assert os.path.exists(video_path), f"Missing video file: {video_path}"
    return [video_path]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    body = f"Watch the full video carefully before answering.\n\nQuestion: {doc['question']}"
    if not doc["is_mcq"]:
        instruction = NUMERICAL_INSTRUCTIONS.get(doc["source_task"])
        if instruction:
            body += f"\n\n{instruction}"
    return f"{pre_prompt}{body}{post_prompt}"


def doc_to_target(doc):
    return doc["answer_text"]


def _extract_mcq_letter(text):
    matches = MCQ_LETTER_PATTERN.findall(str(text).upper())
    return matches[-1] if matches else None


def _extract_last_integer(text):
    matches = [int(m) for m in INTEGER_PATTERN.findall(str(text))]
    return matches[-1] if matches else None


def _parse_prediction(doc, prediction):
    if doc["is_mcq"]:
        return _extract_mcq_letter(prediction)
    if doc.get("target_value") is not None:
        return _extract_last_integer(prediction)
    # open-ended text: normalized lowercase string match
    return str(prediction).strip().lower()


def _is_correct(doc, parsed_prediction):
    if parsed_prediction is None:
        return False
    if doc["is_mcq"]:
        return parsed_prediction == doc["answer_text"]
    if doc.get("target_value") is not None:
        return parsed_prediction == doc["target_value"]
    # open-ended text: normalize and compare
    return parsed_prediction == str(doc["answer_text"]).strip().lower()


def _compute_mra(doc, parsed_prediction):
    if doc["is_mcq"] or doc.get("target_value") is None:
        return None
    gt = doc["target_value"]
    if parsed_prediction is None:
        return 0.0
    if gt == 0:
        return 1.0 if parsed_prediction == 0 else 0.0
    return max(0.0, 1.0 - abs(parsed_prediction - gt) / abs(gt))


def _compute_mae(doc, parsed_prediction):
    if doc["is_mcq"] or doc.get("target_value") is None:
        return None
    gt = doc["target_value"]
    if parsed_prediction is None:
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
