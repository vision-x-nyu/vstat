"""Task helpers for VPI YouTube boxing/volleyball sequence MCQ evaluation.

Brief description:
Flatten the regenerated VPI sequence-options JSON into lmms_eval video MCQ
documents and score predicted option letters with accuracy, MRA, and MAE.

Usage:
Referenced by longvid-reasoning-eval_ytb.yaml via !function utils.*.

Input spec:
/nas2/benchmarks/vpi/ytb-vids/merged_qa/boxing_volleyball_sequence_options.json
with top-level data -> {boxing: [rows], volleyball: [rows]}. Each row contains
video_path, question, answer as A-D, answer_index, choices, and sequence_answer.

Output spec:
Flat Dataset rows with video_path, question, answer_text, source_task, video_id,
choices, and sequence_answer. Metrics aggregate per-row MCQ correctness.
"""

import json
import os
import re

from datasets import Dataset

FINAL_OPTION_PATTERN = re.compile(r"(?:^|\n)\s*\**\(?\s*([A-D])\s*\)?\**\.?\s*$", re.IGNORECASE)
EXPLICIT_OPTION_PATTERN = re.compile(
    r"(?:answer|option|choice|matches)\s*(?:is|:|=|option|choice)?\W*\(?\s*([A-D])\s*\)?",
    re.IGNORECASE,
)


def _source_payload(dataset):
    if len(dataset) == 1 and "data" in dataset[0]:
        data = dataset[0]["data"]
        if isinstance(data, str):
            return json.loads(data)
        return data
    checksums = getattr(dataset.info, "download_checksums", None) or {}
    paths = [path for path in checksums if path.endswith(".json")]
    assert paths, f"Cannot recover source JSON path from dataset with {len(dataset)} rows"
    with open(paths[0], encoding="utf-8") as f:
        return json.load(f)["data"]


def process_docs(dataset):
    rows = []
    for source_task, docs in _source_payload(dataset).items():
        for doc in docs:
            answer = str(doc["answer"]).strip().upper()
            assert answer in "ABCD", f"Bad MCQ answer for {doc['video_id']}: {answer}"
            rows.append(
                {
                    "source_task": source_task,
                    "video_id": str(doc["video_id"]),
                    "video_path": doc["video_path"],
                    "question": str(doc["question"]).strip(),
                    "choices": doc["choices"],
                    "answer_text": answer,
                    "sequence_answer": doc.get("sequence_answer"),
                }
            )
    return Dataset.from_list(rows)


def doc_to_visual(doc):
    video_path = doc["video_path"]
    assert os.path.exists(video_path), f"Missing video file: {video_path}"
    return [video_path]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    sections = [
        kwargs.get("pre_prompt", "").strip(),
        "Watch the full video carefully before answering.",
        f"Question: {doc['question']}",
        kwargs.get("mcq_post_prompt", "").strip(),
    ]
    return "\n\n".join(section for section in sections if section)


def doc_to_target(doc):
    return doc["answer_text"]


def _normalize_choice(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _extract_option(text, choices=None):
    raw_text = str(text).strip()
    final_match = FINAL_OPTION_PATTERN.search(raw_text)
    if final_match:
        return final_match.group(1).upper()
    explicit_matches = EXPLICIT_OPTION_PATTERN.findall(raw_text)
    if explicit_matches:
        return explicit_matches[-1].upper()
    if choices:
        normalized_text = _normalize_choice(raw_text)
        matched_letters = [
            "ABCD"[index]
            for index, choice in enumerate(choices)
            if _normalize_choice(choice) in normalized_text
        ]
        if len(set(matched_letters)) == 1:
            return matched_letters[0]
    return None


def process_results(doc, results):
    prediction = str(results[0]).strip() if results else ""
    parsed = _extract_option(prediction, doc.get("choices"))
    correct = parsed == doc["answer_text"]
    accuracy = 1.0 if correct else 0.0
    return {
        "accuracy": {"score": accuracy, "parsed": parsed, "target": doc["answer_text"]},
        "mra": {"score": accuracy},
        "mae": {"score": 0.0 if correct else 1.0},
    }


def aggregate_accuracy(results):
    return round(sum(result["score"] for result in results) / len(results), 3) if results else 0.0


def aggregate_mra(results):
    return aggregate_accuracy(results)


def aggregate_mae(results):
    return round(sum(result["score"] for result in results) / len(results), 3) if results else 0.0
