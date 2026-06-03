"""Task helpers for VSTAT (vstat).

The benchmark has no duration subfolders. Video paths in the QA JSON are
relative to the local VSTAT folder, and examples are a mix of MCQ,
numerical, and open-ended text rows, so the scoring mode is determined
per doc.
"""

import json
import os
import re
from pathlib import Path

from datasets import Dataset, DatasetDict
from lmms_eval.api.task import ConfigurableTask

import numpy as np

# Video root resolution order:
#   1. $VSTAT_VIDEO_ROOT (absolute directory) — used as the prefix for relative paths.
#   2. <repo_root>/data/vstat/ — fallback. Videos referenced as e.g.
#      "videos/youtube/basketball/0001_pt1.mp4" resolve to
#      "<root>/videos/youtube/basketball/0001_pt1.mp4".
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_ROOT = _REPO_ROOT / "data" / "vstat"


def _video_root() -> Path:
    override = os.environ.get("VSTAT_VIDEO_ROOT")
    return Path(override) if override else _DATA_ROOT


def _resolve_repo_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else _REPO_ROOT / path


def _qa_path_from_kwargs(dataset_kwargs) -> Path:
    override = os.environ.get("VSTAT_QA_PATH")
    if override:
        return _resolve_repo_path(override)
    data_files = (dataset_kwargs or {}).get("data_files", {})
    if isinstance(data_files, dict):
        path = data_files.get("test") or next(iter(data_files.values()))
    else:
        path = data_files
    if not path:
        path = "data/vstat/vstat_qa_clean.json"
    return _resolve_repo_path(path)


def _normalize_doc_for_arrow(doc):
    doc = dict(doc)
    doc["answer"] = str(doc["answer"])
    doc["choices"] = doc.get("choices") or []
    return doc


class VSTATTask(ConfigurableTask):
    """ConfigurableTask variant that reads the official VSTAT QA JSON."""

    def __init__(self, *args, config=None, **kwargs):
        if config is not None:
            config = dict(config)
            config.pop("class", None)
        super().__init__(*args, config=config, **kwargs)

    def download(self, dataset_kwargs=None) -> None:
        qa_path = _qa_path_from_kwargs(dataset_kwargs)
        with qa_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        data = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
        rows = [
            _normalize_doc_for_arrow(doc)
            for docs in data.values()
            for doc in docs
        ]
        split = self.config.test_split
        self.dataset = DatasetDict({split: Dataset.from_list(rows)})
        if self.config.process_docs is not None:
            self.dataset[split] = self.config.process_docs(self.dataset[split])
        self.dataset_no_image = self.dataset.copy()

INTEGER_PATTERN = re.compile(r"-?\d+")
MCQ_LETTER_PATTERN = re.compile(r"\b([A-D])\b")

# Per-source-task answer instruction appended after the question body.
# MCQ rows already embed "(A) ... (B) ..." in the question text.
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


def process_docs(dataset):
    flat_docs = []
    for doc in dataset:
        question = doc["question"]
        answer = doc["answer"]
        assert question is not None, f"Missing question for {doc.get('source_task')}"
        assert answer is not None, f"Missing answer for {doc.get('source_task')}"
        choices = doc.get("choices") or []
        is_mcq = bool(choices)
        flat_doc = {
            "source_task": str(doc["source_task"]),
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


def doc_to_visual(doc):
    video_path = doc["video_path"]
    if not os.path.isabs(video_path):
        video_path = str(_video_root() / video_path)
    assert os.path.exists(video_path), f"Missing video file: {video_path}"
    return [video_path]


def _mcq_instruction(n_choices):
    letters = "ABCD"[:n_choices]
    if len(letters) < 2:
        return ""
    return f"Please answer with the letter ({', '.join(letters[:-1])}, or {letters[-1]})."


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    body = f"Watch the full video carefully before answering.\n\nQuestion: {doc['question']}"
    if doc["is_mcq"]:
        instruction = _mcq_instruction(len(doc["choices"]))
        post_prompt = kwargs.get("mcq_post_prompt", "")
        if instruction:
            body += f"\n\n{instruction}"
        return f"{pre_prompt}{body}\n\n{post_prompt}"
    else:
        instruction = NUMERICAL_INSTRUCTIONS.get(doc["source_task"])
        post_prompt = kwargs.get("na_post_prompt", "")
        if instruction:
            body += f"\n\n{instruction}"
        return f"{pre_prompt}{body}\n\n{post_prompt}"


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


def abs_dist_norm(pred, target):
    if target == 0:
        return 0.0 if pred == target else float("inf")
    return abs(pred - target) / abs(target)


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


def _numeric_mra(parsed_prediction, target_value):
    if parsed_prediction is None:
        return 0.0
    try:
        return float(mean_relative_accuracy(parsed_prediction, target_value, 0.5, 0.95, 0.05))
    except (OverflowError, TypeError, ValueError, ZeroDivisionError):
        return 0.0


def process_results(doc, results):
    """Per-doc metrics.

    Numeric docs emit `Numeric_MRA` (MRA at .5:.95:.05). Non-numeric docs (MCQ
    and open-ended text) emit `MCQ_ACC` (0/1 accuracy). Every doc emits
    `ALL_Score_avg` set to whichever score applies to it, so the aggregate is
    the unified per-sample average across the whole benchmark.
    """
    prediction = str(results[0]).strip() if results else ""
    parsed_prediction = _parse_prediction(doc, prediction)
    is_numeric = (not doc["is_mcq"]) and doc.get("target_value") is not None

    if is_numeric:
        score = _numeric_mra(parsed_prediction, doc["target_value"])
        return {"Numeric_MRA": score, "ALL_Score_avg": score}

    score = float(_is_correct(doc, parsed_prediction))
    return {"MCQ_ACC": score, "ALL_Score_avg": score}


def aggregate_mean(results):
    return sum(float(r) for r in results) / len(results) if results else 0.0
