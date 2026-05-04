"""Task helpers for ytb Gemini multi-image eval.

The source QA JSON is copied from the video benchmark and augmented with
frame_paths. Each document returns those frames as chronological images while
reusing the ytb scoring functions.
"""

import importlib.util
import os
from pathlib import Path

from datasets import Dataset
from PIL import Image


_YTB_UTILS_PATH = Path(__file__).resolve().parents[1] / "longvid-reasoning-eval_ytb" / "utils.py"
_YTB_SPEC = importlib.util.spec_from_file_location("ytb_video_utils", _YTB_UTILS_PATH)
assert _YTB_SPEC is not None and _YTB_SPEC.loader is not None
_YTB_UTILS = importlib.util.module_from_spec(_YTB_SPEC)
_YTB_SPEC.loader.exec_module(_YTB_UTILS)

DEFAULT_MAX_FRAMES = 512
NUMERICAL_INSTRUCTIONS = _YTB_UTILS.NUMERICAL_INSTRUCTIONS
doc_to_target = _YTB_UTILS.doc_to_target
process_results = _YTB_UTILS.process_results
aggregate_accuracy = _YTB_UTILS.aggregate_accuracy
aggregate_mra = _YTB_UTILS.aggregate_mra
aggregate_mae = _YTB_UTILS.aggregate_mae


def _source_task_from_video_path(video_path):
    parts = str(video_path).replace("\\", "/").split("/")
    assert len(parts) >= 2, f"Cannot infer source task from video_path: {video_path}"
    return parts[-2]


def _iter_source_docs(dataset):
    for row in dataset:
        data = row["data"]
        if isinstance(data, dict):
            for source_task, docs in data.items():
                for doc in docs:
                    yield source_task, doc
        elif isinstance(data, list):
            for doc in data:
                yield _source_task_from_video_path(doc["video_path"]), doc
        else:
            raise TypeError(f"Unsupported data field type: {type(data)}")


def _build_flat_doc(source_task, doc):
    question = doc["question"]
    answer = doc["answer"]
    frame_paths = doc["frame_paths"]
    assert question is not None, f"Missing question for {source_task}"
    assert answer is not None, f"Missing answer for {source_task}"
    assert frame_paths, f"Missing frame_paths for {source_task}/{doc['video_id']}"

    choices = doc.get("choices")
    is_mcq = choices is not None
    flat_doc = {
        "source_task": source_task,
        "video_id": str(doc["video_id"]),
        "video_path": doc["video_path"],
        "frame_paths": list(frame_paths),
        "question": question.strip(),
        "is_mcq": is_mcq,
        "choices": choices,
        "answer_index": doc.get("answer_index"),
    }
    if is_mcq:
        flat_doc["answer_text"] = str(answer).strip()
        flat_doc["target_value"] = None
        return flat_doc

    text_answer = str(answer).strip()
    try:
        value = int(text_answer)
        flat_doc["target_value"] = value
        flat_doc["answer_text"] = str(value)
    except ValueError:
        flat_doc["target_value"] = None
        flat_doc["answer_text"] = text_answer
    return flat_doc


def process_docs(dataset):
    flat_docs = [_build_flat_doc(source_task, doc) for source_task, doc in _iter_source_docs(dataset)]
    return Dataset.from_list(flat_docs)


def _max_frames():
    value = os.environ.get("GEMINI_MULTI_IMAGES_MAX_FRAMES")
    if value is None:
        return DEFAULT_MAX_FRAMES
    max_frames = int(value)
    assert max_frames > 0, "GEMINI_MULTI_IMAGES_MAX_FRAMES must be positive"
    return max_frames


def _sample_frame_paths(frame_paths, max_frames):
    if len(frame_paths) <= max_frames:
        return frame_paths
    if max_frames == 1:
        return [frame_paths[0]]
    last_index = len(frame_paths) - 1
    return [
        frame_paths[round(index * last_index / (max_frames - 1))]
        for index in range(max_frames)
    ]


def doc_to_visual(doc):
    frame_paths = _sample_frame_paths(doc["frame_paths"], _max_frames())
    images = []
    for frame_path in frame_paths:
        assert os.path.exists(frame_path), f"Missing frame file: {frame_path}"
        images.append(Image.open(frame_path).convert("RGB"))
    return images


def _mcq_instruction(n_choices):
    letters = "ABCD"[:n_choices]
    if len(letters) < 2:
        return ""
    return f"Please answer with the letter ({', '.join(letters[:-1])}, or {letters[-1]})."


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    body = (
        "Look at the video frames in chronological order before answering.\n\n"
        f"Question: {doc['question']}"
    )
    if doc["is_mcq"]:
        instruction = _mcq_instruction(len(doc["choices"]))
        post_prompt = kwargs.get("mcq_post_prompt", "")
        if instruction:
            body += f"\n\n{instruction}"
        return f"{pre_prompt}{body}\n\n{post_prompt}"

    instruction = NUMERICAL_INSTRUCTIONS.get(doc["source_task"])
    post_prompt = kwargs.get("na_post_prompt", "")
    if instruction:
        body += f"\n\n{instruction}"
    return f"{pre_prompt}{body}\n\n{post_prompt}"
