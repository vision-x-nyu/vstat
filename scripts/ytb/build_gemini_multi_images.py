"""Build frame-based multi-image data for the ytb Gemini eval.

Usage:
    python scripts/ytb/build_gemini_multi_images.py --fps 1
    python scripts/ytb/build_gemini_multi_images.py --all-frames

Input spec:
    Reads a merged QA JSON with schema
    {"data": {"task_name": [{"video_path": "...", "question": "...", ...}]}}.
    By default, every source task in the input JSON is included.
    Video files are read only and are never modified.

Output spec:
    Writes a new derived JSON under data/ytb_gemini_multi_images/<preset>/.
    Each copied QA entry gains frame_paths, a chronological list of JPEG frames
    extracted under data/ytb_gemini_multi_images/<preset>/frames/.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("/nas2/benchmarks/vpi/real/merged_qa/vpi_qa_redact.json")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "ytb_gemini_multi_images"
YTB_TASKS_YAML = (
    REPO_ROOT
    / "lmms_eval"
    / "tasks"
    / "longvid-reasoning-eval_ytb"
    / "longvid-reasoning-eval_ytb.yaml"
)
LEGACY_VIDEO_ROOT = "/nas2/benchmarks/vpi/ytb-vids/processed/"
CURRENT_VIDEO_ROOT = "/nas2/benchmarks/vpi/real/processed/"


@dataclass(frozen=True)
class VideoJob:
    video_path: str
    frame_dir: Path
    fps: float
    all_frames: bool
    jpeg_quality: int
    force: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--tasks-yaml",
        type=Path,
        default=None,
        help="Optional ytb group YAML used to filter source tasks. Defaults to no filtering.",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def preset_name(fps: float, all_frames: bool) -> str:
    if all_frames:
        return "all_frames"
    fps_text = ("%g" % fps).replace(".", "p")
    return f"fps{fps_text}"


def resolve_video_path(video_path: str) -> str:
    if video_path.startswith(LEGACY_VIDEO_ROOT):
        return CURRENT_VIDEO_ROOT + video_path[len(LEGACY_VIDEO_ROOT):]
    return video_path


def load_source_tasks(tasks_yaml: Path) -> set[str]:
    prefix = "longvid-reasoning-eval_ytb_"
    source_tasks = set()
    for line in tasks_yaml.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        task_name = stripped[2:].split("#", 1)[0].strip()
        if task_name.startswith(prefix):
            source_tasks.add(task_name[len(prefix):])
    assert source_tasks, f"No ytb source tasks found in {tasks_yaml}"
    return source_tasks


def filter_data(data: dict, source_tasks: set[str]) -> dict:
    output = dict(data)
    output["data"] = {
        source_task: docs
        for source_task, docs in data["data"].items()
        if source_task in source_tasks
    }
    missing = sorted(source_tasks - set(output["data"]))
    assert not missing, f"Missing source tasks in input JSON: {missing}"
    return output


def frame_dir_for_video(frame_root: Path, source_task: str, video_path: str) -> Path:
    resolved = resolve_video_path(video_path)
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:8]
    stem = Path(resolved).stem
    return frame_root / source_task / f"{stem}_{digest}"


def list_frames(frame_dir: Path) -> list[str]:
    return [str(path) for path in sorted(frame_dir.glob("frame_*.jpg"))]


def extract_video(job: VideoJob) -> tuple[str, list[str]]:
    video_path = resolve_video_path(job.video_path)
    assert os.path.exists(video_path), f"Missing video file: {video_path}"

    if job.force and job.frame_dir.exists():
        shutil.rmtree(job.frame_dir)
    job.frame_dir.mkdir(parents=True, exist_ok=True)

    complete_path = job.frame_dir / ".complete"
    frames = list_frames(job.frame_dir)
    if complete_path.exists() and frames:
        return job.video_path, frames

    output_pattern = str(job.frame_dir / "frame_%06d.jpg")
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
    ]
    if not job.all_frames:
        command.extend(["-vf", f"fps={job.fps:g}"])
    command.extend(["-q:v", str(job.jpeg_quality), output_pattern])

    subprocess.run(command, check=True)
    frames = list_frames(job.frame_dir)
    assert frames, f"No frames extracted for {video_path}"
    complete_path.write_text(f"{len(frames)}\n")
    return job.video_path, frames


def iter_docs(data: dict):
    for source_task, docs in data["data"].items():
        for doc in docs:
            yield source_task, doc


def build_jobs(data: dict, frame_root: Path, args: argparse.Namespace) -> list[VideoJob]:
    jobs_by_video = {}
    for source_task, doc in iter_docs(data):
        video_path = doc["video_path"]
        if video_path in jobs_by_video:
            continue
        jobs_by_video[video_path] = VideoJob(
            video_path=video_path,
            frame_dir=frame_dir_for_video(frame_root, source_task, video_path),
            fps=args.fps,
            all_frames=args.all_frames,
            jpeg_quality=args.jpeg_quality,
            force=args.force,
        )
    return list(jobs_by_video.values())


def run_extraction(jobs: list[VideoJob], workers: int) -> dict[str, list[str]]:
    frames_by_video = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extract_video, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            video_path, frames = future.result()
            frames_by_video[video_path] = frames
            print(f"[{index}/{len(jobs)}] {video_path} -> {len(frames)} frames", flush=True)
    return frames_by_video


def attach_frames(data: dict, frames_by_video: dict[str, list[str]], preset: str) -> dict:
    output = dict(data)
    output["frame_preset"] = preset
    output["data"] = {}
    for source_task, docs in data["data"].items():
        output_docs = []
        for doc in docs:
            copied = dict(doc)
            copied["frame_paths"] = frames_by_video[doc["video_path"]]
            output_docs.append(copied)
        output["data"][source_task] = output_docs
    return output


def main() -> None:
    args = parse_args()
    assert args.fps > 0, "--fps must be positive"
    assert args.workers > 0, "--workers must be positive"
    assert args.input.exists(), f"Missing input JSON: {args.input}"
    if args.tasks_yaml is not None:
        assert args.tasks_yaml.exists(), f"Missing tasks YAML: {args.tasks_yaml}"

    preset = preset_name(args.fps, args.all_frames)
    preset_root = args.output_root / preset
    frame_root = preset_root / "frames"
    output_path = preset_root / "merged_qa.json"

    with args.input.open() as fp:
        data = json.load(fp)
    if args.tasks_yaml is not None:
        data = filter_data(data, load_source_tasks(args.tasks_yaml))

    jobs = build_jobs(data, frame_root, args)
    print(f"preset={preset} videos={len(jobs)} workers={args.workers}")
    frames_by_video = run_extraction(jobs, args.workers)
    output = attach_frames(data, frames_by_video, preset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fp:
        json.dump(output, fp, indent=2, ensure_ascii=False)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
