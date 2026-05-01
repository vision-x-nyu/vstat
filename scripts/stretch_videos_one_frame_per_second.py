import argparse
import shlex
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read every video in a directory and re-save so each original frame "
            "stays on screen for a fixed duration in seconds "
            "(e.g. 1.0 sec => output duration ~= input duration * fps)."
        )
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing source videos")
    parser.add_argument("--output_dir", required=True, help="Directory to save converted videos")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search videos recursively under input_dir",
    )
    parser.add_argument(
        "--suffix",
        default="_1fps_framehold",
        help="Suffix added to output filename stem",
    )
    parser.add_argument(
        "--keep_name",
        action="store_true",
        help="Keep output filename exactly same as source filename (ignore --suffix)",
    )
    parser.add_argument(
        "--hold_seconds",
        type=float,
        default=1.0,
        help="How long each original frame should stay on screen (seconds), e.g. 0.5",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--keep_audio",
        action="store_true",
        help="Keep audio track (audio will likely end earlier than stretched video)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned ffmpeg commands without running them",
    )
    return parser.parse_args()


def get_fps(video_path: Path) -> Fraction:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    fps_text = result.stdout.strip()
    fps = Fraction(fps_text)
    if fps <= 0:
        raise ValueError(f"Invalid FPS from ffprobe for {video_path}: {fps_text}")
    return fps


def build_ffmpeg_command(
    src: Path,
    dst: Path,
    fps: Fraction,
    hold_seconds: float,
    keep_audio: bool,
    overwrite: bool,
) -> list[str]:
    fps_float = float(fps)
    setpts_multiplier = fps_float * hold_seconds
    setpts_expr = f"setpts={setpts_multiplier:.12f}*PTS"

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-i",
        str(src),
        "-vf",
        setpts_expr,
        "-r",
        f"{fps_float:.12f}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]

    if keep_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
    else:
        cmd.append("-an")

    cmd.append(str(dst))
    return cmd


def collect_videos(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = [path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(files)


def collect_json_files(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = [path for path in iterator if path.is_file() and path.suffix.lower() == ".json"]
    return sorted(files)


def main() -> None:
    args = parse_args()

    if args.hold_seconds <= 0:
        raise ValueError("--hold_seconds must be > 0")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    videos = collect_videos(input_dir, args.recursive)
    json_files = collect_json_files(input_dir, args.recursive)

    if not videos and not json_files:
        print(f"No video/json files found in {input_dir}")
        return

    print(f"Found {len(videos)} video(s) and {len(json_files)} json file(s) in {input_dir}")
    converted = 0
    skipped = 0
    json_copied = 0
    json_skipped = 0

    for video_path in videos:
        relative_parent = video_path.parent.relative_to(input_dir)
        target_parent = output_dir / relative_parent
        target_parent.mkdir(parents=True, exist_ok=True)

        if args.keep_name:
            dst_name = video_path.name
        else:
            dst_name = f"{video_path.stem}{args.suffix}{video_path.suffix}"
        dst_path = target_parent / dst_name

        if dst_path.exists() and not args.overwrite:
            print(f"[SKIP] Exists: {dst_path}")
            skipped += 1
            continue

        fps = get_fps(video_path)
        cmd = build_ffmpeg_command(
            src=video_path,
            dst=dst_path,
            fps=fps,
            hold_seconds=args.hold_seconds,
            keep_audio=args.keep_audio,
            overwrite=args.overwrite,
        )

        print(
            f"[RUN ] {video_path.name} | fps={float(fps):.6f}, hold={args.hold_seconds:.3f}s "
            f"-> {dst_path.name}"
        )
        print("       " + " ".join(shlex.quote(part) for part in cmd))

        if args.dry_run:
            converted += 1
            continue

        subprocess.run(cmd, check=True)
        converted += 1

    for json_path in json_files:
        relative_path = json_path.relative_to(input_dir)
        dst_path = output_dir / relative_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not args.overwrite:
            print(f"[SKIP] JSON exists: {dst_path}")
            json_skipped += 1
            continue

        print(f"[COPY] {json_path.name} -> {dst_path}")
        if args.dry_run:
            json_copied += 1
            continue

        shutil.copy2(json_path, dst_path)
        json_copied += 1

    print(
        f"Done. converted={converted}, skipped={skipped}, total_videos={len(videos)}, "
        f"json_copied={json_copied}, json_skipped={json_skipped}, total_json={len(json_files)}"
    )


if __name__ == "__main__":
    main()
