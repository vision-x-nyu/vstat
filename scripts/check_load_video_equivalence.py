"""Verify the new sequential-read load_video matches the old random-access version.

For each test video, decode the same frame indices using:
- old: ``vr[i].asnumpy()`` per index (random seek_accurate)
- new: ``vr.next()`` sequential + skip until target index

Then compute per-frame absolute pixel diff and report max/mean diff.

The old code path is run with a SIGALRM watchdog so corrupt-file hangs surface
as ``hang`` instead of blocking the whole verification. SIGALRM only fires when
Python checks signals, so a deadlocked C call cannot be interrupted; in that
case the script must be killed manually after the row is printed for the next
video. For the known-corrupt fixture file we use a child process via os.fork
so the hang can be hard-killed.

Usage:
    python scripts/check_load_video_equivalence.py
    python scripts/check_load_video_equivalence.py --num-segments 32 --timeout-sec 30 \
        --videos /path/to/foo.mp4 /path/to/bar.mp4

Output (stdout):
- One row per video: status (ok/hang/err), max pixel diff, mean pixel diff,
  number of frames that differ. Matching decoders produce max_diff == 0.
"""

import argparse
import os
import signal
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
from decord import VideoReader, cpu


DEFAULT_VIDEOS = [
    "/nas2/benchmarks/vpi/ytb-vids/processed/table_tennis/0001_pt1.mp4",
    "/nas2/benchmarks/vpi/ytb-vids/processed/table_tennis/0002_pt5.mp4",
    "/nas2/benchmarks/vpi/ytb-vids/processed/basketball/0001_pt1.mp4",
    "/nas2/benchmarks/vpi/ytb-vids/processed/soccer/0001_pt1.mp4",
    "/nas2/benchmarks/vpi/ytb-vids/processed/table_tennis/0003_pt5.mp4",  # corrupt fixture
]

KNOWN_HANG_VIDEOS = {
    "/nas2/benchmarks/vpi/ytb-vids/processed/table_tennis/0003_pt5.mp4",
}


def _get_index(num_frames: int, num_segments: int) -> np.ndarray:
    """Mirror internvl3.get_index with bound=None, first_idx=0."""
    end_idx = num_frames - 1
    seg = float(end_idx) / num_segments
    return np.array(
        [int((seg / 2) + np.round(seg * idx)) for idx in range(num_segments)],
        dtype=np.int64,
    )


def _load_old(vr: VideoReader, indices: np.ndarray) -> np.ndarray:
    return np.stack([vr[int(i)].asnumpy() for i in indices], axis=0)


def _load_new(vr: VideoReader, indices: np.ndarray) -> np.ndarray:
    unique_targets = sorted({int(i) for i in indices})
    frames_by_idx = {}
    cursor = 0
    for target in unique_targets:
        while cursor < target:
            vr.next()
            cursor += 1
        frames_by_idx[target] = vr.next().asnumpy()
        cursor += 1
    return np.stack([frames_by_idx[int(i)] for i in indices], axis=0)


def _alarm_handler(signum, frame):
    raise TimeoutError("decoder hung past timeout")


def _try_with_alarm(fn, *args, timeout_sec: float):
    """Run fn under SIGALRM. Works for pure-Python hangs; C-level deadlocks
    will not be interrupted (the alarm fires once Python regains control)."""
    prev = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        return ("ok", fn(*args))
    except TimeoutError as e:
        return ("hang", str(e))
    except Exception as e:
        return ("err", f"{type(e).__name__}: {e}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def _run_in_fork(fn, path: str, num_segments: int, timeout_sec: float):
    """Hard-isolate the call in a forked child and read the result from a
    tempfile. Used for KNOWN_HANG_VIDEOS so a C-level deadlock can be SIGKILLed.
    Tempfile (not pipe) avoids buffer-full deadlocks for multi-MB arrays."""
    import tempfile

    fd, tmp_path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    pid = os.fork()
    if pid == 0:
        try:
            vr = VideoReader(path, ctx=cpu(0), num_threads=1)
            indices = _get_index(len(vr), num_segments)
            arr = fn(vr, indices)
            np.save(tmp_path, arr, allow_pickle=False)
            os._exit(0)
        except BaseException as exc:
            sys.stderr.write(f"child err: {type(exc).__name__}: {exc}\n")
            os._exit(2)

    deadline = time.time() + timeout_sec
    status = 0
    while True:
        try:
            wpid, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return ("err", "lost child", None)
        if wpid != 0:
            break
        if time.time() > deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return ("hang", f"timeout {timeout_sec}s", None)
        time.sleep(0.1)

    try:
        if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 and os.path.getsize(tmp_path) > 0:
            arr = np.load(tmp_path, allow_pickle=False)
            return ("ok", "", arr)
        return ("err", f"exit {os.WEXITSTATUS(status) if os.WIFEXITED(status) else 'sig'}", None)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _diff(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    if a.shape != b.shape:
        return float("nan"), float("nan"), -1
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return float(d.max()), float(d.mean()), int((d.sum(axis=(1, 2, 3)) > 0).sum())


def _alignment(old: np.ndarray, new: np.ndarray) -> List[int]:
    """For each frame in `new`, find the index in `old` that matches best
    (lowest mean abs pixel diff). If decord's random `vr[i]` returns a
    neighbouring frame instead of frame i, alignment_offset = best_idx - i
    will be non-zero for those frames."""
    n = new.shape[0]
    offsets = []
    for i in range(n):
        diffs = np.abs(old.astype(np.int32) - new[i].astype(np.int32)).mean(axis=(1, 2, 3))
        best = int(diffs.argmin())
        offsets.append(best - i)
    return offsets


def _evaluate(path: str, num_segments: int, timeout_sec: float):
    if path in KNOWN_HANG_VIDEOS:
        s_old, msg_old, old = _run_in_fork(_load_old, path, num_segments, timeout_sec)
        s_new, msg_new, new = _run_in_fork(_load_new, path, num_segments, timeout_sec)
    else:
        vr_old = VideoReader(path, ctx=cpu(0), num_threads=1)
        indices = _get_index(len(vr_old), num_segments)
        s_old, payload_old = _try_with_alarm(_load_old, vr_old, indices, timeout_sec=timeout_sec)
        old = payload_old if s_old == "ok" else None

        vr_new = VideoReader(path, ctx=cpu(0), num_threads=1)
        s_new, payload_new = _try_with_alarm(_load_new, vr_new, indices, timeout_sec=timeout_sec)
        new = payload_new if s_new == "ok" else None
    return s_old, old, s_new, new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="*", default=DEFAULT_VIDEOS)
    parser.add_argument("--num-segments", type=int, default=32)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    args = parser.parse_args()

    header = f"{'video':<70} {'old':<6} {'new':<6} {'max_diff':>9} {'mean_diff':>10} {'frames_diff':>12} {'aligned_diff':>13} {'offsets':>20}"
    print(header)
    print("-" * len(header))

    for path in args.videos:
        if not os.path.exists(path):
            print(f"{path:<70} MISSING")
            continue
        s_old, old, s_new, new = _evaluate(path, args.num_segments, args.timeout_sec)
        if s_old == "ok" and s_new == "ok":
            mx, mn, nd = _diff(old, new)
            offsets = _alignment(old, new)
            # Re-diff using best-aligned old frames for each new frame.
            old_aligned = np.stack([old[max(0, min(len(old) - 1, i + offsets[i]))] for i in range(len(new))], axis=0)
            _, mn_aligned, _ = _diff(old_aligned, new)
            offset_summary = f"{min(offsets):+d}..{max(offsets):+d}"
            print(f"{path:<70} {'ok':<6} {'ok':<6} {mx:>9.3f} {mn:>10.5f} {nd:>12d} {mn_aligned:>13.5f} {offset_summary:>20}")
        else:
            print(f"{path:<70} {s_old:<6} {s_new:<6} {'-':>9} {'-':>10} {'-':>12} {'-':>13} {'-':>20}")


if __name__ == "__main__":
    main()
