"""Extract a sharp, evenly spread set of frames from a video for sim-capture.

Freehand video is the easiest source of overlapping photos, but consecutive frames
are near-duplicates and motion blur wrecks feature matching. This samples the
video at ``--fps``, scores each sample by Laplacian variance, and keeps the
sharpest frame in each of ``--max-frames`` equal time bins.

    python -m sim_capture.tools.frames_from_video clip.mp4 --out photos --max-frames 90
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def sharpness(frame: NDArray[np.uint8]) -> float:
    """Variance of the Laplacian on a fixed-size gray copy (comparable across frames)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (640, int(640 * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_sharpest_per_bin(scores: Sequence[float], max_frames: int) -> list[int]:
    """Indices into ``scores``: the best one in each of ``max_frames`` equal bins (all if fewer)."""
    count = len(scores)
    if count <= max_frames:
        return list(range(count))
    edges = np.linspace(0, count, max_frames + 1).astype(int)
    return [int(lo + np.argmax(scores[lo:hi])) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]


def extract(
    video: Path,
    out_dir: Path,
    fps: float = 4.0,
    max_frames: int = 100,
    start_s: float = 0.0,
    end_s: float | None = None,
    max_width: int | None = None,
) -> list[Path]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video {video}")
    native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native_fps / fps))
    first = int(start_s * native_fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    last = None if end_s is None else int(end_s * native_fps)

    candidates: list[tuple[float, NDArray[np.uint8]]] = []
    index = first
    while last is None or index < last:
        ok, raw = capture.read()
        if not ok:
            break
        if (index - first) % step == 0:
            frame = np.asarray(raw, dtype=np.uint8)
            candidates.append((sharpness(frame), frame))
        index += 1
    capture.release()
    if len(candidates) < 3:
        raise ValueError(f"only {len(candidates)} frames sampled from {video}; need at least 3")

    chosen = select_sharpest_per_bin([score for score, _ in candidates], max_frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for number, candidate_index in enumerate(chosen, start=1):
        frame = candidates[candidate_index][1]
        if max_width is not None and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        path = out_dir / f"frame_{number:04d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="frames_from_video", description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=4.0, help="candidate sampling rate (default 4)")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--start", type=float, default=0.0, help="start second")
    parser.add_argument("--end", type=float, default=None, help="end second")
    parser.add_argument("--max-width", type=int, default=None, help="downscale wider frames (saves disk)")
    args = parser.parse_args(argv)
    try:
        written = extract(args.video, args.out, args.fps, args.max_frames, args.start, args.end, args.max_width)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"wrote {len(written)} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
