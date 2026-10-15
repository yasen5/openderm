"""Explicit, pickle-free checkpoint for repeating texture work without registration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .registration_features import Frame
from .registration_geometry import BundleAdjustmentResult, FloatArray, RigModel


def save_registration(
    path: Path, frames: Sequence[Frame], rig: RigModel, result: BundleAdjustmentResult,
) -> None:
    """Save refined geometry, calibration and photo identity, excluding texture settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, version=np.array([1]),
        images=np.asarray([Path(frame.image_path).resolve().as_posix() for frame in frames]),
        calibration=np.array([rig.fx, rig.k1, rig.cx, rig.cy, rig.downscale]),
        depth_mm=np.asarray([rig.depth(frame) for frame in frames]),
        base_R=rig.base_R, base_t=rig.base_t,
        X=result["X"], track_err=result["track_err"], obs_frame=result["obs_frame"],
        obs_uv=result["obs_uv"], obs_track=result["obs_track"], err=result["err"],
        R=result["R"], C=result["C"], rms=np.array([result["rms"]]), history=np.asarray(result["history"]),
    )


def load_registration(path: Path, frames: Sequence[Frame]) -> tuple[RigModel, BundleAdjustmentResult]:
    """Validate an explicit checkpoint against the selected photos before reuse."""
    with np.load(path, allow_pickle=False) as data:
        required = {"version", "images", "calibration", "depth_mm", "X", "track_err", "obs_frame", "obs_uv",
                    "obs_track", "err", "R", "C", "rms", "history"}
        if not required.issubset(data.files):
            raise ValueError("incomplete registration checkpoint")
        # np.load is the dynamic third-party boundary; each array is validated
        # before it becomes part of the domain result.
        def numbers(name: str, shape: tuple[int, ...], *, finite: bool = True) -> FloatArray:
            raw = np.asarray(data[name])
            if raw.dtype.kind not in "fiu" or raw.shape != shape:
                raise ValueError(f"invalid registration array {name}: expected {shape}")
            value = np.asarray(raw, dtype=np.float64)
            if finite and not np.isfinite(value).all():
                raise ValueError(f"nonfinite registration array {name}")
            return value

        if not np.array_equal(data["version"], [1]):
            raise ValueError("unsupported registration checkpoint version")
        expected = [Path(frame.image_path).resolve().as_posix() for frame in frames]
        if not np.array_equal(data["images"], expected):
            raise ValueError("registration checkpoint belongs to different photos or frame selection")
        calibration = numbers("calibration", (5,))
        fx, k1, cx, cy, downscale = calibration
        if fx <= 0 or cx <= 0 or cy <= 0 or downscale < 1 or downscale != int(downscale):
            raise ValueError("invalid registration calibration")
        n = len(frames)
        depths = numbers("depth_mm", (n,))
        if np.any(depths <= 0):
            raise ValueError("invalid registration camera depths")
        raw_points = np.asarray(data["X"])
        raw_uv = np.asarray(data["obs_uv"])
        if raw_points.ndim != 2 or raw_uv.ndim != 2:
            raise ValueError("invalid registration landmark/observation dimensions")
        m, o = len(raw_points), len(raw_uv)
        points = numbers("X", (m, 3), finite=False)  # rejected tracks may retain NaNs
        track_errors = numbers("track_err", (m,), finite=False)

        def indices(name: str, upper: int) -> NDArray[np.int32]:
            raw = np.asarray(data[name])
            if raw.dtype.kind not in "iu" or raw.shape != (o,):
                raise ValueError(f"invalid registration indices {name}")
            if np.any(raw < 0) or np.any(raw >= upper):
                raise ValueError(f"out-of-range registration indices {name}")
            return np.asarray(raw, dtype=np.int32)

        rotations = numbers("R", (n, 3, 3))
        if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-6)
                or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6)):
            raise ValueError("invalid registration camera rotations")
        history = np.asarray(data["history"])
        if history.ndim != 1:
            raise ValueError("invalid registration history")
        base_rotation = numbers("base_R", (3, 3)) if "base_R" in data.files else np.eye(3)
        base_translation = numbers("base_t", (3,)) if "base_t" in data.files else np.zeros(3)
        if (not np.allclose(base_rotation.T @ base_rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(base_rotation), 1, atol=1e-6)):
            raise ValueError("invalid registration base rotation")
        result = BundleAdjustmentResult(
            X=points, track_err=track_errors, obs_frame=indices("obs_frame", n),
            obs_uv=numbers("obs_uv", (o, 2)), obs_track=indices("obs_track", m),
            err=numbers("err", (o,), finite=False), R=rotations, C=numbers("C", (n, 3)),
            rms=float(numbers("rms", (1,))[0]), history=numbers("history", history.shape).tolist(),
        )
    rig = RigModel(float(fx), float(k1), float(cx), float(cy), 1.0, np.zeros(3), np.eye(3), 0,
                   base_R=base_rotation, base_t=base_translation,
                   downscale=int(downscale), fixed_poses=(result["R"], result["C"]))
    for frame, depth in zip(frames, depths):
        frame.standoff = float(depth)
    return rig, result
