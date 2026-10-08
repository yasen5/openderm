"""Externally supplied camera poses (e.g. freehand SfM) for the registration.

A capture made without the gantry has no proprioception to derive poses from.
``--poses-from poses.json`` supplies them directly, in the same gauge the
registration works in: millimetres, camera-to-world rotations, a frame in which
the subject is roughly a heightfield z = f(x, y) seen from +z.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from .registration_features import Frame


class ExternalPoseFrameJson(TypedDict):
    image: str  # basename, matched against the sidecar's image
    R_cam2world: list[list[float]]
    C_mm: list[float]
    depth_mm: float  # median camera-frame depth of the frame's visible surface


class ExternalPosesDocument(TypedDict):
    source: str
    image_size: list[int]  # [width, height], full resolution
    fx_full: float  # full-resolution focal length, px
    k1: float  # f * x * (1 + k1 * r^2) on normalised coordinates
    frames: list[ExternalPoseFrameJson]


@dataclass
class ExternalPoses:
    fx_full: float
    k1: float
    image_size: tuple[int, int]
    rotations_cam2world: NDArray[np.float64]  # (n,3,3), indexed by Frame.idx
    camera_centers_mm: NDArray[np.float64]  # (n,3)
    surface_depths_mm: NDArray[np.float64]  # (n,), median camera-frame depth


def _json_object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object")
    return cast(Mapping[str, object], value)


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite number")
    return float(value)


def _parse_poses_document(value: object, path: str) -> ExternalPosesDocument:
    document = _json_object(value, path)
    source = document.get("source")
    if not isinstance(source, str):
        raise ValueError(f"{path}: source must be a string")

    raw_image_size = document.get("image_size")
    if (
        not isinstance(raw_image_size, list)
        or len(raw_image_size) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in raw_image_size)
    ):
        raise ValueError(f"{path}: image_size must be [positive width, positive height]")
    image_size = cast(list[int], raw_image_size)

    raw_frames = document.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError(f"{path}: frames must be a list")
    frames: list[ExternalPoseFrameJson] = []
    for index, raw_frame in enumerate(raw_frames):
        context = f"{path}: frames[{index}]"
        entry = _json_object(raw_frame, context)
        image = entry.get("image")
        if not isinstance(image, str) or not image:
            raise ValueError(f"{context}.image must be a non-empty string")

        raw_rotation = entry.get("R_cam2world")
        if not isinstance(raw_rotation, list) or len(raw_rotation) != 3:
            raise ValueError(f"{context}.R_cam2world must be a 3x3 matrix")
        rotation_rows: list[list[float]] = []
        for row_index, raw_row in enumerate(raw_rotation):
            if not isinstance(raw_row, list) or len(raw_row) != 3:
                raise ValueError(f"{context}.R_cam2world[{row_index}] must contain 3 values")
            rotation_rows.append(
                [_finite_number(component, f"{context}.R_cam2world[{row_index}]") for component in raw_row]
            )

        raw_center = entry.get("C_mm")
        if not isinstance(raw_center, list) or len(raw_center) != 3:
            raise ValueError(f"{context}.C_mm must contain 3 values")
        center_mm = [_finite_number(component, f"{context}.C_mm") for component in raw_center]
        surface_depth_mm = _finite_number(entry.get("depth_mm"), f"{context}.depth_mm")
        frames.append(
            {
                "image": image,
                "R_cam2world": rotation_rows,
                "C_mm": center_mm,
                "depth_mm": surface_depth_mm,
            }
        )

    focal_length_px = _finite_number(document.get("fx_full"), f"{path}.fx_full")
    if focal_length_px <= 0:
        raise ValueError(f"{path}: fx_full must be positive")
    return {
        "source": source,
        "image_size": image_size,
        "fx_full": focal_length_px,
        "k1": _finite_number(document.get("k1"), f"{path}.k1"),
        "frames": frames,
    }


def load_external_poses(path: str, frames: list[Frame]) -> ExternalPoses:
    """Read a poses document and align it to ``frames`` by image basename."""
    with open(path) as poses_file:
        document = _parse_poses_document(json.load(poses_file), path)
    by_image: dict[str, ExternalPoseFrameJson] = {}
    for entry in document["frames"]:
        image_basename = os.path.basename(entry["image"])
        if image_basename in by_image:
            raise ValueError(f"{path}: multiple poses use the image basename {image_basename!r}")
        by_image[image_basename] = entry
    missing = [os.path.basename(frame.image_path) for frame in frames if os.path.basename(frame.image_path) not in by_image]
    if missing:
        raise ValueError(f"{path} has no pose for {len(missing)} frame(s), e.g. {missing[:3]}")
    rotations_cam2world = np.zeros((len(frames), 3, 3))
    camera_centers_mm = np.zeros((len(frames), 3))
    surface_depths_mm = np.zeros(len(frames))
    for frame in frames:
        if not 0 <= frame.idx < len(frames):
            raise ValueError(f"{path}: frame index {frame.idx} is outside the pose array")
        entry = by_image[os.path.basename(frame.image_path)]
        rotation = np.asarray(entry["R_cam2world"], dtype=np.float64)
        if (
            rotation.shape != (3, 3)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        ):
            raise ValueError(f"{path}: R_cam2world for {entry['image']} is not a 3x3 rotation")
        rotations_cam2world[frame.idx] = rotation
        camera_center_mm = np.asarray(entry["C_mm"], dtype=np.float64)
        if camera_center_mm.shape != (3,):
            raise ValueError(f"{path}: C_mm for {entry['image']} is not a 3D camera center")
        camera_centers_mm[frame.idx] = camera_center_mm
        surface_depths_mm[frame.idx] = float(entry["depth_mm"])
    if not (surface_depths_mm > 0).all():
        raise ValueError(f"{path}: every frame needs depth_mm > 0")
    width, height = document["image_size"]
    return ExternalPoses(
        fx_full=float(document["fx_full"]),
        k1=float(document["k1"]),
        image_size=(int(width), int(height)),
        rotations_cam2world=rotations_cam2world,
        camera_centers_mm=camera_centers_mm,
        surface_depths_mm=surface_depths_mm,
    )
