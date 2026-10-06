"""Externally supplied camera poses (e.g. freehand SfM) for the registration.

A capture made without the gantry has no proprioception to derive poses from.
``--poses-from poses.json`` supplies them directly, in the same gauge the
registration works in: millimetres, camera-to-world rotations, a frame in which
the subject is roughly a heightfield z = f(x, y) seen from +z.
"""

from __future__ import annotations

import json
import os
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
    rotations: NDArray[np.float64]  # (n,3,3) camera-to-world, indexed by Frame.idx
    centers: NDArray[np.float64]  # (n,3) mm
    depths: NDArray[np.float64]  # (n,) mm


def load_external_poses(path: str, frames: list[Frame]) -> ExternalPoses:
    """Read a poses document and align it to ``frames`` by image basename."""
    with open(path) as poses_file:
        document = cast(ExternalPosesDocument, json.load(poses_file))
    by_image = {os.path.basename(entry["image"]): entry for entry in document["frames"]}
    missing = [os.path.basename(frame.image_path) for frame in frames if os.path.basename(frame.image_path) not in by_image]
    if missing:
        raise ValueError(f"{path} has no pose for {len(missing)} frame(s), e.g. {missing[:3]}")
    rotations = np.zeros((len(frames), 3, 3))
    centers = np.zeros((len(frames), 3))
    depths = np.zeros(len(frames))
    for frame in frames:
        entry = by_image[os.path.basename(frame.image_path)]
        rotation = np.asarray(entry["R_cam2world"], dtype=np.float64)
        if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError(f"{path}: R_cam2world for {entry['image']} is not a 3x3 rotation")
        rotations[frame.idx] = rotation
        centers[frame.idx] = np.asarray(entry["C_mm"], dtype=np.float64)
        depths[frame.idx] = float(entry["depth_mm"])
    if not (depths > 0).all():
        raise ValueError(f"{path}: every frame needs depth_mm > 0")
    width, height = document["image_size"]
    return ExternalPoses(
        fx_full=float(document["fx_full"]),
        k1=float(document["k1"]),
        image_size=(int(width), int(height)),
        rotations=rotations,
        centers=centers,
        depths=depths,
    )
