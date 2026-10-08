"""Plain data carried between the COLMAP wrapper and the rest of sim-capture.

Nothing here imports pycolmap, so alignment and export are testable without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class AlignmentDiagnostics(TypedDict):
    relief_ratio: float
    max_view_angle_deg: float
    median_view_angle_deg: float
    cameras_below_surface: float
    surface_extent_x_mm: float
    surface_extent_y_mm: float


@dataclass
class SparseModel:
    """A COLMAP reconstruction in COLMAP's own (arbitrary scale/gauge) frame.

    ``X_cam = R_cam_from_world @ X_world + t_cam_from_world`` (COLMAP convention).
    Only registered images appear. ``visible_points[i]`` indexes ``points`` rows
    seen by image ``i``.
    """

    image_names: list[str]
    rotations_cam_from_world: FloatArray  # (n,3,3)
    translations_cam_from_world: FloatArray  # (n,3)
    points: FloatArray  # (m,3)
    point_errors_px: FloatArray  # (m,)
    point_track_lengths: IntArray  # (m,)
    point_colors: NDArray[np.uint8]  # (m,3) RGB
    visible_points: list[IntArray]
    fx: float  # full-resolution px (SIMPLE_RADIAL: one focal length)
    cx: float
    cy: float
    k1: float  # f * x * (1 + k1 * r^2) on normalised coordinates
    width: int
    height: int
    mean_reprojection_error_px: float
    unregistered_images: list[str] = field(default_factory=list)
    extra_model_sizes: list[int] = field(default_factory=list)  # other disconnected models
    visible_xy: list[FloatArray] = field(default_factory=list)  # (k,2) px of each visible point, aligned with visible_points


@dataclass
class CanonicalPoses:
    """The model in processing's gauge: millimetres, +z toward the cameras."""

    scale_mm_per_colmap_unit: float
    rotations_cam2world: FloatArray  # (n,3,3)
    camera_centers_mm: FloatArray  # (n,3)
    surface_depths_mm: FloatArray  # (n,) median camera-frame depth of each image's surface
    points_mm: FloatArray  # (m,3), same row order as SparseModel.points
    surface_point_mask: NDArray[np.bool_]  # (m,) points trusted for the PCA / surface
    diagnostics: AlignmentDiagnostics
    warnings: list[str] = field(default_factory=list)
