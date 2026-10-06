"""Put a COLMAP reconstruction into processing's gauge.

COLMAP's frame is arbitrary in scale, rotation and origin. Processing expects
millimetres and a world in which the subject is roughly a heightfield
z = f(x, y) seen from +z (texture u = x, v = arc length along y). We get there
with a similarity transform: scale from the user's standoff, axes from a PCA of
the sparse surface.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .model import CanonicalPoses, FloatArray, SparseModel

MIN_POINTS_FOR_FIT = 20
RELIEF_WARN_RATIO = 0.35  # sigma_z / sigma_x above this is not heightfield-like
VIEW_ANGLE_WARN_DEG = 70.0


def _trusted_surface_points(model: SparseModel) -> NDArray[np.bool_]:
    """Well-observed, low-error, non-outlying points for the plane fit."""
    mask = np.ones(len(model.points), bool)
    well_tracked = model.point_track_lengths >= 3
    if well_tracked.sum() >= MIN_POINTS_FOR_FIT:
        mask &= well_tracked
    error_limit = max(1.0, 2.0 * float(np.median(model.point_errors_px[mask])))
    low_error = model.point_errors_px <= error_limit
    if (mask & low_error).sum() >= MIN_POINTS_FOR_FIT:
        mask &= low_error
    center = np.median(model.points[mask], axis=0)
    spread = 1.4826 * np.median(np.abs(model.points[mask] - center), axis=0) + 1e-12
    within = np.all(np.abs(model.points - center) < 6.0 * spread, axis=1)
    if (mask & within).sum() >= MIN_POINTS_FOR_FIT:
        mask &= within
    return mask


def canonicalize(
    model: SparseModel, standoff_mm: float, point_filter: NDArray[np.bool_] | None = None
) -> CanonicalPoses:
    """Scale to ``standoff_mm`` mean camera-to-surface depth and align the axes.

    ``point_filter`` (m,) restricts the points that define the surface, i.e. the
    depth that ``standoff_mm`` refers to and the plane that sets the axes, for
    example to the sparse points that land on skin rather than on the table.
    """
    if standoff_mm <= 0:
        raise ValueError("standoff_mm must be positive")
    if len(model.image_names) < 2 or len(model.points) < MIN_POINTS_FOR_FIT:
        raise ValueError("reconstruction is too small to canonicalize")

    rotations_c2w = np.transpose(model.rotations_cam_from_world, (0, 2, 1))  # R^T
    centers = -np.einsum("nij,nj->ni", rotations_c2w, model.translations_cam_from_world)

    if point_filter is not None and int(point_filter.sum()) < MIN_POINTS_FOR_FIT:
        raise ValueError(
            f"only {int(point_filter.sum())} COLMAP points lie on the masked surface (need "
            f"{MIN_POINTS_FOR_FIT}); the masked region has too little texture for COLMAP, "
            "so scale and axes cannot be taken from it"
        )
    depths = np.full(len(model.image_names), np.nan)
    for index, visible in enumerate(model.visible_points):
        if point_filter is not None:
            visible = visible[point_filter[visible]]
        if len(visible) >= 10:
            camera_points = model.points[visible] @ model.rotations_cam_from_world[index].T
            camera_points = camera_points + model.translations_cam_from_world[index]
            depths[index] = np.median(camera_points[:, 2])
    valid_depths = depths[np.isfinite(depths) & (depths > 0)]
    if len(valid_depths) == 0:
        raise ValueError("no image sees enough points in front of it to measure depth")
    depths = np.where(np.isfinite(depths) & (depths > 0), depths, np.median(valid_depths))
    scale = standoff_mm / float(np.median(valid_depths))

    trusted = _trusted_surface_points(model)
    if point_filter is not None:
        trusted = trusted & point_filter
        if int(trusted.sum()) < MIN_POINTS_FOR_FIT:
            raise ValueError("too few well-tracked points on the masked surface to fit its plane")
    origin = model.points[trusted].mean(axis=0)
    _, _, principal_axes = np.linalg.svd(model.points[trusted] - origin, full_matrices=False)
    z_axis = principal_axes[2]
    if float(np.mean((centers - origin) @ z_axis)) < 0:
        z_axis = -z_axis
    x_axis = principal_axes[0] - (principal_axes[0] @ z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    path = centers[-1] - centers[0]
    if float(path @ x_axis) < 0:  # +x follows the capture order
        x_axis = -x_axis
    y_axis = np.cross(z_axis, x_axis)
    canonical_from_world: FloatArray = np.stack([x_axis, y_axis, z_axis], axis=0)  # rows; det = +1

    centers_mm = scale * (centers - origin) @ canonical_from_world.T
    rotations_mm = np.einsum("ij,njk->nik", canonical_from_world, rotations_c2w)
    points_mm = scale * (model.points - origin) @ canonical_from_world.T

    surface = points_mm[trusted]
    relief_ratio = float(np.std(surface[:, 2]) / max(np.std(surface[:, 0]), 1e-9))
    optical_axes = rotations_mm[:, :, 2]
    view_angles = np.degrees(np.arccos(np.clip(-optical_axes[:, 2], -1.0, 1.0)))
    below_plane = int(np.sum(centers_mm[:, 2] <= 0))
    diagnostics = {
        "relief_ratio": relief_ratio,
        "max_view_angle_deg": float(view_angles.max()),
        "median_view_angle_deg": float(np.median(view_angles)),
        "cameras_below_surface": float(below_plane),
        "surface_extent_x_mm": float(np.ptp(surface[:, 0])),
        "surface_extent_y_mm": float(np.ptp(surface[:, 1])),
    }
    warnings: list[str] = []
    if below_plane:
        warnings.append(
            f"{below_plane} camera(s) sit behind the fitted surface plane: the capture wraps around the "
            "subject, but processing models a one-sided heightfield"
        )
    if relief_ratio > RELIEF_WARN_RATIO:
        warnings.append(
            f"surface relief is large (sigma_z/sigma_x = {relief_ratio:.2f}): the subject is strongly "
            "3D, so the heightfield model may fold it"
        )
    if view_angles.max() > VIEW_ANGLE_WARN_DEG:
        warnings.append(
            f"some views are {view_angles.max():.0f} deg from the surface normal; very oblique frames "
            "texture poorly"
        )
    return CanonicalPoses(
        scale_mm_per_unit=scale,
        rotations_cam2world=rotations_mm,
        centers_mm=centers_mm,
        depths_mm=scale * depths,
        points_mm=points_mm,
        surface_point_mask=trusted,
        diagnostics=diagnostics,
        warnings=warnings,
    )
