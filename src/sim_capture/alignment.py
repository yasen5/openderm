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

from .model import AlignmentDiagnostics, CanonicalPoses, FloatArray, SparseModel

MIN_POINTS_FOR_FIT = 20
RELIEF_WARN_RATIO = 0.35  # sigma_z / sigma_x above this is not heightfield-like
VIEW_ANGLE_WARN_DEG = 70.0


def _trusted_surface_points(model: SparseModel) -> NDArray[np.bool_]:
    """Well-observed, low-error, non-outlying points for the plane fit."""
    trusted_point_mask = np.ones(len(model.points), bool)
    well_tracked = model.point_track_lengths >= 3
    if well_tracked.sum() >= MIN_POINTS_FOR_FIT:
        trusted_point_mask &= well_tracked
    reprojection_error_limit_px = max(1.0, 2.0 * float(np.median(model.point_errors_px[trusted_point_mask])))
    low_reprojection_error = model.point_errors_px <= reprojection_error_limit_px
    if (trusted_point_mask & low_reprojection_error).sum() >= MIN_POINTS_FOR_FIT:
        trusted_point_mask &= low_reprojection_error
    robust_point_center = np.median(model.points[trusted_point_mask], axis=0)
    robust_point_spread = 1.4826 * np.median(np.abs(model.points[trusted_point_mask] - robust_point_center), axis=0) + 1e-12
    non_outlier = np.all(np.abs(model.points - robust_point_center) < 6.0 * robust_point_spread, axis=1)
    if (trusted_point_mask & non_outlier).sum() >= MIN_POINTS_FOR_FIT:
        trusted_point_mask &= non_outlier
    return trusted_point_mask


def canonicalize(
    model: SparseModel, standoff_mm: float, surface_point_mask: NDArray[np.bool_] | None = None
) -> CanonicalPoses:
    """Scale to ``standoff_mm`` mean camera-to-surface depth and align the axes.

    ``surface_point_mask`` (m,) restricts the points that define the surface, i.e. the
    depth that ``standoff_mm`` refers to and the plane that sets the axes, for
    example to the sparse points that land on skin rather than on the table.
    """
    if standoff_mm <= 0:
        raise ValueError("standoff_mm must be positive")
    if len(model.image_names) < 2 or len(model.points) < MIN_POINTS_FOR_FIT:
        raise ValueError("reconstruction is too small to canonicalize")

    rotations_cam2world_colmap = np.transpose(model.rotations_cam_from_world, (0, 2, 1))  # R^T
    camera_centers_colmap_units = -np.einsum(
        "nij,nj->ni", rotations_cam2world_colmap, model.translations_cam_from_world
    )

    if surface_point_mask is not None and int(surface_point_mask.sum()) < MIN_POINTS_FOR_FIT:
        raise ValueError(
            f"only {int(surface_point_mask.sum())} COLMAP points lie on the masked surface (need "
            f"{MIN_POINTS_FOR_FIT}); the masked region has too little texture for COLMAP, "
            "so scale and axes cannot be taken from it"
        )
    visible_surface_depths_colmap_units = np.full(len(model.image_names), np.nan)
    for index, visible in enumerate(model.visible_points):
        if surface_point_mask is not None:
            visible = visible[surface_point_mask[visible]]
        if len(visible) >= 10:
            camera_points = model.points[visible] @ model.rotations_cam_from_world[index].T
            camera_points = camera_points + model.translations_cam_from_world[index]
            visible_surface_depths_colmap_units[index] = np.median(camera_points[:, 2])
    valid_surface_depths_colmap_units = visible_surface_depths_colmap_units[
        np.isfinite(visible_surface_depths_colmap_units) & (visible_surface_depths_colmap_units > 0)
    ]
    if len(valid_surface_depths_colmap_units) == 0:
        raise ValueError("no image sees enough points in front of it to measure depth")
    visible_surface_depths_colmap_units = np.where(
        np.isfinite(visible_surface_depths_colmap_units) & (visible_surface_depths_colmap_units > 0),
        visible_surface_depths_colmap_units,
        np.median(valid_surface_depths_colmap_units),
    )
    mm_per_colmap_unit = standoff_mm / float(np.median(valid_surface_depths_colmap_units))

    trusted_surface_point_mask = _trusted_surface_points(model)
    if surface_point_mask is not None:
        trusted_surface_point_mask &= surface_point_mask
        if int(trusted_surface_point_mask.sum()) < MIN_POINTS_FOR_FIT:
            raise ValueError("too few well-tracked points on the masked surface to fit its plane")
    surface_origin_colmap_units = model.points[trusted_surface_point_mask].mean(axis=0)
    _, _, principal_axes = np.linalg.svd(
        model.points[trusted_surface_point_mask] - surface_origin_colmap_units, full_matrices=False
    )
    z_axis = principal_axes[2]
    if float(np.mean((camera_centers_colmap_units - surface_origin_colmap_units) @ z_axis)) < 0:
        z_axis = -z_axis
    x_axis = principal_axes[0] - (principal_axes[0] @ z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    camera_path_colmap_units = camera_centers_colmap_units[-1] - camera_centers_colmap_units[0]
    if float(camera_path_colmap_units @ x_axis) < 0:  # +x follows the capture order
        x_axis = -x_axis
    y_axis = np.cross(z_axis, x_axis)
    canonical_from_world: FloatArray = np.stack([x_axis, y_axis, z_axis], axis=0)  # rows; det = +1

    camera_centers_mm = (
        mm_per_colmap_unit * (camera_centers_colmap_units - surface_origin_colmap_units) @ canonical_from_world.T
    )
    rotations_cam2world_canonical = np.einsum("ij,njk->nik", canonical_from_world, rotations_cam2world_colmap)
    points_mm = mm_per_colmap_unit * (model.points - surface_origin_colmap_units) @ canonical_from_world.T

    surface = points_mm[trusted_surface_point_mask]
    relief_ratio = float(np.std(surface[:, 2]) / max(np.std(surface[:, 0]), 1e-9))
    optical_axes = rotations_cam2world_canonical[:, :, 2]
    view_angles = np.degrees(np.arccos(np.clip(-optical_axes[:, 2], -1.0, 1.0)))
    below_plane = int(np.sum(camera_centers_mm[:, 2] <= 0))
    diagnostics: AlignmentDiagnostics = {
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
        scale_mm_per_colmap_unit=mm_per_colmap_unit,
        rotations_cam2world=rotations_cam2world_canonical,
        camera_centers_mm=camera_centers_mm,
        surface_depths_mm=mm_per_colmap_unit * visible_surface_depths_colmap_units,
        points_mm=points_mm,
        surface_point_mask=trusted_surface_point_mask,
        diagnostics=diagnostics,
        warnings=warnings,
    )
