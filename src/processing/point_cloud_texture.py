"""Photo texture on local 3D landmark triangles, without a global heightfield.

Each view triangulates only the landmarks it actually observes. Sensor samples
are lifted onto these triangles using perspective-correct interpolation, then
fused in world-space voxels. Geometry between landmarks is piecewise planar;
this increases texture density, not the measured geometric detail.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError

from .registration_features import Frame
from .registration_geometry import (
    BundleAdjustmentResult,
    FloatArray,
    RigModel,
    project_world_points_into_camera,
    undistort_image_points_to_normalized_camera,
)
from .registration_texture import (
    apply_camera_frame_texture_gain,
    camera_frame_texture_weights,
    load_camera_frame_mask,
)


@dataclass(frozen=True)
class PointCloudTextureOptions:
    point_pitch_mm: float = 0.25  # voxel side and approximate surface sample spacing
    max_edge_mm: float = 10.0  # do not bridge poorly supported regions / depth jumps
    max_reprojection_px: float = 3.0  # at feature-analysis resolution
    max_incidence_deg: float = 65.0
    blend_sharpness: float = 20.0
    max_points: int = 5_000_000

    def __post_init__(self) -> None:
        for name, value in (
            ("point_pitch_mm", self.point_pitch_mm),
            ("max_edge_mm", self.max_edge_mm),
            ("max_reprojection_px", self.max_reprojection_px),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.max_incidence_deg) or not 0 <= self.max_incidence_deg < 90:
            raise ValueError("max_incidence_deg must be in [0, 90)")
        if not math.isfinite(self.blend_sharpness) or self.blend_sharpness < 0:
            raise ValueError("blend_sharpness must be finite and nonnegative")
        if self.max_points < 1:
            raise ValueError("max_points must be positive")


@dataclass
class ColoredPointCloud:
    points_mm: FloatArray  # (N, 3), processing's world gauge, millimetres
    colors_rgb: NDArray[np.uint8]  # (N, 3), display RGB
    weights: FloatArray  # (N,), border / centre / incidence preference


def _fuse_voxels(
    clouds: Sequence[ColoredPointCloud], options: PointCloudTextureOptions
) -> ColoredPointCloud:
    """Keep the best photo sample per world-space voxel; never average away detail."""
    points = np.concatenate([cloud.points_mm for cloud in clouds])
    colors = np.concatenate([cloud.colors_rgb for cloud in clouds])
    weights = np.concatenate([cloud.weights for cloud in clouds])
    voxels = np.floor(points / options.point_pitch_mm).astype(np.int64)
    order = np.lexsort((-weights, voxels[:, 2], voxels[:, 1], voxels[:, 0]))
    sorted_voxels = voxels[order]
    first = np.r_[True, np.any(sorted_voxels[1:] != sorted_voxels[:-1], axis=1)]
    selected = order[first]
    if len(selected) > options.max_points:
        raise ValueError(
            f"point cloud exceeds --max-points {options.max_points}; "
            "increase --point-pitch-mm or raise --max-points"
        )
    return ColoredPointCloud(points[selected], colors[selected], weights[selected])


def _sample_frame(
    frame: Frame,
    points_mm: FloatArray,
    rotation: FloatArray,
    center_mm: FloatArray,
    rig: RigModel,
    options: PointCloudTextureOptions,
    gain: NDArray[np.float32] | None,
) -> Iterator[ColoredPointCloud]:
    image = cv2.imread(frame.image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"could not read image {frame.image_path}")
    height, width = image.shape[:2]
    fx, cx, cy = rig.fx * rig.downscale, rig.cx * rig.downscale, rig.cy * rig.downscale
    if abs(width - 2 * cx) > rig.downscale or abs(height - 2 * cy) > rig.downscale:
        raise ValueError(f"image dimensions differ from pose calibration: {frame.image_path}")
    _, depths = project_world_points_into_camera(points_mm, rotation, center_mm, fx, rig.k1, cx, cy)
    points_mm, depths = points_mm[depths > 0], depths[depths > 0]
    if len(points_mm) < 3:
        return
    camera_points = (points_mm - center_mm) @ rotation
    normalized = camera_points[:, :2] / depths[:, None]
    try:
        triangulation = Delaunay(normalized)
    except QhullError:
        return  # too few noncollinear visible landmarks
    vertices = points_mm[triangulation.simplices]
    edges = vertices - np.roll(vertices, 1, axis=1)
    valid_faces = np.linalg.norm(edges, axis=2).max(axis=1) <= options.max_edge_mm
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_faces &= normal_lengths > 1e-9
    normals /= np.maximum(normal_lengths[:, None], 1e-9)
    if not valid_faces.any():
        return

    # Process a narrow strip at a time, even for full-resolution photographs.
    stride = max(1, int(fx * options.point_pitch_mm / float(np.median(depths))))
    if math.ceil(width / stride) * math.ceil(height / stride) > 4 * options.max_points:
        raise ValueError("frame sampling exceeds point budget; increase --point-pitch-mm")
    mask = (
        None if frame.mask_path is None else load_camera_frame_mask(frame.mask_path, width, height)
    )
    xs = np.arange(stride // 2, width, stride, dtype=np.float64)
    for row in range(stride // 2, height, 32 * stride):
        ys = np.arange(row, min(row + 32 * stride, height), stride, dtype=np.float64)
        pixel_x, pixel_y = np.meshgrid(xs, ys)
        uv = np.stack([pixel_x.ravel(), pixel_y.ravel()], axis=1)
        queries = undistort_image_points_to_normalized_camera(uv, fx, rig.k1, cx, cy)
        simplex = triangulation.find_simplex(queries)
        valid = (simplex >= 0) & valid_faces[np.maximum(simplex, 0)]
        if mask is not None:
            valid &= mask[uv[:, 1].astype(int), uv[:, 0].astype(int)] > 0.5
        if not valid.any():
            continue
        uv, queries, simplex = uv[valid], queries[valid], simplex[valid]
        transforms = triangulation.transform[simplex]
        barycentric_xy = np.einsum("nij,nj->ni", transforms[:, :2], queries - transforms[:, 2])
        barycentric = np.c_[barycentric_xy, 1 - barycentric_xy.sum(axis=1)]
        # Image-space barycentric coordinates interpolate reciprocal depth.
        corners = triangulation.simplices[simplex]
        perspective = barycentric / depths[corners]
        perspective /= perspective.sum(axis=1, keepdims=True)
        points = np.einsum("ni,nij->nj", perspective, points_mm[corners])
        feather, preference = camera_frame_texture_weights(
            uv, width, height, options.blend_sharpness
        )
        weights = feather * preference
        if options.max_incidence_deg > 0:
            rays = points - center_mm
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            cosine = np.abs(np.einsum("ni,ni->n", normals[simplex], rays))
            high = math.cos(math.radians(max(options.max_incidence_deg - 15, 1)))
            low = math.cos(math.radians(options.max_incidence_deg))
            weights *= np.clip((cosine - low) / max(high - low, 1e-6), 0, 1)
        keep = weights > 0
        if not keep.any():
            continue
        uv, points, weights = uv[keep], points[keep], weights[keep]
        # Sample from the original sensor image (including radial distortion).
        # Nx1 remap arrays avoid OpenCV's 32767-pixel dimension limit.
        mapx = uv[:, 0].astype(np.float32).reshape(-1, 1)
        mapy = uv[:, 1].astype(np.float32).reshape(-1, 1)
        # Strip width can still exceed that limit on very wide photographs.
        for start in range(0, len(uv), 30_000):
            stop = start + 30_000
            color = cv2.remap(image, mapx[start:stop], mapy[start:stop], cv2.INTER_LINEAR).astype(
                np.float32
            )
            apply_camera_frame_texture_gain(
                color, mapx[start:stop], mapy[start:stop], width, height, gain
            )
            rgb = np.clip(np.rint(color[:, 0, ::-1]), 0, 255).astype(np.uint8)
            yield ColoredPointCloud(points[start:stop], rgb, weights[start:stop])


def render_point_cloud_texture(
    frames: Sequence[Frame],
    rig: RigModel,
    reconstruction: BundleAdjustmentResult,
    options: PointCloudTextureOptions,
    frame_gains: NDArray[np.float32] | None = None,
) -> ColoredPointCloud:
    """Texture locally supported 3D patches, preserving folds and separate sides."""
    points = reconstruction["X"]
    counts = np.bincount(reconstruction["obs_track"], minlength=len(points))
    trusted = (
        np.isfinite(points).all(axis=1)
        & (counts >= 2)
        & (reconstruction["track_err"] <= options.max_reprojection_px)
    )
    cloud: ColoredPointCloud | None = None
    for frame in frames:
        observed = np.unique(reconstruction["obs_track"][reconstruction["obs_frame"] == frame.idx])
        observed = observed[trusted[observed]]
        if len(observed) < 3:
            continue
        patches = list(
            _sample_frame(
                frame,
                points[observed],
                reconstruction["R"][frame.idx],
                reconstruction["C"][frame.idx],
                rig,
                options,
                None if frame_gains is None else frame_gains[frame.idx],
            )
        )
        if not patches:
            continue
        if cloud is not None:
            patches.insert(0, cloud)
        cloud = _fuse_voxels(patches, options)
        print(f"      frame {frame.idx + 1}/{len(frames)}: {len(cloud.points_mm)} textured points")
    if cloud is None:
        raise ValueError(
            "no supported 3D patches; check overlap, masks, --max-edge-mm and --max-reprojection-px"
        )
    return cloud
