"""Project every posed photo onto a shared, depth-tested 3D landmark surface.

Matches establish geometry once. Photos can texture visible triangles without
having observed their landmarks. Interpolation adds texture, not measured depth.
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
    frame_indices: NDArray[np.int32]  # (N,), winning photo, including matchless views


@dataclass(frozen=True)
class LandmarkSurface:
    vertices_mm: FloatArray  # (V,3), world millimetres
    faces: NDArray[np.int32]  # (F,3), vertex indices


def _fuse_voxels(
    clouds: Sequence[ColoredPointCloud], options: PointCloudTextureOptions
) -> ColoredPointCloud:
    """Keep the best photo sample per world-space voxel; never average away detail."""
    points = np.concatenate([cloud.points_mm for cloud in clouds])
    colors = np.concatenate([cloud.colors_rgb for cloud in clouds])
    weights = np.concatenate([cloud.weights for cloud in clouds])
    frame_indices = np.concatenate([cloud.frame_indices for cloud in clouds])
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
    return ColoredPointCloud(points[selected], colors[selected], weights[selected], frame_indices[selected])


def build_landmark_surface(
    frames: Sequence[Frame], reconstruction: BundleAdjustmentResult,
    options: PointCloudTextureOptions,
) -> LandmarkSurface:
    """Union the supported triangles from all views; do not flatten or bridge gaps."""
    points = reconstruction["X"]
    counts = np.bincount(reconstruction["obs_track"], minlength=len(points))
    trusted = (np.isfinite(points).all(axis=1) & (counts >= 2)
               & (reconstruction["track_err"] <= options.max_reprojection_px))
    faces: list[NDArray[np.int32]] = []
    for frame in frames:
        observed = np.unique(reconstruction["obs_track"][reconstruction["obs_frame"] == frame.idx])
        observed = observed[trusted[observed]]
        camera = (points[observed] - reconstruction["C"][frame.idx]) @ reconstruction["R"][frame.idx]
        observed, camera = observed[camera[:, 2] > 0], camera[camera[:, 2] > 0]
        if len(observed) < 3:
            continue
        try:
            triangles = Delaunay(camera[:, :2] / camera[:, 2:])
        except QhullError:
            continue
        indices = observed[triangles.simplices]
        vertices = points[indices]
        edges = vertices - np.roll(vertices, 1, axis=1)
        good = np.linalg.norm(edges, axis=2).max(axis=1) <= options.max_edge_mm
        good &= np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1) > 1e-9
        faces.append(np.asarray(indices[good], dtype=np.int32))
    if not faces or not any(len(part) for part in faces):
        raise ValueError("no supported 3D surface; check overlap, --max-edge-mm and --max-reprojection-px")
    unique = np.unique(np.sort(np.concatenate(faces), axis=1), axis=0)
    return LandmarkSurface(points, unique)


def _visible_faces(
    camera_vertices: FloatArray, faces: NDArray[np.int32],
    fx: float, cx: float, cy: float, width: int, height: int, stride: int,
) -> NDArray[np.int32]:
    """Rasterize an undistorted sensor grid with perspective-correct z buffering."""
    owners = np.full((math.ceil(height / stride), math.ceil(width / stride)), -1, np.int32)
    depth = np.full(owners.shape, np.inf)
    projected = fx * camera_vertices[:, :2] / np.maximum(camera_vertices[:, 2:], 1e-9)
    projected = (projected + [cx, cy] - stride / 2) / stride
    for face_index, face in enumerate(faces):
        z = camera_vertices[face, 2]
        if np.any(z <= 1e-6):
            continue
        triangle = projected[face]
        x0, y0 = np.maximum(np.ceil(triangle.min(axis=0)), 0).astype(int)
        x1, y1 = np.minimum(np.floor(triangle.max(axis=0)), np.array(owners.shape[::-1]) - 1).astype(int)
        if x1 < x0 or y1 < y0:
            continue
        a, b, c = triangle
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
        u = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
        v = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
        w = 1 - u - v
        inside = (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
        candidate_depth = 1 / np.maximum(u / z[0] + v / z[1] + w / z[2], 1e-12)
        region = depth[y0:y1 + 1, x0:x1 + 1]
        closer = inside & (candidate_depth < region)
        region[closer] = candidate_depth[closer]
        owners[y0:y1 + 1, x0:x1 + 1][closer] = face_index
    return owners


def _sample_frame(
    frame: Frame,
    surface: LandmarkSurface,
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
    camera_points = (surface.vertices_mm - center_mm) @ rotation
    depths = camera_points[np.unique(surface.faces), 2]
    depths = depths[depths > 0]
    if len(depths) == 0:
        return
    vertices = surface.vertices_mm[surface.faces]
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(normal_lengths[:, None], 1e-9)
    stride = max(1, int(fx * options.point_pitch_mm / float(np.median(depths))))
    if math.ceil(width / stride) * math.ceil(height / stride) > 4 * options.max_points:
        raise ValueError("frame sampling exceeds point budget; increase --point-pitch-mm")
    mask = (
        None if frame.mask_path is None else load_camera_frame_mask(frame.mask_path, width, height)
    )
    # Extend the undistorted sensor bounds so barrel distortion does not crop
    # valid source pixels at the photo edges.
    border = np.linspace(0, 1, 32)
    perimeter = np.concatenate([
        np.column_stack((border * width, np.zeros(32))),
        np.column_stack((border * width, np.full(32, height))),
        np.column_stack((np.zeros(32), border * height)),
        np.column_stack((np.full(32, width), border * height)),
    ])
    undistorted = undistort_image_points_to_normalized_camera(perimeter, fx, rig.k1, cx, cy) * fx + [cx, cy]
    origin = np.floor(undistorted.min(axis=0))
    extent = np.ceil(undistorted.max(axis=0) - origin).astype(int)
    if math.ceil(extent[0] / stride) * math.ceil(extent[1] / stride) > 4 * options.max_points:
        raise ValueError("undistorted frame exceeds point budget; increase --point-pitch-mm")
    ownership = _visible_faces(camera_points, surface.faces, fx, cx - origin[0], cy - origin[1],
                               int(extent[0]), int(extent[1]), stride)
    camera_triangles = camera_points[surface.faces]
    for row in range(0, ownership.shape[0], 32):
        block = ownership[row:row + 32]
        yy, xx = np.nonzero(block >= 0)
        if not len(xx):
            continue
        simplex = block[yy, xx]
        queries = (np.column_stack((xx, yy + row)) * stride + stride / 2 + origin - [cx, cy]) / fx
        triangles = camera_triangles[simplex]
        # A ray intersects the selected triangle's plane. The z buffer has
        # already discarded all farther triangles at the same ray.
        camera_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        rays = np.column_stack((queries, np.ones(len(queries))))
        ray_depth = np.sum(camera_normals * triangles[:, 0], axis=1) / np.sum(camera_normals * rays, axis=1)
        points = (rays * ray_depth[:, None]) @ rotation.T + center_mm
        uv, _ = project_world_points_into_camera(points, rotation, center_mm, fx, rig.k1, cx, cy)
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < height - 1)
        if mask is not None:
            valid &= mask[np.clip(np.rint(uv[:, 1]).astype(int), 0, height - 1),
                          np.clip(np.rint(uv[:, 0]).astype(int), 0, width - 1)] > 0.5
        if not valid.any():
            continue
        uv, points, simplex = uv[valid], points[valid], simplex[valid]
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
            yield ColoredPointCloud(points[start:stop], rgb, weights[start:stop],
                                    np.full(len(rgb), frame.idx, np.int32))


def render_point_cloud_texture(
    frames: Sequence[Frame],
    rig: RigModel,
    reconstruction: BundleAdjustmentResult,
    options: PointCloudTextureOptions,
    frame_gains: NDArray[np.float32] | None = None,
) -> ColoredPointCloud:
    """Texture the shared surface from every photo, independent of its matches."""
    surface = build_landmark_surface(frames, reconstruction, options)
    print(f"      shared surface: {len(surface.faces)} supported triangles")
    cloud: ColoredPointCloud | None = None
    for frame in frames:
        patches = list(
            _sample_frame(
                frame,
                surface,
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
