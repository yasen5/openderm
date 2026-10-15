"""Photo-colored 3D patches directly from a saved COLMAP SIMPLE_RADIAL model.

No feature extraction, matching, or optimization. Geometry is interpolated
between observed sparse landmarks; extra samples do not add measured detail.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, QhullError

from .alignment import canonicalize
from .colmap_runner import _ColmapReconstruction, _read_model
from .masks import check_masks, skin_point_filter
from .model import FloatArray, SparseModel


@dataclass(frozen=True)
class TextureOptions:
    standoff_mm: float = 500.0
    pixel_step: int = 4
    voxel_mm: float = 0.5
    max_edge_mm: float = 15.0
    max_error_px: float = 3.0

    def __post_init__(self) -> None:
        if self.pixel_step < 1:
            raise ValueError("pixel_step must be positive")
        for value in (self.standoff_mm, self.voxel_mm, self.max_edge_mm, self.max_error_px):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("distances and error limits must be finite and positive")


class FrameReport(TypedDict):
    image: str
    sampled_points: int
    retained_points: int


class TextureReport(TypedDict):
    model: str
    images: str
    masks: str | None
    units: str
    geometry: str
    standoff_mm: float
    scale_mm_per_colmap_unit: float
    pixel_step: int
    voxel_mm: float
    max_edge_mm: float
    max_error_px: float
    registered_frames: int
    sampled_frames: int
    contributing_frames: int
    landmarks: int
    points: int
    frames: list[FrameReport]


def _sample_view(
    model: SparseModel,
    index: int,
    points_mm: FloatArray,
    trusted: NDArray[np.bool_],
    image_dir: Path,
    mask_dir: Path | None,
    options: TextureOptions,
) -> tuple[FloatArray, NDArray[np.uint8]]:
    """Intersect undistorted photo rays with local landmark triangles."""
    empty = (np.empty((0, 3), np.float64), np.empty((0, 3), np.uint8))
    name = model.image_names[index]
    photo = cv2.imread(str(image_dir / name), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if photo is None or photo.shape[:2] != (model.height, model.width):
        raise ValueError(f"missing photo or dimensions differ from COLMAP: {image_dir / name}")
    mask = None
    if mask_dir is not None:
        mask = cv2.imread(str(mask_dir / f"{Path(name).stem}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"cannot read mask for {name}")
        mask = cv2.resize(mask, (model.width, model.height), interpolation=cv2.INTER_NEAREST)

    visible = model.visible_points[index]
    camera_points = (
        model.points[visible] @ model.rotations_cam_from_world[index].T
        + model.translations_cam_from_world[index]
    )
    in_front = camera_points[:, 2] > 0
    safe_depth = np.maximum(camera_points[:, 2:], 1e-12)
    normalized = camera_points[:, :2] / safe_depth
    projected = model.fx * normalized * (1 + model.k1 * np.sum(normalized**2, axis=1, keepdims=True))
    projected += [model.cx, model.cy]
    good = trusted[visible] & in_front
    good &= np.linalg.norm(projected - model.visible_xy[index], axis=1) <= options.max_error_px
    if mask is not None:
        xy = np.rint(projected).astype(np.int64)
        good &= mask[np.clip(xy[:, 1], 0, model.height - 1), np.clip(xy[:, 0], 0, model.width - 1)] > 127
    if good.sum() < 3:
        return empty
    visible, camera_points, normalized = visible[good], camera_points[good], normalized[good]
    try:
        triangles = Delaunay(normalized)
    except QhullError:
        return empty
    vertices = points_mm[visible][triangles.simplices]
    edges = vertices - np.roll(vertices, 1, axis=1)
    allowed = np.max(np.linalg.norm(edges, axis=2), axis=1) <= options.max_edge_mm
    # Reject grazing triangles, whose interpolated depth is poorly constrained.
    camera_vertices = camera_points[triangles.simplices]
    normal = np.cross(camera_vertices[:, 1] - camera_vertices[:, 0], camera_vertices[:, 2] - camera_vertices[:, 0])
    center = camera_vertices.mean(axis=1)
    cosine = np.abs(np.sum(normal * center, axis=1)) / np.maximum(
        np.linalg.norm(normal, axis=1) * np.linalg.norm(center, axis=1), 1e-12
    )
    allowed &= cosine >= math.cos(math.radians(75))

    columns, rows = np.meshgrid(
        np.arange(0, model.width, options.pixel_step), np.arange(0, model.height, options.pixel_step)
    )
    pixels = np.column_stack((columns.ravel(), rows.ravel()))
    if mask is not None:
        pixels = pixels[mask[pixels[:, 1], pixels[:, 0]] > 127]
    intrinsics = np.array([[model.fx, 0, model.cx], [0, model.fx, model.cy], [0, 0, 1.0]])
    rays = cv2.undistortPoints(
        pixels.astype(np.float64).reshape(-1, 1, 2), intrinsics, np.array([model.k1, 0., 0., 0.])
    ).reshape(-1, 2) if len(pixels) else np.empty((0, 2), np.float64)
    simplex = triangles.find_simplex(rays)
    valid = (simplex >= 0) & allowed[np.maximum(simplex, 0)]
    pixels, rays, simplex = pixels[valid], rays[valid], simplex[valid]
    transform = triangles.transform[simplex]
    barycentric = np.einsum("nij,nj->ni", transform[:, :2], rays - transform[:, 2])
    weights = np.column_stack((barycentric, 1 - barycentric.sum(axis=1)))
    weights /= camera_vertices[simplex, :, 2]
    weights /= weights.sum(axis=1, keepdims=True)
    xyz = np.einsum("ni,nij->nj", weights, vertices[simplex])
    rgb = photo[pixels[:, 1], pixels[:, 0], ::-1]
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)


def reconstruct_texture(
    model_dir: Path, image_dir: Path, out_dir: Path,
    options: TextureOptions = TextureOptions(), mask_dir: Path | None = None,
) -> TextureReport:
    """Load saved COLMAP geometry, sample photos, and export a binary RGB PLY."""
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_dir))
    # pycolmap's extension types lack complete stubs; narrow at this boundary.
    model = _read_model(cast(_ColmapReconstruction, reconstruction), [], [])
    surface = None
    if mask_dir is not None:
        check_masks(mask_dir, model.image_names)
        surface = skin_point_filter(model, mask_dir)
    canonical = canonicalize(model, options.standoff_mm, surface)
    trusted = (model.point_track_lengths >= 3) & (model.point_errors_px <= options.max_error_px)
    trusted &= np.isfinite(model.points).all(axis=1)
    if surface is not None:
        trusted &= surface
    clouds: list[FloatArray] = []
    colors: list[NDArray[np.uint8]] = []
    owners: list[NDArray[np.int32]] = []
    frames: list[FrameReport] = []
    print(f"Loaded {len(model.image_names)} COLMAP views, {int(trusted.sum())} trusted landmarks", flush=True)
    for index, name in enumerate(model.image_names):
        xyz, rgb = _sample_view(
            model, index, canonical.points_mm, trusted,
            image_dir, mask_dir, options,
        )
        frames.append(FrameReport(image=name, sampled_points=len(xyz), retained_points=0))
        if len(xyz):
            _, keep = np.unique(np.floor(xyz / options.voxel_mm).astype(np.int64), axis=0, return_index=True)
            clouds.append(xyz[keep])
            colors.append(rgb[keep])
            owners.append(np.full(len(keep), index, np.int32))
        if (index + 1) % 25 == 0 or index + 1 == len(model.image_names):
            print(f"Sampled {index + 1}/{len(model.image_names)} views", flush=True)
    if not clouds:
        raise ValueError("no supported patches; check masks, landmark errors, and max-edge-mm")
    xyz_all, rgb_all, owner_all = np.concatenate(clouds), np.concatenate(colors), np.concatenate(owners)
    # One final fusion pass. First registered view wins duplicate voxel colors.
    _, keep = np.unique(np.floor(xyz_all / options.voxel_mm).astype(np.int64), axis=0, return_index=True)
    counts = np.bincount(owner_all[keep], minlength=len(frames))
    for frame, count in zip(frames, counts):
        frame["retained_points"] = int(count)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = np.empty(len(keep), dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
    records["xyz"], records["rgb"] = xyz_all[keep], rgb_all[keep]
    with (out_dir / "texture.ply").open("wb") as stream:
        stream.write((
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(records)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        ).encode("ascii"))
        records.tofile(stream)
    report = TextureReport(
        model=str(model_dir.resolve()), images=str(image_dir.resolve()),
        masks=str(mask_dir.resolve()) if mask_dir else None, units="mm",
        geometry="piecewise-planar interpolation of unchanged COLMAP landmarks",
        standoff_mm=options.standoff_mm, scale_mm_per_colmap_unit=canonical.scale_mm_per_colmap_unit,
        pixel_step=options.pixel_step, voxel_mm=options.voxel_mm, max_edge_mm=options.max_edge_mm,
        max_error_px=options.max_error_px, registered_frames=len(frames),
        sampled_frames=sum(frame["sampled_points"] > 0 for frame in frames),
        contributing_frames=int(np.count_nonzero(counts)), landmarks=int(trusted.sum()),
        points=len(records), frames=frames,
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(records):,} points from {report['contributing_frames']} views to {out_dir / 'texture.ply'}", flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="saved COLMAP sparse model directory")
    parser.add_argument("images", type=Path, help="original photos with COLMAP image names and dimensions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--masks", type=Path, help="optional <image-stem>.png masks")
    parser.add_argument("--standoff-mm", type=float, default=500, help="assumed median camera depth; sets scale")
    parser.add_argument("--pixel-step", type=int, default=4)
    parser.add_argument("--voxel-mm", type=float, default=0.5)
    parser.add_argument("--max-edge-mm", type=float, default=15)
    parser.add_argument("--max-error-px", type=float, default=3)
    args = parser.parse_args(argv)
    try:
        reconstruct_texture(
            args.model, args.images, args.out,
            TextureOptions(args.standoff_mm, args.pixel_step, args.voxel_mm, args.max_edge_mm, args.max_error_px),
            args.masks,
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"openderm-colmap-texture: {error}\n")


if __name__ == "__main__":
    main()
