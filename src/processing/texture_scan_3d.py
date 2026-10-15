"""Build a photo-colored 3D point cloud from a sim-capture folder.

Reuses registration through bundle adjustment, then samples local landmark
triangles in each photograph instead of fitting z=f(x,y) and unwrapping it.
Output: <capture_dir>/texture3d/texture.ply (RGB, millimetres) and report.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np

from ._reconstruction.parser import CameraRegistrationArguments, add_camera_registration_arguments
from ._reconstruction.problem import build_scan_reconstruction_problem
from .point_cloud_texture import PointCloudTextureOptions, render_point_cloud_texture
from .registration_export import export_colored_point_cloud_ply
from .registration_features import _apply_processing_memory_limit
from .registration_geometry import bundle_adjust_camera_poses_and_feature_tracks
from .registration_texture import estimate_camera_frame_texture_gains


class PointCloudCliArguments(CameraRegistrationArguments):
    point_pitch_mm: float
    max_edge_mm: float
    max_reprojection_px: float
    max_points: int


class PointCloudFrameReport(TypedDict):
    image: str
    R_cam2world: list[list[float]]
    C_mm: list[float]


class PointCloudReport(TypedDict):
    capture_dir: str
    poses_from: str
    mask_dir: str | None
    units: str
    geometry: str
    color_selection: str
    points: int
    landmarks: int
    reprojection_rms_px: float
    fx_full_px: float
    image_size: list[int]
    k1: float
    downscale: int
    point_pitch_mm: float
    max_edge_mm: float
    max_reprojection_px: float
    max_incidence_deg: float
    frames: list[PointCloudFrameReport]


def parse_arguments(argv: Sequence[str] | None = None) -> PointCloudCliArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    add_camera_registration_arguments(parser)
    parser.set_defaults(
        downscale=1,
        group_align="none",
        sigma_r=2.0,
        max_incidence_deg=65.0,
        blend_sharpness=20.0,
    )
    parser.add_argument(
        "--point-pitch-mm",
        type=float,
        default=0.25,
        help="approximate sample spacing and fusion voxel size, mm (default: 0.25)",
    )
    parser.add_argument(
        "--max-edge-mm",
        type=float,
        default=10.0,
        help="largest supported triangle edge, mm (default: 10); larger fills more gaps",
    )
    parser.add_argument(
        "--max-reprojection-px",
        type=float,
        default=3.0,
        help="maximum landmark error at --downscale analysis resolution (default: 3)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5_000_000,
        help="abort if the fused cloud exceeds this budget (default: 5000000)",
    )
    args = parser.parse_args(argv, namespace=PointCloudCliArguments())
    capture = Path(args.capture_dir)
    if args.rig_from:
        parser.error(
            "this entry point takes sim-capture output; use --poses-from instead of --rig-from"
        )
    args.poses_from = args.poses_from or str(capture / "sim" / "poses.json")
    if not Path(args.poses_from).is_file():
        parser.error(f"poses file not found: {args.poses_from}")
    if args.mask_dir is None and (capture / "sim" / "masks").is_dir():
        args.mask_dir = str(capture / "sim" / "masks")
    args.out = args.out or str(capture / "texture3d")
    # A nonzero fx_full locks the supplied COLMAP intrinsics in the shared BA;
    # its numeric value is ignored for external poses. Zero enables refinement.
    for name, value in (
        ("downscale", args.downscale),
        ("rounds", args.rounds),
        ("nfeatures", args.nfeatures),
        ("min_inliers", args.min_inliers),
        ("max_partners", args.max_partners),
        ("max_corr_per_pair", args.max_corr_per_pair),
    ):
        if value < 1:
            parser.error(f"{name} must be positive")
    for name, value in (
        ("sigma_px", args.sigma_px),
        ("sigma_t", args.sigma_t),
        ("sigma_r", args.sigma_r),
    ):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    try:
        _texture_options(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def _texture_options(args: PointCloudCliArguments) -> PointCloudTextureOptions:
    return PointCloudTextureOptions(
        point_pitch_mm=args.point_pitch_mm,
        max_edge_mm=args.max_edge_mm,
        max_reprojection_px=args.max_reprojection_px,
        max_incidence_deg=args.max_incidence_deg,
        blend_sharpness=args.blend_sharpness,
        max_points=args.max_points,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    cap = _apply_processing_memory_limit()
    try:
        problem = build_scan_reconstruction_problem(args, cap, spatial_overlap=True)
        print("[6/9] refining 3D landmarks and camera poses")
        result = bundle_adjust_camera_poses_and_feature_tracks(
            problem.frames, problem.pairs, problem.rig_model, problem.R0, problem.C0, args
        )
        if len(result["X"]) < 3 or not np.isfinite(result["rms"]):
            raise ValueError("too few reliable 3D landmarks; check photo overlap and masks")
        gains = None
        if args.lf_gain != "off":
            gains = estimate_camera_frame_texture_gains(
                problem.frames,
                problem.rig_model,
                result["obs_frame"],
                result["obs_uv"],
                result["obs_track"],
                result["err"],
                mode=args.lf_gain,
            )
        print("[7/9] sampling photo texture on local 3D patches")
        cloud = render_point_cloud_texture(
            problem.frames, problem.rig_model, result, _texture_options(args), gains
        )
        output = Path(problem.out_dir)
        print("[8/9] writing RGB point cloud (millimetres)")
        export_colored_point_cloud_ply(output / "texture.ply", cloud.points_mm, cloud.colors_rgb)
        assert args.poses_from is not None
        report: PointCloudReport = {
            "capture_dir": str(Path(args.capture_dir).resolve()),
            "poses_from": str(Path(args.poses_from).resolve()),
            "mask_dir": args.mask_dir,
            "units": "mm",
            "geometry": "per-view piecewise-planar landmark triangles",
            "color_selection": "best border/centre/incidence weight per 3D voxel",
            "points": len(cloud.points_mm),
            "landmarks": len(result["X"]),
            "reprojection_rms_px": result["rms"],
            "fx_full_px": problem.rig_model.fx * args.downscale,
            "k1": problem.rig_model.k1,
            "image_size": [
                round(2 * problem.rig_model.cx * args.downscale),
                round(2 * problem.rig_model.cy * args.downscale),
            ],
            "downscale": args.downscale,
            "point_pitch_mm": args.point_pitch_mm,
            "max_edge_mm": args.max_edge_mm,
            "max_reprojection_px": args.max_reprojection_px,
            "max_incidence_deg": args.max_incidence_deg,
            "frames": [
                {
                    "image": Path(frame.image_path).name,
                    "R_cam2world": result["R"][frame.idx].tolist(),
                    "C_mm": result["C"][frame.idx].tolist(),
                }
                for frame in problem.frames
            ],
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"[9/9] done: {output / 'texture.ply'}")
    except MemoryError:
        sys.exit("openderm-texture-3d: RAM cap reached; increase --point-pitch-mm or --downscale")
    except (ValueError, OSError) as error:
        sys.exit(f"openderm-texture-3d: {error}")


if __name__ == "__main__":
    main()
