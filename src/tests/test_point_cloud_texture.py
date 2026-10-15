"""Full photo -> sim-capture export -> registered RGB cloud regression."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from freehand_scene import Camera, look_at, make_scene, render_view
from processing.registration_surface import Surface
from numpy.typing import NDArray
from sim_capture.export import write_capture_folder
from sim_capture.model import CanonicalPoses, SparseModel


def test_photo_pipeline_preserves_two_sides_and_colors_in_a_vertical_gauge(tmp_path: Path) -> None:
    """Two textured sheets share XY, then rotate vertical: no world heightfield works."""
    images = tmp_path / "photos"
    images.mkdir()
    capture = tmp_path / "capture"
    capture.mkdir()
    rng = np.random.default_rng(4)
    xy = rng.uniform([-24, -20], [24, 20], (250, 2))
    points = np.concatenate(
        [
            np.c_[xy, 8 + 0.05 * xy[:, 0]],
            np.c_[xy, -8 - 0.05 * xy[:, 0]],
        ]
    )
    # Turn the common surface normal to world X. Each sheet has many heights
    # at the same world (x,y), and the two sheets must never merge.
    gauge = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    cameras: list[Camera] = []
    names: list[str] = []
    visible: list[NDArray[np.int64]] = []
    for side, sign in enumerate((1, -1)):
        scene = make_scene(seed=3 + side, fx=420, k1=-0.04, size=(320, 240))
        xx, _ = np.meshgrid(scene.surface.xs, scene.surface.ys)
        scene.surface = Surface(scene.surface.xs, scene.surface.ys, sign * (8 + 0.05 * xx))
        # Strongly distinct BGR hues allow an RGB-order / back-side leakage check.
        texture = scene.texture[:, :, 2].copy()
        channels = (0.35, 0.6, 1.0) if side == 0 else (1.0, 0.6, 0.35)
        scene.texture = np.stack([texture * channel for channel in channels], axis=-1).astype(
            np.uint8
        )
        for offset in (-15, 0, 15):
            center = np.array([float(offset), 0.0, sign * 100.0])
            camera = Camera(
                look_at(center, np.array([0.0, 0.0, sign * 8.0]), roll_deg=offset), center
            )
            name = f"v{len(names)}.jpg"
            assert cv2.imwrite(str(images / name), render_view(scene, camera))
            cameras.append(Camera(gauge @ camera.R, gauge @ camera.C))
            names.append(name)
            visible.append(np.arange(side * 250, (side + 1) * 250, dtype=np.int64))
    world_points = points @ gauge.T
    centers = np.stack([camera.C for camera in cameras])
    rotations = np.stack([camera.R for camera in cameras])
    model = SparseModel(
        image_names=names,
        rotations_cam_from_world=rotations.transpose(0, 2, 1),
        translations_cam_from_world=-np.einsum("nji,nj->ni", rotations, centers),
        points=world_points,
        point_errors_px=np.zeros(500),
        point_track_lengths=np.full(500, 3, dtype=np.int64),
        point_colors=np.full((500, 3), 128, dtype=np.uint8),
        visible_points=visible,
        fx=420.0,
        cx=160.0,
        cy=120.0,
        k1=-0.04,
        width=320,
        height=240,
        mean_reprojection_error_px=0.0,
    )
    canonical = CanonicalPoses(
        scale_mm_per_colmap_unit=1.0,
        rotations_cam2world=rotations,
        camera_centers_mm=centers,
        surface_depths_mm=np.full(6, 92.0),
        points_mm=world_points,
        surface_point_mask=np.ones(500, dtype=bool),
        diagnostics={
            "relief_ratio": 1.0,
            "max_view_angle_deg": 180.0,
            "median_view_angle_deg": 0.0,
            "cameras_below_surface": 0.5,
            "surface_extent_x_mm": 20.0,
            "surface_extent_y_mm": 40.0,
        },
    )
    write_capture_folder(capture, images, model, canonical, 92.0)
    masks = capture / "sim" / "masks"
    masks.mkdir()
    # Auto-detected masks remove the top half of each sensor, including pixels
    # inside the convex hull of the surviving keypoints.
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[125:220, 30:290] = 255
    mask[160:180, 130:190] = 0  # internal hole must stay empty
    for name in names:
        assert cv2.imwrite(str(masks / f"{Path(name).stem}.png"), mask)

    # Exercise the real entry point, SIFT, epipolar verification and BA. No mocks.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "processing.texture_scan_3d",
            str(capture),
            "--rounds",
            "3",
            "--min-inliers",
            "15",
            "--nfeatures",
            "3000",
            "--max-corr-per-pair",
            "600",
            "--point-pitch-mm",
            "0.6",
            "--max-edge-mm",
            "15",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    output = capture / "texture3d"
    report = json.loads((output / "report.json").read_text())
    assert report["mask_dir"] == str(masks)
    assert report["reprojection_rms_px"] < 1.5
    assert report["points"] > 1000
    assert report["landmarks"] > 80
    raw = (output / "texture.ply").read_bytes()
    header, payload = raw.split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0" in header
    records = np.frombuffer(payload, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    cloud, colors = records["xyz"], records["rgb"].astype(float)
    assert len(cloud) == report["points"]
    assert np.isfinite(cloud).all()
    positive, negative = cloud[:, 0] > 0, cloud[:, 0] < 0
    assert positive.sum() > 500 and negative.sum() > 500
    # True surface equation after the rigid rotation, in mm. No in-between sheet.
    error = np.abs(np.abs(cloud[:, 0]) - (8 - 0.05 * cloud[:, 2]))
    assert np.percentile(error, 95) < 1.0
    assert np.mean(np.abs(cloud[:, 0]) > 6) > 0.99
    assert np.mean(colors[positive, 0] - colors[positive, 2]) > 45
    assert np.mean(colors[negative, 2] - colors[negative, 0]) > 45
    assert np.std(colors[positive, 0]) > 15  # photo detail, not a uniform landmark tint
    # A point must project into the mask of at least one camera on its own side.
    from processing.registration_geometry import project_world_points_into_camera

    covered = np.zeros(len(cloud), dtype=bool)
    for camera in report["frames"]:
        rotation = np.asarray(camera["R_cam2world"])
        center = np.asarray(camera["C_mm"])
        pixels, depths = project_world_points_into_camera(
            cloud.astype(np.float64),
            rotation,
            center,
            report["fx_full_px"],
            report["k1"],
            160.0,
            120.0,
        )
        px, py = np.rint(pixels).astype(int).T
        in_image = (depths > 0) & (px >= 0) & (px < 320) & (py >= 0) & (py < 240)
        on_side = cloud[:, 0] * center[0] > 0
        indices = np.where(in_image & on_side)[0]
        covered[indices] |= mask[py[indices], px[indices]] > 0
    assert covered.all()
