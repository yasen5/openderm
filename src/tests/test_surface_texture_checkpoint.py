"""Cached registration -> original texture projection -> textured mesh -> HTTP."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from urllib.request import urlopen

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from freehand_scene import look_at
from processing.registration_checkpoint import save_registration
from processing.registration_features import Frame
from processing.registration_geometry import (
    BundleAdjustmentResult,
    RigModel,
    project_world_points_into_camera,
)
from processing.view_point_cloud import create_viewer_server


def test_cached_surface_mesh_projects_matchless_photo_and_serves_viewer(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    capture.mkdir()
    masks = capture / "masks"
    masks.mkdir()
    xx, yy = np.meshgrid(np.linspace(-24, 24, 5), np.linspace(-24, 24, 5))
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    xy = xy[np.any(xy != 0, axis=1)]  # no landmark at the photo feature's centre
    points = np.column_stack((xy, 0.08 * xy[:, 0] + 0.04 * xy[:, 1]))
    frames: list[Frame] = []
    rotations, centers, observations = [], [], []
    for index, offset in enumerate((-8, 8, 0)):
        path = capture / f"v{index}.png"
        image = np.full((240, 320, 3), (30, 80, 200), np.uint8)
        image[110:130, 150:170] = (200, 40, 20)  # unmatched blue photo feature
        assert cv2.imwrite(str(path), image)
        mask = np.zeros((240, 320), np.uint8)
        if index == 2:
            mask[35:205, 45:275] = 255
            mask[80:100, 190:210] = 0
        assert cv2.imwrite(str(masks / path.name), mask)
        center = np.array([float(offset), 0., 100.])
        rotation = look_at(center, np.zeros(3))
        rotations.append(rotation)
        centers.append(center)
        frames.append(Frame(
            idx=index, station=index + 1, row=1, col=index + 1, phase="+y", sidecar="",
            image_path=str(path), gauge=np.zeros(3), rx=0., standoff=100., settled=True,
        ))
        (capture / f"v{index}.json").write_text(json.dumps({
            "image": str(path), "station": index + 1, "row": 1, "col": index + 1,
            "x_mm": 0, "y_mm": 0, "z_mm": 0, "rx_rad": 0,
            "sensor1_mm": 100, "sensor2_mm": 100,
        }))
        if index < 2:
            pixels, _ = project_world_points_into_camera(points, rotation, center, 420., 0., 160., 120.)
            observations.append(pixels)
    count = len(points)
    result = BundleAdjustmentResult(
        X=points, track_err=np.zeros(count), obs_frame=np.repeat([0, 1], count).astype(np.int32),
        obs_uv=np.concatenate(observations), obs_track=np.tile(np.arange(count, dtype=np.int32), 2),
        err=np.zeros(2 * count), R=np.stack(rotations), C=np.stack(centers), rms=0., history=[],
    )
    checkpoint = tmp_path / "registration.npz"
    base_rotation = np.array([[np.cos(.1), -np.sin(.1), 0.], [np.sin(.1), np.cos(.1), 0.], [0., 0., 1.]])
    base_translation = np.array([3., 5., 0.])
    save_registration(checkpoint, frames, RigModel(
        420., 0., 160., 120., 1., np.zeros(3), np.eye(3), 0.,
        base_R=base_rotation, base_t=base_translation,
    ), result)
    output = tmp_path / "surface"
    completed = subprocess.run([
        sys.executable, "-m", "processing.register_scan_3d", str(capture),
        "--registration-from", str(checkpoint), "--out", str(output), "--mask-dir", str(masks),
        "--surface-pitch", "9", "--texture-ppmm", "4", "--mesh-pitch", "0.5",
        "--blend", "two-band", "--device", "cpu",
    ], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "feature extraction and matching skipped" in completed.stdout
    assert "pose refinement skipped" in completed.stdout
    assert "extracting CLAHE-SIFT" not in completed.stdout
    placements = json.loads((output / "placements3d.json").read_text())
    assert placements["pose_source"] == "checkpoint"
    np.testing.assert_allclose(placements["rig_model"]["base_R"], base_rotation)
    np.testing.assert_allclose(placements["rig_model"]["base_t"], base_translation)
    for frame, center, rotation in zip(placements["frames"], centers, rotations):
        np.testing.assert_allclose(frame["C_mm"], center)
        np.testing.assert_allclose(frame["R_cam2world"], rotation)
    # Baseline photos are entirely masked: all texture must come from frame 2,
    # despite it having zero observations. Its unmatched centre feature survives.
    texture = cv2.imread(str(output / "texture.jpg"))
    coverage = cv2.imread(str(output / "coverage.png"), cv2.IMREAD_GRAYSCALE)
    assert texture is not None and coverage is not None
    covered = coverage > 0
    assert covered.sum() > 10_000
    assert np.count_nonzero(covered & (texture[:, :, 0] > texture[:, :, 2].astype(float) + 50)) > 100
    obj = (output / "surface_mesh.obj").read_text()
    vertices = np.asarray([[float(v) for v in line.split()[1:]] for line in obj.splitlines() if line.startswith("v ")])
    assert "mtllib surface_mesh.mtl" in obj and "\nf " in obj
    assert "map_Kd texture.jpg" in (output / "surface_mesh.mtl").read_text()
    # Only referenced vertices form the displayed mesh; the export also stores
    # unused grid vertices outside the textured region.
    faces = np.asarray([[int(v.split("/")[0]) - 1 for v in line.split()[1:]]
                        for line in obj.splitlines() if line.startswith("f ")])
    used = vertices[np.unique(faces)]
    np.testing.assert_allclose(used[:, 2], .08 * used[:, 0] + .04 * used[:, 1], atol=.01)
    assert np.ptp(vertices[:, 2]) > 3  # surface retains depth instead of flattening
    server = create_viewer_server(output, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5) as response:
            html = response.read().decode()
            assert response.headers.get_content_type() == "text/html"
        assert html == (output / "viewer.html").read_text()
        assert "__PAYLOAD__" not in html and 'id="three_b64"' in html
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
