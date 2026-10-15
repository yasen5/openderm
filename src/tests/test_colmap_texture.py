"""Saved model -> textured PLY, without invoking reconstruction or processing."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pycolmap = pytest.importorskip("pycolmap")

from sim_capture.alignment import canonicalize
from sim_capture.colmap_runner import _ColmapReconstruction, _read_model
from sim_capture.masks import skin_point_filter


def test_saved_colmap_model_to_photo_colored_3d(tmp_path: Path) -> None:
    """Two separated, tilted sheets exercise depth, distortion, masks and RGB."""
    model_dir, images, masks, output = [tmp_path / name for name in ("model", "images", "masks", "out")]
    for directory in (model_dir, images, masks):
        directory.mkdir()
    (model_dir / "cameras.txt").write_text("1 SIMPLE_RADIAL 320 240 250 160 120 -0.03\n")
    xx, yy = np.meshgrid(np.r_[np.linspace(-30, -10, 6), np.linspace(10, 30, 6)], np.linspace(-20, 20, 7))
    zz = np.where(xx < 0, 100 + 0.4 * xx, 140 - 0.2 * xx)
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    rows, columns = np.mgrid[:240, :320]
    # Lossless photo values identify source pixel and RGB channel order.
    rgb = np.stack((columns % 256, rows, np.full_like(rows, 73)), axis=2).astype(np.uint8)
    mask = np.full((240, 320), 255, np.uint8)
    mask[108:133, :] = 0
    image_lines: list[str] = []
    for index, camera_x in enumerate((-3., 0., 3.), start=1):
        camera_points = points - [camera_x, 0, 0]
        normalized = camera_points[:, :2] / camera_points[:, 2:]
        xy = 250 * normalized * (1 - 0.03 * np.sum(normalized**2, axis=1, keepdims=True)) + [160, 120]
        name = f"view{index}.png"
        image_lines += [f"{index} 1 0 0 0 {-camera_x} 0 0 1 {name}",
                        " ".join(f"{u} {v} {point_id}" for point_id, (u, v) in enumerate(xy, start=1))]
        assert cv2.imwrite(str(images / name), rgb[:, :, ::-1])
        assert cv2.imwrite(str(masks / name), mask)
    (model_dir / "images.txt").write_text("\n".join(image_lines) + "\n")
    (model_dir / "points3D.txt").write_text("\n".join(
        f"{i + 1} {x} {y} {z} 0 0 0 0.1 1 {i} 2 {i} 3 {i}"
        for i, (x, y, z) in enumerate(points)
    ) + "\n")
    before = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in model_dir.iterdir()}
    # Fail if the standalone CLI imports any processing module or calls COLMAP
    # reconstruction functions. Reading Reconstruction is intentionally allowed.
    script = """
import importlib.abc
import sys
import pycolmap
class NoProcessing(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'processing' or fullname.startswith('processing.'):
            raise AssertionError('processing imported')
sys.meta_path.insert(0, NoProcessing())
def forbidden(*args, **kwargs):
    raise AssertionError('COLMAP reconstruction invoked')
for name in ('extract_features', 'match_exhaustive', 'match_sequential', 'incremental_mapping'):
    setattr(pycolmap, name, forbidden)
from sim_capture.texture import main
main(sys.argv[1:])
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(model_dir), str(images), "--out", str(output),
         "--masks", str(masks), "--standoff-mm", "120", "--pixel-step", "2", "--voxel-mm", "0.3"],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert before == {p.name: hashlib.sha256(p.read_bytes()).digest() for p in model_dir.iterdir()}
    report = json.loads((output / "report.json").read_text())
    assert report["registered_frames"] == report["sampled_frames"] == 3
    assert report["points"] > 1000
    raw = (output / "texture.ply").read_bytes()
    records = np.frombuffer(raw, dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,))],
                            offset=raw.index(b"end_header\n") + len(b"end_header\n"))
    assert len(records) == report["points"]
    assert np.isfinite(records["xyz"]).all()
    model = _read_model(cast(_ColmapReconstruction, pycolmap.Reconstruction(str(model_dir))), [], [])
    canonical = canonicalize(model, 120, skin_point_filter(model, masks))
    # Undo the export's similarity transform to inspect original COLMAP space.
    transform, _, _, _ = np.linalg.lstsq(np.column_stack((model.points, np.ones(len(points)))),
                                        canonical.points_mm, rcond=None)
    recovered = (records["xyz"] - transform[3]) @ np.linalg.inv(transform[:3])
    left = recovered[:, 0] < 0
    assert left.any() and (~left).any()
    expected_z = np.where(left, 100 + 0.4 * recovered[:, 0], 140 - 0.2 * recovered[:, 0])
    np.testing.assert_allclose(recovered[:, 2], expected_z, atol=0.001)
    assert np.all(np.abs(recovered[:, 0]) >= 9.99)  # no interpolated bridge
    assert np.ptp(records["xyz"][:, 2]) > 5  # retained 3D relief
    # Every retained sample matches its owning photograph and lies inside its mask.
    for index, frame in enumerate(report["frames"]):
        if frame["retained_points"] == 0:
            continue
        camera_points = recovered - [(-3., 0., 3.)[index], 0, 0]
        normalized = camera_points[:, :2] / camera_points[:, 2:]
        xy = 250 * normalized * (1 - 0.03 * np.sum(normalized**2, axis=1, keepdims=True)) + [160, 120]
        pixel = np.rint(xy).astype(int)
        matched = (np.abs(xy - pixel).max(axis=1) < 0.001)
        matched &= np.all(records["rgb"] == rgb[pixel[:, 1], pixel[:, 0]], axis=1)
        assert matched.sum() >= frame["retained_points"]
        assert np.all(mask[pixel[matched, 1], pixel[matched, 0]] > 127)
