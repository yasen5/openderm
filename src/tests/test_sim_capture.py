from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

cv2 = pytest.importorskip(
    "cv2",
    reason='sim-capture tests require the optional "vision"/"sim" dependencies',
)
np = pytest.importorskip("numpy")
scipy_rotation = pytest.importorskip("scipy.spatial.transform")

from freehand_scene import Camera, look_at, make_scene, random_freehand_cameras, render_view

from processing.external_poses import load_external_poses
from processing.registration_features import load_scan_camera_frames
from sim_capture.alignment import canonicalize
from sim_capture.export import OutputDirectoryError, prepare_output_dir, write_capture_folder
from sim_capture.model import SparseModel
from sim_capture.process import registration_command

Rotation = scipy_rotation.Rotation


def _model_in_gauge(
    cameras: list[Camera], gauge_seed: int, scale: float, point_count: int = 600
) -> tuple[SparseModel, np.ndarray]:
    """The synthetic scene as COLMAP would report it, in an arbitrary scale/gauge.

    The surface points are the same for every ``gauge_seed``: only the (arbitrary)
    COLMAP scale, rotation and origin change, as between two COLMAP runs."""
    points_rng = np.random.default_rng(0)
    rng = np.random.default_rng(gauge_seed)
    scene = make_scene(seed=0)
    xy = np.column_stack([points_rng.uniform(-45, 45, point_count), points_rng.uniform(-35, 35, point_count)])
    truth_points = np.column_stack([xy, scene.surface.height(xy[:, 0], xy[:, 1])])

    gauge_rotation = Rotation.random(random_state=gauge_seed).as_matrix()
    gauge_translation = rng.normal(0, 50, 3)

    def to_gauge(points: np.ndarray) -> np.ndarray:
        return scale * points @ gauge_rotation.T + gauge_translation

    points = to_gauge(truth_points)
    rotations_cw, translations_cw = [], []
    for camera in cameras:
        center = to_gauge(camera.C[None, :])[0]
        rotation_cw = (gauge_rotation @ camera.R).T  # COLMAP: X_cam = R_cw X_world + t
        rotations_cw.append(rotation_cw)
        translations_cw.append(-rotation_cw @ center)
    everything = np.arange(point_count)
    model = SparseModel(
        image_names=[f"v{i:03d}.jpg" for i in range(len(cameras))],
        rotations_cam_from_world=np.stack(rotations_cw),
        translations_cam_from_world=np.stack(translations_cw),
        points=points,
        point_errors_px=np.full(point_count, 0.3),
        point_track_lengths=np.full(point_count, 5, dtype=np.int64),
        point_colors=np.full((point_count, 3), 128, dtype=np.uint8),
        visible_points=[everything for _ in cameras],
        fx=900.0,
        cx=320.0,
        cy=240.0,
        k1=-0.01,
        width=640,
        height=480,
        mean_reprojection_error_px=0.3,
    )
    return model, truth_points


def test_canonicalize_scales_to_standoff_and_puts_cameras_on_positive_z() -> None:
    cameras = random_freehand_cameras(16, seed=3)
    model, _ = _model_in_gauge(cameras, gauge_seed=11, scale=0.37)

    canonical = canonicalize(model, standoff_mm=105.0)

    assert np.median(canonical.depths_mm) == pytest.approx(105.0)
    rotations = canonical.rotations_cam2world
    np.testing.assert_allclose(np.einsum("nij,nkj->nik", rotations, rotations), np.broadcast_to(np.eye(3), rotations.shape), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-9)
    assert (canonical.centers_mm[:, 2] > 0).all()  # +z faces the cameras
    assert canonical.warnings == []
    # every camera looks roughly down the surface normal (-z) in this capture
    assert (rotations[:, 2, 2] < -0.8).all()
    assert canonical.diagnostics["cameras_below_surface"] == 0


def test_canonicalize_is_independent_of_colmap_gauge() -> None:
    cameras = random_freehand_cameras(16, seed=3)
    first, _ = _model_in_gauge(cameras, gauge_seed=11, scale=0.37)
    second, _ = _model_in_gauge(cameras, gauge_seed=29, scale=4.2)

    a = canonicalize(first, 105.0)
    b = canonicalize(second, 105.0)

    np.testing.assert_allclose(a.centers_mm, b.centers_mm, atol=1e-6)
    np.testing.assert_allclose(a.rotations_cam2world, b.rotations_cam2world, atol=1e-8)
    np.testing.assert_allclose(a.depths_mm, b.depths_mm, atol=1e-6)


def test_canonicalize_preserves_shape_up_to_the_standoff_scale() -> None:
    cameras = random_freehand_cameras(12, seed=5)
    model, _ = _model_in_gauge(cameras, gauge_seed=2, scale=1.0)
    canonical = canonicalize(model, standoff_mm=105.0)

    truth_centers = np.stack([camera.C for camera in cameras])
    true_distances = np.linalg.norm(truth_centers[:, None] - truth_centers[None, :], axis=-1)
    recovered = np.linalg.norm(canonical.centers_mm[:, None] - canonical.centers_mm[None, :], axis=-1)
    off_diagonal = ~np.eye(len(cameras), dtype=bool)
    ratios = recovered[off_diagonal] / true_distances[off_diagonal]
    assert np.std(ratios) < 1e-9  # a similarity: one global scale, no distortion
    assert ratios[0] == pytest.approx(canonical.scale_mm_per_unit / 1.0)


def test_canonicalize_warns_when_the_capture_wraps_around_the_subject() -> None:
    cameras = random_freehand_cameras(12, seed=3)
    behind = [
        Camera(look_at(np.array([x, 0.0, -100.0]), np.zeros(3)), np.array([x, 0.0, -100.0]))
        for x in (-30.0, 0.0, 30.0, 15.0, -15.0, 45.0)
    ]
    model, _ = _model_in_gauge(cameras + behind, gauge_seed=7, scale=1.0)

    canonical = canonicalize(model, 105.0)

    assert any("behind" in warning for warning in canonical.warnings)


def _write_images(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        cv2.imwrite(str(directory / name), np.full((48, 64, 3), 90, np.uint8))


def test_exported_folder_loads_in_processing_and_poses_round_trip(tmp_path: Path) -> None:
    cameras = random_freehand_cameras(6, seed=1)
    model, _ = _model_in_gauge(cameras, gauge_seed=4, scale=1.0)
    model.width, model.height = 64, 48
    canonical = canonicalize(model, 105.0)
    source, out = tmp_path / "photos", tmp_path / "scan"
    _write_images(source, model.image_names)
    prepare_output_dir(out, force=False)

    report = write_capture_folder(out, source, model, canonical, 105.0)

    frames = load_scan_camera_frames(str(out))
    assert [frame.station for frame in frames] == [1, 2, 3, 4, 5, 6]
    assert all(os.path.isabs(frame.image_path) and os.path.exists(frame.image_path) for frame in frames)
    poses = load_external_poses(str(out / "sim" / "poses.json"), frames)
    np.testing.assert_allclose(poses.centers, canonical.centers_mm, atol=1e-9)
    np.testing.assert_allclose(poses.rotations, canonical.rotations_cam2world, atol=1e-12)
    np.testing.assert_allclose([frame.standoff for frame in frames], canonical.depths_mm)
    assert poses.fx_full == 900.0 and poses.image_size == (64, 48)
    assert report["images_registered"] == 6 and (out / "sim" / "sparse.ply").exists()
    # nothing but sidecars matches processing's top-level *.json glob
    assert sorted(path.name for path in out.glob("*.json")) == sorted(f"v{i:03d}.json" for i in range(6))


def test_prepare_output_dir_protects_foreign_json_and_replaces_only_its_own(tmp_path: Path) -> None:
    out = tmp_path / "scan"
    out.mkdir()
    (out / "gantry-shot.json").write_text(json.dumps({"station": 1, "image": "x.jpg"}))
    with pytest.raises(OutputDirectoryError, match="non-sim-capture JSON"):
        prepare_output_dir(out, force=True)  # even --force must not delete a real scan's sidecars
    assert (out / "gantry-shot.json").exists()

    (out / "gantry-shot.json").unlink()
    (out / "old.json").write_text(json.dumps({"sim": True, "image": "old.jpg"}))
    with pytest.raises(OutputDirectoryError, match="--force"):
        prepare_output_dir(out, force=False)
    prepare_output_dir(out, force=True)
    assert not (out / "old.json").exists()


def test_registration_command_scales_mm_parameters_with_the_standoff() -> None:
    near = registration_command(Path("scan"), fx_full=39237.0, image_width=6000, standoff_mm=110.0)
    far = registration_command(Path("scan"), fx_full=3000.0, image_width=4000, standoff_mm=330.0)

    def value(command: list[str], flag: str) -> str:
        return command[command.index(flag) + 1]

    assert value(near, "--poses-from") == str(Path("scan") / "sim" / "poses.json")
    assert value(near, "--reject-pose-mm") == "0" and value(near, "--reject-rot-deg") == "0"
    assert "--group-by-row" not in near and "--rig-from" not in near
    assert float(value(far, "--surface-pitch")) == pytest.approx(3 * float(value(near, "--surface-pitch")), rel=1e-2)
    assert float(value(far, "--sigma-t")) == pytest.approx(3 * float(value(near, "--sigma-t")), rel=1e-2)
    assert value(far, "--downscale") == "3"  # 4000 px wide -> ~1280 px analysis resolution
    assert registration_command(Path("scan"), 3000.0, 4000, 330.0, quality="full").count("--downscale") == 1
    assert value(registration_command(Path("scan"), 3000.0, 4000, 330.0, quality="full"), "--downscale") == "1"


# ---------------------------------------------------------------------------
# End to end: images -> COLMAP -> sim-capture folder -> processing --poses-from
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_sim_capture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    pytest.importorskip("pycolmap", reason='end-to-end sim-capture needs the optional "sim" extra')
    from sim_capture.cli import main

    root = tmp_path_factory.mktemp("sim")
    scene = make_scene(seed=0, k1=-0.03, size=(480, 360))
    scene.fx = 675.0  # same ~40 deg field of view at the smaller size
    cameras = random_freehand_cameras(14, seed=1)
    images = root / "photos"
    images.mkdir()
    for index, camera in enumerate(cameras):
        cv2.imwrite(str(images / f"view_{index:03d}.jpg"), render_view(scene, camera), [cv2.IMWRITE_JPEG_QUALITY, 95])
    out = root / "scan"
    exit_code = main([str(images), "--out", str(out), "--standoff-mm", "105", "--device", "cpu"])
    assert exit_code == 0
    return {"out": out, "cameras": cameras, "scene": scene}


def _similarity_fit(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama: scale, R, t with target ~ scale * R @ source + t."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_centered, target_centered = source - source_mean, target - target_mean
    u, singular, vt = np.linalg.svd(target_centered.T @ source_centered / len(source))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ flip @ vt
    scale = float((singular * np.diag(flip)).sum() / source_centered.var(0).sum())
    return scale, rotation, target_mean - scale * rotation @ source_mean


def test_sim_capture_recovers_freehand_poses_and_intrinsics(synthetic_sim_capture: dict[str, object]) -> None:
    out = synthetic_sim_capture["out"]
    cameras = synthetic_sim_capture["cameras"]
    assert isinstance(out, Path) and isinstance(cameras, list)
    poses = json.loads((out / "sim" / "poses.json").read_text())
    report = json.loads((out / "sim" / "report.json").read_text())

    assert report["images_registered"] == 14 and report["images_unregistered"] == []
    assert report["mean_reprojection_error_px"] < 0.5
    assert poses["fx_full"] == pytest.approx(675.0, rel=0.01)
    assert poses["k1"] == pytest.approx(-0.03, abs=0.01)

    by_name = {frame["image"]: frame for frame in poses["frames"]}
    recovered = np.array([by_name[f"view_{i:03d}.jpg"]["C_mm"] for i in range(14)])
    truth = np.array([camera.C for camera in cameras])
    scale, rotation, translation = _similarity_fit(recovered, truth)
    residual = scale * recovered @ rotation.T + translation - truth
    assert np.linalg.norm(residual, axis=1).max() < 0.5  # mm, on a ~100 mm scene
    angle_errors = [
        np.degrees(np.linalg.norm(Rotation.from_matrix((rotation @ np.array(by_name[f"view_{i:03d}.jpg"]["R_cam2world"])).T @ cameras[i].R).as_rotvec()))
        for i in range(14)
    ]
    assert max(angle_errors) < 0.5
    # metric scale is exactly as good as the standoff guess (true median depth ~102 vs the 105 given)
    assert scale == pytest.approx(1.0, abs=0.06)


def test_processing_runs_on_sim_capture_with_poses_from(synthetic_sim_capture: dict[str, object]) -> None:
    out = synthetic_sim_capture["out"]
    assert isinstance(out, Path)
    poses = json.loads((out / "sim" / "poses.json").read_text())
    command = registration_command(out, poses["fx_full"], 480, 105.0, device="cpu")
    command += ["--nfeatures", "3000", "--rounds", "6", "--no-cache"]
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}

    result = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)

    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    placements = json.loads((out / "registration3d" / "placements3d.json").read_text())
    assert placements["pose_source"] == "external"
    assert len(placements["frames"]) == 14
    assert placements["ba"]["rms_px"] < 1.0
    assert placements["surface"]["landmark_rms_mm"] < 1.0
    for artifact in ("texture.jpg", "coverage.png", "surface_mesh.obj", "landmarks.ply"):
        assert (out / "registration3d" / artifact).exists()
    # poses barely move: COLMAP's poses were already consistent with the images
    from processing.tex_anchor import load_gauge

    gauge = load_gauge(str(out / "registration3d"))
    assert gauge.ppmm > 0


def test_frames_from_video_keeps_the_sharpest_frame_per_bin(tmp_path: Path) -> None:
    from sim_capture.tools.frames_from_video import extract, select_sharpest_per_bin

    assert select_sharpest_per_bin([1.0, 5.0, 2.0, 9.0, 3.0, 4.0], 3) == [1, 3, 5]
    assert select_sharpest_per_bin([1.0, 2.0], 5) == [0, 1]

    scene = make_scene(seed=0, size=(320, 240))
    scene.fx = 450.0
    cameras = random_freehand_cameras(24, seed=2)
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (320, 240))
    assert writer.isOpened()
    blurred = set(range(1, 24, 2))  # every other frame is motion-blurred
    for index, camera in enumerate(cameras):
        frame = render_view(scene, camera)
        writer.write(cv2.GaussianBlur(frame, (0, 0), 4.0) if index in blurred else frame)
    writer.release()

    written = extract(video, tmp_path / "frames", fps=8.0, max_frames=12)

    assert len(written) == 12
    from sim_capture.tools.frames_from_video import sharpness

    kept = [sharpness(cv2.imread(str(path))) for path in written]
    blurry = sharpness(cv2.GaussianBlur(render_view(scene, cameras[0]), (0, 0), 4.0))
    assert min(kept) > 3 * blurry  # every kept frame is a sharp one, none of the blurred ones
