from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


cv2 = pytest.importorskip(
    "cv2",
    reason='reconstruction tests require the optional "vision" dependencies',
)
np = pytest.importorskip("numpy")

from processing import register_scan_3d
from processing.registration_export import build_surface_mesh
from processing.registration_features import Frame, Pair, load_scan_camera_frames
from processing.registration_geometry import (
    RigModel,
    build_3d_feature_tracks,
    project_world_points_into_camera,
    compute_feature_track_reprojection_errors,
    triangulate_3d_feature_tracks,
    undistort_image_points_to_normalized_camera,
)
from processing.registration_surface import (
    Surface,
    TexParam,
    fit_surface_heightfield,
    ray_surface_intersect,
)
from processing.registration_texture import write_texture_outputs


def _frame(
    index: int,
    *,
    gantry: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rx: float = 0.0,
    standoff: float = 100.0,
) -> Frame:
    return Frame(
        idx=index,
        station=index + 1,
        row=1,
        col=index + 1,
        phase="+y",
        sidecar="",
        image_path="",
        g=np.asarray(gantry, dtype=float),
        rx=rx,
        standoff=standoff,
        settled=True,
    )


def _camera_model() -> RigModel:
    return RigModel(
        fx=800.0,
        k1=0.0,
        cx=320.0,
        cy=240.0,
        sign=1.0,
        lever=np.zeros(3),
        Rm=np.eye(3),
        dz0=0.0,
    )


def test_load_frames_deduplicates_and_uses_only_in_range_sensors(
    tmp_path: Path,
) -> None:
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"placeholder")

    base = {
        "station": 7,
        "row": 2,
        "col": 3,
        "phase": "descend",
        "image": image.name,
        "x_mm": 10,
        "y_mm": 20,
        "z_mm": 30,
        "rx_rad": 0.25,
        "sensor1_mm": 104,
        "sensor2_mm": 160,
        "sensor1_in_range": True,
        "sensor2_in_range": False,
        "settled": True,
    }
    (tmp_path / "001.json").write_text(json.dumps({**base, "x_mm": 1}))
    (tmp_path / "002.json").write_text(json.dumps(base))
    (tmp_path / "003.json").write_text("{truncated")

    frames = load_scan_camera_frames(str(tmp_path))

    assert len(frames) == 1
    assert frames[0].station == 7
    assert frames[0].g.tolist() == [10.0, 20.0, 30.0]
    assert frames[0].standoff == 104.0
    assert frames[0].sensor_mm == {"sensor1": 104, "sensor2": None}


def test_projection_undistortion_round_trip() -> None:
    model = _camera_model()
    model.k1 = -0.08
    points = np.array(
        [
            [-10.0, -5.0, 100.0],
            [0.0, 0.0, 120.0],
            [8.0, 6.0, 90.0],
        ]
    )

    pixels, depth = project_world_points_into_camera(
        points,
        np.eye(3),
        np.zeros(3),
        model.fx,
        model.k1,
        model.cx,
        model.cy,
    )
    normalized = undistort_image_points_to_normalized_camera(
        pixels,
        model.fx,
        model.k1,
        model.cx,
        model.cy,
    )

    np.testing.assert_allclose(normalized, points[:, :2] / points[:, 2, None])
    np.testing.assert_allclose(depth, points[:, 2])


def test_tracks_triangulate_back_to_synthetic_landmarks() -> None:
    frames = [_frame(0), _frame(1), _frame(2)]
    keypoints = [
        [(100.0, 120.0), (200.0, 220.0)],
        [(101.0, 120.0), (201.0, 220.0)],
        [(102.0, 120.0), (202.0, 220.0)],
    ]
    for frame, points in zip(frames, keypoints):
        frame.kp = [SimpleNamespace(pt=point) for point in points]
    pairs = [
        Pair(
            i=i,
            j=i + 1,
            n_good=2,
            n_inlier=2,
            tx=1.0,
            ty=0.0,
            rot_deg=0.0,
            scale=1.0,
            src=np.asarray(keypoints[i]),
            dst=np.asarray(keypoints[i + 1]),
            src_kp=np.array([0, 1]),
            dst_kp=np.array([0, 1]),
        )
        for i in range(2)
    ]
    obs_frame, _, obs_track, track_count = build_3d_feature_tracks(frames, pairs, 10)
    assert track_count == 2
    for track in range(track_count):
        assert set(obs_frame[obs_track == track]) == {0, 1, 2}

    model = _camera_model()
    landmarks = np.array(
        [
            [-12.0, -8.0, 100.0],
            [4.0, 6.0, 110.0],
            [16.0, -3.0, 95.0],
        ]
    )
    rotations = np.repeat(np.eye(3)[None], 2, axis=0)
    centers = np.array([[-15.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    observations = []
    observation_frames = []
    observation_tracks = []
    for frame_index in range(2):
        pixels, _ = project_world_points_into_camera(
            landmarks,
            rotations[frame_index],
            centers[frame_index],
            model.fx,
            model.k1,
            model.cx,
            model.cy,
        )
        observations.append(pixels)
        observation_frames.extend([frame_index] * len(landmarks))
        observation_tracks.extend(range(len(landmarks)))
    observation_frames = np.asarray(observation_frames, dtype=np.int32)
    observation_tracks = np.asarray(observation_tracks, dtype=np.int32)
    observations = np.concatenate(observations)

    recovered = triangulate_3d_feature_tracks(
        observation_frames,
        observations,
        observation_tracks,
        len(landmarks),
        rotations,
        centers,
        model,
    )
    errors, depths = compute_feature_track_reprojection_errors(
        recovered,
        observation_frames,
        observations,
        observation_tracks,
        rotations,
        centers,
        model,
    )

    np.testing.assert_allclose(recovered, landmarks, atol=1e-5)
    assert np.max(errors) < 1e-5
    assert np.all(depths > 0)


def test_surface_fit_unwrap_and_ray_intersection() -> None:
    coordinates = np.linspace(0.0, 20.0, 5)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    z_grid = 10.0 + 0.1 * x_grid - 0.2 * y_grid
    landmarks = np.stack(
        [x_grid.ravel(), y_grid.ravel(), z_grid.ravel()],
        axis=1,
    )

    surface, rms = fit_surface_heightfield(
        landmarks,
        bounds=(0.0, 20.0, 0.0, 20.0),
        pitch=5.0,
        smooth=0.01,
    )
    assert rms < 0.01
    np.testing.assert_allclose(
        surface.height(np.array([5.0, 15.0]), np.array([5.0, 15.0])),
        np.array([9.5, 8.5]),
        atol=0.02,
    )
    normals = surface.normal(np.array([10.0]), np.array([10.0]))
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)

    texture_parameterization = TexParam(surface)
    x = np.array([4.0, 10.0, 16.0])
    y = np.array([3.0, 11.0, 17.0])
    u, v = texture_parameterization.to_uv(x, y)
    round_trip_x, round_trip_y = texture_parameterization.to_xy(u, v)
    np.testing.assert_allclose(round_trip_x, x, atol=0.05)
    np.testing.assert_allclose(round_trip_y, y, atol=0.05)

    hit = surface.height(np.array([10.0]), np.array([10.0]))[0]
    intersection = ray_surface_intersect(
        np.array([10.0, 10.0, 0.0]),
        np.array([[0.0, 0.0, 1.0]]),
        surface,
        t0=10.0,
    )
    np.testing.assert_allclose(intersection[0], [10.0, 10.0, hit], atol=1e-6)


def test_mesh_and_texture_outputs_are_complete(tmp_path: Path) -> None:
    xs = ys = np.linspace(0.0, 10.0, 3)
    surface = Surface(xs, ys, np.zeros((3, 3)))
    parameterization = TexParam(surface)
    coverage = np.ones((11, 11), dtype=np.float32)
    positions, normals, uv, faces = build_surface_mesh(
        surface,
        parameterization,
        tex_bounds=(0.0, 0.0, 10.0, 10.0),
        wacc=coverage,
        ppmm=1.0,
        mesh_pitch=5.0,
        up_sign=1.0,
    )
    assert positions.shape == (9, 3)
    assert faces.shape == (8, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.all((uv >= 0.0) & (uv <= 1.0))

    frame = _frame(0)
    state = SimpleNamespace(
        acc=np.full((3, 4, 3), 100.0, dtype=np.float32),
        wacc=np.ones((3, 4), dtype=np.float32),
        blend_mode="soft",
        hf_best=None,
        frames=[frame],
        foot_uv={frame.idx: np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]])},
        umin=0.0,
        vmin=0.0,
        ppmm=1.0,
        bounds=(0.0, 0.0, 4.0, 3.0),
    )
    texture, weights, bounds = write_texture_outputs(state, str(tmp_path))
    assert texture.shape == (3, 4, 3)
    np.testing.assert_array_equal(weights, state.wacc)
    assert bounds == state.bounds
    for filename in ("texture.jpg", "coverage.png", "texture_index.jpg"):
        assert (tmp_path / filename).is_file()
    coverage_image = cv2.imread(
        str(tmp_path / "coverage.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    np.testing.assert_array_equal(coverage_image, np.full((3, 4), 255))


def test_registration_main_connects_all_pipeline_stages() -> None:
    args = SimpleNamespace(capture_dir="captures")
    problem = SimpleNamespace(frames=["frame"])
    solution = SimpleNamespace(texture="texture")

    with (
        mock.patch.object(register_scan_3d, "parse_scan_cli_arguments", return_value=args),
        mock.patch.object(
            register_scan_3d,
            "build_scan_reconstruction_problem",
            return_value=problem,
        ) as prepare,
        mock.patch.object(
            register_scan_3d,
            "reconstruct_surface_from_camera_frames",
            return_value=solution,
        ) as solve,
        mock.patch.object(
            register_scan_3d,
            "export_scan_reconstruction_artifacts",
        ) as write,
    ):
        register_scan_3d._main(mem_cap=1234)

    prepare.assert_called_once_with(args, 1234)
    solve.assert_called_once_with(args, problem)
    write.assert_called_once_with(args, problem, solution)
