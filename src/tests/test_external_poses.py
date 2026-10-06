from __future__ import annotations

import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip(
    "cv2",
    reason='external-pose tests require the optional "vision" dependencies',
)
np = pytest.importorskip("numpy")

from freehand_scene import Camera, look_at, make_scene, render_view

from processing.external_poses import load_external_poses
from processing.registration_features import (
    Frame,
    _extract_camera_frame_sift_keypoints,
    _KPt,
    match_camera_frame_pair_keypoints,
)
from processing.registration_geometry import (
    RigModel,
    project_world_points_into_camera,
    verify_pair_with_known_poses,
)
from processing.registration_surface import ray_surface_intersect


def _frame(index: int, image_path: str = "", standoff: float = 100.0) -> Frame:
    return Frame(
        idx=index,
        station=index + 1,
        row=0,
        col=index + 1,
        phase="+y",
        sidecar="",
        image_path=image_path,
        g=np.zeros(3),
        rx=0.0,
        standoff=standoff,
        settled=True,
    )


def _rig(rotations: np.ndarray | None = None, centers: np.ndarray | None = None) -> RigModel:
    return RigModel(
        fx=900.0,
        k1=0.0,
        cx=320.0,
        cy=240.0,
        sign=1.0,
        lever=np.zeros(3),
        Rm=np.eye(3),
        dz0=0.0,
        fixed_poses=None if rotations is None or centers is None else (rotations, centers),
    )


def test_fixed_poses_bypass_gantry_kinematics() -> None:
    rotations = np.stack([look_at(np.array([5.0 * i, 0.0, 100.0]), np.zeros(3), 10.0 * i) for i in range(3)])
    centers = np.array([[5.0 * i, 0.0, 100.0] for i in range(3)])
    frames = [_frame(i) for i in range(3)]
    frames[1].rx = 1.2  # would swing a gantry-modelled camera; must be ignored
    frames[1].gauge = np.array([50.0, 60.0, 70.0])

    poses_rotations, poses_centers = _rig(rotations, centers).poses(frames)

    np.testing.assert_allclose(poses_rotations, rotations)
    np.testing.assert_allclose(poses_centers, centers)
    # callers mutate the returned arrays (bundle adjustment): they must be copies
    poses_centers[0] += 1.0
    assert not np.allclose(centers[0], poses_centers[0])


def _write_poses(path: Path, names: list[str], rotation_override: dict[str, list[list[float]]] | None = None) -> None:
    frames = []
    for index, name in enumerate(names):
        rotation = look_at(np.array([4.0 * index, 0.0, 100.0]), np.zeros(3), 7.0 * index).tolist()
        frames.append(
            {
                "image": name,
                "R_cam2world": (rotation_override or {}).get(name, rotation),
                "C_mm": [4.0 * index, 0.0, 100.0],
                "depth_mm": 100.0 + index,
            }
        )
    path.write_text(
        json.dumps(
            {"source": "test", "image_size": [640, 480], "fx_full": 900.0, "k1": -0.02, "frames": frames}
        )
    )


def test_load_external_poses_aligns_by_basename_and_validates(tmp_path: Path) -> None:
    poses_path = tmp_path / "poses.json"
    _write_poses(poses_path, ["b.jpg", "a.jpg"])
    # frame order differs from file order, and image_path carries a directory
    frames = [_frame(0, str(tmp_path / "a.jpg")), _frame(1, str(tmp_path / "b.jpg"))]

    loaded = load_external_poses(str(poses_path), frames)

    assert loaded.fx_full == 900.0 and loaded.k1 == -0.02 and loaded.image_size == (640, 480)
    np.testing.assert_allclose(loaded.centers[0], [4.0, 0.0, 100.0])  # a.jpg is file index 1
    np.testing.assert_allclose(loaded.centers[1], [0.0, 0.0, 100.0])
    np.testing.assert_allclose(loaded.depths, [101.0, 100.0])

    with pytest.raises(ValueError, match="no pose for 1 frame"):
        load_external_poses(str(poses_path), frames + [_frame(2, str(tmp_path / "missing.jpg"))])

    _write_poses(poses_path, ["a.jpg", "b.jpg"], {"a.jpg": [[1, 1, 0], [0, 1, 0], [0, 0, 1]]})
    with pytest.raises(ValueError, match="not a 3x3 rotation"):
        load_external_poses(str(poses_path), frames)


def _frame_with_features(image: np.ndarray, index: int, tmp_path: Path) -> Frame:
    path = tmp_path / f"view{index}.png"
    cv2.imwrite(str(path), image)
    points, descriptors, shape = _extract_camera_frame_sift_keypoints(str(path), 1, 4000)
    frame = _frame(index, str(path))
    frame.kp = [_KPt((float(x), float(y))) for x, y in points]
    frame.des = descriptors
    frame.shape = shape
    return frame


def test_pose_verifier_survives_perspective_change_that_defeats_similarity_ransac(tmp_path: Path) -> None:
    scene = make_scene(seed=0)
    camera_a = Camera(look_at(np.array([-12.0, -4.0, 100.0]), np.zeros(3), 0.0), np.array([-12.0, -4.0, 100.0]))
    camera_b = Camera(look_at(np.array([34.0, 10.0, 92.0]), np.zeros(3), 24.0), np.array([34.0, 10.0, 92.0]))
    frames = [
        _frame_with_features(render_view(scene, camera_a), 0, tmp_path),
        _frame_with_features(render_view(scene, camera_b), 1, tmp_path),
    ]
    rotations = np.stack([camera_a.R, camera_b.R])
    centers = np.stack([camera_a.C, camera_b.C])
    rig = _rig(rotations, centers)

    verified = verify_pair_with_known_poses(rig, frames, rotations, centers, 0, 1, ratio=0.8, min_inliers=25)
    similarity = match_camera_frame_pair_keypoints(frames[0], frames[1], 0.8, 25)

    assert verified is not None
    assert verified.n_inlier >= 150

    # ground truth: lift each verified source point onto the surface, project into view B
    xu = (verified.src.astype(np.float64) - [320.0, 240.0]) / 900.0
    rays = np.concatenate([xu, np.ones((len(xu), 1))], 1) @ camera_a.R.T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    hits = ray_surface_intersect(camera_a.C, rays, scene.surface, 100.0)
    expected_b, _ = project_world_points_into_camera(hits, camera_b.R, camera_b.C, 900.0, 0.0, 320.0, 240.0)
    error_px = np.linalg.norm(expected_b - verified.dst, axis=1)
    precision_2px = float(np.mean(error_px < 2.0))
    assert precision_2px > 0.9, f"only {precision_2px:.0%} of verified matches are true correspondences"

    # the point of the new verifier: the similarity gate keeps far fewer of them here
    assert similarity is None or similarity.n_inlier < 0.5 * verified.n_inlier


def test_pose_verifier_rejects_wrong_matches_and_pure_rotation(tmp_path: Path) -> None:
    scene = make_scene(seed=0)
    camera_a = Camera(look_at(np.array([0.0, 0.0, 100.0]), np.zeros(3)), np.array([0.0, 0.0, 100.0]))
    camera_b = Camera(look_at(np.array([20.0, 5.0, 100.0]), np.zeros(3), 10.0), np.array([20.0, 5.0, 100.0]))
    frames = [
        _frame_with_features(render_view(scene, camera_a), 0, tmp_path),
        _frame_with_features(render_view(scene, camera_b), 1, tmp_path),
    ]
    rig = _rig(np.stack([camera_a.R, camera_b.R]), np.stack([camera_a.C, camera_b.C]))

    # claim the cameras are the SAME place (pure rotation): nothing to verify against
    colocated_centers = np.stack([camera_a.C, camera_a.C])
    rig_colocated = _rig(np.stack([camera_a.R, camera_b.R]), colocated_centers)
    assert verify_pair_with_known_poses(rig_colocated, frames, rig_colocated.poses(frames)[0], colocated_centers, 0, 1, 0.8, 25) is None

    # claim wildly wrong poses: true matches violate the epipolar geometry and get rejected
    wrong_centers = np.stack([camera_a.C, camera_b.C + np.array([0.0, 40.0, -30.0])])
    wrong_rotations = np.stack([camera_a.R, look_at(wrong_centers[1], np.array([30.0, -20.0, 0.0]), -35.0)])
    rig_wrong = _rig(wrong_rotations, wrong_centers)
    right = verify_pair_with_known_poses(rig, frames, rig.poses(frames)[0], rig.poses(frames)[1], 0, 1, 0.8, 25)
    wrong = verify_pair_with_known_poses(rig_wrong, frames, wrong_rotations, wrong_centers, 0, 1, 0.8, 25)
    assert right is not None
    assert wrong is None or wrong.n_inlier < 0.2 * right.n_inlier
