from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2", reason='mask tests require the optional "vision"/"sim" dependencies')
np = pytest.importorskip("numpy")
pytest.importorskip("scipy.spatial.transform")

from freehand_scene import random_freehand_cameras
from test_sim_capture import _model_in_gauge

from processing.registration_features import _extract_camera_frame_sift_keypoints, attach_frame_masks
from sim_capture.alignment import canonicalize
from sim_capture.masks import MaskError, check_masks, install_masks, skin_point_filter
from sim_capture.model import SparseModel
from sim_capture.tools.sapiens_seg_preview import CLASS_NAMES, SKIN_CLASSES
from sim_capture.tools.skin_masks import clean_skin_mask

ARM = CLASS_NAMES.index("Left_Lower_Arm")


def test_skin_classes_are_body_parts_only() -> None:
    names = {CLASS_NAMES[i] for i in SKIN_CLASSES}
    assert {"Left_Lower_Arm", "Right_Hand", "Face_Neck", "Torso"} <= names
    assert not names & {"Background", "Apparel", "Upper_Clothing", "Hair", "Left_Shoe", "Eyeglass"}


def test_clean_skin_mask_fills_speckle_drops_islands_and_trims_the_edge() -> None:
    labels = np.zeros((300, 400), np.int64)
    labels[100:250, 20:380] = ARM
    labels[150:152:, 100:300:8] = 0  # patch-grid speckle inside the arm
    labels[170:176, 200:206] = CLASS_NAMES.index("Hair")  # a mislabelled blotch inside
    labels[10:20, 10:20] = ARM  # floor-coloured island far from the arm

    mask = clean_skin_mask(labels, erode_fraction=0.01)

    assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 255}
    assert (mask[150:152, 100:300] == 255).all(), "speckle should be filled"
    assert (mask[170:176, 200:206] == 255).all(), "enclosed mislabelled blotch should be filled"
    assert (mask[10:20, 10:20] == 0).all(), "small island should be dropped"
    assert mask[175, 200] == 255 and mask[50, 200] == 0
    assert mask[100, 200] == 0 and mask[104, 200] == 255, "the boundary is eroded by ~4 px"


def _two_view_model() -> tuple[SparseModel, np.ndarray]:
    points = np.arange(12, dtype=float).reshape(4, 3)
    return (
        SparseModel(
            image_names=["a.jpg", "b.jpg"],
            rotations_cam_from_world=np.stack([np.eye(3)] * 2),
            translations_cam_from_world=np.zeros((2, 3)),
            points=points,
            point_errors_px=np.full(4, 0.3),
            point_track_lengths=np.full(4, 2, dtype=np.int64),
            point_colors=np.zeros((4, 3), np.uint8),
            visible_points=[np.arange(4), np.arange(4)],
            fx=100.0,
            cx=50.0,
            cy=40.0,
            k1=0.0,
            width=100,
            height=80,
            mean_reprojection_error_px=0.3,
            # points 0,1 sit at x=20 (left half), points 2,3 at x=80 (right half)
            visible_xy=[np.array([[20, 10], [20, 60], [80, 10], [80, 60]], float)] * 2,
        ),
        np.array([True, True, False, False]),
    )


def _write_half_masks(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((80, 100), np.uint8)
    mask[:, :50] = 255  # left half is skin
    for name in names:
        cv2.imwrite(str(directory / f"{Path(name).stem}.png"), mask)


def test_skin_point_filter_keeps_points_that_land_in_the_mask(tmp_path: Path) -> None:
    model, expected = _two_view_model()
    _write_half_masks(tmp_path, model.image_names)
    assert (skin_point_filter(model, tmp_path) == expected).all()


def test_masks_must_cover_every_image_and_copy_over(tmp_path: Path) -> None:
    model, _ = _two_view_model()
    _write_half_masks(tmp_path / "m", ["a.jpg"])
    with pytest.raises(MaskError, match="b.jpg"):
        check_masks(tmp_path / "m", model.image_names)
    _write_half_masks(tmp_path / "m", model.image_names)
    check_masks(tmp_path / "m", model.image_names)
    install_masks(tmp_path / "m", tmp_path / "out", model.image_names)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a.png", "b.png"]


def test_canonicalize_takes_scale_from_the_filtered_points_only() -> None:
    cameras = random_freehand_cameras(8, seed=3)
    model, _ = _model_in_gauge(cameras, gauge_seed=1, scale=0.37)
    keep = np.zeros(len(model.points), bool)
    keep[::3] = True

    depths = [
        np.median((model.points[keep] @ rotation.T + translation)[:, 2])
        for rotation, translation in zip(model.rotations_cam_from_world, model.translations_cam_from_world)
    ]
    canonical = canonicalize(model, 100.0, point_filter=keep)
    assert canonical.scale_mm_per_unit == pytest.approx(100.0 / np.median(depths), rel=1e-6)
    assert np.median(canonical.depths_mm) == pytest.approx(100.0, rel=1e-6)
    assert not canonical.surface_point_mask[~keep].any()

    everything = canonicalize(model, 100.0, point_filter=np.ones(len(model.points), bool))
    plain = canonicalize(model, 100.0)
    assert everything.scale_mm_per_unit == pytest.approx(plain.scale_mm_per_unit)


def test_canonicalize_refuses_a_mask_with_almost_no_points() -> None:
    cameras = random_freehand_cameras(6, seed=3)
    model, _ = _model_in_gauge(cameras, gauge_seed=1, scale=1.0)
    few = np.zeros(len(model.points), bool)
    few[:5] = True
    with pytest.raises(ValueError, match="masked surface"):
        canonicalize(model, 100.0, point_filter=few)


def test_attach_frame_masks_requires_a_mask_for_every_frame(tmp_path: Path) -> None:
    frames = [SimpleNamespace(image_path=str(tmp_path / f"f{i}.jpg"), mask_path=None) for i in range(3)]
    masks = tmp_path / "masks"
    masks.mkdir()
    for frame in frames[:2]:
        cv2.imwrite(str(masks / (Path(frame.image_path).stem + ".png")), np.full((4, 4), 255, np.uint8))
    with pytest.raises(SystemExit, match="no mask for 1 frame"):
        attach_frame_masks(frames, str(masks))  # type: ignore[arg-type]
    cv2.imwrite(str(masks / "f2.png"), np.full((4, 4), 255, np.uint8))
    token = attach_frame_masks(frames, str(masks))  # type: ignore[arg-type]
    assert all(frame.mask_path and frame.mask_path.endswith(".png") for frame in frames) and token


def test_sift_keypoints_stay_inside_the_mask(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.integers(0, 255, (480, 640), dtype=np.uint8), (0, 0), 2.0)
    path = tmp_path / "t.jpg"
    cv2.imwrite(str(path), texture)
    mask = np.zeros((480, 640), np.uint8)
    mask[:, :320] = 255
    mask_path = tmp_path / "t.png"
    cv2.imwrite(str(mask_path), mask)

    everywhere, _, _ = _extract_camera_frame_sift_keypoints(str(path), 1, 500)
    masked, _, _ = _extract_camera_frame_sift_keypoints(str(path), 1, 500, mask_path=str(mask_path))
    assert everywhere[:, 0].max() > 400
    assert len(masked) > 50 and masked[:, 0].max() < 321
