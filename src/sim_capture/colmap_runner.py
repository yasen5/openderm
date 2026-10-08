"""Run COLMAP (via pycolmap) on a folder of freehand images -> SparseModel.

COLMAP does the feature matching, geometric verification and incremental
mapping; this module only drives it and reads the result back as plain arrays.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .model import IntArray, SparseModel

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
EXHAUSTIVE_LIMIT = 200  # exhaustive matching is O(n^2); beyond this assume an ordered sequence
SEQUENTIAL_OVERLAP = 20
DEFAULT_THREADS = 4  # SIFT memory scales with threads x image size; 16 threads OOMed 4K-class frames

Matcher = Literal["auto", "exhaustive", "sequential"]
Device = Literal["auto", "cpu"]


class _ColmapTrack(Protocol):
    def length(self) -> int: ...


class _ColmapPoint3D(Protocol):
    xyz: Sequence[float]
    error: float
    track: _ColmapTrack
    color: Sequence[int]


class _ColmapPoint2D(Protocol):
    xy: Sequence[float]
    point3D_id: int

    def has_point3D(self) -> bool: ...


class _ColmapImage(Protocol):
    name: str
    has_pose: bool
    points2D: Sequence[_ColmapPoint2D]

    def cam_from_world(self) -> _ColmapPose: ...


class _ColmapCamera(Protocol):
    model: object
    params: Sequence[float]
    width: int
    height: int


class _ColmapRotation(Protocol):
    def matrix(self) -> NDArray[np.float64]: ...


class _ColmapPose(Protocol):
    rotation: _ColmapRotation
    translation: Sequence[float]


class _ColmapReconstruction(Protocol):
    cameras: Mapping[int, _ColmapCamera]
    images: Mapping[int, _ColmapImage]
    points3D: Mapping[int, _ColmapPoint3D]

    def compute_mean_reprojection_error(self) -> float: ...


class ReconstructionError(RuntimeError):
    """COLMAP could not produce a usable reconstruction."""


def natural_key(name: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name.lower())]


def list_images(image_dir: Path) -> list[str]:
    """Top-level image file names in natural sort order (frame_2 before frame_10)."""
    names = [p.name for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(names, key=natural_key)


def reconstruct(
    image_dir: Path,
    workspace: Path,
    image_names: list[str],
    matcher: Matcher = "auto",
    device: Device = "auto",
    verbose: bool = False,
    num_threads: int = DEFAULT_THREADS,
) -> SparseModel:
    """Single-camera SIMPLE_RADIAL reconstruction of ``image_names`` in ``image_dir``.

    SIMPLE_RADIAL (f, cx, cy, k) with the principal point held at the image centre
    is exactly the intrinsics model processing uses: u = f * x * (1 + k * r^2) + c.
    """
    import pycolmap  # optional dependency: the `sim` extra

    pycolmap.logging.minloglevel = 0 if verbose else 3
    pycolmap.logging.logtostderr = verbose
    colmap_device = pycolmap.Device.cpu if device == "cpu" else pycolmap.Device.auto

    workspace.mkdir(parents=True, exist_ok=True)
    database = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    database.unlink(missing_ok=True)

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = num_threads  # SIFT needs ~1 GB per thread at 1440x2560
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = num_threads
    pycolmap.extract_features(
        str(database),
        str(image_dir),
        image_names=image_names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
        device=colmap_device,
    )

    use_exhaustive = matcher == "exhaustive" or (matcher == "auto" and len(image_names) <= EXHAUSTIVE_LIMIT)
    if use_exhaustive:
        pycolmap.match_exhaustive(str(database), matching_options=matching, device=colmap_device)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = SEQUENTIAL_OVERLAP
        pairing.loop_detection = False  # needs a vocabulary tree that is not bundled
        pycolmap.match_sequential(
            str(database), matching_options=matching, pairing_options=pairing, device=colmap_device
        )

    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.num_threads = num_threads
    # pycolmap exposes extension objects without useful static stubs; narrow
    # its mapper result to the protocol used by the reader below.
    reconstructions = cast(
        Mapping[int, _ColmapReconstruction],
        pycolmap.incremental_mapping(str(database), str(image_dir), str(sparse_dir), mapping),
    )
    if not reconstructions:
        raise ReconstructionError(
            "COLMAP could not build any model: not enough overlapping, textured views "
            "(need >=3 images sharing a surface with ~60% overlap between neighbours)"
        )

    def registered(reconstruction: _ColmapReconstruction) -> int:
        return sum(1 for image in reconstruction.images.values() if image.has_pose)

    best_id = max(reconstructions, key=lambda key: registered(reconstructions[key]))
    best = reconstructions[best_id]
    extra_sizes = sorted((registered(r) for key, r in reconstructions.items() if key != best_id), reverse=True)
    return _read_model(best, image_names, extra_sizes)


def _read_model(reconstruction: _ColmapReconstruction, requested: list[str], extra_sizes: list[int]) -> SparseModel:
    import pycolmap

    cameras = list(reconstruction.cameras.values())
    if len(cameras) != 1:
        raise ReconstructionError(f"expected one shared camera, COLMAP produced {len(cameras)}")
    camera = cameras[0]
    if camera.model != pycolmap.CameraModelId.SIMPLE_RADIAL:
        raise ReconstructionError(f"unexpected camera model {camera.model}")
    focal, center_x, center_y, k1 = (float(value) for value in camera.params)

    point_ids = sorted(reconstruction.points3D)
    row_of_point = {point_id: row for row, point_id in enumerate(point_ids)}
    points = np.array([reconstruction.points3D[i].xyz for i in point_ids], dtype=np.float64).reshape(-1, 3)
    errors = np.array([reconstruction.points3D[i].error for i in point_ids], dtype=np.float64)
    track_lengths = np.array([reconstruction.points3D[i].track.length() for i in point_ids], dtype=np.int64)
    colors = np.array([reconstruction.points3D[i].color for i in point_ids], dtype=np.uint8).reshape(-1, 3)

    posed = sorted(
        (image for image in reconstruction.images.values() if image.has_pose),
        key=lambda image: natural_key(image.name),
    )
    rotations = np.zeros((len(posed), 3, 3))
    translations = np.zeros((len(posed), 3))
    visible: list[IntArray] = []
    visible_xy: list[NDArray[np.float64]] = []
    for index, image in enumerate(posed):
        pose = image.cam_from_world()
        rotations[index] = pose.rotation.matrix()
        translations[index] = pose.translation
        observed = [p for p in image.points2D if p.has_point3D()]
        visible.append(np.asarray([row_of_point[p.point3D_id] for p in observed], dtype=np.int64))
        visible_xy.append(np.asarray([p.xy for p in observed], dtype=np.float64).reshape(-1, 2))
    names = [image.name for image in posed]
    registered_names = set(names)
    return SparseModel(
        image_names=names,
        rotations_cam_from_world=rotations,
        translations_cam_from_world=translations,
        points=points,
        point_errors_px=errors,
        point_track_lengths=track_lengths,
        point_colors=colors,
        visible_points=visible,
        fx=focal,
        cx=center_x,
        cy=center_y,
        k1=k1,
        width=int(camera.width),
        height=int(camera.height),
        mean_reprojection_error_px=float(reconstruction.compute_mean_reprojection_error()),
        unregistered_images=[name for name in requested if name not in registered_names],
        extra_model_sizes=extra_sizes,
        visible_xy=visible_xy,
    )
