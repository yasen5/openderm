"""Frame loading, feature extraction, and pairwise matching."""

from __future__ import annotations

import glob
import json
import math
import os
import time
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Protocol, TypedDict, cast

import cv2
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


class CaptureMetadata(TypedDict, total=False):
    """Known fields in a capture sidecar; extra sidecar keys are ignored."""

    image: str
    station: int
    row: int
    col: int
    phase: str
    x_mm: float
    y_mm: float
    z_mm: float
    rx_rad: float
    sensor1_mm: float | None
    sensor2_mm: float | None
    sensor1_in_range: bool
    sensor2_in_range: bool
    target_mm: float
    settled: bool


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
@dataclass(init=False)
class Frame:
    idx: int
    station: int
    row: int
    col: int
    phase: str  # contour-scan y sweep phase ('+y'/'-y'); backs have none
    sidecar: str
    image_path: str
    gauge: NDArray[np.float64]  # gantry (x, y, z) mm
    rx: float  # radians
    standoff: float  # mean of in-range distance sensors, mm
    settled: bool
    sensor_mm: dict[str, float | None] = field(default_factory=dict)  # None means out of range
    # filled later
    gray: NDArray[np.uint8] | None = field(default=None, repr=False)
    kp: list[_KPt] | None = field(default=None, repr=False)
    des: FloatArray | None = field(default=None, repr=False)
    shape: tuple[int, ...] | None = None

    def __init__(self, *positional_values: object, **frame_values: object) -> None:
        # ``g`` was the original capture-side field name. Accept it when
        # reading older callers while keeping the stored field descriptive.
        if "g" in frame_values:
            if "gauge" in frame_values:
                raise TypeError("Frame received both 'g' and 'gauge'")
            frame_values["gauge"] = frame_values.pop("g")
        frame_fields = fields(type(self))
        if len(positional_values) > len(frame_fields):
            raise TypeError("Frame received too many positional arguments")
        for frame_field, field_value in zip(frame_fields, positional_values):
            if frame_field.name in frame_values:
                raise TypeError(f"Frame received multiple values for {frame_field.name!r}")
            frame_values[frame_field.name] = field_value
        for frame_field in frame_fields:
            if frame_field.name in frame_values:
                field_value = frame_values.pop(frame_field.name)
            elif frame_field.default is not MISSING:
                field_value = frame_field.default
            elif frame_field.default_factory is not MISSING:
                field_value = frame_field.default_factory()
            else:
                raise TypeError(f"Frame is missing required field {frame_field.name!r}")
            setattr(self, frame_field.name, field_value)
        if frame_values:
            unexpected_field = next(iter(frame_values))
            raise TypeError(f"Frame got an unexpected keyword argument {unexpected_field!r}")

    def __getattr__(self, field_name: str) -> Any:
        if field_name == "g":
            return self.gauge
        raise AttributeError(field_name)


def load_scan_camera_frames(
    capture_dir: str,
    limit_rows: int = 0,
    row_range: tuple[int, int] | None = None,
    station_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
) -> list[Frame]:
    sidecars = sorted(glob.glob(os.path.join(capture_dir, "*.json")))
    capture_records: list[tuple[str, str, CaptureMetadata]] = []
    for sidecar_path in sidecars:
        try:
            with open(sidecar_path) as fh:
                sidecar_text = fh.read()
            if not sidecar_text.strip():
                raise ValueError("empty file")
            frame_metadata = json.loads(sidecar_text)
        except (ValueError, json.JSONDecodeError) as caught_exception:
            # Capture glitches can leave zero-byte or truncated sidecars; skip
            # them instead of crashing the run.
            print(f"  ! skipping unreadable sidecar {os.path.basename(sidecar_path)} ({caught_exception})")
            continue
        if not isinstance(frame_metadata, dict) or not all(isinstance(key, str) for key in frame_metadata):
            continue
        metadata = cast(CaptureMetadata, frame_metadata)
        image_reference = metadata.get("image")
        if image_reference and not os.path.isabs(image_reference):
            image_path = image_reference if os.path.exists(image_reference) else os.path.join(capture_dir, os.path.basename(image_reference))
        else:
            image_path = image_reference
        if not image_path or not os.path.exists(image_path):
            print(f"  ! skipping {os.path.basename(sidecar_path)} (image not found: {image_reference})")
            continue
        capture_records.append((sidecar_path, image_path, metadata))
    # a station re-captured later (aborted run restarted) supersedes the
    # earlier shot; sidecars are timestamp-sorted so the last one wins
    by_station: dict[int | str, tuple[str, str, CaptureMetadata]] = {}
    for sidecar_path, image_path, frame_metadata in capture_records:
        station_key = frame_metadata.get("station", os.path.basename(sidecar_path))
        if station_key in by_station:
            print(f"  ! dropping superseded station {station_key}: {os.path.basename(by_station[station_key][0])}")
        by_station[station_key] = (sidecar_path, image_path, frame_metadata)
    capture_records = [by_station[item_index] for item_index in sorted(by_station, key=str)]
    capture_records.sort(key=lambda capture_record: capture_record[0])
    frames: list[Frame] = []
    for sidecar_path, image_path, frame_metadata in capture_records:
        row_number = int(frame_metadata.get("row", 1))
        if limit_rows and row_number > limit_rows:
            continue
        if row_range and not (row_range[0] <= row_number <= row_range[1]):
            continue
        station_number = int(frame_metadata.get("station", 0))
        if station_range and not (station_range[0] <= station_number <= station_range[1]):
            continue
        column_index = int(frame_metadata.get("col", 0))
        if col_range and not (col_range[0] <= column_index <= col_range[1]):
            continue
        sensor1_mm, sensor2_mm = frame_metadata.get("sensor1_mm"), frame_metadata.get("sensor2_mm")
        sensor1_in_range = frame_metadata.get("sensor1_in_range", True)
        sensor2_in_range = frame_metadata.get("sensor2_in_range", True)
        sensor_readings: list[float] = [
            reading
            for reading, in_range in (
                (sensor1_mm, sensor1_in_range), (sensor2_mm, sensor2_in_range)
            )
            if reading is not None and in_range
        ]
        standoff = float(np.mean(sensor_readings)) if sensor_readings else float(frame_metadata.get("target_mm", 110.0))
        frames.append(
            Frame(
                idx=len(frames),
                station=int(frame_metadata.get("station", len(frames))),
                row=row_number,
                col=int(frame_metadata.get("col", 0)),
                phase=str(frame_metadata.get("phase", "+y")),
                sidecar=sidecar_path,
                image_path=image_path,
                g=np.array([
                    float(cast(float, frame_metadata.get("x_mm"))),
                    float(cast(float, frame_metadata.get("y_mm"))),
                    float(cast(float, frame_metadata.get("z_mm"))),
                ]),
                rx=float(cast(float, frame_metadata.get("rx_rad"))),
                standoff=standoff,
                settled=bool(frame_metadata.get("settled", True)),
                sensor_mm={
                    "sensor1": sensor1_mm if (sensor1_mm is not None and sensor1_in_range) else None,
                    "sensor2": sensor2_mm if (sensor2_mm is not None and sensor2_in_range) else None,
                },
            )
        )
    return frames


# ----------------------------------------------------------------------------
# feature extraction + pairwise matching (CLAHE-SIFT, same recipe as 2D script)
# ----------------------------------------------------------------------------
def _get_available_system_memory_bytes() -> float:
    """MemAvailable from /proc/meminfo; +inf when unreadable (non-Linux)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable"):
                    return float(line.split()[1]) * 1024.0
    except OSError:
        pass
    return float("inf")


def _apply_processing_memory_limit(frac: float = 0.7, budget: int | None = None) -> int | None:
    """Cap THIS process's data segment (heap + anonymous mmaps) so a runaway
    allocation dies here with a MemoryError instead of driving the kernel OOM
    killer into other users' processes (a full-res extract_camera_frame_sift_keypoints pool once swapped
    the box and took out every tmux session on it). Returns the cap or None."""
    try:
        import resource

        if budget is None:
            avail = _get_available_system_memory_bytes()
            if not math.isfinite(avail):
                return None
            vmdata = 0.0
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmData"):
                        vmdata = float(line.split()[1]) * 1024.0
                        break
            budget = int(vmdata + frac * avail)
        resource.setrlimit(resource.RLIMIT_DATA, (budget, budget))
        return budget
    except Exception:
        return None


class _KPt:
    """Minimal keypoint: downstream only ever reads .pt (matching, track
    building, the pair cache all key on coordinates)."""

    __slots__ = ("pt",)

    def __init__(self, pt: tuple[float, float]) -> None:
        self.pt = pt


def _extract_camera_frame_sift_keypoints(
    path: str, downscale: int, nfeatures: int, mem_budget: int | None = None
) -> tuple[FloatArray, FloatArray | None, tuple[int, ...]]:
    """One frame's CLAHE-SIFT; module-level so a process pool can run it.
    Returns (pts float32 (N,2), des float32 (N,128), shape)."""
    cv2.setNumThreads(1)  # the pool is the parallelism
    if mem_budget:
        # A worker that outgrows its share dies alone with MemoryError so the
        # caller can fall back to serial extraction.
        _apply_processing_memory_limit(budget=mem_budget)
    # IGNORE_ORIENTATION: the camera mount never rotates, but the Canon's
    # orientation sensor flips the EXIF auto-rotate tag mid-scan as rx
    # tilts past ~45deg -- honouring it feeds the pipeline a mix of
    # portrait/landscape frames, which would rotate cross-band matches and
    # cause the prior gate to reject valid links.
    im = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if im is None:
        raise RuntimeError(f"could not read {path}")
    gauge = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if downscale != 1:
        gauge = cv2.resize(gauge, None, fx=1 / downscale, fy=1 / downscale, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    gauge = clahe.apply(gauge)
    cv_api: Any = cv2
    sift = cv_api.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.008, edgeThreshold=20)
    kp, raw_descriptors = sift.detectAndCompute(gauge, None)
    pts = cast(FloatArray, np.asarray([item_index.pt for item_index in kp], dtype=np.float32))
    descriptors = cast(FloatArray | None, raw_descriptors)
    return pts, descriptors, gauge.shape


def extract_camera_frame_sift_keypoints(frames: list[Frame], downscale: int, nfeatures: int) -> None:
    """CLAHE-SIFT for every frame, fanned out over a process pool (identical
    per-frame results to the serial path; SIFT itself is deterministic)."""
    t0 = time.time()
    results: list[tuple[FloatArray, FloatArray | None, tuple[int, ...]]] | None = None
    if len(frames) > 3:
        try:
            import concurrent.futures as cf
            import multiprocessing as mp

            # cap workers by RAM, not just cores: 22 concurrent full-res
            # extractions once swapped a 128GB box (observed: pool 3x SLOWER
            # than serial). Estimate per-worker peak from one image's pixel
            # count and keep the pool inside half the available RAM. Peak =
            #   spawn interpreter + numpy/cv2 imports        ~0.4 GB
            # + full-res decode BEFORE downscaling
            #   (BGR + gray + libjpeg temps)                 ~8 B/full-res px
            # + SIFT, which 2x-upsamples to a float32 base and allocates the
            #   whole Gaussian (6/octave) + DoG (5/octave) pyramids up front:
            #   11 planes x 4/3 octave series x 4x px x 4 B  ~240 B/ds px
            # A flat bytes-per-pixel estimate can starve workers because SIFT
            # allocates its full Gaussian and DoG pyramids up front.
            im0 = cv2.imread(frames[0].image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if im0 is None:
                raise RuntimeError(f"could not read {frames[0].image_path}")
            px_full = float(im0.shape[0] * im0.shape[1])
            px_ds = px_full / max(1, downscale * downscale)
            per_worker = 0.4e9 + px_full * 8.0 + px_ds * 240.0
            avail = 0.5e9 * 100  # fallback: assume plenty
            try:
                with open("/proc/meminfo") as fh:
                    for line in fh:
                        if line.startswith("MemAvailable"):
                            avail = float(line.split()[1]) * 1024.0
                            break
            except OSError:
                pass
            nw = min(
                len(frames),
                max(1, (os.cpu_count() or 8) - 2),
                max(1, int(0.5 * avail / per_worker)),
            )
            budget = int(per_worker * 2)
            # spawn workers re-import numpy/cv2, whose BLAS/OpenMP runtimes
            # each start a core-count thread pool, and glibc grows a 64MB
            # malloc arena per contending thread: on a 48-core box that is
            # ~3GB of anon mmaps PER WORKER, charged against RLIMIT_DATA
            # before a single pixel is read (workers failed 97MB decode
            # allocs with GBs "in use"). The pool is the parallelism --
            # worker-internal threads are pure overhead, so pin children to
            # one thread / two arenas via env the spawn re-import reads.
            one_thread: dict[str, str] = {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "MALLOC_ARENA_MAX": "2",
            }
            saved_env: dict[str, str | None] = {item_index: os.environ.get(item_index) for item_index in one_thread}
            os.environ.update(one_thread)
            try:
                with cf.ProcessPoolExecutor(
                    max_workers=nw, mp_context=mp.get_context("spawn")
                ) as ex:
                    results = list(
                        ex.map(
                            _extract_camera_frame_sift_keypoints,
                            [frame.image_path for frame in frames],
                            [downscale] * len(frames),
                            [nfeatures] * len(frames),
                            [budget] * len(frames),
                            chunksize=1,
                        )
                    )
            finally:
                for item_index, texture_v in saved_env.items():
                    if texture_v is None:
                        os.environ.pop(item_index, None)
                    else:
                        os.environ[item_index] = texture_v
        except Exception as caught_exception:  # pool unavailable: go serial
            print(f"  ! extract_camera_frame_sift_keypoints pool failed ({caught_exception}); extracting serially")
            results = None
    if results is None:
        results = [_extract_camera_frame_sift_keypoints(frame.image_path, downscale, nfeatures) for frame in frames]
    for frame, (pts, des, shape) in zip(frames, results):
        frame.gray = None  # unused downstream; skip the RAM
        frame.shape = shape
        frame.kp = [_KPt((float(point[0]), float(point[1]))) for point in pts]
        frame.des = des
        if frame.idx % 20 == 0 or frame.idx == len(frames) - 1:
            print(
                f"  frame {frame.idx:>3}/{len(frames)}: {shape[1]}x{shape[0]}, "
                f"{len(frame.kp)} kp  ({time.time() - t0:.0f}s)"
            )


@dataclass(init=False)
class Pair:
    index: int
    neighbor_index: int
    n_good: int
    n_inlier: int
    tx: float  # affine translation i->j, downscaled px
    ty: float
    rot_deg: float
    scale: float
    src: FloatArray  # inlier pts in frame i (downscaled px)
    dst: FloatArray  # inlier pts in frame j
    src_kp: NDArray[np.int32]  # keypoint indices in frame i (for track building)
    dst_kp: NDArray[np.int32]

    def __init__(self, *positional_values: object, **pair_values: object) -> None:
        # Keep the original pair endpoint keywords available to callers.
        for legacy_name, descriptive_name in (("i", "index"), ("j", "neighbor_index")):
            if legacy_name in pair_values:
                if descriptive_name in pair_values:
                    raise TypeError(f"Pair received both {legacy_name!r} and {descriptive_name!r}")
                pair_values[descriptive_name] = pair_values.pop(legacy_name)
        pair_fields = fields(type(self))
        if len(positional_values) > len(pair_fields):
            raise TypeError("Pair received too many positional arguments")
        for pair_field, field_value in zip(pair_fields, positional_values):
            if pair_field.name in pair_values:
                raise TypeError(f"Pair received multiple values for {pair_field.name!r}")
            pair_values[pair_field.name] = field_value
        for pair_field in pair_fields:
            if pair_field.name not in pair_values:
                raise TypeError(f"Pair is missing required field {pair_field.name!r}")
            setattr(self, pair_field.name, pair_values.pop(pair_field.name))
        if pair_values:
            unexpected_field = next(iter(pair_values))
            raise TypeError(f"Pair got an unexpected keyword argument {unexpected_field!r}")

    def __getattr__(self, field_name: str) -> Any:
        if field_name == "i":
            return self.index
        if field_name == "j":
            return self.neighbor_index
        raise AttributeError(field_name)


def _create_feature_matching_flann_index() -> cv2.FlannBasedMatcher:
    return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))


class GpuFeatureMatcher(Protocol):
    def knn2(
        self, key_q: int, des_q: FloatArray, key_t: int, des_t: FloatArray
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]: ...


_GPU_MATCHER: GpuFeatureMatcher | None = None  # set by main() when --device cuda


def configure_gpu_feature_matcher(mask: GpuFeatureMatcher | None) -> None:
    global _GPU_MATCHER
    _GPU_MATCHER = mask


def match_camera_frame_pair_keypoints(
    source_frame: Frame,
    target_frame: Frame,
    ratio: float,
    min_inliers: int,
    prior_xy: tuple[float, float] | None = None,
    prior_tol: float = 0.0,
) -> Pair | None:
    if (
        source_frame.des is None
        or target_frame.des is None
        or source_frame.kp is None
        or target_frame.kp is None
        or len(source_frame.kp) < 2
        or len(target_frame.kp) < 2
    ):
        return None
    if _GPU_MATCHER is not None:
        nearest_distances, second_nearest_distances, nearest_neighbor_indices = _GPU_MATCHER.knn2(
            source_frame.idx, source_frame.des, target_frame.idx, target_frame.des
        )
        source_keypoint_indices = cast(
            NDArray[np.int32], np.where(nearest_distances < ratio * second_nearest_distances)[0].astype(np.int32)
        )
        target_keypoint_indices = cast(
            NDArray[np.int32], nearest_neighbor_indices[source_keypoint_indices].astype(np.int32)
        )
    else:
        neighbor_matches = _create_feature_matching_flann_index().knnMatch(source_frame.des, target_frame.des, k=2)
        good_matches = [
            first_match
            for first_match, second_match in neighbor_matches
            if first_match.distance < ratio * second_match.distance
        ]
        source_keypoint_indices = cast(
            NDArray[np.int32], np.asarray([match.queryIdx for match in good_matches], dtype=np.int32)
        )
        target_keypoint_indices = cast(
            NDArray[np.int32], np.asarray([match.trainIdx for match in good_matches], dtype=np.int32)
        )
    if len(source_keypoint_indices) < min_inliers:
        return None
    source_image_points = cast(
        FloatArray,
        np.asarray([source_frame.kp[keypoint_index].pt for keypoint_index in source_keypoint_indices], dtype=np.float32),
    )
    target_image_points = cast(
        FloatArray,
        np.asarray([target_frame.kp[keypoint_index].pt for keypoint_index in target_keypoint_indices], dtype=np.float32),
    )
    if prior_xy is not None and prior_tol > 0:
        observed_translations = target_image_points - source_image_points
        within_translation_prior = (
            np.hypot(
                observed_translations[:, 0] - prior_xy[0],
                observed_translations[:, 1] - prior_xy[1],
            )
            < prior_tol
        )
        if within_translation_prior.sum() >= min_inliers:
            source_image_points = source_image_points[within_translation_prior]
            target_image_points = target_image_points[within_translation_prior]
            source_keypoint_indices = source_keypoint_indices[within_translation_prior]
            target_keypoint_indices = target_keypoint_indices[within_translation_prior]
    cv_api: Any = cv2
    affine_transform, inlier_mask = cv_api.estimateAffinePartial2D(
        source_image_points, target_image_points, method=cv2.RANSAC, ransacReprojThreshold=4, maxIters=5000, confidence=0.999
    )
    if affine_transform is None or inlier_mask is None:
        return None
    inlier_mask = inlier_mask.ravel().astype(bool)
    if int(inlier_mask.sum()) < min_inliers:
        return None
    return Pair(
        i=source_frame.idx,
        j=target_frame.idx,
        n_good=len(source_image_points),
        n_inlier=int(inlier_mask.sum()),
        tx=float(affine_transform[0, 2]),
        ty=float(affine_transform[1, 2]),
        rot_deg=math.degrees(math.atan2(affine_transform[1, 0], affine_transform[0, 0])),
        scale=float(math.hypot(affine_transform[0, 0], affine_transform[1, 0])),
        src=source_image_points[inlier_mask],
        dst=target_image_points[inlier_mask],
        src_kp=source_keypoint_indices[inlier_mask],
        dst_kp=target_keypoint_indices[inlier_mask],
    )
