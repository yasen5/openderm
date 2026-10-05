"""Frame loading, feature extraction, and pairwise matching."""

from __future__ import annotations

import glob
import json
import math
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
@dataclass
class Frame:
    idx: int
    station: int
    row: int
    col: int
    phase: str  # contour-scan y sweep phase ('+y'/'-y'); backs have none
    sidecar: str
    image_path: str
    g: np.ndarray  # gantry (x, y, z) mm
    rx: float  # radians
    standoff: float  # mean of in-range distance sensors, mm
    settled: bool
    sensor_mm: dict = field(default_factory=dict)  # per-sensor reading or None (OOR)
    # filled later
    gray: np.ndarray = field(default=None, repr=False)
    kp: list = field(default=None, repr=False)
    des: np.ndarray = field(default=None, repr=False)
    shape: tuple = None


def load_scan_camera_frames(
    capture_dir: str,
    limit_rows: int = 0,
    row_range: tuple[int, int] | None = None,
    station_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
) -> list[Frame]:
    sidecars = sorted(glob.glob(os.path.join(capture_dir, "*.json")))
    records = []
    for sc in sidecars:
        try:
            with open(sc) as fh:
                txt = fh.read()
            if not txt.strip():
                raise ValueError("empty file")
            d = json.loads(txt)
        except (ValueError, json.JSONDecodeError) as e:
            # Capture glitches can leave zero-byte or truncated sidecars; skip
            # them instead of crashing the run.
            print(f"  ! skipping unreadable sidecar {os.path.basename(sc)} ({e})")
            continue
        if not isinstance(d, dict):
            continue
        img = d.get("image")
        if img and not os.path.isabs(img):
            cand = img if os.path.exists(img) else os.path.join(capture_dir, os.path.basename(img))
        else:
            cand = img
        if not cand or not os.path.exists(cand):
            print(f"  ! skipping {os.path.basename(sc)} (image not found: {img})")
            continue
        records.append((sc, cand, d))
    # a station re-captured later (aborted run restarted) supersedes the
    # earlier shot; sidecars are timestamp-sorted so the last one wins
    by_station: dict = {}
    for sc, cand, d in records:
        key = d.get("station", os.path.basename(sc))
        if key in by_station:
            print(f"  ! dropping superseded station {key}: {os.path.basename(by_station[key][0])}")
        by_station[key] = (sc, cand, d)
    records = [by_station[k] for k in sorted(by_station, key=str)]
    records.sort(key=lambda r: r[0])
    frames = []
    for sc, cand, d in records:
        row = int(d.get("row", 1))
        if limit_rows and row > limit_rows:
            continue
        if row_range and not (row_range[0] <= row <= row_range[1]):
            continue
        st = int(d.get("station", 0))
        if station_range and not (station_range[0] <= st <= station_range[1]):
            continue
        col = int(d.get("col", 0))
        if col_range and not (col_range[0] <= col <= col_range[1]):
            continue
        s1, s2 = d.get("sensor1_mm"), d.get("sensor2_mm")
        in1, in2 = d.get("sensor1_in_range", True), d.get("sensor2_in_range", True)
        vals = [s for s, ok in ((s1, in1), (s2, in2)) if s is not None and ok]
        standoff = float(np.mean(vals)) if vals else float(d.get("target_mm", 110.0))
        frames.append(
            Frame(
                idx=len(frames),
                station=int(d.get("station", len(frames))),
                row=row,
                col=int(d.get("col", 0)),
                phase=str(d.get("phase", "+y")),
                sidecar=sc,
                image_path=cand,
                g=np.array([float(d["x_mm"]), float(d["y_mm"]), float(d["z_mm"])]),
                rx=float(d["rx_rad"]),
                standoff=standoff,
                settled=bool(d.get("settled", True)),
                sensor_mm={
                    "sensor1": s1 if (s1 is not None and in1) else None,
                    "sensor2": s2 if (s2 is not None and in2) else None,
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


def _apply_processing_memory_limit(frac: float = 0.7, budget: int | None = None):
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

    def __init__(self, pt):
        self.pt = pt


def _extract_camera_frame_sift_keypoints(path: str, downscale: int, nfeatures: int, mem_budget: int | None = None):
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
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if downscale != 1:
        g = cv2.resize(g, None, fx=1 / downscale, fy=1 / downscale, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    g = clahe.apply(g)
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.008, edgeThreshold=20)
    kp, des = sift.detectAndCompute(g, None)
    pts = np.float32([k.pt for k in kp])
    return pts, des, g.shape


def extract_camera_frame_sift_keypoints(frames: list[Frame], downscale: int, nfeatures: int) -> None:
    """CLAHE-SIFT for every frame, fanned out over a process pool (identical
    per-frame results to the serial path; SIFT itself is deterministic)."""
    t0 = time.time()
    results = None
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
            one_thread = {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "MALLOC_ARENA_MAX": "2",
            }
            saved_env = {k: os.environ.get(k) for k in one_thread}
            os.environ.update(one_thread)
            try:
                with cf.ProcessPoolExecutor(
                    max_workers=nw, mp_context=mp.get_context("spawn")
                ) as ex:
                    results = list(
                        ex.map(
                            _extract_camera_frame_sift_keypoints,
                            [f.image_path for f in frames],
                            [downscale] * len(frames),
                            [nfeatures] * len(frames),
                            [budget] * len(frames),
                            chunksize=1,
                        )
                    )
            finally:
                for k, v in saved_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        except Exception as e:  # pool unavailable: go serial
            print(f"  ! extract_camera_frame_sift_keypoints pool failed ({e}); extracting serially")
            results = None
    if results is None:
        results = [_extract_camera_frame_sift_keypoints(f.image_path, downscale, nfeatures) for f in frames]
    for f, (pts, des, shape) in zip(frames, results):
        f.gray = None  # unused downstream; skip the RAM
        f.shape = shape
        f.kp = [_KPt(tuple(p)) for p in pts]
        f.des = des
        if f.idx % 20 == 0 or f.idx == len(frames) - 1:
            print(
                f"  frame {f.idx:>3}/{len(frames)}: {shape[1]}x{shape[0]}, "
                f"{len(f.kp)} kp  ({time.time() - t0:.0f}s)"
            )


@dataclass
class Pair:
    i: int
    j: int
    n_good: int
    n_inlier: int
    tx: float  # affine translation i->j, downscaled px
    ty: float
    rot_deg: float
    scale: float
    src: np.ndarray  # inlier pts in frame i (downscaled px)
    dst: np.ndarray  # inlier pts in frame j
    src_kp: np.ndarray  # keypoint indices in frame i (for track building)
    dst_kp: np.ndarray


def _create_feature_matching_flann_index():
    return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))


_GPU_MATCHER = None  # set by main() when --device cuda


def configure_gpu_feature_matcher(m) -> None:
    global _GPU_MATCHER
    _GPU_MATCHER = m


def match_camera_frame_pair_keypoints(
    fi: Frame, fj: Frame, ratio: float, min_inliers: int, prior_xy=None, prior_tol: float = 0.0
) -> Pair | None:
    if fi.des is None or fj.des is None or len(fi.kp) < 2 or len(fj.kp) < 2:
        return None
    if _GPU_MATCHER is not None:
        d1, d2, nn1 = _GPU_MATCHER.knn2(fi.idx, fi.des, fj.idx, fj.des)
        k1i = np.where(d1 < ratio * d2)[0].astype(np.int32)
        k2i = nn1[k1i].astype(np.int32)
    else:
        knn = _create_feature_matching_flann_index().knnMatch(fi.des, fj.des, k=2)
        good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
        k1i = np.int32([m.queryIdx for m in good])
        k2i = np.int32([m.trainIdx for m in good])
    if len(k1i) < min_inliers:
        return None
    p1 = np.float32([fi.kp[q].pt for q in k1i])
    p2 = np.float32([fj.kp[t].pt for t in k2i])
    if prior_xy is not None and prior_tol > 0:
        d = p2 - p1
        keep = np.hypot(d[:, 0] - prior_xy[0], d[:, 1] - prior_xy[1]) < prior_tol
        if keep.sum() >= min_inliers:
            p1, p2, k1i, k2i = p1[keep], p2[keep], k1i[keep], k2i[keep]
    M, inl = cv2.estimateAffinePartial2D(
        p1, p2, method=cv2.RANSAC, ransacReprojThreshold=4, maxIters=5000, confidence=0.999
    )
    if M is None or inl is None:
        return None
    inl = inl.ravel().astype(bool)
    if int(inl.sum()) < min_inliers:
        return None
    return Pair(
        i=fi.idx,
        j=fj.idx,
        n_good=len(p1),
        n_inlier=int(inl.sum()),
        tx=float(M[0, 2]),
        ty=float(M[1, 2]),
        rot_deg=math.degrees(math.atan2(M[1, 0], M[0, 0])),
        scale=float(math.hypot(M[0, 0], M[1, 0])),
        src=p1[inl],
        dst=p2[inl],
        src_kp=k1i[inl],
        dst_kp=k2i[inl],
    )
