"""Capture loading, feature matching, and rig-prior construction."""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from .. import registration_features
from ..registration_features import (
    Frame,
    Pair,
    _KPt,
    extract_camera_frame_sift_keypoints,
    load_scan_camera_frames,
    match_camera_frame_pair_keypoints,
    configure_gpu_feature_matcher,
)
from ..registration_geometry import (
    RigModel,
    find_overlapping_frame_pairs,
    predict_frame_pair_translation,
    estimate_initial_rig_camera_model,
)
from .parser import ScanCliArguments


class _CacheKey(TypedDict):
    version: int
    downscale: int
    nfeatures: int
    ratio: float
    min_inliers: int
    overlap_frac: float
    max_partners: int
    n: int
    src: str
    stations: tuple[int, ...]


class _PairCache(TypedDict):
    key: _CacheKey
    cpairs: list[Pair]
    xpairs: list[Pair]
    pairs: list[Pair]
    shape: tuple[int, int]
    kps: list[list[tuple[float, float]]]


class _RigModelJson(TypedDict):
    fx_fullres_px: float
    k1: float
    rx_sign: float
    lever_mm: list[float]
    Rm: list[list[float]]
    dz0_mm: float
    base_R: list[list[float]]
    base_t: list[float]


class _RigDocument(TypedDict):
    rig_model: _RigModelJson


class _ClearableGpuMatcher(Protocol):
    def clear(self) -> None: ...


@dataclass
class ReconstructionProblem:
    out_dir: str
    frames: list[Frame]
    pairs: list[Pair]
    rig_model: RigModel
    prefit_rms: float
    R0: NDArray[np.float64]
    C0: NDArray[np.float64]


def build_scan_reconstruction_problem(
    args: ScanCliArguments, mem_cap: int | None = None
) -> ReconstructionProblem:
    """Load captures, match features, and establish the rig prior."""
    out_dir = args.out or os.path.join(args.capture_dir, "registration3d")
    os.makedirs(out_dir, exist_ok=True)
    if mem_cap:
        print(
            f"      RAM guard: data segment capped at {mem_cap / 1e9:.1f} GB "
            "(MemoryError here beats the kernel OOM killer there)"
        )
    row_range: tuple[int, int] | None = None
    station_range: tuple[int, int] | None = None
    col_range: tuple[int, int] | None = None
    if args.rows:
        lo, _, hi = args.rows.partition(":")
        row_range = (int(lo), int(hi or lo))
    if args.stations:
        lo, _, hi = args.stations.partition(":")
        station_range = (int(lo), int(hi or lo))
    if args.cols:
        lo, _, hi = args.cols.partition(":")
        col_range = (int(lo), int(hi or lo))

    print(f"[1/9] loading frames from {args.capture_dir}")
    frames = load_scan_camera_frames(args.capture_dir, args.limit_rows, row_range, station_range, col_range)
    if len(frames) < 2:
        print("need >=2 frames; aborting")
        sys.exit(1)
    print(
        f"      {len(frames)} frames, rows "
        f"{min(frame.row for frame in frames)}..{max(frame.row for frame in frames)}"
    )

    # Features and matches are computed in the sensor frame with EXIF
    # orientation ignored.
    cache_key: _CacheKey = {
        "version": 3,
        "downscale": args.downscale,
        "nfeatures": args.nfeatures,
        "ratio": args.ratio,
        "min_inliers": args.min_inliers,
        "overlap_frac": args.overlap_frac,
        "max_partners": args.max_partners,
        "n": len(frames),
        "src": "jpeg",
        "stations": tuple(frame.station for frame in frames),
    }
    cache_path = os.path.join(out_dir, "cache_pairs.pkl")
    cached: _PairCache | None = None
    if not args.no_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                blob = cast(_PairCache, pickle.load(fh))
            if blob["key"] == cache_key:
                cached = blob
                print("      (using cached matches)")
        except Exception:
            cached = None

    if cached is None:
        print(f"[2/9] extracting CLAHE-SIFT features (downscale={args.downscale})")
        extract_camera_frame_sift_keypoints(frames, args.downscale, args.nfeatures)
        if args.device != "cpu":
            from processing.gpu_match import GpuMatcher, gpu_matcher_available as _gm_ok

            if _gm_ok():
                configure_gpu_feature_matcher(GpuMatcher())
                print("      descriptor matching: exact 2-NN on cuda (torch)")
        frame_shape = frames[0].shape
        assert frame_shape is not None
        h0, w0 = frame_shape
    else:
        h0, w0 = cached["shape"]
        for frame in frames:
            frame.shape = (h0, w0)
    diag = math.hypot(w0, h0)

    if cached is None:
        print("[3/9] matching consecutive pairs")
        cpairs: list[Pair] = []
        for item_index in range(len(frames) - 1):
            point = match_camera_frame_pair_keypoints(frames[item_index], frames[item_index + 1], args.ratio, args.min_inliers)
            if point is None:
                print(f"  ! consecutive pair {item_index}->{item_index + 1} FAILED")
                continue
            cpairs.append(point)
        print(f"      {len(cpairs)}/{len(frames) - 1} consecutive pairs matched")
        # same-column partners in the next row: these excite the y direction,
        # without which the rig-model pre-fit is rank-deficient (in-row pairs
        # only sample dx/dz/drx). Serpentine scan -> nearest gantry xy match.
        print("      matching cross-row pairs for the pre-fit")
        xpairs: list[Pair] = []
        byrow: dict[int, list[Frame]] = {}
        for frame in frames:
            byrow.setdefault(frame.row, []).append(frame)
        for camera_rotation in sorted(byrow):
            if camera_rotation + 1 not in byrow:
                continue
            for frame in byrow[camera_rotation][::2]:
                g_ = min(byrow[camera_rotation + 1], key=lambda query: abs(query.g[0] - frame.g[0]))
                point = match_camera_frame_pair_keypoints(frame, g_, args.ratio, args.min_inliers)
                if point is not None:
                    xpairs.append(point)
        print(f"      {len(xpairs)} cross-row pairs matched")
    else:
        cpairs = cached["cpairs"]
        xpairs = cached["xpairs"]

    if args.rig_from:
        print(f"[4/9] rig model loaded from {args.rig_from} (pre-fit skipped)")
        with open(args.rig_from) as rig_file:
            rig_document = cast(_RigDocument, json.load(rig_file))
        rj = rig_document["rig_model"]
        rig_model = RigModel(
            fx=float(rj["fx_fullres_px"]) / args.downscale,
            k1=float(rj["k1"]),
            cx=w0 / 2.0,
            cy=h0 / 2.0,
            sign=float(rj["rx_sign"]),
            lever=np.array(rj["lever_mm"]),
            Rm=np.array(rj["Rm"]),
            dz0=float(rj["dz0_mm"]),
            downscale=args.downscale,
            base_R=np.array(rj["base_R"]),
            base_t=np.array(rj["base_t"]),
        )
        prefit_rms = float("nan")
    else:
        print("[4/9] rig-model pre-fit (fx seed, lever arm, mount, rx sign)")
        # rough fx init: median consecutive-pair shift per 10mm gantry step at Z~110
        ts: NDArray[np.float64] = np.array([[point.tx, point.ty] for point in cpairs])
        fx0 = float(np.median(np.linalg.norm(ts, axis=1)) / 10.0 * 110.0)
        rig_model, prefit_rms = estimate_initial_rig_camera_model(frames, cpairs + xpairs, w0, h0, args.downscale, fx0)
    if args.fx_full:
        rig_model.fx = args.fx_full / args.downscale
        print(
            f"      fx LOCKED to {args.fx_full:.0f} full px ({rig_model.fx:.1f} ds-px); "
            f"lever/dz0 will re-anchor during BA"
        )
    R0, C0 = rig_model.poses(frames)
    axis = R0[len(frames) // 2][:, 2]
    print(
        f"      fx={rig_model.fx:.0f}ds-px ({rig_model.fx * args.downscale:.0f} full px), "
        f"px/mm≈{rig_model.fx / 110:.1f} (ds), lever=({rig_model.lever[0]:.1f},{rig_model.lever[1]:.1f},"
        f"{rig_model.lever[2]:.1f})mm |{np.linalg.norm(rig_model.lever):.1f}mm|, dz0={rig_model.dz0:.1f}mm"
    )
    print(f"      mid-scan optical axis (world): ({axis[0]:+.3f},{axis[1]:+.3f},{axis[2]:+.3f})")

    if cached is None:
        print(
            f"[5/9] matching overlapping pairs (overlap>={args.overlap_frac}, "
            f"top-{args.max_partners}/frame)"
        )
        keep, overlaps = find_overlapping_frame_pairs(
            rig_model, frames, R0, C0, args.overlap_frac, args.max_partners
        )
        existing: set[tuple[int, int]] = {
            (point.i, point.j) for point in cpairs
        } | {(point.i, point.j) for point in xpairs}
        to_match = sorted(ij for ij in keep if ij not in existing)
        print(f"      {len(overlaps)} candidate pairs -> matching {len(to_match)} extra")
        pairs: list[Pair] = list(cpairs) + list(xpairs)
        added = 0
        rejected = 0
        t0 = time.time()
        for n_, (index, neighbor_index) in enumerate(to_match):
            tpred = predict_frame_pair_translation(rig_model, frames, R0, C0, index, neighbor_index)
            point = match_camera_frame_pair_keypoints(
                frames[index],
                frames[neighbor_index],
                args.ratio,
                args.min_inliers,
                prior_xy=tuple(tpred),
                prior_tol=0.25 * diag,
            )
            if point is None:
                continue
            if math.hypot(point.tx - tpred[0], point.ty - tpred[1]) > 0.2 * diag:
                rejected += 1
                continue
            pairs.append(point)
            added += 1
            if n_ % 100 == 99:
                print(f"        {n_ + 1}/{len(to_match)} ({time.time() - t0:.0f}s)")
        print(f"      added {added} ({rejected} rejected by prior gate); total {len(pairs)} pairs")
        if not args.no_cache:
            with open(cache_path, "wb") as fh:
                pickle.dump(
                    dict(
                        key=cache_key,
                        cpairs=cpairs,
                        xpairs=xpairs,
                        pairs=pairs,
                        shape=(h0, w0),
                        kps=[[kp.pt for kp in (frame.kp or [])] for frame in frames],
                    ),
                    fh,
                )
    else:
        pairs = cached["pairs"]
        # rebuild minimal kp lists for track building
        for frame, kps in zip(frames, cached["kps"]):
            frame.kp = [_KPt((float(point[0]), float(point[1]))) for point in kps]

    if registration_features._GPU_MATCHER is not None:
        cast(_ClearableGpuMatcher, registration_features._GPU_MATCHER).clear()
        # free descriptor VRAM before render
        configure_gpu_feature_matcher(None)
    return ReconstructionProblem(
        out_dir=out_dir,
        frames=frames,
        pairs=pairs,
        rig_model=rig_model,
        prefit_rms=prefit_rms,
        R0=R0,
        C0=C0,
    )
