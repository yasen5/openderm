"""Capture loading, feature matching, and rig-prior construction."""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np

from .. import registration_features
from ..registration_features import extract, load_frames, match_pair, set_gpu_matcher
from ..registration_geometry import (
    RigModel,
    find_overlap_pairs,
    predicted_pair_translation,
    prefit_rig_model,
)


def _prepare_problem(args, mem_cap=None):
    """Load captures, match features, and establish the rig prior."""
    out_dir = args.out or os.path.join(args.capture_dir, "registration3d")
    os.makedirs(out_dir, exist_ok=True)
    if mem_cap:
        print(
            f"      RAM guard: data segment capped at {mem_cap / 1e9:.1f} GB "
            "(MemoryError here beats the kernel OOM killer there)"
        )
    row_range = station_range = col_range = None
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
    frames = load_frames(args.capture_dir, args.limit_rows, row_range, station_range, col_range)
    if len(frames) < 2:
        print("need >=2 frames; aborting")
        sys.exit(1)
    print(
        f"      {len(frames)} frames, rows "
        f"{min(f.row for f in frames)}..{max(f.row for f in frames)}"
    )

    # Features and matches are computed in the sensor frame with EXIF
    # orientation ignored.
    cache_key = dict(
        version=3,
        downscale=args.downscale,
        nfeatures=args.nfeatures,
        ratio=args.ratio,
        min_inliers=args.min_inliers,
        overlap_frac=args.overlap_frac,
        max_partners=args.max_partners,
        n=len(frames),
        src="jpeg",
        stations=tuple(f.station for f in frames),
    )
    cache_path = os.path.join(out_dir, "cache_pairs.pkl")
    cached = None
    if not args.no_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                blob = pickle.load(fh)
            if blob["key"] == cache_key:
                cached = blob
                print("      (using cached matches)")
        except Exception:
            cached = None

    if cached is None:
        print(f"[2/9] extracting CLAHE-SIFT features (downscale={args.downscale})")
        extract(frames, args.downscale, args.nfeatures)
        if args.device != "cpu":
            from skinmap.gpu_match import GpuMatcher, available as _gm_ok

            if _gm_ok():
                set_gpu_matcher(GpuMatcher())
                print("      descriptor matching: exact 2-NN on cuda (torch)")
        h0, w0 = frames[0].shape
    else:
        h0, w0 = cached["shape"]
        for f in frames:
            f.shape = (h0, w0)
    diag = math.hypot(w0, h0)

    if cached is None:
        print("[3/9] matching consecutive pairs")
        cpairs = []
        for k in range(len(frames) - 1):
            p = match_pair(frames[k], frames[k + 1], args.ratio, args.min_inliers)
            if p is None:
                print(f"  ! consecutive pair {k}->{k + 1} FAILED")
                continue
            cpairs.append(p)
        print(f"      {len(cpairs)}/{len(frames) - 1} consecutive pairs matched")
        # same-column partners in the next row: these excite the y direction,
        # without which the rig-model pre-fit is rank-deficient (in-row pairs
        # only sample dx/dz/drx). Serpentine scan -> nearest gantry xy match.
        print("      matching cross-row pairs for the pre-fit")
        xpairs = []
        byrow: dict[int, list] = {}
        for f in frames:
            byrow.setdefault(f.row, []).append(f)
        for r in sorted(byrow):
            if r + 1 not in byrow:
                continue
            for f in byrow[r][::2]:
                g_ = min(byrow[r + 1], key=lambda q: abs(q.g[0] - f.g[0]))
                p = match_pair(f, g_, args.ratio, args.min_inliers)
                if p is not None:
                    xpairs.append(p)
        print(f"      {len(xpairs)} cross-row pairs matched")
    else:
        cpairs = cached["cpairs"]
        xpairs = cached["xpairs"]

    if args.rig_from:
        print(f"[4/9] rig model loaded from {args.rig_from} (pre-fit skipped)")
        rj = json.load(open(args.rig_from))["rig_model"]
        mdl = RigModel(
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
        ts = np.array([[p.tx, p.ty] for p in cpairs])
        fx0 = float(np.median(np.linalg.norm(ts, axis=1)) / 10.0 * 110.0)
        mdl, prefit_rms = prefit_rig_model(frames, cpairs + xpairs, w0, h0, args.downscale, fx0)
    if args.fx_full:
        mdl.fx = args.fx_full / args.downscale
        print(
            f"      fx LOCKED to {args.fx_full:.0f} full px ({mdl.fx:.1f} ds-px); "
            f"lever/dz0 will re-anchor during BA"
        )
    R0, C0 = mdl.poses(frames)
    axis = R0[len(frames) // 2][:, 2]
    print(
        f"      fx={mdl.fx:.0f}ds-px ({mdl.fx * args.downscale:.0f} full px), "
        f"px/mm≈{mdl.fx / 110:.1f} (ds), lever=({mdl.lever[0]:.1f},{mdl.lever[1]:.1f},"
        f"{mdl.lever[2]:.1f})mm |{np.linalg.norm(mdl.lever):.1f}mm|, dz0={mdl.dz0:.1f}mm"
    )
    print(f"      mid-scan optical axis (world): ({axis[0]:+.3f},{axis[1]:+.3f},{axis[2]:+.3f})")

    if cached is None:
        print(
            f"[5/9] matching overlapping pairs (overlap>={args.overlap_frac}, "
            f"top-{args.max_partners}/frame)"
        )
        keep, overlaps = find_overlap_pairs(
            mdl, frames, R0, C0, args.overlap_frac, args.max_partners
        )
        existing = {(p.i, p.j) for p in cpairs} | {(p.i, p.j) for p in xpairs}
        to_match = sorted(ij for ij in keep if ij not in existing)
        print(f"      {len(overlaps)} candidate pairs -> matching {len(to_match)} extra")
        pairs = list(cpairs) + list(xpairs)
        added = rejected = 0
        t0 = time.time()
        for n_, (i, j) in enumerate(to_match):
            tpred = predicted_pair_translation(mdl, frames, R0, C0, i, j)
            p = match_pair(
                frames[i],
                frames[j],
                args.ratio,
                args.min_inliers,
                prior_xy=tuple(tpred),
                prior_tol=0.25 * diag,
            )
            if p is None:
                continue
            if math.hypot(p.tx - tpred[0], p.ty - tpred[1]) > 0.2 * diag:
                rejected += 1
                continue
            pairs.append(p)
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
                        kps=[[kp.pt for kp in f.kp] for f in frames],
                    ),
                    fh,
                )
    else:
        pairs = cached["pairs"]
        # rebuild minimal kp lists for track building
        for f, kps in zip(frames, cached["kps"]):
            f.kp = [registration_features._KPt(tuple(p)) for p in kps]

    if registration_features._GPU_MATCHER is not None:
        registration_features._GPU_MATCHER.clear()  # free descriptor VRAM before render
        set_gpu_matcher(None)
    return SimpleNamespace(
        out_dir=out_dir,
        frames=frames,
        pairs=pairs,
        mdl=mdl,
        prefit_rms=prefit_rms,
        R0=R0,
        C0=C0,
    )
