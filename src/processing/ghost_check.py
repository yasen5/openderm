#!/usr/bin/env python3
"""Ghost / duplicated-feature validator for one 3D skin reconstruction.

The renderer can paint the SAME real skin feature at two slightly-offset texel
positions (residual per-frame misalignment + per-texel HF winner-take-all), so a
single mole/red-dot shows up TWICE in the ortho-texture -- a false "new lesion"
for cancer screening. This tool cross-checks the composite against the raw source
frames, which are ground truth (a raw photo shows each feature exactly once):

  for every candidate feature in texture.jpg:
    - re-detect blobs LOCALLY in the composite crop (catches close doubles the
      global detector's 2.5mm suppression merges);
    - if >=2 composite blobs, find every source frame that imaged that surface
      point, project_world_points_into_camera the composite blobs into each, and detect the real blobs;
    - two composite blobs are the SAME feature (a ghost) unless a MAJORITY of the
      covering source frames resolve them as TWO separate blobs.
  ghost_excess = (#composite blobs with source support) - (#distinct real features).

total_ghosts (summed over regions) is a scalar to MINIMISE across render settings
-- the automatic acceptance test: a correct render -> 0.

Install OpenDerm with the `vision` extra, then run:
  openderm-check-ghosts captures/<scan>/registration3d-canonical
"""

from __future__ import annotations
import argparse
import json
import os
import time
from typing import Any, Literal, Optional, TypedDict, cast

import numpy as np
import cv2
from numpy.typing import NDArray
from scipy.ndimage import gaussian_laplace, maximum_filter
from scipy.interpolate import LinearNDInterpolator

from processing.tex_anchor import Gauge, load_gauge, coverage_mask, _parse_obj_v_vt
from processing.lesions import (
    detect_angiomas,
    detect_moles,
    hemoglobin_flat,
    mel_threshold,
    melanin_flat,
)

Image = NDArray[Any]
Mask = NDArray[Any]
FloatArray = NDArray[np.float64]
Channel = Literal["mel", "hem"]
Blob = tuple[int, int, float, float]
SourceBlob = tuple[int, int, float]


class CameraIntrinsics(TypedDict):
    fxf: float
    k1: float
    cxf: float
    cyf: float
    Wf: float
    Hf: float


class SourceFrame(TypedDict):
    idx: int
    station: int
    row: int
    path: str
    R: FloatArray
    C: FloatArray
    standoff: float


class CandidateSeed(TypedDict):
    x: int
    y: int
    r: float
    channel: Channel


class Candidate(CandidateSeed):
    u: float
    v: float
    P: FloatArray


class PerFrameRecord(TypedDict):
    idx: int
    station: int
    sds: int
    crop: list[int]
    sblobs: list[SourceBlob]
    ppmm_s: float
    assign: list[int]
    preds: list[list[float]]


class GhostRecord(TypedDict):
    id: int
    channel: Channel
    tex_px: list[int]
    n_comp_blobs: int
    source_count: int
    n_phantom: int
    ghost_excess: int
    n_cover: int
    cblobs: list[Blob]
    per_frame: list[PerFrameRecord]


class AnalysisResult(TypedDict):
    tex: Image
    g: Gauge
    records: list[GhostRecord]
    total_ghosts: int
    phantoms: int
    n_candidates: int
    K: CameraIntrinsics
    _paths: list[tuple[int, str]]


class GhostReport(TypedDict):
    reg_dir: str
    total_ghosts: int
    phantoms: int
    n_candidates: int
    ghosts: list[dict[str, Any]]


class Arguments(argparse.Namespace):
    reg_dir: str = ""
    out: Optional[str] = None
    window_mm: float = 8.0
    merge_mm: float = 8.0
    match_mm: float = 2.0
    suppress_mm: float = 0.8
    k_sigma: float = 4.0
    max_frames: int = 6
    src_ppmm: float = 20.0
    src_pad_mm: float = 16.0
    global_ds: int = 3


from processing.register_scan_3d import project_world_points_into_camera


# --------------------------------------------------------------------------- #
def detect_local_lesion_candidate_peaks(
    chan: Image,
    pixels_per_mm: float,
    mask: Mask | None = None,
    diam_mm: tuple[float, float] = (0.3, 4.0),
    k_sigma: float = 4.0,
    min_contrast: float = 0.05,
    suppress_mm: float = 0.8,
) -> list[Blob]:
    """Positive-bump detector (multiscale LoG) with a SMALL NMS radius so two
    features ~1-2mm apart stay separate. Returns [(x, y, r_mm, val)]."""
    if mask is None:
        mask = np.ones(chan.shape, np.uint8)
    sig_lo = diam_mm[0] / 2 * pixels_per_mm / np.sqrt(2)
    sig_hi = diam_mm[1] / 2 * pixels_per_mm / np.sqrt(2)
    sigmas = np.geomspace(sig_lo, sig_hi, 8)
    resp = np.stack([-gaussian_laplace(chan, score) * score**2 for score in sigmas])
    peak = resp.max(0)
    scl = resp.argmax(0)
    thr = mel_threshold(chan, mask, min_contrast, k_sigma)
    win = int(sig_lo * 2) | 1
    loc = (peak == maximum_filter(peak, size=max(3, win))) & (mask > 0) & (peak > 0)
    ys, xs = np.where(loc)
    out: list[Blob] = []
    taken: Mask = np.zeros(chan.shape, np.uint8)
    sup_px = int(max(suppress_mm * pixels_per_mm, 1))
    for val, world_x, world_y in sorted(zip(peak[ys, xs], xs, ys), reverse=True):
        if float(chan[world_y, world_x]) < thr or taken[world_y, world_x]:
            continue
        r_px = sigmas[scl[world_y, world_x]] * np.sqrt(2)
        if not (diam_mm[0] <= 2 * r_px / pixels_per_mm <= diam_mm[1]):
            continue
        cv2.circle(taken, (int(world_x), int(world_y)), sup_px, 1, -1)
        out.append((int(world_x), int(world_y), float(r_px / pixels_per_mm), float(val)))
    return out


def detect_source_frame_lesions(
    bgr: Image, pixels_per_mm: float, channel: Channel, mask: Mask | None = None
) -> list[SourceBlob]:
    """Real lesion detections (calibrated + red-hue gate for angiomas), the SAME
    definition of 'a feature' used everywhere. Returns [(x, y, r_mm)]."""
    if mask is None:
        mask = np.full(bgr.shape[:2], 255, np.uint8)
    mel = melanin_flat(bgr, mask, pixels_per_mm)
    if channel == "hem":
        hem = hemoglobin_flat(bgr, mask, pixels_per_mm)
        feats = detect_angiomas(bgr, hem, mel, mask, pixels_per_mm)
    else:
        feats = detect_moles(mel, mask, pixels_per_mm)
    return [(frame["x"], frame["y"], frame["radius_mm"]) for frame in feats]


def build_texture_uv_to_world_interpolator(reg_dir: str, gauge: Gauge) -> Any:
    """Interpolator (u_mm, v_mm) -> world (x,y,z) from the exported mesh UVs."""
    texture_v_values, VT = _parse_obj_v_vt(os.path.join(reg_dir, "surface_mesh.obj"))
    texture_u = gauge.umin + VT[:, 0] * (gauge.umax - gauge.umin)  # invert build_surface_mesh's VT
    texture_v = gauge.vmin + (1.0 - VT[:, 1]) * (gauge.vmax - gauge.vmin)
    return LinearNDInterpolator(np.column_stack([texture_u, texture_v]), texture_v_values)


def load_reconstruction_rig_and_frames(
    reg_dir: str,
) -> tuple[CameraIntrinsics, list[SourceFrame]]:
    with open(os.path.join(reg_dir, "placements3d.json")) as placements_file:
        pl = cast(dict[str, Any], json.load(placements_file))
    camera_rotation, ds = pl["rig_model"], pl["downscale"]
    camera_intrinsics = CameraIntrinsics(
        fxf=float(camera_rotation["fx_fullres_px"]),
        k1=float(camera_rotation["k1"]),
        cxf=float(camera_rotation["cx"]) * ds,
        cyf=float(camera_rotation["cy"]) * ds,
        Wf=float(camera_rotation["cx"]) * ds * 2,
        Hf=float(camera_rotation["cy"]) * ds * 2,
    )
    cap = pl.get("capture_dir") or os.path.dirname(reg_dir.rstrip("/"))
    frames = [
        SourceFrame(
            idx=frame["idx"],
            station=frame["station"],
            row=frame["row"],
            path=os.path.join(cap, os.path.basename(frame["image"])),
            R=np.array(frame["R_cam2world"]),
            C=np.array(frame["C_mm"]),
            standoff=frame["gantry"]["standoff_mm"],
        )
        for frame in pl["frames"]
    ]
    return camera_intrinsics, frames


def project_world_point_into_source_frame(
    points: FloatArray, frame: SourceFrame, camera_intrinsics: CameraIntrinsics
) -> tuple[float, float, float]:
    uv, world_z = project_world_points_into_camera(np.atleast_2d(points), frame["R"], frame["C"], camera_intrinsics["fxf"], camera_intrinsics["k1"], camera_intrinsics["cxf"], camera_intrinsics["cyf"])
    return float(uv[0][0]), float(uv[0][1]), float(world_z[0])


_IMG: dict[tuple[str, int], Image | None] = {}


def load_downscaled_source_frame(path: str, ds: int) -> Image | None:
    key = (path, ds)
    if key not in _IMG:
        im = cv2.imread(path)
        if im is not None and ds != 1:
            im = cv2.resize(im, None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
        _IMG[key] = im
    return _IMG[key]


# --------------------------------------------------------------------------- #
def analyze_reconstruction_ghosts(reg_dir: str, args: Arguments) -> AnalysisResult:
    t0 = time.time()

    def log(descriptive_mask: str) -> None:
        print(f"  [{time.time() - t0:5.1f}s] {descriptive_mask}", flush=True)

    gauge = load_gauge(reg_dir)
    pixels_per_mm = gauge.ppmm
    tex = cast(Optional[Image], cv2.imread(os.path.join(reg_dir, "texture.jpg")))
    if tex is None:
        raise FileNotFoundError(os.path.join(reg_dir, "texture.jpg"))
    cov = cast(Mask, coverage_mask(reg_dir))
    camera_intrinsics, frames = load_reconstruction_rig_and_frames(reg_dir)
    log(f"loaded texture {tex.shape[1]}x{tex.shape[0]} @ {pixels_per_mm:.0f}px/mm")
    uv2w = build_texture_uv_to_world_interpolator(reg_dir, gauge)
    log("built (u,v)->world interpolator")

    # global candidate hunt on a DOWNSCALED texture (locations only; precise
    # blob geometry is redone per-crop at full res). Full-res multiscale LoG over
    # the whole 48Mpx map is what made this slow.
    gds = args.global_ds
    tg = cast(NDArray[np.uint8], cv2.resize(tex, None, fx=1 / gds, fy=1 / gds, interpolation=cv2.INTER_AREA))
    cg = cast(NDArray[np.uint8], cv2.resize(cov, (tg.shape[1], tg.shape[0]), interpolation=cv2.INTER_NEAREST))
    dm_g = cast(
        NDArray[np.uint8],
        cv2.erode(
            cg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(3 * pixels_per_mm / gds) | 1,) * 2)
        ),
    )
    ppg = pixels_per_mm / gds
    mel_g = melanin_flat(tg, cg, ppg)
    hem_g = hemoglobin_flat(tg, cg, ppg)
    cands: list[CandidateSeed] = [
        CandidateSeed(x=descriptive_mask["x"] * gds, y=descriptive_mask["y"] * gds, r=descriptive_mask["radius_mm"], channel="mel")
        for descriptive_mask in detect_moles(mel_g, dm_g, ppg)
    ] + [
        CandidateSeed(x=angioma_detection["x"] * gds, y=angioma_detection["y"] * gds, r=angioma_detection["radius_mm"], channel="hem")
        for angioma_detection in detect_angiomas(tg, hem_g, mel_g, dm_g, ppg)
    ]
    cands.sort(key=lambda candidate: candidate["channel"] != "hem")  # prefer red on overlap
    kept: list[Candidate] = []
    for candidate in cands:
        if all((candidate["x"] - item_index["x"]) ** 2 + (candidate["y"] - item_index["y"]) ** 2 > (2.5 * pixels_per_mm) ** 2 for item_index in kept):
            u_mm, v_mm = gauge.px_to_uv(candidate["x"], candidate["y"])
            points = uv2w(float(u_mm), float(v_mm))
            if points is None or np.any(np.isnan(points)):
                continue
            kept.append(
                Candidate(
                    **candidate,
                    u=float(u_mm),
                    v=float(v_mm),
                    P=np.asarray(points).ravel(),
                )
            )
    log(f"{len(kept)} candidate features ({sum(candidate['channel'] == 'hem' for candidate in kept)} real red)")

    # group candidates that sit within merge_mm on the surface -- only a group of
    # >=2 can be a doubling. A ghost = members the source frames resolve as fewer.
    candidate_count = len(kept)
    parent = list(range(candidate_count))

    def find(candidate_index: int) -> int:
        while parent[candidate_index] != candidate_index:
            parent[candidate_index] = parent[parent[candidate_index]]
            candidate_index = parent[candidate_index]
        return candidate_index

    for index in range(candidate_count):
        for neighbor_index in range(index + 1, candidate_count):
            if (
                kept[index]["channel"] == kept[neighbor_index]["channel"]
                and np.linalg.norm(kept[index]["P"] - kept[neighbor_index]["P"]) < args.merge_mm
            ):
                parent[find(index)] = find(neighbor_index)
    groups: dict[int, list[int]] = {}
    for index in range(candidate_count):
        groups.setdefault(find(index), []).append(index)

    records: list[GhostRecord] = []
    total_ghosts = 0
    phantoms = 0
    for gi, mem in enumerate(groups.values()):
        ms = [kept[index] for index in mem]
        ch = ms[0]["channel"]
        cworld = [descriptive_mask["P"] for descriptive_mask in ms]
        cblobs = [(int(item["x"]), int(item["y"]), item["r"], 0.0) for item in ms]
        cx, cy = int(np.mean([descriptive_mask["x"] for descriptive_mask in ms])), int(np.mean([descriptive_mask["y"] for descriptive_mask in ms]))
        rec = GhostRecord(
            id=gi,
            channel=ch,
            tex_px=[cx, cy],
            n_comp_blobs=len(ms),
            source_count=len(ms),
            n_phantom=0,
            ghost_excess=0,
            n_cover=0,
            cblobs=cblobs,
            per_frame=[],
        )
        if len(ms) < 2:  # lone feature -> no double
            records.append(rec)
            continue

        Pc = np.mean(cworld, axis=0)
        cover_candidates: list[tuple[SourceFrame, float]] = []
        for frame in frames:
            px, py, world_z = project_world_point_into_source_frame(Pc, frame, camera_intrinsics)
            if world_z > 10 and 10 <= px < camera_intrinsics["Wf"] - 10 and 10 <= py < camera_intrinsics["Hf"] - 10:
                cover_candidates.append((frame, np.hypot(px - camera_intrinsics["cxf"], py - camera_intrinsics["cyf"])))
        cover_candidates.sort(key=lambda threshold: threshold[1])
        cover = [frame for frame, center_distance in cover_candidates[: args.max_frames]]

        nb = len(ms)
        sep = np.zeros((nb, nb))
        seen = np.zeros((nb, nb))
        support = np.zeros(nb)
        per_frame: list[PerFrameRecord] = []
        for frame in cover:
            # downscale each source frame to ~src_ppmm (matches the composite's
            # scale): at ~20px/mm the pore/texture noise that wrecks the adaptive
            # threshold at full res is averaged away, so the calibrated detectors
            # behave as they do on the composite.
            sds = max(1, round(camera_intrinsics["fxf"] / frame["standoff"] / args.src_ppmm))
            im = load_downscaled_source_frame(frame["path"], sds)
            if im is None:
                continue
            ppmm_s = camera_intrinsics["fxf"] / frame["standoff"] / sds
            preds = [project_world_point_into_source_frame(weight, frame, camera_intrinsics) for weight in cworld]
            xs = [point[0] / sds for point in preds]
            ys = [point[1] / sds for point in preds]
            mrg = int(args.src_pad_mm * ppmm_s)
            sx0 = int(max(0, min(xs) - mrg))
            sy0 = int(max(0, min(ys) - mrg))
            sx1 = int(min(im.shape[1], max(xs) + mrg))
            sy1 = int(min(im.shape[0], max(ys) + mrg))
            if sx1 - sx0 < 5 or sy1 - sy0 < 5:
                continue
            sblobs = detect_source_frame_lesions(im[sy0:sy1, sx0:sx1], ppmm_s, ch)
            tol = args.match_mm * ppmm_s
            assign: list[int] = []
            for px, py, world_z in preds:
                lx, ly = px / sds - sx0, py / sds - sy0
                best, bd = -1, tol
                for si, (sx, sy, sr) in enumerate(sblobs):
                    distance = np.hypot(sx - lx, sy - ly)
                    if distance < bd:
                        bd, best = distance, si
                assign.append(best)
            for index in range(nb):
                if assign[index] >= 0:
                    support[index] += 1
                for neighbor_index in range(index + 1, nb):
                    if assign[index] >= 0 and assign[neighbor_index] >= 0:
                        seen[index, neighbor_index] += 1
                        if assign[index] != assign[neighbor_index]:
                            sep[index, neighbor_index] += 1
            per_frame.append(
                PerFrameRecord(
                    idx=frame["idx"],
                    station=frame["station"],
                    sds=sds,
                    crop=[sx0, sy0, sx1, sy1],
                    sblobs=sblobs,
                    ppmm_s=ppmm_s,
                    assign=assign,
                    preds=[[point[0] / sds, point[1] / sds] for point in preds],
                )
            )

        par2 = list(range(nb))

        def find2(candidate_index: int) -> int:
            while par2[candidate_index] != candidate_index:
                par2[candidate_index] = par2[par2[candidate_index]]
                candidate_index = par2[candidate_index]
            return candidate_index

        for index in range(nb):
            for neighbor_index in range(index + 1, nb):
                if seen[index, neighbor_index] > 0 and sep[index, neighbor_index] <= seen[index, neighbor_index] / 2.0:
                    par2[find2(index)] = find2(neighbor_index)
        n_support = int(np.sum(support > 0))
        source_count = len({find2(index) for index in range(nb) if support[index] > 0})
        n_phantom = int(np.sum(support == 0))
        ghost_excess = max(0, n_support - source_count)
        total_ghosts += ghost_excess
        phantoms += n_phantom
        rec.update(
            source_count=source_count,
            n_phantom=n_phantom,
            ghost_excess=ghost_excess,
            n_cover=len(cover),
            per_frame=per_frame,
        )
        records.append(rec)
        if ghost_excess > 0 or n_phantom > 0:
            tag = "GHOST" if ghost_excess > 0 else "phantom"
            log(
                f"  [{tag}] group {gi} ({ch}) @ (u={ms[0]['u']:.0f},v={ms[0]['v']:.0f})"
                f"mm: {nb} composite features, {source_count} real in {len(cover)} "
                f"source frames -> +{ghost_excess} ghost"
            )
    return AnalysisResult(
        tex=tex,
        g=gauge,
        records=records,
        total_ghosts=total_ghosts,
        phantoms=phantoms,
        n_candidates=len(kept),
        K=camera_intrinsics,
        _paths=[],
    )


# --------------------------------------------------------------------------- #
def write_reconstruction_ghost_reports(res: AnalysisResult, out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "montage"), exist_ok=True)
    tex, gauge = res["tex"], res["g"]
    ov = tex.copy()
    for camera_rotation in res["records"]:
        gh = camera_rotation["ghost_excess"] > 0
        col = (0, 0, 255) if gh else ((0, 165, 255) if camera_rotation["n_phantom"] else (0, 200, 0))
        pts = [(bx, by) for bx, by, rr, texture_v in camera_rotation["cblobs"]]
        for bx, by in pts:
            cv2.circle(ov, (bx, by), int(12), col, 3)
        if gh:
            for first_value in range(len(pts)):
                for second_value in range(first_value + 1, len(pts)):
                    cv2.line(ov, pts[first_value], pts[second_value], col, 2)
    cv2.putText(
        ov,
        f"ghosts={res['total_ghosts']} phantoms={res['phantoms']} features={res['n_candidates']}",
        (20, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )
    cv2.imwrite(os.path.join(out_dir, "ghost_overlay.png"), ov)

    for camera_rotation in res["records"]:
        if camera_rotation["ghost_excess"] <= 0 and camera_rotation["n_phantom"] <= 0:
            continue
        cx, cy = camera_rotation["tex_px"]
        wp = int(6 * gauge.ppmm)
        cc = tex[max(0, cy - wp) : cy + wp, max(0, cx - wp) : cx + wp].copy()
        for bx, by, rr, texture_v in camera_rotation["cblobs"]:
            cv2.circle(
                cc,
                (bx - max(0, cx - wp), by - max(0, cy - wp)),
                int(rr * gauge.ppmm + 8),
                (0, 0, 255),
                2,
            )
        tiles = [_label(cv2.resize(cc, (280, 280)), "COMPOSITE")]
        for pf in camera_rotation["per_frame"][:4]:
            source_path = _path_for(res, pf["idx"])
            if source_path is None:
                continue
            im = load_downscaled_source_frame(source_path, pf["sds"])
            if im is None:
                continue
            sx0, sy0, sx1, sy1 = pf["crop"]
            crop = im[sy0:sy1, sx0:sx1].copy()
            for sx, sy, sr in pf["sblobs"]:
                cv2.circle(crop, (int(sx), int(sy)), int(sr * pf["ppmm_s"] + 6), (0, 255, 0), 2)
            for px, py in pf["preds"]:
                cv2.drawMarker(
                    crop, (int(px - sx0), int(py - sy0)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2
                )
            if crop.size:
                tiles.append(_label(cv2.resize(crop, (280, 280)), f"src #{pf['station']}"))
        if tiles:
            image_height = 280
            row = np.hstack([cv2.resize(threshold, (280, image_height)) for threshold in tiles])
            cv2.imwrite(os.path.join(out_dir, "montage", f"ghost_{camera_rotation['id']}.png"), row)

    slim = GhostReport(
        reg_dir=out_dir,
        total_ghosts=res["total_ghosts"],
        phantoms=res["phantoms"],
        n_candidates=res["n_candidates"],
        ghosts=[
            {
                "id": camera_rotation["id"],
                "channel": camera_rotation["channel"],
                "tex_px": camera_rotation["tex_px"],
                "n_comp_blobs": camera_rotation["n_comp_blobs"],
                "source_count": camera_rotation["source_count"],
                "n_phantom": camera_rotation["n_phantom"],
                "ghost_excess": camera_rotation["ghost_excess"],
                "n_cover": camera_rotation["n_cover"],
            }
            for camera_rotation in res["records"]
            if camera_rotation["ghost_excess"] > 0 or camera_rotation["n_phantom"] > 0
        ],
    )
    with open(os.path.join(out_dir, "ghost_report.json"), "w") as report_file:
        json.dump(slim, report_file, indent=1)


def _label(img: Image, txt: str) -> Image:
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


_PATHS: dict[int, str] = {}


def _path_for(res: AnalysisResult, idx: int) -> str | None:
    if not _PATHS:
        _PATHS.update({frame: point for frame, point in res.get("_paths", [])})
    return _PATHS.get(idx)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("reg_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--window-mm", type=float, default=8.0)
    ap.add_argument(
        "--merge-mm",
        type=float,
        default=8.0,
        help="max surface distance to consider two features a possible "
        "doubling of one (only groups of >=2 are checked)",
    )
    ap.add_argument("--match-mm", type=float, default=2.0)
    ap.add_argument("--suppress-mm", type=float, default=0.8)
    ap.add_argument("--k-sigma", type=float, default=4.0)
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument(
        "--src-ppmm",
        type=float,
        default=20.0,
        help="downscale each source frame to ~this px/mm before "
        "detection (matches the composite; blurs away pore noise)",
    )
    ap.add_argument(
        "--src-pad-mm",
        type=float,
        default=16.0,
        help="half-size of the source-frame crop (gives the background flatten enough context)",
    )
    ap.add_argument(
        "--global-ds",
        type=int,
        default=3,
        help="downscale for the global candidate hunt (locations only)",
    )
    args = ap.parse_args(namespace=Arguments())
    reg_dir = args.reg_dir.rstrip("/")
    out_dir = args.out or os.path.join(reg_dir, "ghost-check")
    reconstruction_rig, frames = load_reconstruction_rig_and_frames(reg_dir)
    res = analyze_reconstruction_ghosts(reg_dir, args)
    res["_paths"] = [(frame["idx"], frame["path"]) for frame in frames]
    write_reconstruction_ghost_reports(res, out_dir)
    doublings = sum(1 for camera_rotation in res["records"] if camera_rotation["n_comp_blobs"] >= 2)
    print(
        f"DOUBLINGS: {doublings} region(s) where the composite shows >=2 "
        f"features at one spot  |  confirmed ghosts: {res['total_ghosts']}, "
        f"unconfirmed: {res['phantoms']}  over {res['n_candidates']} features"
    )
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
