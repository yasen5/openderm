#!/usr/bin/env python3
"""Find rendered feature DOUBLINGS in a skin ortho-texture (source-free).

A rendering doubling paints the SAME skin (a mole/dot/scar + its surrounding
pores) at two nearby texture locations, so the two patches are near-IDENTICAL.
Two genuinely-distinct lesions are never pixel-identical in their surroundings.
So: for every detected feature, template-match its patch against the local
neighbourhood; a strong secondary correlation peak (besides itself) is a
duplicate. Robust, and needs no source frames.

total_doublings is the scalar to drive to ZERO across render settings.

  openderm-check-doublings captures/<scan>/registration3d-canonical
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np
import cv2
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter

from processing.tex_anchor import Gauge, load_gauge, coverage_mask
from processing.lesions import (
    detect_angiomas,
    detect_moles,
    hemoglobin_flat,
    melanin_flat,
)
from processing.ghost_check import load_reconstruction_rig_and_frames, build_texture_uv_to_world_interpolator, project_world_point_into_source_frame, load_downscaled_source_frame

Image = NDArray[Any]
Mask = NDArray[np.uint8]
ChannelName = Literal["mel", "hem"]
Verdict = Literal["unknown", "distinct", "DOUBLING", "weak"]
Pair = tuple[float, int, int, int, int, float, ChannelName]
ConfirmedPair = tuple[float, int, int, int, int, float, ChannelName, Verdict, int, int, int]
GridCell = tuple[int, int]
PairKey = tuple[GridCell, GridCell]


class Lesion(TypedDict):
    x: int
    y: int
    radius_mm: float


class Candidate(TypedDict):
    pixel_x: int
    pixel_y: int
    radius: float
    ch: ChannelName


class DoublingResult(TypedDict):
    texture_gauge: Gauge
    tex: Image
    pixels_per_mm: float
    pairs: list[Pair]
    n_seeds: int
    confirmed: list[ConfirmedPair]


@dataclass
class Arguments:
    reg_dir: str = ""
    out: str | None = None
    patch_mm: float = 5.0
    search_mm: float = 22.0
    min_offset_mm: float = 3.0
    ncc: float = 0.5
    k_sigma: float = 3.5
    min_std: float = 4.0
    no_confirm: bool = False
    z_thresh: float = 3.5
    src_ppmm: float = 20.0
    min_cover: int = 3


def measure_source_feature_contrast_zscore(
    source_image: Image,
    pixel_x: float,
    pixel_y: float,
    pixels_per_mm: float,
    channel_name: ChannelName,
) -> float:
    """Feature-vs-skin z-score at a KNOWN pixel: how much darker (mole) or redder
    (angioma) the spot is than its own local skin ring, in robust-sigma units.
    Robust to raw-frame pore noise because it's a local contrast at a fixed
    location -- no blind peak detection / adaptive global threshold."""
    red_channel = source_image[:, :, 2].astype(np.float32)
    green_channel = source_image[:, :, 1].astype(np.float32)
    contrast_map = (
        (np.log(np.maximum(red_channel, 1)) - np.log(np.maximum(green_channel, 1)))
        if channel_name == "hem"
        else -np.log(np.maximum(red_channel, 1))
    )  # mole = darker = higher
    core_radius_px = max(2, int(0.7 * pixels_per_mm))
    outer_radius_px = int(4.5 * pixels_per_mm)
    inner_ring_radius_px = int(2.0 * pixels_per_mm)
    pixel_x, pixel_y = int(round(pixel_x)), int(round(pixel_y))
    crop_y_min, crop_y_max = max(0, pixel_y - outer_radius_px), min(contrast_map.shape[0], pixel_y + outer_radius_px)
    crop_x_min, crop_x_max = max(0, pixel_x - outer_radius_px), min(contrast_map.shape[1], pixel_x + outer_radius_px)
    contrast_crop = contrast_map[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    if contrast_crop.size < 25:
        return -99.0
    pixel_y_grid, pixel_x_grid = np.ogrid[: contrast_crop.shape[0], : contrast_crop.shape[1]]
    center_distance_px = np.hypot(pixel_x_grid - (pixel_x - crop_x_min), pixel_y_grid - (pixel_y - crop_y_min))
    feature_core_values = contrast_crop[center_distance_px <= core_radius_px]
    surrounding_skin_values = contrast_crop[
        (center_distance_px >= inner_ring_radius_px) & (center_distance_px <= outer_radius_px)
    ]
    if feature_core_values.size < 3 or surrounding_skin_values.size < 20:
        return -99.0
    skin_median = float(np.median(surrounding_skin_values))
    skin_mad = 1.4826 * float(np.median(np.abs(surrounding_skin_values - skin_median))) + 1e-6
    return (float(np.median(feature_core_values)) - skin_median) / skin_mad


def detect_texture_doubling_candidates(
    texture: Image, coverage: Mask, pixels_per_mm: float, robust_sigma_multiplier: float
) -> tuple[list[Candidate], Mask]:
    er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(3 * pixels_per_mm) | 1,) * 2)
    detection_coverage_mask: Mask = np.asarray(cv2.erode(coverage, er), dtype=np.uint8)
    melanin_channel = melanin_flat(texture, coverage, pixels_per_mm)
    hemoglobin_channel = hemoglobin_flat(texture, coverage, pixels_per_mm)
    candidate_features = [
        Candidate(pixel_x=mole["x"], pixel_y=mole["y"], radius=mole["radius_mm"], ch="mel")
        for mole in detect_moles(
            melanin_channel, detection_coverage_mask, pixels_per_mm,
            robust_sigma_multiplier=robust_sigma_multiplier
        )
    ] + [
        Candidate(pixel_x=angioma["x"], pixel_y=angioma["y"], radius=angioma["radius_mm"], ch="hem")
        for angioma in detect_angiomas(
            texture, hemoglobin_channel, melanin_channel, detection_coverage_mask,
            pixels_per_mm, k_sigma=robust_sigma_multiplier
        )
    ]
    return candidate_features, detection_coverage_mask


def find_texture_doublings(reg_dir: str, args: Arguments) -> DoublingResult:
    texture_gauge = load_gauge(reg_dir)
    pixels_per_mm = texture_gauge.ppmm
    tex = cv2.imread(os.path.join(reg_dir, "texture.jpg"))
    if tex is None:
        raise ValueError(f"Could not read texture image from {reg_dir!r}")
    cov = coverage_mask(reg_dir)
    gray = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cand, dm = detect_texture_doubling_candidates(tex, cov, pixels_per_mm, args.k_sigma)
    covm = dm > 0
    print(f"  {len(cand)} seed features @ {pixels_per_mm:.0f}px/mm", flush=True)

    pr = int(args.patch_mm * pixels_per_mm)  # half patch
    sr = int(args.search_mm * pixels_per_mm)  # half search window
    excl = int(args.min_offset_mm * pixels_per_mm)
    image_height, image_width = gray.shape
    pairs: list[Pair] = []
    for score in cand:
        pixel_x, pixel_y = score["pixel_x"], score["pixel_y"]
        if pixel_x - pr < 0 or pixel_y - pr < 0 or pixel_x + pr >= image_width or pixel_y + pr >= image_height:
            continue
        patch = gray[pixel_y - pr : pixel_y + pr, pixel_x - pr : pixel_x + pr]
        if patch.std() < args.min_std:  # featureless skin -> skip
            continue
        wx0, wy0 = max(0, pixel_x - sr), max(0, pixel_y - sr)
        win = gray[wy0 : min(image_height, pixel_y + sr), wx0 : min(image_width, pixel_x + sr)]
        if win.shape[0] <= patch.shape[0] or win.shape[1] <= patch.shape[1]:
            continue
        res = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
        # peak coords are top-left of the matched patch; centre offset:
        pk = (res == maximum_filter(res, size=int(2 * excl) | 1)) & (res > args.ncc)
        ys, xs = np.where(pk)
        for row_idx, col_idx in zip(ys, xs):
            dx = (col_idx + pr) - (pixel_x - wx0)  # dup_centre - seed_centre (px)
            dy = (row_idx + pr) - (pixel_y - wy0)
            off = np.hypot(dx, dy) / pixels_per_mm
            if off < args.min_offset_mm or off > args.search_mm:
                continue
            dupx, dupy = pixel_x + dx, pixel_y + dy
            if not (0 <= dupx < image_width and 0 <= dupy < image_height and covm[dupy, dupx]):
                continue
            pairs.append(
                (float(res[row_idx, col_idx]), pixel_x, pixel_y, int(dupx), int(dupy), float(off), score["ch"])
            )
    # dedupe symmetric / overlapping pairs: canonical unordered key at ~3mm grid
    pairs.sort(reverse=True)
    seen: set[PairKey] = set()
    uniq: list[Pair] = []
    deduplication_grid_size_px = max(1, int(3 * pixels_per_mm))
    for point in pairs:
        ncc, pixel_x, pixel_y, dx, dy, off, ch = point
        first_endpoint_grid_cell = (
            min(pixel_x // deduplication_grid_size_px, dx // deduplication_grid_size_px),
            min(pixel_y // deduplication_grid_size_px, dy // deduplication_grid_size_px),
        )
        second_endpoint_grid_cell = (
            max(pixel_x // deduplication_grid_size_px, dx // deduplication_grid_size_px),
            max(pixel_y // deduplication_grid_size_px, dy // deduplication_grid_size_px),
        )
        key = (first_endpoint_grid_cell, second_endpoint_grid_cell)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(point)
    return DoublingResult(
        texture_gauge=texture_gauge,
        tex=tex,
        pixels_per_mm=pixels_per_mm,
        pairs=uniq,
        n_seeds=len(cand),
        confirmed=[],
    )


def write_texture_doubling_reports(res: DoublingResult, out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "pairs"), exist_ok=True)
    tex, pixels_per_mm = res["tex"], res["pixels_per_mm"]
    ov = tex.copy()
    for item_index, (ncc, pixel_x, pixel_y, dx, dy, off, ch) in enumerate(res["pairs"]):
        cv2.circle(ov, (pixel_x, pixel_y), 16, (0, 0, 255), 3)
        cv2.circle(ov, (dx, dy), 16, (0, 0, 255), 3)
        cv2.line(ov, (pixel_x, pixel_y), (dx, dy), (0, 0, 255), 2)
        cv2.putText(
            ov,
            f"{item_index}:{ncc:.2f}",
            (min(pixel_x, dx), min(pixel_y, dy) - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        crop_radius_px = int(6 * pixels_per_mm)
        first_patch = tex[max(0, pixel_y - crop_radius_px) : pixel_y + crop_radius_px, max(0, pixel_x - crop_radius_px) : pixel_x + crop_radius_px]
        second_patch = tex[max(0, dy - crop_radius_px) : dy + crop_radius_px, max(0, dx - crop_radius_px) : dx + crop_radius_px]
        if first_patch.size and second_patch.size:
            first_patch = cv2.resize(first_patch, (240, 240))
            second_patch = cv2.resize(second_patch, (240, 240))
            cv2.putText(first_patch, "copy A", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(second_patch, "copy B", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(
                os.path.join(out_dir, "pairs", f"pair_{item_index}_ncc{ncc:.2f}.png"), np.hstack([first_patch, second_patch])
            )
    cv2.putText(
        ov,
        f"doublings={len(res['pairs'])}",
        (20, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )
    cv2.imwrite(os.path.join(out_dir, "doublings_overlay.png"), ov)


def confirm_texture_doublings_from_source_frames(
    res: DoublingResult, reg_dir: str, args: Arguments
) -> list[ConfirmedPair]:
    """For each NCC candidate pair A<->B, use the SOURCE frames as ground truth:
    if it's one real feature doubled, NO single frame can show a strong feature
    at BOTH placements; two genuinely-distinct lesions appear at both in most
    covering frames. Returns pairs with a verdict + evidence."""
    texture_gauge = res["texture_gauge"]
    camera_intrinsics, frames = load_reconstruction_rig_and_frames(reg_dir)
    uv2w = build_texture_uv_to_world_interpolator(reg_dir, texture_gauge)
    zt = args.z_thresh
    out: list[ConfirmedPair] = []
    for ncc, pixel_x, pixel_y, dx, dy, off, ch in res["pairs"]:
        uA = texture_gauge.px_to_uv(pixel_x, pixel_y)
        uB = texture_gauge.px_to_uv(dx, dy)
        PA = uv2w(float(uA[0]), float(uA[1]))
        PB = uv2w(float(uB[0]), float(uB[1]))
        if PA is None or PB is None or np.any(np.isnan(PA)) or np.any(np.isnan(PB)):
            out.append((ncc, pixel_x, pixel_y, dx, dy, off, ch, "unknown", 0, 0, 0))
            continue
        PA = np.asarray(PA).ravel()
        PB = np.asarray(PB).ravel()
        both = a_only = b_only = ncov = 0
        for frame in frames:
            aX, aY, aZ = project_world_point_into_source_frame(PA, frame, camera_intrinsics)
            bX, bY, bZ = project_world_point_into_source_frame(PB, frame, camera_intrinsics)
            inb = lambda px, py, world_z: world_z > 10 and 12 <= px < camera_intrinsics["Wf"] - 12 and 12 <= py < camera_intrinsics["Hf"] - 12
            if not (inb(aX, aY, aZ) and inb(bX, bY, bZ)):
                continue
            sds = max(1, round(camera_intrinsics["fxf"] / frame["standoff"] / args.src_ppmm))
            im = load_downscaled_source_frame(frame["path"], sds)
            if im is None:
                continue
            pp = camera_intrinsics["fxf"] / frame["standoff"] / sds
            zA = measure_source_feature_contrast_zscore(im, aX / sds, aY / sds, pp, ch)
            zB = measure_source_feature_contrast_zscore(im, bX / sds, bY / sds, pp, ch)
            ncov += 1
            if zA > zt and zB > zt:
                both += 1
            elif zA > zt:
                a_only += 1
            elif zB > zt:
                b_only += 1
        if ncov < args.min_cover:
            verdict = "unknown"
        elif both >= max(2, 0.35 * ncov):
            verdict = "distinct"  # both real in many frames
        elif a_only + b_only >= max(2, 0.35 * ncov) and both <= 0.15 * ncov:
            verdict = "DOUBLING"  # each seen alone, never together
        else:
            verdict = "weak"  # not clearly a feature
        out.append((ncc, pixel_x, pixel_y, dx, dy, off, ch, verdict, both, a_only + b_only, ncov))
    res["confirmed"] = out
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("reg_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--patch-mm", type=float, default=5.0)
    ap.add_argument("--search-mm", type=float, default=22.0)
    ap.add_argument("--min-offset-mm", type=float, default=3.0)
    ap.add_argument("--ncc", type=float, default=0.5)
    ap.add_argument("--k-sigma", type=float, default=3.5)
    ap.add_argument("--min-std", type=float, default=4.0)
    ap.add_argument(
        "--no-confirm_texture_doublings_from_source_frames", dest="no_confirm", action="store_true", help="skip the source-frame ground-truth confirmation"
    )
    ap.add_argument(
        "--z-thresh",
        type=float,
        default=3.5,
        help="matched-filter z for 'a feature is present here'",
    )
    ap.add_argument("--src-ppmm", type=float, default=20.0)
    ap.add_argument("--min-cover", type=int, default=3)
    args = ap.parse_args(namespace=Arguments())
    reg = args.reg_dir.rstrip("/")
    out = args.out or os.path.join(reg, "doublings")
    res = find_texture_doublings(reg, args)
    print(f"  {len(res['pairs'])} NCC candidate pairs; confirming vs source...", flush=True)
    if args.no_confirm:
        write_texture_doubling_reports(res, out)
        print(f"DOUBLINGS(NCC only): {len(res['pairs'])}  -> {out}", flush=True)
        return
    conf = confirm_texture_doublings_from_source_frames(res, reg, args)
    write_texture_doubling_reports(res, out)
    order = {"DOUBLING": 0, "distinct": 1, "weak": 2, "unknown": 3}
    conf.sort(key=lambda candidate: (order[candidate[7]], -candidate[0]))
    ndbl = 0
    for ncc, pixel_x, pixel_y, dx, dy, off, ch, verdict, both, alone, ncov in conf:
        texture_u, texture_v = map_gauge_coordinates_to_texture_uv(res["texture_gauge"], pixel_x, pixel_y)
        if verdict == "DOUBLING":
            ndbl += 1
        if verdict in ("DOUBLING", "distinct"):
            print(
                f"    [{verdict:8}] {ch} off={off:4.1f}mm ncc={ncc:.2f} @ "
                f"(u={texture_u:.0f},v={texture_v:.0f})mm  frames both={both} alone={alone}/{ncov}",
                flush=True,
            )
    print(
        f"CONFIRMED DOUBLINGS: {ndbl}   (NCC candidates {len(res['pairs'])}, "
        f"detect_texture_doubling_candidates {res['n_seeds']})  -> {out}",
        flush=True,
    )


def map_gauge_coordinates_to_texture_uv(
    texture_gauge: Gauge, pixel_x: int, pixel_y: int
) -> tuple[float, float]:
    texture_u, texture_v = texture_gauge.px_to_uv(pixel_x, pixel_y)
    return float(texture_u), float(texture_v)


if __name__ == "__main__":
    main()
