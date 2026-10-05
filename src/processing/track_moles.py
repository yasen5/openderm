#!/usr/bin/env python
"""M2 lesion size-change tracking: measure per-mole size/shape change between two
registered scans, with a per-mole uncertainty and a significance flag.

Load-bearing rule: every size/shape number is measured in each scan's OWN
un-warped melanin frame; the transform/warp is used ONLY to match moles and gate
coverage, never to resample lesion pixels. That is what keeps a growing rim from
being pulled back to its old size.

Exposure invariance: melanin_flat is a log-ratio map (log(R_skin/R)), so a
multiplicative exposure/white-balance gain g cancels (log(g*R_skin/g*R)). Hence a
mole's peak and any FRACTION-of-peak contour are gain-invariant -- we measure the
lesion footprint at half its own peak contrast, not an absolute threshold (which
would make a brighter scan read every mole as shrunk).

This module is consumed by ``compare_scans.py``.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.interpolate import RBFInterpolator


# --------------------------------------------------------------------------- #
# Size & shape measurement (one scan's own un-warped frame)
# --------------------------------------------------------------------------- #
def _disk_mask(shape, center_x_px, center_y_px, radius_px):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - center_x_px) ** 2 + (yy - center_y_px) ** 2 <= radius_px * radius_px


def _local_baseline(bump, cov, cx, cy, r_in, r_out, other_mask):
    """Robust skin baseline from an annulus around the mole (inside coverage,
    excluding other moles)."""
    yy, xx = np.ogrid[: bump.shape[0], : bump.shape[1]]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    annulus = (d2 >= r_in * r_in) & (d2 <= r_out * r_out) & (cov > 0) & (~other_mask)
    vals = bump[annulus]
    return float(np.median(vals)) if vals.size >= 20 else 0.0


def _iso_region(bump, level, cx, cy, pixels_per_mm):
    """Half-max connected component containing (cx,cy), closed, + its contour."""
    binary = (bump >= level).astype(np.uint8)
    lab, item_count = label(binary)
    if item_count == 0:
        return None, None, 0
    cl = lab[int(round(cy)), int(round(cx))]
    if cl == 0:  # center fell below level
        return None, None, 0
    mask = (lab == cl).astype(np.uint8)
    item_index = max(1, int(round(0.2 * pixels_per_mm)))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (item_index | 1, item_index | 1))
    )
    contours, contour_hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask, None, int(mask.sum())
    contour = max(contours, key=cv2.contourArea)
    return mask, contour, int(mask.sum())


def measure_mole_size(
    melanin_image, coverage_mask, mole_detection, pixels_per_mm, iso_fraction=0.5, presmoothing_sigma_mm=0.35, peak_core_radius_mm=0.5, other_mole_detections=None
):
    """Half-max iso-contour measurement of one detected mole in one scan's melanin map.

    Returns equivalent_diameter_mm, region_area_mm2, eccentricity, major/minor_axis_length_mm, border_irregularity,
    region_perimeter_mm, peak, center, plus per-measurement sigmas and a `measurement_valid` flag."""
    mole_radius_mm = max(mole_detection["radius_mm"], 0.3)
    crop_half_size_px = int(math.ceil(max(3 * mole_radius_mm, 4.0) * pixels_per_mm))
    image_height, image_width = melanin_image.shape
    mole_pixel_x, mole_pixel_y = int(round(mole_detection["x"])), int(round(mole_detection["y"]))
    x0, y0 = mole_pixel_x - crop_half_size_px, mole_pixel_y - crop_half_size_px
    x1, y1 = mole_pixel_x + crop_half_size_px, mole_pixel_y + crop_half_size_px
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(image_width, x1), min(image_height, y1)
    sz = 2 * crop_half_size_px
    patch = np.zeros((sz, sz), np.float32)
    patch_coverage = np.zeros((sz, sz), np.uint8)
    patch[pad_t : pad_t + (sy1 - sy0), pad_l : pad_l + (sx1 - sx0)] = melanin_image[sy0:sy1, sx0:sx1]
    patch_coverage[pad_t : pad_t + (sy1 - sy0), pad_l : pad_l + (sx1 - sx0)] = coverage_mask[sy0:sy1, sx0:sx1]
    patch_center_x, patch_center_y = crop_half_size_px, crop_half_size_px  # mole center in patch coordinates

    # mask out OTHER moles inside the patch (so baseline/region don't merge them)
    other_mole_exclusion_mask = np.zeros((sz, sz), bool)
    if other_mole_detections:
        for other_mole in other_mole_detections:
            if other_mole is mole_detection:
                continue
            other_mole_patch_x, other_mole_patch_y = other_mole["x"] - x0, other_mole["y"] - y0
            if -crop_half_size_px < other_mole_patch_x < sz + crop_half_size_px and -crop_half_size_px < other_mole_patch_y < sz + crop_half_size_px:
                other_mole_exclusion_mask |= _disk_mask((sz, sz), other_mole_patch_x, other_mole_patch_y, max(other_mole["radius_mm"], 0.3) * pixels_per_mm * 1.5)

    baseline_inner_radius_px, baseline_outer_radius_px = 2.0 * mole_radius_mm * pixels_per_mm, crop_half_size_px
    baseline_melanin_value = _local_baseline(patch, patch_coverage, patch_center_x, patch_center_y, baseline_inner_radius_px, baseline_outer_radius_px, other_mole_exclusion_mask)
    contrast_bump = gaussian_filter(patch - baseline_melanin_value, presmoothing_sigma_mm * pixels_per_mm)

    # robust peak in a small disk at the LoG center; refine the center using the core mask
    peak_core_mask = _disk_mask((sz, sz), patch_center_x, patch_center_y, peak_core_radius_mm * pixels_per_mm)
    peak_contrast = float(np.percentile(contrast_bump[peak_core_mask], 95)) if peak_core_mask.any() else float(contrast_bump.max())
    measurement_result = dict(
        valid=False,
        peak=peak_contrast,
        d_eq_mm=float("nan"),
        area_mm2=float("nan"),
        eccentricity=float("nan"),
        major_mm=float("nan"),
        minor_mm=float("nan"),
        border_irregularity=float("nan"),
        perim_mm=float("nan"),
        center=(mole_pixel_x, mole_pixel_y),
        sigma_diam_meas=float("nan"),
        sigma_area_meas=float("nan"),
        sigma_m=float("nan"),
        gbar=float("nan"),
        n_pixels=0,
        reason="",
    )
    if not np.isfinite(peak_contrast) or peak_contrast <= 0:
        measurement_result["reason"] = "non-positive peak"
        return measurement_result
    core_mask = (contrast_bump >= 0.7 * peak_contrast) & _disk_mask((sz, sz), patch_center_x, patch_center_y, max(3 * mole_radius_mm * pixels_per_mm, crop_half_size_px))
    if core_mask.sum() >= 3:
        yy, xx = np.nonzero(core_mask)
        patch_center_x, patch_center_y = float(xx.mean()), float(yy.mean())

    mask, contour, region_area_pixels = _iso_region(contrast_bump, iso_fraction * peak_contrast, patch_center_x, patch_center_y, pixels_per_mm)
    if mask is None or region_area_pixels < 3 or contour is None or len(contour) < 5:
        measurement_result["reason"] = "no iso-region"
        return measurement_result

    region_area_mm2 = region_area_pixels / pixels_per_mm**2
    equivalent_diameter_mm = 2.0 * math.sqrt(region_area_mm2 / math.pi)
    region_pixel_y, region_pixel_x = np.nonzero(mask)
    centered_region_points = np.stack([region_pixel_x - region_pixel_x.mean(), region_pixel_y - region_pixel_y.mean()], 1).astype(np.float64)
    region_covariance = (centered_region_points.T @ centered_region_points) / len(centered_region_points)
    region_eigenvalues = np.sort(np.linalg.eigvalsh(region_covariance))[::-1]  # lam1 >= lam2
    region_eigenvalues = np.clip(region_eigenvalues, 1e-6, None)
    region_eccentricity = math.sqrt(max(0.0, 1.0 - region_eigenvalues[1] / region_eigenvalues[0]))
    major_axis_length_mm, minor_axis_length_mm = 4 * math.sqrt(region_eigenvalues[0]) / pixels_per_mm, 4 * math.sqrt(region_eigenvalues[1]) / pixels_per_mm
    region_perimeter_mm = cv2.arcLength(contour, True) / pixels_per_mm
    region_circularity = 4 * math.pi * region_area_mm2 / max(region_perimeter_mm**2, 1e-9)
    boundary_irregularity = 1.0 / max(region_circularity, 1e-6) - 1.0

    # boundary-localization sigma from the local melanin noise band
    annulus = (
        _disk_mask((sz, sz), patch_center_x, patch_center_y, baseline_outer_radius_px)
        & ~_disk_mask((sz, sz), patch_center_x, patch_center_y, baseline_inner_radius_px)
        & (patch_coverage > 0)
        & ~other_mole_exclusion_mask
    )
    boundary_contrast_sigma = (
        1.4826 * np.median(np.abs(contrast_bump[annulus] - np.median(contrast_bump[annulus])))
        if annulus.sum() >= 20
        else 0.02
    )
    gradient_y, gradient_x = np.gradient(contrast_bump)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    contour_points = contour.reshape(-1, 2)
    mean_boundary_gradient = float(np.mean(gradient_magnitude[contour_points[:, 1], contour_points[:, 0]]))
    mean_boundary_gradient = max(mean_boundary_gradient, 1e-4)
    boundary_radius_sigma_mm = math.sqrt((boundary_contrast_sigma / mean_boundary_gradient) ** 2 + (iso_fraction * boundary_contrast_sigma / mean_boundary_gradient) ** 2) / pixels_per_mm
    diameter_measurement_sigma_mm = 2 * boundary_radius_sigma_mm
    area_measurement_sigma_mm = region_perimeter_mm * boundary_radius_sigma_mm

    # validity: region must not touch the eroded-coverage edge or be too ragged
    eroded_coverage = cv2.erode(patch_coverage, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    touches_edge = (mask.astype(bool) & (eroded_coverage == 0)).any()
    convex_hull = cv2.convexHull(contour)
    region_solidity = region_area_pixels / max(cv2.contourArea(convex_hull), 1.0)
    measurement_valid = (not touches_edge) and region_solidity >= 0.7 and equivalent_diameter_mm >= 0.5

    measurement_result.update(
        valid=bool(measurement_valid),
        d_eq_mm=equivalent_diameter_mm,
        area_mm2=region_area_mm2,
        eccentricity=region_eccentricity,
        major_mm=major_axis_length_mm,
        minor_mm=minor_axis_length_mm,
        border_irregularity=boundary_irregularity,
        perim_mm=region_perimeter_mm,
        peak=peak_contrast,
        center=(int(round(x0 + patch_center_x)), int(round(y0 + patch_center_y))),
        sigma_diam_meas=diameter_measurement_sigma_mm,
        sigma_area_meas=area_measurement_sigma_mm,
        sigma_m=boundary_contrast_sigma,
        gbar=mean_boundary_gradient,
        n_pixels=region_area_pixels,
        reason=("" if measurement_valid else ("edge" if touches_edge else f"solidity {region_solidity:.2f}")),
    )
    return measurement_result


# --------------------------------------------------------------------------- #
# Uncertainty model
# --------------------------------------------------------------------------- #
def load_mole_change_calibration(path=None):
    """Same-scan-null calibration {k, floor_diam, floor_area}; conservative
    defaults if no calibration file exists."""
    import json

    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "null_calibration.json")
    if os.path.exists(path):
        distance = json.load(open(path))
        distance.setdefault("uncalibrated", False)
        return distance
    return dict(k=2.5, floor_diam=0.30, floor_area=None, uncalibrated=True)


def estimate_mole_detection_position_sigma(mel, cov, mole, pixels_per_mm, jittered_sample_count=24, random_seed=0):
    """Detector-repeatability sigma: std of the size measurement over N rigidly-
    jittered, noise-added copies of the mole's own melanin_crop. Uses the scan's own
    robust skin noise, so a fainter/brighter scan gets the appropriately larger
    error bar (the exposure robustness, expressed as uncertainty)."""
    covered_skin_mask = cov > 0
    skin_noise_sigma = 1.4826 * np.median(np.abs(mel[covered_skin_mask] - np.median(mel[covered_skin_mask]))) if covered_skin_mask.any() else 0.02
    mole_radius_mm = max(mole["radius_mm"], 0.3)
    crop_half_size_px = int(math.ceil(max(3 * mole_radius_mm, 4.0) * pixels_per_mm)) + int(math.ceil(2 * pixels_per_mm))
    image_height, image_width = mel.shape
    patch_center_x, patch_center_y = int(round(mole["x"])), int(round(mole["y"]))
    crop_origin_x, crop_origin_y = patch_center_x - crop_half_size_px, patch_center_y - crop_half_size_px
    melanin_crop = np.zeros((2 * crop_half_size_px, 2 * crop_half_size_px), np.float32)
    coverage_crop = np.zeros((2 * crop_half_size_px, 2 * crop_half_size_px), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, crop_origin_x), max(0, crop_origin_y), min(image_width, patch_center_x + crop_half_size_px), min(image_height, patch_center_y + crop_half_size_px)
    melanin_crop[sy0 - crop_origin_y : sy1 - crop_origin_y, sx0 - crop_origin_x : sx1 - crop_origin_x] = mel[sy0:sy1, sx0:sx1]
    coverage_crop[sy0 - crop_origin_y : sy1 - crop_origin_y, sx0 - crop_origin_x : sx1 - crop_origin_x] = cov[sy0:sy1, sx0:sx1]
    random_generator = np.random.default_rng(random_seed)
    measurement_names = ("d_eq_mm", "area_mm2", "eccentricity", "border_irregularity")
    measurement_samples = {measurement_name: [] for measurement_name in measurement_names}
    patch_center_px = crop_half_size_px
    synthetic_mole = dict(x=patch_center_px, y=patch_center_px, radius_mm=mole_radius_mm)
    for iteration_index in range(jittered_sample_count):
        translation_jitter_x, translation_jitter_y = random_generator.uniform(-0.5, 0.5, 2)
        rotation_angle_degrees = random_generator.uniform(-1, 1)
        affine_warp_matrix = cv2.getRotationMatrix2D((patch_center_px, patch_center_px), rotation_angle_degrees, 1.0)
        affine_warp_matrix[0, 2] += translation_jitter_x
        affine_warp_matrix[1, 2] += translation_jitter_y
        warped_melanin_crop = cv2.warpAffine(
            melanin_crop, affine_warp_matrix, (2 * crop_half_size_px, 2 * crop_half_size_px), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        warped_coverage_crop = cv2.warpAffine(coverage_crop, affine_warp_matrix, (2 * crop_half_size_px, 2 * crop_half_size_px), flags=cv2.INTER_NEAREST)
        warped_melanin_crop = warped_melanin_crop + random_generator.normal(0, skin_noise_sigma, warped_melanin_crop.shape).astype(np.float32)
        mole_measurement = measure_mole_size(warped_melanin_crop, warped_coverage_crop, synthetic_mole, pixels_per_mm)
        if mole_measurement["valid"] and np.isfinite(mole_measurement["d_eq_mm"]):
            for measurement_name in measurement_names:
                measurement_samples[measurement_name].append(mole_measurement[measurement_name])
    measurement_sigmas = {
        "sigma_" + measurement_name: (float(np.std(measurement_samples[measurement_name])) if len(measurement_samples[measurement_name]) >= 5 else float("nan")) for measurement_name in measurement_names
    }
    measurement_sigmas["n"] = len(measurement_samples["d_eq_mm"])
    return measurement_sigmas


def build_mole_alignment_residual_field(corr, pixels_per_mm):
    """Smooth field: A-frame position (px) -> registration sigma (mm), from the
    winning alignment's per-correspondence residual. RBF when SIFT-dense & wide;
    IDW for the sparse mole-constellation; flat 1mm when uncorroborated."""
    if not corr or len(corr.get("dst_px", [])) < 3:
        return lambda points: np.full(len(np.atleast_2d(points)), 1.0)
    src = np.asarray(corr["dst_px"], float)
    res = np.asarray(corr["resid_mm"], float)
    spread = float(np.hypot(*(src.max(0) - src.min(0))) / pixels_per_mm)
    if len(src) >= 8 and spread >= 40:
        score = float(np.median(res)) ** 2 * len(src) + 1e-6
        rbf = RBFInterpolator(src, res, kernel="thin_plate_spline", smoothing=score)
        return lambda points: np.clip(rbf(np.atleast_2d(points)), 0, None)

    def idw(points):
        points = np.atleast_2d(points)
        estimated_uncertainties = []
        for point in points:
            d2 = ((src - point) ** 2).sum(1)
            weight = 1.0 / (d2 + (5 * pixels_per_mm) ** 2)
            estimated_uncertainties.append(math.sqrt(max(0.0, (weight * res**2).sum() / weight.sum())))
        return np.array(estimated_uncertainties)

    return idw


def combine_mole_measurements_and_flag_change(sa, sb, da, db, sigma_pos_mm, calib):
    """Combine registration + detector + null-floor sigmas in quadrature; classify
    grew/shrank/stable with a significance gate (|z|>=k AND |delta|>=floor)."""
    item_index = calib.get("k", 2.5)
    floor_d = calib.get("floor_diam", 0.30)
    d_ref = max(sa["d_eq_mm"] if np.isfinite(sa["d_eq_mm"]) else 1.0, 0.5)
    floor_a = calib.get("floor_area") or (math.pi * d_ref / 2) * floor_d
    sig_reg_d = math.sqrt(2) * sigma_pos_mm
    sig_reg_a = (math.pi * d_ref / 2) * sig_reg_d

    def quadrature_sum(*uncertainty_components):
        finite_component_squares = (
            (component or 0.0) ** 2
            for component in uncertainty_components
            if component is None or np.isfinite(component)
        )
        return math.sqrt(sum(finite_component_squares))

    sig_d = quadrature_sum(sig_reg_d, da.get("sigma_d_eq_mm"), db.get("sigma_d_eq_mm"), floor_d)
    sig_a = quadrature_sum(sig_reg_a, da.get("sigma_area_mm2"), db.get("sigma_area_mm2"), floor_a)
    valid = bool(sa["valid"] and sb["valid"])
    dd = sb["d_eq_mm"] - sa["d_eq_mm"]
    dA = sb["area_mm2"] - sa["area_mm2"]
    world_z = dd / sig_d if sig_d > 0 else 0.0
    sig = bool(valid and abs(world_z) >= item_index and abs(dd) >= floor_d)
    cls = (
        "uncertain"
        if not valid
        else ("grew" if sig and dd > 0 else "shrank" if sig and dd < 0 else "stable")
    )
    return dict(
        d_eq_a_mm=round(sa["d_eq_mm"], 2),
        d_eq_b_mm=round(sb["d_eq_mm"], 2),
        delta_diam_mm=round(dd, 2),
        sigma_diam_mm=round(sig_d, 2),
        z_diam=round(world_z, 1),
        area_a_mm2=round(sa["area_mm2"], 2),
        area_b_mm2=round(sb["area_mm2"], 2),
        delta_area_mm2=round(dA, 2),
        sigma_area_mm2=round(sig_a, 2),
        ecc_a=round(sa["eccentricity"], 2),
        ecc_b=round(sb["eccentricity"], 2),
        border_a=round(sa["border_irregularity"], 2),
        border_b=round(sb["border_irregularity"], 2),
        sigma_reg_mm=round(sigma_pos_mm, 2),
        significant=sig,
        valid=valid,
        **{"class": cls},
    )


def assign_stable_mole_lesion_id(uv):
    return f"M{int(round(uv[0]))}_{int(round(uv[1]))}"


def measure_longitudinal_mole_changes(cc, melA, melB, detA, detB, molesA, molesB, pairs, al, pixels_per_mm, calib=None):
    """Per matched pair: measure size in each scan's OWN un-warped frame, attach
    registration + detector uncertainty, classify change. Returns (list, calib)."""
    calib = calib or load_mole_change_calibration()
    field = build_mole_alignment_residual_field(al.get("corr"), pixels_per_mm)
    out = []
    for idx, point in enumerate(pairs):
        mA, mB = molesA[point["iA"]], molesB[point["iB"]]
        sa = measure_mole_size(melA, detA, mA, pixels_per_mm, other_mole_detections=molesA)
        sb = measure_mole_size(melB, detB, mB, pixels_per_mm, other_mole_detections=molesB)
        da = estimate_mole_detection_position_sigma(melA, detA, mA, pixels_per_mm, random_seed=1 + idx)
        db = estimate_mole_detection_position_sigma(melB, detB, mB, pixels_per_mm, random_seed=1001 + idx)
        sigma_pos = float(field(np.array([[mA["x"], mA["y"]]]))[0])
        ch = combine_mole_measurements_and_flag_change(sa, sb, da, db, sigma_pos, calib)
        texture_u = round(cc["umin"] + mA["x"] / pixels_per_mm, 1)
        texture_v = round(cc["vmin"] + mA["y"] / pixels_per_mm, 1)
        ch["id"] = assign_stable_mole_lesion_id((texture_u, texture_v))
        ch["uv_a_mm"] = [texture_u, texture_v]
        ch["resid_mm"] = round(point["resid_mm"], 2)
        out.append(ch)
    return out, calib


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
_CLASS_BGR = {
    "stable": (0, 200, 0),
    "grew": (0, 140, 255),
    "shrank": (255, 200, 0),
    "uncertain": (160, 160, 160),
    "new": (0, 0, 255),
    "disappeared": (255, 80, 0),
}
_IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def write_mole_change_results_csv(path, cc, changes, molesB, molesA, newJ, disJ):
    import csv

    pp = cc["ppmm"]
    with open(path, "w", newline="") as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(
            [
                "id",
                "type",
                "u_mm",
                "v_mm",
                "d_eq_a_mm",
                "d_eq_b_mm",
                "delta_diam_mm",
                "sigma_diam_mm",
                "z_diam",
                "significant",
                "class",
            ]
        )
        for candidate in changes:
            csv_writer.writerow(
                [
                    candidate["id"],
                    candidate["class"],
                    candidate["uv_a_mm"][0],
                    candidate["uv_a_mm"][1],
                    candidate["d_eq_a_mm"],
                    candidate["d_eq_b_mm"],
                    candidate["delta_diam_mm"],
                    candidate["sigma_diam_mm"],
                    candidate["z_diam"],
                    candidate["significant"],
                    candidate["class"],
                ]
            )
        for neighbor_index in newJ:
            texture_u = round(cc["umin"] + molesB[neighbor_index]["x"] / pp, 1)
            texture_v = round(cc["vmin"] + molesB[neighbor_index]["y"] / pp, 1)
            csv_writer.writerow(
                [
                    assign_stable_mole_lesion_id((texture_u, texture_v)),
                    "new",
                    texture_u,
                    texture_v,
                    "",
                    round(molesB[neighbor_index]["diam_mm"], 2),
                    "",
                    "",
                    "",
                    True,
                    "new",
                ]
            )
        for index in disJ:
            texture_u = round(cc["umin"] + molesA[index]["x"] / pp, 1)
            texture_v = round(cc["vmin"] + molesA[index]["y"] / pp, 1)
            csv_writer.writerow(
                [
                    assign_stable_mole_lesion_id((texture_u, texture_v)),
                    "disappeared",
                    texture_u,
                    texture_v,
                    round(molesA[index]["diam_mm"], 2),
                    "",
                    "",
                    "",
                    "",
                    True,
                    "disappeared",
                ]
            )


def render_mole_change_overlay(
    out,
    cc,
    mutual,
    changes,
    molesA,
    molesB,
    pairs,
    newJ,
    disJ,
    transform_matrix=None,
    pixels_per_mm=None,
    labels=("A", "B"),
    **legacy_options,
):
    transform_matrix = legacy_options.pop("T", transform_matrix)
    pixels_per_mm = legacy_options.pop("ppmm", pixels_per_mm)
    if legacy_options:
        unexpected_option = next(iter(legacy_options))
        raise TypeError(f"render_mole_change_overlay got an unexpected keyword argument {unexpected_option!r}")
    if pixels_per_mm is None:
        raise TypeError("render_mole_change_overlay requires pixels_per_mm")
    base = cc["texA"].copy()
    dim = (base.astype(np.float32) * 0.45).astype(np.uint8)
    canvas = np.where(mutual[..., None].astype(bool), base, dim)
    for candidate, point in zip(changes, pairs):
        mA = molesA[point["iA"]]
        col = _CLASS_BGR.get(candidate["class"], (0, 200, 0))
        ra = int(max(candidate["d_eq_a_mm"], 0.5) / 2 * pixels_per_mm) + 2
        rb = int(max(candidate["d_eq_b_mm"], 0.5) / 2 * pixels_per_mm) + 2
        cv2.circle(canvas, (mA["x"], mA["y"]), ra, col, 2)  # baseline size
        cv2.circle(canvas, (mA["x"], mA["y"]), rb, col, 1)  # follow-up size
        tag = f"{candidate['delta_diam_mm']:+.1f}mm" if candidate["class"] in ("grew", "shrank") else candidate["class"][:4]
        star = "*" if candidate["significant"] else ""
        cv2.putText(
            canvas,
            f"{tag}{star}",
            (mA["x"] + rb + 4, mA["y"]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            col,
            2,
            cv2.LINE_AA,
        )
    for neighbor_index in newJ:
        transform = np.asarray(transform_matrix) if transform_matrix is not None else _IDENTITY
        xy = (
            np.asarray([[molesB[neighbor_index]["x"], molesB[neighbor_index]["y"]]], float) @ transform[:, :2].T
        ) + transform[:, 2]
        aligned_pixel_x, aligned_pixel_y = int(xy[0, 0]), int(xy[0, 1])
        cv2.circle(canvas, (aligned_pixel_x, aligned_pixel_y), int(molesB[neighbor_index]["radius_mm"] * pixels_per_mm) + 6, _CLASS_BGR["new"], 3)
    for index in disJ:
        cv2.circle(
            canvas,
            (molesA[index]["x"], molesA[index]["y"]),
            int(molesA[index]["radius_mm"] * pixels_per_mm) + 6,
            _CLASS_BGR["disappeared"],
            3,
        )
    cv2.putText(
        canvas,
        f"{labels[0]}->{labels[1]}:  green=stable  orange=grew  "
        "cyan=shrank  red=new  blue=gone  (*=significant; inner=baseline, "
        "outer=follow-up size)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(os.path.join(out, "change_overlay.png"), canvas)


def _crop(img, cx, cy, half):
    image_height, image_width = img.shape[:2]
    crop = np.full((2 * half, 2 * half, 3), (40, 40, 40), np.uint8)
    x0, y0 = int(cx) - half, int(cy) - half
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(image_width, x0 + 2 * half), min(image_height, y0 + 2 * half)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return crop


def render_mole_change_montage(
    out, cc, changes, molesA, molesB, pairs, transform_matrix, pixels_per_mm, crop_mm=12.0, labels=("A", "B")
):
    half = int(crop_mm / 2 * pixels_per_mm)
    order = sorted(range(len(changes)), key=lambda index: -abs(changes[index]["z_diam"]))
    rows = []
    for index in order:
        candidate, point = changes[index], pairs[index]
        mA, mB = molesA[point["iA"]], molesB[point["iB"]]
        ca = _crop(cc["texA"], mA["x"], mA["y"], half)
        cb = _crop(cc["texB"], mB["x"], mB["y"], half)
        col = _CLASS_BGR.get(candidate["class"], (0, 200, 0))
        cv2.circle(ca, (half, half), int(max(candidate["d_eq_a_mm"], 0.5) / 2 * pixels_per_mm), col, 1)
        cv2.circle(cb, (half, half), int(max(candidate["d_eq_b_mm"], 0.5) / 2 * pixels_per_mm), col, 1)
        row = np.hstack([ca, np.full((2 * half, 6, 3), 30, np.uint8), cb])
        bar = np.full((26, row.shape[1], 3), 30, np.uint8)
        star = " *" if candidate["significant"] else ""
        cv2.putText(
            bar,
            f"{candidate['id']} {candidate['uv_a_mm']}mm  d:{candidate['d_eq_a_mm']}->"
            f"{candidate['d_eq_b_mm']}mm  d{candidate['delta_diam_mm']:+.2f}+-"
            f"{candidate['sigma_diam_mm']}mm  {candidate['class'].upper()}{star}",
            (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.vstack([bar, row]))
    if not rows:
        return
    hdr = np.full((28, rows[0].shape[1], 3), 55, np.uint8)
    cv2.putText(
        hdr,
        f"SIZE CHANGE  LEFT={labels[0]}  RIGHT={labels[1]}  (circle = half-max equiv. diameter)",
        (6, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(os.path.join(out, "change_montage.png"), np.vstack([hdr] + rows))
