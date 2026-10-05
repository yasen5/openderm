"""Shared lesion-channel processing and detection."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter


def melanin_flat(texture_bgr, coverage_mask, pixels_per_mm, background_scale_mm=12.0):
    """Melanin proxy M=-log(R), high-passed by a >=2x-largest-blob background.

    The background is estimated from COVERED skin only (normalized/masked blur),
    so the uncovered fill and coverage-edge wedges don't bias the flatten."""
    red_channel = texture_bgr[:, :, 2].astype(np.float32)  # BGR -> red channel
    negative_log_red = -np.log(np.maximum(red_channel, 1.0))
    covered_pixels = (coverage_mask > 0).astype(np.float32)
    downsample_factor = 8
    background_sigma_pixels = max(1.0, background_scale_mm * pixels_per_mm / downsample_factor)
    image_height, image_width = negative_log_red.shape
    reduced_red = cv2.resize(negative_log_red * covered_pixels, (image_width // downsample_factor, image_height // downsample_factor), interpolation=cv2.INTER_AREA)
    reduced_coverage = cv2.resize(covered_pixels, (image_width // downsample_factor, image_height // downsample_factor), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(reduced_red, (0, 0), background_sigma_pixels) / (cv2.GaussianBlur(reduced_coverage, (0, 0), background_sigma_pixels) + 1e-6)
    bg = cv2.resize(bg, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    out = negative_log_red - bg
    out[coverage_mask == 0] = 0.0
    return out  # moles -> positive bumps


def hemoglobin_flat(texture_bgr, coverage_mask, pixels_per_mm, background_scale_mm=12.0):
    """Erythema/hemoglobin proxy He = log(R) - log(G) = log(R/G), high-passed the
    same masked way as melanin_flat. Cherry angiomas (red: high R, low G) are
    POSITIVE bumps; brown moles are not. Being a log-RATIO, a multiplicative
    exposure/white-balance gain cancels -> exposure-invariant like melanin."""
    red_channel = texture_bgr[:, :, 2].astype(np.float32)
    green_channel = texture_bgr[:, :, 1].astype(np.float32)
    hemoglobin_log_ratio = np.log(np.maximum(red_channel, 1.0)) - np.log(np.maximum(green_channel, 1.0))
    covered_pixels = (coverage_mask > 0).astype(np.float32)
    downsample_factor = 8
    background_sigma_pixels = max(1.0, background_scale_mm * pixels_per_mm / downsample_factor)
    image_height, image_width = hemoglobin_log_ratio.shape
    reduced_ratio = cv2.resize(hemoglobin_log_ratio * covered_pixels, (image_width // downsample_factor, image_height // downsample_factor), interpolation=cv2.INTER_AREA)
    reduced_coverage = cv2.resize(covered_pixels, (image_width // downsample_factor, image_height // downsample_factor), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(reduced_ratio, (0, 0), background_sigma_pixels) / (cv2.GaussianBlur(reduced_coverage, (0, 0), background_sigma_pixels) + 1e-6)
    bg = cv2.resize(bg, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    out = hemoglobin_log_ratio - bg
    out[coverage_mask == 0] = 0.0
    return out


def detect_angiomas(
    texture_bgr,
    hemoglobin_channel,
    melanin_channel,
    coverage_mask,
    pixels_per_mm,
    diam_mm=(0.4, 4.0),
    min_contrast=0.08,
    k_sigma=6.0,
    mel_max=0.07,
    hue_max=8,
    sat_min=50,
):
    """Detect cherry angiomas: blobs in the hemoglobin channel that are (a) NOT
    dark (melanin bump < mel_max -- the red channel isn't suppressed, so brown
    moles are rejected) and (b) genuinely RED in hue (excludes tan/orange moles
    that also raise R/G). Returns the same dict shape as detect_moles."""
    candidates = detect_moles(hemoglobin_channel, coverage_mask, pixels_per_mm, diam_mm, min_contrast, k_sigma)
    smoothed_melanin = gaussian_filter(melanin_channel, 0.35 * pixels_per_mm)
    hsv = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2HSV)
    angiomas = []
    for candidate in candidates:
        pixel_x, pixel_y = candidate["x"], candidate["y"]
        if smoothed_melanin[pixel_y, pixel_x] >= mel_max:  # dark -> brown mole, not angioma
            continue
        sample_radius_px = max(int(candidate["radius_mm"] * pixels_per_mm), 3)
        disk = hsv[
            max(0, pixel_y - sample_radius_px) : pixel_y + sample_radius_px,
            max(0, pixel_x - sample_radius_px) : pixel_x + sample_radius_px,
        ].reshape(-1, 3)
        hue = float(np.median(disk[:, 0]))  # OpenCV hue 0..180
        sat = float(np.median(disk[:, 1]))
        if (hue <= hue_max or hue >= 180 - hue_max) and sat >= sat_min:
            angiomas.append(candidate)
    return angiomas


def mel_threshold(mel, mask, min_contrast, k_sigma):
    """Per-scan adaptive melanin-contrast threshold (k robust-sigmas above this
    scan's own skin noise, floored at min_contrast)."""
    covered_melanin_values = mel[mask > 0]
    sigma = 1.4826 * np.median(np.abs(covered_melanin_values - np.median(covered_melanin_values)))
    return max(min_contrast, k_sigma * float(sigma))


def detect_moles(
    melanin_image,
    coverage_mask,
    pixels_per_mm,
    diameter_range_mm=(0.4, 5.0),
    minimum_contrast=0.08,
    robust_sigma_multiplier=6.0,
    presmoothing_width_mm=0.35,
    maximum_candidate_count=400,
):
    """Multiscale-LoG blob detection on the flattened melanin map.

    The LoG locates candidate spots and their scale. The keep decision is a
    PER-SCAN ADAPTIVE threshold: a mole must stand k_sigma robust-sigmas above
    that scan's own skin-noise floor (with a small absolute floor). This is
    essential because there is no cross-scan white-balance -- a brighter capture
    compresses absolute melanin contrast, so a fixed threshold would detect far
    fewer moles there and fabricate spurious 'disappeared' calls.

    A light presmooth (< smallest mole) is applied first so per-pixel texture
    noise -- which varies between scans/renders -- doesn't inflate the robust
    sigma and make the detection count swing wildly with the threshold."""
    if presmoothing_width_mm > 0:
        melanin_image = gaussian_filter(melanin_image, presmoothing_width_mm * pixels_per_mm)
    minimum_gaussian_scale_px = diameter_range_mm[0] / 2 * pixels_per_mm / np.sqrt(2)
    maximum_gaussian_scale_px = diameter_range_mm[1] / 2 * pixels_per_mm / np.sqrt(2)
    gaussian_scales_px = np.geomspace(minimum_gaussian_scale_px, maximum_gaussian_scale_px, 8)
    log_response_by_scale = np.stack([-gaussian_laplace(melanin_image, gaussian_scale) * gaussian_scale**2 for gaussian_scale in gaussian_scales_px])
    maximum_log_response = log_response_by_scale.max(0)
    winning_scale_indices = log_response_by_scale.argmax(0)
    covered_pixel_mask = coverage_mask > 0
    contrast_threshold = mel_threshold(melanin_image, coverage_mask, minimum_contrast, robust_sigma_multiplier)
    local_maximum_window = int(minimum_gaussian_scale_px * 2) | 1
    candidate_peak_mask = (maximum_log_response == maximum_filter(maximum_log_response, size=max(3, local_maximum_window))) & covered_pixel_mask & (maximum_log_response > 0)
    candidate_pixel_y, candidate_pixel_x = np.where(candidate_peak_mask)
    # rank by scale-normalized LoG response (peaks at the blob's true scale), so a
    # large mole is accepted before the sub-structure/noise blobs sitting on it.
    ranked_candidates = sorted(zip(maximum_log_response[candidate_pixel_y, candidate_pixel_x], candidate_pixel_x, candidate_pixel_y), reverse=True)
    detected_moles = []
    suppression_mask = np.zeros(coverage_mask.shape, np.uint8)
    for peak_response, pixel_x, pixel_y in ranked_candidates:
        melanin_contrast = float(melanin_image[pixel_y, pixel_x])
        if melanin_contrast < contrast_threshold or suppression_mask[pixel_y, pixel_x]:
            continue
        radius_px = gaussian_scales_px[winning_scale_indices[pixel_y, pixel_x]] * np.sqrt(2)
        diameter_mm = 2 * radius_px / pixels_per_mm
        if not (diameter_range_mm[0] <= diameter_mm <= diameter_range_mm[1]):
            continue
        # suppress everything within this blob's radius (>= a 2.5mm floor so a
        # single lesion isn't reported as a cluster of sub-detections)
        suppression_radius_px = int(max(radius_px * 1.5, 2.5 * pixels_per_mm)) + 1
        cv2.circle(suppression_mask, (int(pixel_x), int(pixel_y)), suppression_radius_px, 1, -1)
        detected_moles.append(
            dict(
                x=int(pixel_x),
                y=int(pixel_y),
                radius_mm=float(radius_px / pixels_per_mm),
                diam_mm=float(diameter_mm),
                area_mm2=float(np.pi * (radius_px / pixels_per_mm) ** 2),
                contrast=melanin_contrast,
                response=float(peak_response),
            )
        )
        if len(detected_moles) >= maximum_candidate_count:
            break
    return detected_moles


def present_in_other(other_melanin_image, feature_pixel_xy, pixels_per_mm, other_threshold, search_radius_mm=2.5, required_fraction=0.5):
    """Is there a (possibly sub-threshold) pigment bump at xy in the other scan?
    Guards new/disappeared against detector misses and cross-scan exposure
    differences: a mole below the other scan's threshold is NOT 'disappeared'."""
    pixel_x, pixel_y = int(round(feature_pixel_xy[0])), int(round(feature_pixel_xy[1]))
    search_radius_px = int(search_radius_mm * pixels_per_mm)
    image_height, image_width = other_melanin_image.shape
    if not (0 <= pixel_x < image_width and 0 <= pixel_y < image_height):
        return False
    patch = other_melanin_image[
        max(0, pixel_y - search_radius_px) : pixel_y + search_radius_px + 1,
        max(0, pixel_x - search_radius_px) : pixel_x + search_radius_px + 1,
    ]
    return patch.size > 0 and float(patch.max()) >= required_fraction * other_threshold
