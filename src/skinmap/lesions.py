"""Shared lesion-channel processing and detection."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter


def melanin_flat(tex_bgr, cov, ppmm, bg_mm=12.0):
    """Melanin proxy M=-log(R), high-passed by a >=2x-largest-blob background.

    The background is estimated from COVERED skin only (normalized/masked blur),
    so the uncovered fill and coverage-edge wedges don't bias the flatten."""
    R = tex_bgr[:, :, 2].astype(np.float32)  # BGR -> red channel
    M = -np.log(np.maximum(R, 1.0))
    m = (cov > 0).astype(np.float32)
    ds = 8
    sig = max(1.0, bg_mm * ppmm / ds)
    h, w = M.shape
    Ms = cv2.resize(M * m, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(m, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(Ms, (0, 0), sig) / (cv2.GaussianBlur(ms, (0, 0), sig) + 1e-6)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)
    out = M - bg
    out[cov == 0] = 0.0
    return out  # moles -> positive bumps


def hemoglobin_flat(tex_bgr, cov, ppmm, bg_mm=12.0):
    """Erythema/hemoglobin proxy He = log(R) - log(G) = log(R/G), high-passed the
    same masked way as melanin_flat. Cherry angiomas (red: high R, low G) are
    POSITIVE bumps; brown moles are not. Being a log-RATIO, a multiplicative
    exposure/white-balance gain cancels -> exposure-invariant like melanin."""
    R = tex_bgr[:, :, 2].astype(np.float32)
    G = tex_bgr[:, :, 1].astype(np.float32)
    He = np.log(np.maximum(R, 1.0)) - np.log(np.maximum(G, 1.0))
    m = (cov > 0).astype(np.float32)
    ds = 8
    sig = max(1.0, bg_mm * ppmm / ds)
    h, w = He.shape
    Hs = cv2.resize(He * m, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(m, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(Hs, (0, 0), sig) / (cv2.GaussianBlur(ms, (0, 0), sig) + 1e-6)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)
    out = He - bg
    out[cov == 0] = 0.0
    return out


def detect_angiomas(
    tex_bgr,
    hem,
    mel,
    mask,
    ppmm,
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
    cand = detect_moles(hem, mask, ppmm, diam_mm, min_contrast, k_sigma)
    mel_s = gaussian_filter(mel, 0.35 * ppmm)
    hsv = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2HSV)
    out = []
    for a in cand:
        x, y = a["x"], a["y"]
        if mel_s[y, x] >= mel_max:  # dark -> brown mole, not angioma
            continue
        r = max(int(a["radius_mm"] * ppmm), 3)
        disk = hsv[max(0, y - r) : y + r, max(0, x - r) : x + r].reshape(-1, 3)
        hue = float(np.median(disk[:, 0]))  # OpenCV hue 0..180
        sat = float(np.median(disk[:, 1]))
        if (hue <= hue_max or hue >= 180 - hue_max) and sat >= sat_min:
            out.append(a)
    return out


def mel_threshold(mel, mask, min_contrast, k_sigma):
    """Per-scan adaptive melanin-contrast threshold (k robust-sigmas above this
    scan's own skin noise, floored at min_contrast)."""
    v = mel[mask > 0]
    sigma = 1.4826 * np.median(np.abs(v - np.median(v)))
    return max(min_contrast, k_sigma * float(sigma))


def detect_moles(
    mel,
    mask,
    ppmm,
    diam_mm=(0.4, 5.0),
    min_contrast=0.08,
    k_sigma=6.0,
    presmooth_mm=0.35,
    max_keep=400,
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
    if presmooth_mm > 0:
        mel = gaussian_filter(mel, presmooth_mm * ppmm)
    sig_lo = diam_mm[0] / 2 * ppmm / np.sqrt(2)
    sig_hi = diam_mm[1] / 2 * ppmm / np.sqrt(2)
    sigmas = np.geomspace(sig_lo, sig_hi, 8)
    resp = np.stack([-gaussian_laplace(mel, s) * s**2 for s in sigmas])
    peak = resp.max(0)
    scl = resp.argmax(0)
    inside = mask > 0
    thr = mel_threshold(mel, mask, min_contrast, k_sigma)
    win = int(sig_lo * 2) | 1
    loc = (peak == maximum_filter(peak, size=max(3, win))) & inside & (peak > 0)
    ys, xs = np.where(loc)
    # rank by scale-normalized LoG response (peaks at the blob's true scale), so a
    # large mole is accepted before the sub-structure/noise blobs sitting on it.
    cand = sorted(zip(peak[ys, xs], xs, ys), reverse=True)
    moles = []
    taken = np.zeros(mask.shape, np.uint8)
    for resp_val, x, y in cand:
        contrast = float(mel[y, x])
        if contrast < thr or taken[y, x]:
            continue
        r_px = sigmas[scl[y, x]] * np.sqrt(2)
        d_mm = 2 * r_px / ppmm
        if not (diam_mm[0] <= d_mm <= diam_mm[1]):
            continue
        # suppress everything within this blob's radius (>= a 2.5mm floor so a
        # single lesion isn't reported as a cluster of sub-detections)
        sup = int(max(r_px * 1.5, 2.5 * ppmm)) + 1
        cv2.circle(taken, (int(x), int(y)), sup, 1, -1)
        moles.append(
            dict(
                x=int(x),
                y=int(y),
                radius_mm=float(r_px / ppmm),
                diam_mm=float(d_mm),
                area_mm2=float(np.pi * (r_px / ppmm) ** 2),
                contrast=contrast,
                response=float(resp_val),
            )
        )
        if len(moles) >= max_keep:
            break
    return moles


def present_in_other(mel_other, xy, ppmm, thr_other, win_mm=2.5, frac=0.5):
    """Is there a (possibly sub-threshold) pigment bump at xy in the other scan?
    Guards new/disappeared against detector misses and cross-scan exposure
    differences: a mole below the other scan's threshold is NOT 'disappeared'."""
    x, y = int(round(xy[0])), int(round(xy[1]))
    r = int(win_mm * ppmm)
    h, w = mel_other.shape
    if not (0 <= x < w and 0 <= y < h):
        return False
    patch = mel_other[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1]
    return patch.size > 0 and float(patch.max()) >= frac * thr_other
