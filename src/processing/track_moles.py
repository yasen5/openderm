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
def _disk_mask(shape, cx, cy, r):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _local_baseline(bump, cov, cx, cy, r_in, r_out, other_mask):
    """Robust skin baseline from an annulus around the mole (inside coverage,
    excluding other moles)."""
    yy, xx = np.ogrid[: bump.shape[0], : bump.shape[1]]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    annulus = (d2 >= r_in * r_in) & (d2 <= r_out * r_out) & (cov > 0) & (~other_mask)
    vals = bump[annulus]
    return float(np.median(vals)) if vals.size >= 20 else 0.0


def _iso_region(bump, level, cx, cy, ppmm):
    """Half-max connected component containing (cx,cy), closed, + its contour."""
    binary = (bump >= level).astype(np.uint8)
    lab, n = label(binary)
    if n == 0:
        return None, None, 0
    cl = lab[int(round(cy)), int(round(cx))]
    if cl == 0:  # center fell below level
        return None, None, 0
    mask = (lab == cl).astype(np.uint8)
    k = max(1, int(round(0.2 * ppmm)))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))
    )
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return mask, None, int(mask.sum())
    contour = max(cnts, key=cv2.contourArea)
    return mask, contour, int(mask.sum())


def measure_mole_size(
    mel, cov, mole, ppmm, frac=0.5, presmooth_mm=0.35, peak_disk_mm=0.5, other_moles=None
):
    """Half-max iso-contour measurement of one mole in one scan's melanin map.

    Returns d_eq_mm, area_mm2, eccentricity, major/minor_mm, border_irregularity,
    perim_mm, peak, center, plus per-measurement sigmas and a `valid` flag."""
    radius_mm = max(mole["radius_mm"], 0.3)
    half = int(math.ceil(max(3 * radius_mm, 4.0) * ppmm))
    H, W = mel.shape
    cx0, cy0 = int(round(mole["x"])), int(round(mole["y"]))
    x0, y0 = cx0 - half, cy0 - half
    x1, y1 = cx0 + half, cy0 + half
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    sz = 2 * half
    patch = np.zeros((sz, sz), np.float32)
    pc = np.zeros((sz, sz), np.uint8)
    patch[pad_t : pad_t + (sy1 - sy0), pad_l : pad_l + (sx1 - sx0)] = mel[sy0:sy1, sx0:sx1]
    pc[pad_t : pad_t + (sy1 - sy0), pad_l : pad_l + (sx1 - sx0)] = cov[sy0:sy1, sx0:sx1]
    cx, cy = half, half  # mole center in patch coords

    # mask out OTHER moles inside the patch (so baseline/region don't merge them)
    other_mask = np.zeros((sz, sz), bool)
    if other_moles:
        for o in other_moles:
            if o is mole:
                continue
            ox, oy = o["x"] - x0, o["y"] - y0
            if -half < ox < sz + half and -half < oy < sz + half:
                other_mask |= _disk_mask((sz, sz), ox, oy, max(o["radius_mm"], 0.3) * ppmm * 1.5)

    r_in, r_out = 2.0 * radius_mm * ppmm, half
    b0 = _local_baseline(patch, pc, cx, cy, r_in, r_out, other_mask)
    bump = gaussian_filter(patch - b0, presmooth_mm * ppmm)

    # robust peak in a small disk at the LoG center; refine center on the core
    pk_disk = _disk_mask((sz, sz), cx, cy, peak_disk_mm * ppmm)
    P = float(np.percentile(bump[pk_disk], 95)) if pk_disk.any() else float(bump.max())
    res = dict(
        valid=False,
        peak=P,
        d_eq_mm=float("nan"),
        area_mm2=float("nan"),
        eccentricity=float("nan"),
        major_mm=float("nan"),
        minor_mm=float("nan"),
        border_irregularity=float("nan"),
        perim_mm=float("nan"),
        center=(cx0, cy0),
        sigma_diam_meas=float("nan"),
        sigma_area_meas=float("nan"),
        sigma_m=float("nan"),
        gbar=float("nan"),
        n_pixels=0,
        reason="",
    )
    if not np.isfinite(P) or P <= 0:
        res["reason"] = "non-positive peak"
        return res
    core = (bump >= 0.7 * P) & _disk_mask((sz, sz), cx, cy, max(3 * radius_mm * ppmm, half))
    if core.sum() >= 3:
        yy, xx = np.nonzero(core)
        cx, cy = float(xx.mean()), float(yy.mean())

    mask, contour, area_px = _iso_region(bump, frac * P, cx, cy, ppmm)
    if mask is None or area_px < 3 or contour is None or len(contour) < 5:
        res["reason"] = "no iso-region"
        return res

    area_mm2 = area_px / ppmm**2
    d_eq_mm = 2.0 * math.sqrt(area_mm2 / math.pi)
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(np.float64)
    covm = (pts.T @ pts) / len(pts)
    lam = np.sort(np.linalg.eigvalsh(covm))[::-1]  # lam1 >= lam2
    lam = np.clip(lam, 1e-6, None)
    ecc = math.sqrt(max(0.0, 1.0 - lam[1] / lam[0]))
    major_mm, minor_mm = 4 * math.sqrt(lam[0]) / ppmm, 4 * math.sqrt(lam[1]) / ppmm
    perim_mm = cv2.arcLength(contour, True) / ppmm
    circ = 4 * math.pi * area_mm2 / max(perim_mm**2, 1e-9)
    border = 1.0 / max(circ, 1e-6) - 1.0

    # boundary-localization sigma from the local melanin noise band
    annulus = (
        _disk_mask((sz, sz), cx, cy, r_out)
        & ~_disk_mask((sz, sz), cx, cy, r_in)
        & (pc > 0)
        & ~other_mask
    )
    sig_m = (
        1.4826 * np.median(np.abs(bump[annulus] - np.median(bump[annulus])))
        if annulus.sum() >= 20
        else 0.02
    )
    gy, gx = np.gradient(bump)
    gmag = np.hypot(gx, gy)
    cpts = contour.reshape(-1, 2)
    gbar = float(np.mean(gmag[cpts[:, 1], cpts[:, 0]]))
    gbar = max(gbar, 1e-4)
    sigma_radius_mm = math.sqrt((sig_m / gbar) ** 2 + (frac * sig_m / gbar) ** 2) / ppmm
    sigma_diam = 2 * sigma_radius_mm
    sigma_area = perim_mm * sigma_radius_mm

    # validity: region must not touch the eroded-coverage edge or be too ragged
    edge = cv2.erode(pc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    touches_edge = (mask.astype(bool) & (edge == 0)).any()
    hull = cv2.convexHull(contour)
    solidity = area_px / max(cv2.contourArea(hull), 1.0)
    valid = (not touches_edge) and solidity >= 0.7 and d_eq_mm >= 0.5

    res.update(
        valid=bool(valid),
        d_eq_mm=d_eq_mm,
        area_mm2=area_mm2,
        eccentricity=ecc,
        major_mm=major_mm,
        minor_mm=minor_mm,
        border_irregularity=border,
        perim_mm=perim_mm,
        peak=P,
        center=(int(round(x0 + cx)), int(round(y0 + cy))),
        sigma_diam_meas=sigma_diam,
        sigma_area_meas=sigma_area,
        sigma_m=sig_m,
        gbar=gbar,
        n_pixels=area_px,
        reason=("" if valid else ("edge" if touches_edge else f"solidity {solidity:.2f}")),
    )
    return res


# --------------------------------------------------------------------------- #
# Uncertainty model
# --------------------------------------------------------------------------- #
def load_calibration(path=None):
    """Same-scan-null calibration {k, floor_diam, floor_area}; conservative
    defaults if no calibration file exists."""
    import json

    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "null_calibration.json")
    if os.path.exists(path):
        d = json.load(open(path))
        d.setdefault("uncalibrated", False)
        return d
    return dict(k=2.5, floor_diam=0.30, floor_area=None, uncalibrated=True)


def detector_sigma(mel, cov, mole, ppmm, n=24, seed=0):
    """Detector-repeatability sigma: std of the size measurement over N rigidly-
    jittered, noise-added copies of the mole's own crop. Uses the scan's own
    robust skin noise, so a fainter/brighter scan gets the appropriately larger
    error bar (the exposure robustness, expressed as uncertainty)."""
    inside = cov > 0
    sig = 1.4826 * np.median(np.abs(mel[inside] - np.median(mel[inside]))) if inside.any() else 0.02
    radius_mm = max(mole["radius_mm"], 0.3)
    half = int(math.ceil(max(3 * radius_mm, 4.0) * ppmm)) + int(math.ceil(2 * ppmm))
    H, W = mel.shape
    cx, cy = int(round(mole["x"])), int(round(mole["y"]))
    x0, y0 = cx - half, cy - half
    crop = np.zeros((2 * half, 2 * half), np.float32)
    cropc = np.zeros((2 * half, 2 * half), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, cx + half), min(H, cy + half)
    crop[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = mel[sy0:sy1, sx0:sx1]
    cropc[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = cov[sy0:sy1, sx0:sx1]
    rng = np.random.default_rng(seed)
    keys = ("d_eq_mm", "area_mm2", "eccentricity", "border_irregularity")
    vals = {k: [] for k in keys}
    cen = half
    m0 = dict(x=cen, y=cen, radius_mm=radius_mm)
    for _ in range(n):
        dx, dy = rng.uniform(-0.5, 0.5, 2)
        th = rng.uniform(-1, 1)
        M = cv2.getRotationMatrix2D((cen, cen), th, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        j = cv2.warpAffine(
            crop, M, (2 * half, 2 * half), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        jc = cv2.warpAffine(cropc, M, (2 * half, 2 * half), flags=cv2.INTER_NEAREST)
        j = j + rng.normal(0, sig, j.shape).astype(np.float32)
        r = measure_mole_size(j, jc, m0, ppmm)
        if r["valid"] and np.isfinite(r["d_eq_mm"]):
            for k in keys:
                vals[k].append(r[k])
    out = {
        "sigma_" + k: (float(np.std(vals[k])) if len(vals[k]) >= 5 else float("nan")) for k in keys
    }
    out["n"] = len(vals["d_eq_mm"])
    return out


def build_residual_field(corr, ppmm):
    """Smooth field: A-frame position (px) -> registration sigma (mm), from the
    winning alignment's per-correspondence residual. RBF when SIFT-dense & wide;
    IDW for the sparse mole-constellation; flat 1mm when uncorroborated."""
    if not corr or len(corr.get("dst_px", [])) < 3:
        return lambda P: np.full(len(np.atleast_2d(P)), 1.0)
    src = np.asarray(corr["dst_px"], float)
    res = np.asarray(corr["resid_mm"], float)
    spread = float(np.hypot(*(src.max(0) - src.min(0))) / ppmm)
    if len(src) >= 8 and spread >= 40:
        s = float(np.median(res)) ** 2 * len(src) + 1e-6
        rbf = RBFInterpolator(src, res, kernel="thin_plate_spline", smoothing=s)
        return lambda P: np.clip(rbf(np.atleast_2d(P)), 0, None)

    def idw(P):
        P = np.atleast_2d(P)
        o = []
        for p in P:
            d2 = ((src - p) ** 2).sum(1)
            w = 1.0 / (d2 + (5 * ppmm) ** 2)
            o.append(math.sqrt(max(0.0, (w * res**2).sum() / w.sum())))
        return np.array(o)

    return idw


def combine_and_flag(sa, sb, da, db, sigma_pos_mm, calib):
    """Combine registration + detector + null-floor sigmas in quadrature; classify
    grew/shrank/stable with a significance gate (|z|>=k AND |delta|>=floor)."""
    k = calib.get("k", 2.5)
    floor_d = calib.get("floor_diam", 0.30)
    d_ref = max(sa["d_eq_mm"] if np.isfinite(sa["d_eq_mm"]) else 1.0, 0.5)
    floor_a = calib.get("floor_area") or (math.pi * d_ref / 2) * floor_d
    sig_reg_d = math.sqrt(2) * sigma_pos_mm
    sig_reg_a = (math.pi * d_ref / 2) * sig_reg_d

    def q(*xs):
        return math.sqrt(sum((x or 0.0) ** 2 for x in xs if x is None or np.isfinite(x)))

    sig_d = q(sig_reg_d, da.get("sigma_d_eq_mm"), db.get("sigma_d_eq_mm"), floor_d)
    sig_a = q(sig_reg_a, da.get("sigma_area_mm2"), db.get("sigma_area_mm2"), floor_a)
    valid = bool(sa["valid"] and sb["valid"])
    dd = sb["d_eq_mm"] - sa["d_eq_mm"]
    dA = sb["area_mm2"] - sa["area_mm2"]
    z = dd / sig_d if sig_d > 0 else 0.0
    sig = bool(valid and abs(z) >= k and abs(dd) >= floor_d)
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
        z_diam=round(z, 1),
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


def assign_lesion_id(uv):
    return f"M{int(round(uv[0]))}_{int(round(uv[1]))}"


def measure_changes(cc, melA, melB, detA, detB, molesA, molesB, pairs, al, ppmm, calib=None):
    """Per matched pair: measure size in each scan's OWN un-warped frame, attach
    registration + detector uncertainty, classify change. Returns (list, calib)."""
    calib = calib or load_calibration()
    field = build_residual_field(al.get("corr"), ppmm)
    out = []
    for idx, p in enumerate(pairs):
        mA, mB = molesA[p["iA"]], molesB[p["iB"]]
        sa = measure_mole_size(melA, detA, mA, ppmm, other_moles=molesA)
        sb = measure_mole_size(melB, detB, mB, ppmm, other_moles=molesB)
        da = detector_sigma(melA, detA, mA, ppmm, seed=1 + idx)
        db = detector_sigma(melB, detB, mB, ppmm, seed=1001 + idx)
        sigma_pos = float(field(np.array([[mA["x"], mA["y"]]]))[0])
        ch = combine_and_flag(sa, sb, da, db, sigma_pos, calib)
        u = round(cc["umin"] + mA["x"] / ppmm, 1)
        v = round(cc["vmin"] + mA["y"] / ppmm, 1)
        ch["id"] = assign_lesion_id((u, v))
        ch["uv_a_mm"] = [u, v]
        ch["resid_mm"] = round(p["resid_mm"], 2)
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


def write_change_csv(path, cc, changes, molesB, molesA, newJ, disJ):
    import csv

    pp = cc["ppmm"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
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
        for c in changes:
            w.writerow(
                [
                    c["id"],
                    c["class"],
                    c["uv_a_mm"][0],
                    c["uv_a_mm"][1],
                    c["d_eq_a_mm"],
                    c["d_eq_b_mm"],
                    c["delta_diam_mm"],
                    c["sigma_diam_mm"],
                    c["z_diam"],
                    c["significant"],
                    c["class"],
                ]
            )
        for j in newJ:
            u = round(cc["umin"] + molesB[j]["x"] / pp, 1)
            v = round(cc["vmin"] + molesB[j]["y"] / pp, 1)
            w.writerow(
                [
                    assign_lesion_id((u, v)),
                    "new",
                    u,
                    v,
                    "",
                    round(molesB[j]["diam_mm"], 2),
                    "",
                    "",
                    "",
                    True,
                    "new",
                ]
            )
        for i in disJ:
            u = round(cc["umin"] + molesA[i]["x"] / pp, 1)
            v = round(cc["vmin"] + molesA[i]["y"] / pp, 1)
            w.writerow(
                [
                    assign_lesion_id((u, v)),
                    "disappeared",
                    u,
                    v,
                    round(molesA[i]["diam_mm"], 2),
                    "",
                    "",
                    "",
                    "",
                    True,
                    "disappeared",
                ]
            )


def write_change_overlay(
    out, cc, mutual, changes, molesA, molesB, pairs, newJ, disJ, T, ppmm, labels=("A", "B")
):
    base = cc["texA"].copy()
    dim = (base.astype(np.float32) * 0.45).astype(np.uint8)
    canvas = np.where(mutual[..., None].astype(bool), base, dim)
    for c, p in zip(changes, pairs):
        mA = molesA[p["iA"]]
        col = _CLASS_BGR.get(c["class"], (0, 200, 0))
        ra = int(max(c["d_eq_a_mm"], 0.5) / 2 * ppmm) + 2
        rb = int(max(c["d_eq_b_mm"], 0.5) / 2 * ppmm) + 2
        cv2.circle(canvas, (mA["x"], mA["y"]), ra, col, 2)  # baseline size
        cv2.circle(canvas, (mA["x"], mA["y"]), rb, col, 1)  # follow-up size
        tag = f"{c['delta_diam_mm']:+.1f}mm" if c["class"] in ("grew", "shrank") else c["class"][:4]
        star = "*" if c["significant"] else ""
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
    for j in newJ:
        transform = np.asarray(T) if T is not None else _IDENTITY
        xy = (
            np.asarray([[molesB[j]["x"], molesB[j]["y"]]], float) @ transform[:, :2].T
        ) + transform[:, 2]
        x, y = int(xy[0, 0]), int(xy[0, 1])
        cv2.circle(canvas, (x, y), int(molesB[j]["radius_mm"] * ppmm) + 6, _CLASS_BGR["new"], 3)
    for i in disJ:
        cv2.circle(
            canvas,
            (molesA[i]["x"], molesA[i]["y"]),
            int(molesA[i]["radius_mm"] * ppmm) + 6,
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
    h, w = img.shape[:2]
    o = np.full((2 * half, 2 * half, 3), (40, 40, 40), np.uint8)
    x0, y0 = int(cx) - half, int(cy) - half
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + 2 * half), min(h, y0 + 2 * half)
    if sx1 > sx0 and sy1 > sy0:
        o[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return o


def write_change_montage(
    out, cc, changes, molesA, molesB, pairs, T, ppmm, crop_mm=12.0, labels=("A", "B")
):
    half = int(crop_mm / 2 * ppmm)
    order = sorted(range(len(changes)), key=lambda i: -abs(changes[i]["z_diam"]))
    rows = []
    for i in order:
        c, p = changes[i], pairs[i]
        mA, mB = molesA[p["iA"]], molesB[p["iB"]]
        ca = _crop(cc["texA"], mA["x"], mA["y"], half)
        cb = _crop(cc["texB"], mB["x"], mB["y"], half)
        col = _CLASS_BGR.get(c["class"], (0, 200, 0))
        cv2.circle(ca, (half, half), int(max(c["d_eq_a_mm"], 0.5) / 2 * ppmm), col, 1)
        cv2.circle(cb, (half, half), int(max(c["d_eq_b_mm"], 0.5) / 2 * ppmm), col, 1)
        row = np.hstack([ca, np.full((2 * half, 6, 3), 30, np.uint8), cb])
        bar = np.full((26, row.shape[1], 3), 30, np.uint8)
        star = " *" if c["significant"] else ""
        cv2.putText(
            bar,
            f"{c['id']} {c['uv_a_mm']}mm  d:{c['d_eq_a_mm']}->"
            f"{c['d_eq_b_mm']}mm  d{c['delta_diam_mm']:+.2f}+-"
            f"{c['sigma_diam_mm']}mm  {c['class'].upper()}{star}",
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
