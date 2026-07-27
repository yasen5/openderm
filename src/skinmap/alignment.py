"""Shared cross-scan canvas construction and alignment."""

from __future__ import annotations

import os

import cv2
import numpy as np

from .tex_anchor import Gauge, coverage_mask


IDENT = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def resample_to_common(reg_dir, g: Gauge, umin, vmin, Wc, Hc, ppmm):
    """Remap a scan's texture + coverage onto the common (u,v)-mm canvas."""
    tex = cv2.imread(os.path.join(reg_dir, "texture.jpg"))
    cov = coverage_mask(reg_dir)
    # common pixel (I,J) -> (u,v)mm -> source pixel (i,j)
    Jc, Ic = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
    u = umin + (Ic + 0.5) / ppmm
    v = vmin + (Jc + 0.5) / ppmm
    sx = ((u - g.umin) * g.ppmm - 0.5).astype(np.float32)
    sy = ((v - g.vmin) * g.ppmm - 0.5).astype(np.float32)
    tex_c = cv2.remap(tex, sx, sy, cv2.INTER_CUBIC, borderValue=(40, 40, 40))
    cov_c = cv2.remap(cov, sx, sy, cv2.INTER_NEAREST, borderValue=0)
    return tex_c, (cov_c > 127).astype(np.uint8)


def common_canvas(rdA, gA, rdB, gB, ppmm):
    umin = max(gA.umin, gB.umin)
    umax = min(gA.umax, gB.umax)
    vmin = max(gA.vmin, gB.vmin)
    vmax = min(gA.vmax, gB.vmax)
    if umax <= umin or vmax <= vmin:
        raise SystemExit("ABSTAIN: scans share no (u,v) overlap region.")
    Wc = int(round((umax - umin) * ppmm))
    Hc = int(round((vmax - vmin) * ppmm))
    texA, covA = resample_to_common(rdA, gA, umin, vmin, Wc, Hc, ppmm)
    texB, covB = resample_to_common(rdB, gB, umin, vmin, Wc, Hc, ppmm)
    return dict(
        umin=umin, vmin=vmin, ppmm=ppmm, W=Wc, H=Hc, texA=texA, texB=texB, covA=covA, covB=covB
    )


def global_align(texA, texB, covA, covB, ppmm):
    grayA = cv2.cvtColor(texA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(texB, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    grayA, grayB = clahe.apply(grayA), clahe.apply(grayB)
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.008, edgeThreshold=20)
    kA, dA = sift.detectAndCompute(grayA, covA * 255)
    kB, dB = sift.detectAndCompute(grayB, covB * 255)
    info = dict(kpA=len(kA), kpB=len(kB), matches=0, inliers=0, median_resid_mm=None, T_B_to_A=None)
    if dA is None or dB is None or len(kA) < 8 or len(kB) < 8:
        return None, info
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    raw = flann.knnMatch(dB, dA, k=2)  # query=B, train=A
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    info["matches"] = len(good)
    if len(good) < 6:
        return None, info
    ptsB = np.float32([kB[m.queryIdx].pt for m in good])
    ptsA = np.float32([kA[m.trainIdx].pt for m in good])
    T, inl = cv2.estimateAffinePartial2D(
        ptsB, ptsA, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=5000, confidence=0.999
    )
    if T is None:
        return None, info
    inl = inl.ravel().astype(bool)
    mapped = (ptsB[inl] @ T[:, :2].T) + T[:, 2]
    resid = np.linalg.norm(mapped - ptsA[inl], axis=1) / ppmm
    scale = float(np.hypot(T[0, 0], T[0, 1]))
    rot_deg = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
    # spatial spread of inliers: a trustworthy fit is constrained over area, not
    # a degenerate clump (which yields a spuriously tiny residual).
    inl_pts = ptsA[inl]
    spread_mm = (
        float(np.hypot(*(inl_pts.max(0) - inl_pts.min(0))) / ppmm) if inl.sum() >= 2 else 0.0
    )
    info.update(
        inliers=int(inl.sum()),
        median_resid_mm=float(np.median(resid)) if inl.any() else None,
        scale=scale,
        rotation_deg=rot_deg,
        tx_mm=float(
            T[0, 2] / ppmm + 0  # translation reported in canvas mm
        ),
        ty_mm=float(T[1, 2] / ppmm),
        inlier_spread_mm=spread_mm,
        T_B_to_A=T.tolist(),
        # Retained for the registration-uncertainty field (per-inlier
        # A-frame position + B-frame source, so residual can be recomputed
        # against whichever transform is finally chosen).
        inlier_dst_A_px=inl_pts.tolist(),
        inlier_src_B_px=ptsB[inl].tolist(),
        inlier_resid_mm=resid.tolist(),
    )
    return T, info


def mole_align(molesA, molesB, ppmm, tol_mm=3.0, min_inliers=3, min_sep_mm=8.0):
    """Constellation alignment: the moles are their own fiducials (design doc).

    A 2-point RANSAC over SIMILARITY transforms: each pair of A-moles and pair
    of B-moles with matching separation proposes a rotation+scale+translation
    (translation-only would fail -- the subject also rotates a few degrees, which
    over a wide skin site is several mm at the edges). The hypothesis aligning the
    most moles within tol wins, refined on its inliers. Returns (T_B_to_A, n)."""
    if len(molesA) < 2 or len(molesB) < 2:
        return None, 0
    A = np.array([[m["x"], m["y"]] for m in molesA], float)
    B = np.array([[m["x"], m["y"]] for m in molesB], float)
    tol = tol_mm * ppmm
    sep = min_sep_mm * ppmm
    best_T, best_in = None, 0
    for i1 in range(len(A)):
        for i2 in range(len(A)):
            if i1 == i2:
                continue
            dA = np.linalg.norm(A[i1] - A[i2])
            if dA < sep:  # need separated anchors
                continue
            for j1 in range(len(B)):
                for j2 in range(len(B)):
                    if j1 == j2 or abs(np.linalg.norm(B[j1] - B[j2]) - dA) > tol:
                        continue  # edge length must match (~scale 1)
                    T = cv2.estimateAffinePartial2D(
                        np.float32([B[j1], B[j2]]), np.float32([A[i1], A[i2]])
                    )[0]
                    if T is None:
                        continue
                    s = np.hypot(T[0, 0], T[0, 1])
                    rot = abs(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
                    if not (0.9 < s < 1.1) or rot > 15:
                        continue  # implausible vs gantry prior
                    Bm = (B @ T[:, :2].T) + T[:, 2]
                    d = np.linalg.norm(Bm[:, None, :] - A[None, :, :], axis=2)
                    inl = int((d.min(0) <= tol).sum())
                    if inl > best_in:
                        best_in, best_T = inl, T
    if best_in < min_inliers or best_T is None:
        return None, best_in
    # refine on the full inlier set
    Bm = (B @ best_T[:, :2].T) + best_T[:, 2]
    d = np.linalg.norm(Bm[:, None, :] - A[None, :, :], axis=2)
    bi = d.argmin(1)
    pa, pb = [], []
    for j in range(len(B)):
        if d[j, bi[j]] <= tol:
            pb.append(B[j])
            pa.append(A[bi[j]])
    if len(pa) >= 2:
        T2 = cv2.estimateAffinePartial2D(np.float32(pb), np.float32(pa))[0]
        if T2 is not None:
            return T2, best_in
    return best_T, best_in


def count_matches(molesA, molesB, T, ppmm, tol_mm):
    """How many independently-detected moles a transform brings into agreement
    (greedy unique). This is the corroboration signal used to pick the alignment
    -- the geometrically-correct transform aligns the most moles, regardless of
    how few SIFT inliers produced it."""
    return len(match_moles(molesA, molesB, T, ppmm, tol_mm)[0])


def select_alignment(ainfo, molesA, molesB, ppmm, max_resid_mm=3.0, mole_tol_mm=3.0):
    """Choose the B->A transform + confidence (shared by compare_scans and
    new_moles). SIFT and the mole-constellation each propose a candidate; the one
    aligning the most independently-detected moles wins (geometry beats SIFT
    inlier count). Confidence is HIGH when >=5 moles align, OR the two methods
    independently agree, OR SIFT is overwhelming (many inliers, wide spread)."""
    sift_plausible = (
        ainfo["T_B_to_A"] is not None
        and ainfo["median_resid_mm"] is not None
        and ainfo["median_resid_mm"] <= max_resid_mm
        and 0.95 < ainfo["scale"] < 1.05
        and abs(ainfo["rotation_deg"]) < 10
        and ainfo["inlier_spread_mm"] >= 40
        and abs(ainfo["tx_mm"]) < 80
        and abs(ainfo["ty_mm"]) < 80
    )
    Tmole, mole_inl = mole_align(molesA, molesB, ppmm, tol_mm=mole_tol_mm, min_inliers=3)
    cands = [("gantry-only", IDENT)]
    if sift_plausible:
        cands.append(("sift", np.array(ainfo["T_B_to_A"])))
    if Tmole is not None:
        cands.append(("mole-constellation", Tmole))
    scored = sorted(
        ((nm, Tc, count_matches(molesA, molesB, Tc, ppmm, mole_tol_mm)) for nm, Tc in cands),
        key=lambda s: -s[2],
    )
    mode, T, n_corrob = scored[0]
    sift_T = np.array(ainfo["T_B_to_A"]) if sift_plausible else None
    methods_agree = (
        sift_T is not None
        and Tmole is not None
        and np.hypot(sift_T[0, 2] - Tmole[0, 2], sift_T[1, 2] - Tmole[1, 2]) / ppmm <= 5.0
        and count_matches(molesA, molesB, sift_T, ppmm, mole_tol_mm) >= 3
        and count_matches(molesA, molesB, Tmole, ppmm, mole_tol_mm) >= 3
    )
    sift_overwhelming = (
        sift_plausible and ainfo["inliers"] >= 50 and ainfo["inlier_spread_mm"] >= 100
    )
    # gantry-only winning with high corroboration means the subject did NOT
    # reposition (identity IS the right transform) -- that is the best case, not
    # the worst. Only treat gantry-only as a failure when corroboration is low
    # (i.e. it won by default because SIFT was degenerate and moles didn't match).
    if n_corrob >= 5 or methods_agree or sift_overwhelming:
        confidence = "high"
    elif n_corrob >= 3:
        confidence = "medium"
    else:
        confidence = "low"
    if confidence == "low" and mode == "gantry-only":
        T = None
    # Correspondence set in the A frame (A-px positions + per-correspondence
    # residual against the CHOSEN transform T) -- feeds the M2 registration-
    # uncertainty field. Prefer SIFT inliers (dense) whenever SIFT is plausible,
    # even if gantry/mole won the mole-count tie: for a no-reposition pair the
    # chosen T is ~identity and the SIFT residual is ~0, giving a correctly TIGHT
    # registration uncertainty (not the flat 1mm "we don't know" fallback).
    corr = None
    if sift_plausible and "inlier_src_B_px" in ainfo and T is not None:
        srcB = np.asarray(ainfo["inlier_src_B_px"], float)
        dstA = np.asarray(ainfo["inlier_dst_A_px"], float)
        Tn = np.asarray(T)
        mapped = (srcB @ Tn[:, :2].T) + Tn[:, 2]
        res = np.linalg.norm(dstA - mapped, axis=1) / ppmm
        corr = dict(dst_px=dstA.tolist(), resid_mm=res.tolist())
    elif mode == "mole-constellation" and Tmole is not None:
        pr = match_moles(molesA, molesB, Tmole, ppmm, mole_tol_mm + 1.0)[0]
        Bxy = apply_T(Tmole, [[molesB[p["iB"]]["x"], molesB[p["iB"]]["y"]] for p in pr])
        dst = [[molesA[p["iA"]]["x"], molesA[p["iA"]]["y"]] for p in pr]
        res = [float(np.hypot(d[0] - b[0], d[1] - b[1]) / ppmm) for d, b in zip(dst, Bxy)]
        corr = dict(dst_px=dst, resid_mm=res)
    return dict(
        T=T,
        mode=mode,
        confidence=confidence,
        n_corrob=n_corrob,
        methods_agree=methods_agree,
        sift_overwhelming=sift_overwhelming,
        Tmole=Tmole,
        mole_inl=mole_inl,
        sift_plausible=sift_plausible,
        corr=corr,
        scored=[(nm, n) for nm, _, n in scored],
    )


def apply_T(T, pts):
    if T is None:
        return np.asarray(pts, float)
    P = np.asarray(pts, float)
    return (P @ np.array(T)[:, :2].T) + np.array(T)[:, 2]


def match_moles(molesA, molesB, T, ppmm, prior_mm):
    """Match B->A (B mapped into A frame by T). Greedy NN under prior radius."""
    if not molesA or not molesB:
        return [], list(range(len(molesA))), list(range(len(molesB)))
    A = np.array([[m["x"], m["y"]] for m in molesA], float)
    B = apply_T(T, [[m["x"], m["y"]] for m in molesB])
    prior_px = prior_mm * ppmm
    pairs, usedA, usedB = [], set(), set()
    cost = []
    for j in range(len(B)):
        for i in range(len(A)):
            d = np.linalg.norm(B[j] - A[i])
            if d <= prior_px:
                cost.append((d, i, j))
    for d, i, j in sorted(cost):
        if i in usedA or j in usedB:
            continue
        usedA.add(i)
        usedB.add(j)
        pairs.append(dict(iA=i, iB=j, resid_mm=float(d / ppmm)))
    onlyA = [i for i in range(len(molesA)) if i not in usedA]
    onlyB = [j for j in range(len(molesB)) if j not in usedB]
    return pairs, onlyA, onlyB


def constellation_check(molesA, molesB, pairs, ppmm, min_inliers=5):
    """Independent sanity check: do matched moles agree on a rigid map?"""
    if len(pairs) < min_inliers:
        return dict(
            status="uninformative",
            n=len(pairs),
            note="too few matched moles for a discriminative constellation",
        )
    A = np.array([[molesA[p["iA"]]["x"], molesA[p["iA"]]["y"]] for p in pairs], float)
    B = np.array([[molesB[p["iB"]]["x"], molesB[p["iB"]]["y"]] for p in pairs], float)
    if np.linalg.matrix_rank(A - A.mean(0)) < 2:
        return dict(status="uninformative", n=len(pairs), note="collinear moles")
    T2, inl = cv2.estimateAffinePartial2D(B, A, method=cv2.RANSAC, ransacReprojThreshold=3.0 * ppmm)
    inl = inl.ravel().astype(bool) if inl is not None else np.zeros(len(A), bool)
    mapped = (B @ T2[:, :2].T) + T2[:, 2] if T2 is not None else B
    resid = np.linalg.norm(mapped - A, axis=1) / ppmm
    return dict(
        status="ok",
        n=int(inl.sum()),
        median_resid_mm=float(np.median(resid[inl])) if inl.any() else None,
    )
