"""Shared cross-scan canvas construction and alignment."""

from __future__ import annotations

import os

import cv2
import numpy as np

from .tex_anchor import Gauge, coverage_mask


IDENT = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def resample_texture_to_common_canvas(reg_dir, texture_gauge, umin, vmin, Wc, Hc, pixels_per_mm):
    """Remap a scan's texture + coverage onto the common (u,v)-mm canvas."""
    tex = cv2.imread(os.path.join(reg_dir, "texture.jpg"))
    cov = coverage_mask(reg_dir)
    # common pixel (I,J) -> (u,v)mm -> source pixel (i,j)
    Jc, Ic = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
    texture_u = umin + (Ic + 0.5) / pixels_per_mm
    texture_v = vmin + (Jc + 0.5) / pixels_per_mm
    sx = ((texture_u - texture_gauge.umin) * texture_gauge.ppmm - 0.5).astype(np.float32)
    sy = ((texture_v - texture_gauge.vmin) * texture_gauge.ppmm - 0.5).astype(np.float32)
    tex_c = cv2.remap(tex, sx, sy, cv2.INTER_CUBIC, borderValue=(40, 40, 40))
    cov_c = cv2.remap(cov, sx, sy, cv2.INTER_NEAREST, borderValue=0)
    return tex_c, (cov_c > 127).astype(np.uint8)


def compute_shared_texture_canvas(rdA, gA, rdB, gB, pixels_per_mm):
    umin = max(gA.umin, gB.umin)
    umax = min(gA.umax, gB.umax)
    vmin = max(gA.vmin, gB.vmin)
    vmax = min(gA.vmax, gB.vmax)
    if umax <= umin or vmax <= vmin:
        raise SystemExit("ABSTAIN: scans share no (u,v) overlap region.")
    Wc = int(round((umax - umin) * pixels_per_mm))
    Hc = int(round((vmax - vmin) * pixels_per_mm))
    texA, covA = resample_texture_to_common_canvas(rdA, gA, umin, vmin, Wc, Hc, pixels_per_mm)
    texB, covB = resample_texture_to_common_canvas(rdB, gB, umin, vmin, Wc, Hc, pixels_per_mm)
    return dict(
        umin=umin, vmin=vmin, ppmm=pixels_per_mm, W=Wc, H=Hc, texA=texA, texB=texB, covA=covA, covB=covB
    )


def align_scan_textures_globally(texA, texB, covA, covB, pixels_per_mm):
    grayA = cv2.cvtColor(texA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(texB, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    grayA, grayB = clahe.apply(grayA), clahe.apply(grayB)
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.008, edgeThreshold=20)
    keypoints_a, descriptors_a = sift.detectAndCompute(grayA, covA * 255)
    keypoints_b, descriptors_b = sift.detectAndCompute(grayB, covB * 255)
    info = dict(kpA=len(keypoints_a), kpB=len(keypoints_b), matches=0, inliers=0, median_resid_mm=None, T_B_to_A=None)
    if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 8 or len(keypoints_b) < 8:
        return None, info
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    raw = flann.knnMatch(descriptors_b, descriptors_a, k=2)  # query=B, train=A
    good_matches = [
        first_neighbor_match
        for first_neighbor_match, second_neighbor_match in raw
        if first_neighbor_match.distance < 0.75 * second_neighbor_match.distance
    ]
    info["matches"] = len(good_matches)
    if len(good_matches) < 6:
        return None, info
    points_b = np.float32(
        [keypoints_b[match.queryIdx].pt for match in good_matches]
    )
    points_a = np.float32(
        [keypoints_a[match.trainIdx].pt for match in good_matches]
    )
    transform_matrix, inlier_mask = cv2.estimateAffinePartial2D(
        points_b, points_a, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=5000, confidence=0.999
    )
    if transform_matrix is None:
        return None, info
    inlier_mask = inlier_mask.ravel().astype(bool)
    mapped_points = (points_b[inlier_mask] @ transform_matrix[:, :2].T) + transform_matrix[:, 2]
    residuals_mm = np.linalg.norm(mapped_points - points_a[inlier_mask], axis=1) / pixels_per_mm
    scale = float(np.hypot(transform_matrix[0, 0], transform_matrix[0, 1]))
    rot_deg = float(np.degrees(np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])))
    # spatial spread of inliers: a trustworthy fit is constrained over area, not
    # a degenerate clump (which yields a spuriously tiny residual).
    inlier_points_a = points_a[inlier_mask]
    spread_mm = (
        float(np.hypot(*(inlier_points_a.max(0) - inlier_points_a.min(0))) / pixels_per_mm)
        if inlier_mask.sum() >= 2
        else 0.0
    )
    info.update(
        inliers=int(inlier_mask.sum()),
        median_resid_mm=float(np.median(residuals_mm)) if inlier_mask.any() else None,
        scale=scale,
        rotation_deg=rot_deg,
        tx_mm=float(
            transform_matrix[0, 2] / pixels_per_mm + 0  # translation reported in canvas mm
        ),
        ty_mm=float(transform_matrix[1, 2] / pixels_per_mm),
        inlier_spread_mm=spread_mm,
        T_B_to_A=transform_matrix.tolist(),
        # Retained for the registration-uncertainty field (per-inlier
        # A-frame position + B-frame source, so residual can be recomputed
        # against whichever transform is finally chosen).
        inlier_dst_A_px=inlier_points_a.tolist(),
        inlier_src_B_px=points_b[inlier_mask].tolist(),
        inlier_resid_mm=residuals_mm.tolist(),
    )
    return transform_matrix, info


def estimate_scan_alignment_from_moles(molesA, molesB, pixels_per_mm, tol_mm=3.0, min_inliers=3, min_sep_mm=8.0):
    """Constellation alignment: the moles are their own fiducials (design doc).

    A 2-point RANSAC over SIMILARITY transforms: each pair of A-moles and pair
    of B-moles with matching separation proposes a rotation+scale+translation
    (translation-only would fail -- the subject also rotates a few degrees, which
    over a wide skin site is several mm at the edges). The hypothesis aligning the
    most moles within tol wins, refined on its inliers. Returns (T_B_to_A, n)."""
    if len(molesA) < 2 or len(molesB) < 2:
        return None, 0
    points_a = np.array([[mask["x"], mask["y"]] for mask in molesA], float)
    points_b = np.array([[mask["x"], mask["y"]] for mask in molesB], float)
    tol = tol_mm * pixels_per_mm
    sep = min_sep_mm * pixels_per_mm
    best_T, best_in = None, 0
    for i1 in range(len(points_a)):
        for i2 in range(len(points_a)):
            if i1 == i2:
                continue
            dA = np.linalg.norm(points_a[i1] - points_a[i2])
            if dA < sep:  # need separated anchors
                continue
            for j1 in range(len(points_b)):
                for j2 in range(len(points_b)):
                    if j1 == j2 or abs(np.linalg.norm(points_b[j1] - points_b[j2]) - dA) > tol:
                        continue  # edge length must match (~scale 1)
                    transform_matrix = cv2.estimateAffinePartial2D(
                        np.float32([points_b[j1], points_b[j2]]), np.float32([points_a[i1], points_a[i2]])
                    )[0]
                    if transform_matrix is None:
                        continue
                    score = np.hypot(transform_matrix[0, 0], transform_matrix[0, 1])
                    rot = abs(np.degrees(np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])))
                    if not (0.9 < score < 1.1) or rot > 15:
                        continue  # implausible vs gantry prior
                    Bm = (points_b @ transform_matrix[:, :2].T) + transform_matrix[:, 2]
                    distance = np.linalg.norm(Bm[:, None, :] - points_a[None, :, :], axis=2)
                    inl = int((distance.min(0) <= tol).sum())
                    if inl > best_in:
                        best_in, best_T = inl, transform_matrix
    if best_in < min_inliers or best_T is None:
        return None, best_in
    # refine on the full inlier set
    Bm = (points_b @ best_T[:, :2].T) + best_T[:, 2]
    distance = np.linalg.norm(Bm[:, None, :] - points_a[None, :, :], axis=2)
    bi = distance.argmin(1)
    pa, pb = [], []
    for neighbor_index in range(len(points_b)):
        if distance[neighbor_index, bi[neighbor_index]] <= tol:
            pb.append(points_b[neighbor_index])
            pa.append(points_a[bi[neighbor_index]])
    if len(pa) >= 2:
        T2 = cv2.estimateAffinePartial2D(np.float32(pb), np.float32(pa))[0]
        if T2 is not None:
            return T2, best_in
    return best_T, best_in


def count_transformed_mole_matches(molesA, molesB, transform_matrix, pixels_per_mm, tol_mm):
    """How many independently-detected moles a transform brings into agreement
    (greedy unique). This is the corroboration signal used to pick the alignment
    -- the geometrically-correct transform aligns the most moles, regardless of
    how few SIFT inliers produced it."""
    return len(match_scan_moles_after_alignment(molesA, molesB, transform_matrix, pixels_per_mm, tol_mm)[0])


def select_best_scan_alignment(ainfo, molesA, molesB, pixels_per_mm, max_resid_mm=3.0, mole_tol_mm=3.0):
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
    Tmole, mole_inl = estimate_scan_alignment_from_moles(molesA, molesB, pixels_per_mm, tol_mm=mole_tol_mm, min_inliers=3)
    cands = [("gantry-only", IDENT)]
    if sift_plausible:
        cands.append(("sift", np.array(ainfo["T_B_to_A"])))
    if Tmole is not None:
        cands.append(("mole-constellation", Tmole))
    scored = sorted(
        ((nm, Tc, count_transformed_mole_matches(molesA, molesB, Tc, pixels_per_mm, mole_tol_mm)) for nm, Tc in cands),
        key=lambda score: -score[2],
    )
    mode, transform_matrix, n_corrob = scored[0]
    sift_T = np.array(ainfo["T_B_to_A"]) if sift_plausible else None
    methods_agree = (
        sift_T is not None
        and Tmole is not None
        and np.hypot(sift_T[0, 2] - Tmole[0, 2], sift_T[1, 2] - Tmole[1, 2]) / pixels_per_mm <= 5.0
        and count_transformed_mole_matches(molesA, molesB, sift_T, pixels_per_mm, mole_tol_mm) >= 3
        and count_transformed_mole_matches(molesA, molesB, Tmole, pixels_per_mm, mole_tol_mm) >= 3
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
        transform_matrix = None
    # Correspondence set in the A frame (A-px positions + per-correspondence
    # residual against the CHOSEN transform T) -- feeds the M2 registration-
    # uncertainty field. Prefer SIFT inliers (dense) whenever SIFT is plausible,
    # even if gantry/mole won the mole-count tie: for a no-reposition pair the
    # chosen T is ~identity and the SIFT residual is ~0, giving a correctly TIGHT
    # registration uncertainty (not the flat 1mm "we don't know" fallback).
    corr = None
    if sift_plausible and "inlier_src_B_px" in ainfo and transform_matrix is not None:
        srcB = np.asarray(ainfo["inlier_src_B_px"], float)
        dstA = np.asarray(ainfo["inlier_dst_A_px"], float)
        Tn = np.asarray(transform_matrix)
        mapped = (srcB @ Tn[:, :2].T) + Tn[:, 2]
        res = np.linalg.norm(dstA - mapped, axis=1) / pixels_per_mm
        corr = dict(dst_px=dstA.tolist(), resid_mm=res.tolist())
    elif mode == "mole-constellation" and Tmole is not None:
        pr = match_scan_moles_after_alignment(molesA, molesB, Tmole, pixels_per_mm, mole_tol_mm + 1.0)[0]
        Bxy = apply_scan_alignment_transform(Tmole, [[molesB[point["iB"]]["x"], molesB[point["iB"]]["y"]] for point in pr])
        dst = [[molesA[point["iA"]]["x"], molesA[point["iA"]]["y"]] for point in pr]
        res = [float(np.hypot(distance[0] - second_value[0], distance[1] - second_value[1]) / pixels_per_mm) for distance, second_value in zip(dst, Bxy)]
        corr = dict(dst_px=dst, resid_mm=res)
    return dict(
        T=transform_matrix,
        mode=mode,
        confidence=confidence,
        n_corrob=n_corrob,
        methods_agree=methods_agree,
        sift_overwhelming=sift_overwhelming,
        Tmole=Tmole,
        mole_inl=mole_inl,
        sift_plausible=sift_plausible,
        corr=corr,
        scored=[(method_name, corroborating_mole_count) for method_name, candidate_transform, corroborating_mole_count in scored],
    )


def apply_scan_alignment_transform(transform_matrix, pts):
    if transform_matrix is None:
        return np.asarray(pts, float)
    points = np.asarray(pts, float)
    return (points @ np.array(transform_matrix)[:, :2].T) + np.array(transform_matrix)[:, 2]


def match_scan_moles_after_alignment(molesA, molesB, transform_matrix, pixels_per_mm, prior_mm):
    """Match B->A (B mapped into A frame by T). Greedy NN under prior radius."""
    if not molesA or not molesB:
        return [], list(range(len(molesA))), list(range(len(molesB)))
    points_a = np.array([[mask["x"], mask["y"]] for mask in molesA], float)
    points_b = apply_scan_alignment_transform(transform_matrix, [[mask["x"], mask["y"]] for mask in molesB])
    prior_px = prior_mm * pixels_per_mm
    pairs, usedA, usedB = [], set(), set()
    cost = []
    for neighbor_index in range(len(points_b)):
        for index in range(len(points_a)):
            distance = np.linalg.norm(points_b[neighbor_index] - points_a[index])
            if distance <= prior_px:
                cost.append((distance, index, neighbor_index))
    for distance, index, neighbor_index in sorted(cost):
        if index in usedA or neighbor_index in usedB:
            continue
        usedA.add(index)
        usedB.add(neighbor_index)
        pairs.append(dict(iA=index, iB=neighbor_index, resid_mm=float(distance / pixels_per_mm)))
    onlyA = [index for index in range(len(molesA)) if index not in usedA]
    onlyB = [neighbor_index for neighbor_index in range(len(molesB)) if neighbor_index not in usedB]
    return pairs, onlyA, onlyB


def validate_mole_constellation_correspondence(molesA, molesB, pairs, pixels_per_mm, min_inliers=5):
    """Independent sanity check: do matched moles agree on a rigid map?"""
    if len(pairs) < min_inliers:
        return dict(
            status="uninformative",
            n=len(pairs),
            note="too few matched moles for a discriminative constellation",
        )
    points_a = np.array([[molesA[point["iA"]]["x"], molesA[point["iA"]]["y"]] for point in pairs], float)
    points_b = np.array([[molesB[point["iB"]]["x"], molesB[point["iB"]]["y"]] for point in pairs], float)
    if np.linalg.matrix_rank(points_a - points_a.mean(0)) < 2:
        return dict(status="uninformative", n=len(pairs), note="collinear moles")
    T2, inl = cv2.estimateAffinePartial2D(points_b, points_a, method=cv2.RANSAC, ransacReprojThreshold=3.0 * pixels_per_mm)
    inl = inl.ravel().astype(bool) if inl is not None else np.zeros(len(points_a), bool)
    mapped = (points_b @ T2[:, :2].T) + T2[:, 2] if T2 is not None else points_b
    resid = np.linalg.norm(mapped - points_a, axis=1) / pixels_per_mm
    return dict(
        status="ok",
        n=int(inl.sum()),
        median_resid_mm=float(np.median(resid[inl])) if inl.any() else None,
    )
