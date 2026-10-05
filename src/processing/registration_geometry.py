"""Rig fitting, overlap discovery, tracking, and bundle adjustment."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .registration_features import Frame


# ----------------------------------------------------------------------------
# rig model: gantry proprioception -> 6-DOF camera pose
# ----------------------------------------------------------------------------
@dataclass
class RigModel:
    fx: float  # focal length, *downscaled* px
    k1: float  # radial distortion (normalised coords)
    cx: float
    cy: float
    sign: float  # rx rotation sign about world +x
    lever: np.ndarray  # camera centre offset from toolhead, rotates with rx (mm)
    Rm: np.ndarray  # 3x3 camera mount rotation (cam->world at rx=0)
    dz0: float  # standoff sensor -> optical centre depth offset (mm)
    downscale: int = 1
    base_R: np.ndarray = field(default_factory=lambda: np.eye(3))  # gauge fix
    base_t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def poses(self, frames) -> tuple[np.ndarray, np.ndarray]:
        """Nominal (R_cam2world (n,3,3), C (n,3)) for every frame."""
        n = len(frames)
        R = np.zeros((n, 3, 3))
        C = np.zeros((n, 3))
        for f in frames:
            Rrx = Rotation.from_rotvec([self.sign * f.rx, 0, 0]).as_matrix()
            C[f.idx] = self.base_t + self.base_R @ (f.g + Rrx @ self.lever)
            R[f.idx] = self.base_R @ Rrx @ self.Rm
        return R, C

    def depth(self, f: Frame) -> float:
        return f.standoff + self.dz0


def project(X, Rcw, C, fx, k1, cx, cy):
    """World pts (N,3) -> pixel (N,2) + cam-z (N,). Rcw: cam->world 3x3."""
    xc = (X - C) @ Rcw  # = Rcw^T (X - C)
    z = np.maximum(xc[:, 2], 1e-6)
    xn = xc[:, :2] / z[:, None]
    r2 = (xn**2).sum(1)
    f = 1.0 + k1 * r2
    uv = np.empty_like(xn)
    uv[:, 0] = fx * xn[:, 0] * f + cx
    uv[:, 1] = fx * xn[:, 1] * f + cy
    return uv, xc[:, 2]


def undistort_norm(uv, fx, k1, cx, cy):
    """Pixel (N,2) -> normalised undistorted (N,2)."""
    xn = (uv - [cx, cy]) / fx
    xu = xn.copy()
    for _ in range(3):
        r2 = (xu**2).sum(1)
        xu = xn / (1.0 + k1 * r2)[:, None]
    return xu


def prefit_rig_model(frames, cpairs, w, h, downscale, fx0) -> tuple[RigModel, float]:
    """Fit the mechanical rig model from consecutive-pair affine transforms.

    The focal-length estimate here is an initialization for the mechanical fit;
    a supplied --fx-full value replaces it before bundle adjustment. Five sample
    points per pair capture translation, rotation, and scale of the measured
    affine.
    """
    cx, cy = w / 2.0, h / 2.0
    samp = np.array(
        [
            [cx, cy],
            [cx - w / 4, cy - h / 4],
            [cx + w / 4, cy - h / 4],
            [cx + w / 4, cy + h / 4],
            [cx - w / 4, cy + h / 4],
        ],
        float,
    )

    # measured destinations of the sample points under each pair's affine
    meas = []
    for p in cpairs:
        th = math.radians(p.rot_deg)
        c, s = p.scale * math.cos(th), p.scale * math.sin(th)
        A = np.array([[c, -s, p.tx], [s, c, p.ty]])
        meas.append(samp @ A[:, :2].T + A[:, 2])
    meas = np.array(meas)  # (npair, 5, 2)

    g = np.array([f.g for f in frames])
    rx = np.array([f.rx for f in frames])
    so = np.array([f.standoff for f in frames])
    ii = np.array([p.i for p in cpairs])
    jj = np.array([p.j for p in cpairs])

    def base_rm(psi_deg, roll_deg):
        zc = np.array([0, 0, -1.0])
        r = math.radians(roll_deg)
        xc = np.array([math.cos(r), math.sin(r), 0.0])
        yc = np.cross(zc, xc)
        B = np.stack([xc, yc, zc], axis=1)
        return Rotation.from_rotvec([math.radians(psi_deg), 0, 0]).as_matrix() @ B

    def resid_for(sign, Rm_init):
        Rrx = Rotation.from_rotvec(np.outer(sign * rx, [1, 0, 0])).as_matrix()

        def resid(p):
            fx, dz0 = p[0], p[1]
            lever = p[2:5]
            Rm = Rm_init @ Rotation.from_rotvec(p[5:8]).as_matrix()
            C = g + Rrx @ lever  # (n,3)
            R = np.einsum("nij,jk->nik", Rrx, Rm)  # cam->world
            Zi = so[ii] + dz0  # (npair,)
            xu = (samp - [cx, cy]) / fx  # (5,2)
            xc = np.empty((len(ii), 5, 3))
            xc[:, :, :2] = xu[None] * Zi[:, None, None]
            xc[:, :, 2] = Zi[:, None]
            Xw = np.einsum("pij,pkj->pki", R[ii], xc) + C[ii][:, None, :]
            d = Xw - C[jj][:, None, :]
            xcj = np.einsum("pki,pij->pkj", d, R[jj])
            z = np.maximum(xcj[:, :, 2], 1e-6)
            uv = fx * xcj[:, :, :2] / z[..., None] + [cx, cy]
            res = (uv - meas).ravel() / 3.0
            prior = np.concatenate([[(dz0 - 50.0) / 100.0], lever / 200.0, p[5:8] / 0.5])
            return np.concatenate([res, prior])

        return resid

    # dz0 can be large and positive: the pinhole centre (entrance pupil) of a
    # long macro lens sits far behind the standoff sensors' reference plane.
    lb = [fx0 * 0.1, -20, -300, -300, -300, -1.5, -1.5, -1.5]
    ub = [fx0 * 20, 400, 300, 300, 300, 1.5, 1.5, 1.5]
    p0 = np.zeros(8)
    p0[0] = fx0
    trials = []
    for sign in (1.0, -1.0):
        for psi in (-70, -35, 0, 35, 70):
            for roll in (0, 90, 180, 270):
                fn = resid_for(sign, base_rm(psi, roll))
                try:
                    sol = least_squares(
                        fn,
                        p0,
                        method="trf",
                        loss="soft_l1",
                        f_scale=5.0,
                        max_nfev=25,
                        bounds=(lb, ub),
                    )
                except Exception:
                    continue
                trials.append((sol.cost, sol.x, sign, psi, roll))
    trials.sort(key=lambda t: t[0])
    best = None
    for cost0, x0_, sign, psi, roll in trials[:3]:  # polish the 3 best seeds
        fn = resid_for(sign, base_rm(psi, roll))
        sol = least_squares(
            fn, x0_, method="trf", loss="soft_l1", f_scale=5.0, max_nfev=150, bounds=(lb, ub)
        )
        if best is None or sol.cost < best[0]:
            best = (sol.cost, sol, sign, base_rm(psi, roll), psi, roll)
    cost, sol, sign, Rm_init, psi, roll = best
    p = sol.x
    Rm = Rm_init @ Rotation.from_rotvec(p[5:8]).as_matrix()
    mdl = RigModel(
        fx=p[0],
        k1=0.0,
        cx=cx,
        cy=cy,
        sign=sign,
        lever=p[2:5].copy(),
        Rm=Rm,
        dz0=p[1],
        downscale=downscale,
    )
    # robust-ish rms of the pixel part only
    nres = meas.size
    fn = resid_for(sign, Rm_init)
    r = fn(p)[:nres] * 3.0
    rms = float(np.sqrt(np.mean(r**2)))
    print(
        f"      best init: sign={sign:+.0f} psi={psi} roll={roll}; "
        f"fit rms {rms:.2f}px over {len(cpairs)} consecutive pairs"
    )
    return mdl, rms


# ----------------------------------------------------------------------------
# overlap prediction on the (curved) surface
# ----------------------------------------------------------------------------
def predicted_pair_translation(mdl, frames, R, C, i, j):
    """Predicted image translation i->j (mean over sample pts) in ds px."""
    w, h = mdl.cx * 2, mdl.cy * 2
    samp = np.array([[mdl.cx, mdl.cy], [w * 0.3, h * 0.3], [w * 0.7, h * 0.7]])
    Zi = mdl.depth(frames[i])
    xu = undistort_norm(samp, mdl.fx, mdl.k1, mdl.cx, mdl.cy)
    xc = np.concatenate([xu * Zi, np.full((len(samp), 1), Zi)], 1)
    Xw = xc @ R[i].T + C[i]
    uv, _ = project(Xw, R[j], C[j], mdl.fx, mdl.k1, mdl.cx, mdl.cy)
    return (uv - samp).mean(0)


def find_overlap_pairs(mdl, frames, R, C, overlap_frac, max_partners):
    """Predict pairwise overlap by projecting footprints onto the mean plane."""
    n = len(frames)
    w, h = mdl.cx * 2, mdl.cy * 2
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    centers = np.zeros((n, 3))
    quads = np.zeros((n, 4, 3))
    for f in frames:
        Z = mdl.depth(f)
        xu = undistort_norm(corners_px, mdl.fx, mdl.k1, mdl.cx, mdl.cy)
        xc = np.concatenate([xu * Z, np.full((4, 1), Z)], 1)
        quads[f.idx] = xc @ R[f.idx].T + C[f.idx]
        centers[f.idx] = C[f.idx] + Z * R[f.idx][:, 2]
    ctr = centers.mean(0)
    _, _, Vt = np.linalg.svd(centers - ctr, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    q2 = np.stack([(quads - ctr) @ e1, (quads - ctr) @ e2], axis=-1)  # (n,4,2)
    lo = q2.min(1)
    hi = q2.max(1)
    area = (hi - lo).prod(1)
    overlaps = {}
    for i in range(n):
        ix = np.maximum(0.0, np.minimum(hi[i, 0], hi[:, 0]) - np.maximum(lo[i, 0], lo[:, 0]))
        iy = np.maximum(0.0, np.minimum(hi[i, 1], hi[:, 1]) - np.maximum(lo[i, 1], lo[:, 1]))
        frac = ix * iy / np.minimum(area[i], area)
        for j in range(i + 1, n):
            if frac[j] >= overlap_frac:
                overlaps[(i, j)] = float(frac[j])
    partners = {i: [] for i in range(n)}
    for (i, j), fr in overlaps.items():
        partners[i].append((fr, j))
        partners[j].append((fr, i))
    keep = set()
    for k, lst in partners.items():
        for fr, o in sorted(lst, reverse=True)[:max_partners]:
            keep.add((min(k, o), max(k, o)))
    return keep, overlaps


# ----------------------------------------------------------------------------
# tracks (union-find over matched keypoints)
# ----------------------------------------------------------------------------
def build_tracks(frames, pairs, max_corr_per_pair):
    t0 = time.time()
    nkp = [len(f.kp) for f in frames]
    off = np.concatenate([[0], np.cumsum(nkp)])
    parent = np.arange(off[-1], dtype=np.int64)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    rng = np.random.default_rng(0)
    for p in pairs:
        m = len(p.src_kp)
        sel = rng.permutation(m)[:max_corr_per_pair]
        for a, b in zip(p.src_kp[sel], p.dst_kp[sel]):
            ra, rb = find(off[p.i] + a), find(off[p.j] + b)
            if ra != rb:
                parent[rb] = ra

    # collect components
    groups = {}
    used = set()
    for p in pairs:
        for arr, fi in ((p.src_kp, p.i), (p.dst_kp, p.j)):
            for a in arr:
                used.add(off[fi] + a)
    for node in used:
        groups.setdefault(find(node), []).append(node)

    obs_frame, obs_uv, obs_track = [], [], []
    tid = 0
    n_dup = 0
    fr_of = np.searchsorted(off, np.arange(off[-1]), side="right") - 1
    for root, nodes in groups.items():
        if len(nodes) < 2:
            continue
        frs = [int(fr_of[n]) for n in nodes]
        if len(set(frs)) != len(frs):  # same frame twice -> ambiguous
            n_dup += 1
            continue
        if len(set(frs)) < 2:
            continue
        for n_, fi in zip(nodes, frs):
            kpidx = int(n_ - off[fi])
            obs_frame.append(fi)
            obs_uv.append(frames[fi].kp[kpidx].pt)
            obs_track.append(tid)
        tid += 1
    print(
        f"      {tid} tracks, {len(obs_frame)} observations "
        f"({n_dup} ambiguous tracks dropped) [{time.time() - t0:.0f}s]"
    )
    return (
        np.array(obs_frame, np.int32),
        np.array(obs_uv, np.float64),
        np.array(obs_track, np.int32),
        tid,
    )


# ----------------------------------------------------------------------------
# bundle adjustment: alternating intersection / resection
# ----------------------------------------------------------------------------
def triangulate(obs_frame, obs_uv, obs_track, ntracks, R, C, mdl):
    xu = undistort_norm(obs_uv, mdl.fx, mdl.k1, mdl.cx, mdl.cy)
    d_cam = np.concatenate([xu, np.ones((len(xu), 1))], 1)
    d_w = np.einsum("nij,nj->ni", R[obs_frame], d_cam)
    d_w /= np.linalg.norm(d_w, axis=1, keepdims=True)
    M = np.eye(3)[None] - d_w[:, :, None] * d_w[:, None, :]
    A = np.zeros((ntracks, 3, 3))
    b = np.zeros((ntracks, 3))
    np.add.at(A, obs_track, M)
    np.add.at(b, obs_track, np.einsum("nij,nj->ni", M, C[obs_frame]))
    A += np.eye(3)[None] * 1e-9
    # NumPy 2.x interprets a 2-D right-hand side as a stack of matrices,
    # which broadcasts ``(tracks, 3)`` into ``(tracks, tracks, 3)`` here.
    # Make the per-track vector dimension explicit across NumPy versions.
    return np.linalg.solve(A, b[..., None])[..., 0]


def reproj_errors(X, obs_frame, obs_uv, obs_track, R, C, mdl):
    Xo = X[obs_track]
    Co = C[obs_frame]
    xc = np.einsum("ni,nij->nj", Xo - Co, R[obs_frame])
    z = np.maximum(xc[:, 2], 1e-6)
    xn = xc[:, :2] / z[:, None]
    r2 = (xn**2).sum(1)
    uv = mdl.fx * xn * (1 + mdl.k1 * r2)[:, None] + [mdl.cx, mdl.cy]
    err = np.linalg.norm(uv - obs_uv, axis=1)
    return err, xc[:, 2]


def resect_all(
    frames, X, obs_frame, obs_uv, obs_track, R, C, R0, C0, mdl, sigma_px, sigma_t, sigma_r
):
    order = np.argsort(obs_frame, kind="stable")
    of, ou, ot = obs_frame[order], obs_uv[order], obs_track[order]
    bounds = np.searchsorted(of, np.arange(len(frames) + 1))
    n_small = 0
    for f in frames:
        i = f.idx
        s, e = bounds[i], bounds[i + 1]
        if e - s < 20:
            n_small += 1
            continue
        Xi, uvi = X[ot[s:e]], ou[s:e]
        # init delta from current pose relative to nominal anchor
        d0 = np.zeros(6)
        d0[:3] = C[i] - C0[i]
        d0[3:] = Rotation.from_matrix(R0[i].T @ R[i]).as_rotvec()

        def resid(d):
            Rcw = R0[i] @ Rotation.from_rotvec(d[3:]).as_matrix()
            Cc = C0[i] + d[:3]
            uv, _ = project(Xi, Rcw, Cc, mdl.fx, mdl.k1, mdl.cx, mdl.cy)
            r = ((uv - uvi) / sigma_px).ravel()
            return np.concatenate([r, d[:3] / sigma_t, d[3:] / sigma_r])

        # robust loss: early rounds still contain large-residual cross-row
        # obs that the pose must converge towards, not be dragged by
        sol = least_squares(resid, d0, method="trf", loss="soft_l1", f_scale=4.0, max_nfev=40)
        R[i] = R0[i] @ Rotation.from_rotvec(sol.x[3:]).as_matrix()
        C[i] = C0[i] + sol.x[:3]
    if n_small:
        print(f"        ({n_small} frames with <20 obs kept at prior pose)")


def refine_intrinsics(X, obs_frame, obs_uv, obs_track, R, C, mdl, nsub=80000, fit_k1=False):
    """Closed-form refit of fx (and optionally k1): uv-c = [xn, xn*r2]@[fx, fx*k1].

    k1 fitting is off by default: Canon applies lens corrections to JPGs, and
    in practice k1 just absorbs IS systematics and pegs its cap unstably.
    """
    if len(obs_uv) < 2000:
        return
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(obs_uv))[:nsub]
    Xo = X[obs_track[idx]]
    Ro, Co_, uvo = R[obs_frame[idx]], C[obs_frame[idx]], obs_uv[idx]
    xc = np.einsum("ni,nij->nj", Xo - Co_, Ro)
    z = np.maximum(xc[:, 2], 1e-6)
    xn = xc[:, :2] / z[:, None]
    r2 = (xn**2).sum(1)
    if fit_k1:
        A = np.stack([xn.ravel(), (xn * r2[:, None]).ravel()], 1)
    else:
        A = xn.ravel()[:, None]
    b = (uvo - [mdl.cx, mdl.cy]).ravel()
    for _ in range(3):  # MAD-robust outlier trimming
        a, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = A @ a - b
        med = np.median(res)
        s = 1.4826 * np.median(np.abs(res - med)) + 1e-9
        keep = np.abs(res - med) < 4.0 * s
        if keep.all():
            break
        A, b = A[keep], b[keep]
    fx = float(a[0])
    if fx > 100.0:  # sanity: never collapse
        mdl.fx = fx
        if fit_k1:
            mdl.k1 = float(np.clip(float(a[1]) / fx, -0.15, 0.15))


def _polar_rot(M):
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def refit_rig_from_poses(frames, R, C, mdl):
    """Re-anchor the rig model on the current BA poses.

    The pre-fit (consecutive-pair affines only) carries systematic bias; once
    BA has settled, the lever arm, the mount rotation AND a global base
    transform (gauge: rotation Q + translation T of the whole gantry frame)
    are re-estimated from the optimised poses, so the proprioception prior
    stops fighting that bias:  C ~ T + Q (g + Rrx lever),  R ~ Q Rrx Rm.
    """
    Rrx = np.stack([Rotation.from_rotvec([mdl.sign * f.rx, 0, 0]).as_matrix() for f in frames])
    g = np.array([f.g for f in frames])
    Q = np.eye(3)
    T = np.zeros(3)
    lever = mdl.lever.copy()
    for _ in range(4):
        # lever given Q, T (linear)
        A = np.einsum("ij,njk->nik", Q, Rrx).reshape(-1, 3)
        b = (C - T - g @ Q.T).ravel()
        lever, *_ = np.linalg.lstsq(A, b, rcond=None)
        # mount rotation given Q
        M = np.einsum("nji,njk->ik", np.einsum("ij,njk->nik", Q, Rrx), R)
        Rm = _polar_rot(M)
        # base transform given lever (Kabsch on camera centres)
        Cn = g + np.einsum("nij,j->ni", Rrx, lever)
        mc, mn = C.mean(0), Cn.mean(0)
        H = (Cn - mn).T @ (C - mc)
        Q = _polar_rot(H.T)
        T = mc - Q @ mn
    mdl.lever = lever
    mdl.Rm = Rm
    mdl.base_R = Q
    mdl.base_t = T
    return mdl.poses(frames)


def resplit_soft_direction(frames, R, C, R0, C0, Zs, sigma_t=1.5, sigma_r_deg=3.0):
    """Re-split each pose deviation along the view-preserving null direction.

    With an ~8deg FOV, a sideways camera slide t and a counter-rotation
    theta ~ t/Z are nearly indistinguishable in the images, so BA parks
    large (physically impossible, 10-20mm) translations along this valley.
    Sliding back along the valley to the maximum-prior point is
    reprojection-neutral to first order but restores metric sanity: gantry
    translation is trusted to ~sigma_t, while per-frame angular error (AF
    focus breathing, kinematic residue) is left to the rotation.
    """
    st2 = sigma_t**2
    sr2 = math.radians(sigma_r_deg) ** 2
    for f in frames:
        i = f.idx
        Z = Zs[i]
        k = Z * Z / st2 + 1.0 / sr2
        dtc = R0[i].T @ (C[i] - C0[i])
        rv = Rotation.from_matrix(R0[i].T @ R[i]).as_rotvec()
        # valley pair (t_x, th_y): t_x' = t_x - Z s, th_y' = th_y + s
        s = (Z * dtc[0] / st2 - rv[1] / sr2) / k
        dtc[0] -= Z * s
        rv[1] += s
        # valley pair (t_y, th_x): t_y' = t_y + Z s, th_x' = th_x + s
        s = -(Z * dtc[1] / st2 + rv[0] / sr2) / k
        dtc[1] += Z * s
        rv[0] += s
        C[i] = C0[i] + R0[i] @ dtc
        R[i] = R0[i] @ Rotation.from_rotvec(rv).as_matrix()


def bundle_adjust(frames, pairs, mdl, R0, C0, args):
    obs_frame, obs_uv, obs_track, ntracks = build_tracks(frames, pairs, args.max_corr_per_pair)
    R, C = R0.copy(), C0.copy()
    if ntracks == 0:
        # no usable tracks (e.g. a sparse revisit group with zero intra-group
        # matches): the poses stay at the proprioception prior, which is what
        # anchors every group anyway; cross-group alignment is regularised so
        # a track-less group simply gets t~0
        print("      ! 0 tracks -- keeping proprioception-prior poses")
        return dict(
            X=np.zeros((0, 3)),
            track_err=np.zeros(0),
            obs_frame=np.zeros(0, np.int32),
            obs_uv=np.zeros((0, 2)),
            obs_track=np.zeros(0, np.int32),
            err=np.zeros(0),
            R=R,
            C=C,
            rms=0.0,
            history=[],
        )
    R0, C0 = R0.copy(), C0.copy()
    sigma_r = math.radians(args.sigma_r)
    Zs = np.array([mdl.depth(f) for f in frames])
    good = np.ones(len(obs_uv), bool)
    # round 1 keeps everything finite (the pre-fit bias puts genuine cross-row
    # obs at 50-300px; pruning them early starves BA of exactly the
    # constraints it needs); later rounds tighten progressively
    floors = [np.inf, 12.0, 8.0, 6.0, 4.0] + [3.0] * max(0, args.rounds - 5)
    switch = max(1, min(4, args.rounds // 2))
    history = []
    for rnd in range(args.rounds):
        if rnd == switch and getattr(args, "rig_from", None):
            print("        (rig re-anchor skipped: --rig-from)")
        elif rnd == switch:
            Xs_ = triangulate(obs_frame[good], obs_uv[good], obs_track[good], ntracks, R, C, mdl)
            _, zc_ = reproj_errors(Xs_, obs_frame[good], obs_uv[good], obs_track[good], R, C, mdl)
            so_ = np.array([frames[i].standoff for i in obs_frame[good]])
            mdl.dz0 = float(np.clip(np.median(zc_ - so_), -20, 600))
            R0, C0 = refit_rig_from_poses(frames, R, C, mdl)
            Zs = np.array([mdl.depth(f) for f in frames])
            print(
                f"        re-anchored rig model: lever="
                f"({mdl.lever[0]:.1f},{mdl.lever[1]:.1f},{mdl.lever[2]:.1f})mm, "
                f"dz0={mdl.dz0:+.1f}mm"
            )
        # loose prior while the anchor still carries pre-fit bias
        st = args.sigma_t * (2.0 if rnd < switch else 1.0)
        sr = sigma_r * (2.0 if rnd < switch else 1.0)
        X = triangulate(obs_frame[good], obs_uv[good], obs_track[good], ntracks, R, C, mdl)
        cnt = np.bincount(obs_track[good], minlength=ntracks)
        # reproject ALL obs so early aggressive prunes can be re-admitted
        err, zc = reproj_errors(X, obs_frame, obs_uv, obs_track, R, C, mdl)
        err[cnt[obs_track] < 2] = np.inf
        rms_pre = float(np.sqrt(np.mean(np.minimum(err[good], 1e6) ** 2)))
        med_pre = float(np.median(err[good]))
        # robust schedule: floors cap from below, a generous multiple of the
        # median caps from above (rms is unusable when a few tracks blow up)
        thresh = max(floors[rnd], 8.0 * med_pre) if np.isfinite(floors[rnd]) else np.inf
        zlo, zhi = 0.4 * Zs[obs_frame], 2.5 * Zs[obs_frame]
        new_good = (err < thresh) & (zc > zlo) & (zc < zhi)
        cnt = np.bincount(obs_track[new_good], minlength=ntracks)
        new_good &= cnt[obs_track] >= 2
        if new_good.sum() < 0.05 * len(new_good):
            print(
                f"      ! round {rnd + 1}: prune would keep only "
                f"{int(new_good.sum())} obs -- keeping previous set"
            )
        else:
            good = new_good
        e_in = err[good]
        rms_in = float(np.sqrt(np.mean(e_in**2))) if good.any() else 0.0
        print(
            f"      round {rnd + 1}/{args.rounds}: rms {rms_pre:.2f}px, "
            f"thresh {thresh:.1f} -> {int(good.sum())}/{len(good)} obs, "
            f"rms {rms_in:.2f}px (median {np.median(e_in):.2f}px)"
        )
        history.append(rms_in)
        t_rs = time.time()
        resect_all(
            frames,
            X,
            obs_frame[good],
            obs_uv[good],
            obs_track[good],
            R,
            C,
            R0,
            C0,
            mdl,
            args.sigma_px,
            st,
            sr,
        )
        print(f"        resect {time.time() - t_rs:.0f}s")
        if rnd >= 1 and not args.fx_full:
            X = triangulate(obs_frame[good], obs_uv[good], obs_track[good], ntracks, R, C, mdl)
            refine_intrinsics(
                X, obs_frame[good], obs_uv[good], obs_track[good], R, C, mdl, fit_k1=args.fit_k1
            )
            print(
                f"        intrinsics: fx={mdl.fx:.1f}ds-px "
                f"({mdl.fx * mdl.downscale:.0f} full-res px), k1={mdl.k1:+.4f}"
            )
    # re-split the translation/rotation valley to physical values, then one
    # tight-prior polish pass and a second re-split
    err, _ = reproj_errors(
        triangulate(obs_frame[good], obs_uv[good], obs_track[good], ntracks, R, C, mdl),
        obs_frame[good],
        obs_uv[good],
        obs_track[good],
        R,
        C,
        mdl,
    )
    med0 = float(np.median(err))
    resplit_soft_direction(frames, R, C, R0, C0, Zs)
    X = triangulate(obs_frame[good], obs_uv[good], obs_track[good], ntracks, R, C, mdl)
    resect_all(
        frames,
        X,
        obs_frame[good],
        obs_uv[good],
        obs_track[good],
        R,
        C,
        R0,
        C0,
        mdl,
        args.sigma_px,
        1.5,
        math.radians(3.0),
    )
    resplit_soft_direction(frames, R, C, R0, C0, Zs)
    err, _ = reproj_errors(
        triangulate(obs_frame[good], obs_uv[good], obs_track[good], ntracks, R, C, mdl),
        obs_frame[good],
        obs_uv[good],
        obs_track[good],
        R,
        C,
        mdl,
    )
    print(
        f"      re-split soft valley (metric fix): median reproj "
        f"{med0:.2f} -> {float(np.median(err)):.2f}px"
    )
    # final
    X = triangulate(obs_frame[good], obs_uv[good], obs_track[good], ntracks, R, C, mdl)
    cnt = np.bincount(obs_track[good], minlength=ntracks)
    err, zc = reproj_errors(X, obs_frame, obs_uv, obs_track, R, C, mdl)
    err[cnt[obs_track] < 2] = np.inf
    zlo, zhi = 0.4 * Zs[obs_frame], 2.5 * Zs[obs_frame]
    good = (err < max(3.0, 6.0 * float(np.median(err[good])))) & (zc > zlo) & (zc < zhi)
    cnt = np.bincount(obs_track[good], minlength=ntracks)
    good &= cnt[obs_track] >= 2
    obs_frame, obs_uv, obs_track, err = (obs_frame[good], obs_uv[good], obs_track[good], err[good])
    keep_tracks = np.unique(obs_track)
    remap = -np.ones(ntracks, np.int64)
    remap[keep_tracks] = np.arange(len(keep_tracks))
    obs_track = remap[obs_track].astype(np.int32)
    X = X[keep_tracks]
    track_err = np.zeros(len(keep_tracks))
    np.maximum.at(track_err, obs_track, err)
    rms = float(np.sqrt(np.mean(err**2)))
    print(
        f"      final: {len(X)} landmarks, {len(obs_uv)} obs, rms {rms:.2f}px "
        f"(median {np.median(err):.2f}px)"
    )
    return dict(
        X=X,
        track_err=track_err,
        obs_frame=obs_frame,
        obs_uv=obs_uv,
        obs_track=obs_track,
        err=err,
        R=R,
        C=C,
        rms=rms,
        history=history,
    )


def bundle_adjust_grouped(frames, pairs, mdl, R0, C0, args, group_of):
    """Breathing-robust registration. The skin deforms between scan passes, so a
    single rigid bundle can't explain frames captured minutes apart -- it puts
    cross-pass features at compromise 3D positions, which then ghost in the
    texture. Instead we register each breathing-coherent GROUP (a scan row, whose
    frames are seconds apart) rigidly on its OWN tracks, then align the groups to
    each other with a per-group 3D translation solved from the cross-group
    feature matches (the deformable step). Within a group geometry is rigid and
    single; between groups, breathing becomes a small shift the alignment removes
    and the group-owned texture compositing renders once instead of doubling.

    group_of: dict frame.idx -> group id. Returns the bundle_adjust dict shape."""
    gids = sorted(set(group_of.values()))
    intra = {g: [] for g in gids}
    cross = []
    for p in pairs:
        ga, gb = group_of[p.i], group_of[p.j]
        if ga == gb:
            intra[ga].append(p)
        else:
            cross.append(p)

    R = R0.copy()
    C = C0.copy()
    Xparts, ofr, ouv, otr, terr = [], [], [], [], []
    pt2track = {}  # (frame, x_round, y_round) -> global track
    grp_of_track = []  # group id per global track
    tbase = 0

    def key(fi, pt):
        return (int(fi), round(float(pt[0]), 2), round(float(pt[1]), 2))

    for g in gids:
        ng = sum(1 for i in group_of if group_of[i] == g)
        print(f"      -- group {g}: {ng} frames, {len(intra[g])} intra-pairs")
        ba = bundle_adjust(frames, intra[g], mdl, R0, C0, args)
        for i in group_of:
            if group_of[i] == g:
                R[i] = ba["R"][i]
                C[i] = ba["C"][i]
        tt = ba["obs_track"].astype(np.int64) + tbase
        for fi, pt, t in zip(ba["obs_frame"], ba["obs_uv"], tt):
            pt2track[key(fi, pt)] = int(t)
        Xparts.append(ba["X"])
        ofr.append(ba["obs_frame"])
        ouv.append(ba["obs_uv"])
        otr.append(tt)
        terr.append(ba["track_err"])
        grp_of_track.extend([g] * len(ba["X"]))
        tbase += len(ba["X"])

    Xg = np.concatenate(Xparts) if Xparts else np.zeros((0, 3))
    grp_of_track = np.array(grp_of_track)

    # cross-group landmark correspondences (dedup by track pair)
    corr = {}
    for p in cross:
        for a, b in zip(p.src_kp, p.dst_kp):
            ta = pt2track.get(key(p.i, frames[p.i].kp[a].pt))
            tb = pt2track.get(key(p.j, frames[p.j].kp[b].pt))
            if ta is not None and tb is not None and grp_of_track[ta] != grp_of_track[tb]:
                corr[(ta, tb)] = True

    gidx = {g: k for k, g in enumerate(gids)}
    t_solved = np.zeros((len(gids), 3))
    if args.group_align != "none" and corr:
        # per-group translation t_g minimising  sum |(Xa+t_a)-(Xb+t_b)|^2
        #                                        + lam sum |t_g|^2  (proprioception anchor / gauge)
        M = np.zeros((len(gids), len(gids)))
        rhs = np.zeros((len(gids), 3))
        for ta, tb in corr:
            ga, gb = gidx[grp_of_track[ta]], gidx[grp_of_track[tb]]
            d = Xg[ta] - Xg[tb]
            M[ga, ga] += 1
            M[gb, gb] += 1
            M[ga, gb] -= 1
            M[gb, ga] -= 1
            rhs[ga] -= d
            rhs[gb] += d
        lam = 1.0
        t_solved = np.linalg.solve(M + lam * np.eye(len(gids)), rhs)
        # The dot ghosting is an IN-PLANE displacement (a feature's ortho-texture
        # position depends on its x,y, not its depth). The along-optical-axis
        # component, by contrast, is the fx/depth-degenerate direction where each
        # small group floats unreliably -- fitting it drags weakly-constrained end
        # rows many mm. So keep only the tangential correction and leave depth to
        # proprioception (which the per-group BA already pins via the standoff).
        nhat = np.array([R[i][:, 2] for i in group_of]).mean(0)
        nhat /= np.linalg.norm(nhat)
        t_solved -= (t_solved @ nhat)[:, None] * nhat
        for g in gids:
            for i in group_of:
                if group_of[i] == g:
                    C[i] = C[i] + t_solved[gidx[g]]
            Xg[grp_of_track == g] += t_solved[gidx[g]]
        mags = np.linalg.norm(t_solved, axis=1)
        print(
            f"      group alignment ({len(corr)} cross-corr): "
            f"|t| per group " + " ".join(f"{g}:{mags[gidx[g]]:.1f}" for g in gids) + " mm"
        )
    elif args.group_align != "none":
        print("      ! no cross-group correspondences -- groups left at proprioception")

    obs_frame = np.concatenate(ofr).astype(np.int32)
    obs_uv = np.concatenate(ouv)
    obs_track = np.concatenate(otr).astype(np.int32)
    track_err = np.concatenate(terr)
    err, _ = reproj_errors(Xg, obs_frame, obs_uv, obs_track, R, C, mdl)
    rms = float(np.sqrt(np.mean(err**2)))
    print(
        f"      grouped final: {len(gids)} groups, {len(Xg)} landmarks, "
        f"{len(obs_uv)} obs, rms {rms:.2f}px (median {np.median(err):.2f}px)"
    )
    return dict(
        X=Xg,
        track_err=track_err,
        obs_frame=obs_frame,
        obs_uv=obs_uv,
        obs_track=obs_track,
        err=err,
        R=R,
        C=C,
        rms=rms,
        history=[],
    )
