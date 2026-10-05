"""Surface fitting, unwrapping, and deformable map alignment."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .registration_geometry import undistort_image_points_to_normalized_camera


# ----------------------------------------------------------------------------
# surface heightfield fit  z = f(x, y)
# ----------------------------------------------------------------------------
class Surface:
    def __init__(self, xs, ys, zgrid, support=None):
        self.xs, self.ys, self.z = xs, ys, zgrid
        hs_x = xs[1] - xs[0]
        hs_y = ys[1] - ys[0]
        self.gy, self.gx = np.gradient(zgrid, hs_y, hs_x)
        # bool grid: where the heightfield is supported by landmarks (vs
        # extrapolated). Texturing is restricted to this region.
        self.support = support if support is not None else np.ones_like(zgrid, bool)

    def _interp(self, grid, x, y):
        xi = np.clip((x - self.xs[0]) / (self.xs[1] - self.xs[0]), 0, len(self.xs) - 1.001)
        yi = np.clip((y - self.ys[0]) / (self.ys[1] - self.ys[0]), 0, len(self.ys) - 1.001)
        x0 = np.floor(xi).astype(int)
        y0 = np.floor(yi).astype(int)
        fx_ = xi - x0
        fy_ = yi - y0
        g = grid
        return (g[y0, x0] * (1 - fx_) + g[y0, x0 + 1] * fx_) * (1 - fy_) + (
            g[y0 + 1, x0] * (1 - fx_) + g[y0 + 1, x0 + 1] * fx_
        ) * fy_

    def height(self, x, y):
        return self._interp(self.z, x, y)

    def grad(self, x, y):
        return self._interp(self.gx, x, y), self._interp(self.gy, x, y)

    def normal(self, x, y, up_sign=1.0):
        gx, gy = self.grad(x, y)
        n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1) * up_sign
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def supported(self, x, y):
        xi = np.clip(
            np.round((x - self.xs[0]) / (self.xs[1] - self.xs[0])).astype(int), 0, len(self.xs) - 1
        )
        yi = np.clip(
            np.round((y - self.ys[0]) / (self.ys[1] - self.ys[0])).astype(int), 0, len(self.ys) - 1
        )
        return self.support[yi, xi]


def fit_surface_heightfield(X, bounds, pitch, smooth, robust_iters=3, w0=None):
    x0, x1, y0, y1 = bounds
    xs = np.arange(x0, x1 + pitch, pitch)
    ys = np.arange(y0, y1 + pitch, pitch)
    nx, ny = len(xs), len(ys)
    N = nx * ny

    px = np.clip((X[:, 0] - x0) / pitch, 0, nx - 1.001)
    py = np.clip((X[:, 1] - y0) / pitch, 0, ny - 1.001)
    ix = np.floor(px).astype(int)
    iy = np.floor(py).astype(int)
    fx_ = px - ix
    fy_ = py - iy
    rows = np.repeat(np.arange(len(X)), 4)
    cols = np.stack(
        [iy * nx + ix, iy * nx + ix + 1, (iy + 1) * nx + ix, (iy + 1) * nx + ix + 1], 1
    ).ravel()
    vals = np.stack([(1 - fx_) * (1 - fy_), fx_ * (1 - fy_), (1 - fx_) * fy_, fx_ * fy_], 1).ravel()
    A = sp.coo_matrix((vals, (rows, cols)), shape=(len(X), N)).tocsr()

    # second-difference smoothness in x and y
    def second_diff(n_outer, n_inner, stride_outer, stride_inner):
        r, c, v = [], [], []
        k = 0
        for o in range(n_outer):
            for i in range(1, n_inner - 1):
                base = o * stride_outer + i * stride_inner
                r += [k, k, k]
                c += [base - stride_inner, base, base + stride_inner]
                v += [1.0, -2.0, 1.0]
                k += 1
        return sp.coo_matrix((v, (r, c)), shape=(k, N)).tocsr()

    Sx = second_diff(ny, nx, nx, 1)
    Sy = second_diff(nx, ny, 1, nx)
    S = sp.vstack([Sx, Sy]) * smooth
    reg = (S.T @ S + sp.eye(N) * 1e-6).tocsc()

    # base weight: landmarks seen from more views triangulate far more
    # accurately in depth (narrow FOV -> depth noise ~ Z^2/(fx*baseline))
    wb = np.ones(len(X)) if w0 is None else np.asarray(w0, float)
    w = wb.copy()
    z = X[:, 2]
    sol = None
    for it in range(robust_iters):
        W = sp.diags(w)
        lhs = (A.T @ W @ A + reg).tocsc()
        rhs = A.T @ (w * z)
        sol = spla.spsolve(lhs, rhs)
        r = A @ sol - z
        sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = wb * np.where(np.abs(r) < 2.0 * sigma, 1.0, 2.0 * sigma / np.abs(r))
        w[np.abs(r) > 6 * sigma] = 0.0
    r = A @ sol - z
    inl = np.abs(r) < 6 * (1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9)
    rmsr = float(np.sqrt(np.mean(r[inl] ** 2)))
    print(
        f"      surface grid {nx}x{ny} @ {pitch}mm, landmark->surface rms "
        f"{rmsr:.3f}mm ({(~inl).sum()} outliers)"
    )
    # support mask: nodes touched by an inlier landmark, dilated ~6mm
    from scipy.ndimage import binary_dilation

    touched = np.zeros(N, bool)
    touched[cols.reshape(-1, 4)[inl].ravel()] = True
    sup = binary_dilation(touched.reshape(ny, nx), iterations=max(1, int(round(6.0 / pitch))))
    return Surface(xs, ys, sol.reshape(ny, nx), support=sup), rmsr


# ----------------------------------------------------------------------------
# arc-length texture parameterisation
# ----------------------------------------------------------------------------
class TexParam:
    """Unwrap to (u, v) mm. u = gantry x; v = arc length along the surface's mean
    profile in gantry y. Anchored to the gantry (proprioception) frame via the
    rig gauge (base_R, base_t) so the same physical skin maps to the same (u, v)
    across scans -- the stable parameterisation longitudinal lesion tracking needs
    (the per-scan world gauge is removed). base_R=None falls back to world x/y."""

    def __init__(self, surf: Surface, base_R=None, base_t=None):
        self.surf = surf
        self.bR = np.eye(3) if base_R is None else np.asarray(base_R)
        self.bt = np.zeros(3) if base_t is None else np.asarray(base_t)
        # surface grid -> gantry frame: G = base_R^T (P_world - base_t) = (P-bt)@bR
        Xs, Ys = np.meshgrid(surf.xs, surf.ys)
        Pg = (np.stack([Xs, Ys, surf.z], -1) - self.bt) @ self.bR
        gy, gz = Pg[..., 1], Pg[..., 2]
        sup = surf.support
        # mean gz(gy) profile over supported cells, binned in gantry y
        gyf, gzf = gy[sup], gz[sup]
        pitch = abs(surf.ys[1] - surf.ys[0])
        n = max(8, int(np.ptp(gyf) / pitch)) if len(gyf) else 8
        edges = np.linspace(gyf.min(), gyf.max(), n + 1) if len(gyf) else np.linspace(0, 1, n + 1)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(np.digitize(gyf, edges) - 1, 0, n - 1)
        prof = np.array([gzf[idx == k].mean() if np.any(idx == k) else np.nan for k in range(n)])
        ok = ~np.isnan(prof)
        prof = np.interp(ctr, ctr[ok], prof[ok]) if ok.any() else np.zeros(n)
        dy = ctr[1] - ctr[0] if n > 1 else 1.0
        ds = np.sqrt(1.0 + np.gradient(prof, dy) ** 2)
        self.gy = ctr
        self.s = np.concatenate([[0], np.cumsum((ds[1:] + ds[:-1]) * 0.5 * dy)])

    def to_uv(self, x, y):
        z = self.surf.height(x, y)
        Pg = (np.stack([x, y, z], -1) - self.bt) @ self.bR
        return Pg[..., 0], np.interp(Pg[..., 1], self.gy, self.s)

    def to_xy(self, u, v):
        gy = np.interp(v, self.s, self.gy)
        # invert the (near-identity) world<->gantry map by Newton: find world
        # (x,y) whose surface point transforms to gantry (u, gy)
        x = np.array(u, float)
        y = np.array(gy, float)
        for _ in range(6):
            Pg = (np.stack([x, y, self.surf.height(x, y)], -1) - self.bt) @ self.bR
            x = x - (Pg[..., 0] - u)
            y = y - (Pg[..., 1] - gy)
        return x, y


def ray_surface_intersect(Cw, dirs, surf: Surface, t0, iters=8):
    """Intersect rays (origin Cw, unit dirs (N,3)) with the heightfield."""
    t = np.full(len(dirs), float(t0))
    for _ in range(iters):
        P = Cw + dirs * t[:, None]
        gx, gy = surf.grad(P[:, 0], P[:, 1])
        fz = surf.height(P[:, 0], P[:, 1])
        r = P[:, 2] - fz
        denom = dirs[:, 2] - (gx * dirs[:, 0] + gy * dirs[:, 1])
        denom = np.where(np.abs(denom) < 1e-6, np.sign(denom + 1e-12) * 1e-6, denom)
        t = t - r / denom
        t = np.clip(t, 1.0, 4.0 * t0)
    return Cw + dirs * t[:, None]


def compute_camera_frame_surface_footprint(f, R, C, mdl, surf, n_edge=6):
    """Footprint polygon on the surface (world pts), sampled along edges."""
    w, h = mdl.cx * 2, mdl.cy * 2
    ts = np.linspace(0, 1, n_edge, endpoint=False)
    edges = []
    cpx = [(0, 0), (w, 0), (w, h), (0, h)]
    for a, b in zip(cpx, cpx[1:] + cpx[:1]):
        for t in ts:
            edges.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    edges = np.array(edges)
    xu = undistort_image_points_to_normalized_camera(edges, mdl.fx, mdl.k1, mdl.cx, mdl.cy)
    d_cam = np.concatenate([xu, np.ones((len(xu), 1))], 1)
    d_w = d_cam @ R[f.idx].T
    d_w /= np.linalg.norm(d_w, axis=1, keepdims=True)
    return ray_surface_intersect(C[f.idx], d_w, surf, mdl.depth(f))


# ----------------------------------------------------------------------------
# deformable alignment (breathing): smooth per-frame warp in the unwrapped map
# ----------------------------------------------------------------------------
def align_frames_with_deformable_surface_warps(
    frames, pairs, R, C, mdl, surf, tp, reg=2.0, max_per_pair=200, order=1
):
    """Remove the residual breathing misalignment that makes stitch seams cross
    moles. Each overlapping pair gives feature correspondences that the rigid
    registration places at slightly different (u,v) on the unwrapped map (the
    breathing offset). Solve a per-frame AFFINE warp of the map so warped
    correspondences agree, with a corner-displacement penalty keeping each warp
    small and smooth at breathing scale. The 3D landmark surface is untouched --
    only the 2D texture content is aligned. Returns (Mu, Mv, umid, vmid): per-
    frame affine params for the u and v corrections about (umid, vmid), or None.

    corr_u_f(u,v) = Mu[f]@[u-umid, v-vmid, 1];  corr_v_f similarly with Mv[f]."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = len(frames)

    def to_uv(pts, fi):
        xu = undistort_image_points_to_normalized_camera(pts.astype(float), mdl.fx, mdl.k1, mdl.cx, mdl.cy)
        dcam = np.concatenate([xu, np.ones((len(xu), 1))], 1)
        dw = dcam @ R[fi].T
        dw /= np.linalg.norm(dw, axis=1, keepdims=True)
        P = ray_surface_intersect(C[fi], dw, surf, mdl.depth(frames[fi]))
        return np.stack(tp.to_uv(P[:, 0], P[:, 1]), 1)

    # collect correspondences in map (u,v); centre for conditioning
    corr = []
    rng = np.random.default_rng(0)
    for p in pairs:
        m = len(p.src)
        if m == 0:
            continue
        sel = rng.permutation(m)[:max_per_pair]
        corr.append((p.i, to_uv(p.src[sel], p.i), p.j, to_uv(p.dst[sel], p.j)))
    if not corr:
        return None
    allu = np.concatenate([c[1] for c in corr] + [c[3] for c in corr])
    umid, vmid = float(allu[:, 0].mean()), float(allu[:, 1].mean())

    nb = 3 if order == 1 else 6

    def basis(du, dv):
        b = [du, dv, 1.0]
        if nb == 6:
            # quadratic terms scaled to linear-column magnitude (mm^2/50)
            # for LSQR conditioning; geom/GpuGeom apply the same 1/50
            b += [du * du / 50.0, du * dv / 50.0, dv * dv / 50.0]
        return b

    II, JJ, VV, bu, bv = [], [], [], [], []
    row = 0
    for fi, Ui, fj, Uj in corr:
        for k in range(len(Ui)):
            ai = basis(Ui[k, 0] - umid, Ui[k, 1] - vmid)
            aj = basis(Uj[k, 0] - umid, Uj[k, 1] - vmid)
            II += [row] * (2 * nb)
            JJ += [nb * fi + t for t in range(nb)] + [nb * fj + t for t in range(nb)]
            VV += ai + [-x for x in aj]
            bu.append(-(Ui[k, 0] - Uj[k, 0]))
            bv.append(-(Ui[k, 1] - Uj[k, 1]))
            row += 1
    ndata = row
    # regularise: pull each frame's footprint-sample displacement toward 0
    # (penalises warp magnitude uniformly in mm -> small, smooth = breathing
    # only). The quadratic warp gets denser anchors: 6 params/axis need
    # well-spread samples to stay bounded across the whole footprint.
    for f in frames:
        fp = compute_camera_frame_surface_footprint(f, R, C, mdl, surf)
        fu, fv = tp.to_uv(fp[:, 0], fp[:, 1])
        for c in range(0, len(fu), max(1, len(fu) // (4 if nb == 3 else 10))):
            bb = basis(fu[c] - umid, fv[c] - vmid)
            II += [row] * nb
            JJ += [nb * f.idx + t for t in range(nb)]
            VV += [reg * x for x in bb]
            bu.append(0.0)
            bv.append(0.0)
            row += 1

    A = sp.coo_matrix((VV, (II, JJ)), shape=(row, nb * n)).tocsr()
    bu = np.array(bu)
    bv = np.array(bv)
    Mu = spla.lsqr(A, bu)[0].reshape(n, nb)
    Mv = spla.lsqr(A, bv)[0].reshape(n, nb)
    # runaway guard: a per-frame "breathing" correction of tens of mm is
    # never physical -- it means the frame's few (usually grazing-flank,
    # one-sided) correspondences conflict with its BA pose, and warping its
    # content that far smears or tears the map. Those frames keep their
    # proprioception-anchored pose instead.
    mag = np.hypot(Mu[:, 2], Mv[:, 2])
    lim = max(5.0, 8.0 * float(np.median(mag)))
    bad = np.where(mag > lim)[0]
    if len(bad):
        print(
            f"      ! zeroed runaway warp on {len(bad)} frame(s) "
            f"(|centre shift| > {lim:.1f}mm): "
            + ", ".join(f"frame {frames[b].idx} ({mag[b]:.1f}mm)" for b in bad[:6])
        )
        Mu[bad] = 0.0
        Mv[bad] = 0.0
        mag = np.hypot(Mu[:, 2], Mv[:, 2])
    # report correction magnitude + how much pair disagreement the warp
    # removed: the POST residual is what still cuts hairs at HF-winner seams
    pre = np.hypot(bu[:ndata], bv[:ndata])
    Ad = A[:ndata]
    post = np.hypot(Ad @ Mu.ravel() - bu[:ndata], Ad @ Mv.ravel() - bv[:ndata])
    print(
        f"      deformable align (order {order}): {row} eqns, "
        f"|centre shift| median {np.median(mag):.2f} max {mag.max():.2f} mm; "
        f"pair residual median {np.median(pre):.2f} -> "
        f"{np.median(post):.2f} mm (p90 {np.percentile(pre, 90):.2f} -> "
        f"{np.percentile(post, 90):.2f})"
    )
    return Mu, Mv, umid, vmid
