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

    def _interp(self, grid, surface_x, surface_y):
        xi = np.clip((surface_x - self.xs[0]) / (self.xs[1] - self.xs[0]), 0, len(self.xs) - 1.001)
        yi = np.clip((surface_y - self.ys[0]) / (self.ys[1] - self.ys[0]), 0, len(self.ys) - 1.001)
        x0 = np.floor(xi).astype(int)
        y0 = np.floor(yi).astype(int)
        fx_ = xi - x0
        fy_ = yi - y0
        gauge = grid
        return (gauge[y0, x0] * (1 - fx_) + gauge[y0, x0 + 1] * fx_) * (1 - fy_) + (
            gauge[y0 + 1, x0] * (1 - fx_) + gauge[y0 + 1, x0 + 1] * fx_
        ) * fy_

    def height(self, surface_x, surface_y):
        return self._interp(self.z, surface_x, surface_y)

    def grad(self, surface_x, surface_y):
        return self._interp(self.gx, surface_x, surface_y), self._interp(self.gy, surface_x, surface_y)

    def normal(self, surface_x, surface_y, up_sign=1.0):
        gx, gy = self.grad(surface_x, surface_y)
        item_count = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1) * up_sign
        return item_count / np.linalg.norm(item_count, axis=-1, keepdims=True)

    def supported(self, surface_x, surface_y):
        xi = np.clip(
            np.round((surface_x - self.xs[0]) / (self.xs[1] - self.xs[0])).astype(int), 0, len(self.xs) - 1
        )
        yi = np.clip(
            np.round((surface_y - self.ys[0]) / (self.ys[1] - self.ys[0])).astype(int), 0, len(self.ys) - 1
        )
        return self.support[yi, xi]


def fit_surface_heightfield(
    landmark_points,
    surface_bounds=None,
    grid_pitch_mm=None,
    smoothness_weight=None,
    robust_fit_iteration_count=3,
    initial_landmark_weights=None,
    **legacy_options,
):
    surface_bounds = legacy_options.pop("bounds", surface_bounds)
    grid_pitch_mm = legacy_options.pop("pitch", grid_pitch_mm)
    smoothness_weight = legacy_options.pop("smooth", smoothness_weight)
    robust_fit_iteration_count = legacy_options.pop("robust_iters", robust_fit_iteration_count)
    initial_landmark_weights = legacy_options.pop("w0", initial_landmark_weights)
    if legacy_options:
        unexpected_option = next(iter(legacy_options))
        raise TypeError(f"fit_surface_heightfield got an unexpected keyword argument {unexpected_option!r}")
    if surface_bounds is None or grid_pitch_mm is None or smoothness_weight is None:
        raise TypeError("fit_surface_heightfield requires bounds, pitch, and smoothness values")
    surface_x_min, surface_x_max, surface_y_min, surface_y_max = surface_bounds
    surface_x_grid = np.arange(surface_x_min, surface_x_max + grid_pitch_mm, grid_pitch_mm)
    surface_y_grid = np.arange(surface_y_min, surface_y_max + grid_pitch_mm, grid_pitch_mm)
    surface_x_node_count, surface_y_node_count = len(surface_x_grid), len(surface_y_grid)
    surface_grid_node_count = surface_x_node_count * surface_y_node_count

    landmark_grid_x_positions = np.clip((landmark_points[:, 0] - surface_x_min) / grid_pitch_mm, 0, surface_x_node_count - 1.001)
    landmark_grid_y_positions = np.clip((landmark_points[:, 1] - surface_y_min) / grid_pitch_mm, 0, surface_y_node_count - 1.001)
    landmark_grid_x_indices = np.floor(landmark_grid_x_positions).astype(int)
    landmark_grid_y_indices = np.floor(landmark_grid_y_positions).astype(int)
    fractional_grid_x = landmark_grid_x_positions - landmark_grid_x_indices
    fractional_grid_y = landmark_grid_y_positions - landmark_grid_y_indices
    interpolation_row_indices = np.repeat(np.arange(len(landmark_points)), 4)
    interpolation_column_indices = np.stack(
        [landmark_grid_y_indices * surface_x_node_count + landmark_grid_x_indices, landmark_grid_y_indices * surface_x_node_count + landmark_grid_x_indices + 1, (landmark_grid_y_indices + 1) * surface_x_node_count + landmark_grid_x_indices, (landmark_grid_y_indices + 1) * surface_x_node_count + landmark_grid_x_indices + 1], 1
    ).ravel()
    interpolation_weights = np.stack(
        [
            (1 - fractional_grid_x) * (1 - fractional_grid_y),
            fractional_grid_x * (1 - fractional_grid_y),
            (1 - fractional_grid_x) * fractional_grid_y,
            fractional_grid_x * fractional_grid_y,
        ],
        1,
    ).ravel()
    landmark_interpolation_matrix = sp.coo_matrix(
        (interpolation_weights, (interpolation_row_indices, interpolation_column_indices)), shape=(len(landmark_points), surface_grid_node_count)
    ).tocsr()

    # second-difference smoothness in x and y
    def second_diff(n_outer, n_inner, stride_outer, stride_inner):
        row_indices, column_indices, second_difference_values = [], [], []
        constraint_index = 0
        for outer_index in range(n_outer):
            for inner_index in range(1, n_inner - 1):
                surface_grid_node_index = outer_index * stride_outer + inner_index * stride_inner
                row_indices.extend([constraint_index] * 3)
                column_indices.extend(
                    [surface_grid_node_index - stride_inner, surface_grid_node_index, surface_grid_node_index + stride_inner]
                )
                second_difference_values.extend([1.0, -2.0, 1.0])
                constraint_index += 1
        return sp.coo_matrix(
            (second_difference_values, (row_indices, column_indices)),
            shape=(constraint_index, surface_grid_node_count),
        ).tocsr()

    surface_x_second_difference_matrix = second_diff(surface_y_node_count, surface_x_node_count, surface_x_node_count, 1)
    surface_y_second_difference_matrix = second_diff(surface_x_node_count, surface_y_node_count, 1, surface_x_node_count)
    smoothness_matrix = sp.vstack([surface_x_second_difference_matrix, surface_y_second_difference_matrix]) * smoothness_weight
    regularization_matrix = (smoothness_matrix.T @ smoothness_matrix + sp.eye(surface_grid_node_count) * 1e-6).tocsc()

    # base weight: landmarks seen from more views triangulate far more
    # accurately in depth (narrow FOV -> depth noise ~ Z^2/(fx*baseline))
    base_landmark_weights = np.ones(len(landmark_points)) if initial_landmark_weights is None else np.asarray(initial_landmark_weights, float)
    robust_weights = base_landmark_weights.copy()
    landmark_heights = landmark_points[:, 2]
    surface_heights = None
    for robust_fit_iteration in range(robust_fit_iteration_count):
        observation_weight_matrix = sp.diags(robust_weights)
        normal_matrix = (
            landmark_interpolation_matrix.T
            @ observation_weight_matrix
            @ landmark_interpolation_matrix
            + regularization_matrix
        ).tocsc()
        normal_rhs = landmark_interpolation_matrix.T @ (robust_weights * landmark_heights)
        surface_heights = spla.spsolve(normal_matrix, normal_rhs)
        landmark_height_residuals = landmark_interpolation_matrix @ surface_heights - landmark_heights
        sigma = 1.4826 * np.median(
            np.abs(landmark_height_residuals - np.median(landmark_height_residuals))
        ) + 1e-9
        robust_weights = base_landmark_weights * np.where(
            np.abs(landmark_height_residuals) < 2.0 * sigma,
            1.0,
            2.0 * sigma / np.abs(landmark_height_residuals),
        )
        robust_weights[np.abs(landmark_height_residuals) > 6 * sigma] = 0.0
    landmark_height_residuals = landmark_interpolation_matrix @ surface_heights - landmark_heights
    inlier_mask = np.abs(landmark_height_residuals) < 6 * (
        1.4826 * np.median(np.abs(landmark_height_residuals - np.median(landmark_height_residuals))) + 1e-9
    )
    surface_fit_rms = float(np.sqrt(np.mean(landmark_height_residuals[inlier_mask] ** 2)))
    print(
        f"      surface grid {surface_x_node_count}x{surface_y_node_count} @ {grid_pitch_mm}mm, landmark->surface rms "
        f"{surface_fit_rms:.3f}mm ({(~inlier_mask).sum()} outliers)"
    )
    # support mask: nodes touched by an inlier landmark, dilated ~6mm
    from scipy.ndimage import binary_dilation

    touched = np.zeros(surface_grid_node_count, bool)
    touched[interpolation_column_indices.reshape(-1, 4)[inlier_mask].ravel()] = True
    surface_support_mask = binary_dilation(touched.reshape(surface_y_node_count, surface_x_node_count), iterations=max(1, int(round(6.0 / grid_pitch_mm))))
    return Surface(surface_x_grid, surface_y_grid, surface_heights.reshape(surface_y_node_count, surface_x_node_count), support=surface_support_mask), surface_fit_rms


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
        item_count = max(8, int(np.ptp(gyf) / pitch)) if len(gyf) else 8
        edges = np.linspace(gyf.min(), gyf.max(), item_count + 1) if len(gyf) else np.linspace(0, 1, item_count + 1)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(np.digitize(gyf, edges) - 1, 0, item_count - 1)
        prof = np.array([gzf[idx == item_index].mean() if np.any(idx == item_index) else np.nan for item_index in range(item_count)])
        ok = ~np.isnan(prof)
        prof = np.interp(ctr, ctr[ok], prof[ok]) if ok.any() else np.zeros(item_count)
        dy = ctr[1] - ctr[0] if item_count > 1 else 1.0
        ds = np.sqrt(1.0 + np.gradient(prof, dy) ** 2)
        self.gy = ctr
        self.s = np.concatenate([[0], np.cumsum((ds[1:] + ds[:-1]) * 0.5 * dy)])

    def to_uv(self, surface_x, surface_y):
        world_z = self.surf.height(surface_x, surface_y)
        Pg = (np.stack([surface_x, surface_y, world_z], -1) - self.bt) @ self.bR
        return Pg[..., 0], np.interp(Pg[..., 1], self.gy, self.s)

    def to_xy(self, texture_u, texture_v):
        gy = np.interp(texture_v, self.s, self.gy)
        # invert the (near-identity) world<->gantry map by Newton: find world
        # (x,y) whose surface point transforms to gantry (u, gy)
        surface_x = np.array(texture_u, float)
        surface_y = np.array(gy, float)
        for iteration_index in range(6):
            Pg = (np.stack([surface_x, surface_y, self.surf.height(surface_x, surface_y)], -1) - self.bt) @ self.bR
            surface_x = surface_x - (Pg[..., 0] - texture_u)
            surface_y = surface_y - (Pg[..., 1] - gy)
        return surface_x, surface_y


def ray_surface_intersect(Cw, dirs, surf: Surface, t0, iters=8):
    """Intersect rays (origin Cw, unit dirs (N,3)) with the heightfield."""
    threshold = np.full(len(dirs), float(t0))
    for iteration_index in range(iters):
        points = Cw + dirs * threshold[:, None]
        gx, gy = surf.grad(points[:, 0], points[:, 1])
        fz = surf.height(points[:, 0], points[:, 1])
        camera_rotation = points[:, 2] - fz
        denom = dirs[:, 2] - (gx * dirs[:, 0] + gy * dirs[:, 1])
        denom = np.where(np.abs(denom) < 1e-6, np.sign(denom + 1e-12) * 1e-6, denom)
        threshold = threshold - camera_rotation / denom
        threshold = np.clip(threshold, 1.0, 4.0 * t0)
    return Cw + dirs * threshold[:, None]


def compute_camera_frame_surface_footprint(frame, camera_rotations, camera_centers, rig_model, surf, n_edge=6):
    """Footprint polygon on the surface (world pts), sampled along edges."""
    weight, image_height = rig_model.cx * 2, rig_model.cy * 2
    ts = np.linspace(0, 1, n_edge, endpoint=False)
    edges = []
    cpx = [(0, 0), (weight, 0), (weight, image_height), (0, image_height)]
    for first_value, second_value in zip(cpx, cpx[1:] + cpx[:1]):
        for threshold in ts:
            edges.append((first_value[0] + (second_value[0] - first_value[0]) * threshold, first_value[1] + (second_value[1] - first_value[1]) * threshold))
    edges = np.array(edges)
    xu = undistort_image_points_to_normalized_camera(edges, rig_model.fx, rig_model.k1, rig_model.cx, rig_model.cy)
    d_cam = np.concatenate([xu, np.ones((len(xu), 1))], 1)
    d_w = d_cam @ camera_rotations[frame.idx].T
    d_w /= np.linalg.norm(d_w, axis=1, keepdims=True)
    return ray_surface_intersect(camera_centers[frame.idx], d_w, surf, rig_model.depth(frame))


# ----------------------------------------------------------------------------
# deformable alignment (breathing): smooth per-frame warp in the unwrapped map
# ----------------------------------------------------------------------------
def align_frames_with_deformable_surface_warps(
    frames, pairs, camera_rotations, camera_centers, rig_model, surf, texture_parameters, reg=2.0, max_per_pair=200, order=1
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

    item_count = len(frames)

    def to_uv(pts, fi):
        xu = undistort_image_points_to_normalized_camera(pts.astype(float), rig_model.fx, rig_model.k1, rig_model.cx, rig_model.cy)
        dcam = np.concatenate([xu, np.ones((len(xu), 1))], 1)
        dw = dcam @ camera_rotations[fi].T
        dw /= np.linalg.norm(dw, axis=1, keepdims=True)
        points = ray_surface_intersect(camera_centers[fi], dw, surf, rig_model.depth(frames[fi]))
        return np.stack(texture_parameters.to_uv(points[:, 0], points[:, 1]), 1)

    # collect correspondences in map (u,v); centre for conditioning
    corr = []
    rng = np.random.default_rng(0)
    for point in pairs:
        mask = len(point.src)
        if mask == 0:
            continue
        sel = rng.permutation(mask)[:max_per_pair]
        corr.append((point.i, to_uv(point.src[sel], point.i), point.j, to_uv(point.dst[sel], point.j)))
    if not corr:
        return None
    allu = np.concatenate([candidate[1] for candidate in corr] + [candidate[3] for candidate in corr])
    umid, vmid = float(allu[:, 0].mean()), float(allu[:, 1].mean())

    nb = 3 if order == 1 else 6

    def basis(du, dv):
        surface_warp_basis = [du, dv, 1.0]
        if nb == 6:
            # quadratic terms scaled to linear-column magnitude (mm^2/50)
            # for LSQR conditioning; geom/GpuGeom apply the same 1/50
            surface_warp_basis += [du * du / 50.0, du * dv / 50.0, dv * dv / 50.0]
        return surface_warp_basis

    II, JJ, VV, bu, bv = [], [], [], [], []
    row = 0
    for fi, Ui, fj, Uj in corr:
        for item_index in range(len(Ui)):
            ai = basis(Ui[item_index, 0] - umid, Ui[item_index, 1] - vmid)
            aj = basis(Uj[item_index, 0] - umid, Uj[item_index, 1] - vmid)
            II += [row] * (2 * nb)
            JJ += [nb * fi + threshold for threshold in range(nb)] + [nb * fj + threshold for threshold in range(nb)]
            VV += ai + [-surface_x for surface_x in aj]
            bu.append(-(Ui[item_index, 0] - Uj[item_index, 0]))
            bv.append(-(Ui[item_index, 1] - Uj[item_index, 1]))
            row += 1
    ndata = row
    # regularise: pull each frame's footprint-sample displacement toward 0
    # (penalises warp magnitude uniformly in mm -> small, smooth = breathing
    # only). The quadratic warp gets denser anchors: 6 params/axis need
    # well-spread samples to stay bounded across the whole footprint.
    for frame in frames:
        fp = compute_camera_frame_surface_footprint(frame, camera_rotations, camera_centers, rig_model, surf)
        fu, fv = texture_parameters.to_uv(fp[:, 0], fp[:, 1])
        for candidate in range(0, len(fu), max(1, len(fu) // (4 if nb == 3 else 10))):
            bb = basis(fu[candidate] - umid, fv[candidate] - vmid)
            II += [row] * nb
            JJ += [nb * frame.idx + threshold for threshold in range(nb)]
            VV += [reg * surface_x for surface_x in bb]
            bu.append(0.0)
            bv.append(0.0)
            row += 1

    points_a = sp.coo_matrix((VV, (II, JJ)), shape=(row, nb * item_count)).tocsr()
    bu = np.array(bu)
    bv = np.array(bv)
    Mu = spla.lsqr(points_a, bu)[0].reshape(item_count, nb)
    Mv = spla.lsqr(points_a, bv)[0].reshape(item_count, nb)
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
            + ", ".join(f"frame {frames[second_value].idx} ({mag[second_value]:.1f}mm)" for second_value in bad[:6])
        )
        Mu[bad] = 0.0
        Mv[bad] = 0.0
        mag = np.hypot(Mu[:, 2], Mv[:, 2])
    # report correction magnitude + how much pair disagreement the warp
    # removed: the POST residual is what still cuts hairs at HF-winner seams
    pre = np.hypot(bu[:ndata], bv[:ndata])
    Ad = points_a[:ndata]
    post = np.hypot(Ad @ Mu.ravel() - bu[:ndata], Ad @ Mv.ravel() - bv[:ndata])
    print(
        f"      deformable align (order {order}): {row} eqns, "
        f"|centre shift| median {np.median(mag):.2f} max {mag.max():.2f} mm; "
        f"pair residual median {np.median(pre):.2f} -> "
        f"{np.median(post):.2f} mm (p90 {np.percentile(pre, 90):.2f} -> "
        f"{np.percentile(post, 90):.2f})"
    )
    return Mu, Mv, umid, vmid
