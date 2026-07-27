"""Mesh, diagnostic, and interactive-viewer exports."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .registration_geometry import undistort_norm


# ----------------------------------------------------------------------------
# mesh build + exports
# ----------------------------------------------------------------------------
def build_mesh(surf, tp, tex_bounds, wacc, ppmm, mesh_pitch, up_sign, mesh_smooth=(0.0, 0.0)):
    umin, vmin, umax, vmax = tex_bounds
    us = np.arange(umin, umax + mesh_pitch, mesh_pitch)
    vs = np.arange(vmin, vmax + mesh_pitch, mesh_pitch)
    nx, ny = len(us), len(vs)
    Ug, Vg = np.meshgrid(us, vs)
    Xg, Yg = tp.to_xy(Ug.ravel(), Vg.ravel())
    Zg = surf.height(Xg, Yg)
    nrm = surf.normal(Xg, Yg, up_sign).astype(np.float32)
    su, sv = mesh_smooth if mesh_smooth else (0.0, 0.0)
    if su > 0 or sv > 0:
        # DISPLAY-mesh smoothing. The render surface must stay tight to the
        # landmarks (texture placement error scales with surface error x
        # tan(view tilt)), but on a breathing subject that tight fit follows
        # the between-pass motion sheets into valleys/bumps that are not
        # anatomy. Smooth only the displayed height field -- anisotropic:
        # strong along U (normally a gentle taper), gentle across V (where the
        # measured cross-section carries the contour). The texture and its
        # (u,v) placement are untouched.
        Z2 = Zg.reshape(ny, nx).astype(np.float32)
        Z2 = cv2.GaussianBlur(
            Z2,
            (0, 0),
            sigmaX=max(su / mesh_pitch, 1e-6),
            sigmaY=max(sv / mesh_pitch, 1e-6),
            borderType=cv2.BORDER_REPLICATE,
        )
        Zg = Z2.ravel().astype(np.float64)
        dzdx = np.gradient(Z2, mesh_pitch, axis=1).ravel()
        dzdy = np.gradient(Z2, mesh_pitch, axis=0).ravel()
        n2 = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], 1)
        n2 /= np.linalg.norm(n2, axis=1, keepdims=True)
        if float(np.nanmean(np.sum(n2 * nrm, axis=1))) < 0:
            n2 = -n2
        nrm = n2.astype(np.float32)
        print(
            f"      mesh display smoothing: sigma {su:.0f}mm along-u / "
            f"{sv:.0f}mm along-v (render surface untouched)"
        )
    pos = np.stack([Xg, Yg, Zg], 1).astype(np.float32)
    # uv: u right, v measured from texture top row (row 0 = vmin)
    uvn = np.stack(
        [(Ug.ravel() - umin) / (umax - umin), 1.0 - (Vg.ravel() - vmin) / (vmax - vmin)], 1
    ).astype(np.float32)
    # validity from texture coverage
    H, W = wacc.shape
    tu = np.clip(((Ug.ravel() - umin) * ppmm).astype(int), 0, W - 1)
    tv = np.clip(((Vg.ravel() - vmin) * ppmm).astype(int), 0, H - 1)
    valid = wacc[tv, tu] > 0

    faces = []
    for j in range(ny - 1):
        a = j * nx + np.arange(nx - 1)
        quads = np.stack([a, a + 1, a + nx, a + nx + 1], 1)
        ok = valid[quads].all(1)
        q = quads[ok]
        faces.append(np.stack([q[:, 0], q[:, 1], q[:, 2]], 1))
        faces.append(np.stack([q[:, 1], q[:, 3], q[:, 2]], 1))
    faces = np.concatenate(faces).astype(np.int64)
    print(f"      mesh: {len(pos)} verts ({nx}x{ny}), {len(faces)} tris")
    return pos, nrm, uvn, faces


def export_obj(out_dir, pos, nrm, uvn, faces):
    mtl = os.path.join(out_dir, "surface_mesh.mtl")
    with open(mtl, "w") as fh:
        fh.write("newmtl skin\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd texture.jpg\n")
    objp = os.path.join(out_dir, "surface_mesh.obj")
    with open(objp, "w") as fh:
        fh.write("mtllib surface_mesh.mtl\nusemtl skin\n")
        np.savetxt(fh, pos, fmt="v %.4f %.4f %.4f")
        np.savetxt(fh, uvn, fmt="vt %.6f %.6f")
        np.savetxt(fh, nrm, fmt="vn %.4f %.4f %.4f")
        f1 = faces + 1
        np.savetxt(
            fh,
            np.column_stack(
                [
                    f1[:, 0],
                    f1[:, 0],
                    f1[:, 0],
                    f1[:, 1],
                    f1[:, 1],
                    f1[:, 1],
                    f1[:, 2],
                    f1[:, 2],
                    f1[:, 2],
                ]
            ),
            fmt="f %d/%d/%d %d/%d/%d %d/%d/%d",
        )
    print(f"      wrote {objp}")


def export_landmarks_ply(out_dir, X, track_err):
    e = np.clip(track_err / max(track_err.max(), 1e-6), 0, 1)
    col = np.stack([(e * 255), (1 - e) * 255, np.zeros_like(e)], 1).astype(np.uint8)
    path = os.path.join(out_dir, "landmarks.ply")
    with open(path, "wb") as fh:
        fh.write(
            (
                f"ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(X)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n"
            ).encode()
        )
        rec = np.zeros(len(X), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        rec["xyz"] = X.astype(np.float32)
        rec["rgb"] = col
        fh.write(rec.tobytes())
    print(f"      wrote {path}")


def make_overview_png(out_dir, frames, R, C, mdl, surf, X):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    step = max(1, len(surf.xs) // 60)
    Xs, Ys = np.meshgrid(surf.xs[::step], surf.ys[::step])
    ax.plot_surface(
        Xs, Ys, surf.z[::step, ::step], alpha=0.35, color="tan", linewidth=0, antialiased=True
    )
    sub = np.random.default_rng(0).permutation(len(X))[:4000]
    ax.scatter(X[sub, 0], X[sub, 1], X[sub, 2], s=0.5, c="firebrick", alpha=0.4)
    rows = np.array([f.row for f in frames])
    cmap = plt.colormaps["viridis"]
    for f in frames:
        cclr = cmap((f.row - rows.min()) / max(1, rows.max() - rows.min()))
        ax.scatter(*C[f.idx], color=cclr, s=12)
        tip = C[f.idx] + R[f.idx][:, 2] * mdl.depth(f) * 0.6
        ax.plot(*np.stack([C[f.idx], tip], 1), color=cclr, lw=0.5, alpha=0.6)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("3D registration: cameras (coloured by row), landmarks, fitted surface")
    # equal aspect
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    ctr = lims.mean(1)
    rad = (lims[:, 1] - lims[:, 0]).max() / 2
    ax.set_xlim3d(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim3d(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim3d(ctr[2] - rad, ctr[2] + rad)
    fig.tight_layout()
    path = os.path.join(out_dir, "overview.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"      wrote {path}")


# ----------------------------------------------------------------------------
# interactive viewer (self-contained HTML, embedded three.js)
# ----------------------------------------------------------------------------
def _b64(arr) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def export_viewer(out_dir, frames, R, C, mdl, pos, uvn, faces, X, track_err, stats):
    assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    three_b64 = orbit_b64 = ""
    for name, var in (("three.module.min.js", "three"), ("OrbitControls.js", "orbit")):
        p = os.path.join(assets, name)
        if os.path.exists(p):
            data = base64.b64encode(open(p, "rb").read()).decode()
            if var == "three":
                three_b64 = data
            else:
                orbit_b64 = data
    if not three_b64:
        print(
            "      ! scripts/assets/three.module.min.js missing -- viewer will "
            "fall back to CDN (needs internet)"
        )

    # cap the embedded copy at the WebGL max texture size of every desktop
    # GPU/browser of the last decade (16384; older mobile GPUs stop at 8192 --
    # this viewer is a desktop artifact). The full-res texture.jpg on disk is
    # untouched (that's the analysis artifact).
    tex_disk = cv2.imread(os.path.join(out_dir, "texture.jpg"), cv2.IMREAD_COLOR)
    max_dim = 16384
    if max(tex_disk.shape[:2]) > max_dim:
        s = max_dim / max(tex_disk.shape[:2])
        tex_disk = cv2.resize(tex_disk, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        print(
            f"      (viewer texture downscaled to {tex_disk.shape[1]}x"
            f"{tex_disk.shape[0]} for WebGL; texture.jpg keeps full res)"
        )
    ok, buf = cv2.imencode(".jpg", tex_disk, [cv2.IMWRITE_JPEG_QUALITY, 88])
    tex_uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    # camera frusta line segments (8 per frame) + colors by row
    rows = np.array([f.row for f in frames])
    rspan = max(1, rows.max() - rows.min())
    seg_pts, seg_cols, centers = [], [], []
    w, h = mdl.cx * 2, mdl.cy * 2
    cpx = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    for f in frames:
        Z = mdl.depth(f) * 0.35
        xu = undistort_norm(cpx, mdl.fx, mdl.k1, mdl.cx, mdl.cy)
        xc = np.concatenate([xu * Z, np.full((4, 1), Z)], 1)
        cw = xc @ R[f.idx].T + C[f.idx]
        t = (f.row - rows.min()) / rspan
        col = np.array([0.2 + 0.8 * t, 0.9 - 0.6 * t, 1.0 - 0.7 * t])
        for k in range(4):
            seg_pts += [C[f.idx], cw[k], cw[k], cw[(k + 1) % 4]]
            seg_cols += [col, col, col, col]
        centers.append(C[f.idx])
    seg_pts = np.array(seg_pts, np.float32)
    seg_cols = np.array(seg_cols, np.float32)

    sub = np.random.default_rng(0).permutation(len(X))[:40000]
    e = np.clip(track_err[sub] / max(np.percentile(track_err, 95), 1e-6), 0, 1)
    lm_col = np.stack([e, 1 - e, np.full_like(e, 0.15)], 1).astype(np.float32)

    payload = dict(
        mesh=dict(positions=_b64(pos), uvs=_b64(uvn), indices=_b64(faces.astype(np.uint32))),
        texture=tex_uri,
        landmarks=dict(positions=_b64(X[sub].astype(np.float32)), colors=_b64(lm_col)),
        cameras=dict(
            segments=_b64(seg_pts),
            colors=_b64(seg_cols),
            centers=_b64(np.array(centers, np.float32)),
        ),
        stats=stats,
    )
    viewer_template = (Path(__file__).with_name("assets") / "viewer.html").read_text(
        encoding="utf-8"
    )
    html = (
        viewer_template.replace("__PAYLOAD__", json.dumps(payload))
        .replace("__THREE_B64__", three_b64)
        .replace("__ORBIT_B64__", orbit_b64)
        .replace("__TITLE__", f"3D skin scan — {stats['capture']}")
    )
    path = os.path.join(out_dir, "viewer.html")
    with open(path, "w") as fh:
        fh.write(html)
    print(f"      wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


RECOMMENDATIONS = """
-- capture-quality checklist ----------------------------------
1. Lock manual focus at the working distance and disable image stabilization.
   Focus changes alter effective magnification and weaken registration.
2. Calibrate camera intrinsics with an external tool and pass the measured
   full-resolution focal length through --fx-full. Recalibrate the camera and
   RX pivot after changing the camera, lens, focus, sensor mount, or lever arm.
3. Keep capture time and subject motion as low as practical. A consistent
   breathing phase reduces disagreement between passes.
4. Preserve generous overlap between adjacent frames and rows; narrow fields of
   view need cross-row observations to constrain depth.
5. Investigate a high unsettled-frame count before accepting a reconstruction.
6. Keep a small rigid fiducial in frame when an independent metric check is
   required across visits.
7. Choose an aperture that keeps the curved field acceptably sharp, then verify
   exposure and motion blur on an inert target.
"""
