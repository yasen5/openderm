"""Mesh, diagnostic, and interactive-viewer exports."""

from __future__ import annotations

import base64
import json
import os
import sysconfig
from pathlib import Path

import cv2
import numpy as np

from .registration_geometry import undistort_image_points_to_normalized_camera


# ----------------------------------------------------------------------------
# mesh build + exports
# ----------------------------------------------------------------------------
def build_surface_mesh(
    surf,
    texture_parameters,
    tex_bounds,
    wacc,
    pixels_per_mm=None,
    mesh_pitch=None,
    up_sign=1.0,
    mesh_smooth=(0.0, 0.0),
    **legacy_options,
):
    pixels_per_mm = legacy_options.pop("ppmm", pixels_per_mm)
    if legacy_options:
        unexpected_option = next(iter(legacy_options))
        raise TypeError(f"build_surface_mesh got an unexpected keyword argument {unexpected_option!r}")
    if pixels_per_mm is None or mesh_pitch is None:
        raise TypeError("build_surface_mesh requires pixels_per_mm and mesh_pitch")
    umin, vmin, umax, vmax = tex_bounds
    us = np.arange(umin, umax + mesh_pitch, mesh_pitch)
    vs = np.arange(vmin, vmax + mesh_pitch, mesh_pitch)
    nx, ny = len(us), len(vs)
    Ug, Vg = np.meshgrid(us, vs)
    Xg, Yg = texture_parameters.to_xy(Ug.ravel(), Vg.ravel())
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
    image_height, image_width = wacc.shape
    tu = np.clip(((Ug.ravel() - umin) * pixels_per_mm).astype(int), 0, image_width - 1)
    tv = np.clip(((Vg.ravel() - vmin) * pixels_per_mm).astype(int), 0, image_height - 1)
    valid = wacc[tv, tu] > 0

    faces = []
    for neighbor_index in range(ny - 1):
        first_vertex_indices = neighbor_index * nx + np.arange(nx - 1)
        quads = np.stack(
            [first_vertex_indices, first_vertex_indices + 1, first_vertex_indices + nx, first_vertex_indices + nx + 1], 1
        )
        ok = valid[quads].all(1)
        valid_cell_vertex_indices = quads[ok]
        faces.append(np.stack([valid_cell_vertex_indices[:, 0], valid_cell_vertex_indices[:, 1], valid_cell_vertex_indices[:, 2]], 1))
        faces.append(np.stack([valid_cell_vertex_indices[:, 1], valid_cell_vertex_indices[:, 3], valid_cell_vertex_indices[:, 2]], 1))
    faces = np.concatenate(faces).astype(np.int64)
    print(f"      mesh: {len(pos)} verts ({nx}x{ny}), {len(faces)} tris")
    return pos, nrm, uvn, faces


def export_surface_mesh_obj(out_dir, pos, nrm, uvn, faces):
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


def export_surface_landmarks_ply(out_dir, landmark_points, track_err):
    reprojection_error = np.clip(track_err / max(track_err.max(), 1e-6), 0, 1)
    col = np.stack([(reprojection_error * 255), (1 - reprojection_error) * 255, np.zeros_like(reprojection_error)], 1).astype(np.uint8)
    path = os.path.join(out_dir, "landmarks.ply")
    with open(path, "wb") as fh:
        fh.write(
            (
                f"ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(landmark_points)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n"
            ).encode()
        )
        rec = np.zeros(len(landmark_points), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        rec["xyz"] = landmark_points.astype(np.float32)
        rec["rgb"] = col
        fh.write(rec.tobytes())
    print(f"      wrote {path}")


def render_reconstruction_overview_png(out_dir, frames, camera_rotations, camera_centers, rig_model, surf, landmark_points):
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
    sub = np.random.default_rng(0).permutation(len(landmark_points))[:4000]
    ax.scatter(landmark_points[sub, 0], landmark_points[sub, 1], landmark_points[sub, 2], s=0.5, c="firebrick", alpha=0.4)
    rows = np.array([frame.row for frame in frames])
    cmap = plt.colormaps["viridis"]
    for frame in frames:
        cclr = cmap((frame.row - rows.min()) / max(1, rows.max() - rows.min()))
        ax.scatter(*camera_centers[frame.idx], color=cclr, s=12)
        tip = camera_centers[frame.idx] + camera_rotations[frame.idx][:, 2] * rig_model.depth(frame) * 0.6
        ax.plot(*np.stack([camera_centers[frame.idx], tip], 1), color=cclr, lw=0.5, alpha=0.6)
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


def export_surface_viewer_html(out_dir, frames, camera_rotations, camera_centers, rig_model, pos, uvn, faces, landmark_points, track_err, stats):
    repo_libraries = Path(__file__).resolve().parents[2] / "third_party" / "threejs"
    installed_libraries = (
        Path(sysconfig.get_path("data")) / "share" / "openderm" / "third_party" / "threejs"
    )
    libraries = repo_libraries if repo_libraries.is_dir() else installed_libraries
    three_b64 = orbit_b64 = ""
    for name, var in (("three.module.min.js", "three"), ("OrbitControls.js", "orbit")):
        point = libraries / name
        if point.is_file():
            data = base64.b64encode(point.read_bytes()).decode()
            if var == "three":
                three_b64 = data
            else:
                orbit_b64 = data
    if not three_b64 or not orbit_b64:
        three_b64 = orbit_b64 = ""
        print(
            f"      ! Three.js viewer libraries missing from {libraries} -- "
            "viewer will fall back to CDN (needs internet)"
        )

    # cap the embedded copy at the WebGL max texture size of every desktop
    # GPU/browser of the last decade (16384; older mobile GPUs stop at 8192 --
    # this viewer is a desktop artifact). The full-res texture.jpg on disk is
    # untouched (that's the analysis artifact).
    tex_disk = cv2.imread(os.path.join(out_dir, "texture.jpg"), cv2.IMREAD_COLOR)
    max_dim = 16384
    if max(tex_disk.shape[:2]) > max_dim:
        score = max_dim / max(tex_disk.shape[:2])
        tex_disk = cv2.resize(tex_disk, None, fx=score, fy=score, interpolation=cv2.INTER_AREA)
        print(
            f"      (viewer texture downscaled to {tex_disk.shape[1]}x"
            f"{tex_disk.shape[0]} for WebGL; texture.jpg keeps full res)"
        )
    ok, buf = cv2.imencode(".jpg", tex_disk, [cv2.IMWRITE_JPEG_QUALITY, 88])
    tex_uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    # camera frusta line segments (8 per frame) + colors by row
    rows = np.array([frame.row for frame in frames])
    rspan = max(1, rows.max() - rows.min())
    seg_pts, seg_cols, centers = [], [], []
    image_width, image_height = rig_model.cx * 2, rig_model.cy * 2
    image_corners = np.array(
        [[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]], float
    )
    for frame in frames:
        frustum_depth = rig_model.depth(frame) * 0.35
        normalized_corner_points = undistort_image_points_to_normalized_camera(
            image_corners, rig_model.fx, rig_model.k1, rig_model.cx, rig_model.cy
        )
        camera_space_corners = np.concatenate(
            [normalized_corner_points * frustum_depth, np.full((4, 1), frustum_depth)], 1
        )
        world_frustum_corners = camera_space_corners @ camera_rotations[frame.idx].T + camera_centers[frame.idx]
        row_color_fraction = (frame.row - rows.min()) / rspan
        row_color = np.array(
            [0.2 + 0.8 * row_color_fraction, 0.9 - 0.6 * row_color_fraction, 1.0 - 0.7 * row_color_fraction]
        )
        for item_index in range(4):
            seg_pts += [camera_centers[frame.idx], world_frustum_corners[item_index], world_frustum_corners[item_index], world_frustum_corners[(item_index + 1) % 4]]
            seg_cols += [row_color, row_color, row_color, row_color]
        centers.append(camera_centers[frame.idx])
    seg_pts = np.array(seg_pts, np.float32)
    seg_cols = np.array(seg_cols, np.float32)

    sub = np.random.default_rng(0).permutation(len(landmark_points))[:40000]
    reprojection_error = np.clip(track_err[sub] / max(np.percentile(track_err, 95), 1e-6), 0, 1)
    lm_col = np.stack([reprojection_error, 1 - reprojection_error, np.full_like(reprojection_error, 0.15)], 1).astype(np.float32)

    payload = dict(
        mesh=dict(positions=_b64(pos), uvs=_b64(uvn), indices=_b64(faces.astype(np.uint32))),
        texture=tex_uri,
        landmarks=dict(positions=_b64(landmark_points[sub].astype(np.float32)), colors=_b64(lm_col)),
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
