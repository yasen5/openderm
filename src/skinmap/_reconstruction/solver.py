"""Bundle adjustment, surface fitting, and texture reconstruction."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from ..registration_export import build_mesh
from ..registration_geometry import (
    bundle_adjust,
    bundle_adjust_grouped,
    reproj_errors,
    triangulate,
)
from ..registration_surface import TexParam, deformable_align, fit_surface
from ..registration_texture import fit_frame_gains, render_texture


def _solve_registration(args, problem):
    """Run bundle adjustment, reject outliers, fit the surface, and render."""
    out_dir = problem.out_dir
    frames = problem.frames
    pairs = problem.pairs
    mdl = problem.mdl
    R0 = problem.R0
    C0 = problem.C0
    frame_group = None
    if args.group_by_row:
        # group by (row, y-sweep phase), not row alone: a contour scan's -y
        # return pass revisits the row minutes later (breathing moved the subject),
        # and its very-oblique frames overlap the +y pass's skin through the
        # wrap-around -- one rigid group can't hold both, and the LF group gate
        # cannot separate what it cannot distinguish. Scans with no phase change
        # use '+y' throughout, so their grouping is unchanged.
        # A band-order scan (phase '+x'/'-x': x is the fast axis) is the
        # transpose -- there the breathing-coherent unit (frames seconds apart)
        # is the BAND (col); same-row frames sit minutes apart across bands.
        if any(f.phase in ("+x", "-x") for f in frames):
            frame_group = {f.idx: 2 * f.col + (1 if f.phase == "-x" else 0) for f in frames}
        else:
            frame_group = {f.idx: 2 * f.row + (1 if f.phase == "-y" else 0) for f in frames}
    if frame_group is not None and args.group_ba == "group":
        if not args.rig_from:
            print(
                "      ! --group-by-row is best with --rig-from (a single row "
                "can't fit the rig model); proceeding with the global rig"
            )
        print("[6/9] 3D bundle adjustment (per-row groups + inter-group align)")
        ba = bundle_adjust_grouped(frames, pairs, mdl, R0, C0, args, frame_group)
    else:
        # global BA; frame_group (if any) still drives the render-side group
        # ownership/LF gates, which don't need per-group pose solves
        print("[6/9] 3D bundle adjustment (alternating triangulate/resect)")
        ba = bundle_adjust(frames, pairs, mdl, R0, C0, args)
    R, C, X = ba["R"], ba["C"], ba["X"]

    # pose deviation from the (re-anchored) proprioception prior
    R0, C0 = mdl.poses(frames)
    dt = np.linalg.norm(C - C0, axis=1)
    dr = np.array(
        [
            np.degrees(np.linalg.norm(Rotation.from_matrix(R0[i].T @ R[i]).as_rotvec()))
            for i in range(len(frames))
        ]
    )
    print(
        f"      pose deviation from prior: |dt| median {np.median(dt):.2f}mm "
        f"max {dt.max():.2f}mm; |dr| median {np.median(dr):.2f}deg max {dr.max():.2f}deg"
    )

    # --- reject gross-outlier frames -----------------------------------------
    # The gantry encoders locate the camera to ~mm, so a frame whose recovered
    # pose sits far from proprioception is a false feature-lock on repetitive/
    # low-texture skin, not real motion (breathing and stabilization motion are smaller).
    # Trust the gantry for those frames: snap the pose back to the prior (keeps
    # their texture coverage roughly right) and drop their observations, then
    # re-triangulate the surviving landmarks from the trustworthy rays only --
    # otherwise a couple of bad frames bend the subject surface into a spike
    # and smear the ortho-texture.
    if args.reject_pose_mm > 0 or args.reject_rot_deg > 0:
        bad = np.zeros(len(frames), bool)
        if args.reject_pose_mm > 0:
            bad |= dt > args.reject_pose_mm
        if args.reject_rot_deg > 0:
            bad |= dr > args.reject_rot_deg
        if bad.any():
            bad_idx = np.where(bad)[0]
            for i in bad_idx:
                R[i], C[i] = R0[i].copy(), C0[i].copy()  # trust the gantry
            bad_set = set(int(i) for i in bad_idx)
            keep = np.array([int(fi) not in bad_set for fi in ba["obs_frame"]], dtype=bool)
            for k in ("obs_frame", "obs_uv", "obs_track"):
                ba[k] = ba[k][keep]
            # re-triangulate landmarks that still have >=2 trustworthy views;
            # keep the old position for the rest (they are dropped later by the
            # >=3-obs surface-fit gate, but must stay finite for the bounds calc).
            Xr = triangulate(ba["obs_frame"], ba["obs_uv"], ba["obs_track"], len(X), R, C, mdl)
            nobs_g = np.bincount(ba["obs_track"], minlength=len(X))
            ok = (nobs_g >= 2) & np.isfinite(Xr).all(1)
            X = np.where(ok[:, None], Xr, X)
            ba["X"] = X
            print(
                f"      ! rejected {len(bad_idx)} gross-outlier frame(s) "
                f"(>{args.reject_pose_mm:.0f}mm or >{args.reject_rot_deg:.0f}deg "
                f"off gantry): {sorted(bad_set)} -> snapped to proprioception, "
                f"{int((~keep).sum())} obs dropped, {int(ok.sum())} landmarks "
                f"re-triangulated"
            )

    # standoff sensor agreement
    err, zc = reproj_errors(X, ba["obs_frame"], ba["obs_uv"], ba["obs_track"], R, C, mdl)
    zc_per = np.full(len(frames), np.nan)
    for f in frames:
        m = ba["obs_frame"] == f.idx
        if m.sum() > 10:
            zc_per[f.idx] = np.median(zc[m])
    so = np.array([f.standoff for f in frames])
    ok = ~np.isnan(zc_per)
    dzfit = float(np.median(zc_per[ok] - so[ok]))
    so_rms = float(np.sqrt(np.mean((zc_per[ok] - so[ok] - dzfit) ** 2)))
    print(f"      standoff sensors vs recovered depth: offset {dzfit:+.1f}mm, rms {so_rms:.2f}mm")

    # settled vs unsettled
    med_err = np.full(len(frames), np.nan)
    for f in frames:
        m = ba["obs_frame"] == f.idx
        if m.sum():
            med_err[f.idx] = np.median(err[m])
    settled = np.array([f.settled for f in frames])
    msg_settle = ""
    if settled.any() and (~settled).any():
        a = np.nanmedian(med_err[settled])
        b = np.nanmedian(med_err[~settled])
        msg_settle = f"median reproj: settled {a:.2f}px vs unsettled {b:.2f}px"
        print(f"      {msg_settle}")

    print("[7/9] fitting surface heightfield")
    # orient: does +z normal face the cameras?
    up_sign = 1.0 if (C[:, 2].mean() > np.median(X[:, 2])) else -1.0
    # Auto contour metric: a strongly curved subject scanned with the rig tilting
    # to follow the surface has a wide RX spread. Surface RMS provides an
    # independent confirmation. The gate only changes surface smoothing, so the
    # bounds and landmark selection remain identical.
    rx_all = np.array([f.rx for f in frames])
    rx_spread = float(np.degrees(rx_all.max() - rx_all.min()))
    # robust bounds + well-supported landmarks only: a handful of blown-up
    # tracks must not inflate the grid (oscillating extrapolation wrecks the
    # arc-length parameterisation downstream)
    qx = np.percentile(X[:, 0], [0.5, 99.5])
    qy = np.percentile(X[:, 1], [0.5, 99.5])
    bx0, bx1 = qx[0] - 8, qx[1] + 8
    by0, by1 = qy[0] - 8, qy[1] + 8
    nobs_track = np.bincount(ba["obs_track"], minlength=len(X))
    w_track = np.clip(nobs_track - 1, 1, 8).astype(float)
    sel = (X[:, 0] >= bx0) & (X[:, 0] <= bx1) & (X[:, 1] >= by0) & (X[:, 1] <= by1)
    strong = nobs_track >= 3
    if (sel & strong).mean() > 0.3:
        sel &= strong
    print(
        f"      using {sel.sum()}/{len(X)} landmarks "
        f"(robust bounds x[{bx0:.0f},{bx1:.0f}] y[{by0:.0f},{by1:.0f}], "
        f">=3-obs tracks)"
    )
    surf, surf_rms = fit_surface(
        X[sel], (bx0, bx1, by0, by1), args.surface_pitch, args.surface_smooth, w0=w_track[sel]
    )
    # Finalize the contour gate from both RX spread and landmark-surface RMS so
    # isolated noisy landmarks cannot enable it on their own.
    contour = (args.contour == "on") or (
        args.contour == "auto"
        and rx_spread > args.contour_rx_thresh_deg
        and surf_rms > args.contour_rms_thresh
    )
    if contour:
        # Refit with stronger smoothing so depth noise and breathing do not turn
        # the broad measured contour into sharp peaks and valleys.
        surf, _ = fit_surface(
            X[sel], (bx0, bx1, by0, by1), args.surface_pitch, args.contour_smooth, w0=w_track[sel]
        )
        print(
            f"      [contour] enabled (RX spread {rx_spread:.0f}deg, surface RMS "
            f"{surf_rms:.0f}mm): landmark surface smoothing="
            f"{args.contour_smooth:.0f}"
        )
    tp = TexParam(surf, mdl.base_R, mdl.base_t)

    warp = None
    if args.deformable:
        print("      deformable alignment (smooth per-frame map warp)")
        warp = deformable_align(
            frames, pairs, R, C, mdl, surf, tp, reg=args.deformable_reg, order=args.deformable_order
        )

    frame_gain = None
    if args.lf_gain != "off":
        frame_gain = fit_frame_gains(
            frames,
            mdl,
            ba["obs_frame"],
            ba["obs_uv"],
            ba["obs_track"],
            ba.get("err"),
            mode=args.lf_gain,
        )

    print("[8/9] rendering ortho-texture")
    tex, wacc, tex_bounds = render_texture(
        frames,
        R,
        C,
        mdl,
        surf,
        tp,
        args.texture_ppmm,
        up_sign,
        out_dir,
        args.blend_sharpness,
        args.focus_weight,
        args.blend,
        frame_group,
        warp,
        args.hf_coherence_mm,
        args.hf_cross_group,
        args.group_feather_mm,
        args.max_incidence_deg,
        device=args.device,
        frame_gain=frame_gain,
    )
    pos, nrm, uvn, faces = build_mesh(
        surf,
        tp,
        tex_bounds,
        wacc,
        args.texture_ppmm,
        args.mesh_pitch,
        up_sign,
        tuple(args.mesh_smooth),
    )
    return SimpleNamespace(
        ba=ba,
        R=R,
        C=C,
        X=X,
        dt=dt,
        dr=dr,
        med_err=med_err,
        surf=surf,
        surf_rms=surf_rms,
        tp=tp,
        tex=tex,
        wacc=wacc,
        tex_bounds=tex_bounds,
        pos=pos,
        nrm=nrm,
        uvn=uvn,
        faces=faces,
        msg_settle=msg_settle,
        dzfit=dzfit,
        so_rms=so_rms,
    )
