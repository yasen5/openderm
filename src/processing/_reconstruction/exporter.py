"""Write reconstruction artifacts and quality reports."""

from __future__ import annotations

import json
import math
import os

import numpy as np

from ..registration_export import (
    RECOMMENDATIONS,
    export_surface_landmarks_ply,
    export_surface_mesh_obj,
    export_surface_viewer_html,
    render_reconstruction_overview_png,
)


def export_scan_reconstruction_artifacts(args, problem, solution):
    """Write meshes, diagnostics, placements, and the human-readable report."""
    out_dir = problem.out_dir
    frames = problem.frames
    pairs = problem.pairs
    mdl = problem.mdl
    prefit_rms = problem.prefit_rms
    ba = solution.ba
    R = solution.R
    C = solution.C
    X = solution.X
    dt = solution.dt
    dr = solution.dr
    med_err = solution.med_err
    surf = solution.surf
    surf_rms = solution.surf_rms
    tp = solution.tp
    tex = solution.tex
    wacc = solution.wacc
    tex_bounds = solution.tex_bounds
    pos = solution.pos
    nrm = solution.nrm
    uvn = solution.uvn
    faces = solution.faces
    msg_settle = solution.msg_settle
    dzfit = solution.dzfit
    so_rms = solution.so_rms
    print("[9/9] writing outputs")
    export_surface_mesh_obj(out_dir, pos, nrm, uvn, faces)
    export_surface_landmarks_ply(out_dir, X, ba["track_err"])
    render_reconstruction_overview_png(out_dir, frames, R, C, mdl, surf, X)

    rms_mm = ba["rms"] / (mdl.fx / np.mean([mdl.depth(f) for f in frames]))
    area_cm2 = (wacc > 0).sum() / (args.texture_ppmm**2) / 100.0
    stats = dict(
        capture=os.path.basename(os.path.normpath(args.capture_dir)),
        n_frames=len(frames),
        n_landmarks=int(len(X)),
        rms_px=f"{ba['rms']:.2f}",
        rms_mm=f"{rms_mm:.3f}",
        px_per_mm=f"{mdl.fx * args.downscale / np.mean([mdl.depth(f) for f in frames]):.0f}",
        area_cm2=f"{area_cm2:.0f}",
    )
    export_surface_viewer_html(out_dir, frames, R, C, mdl, pos, uvn, faces, X, ba["track_err"], stats)

    placements = dict(
        capture_dir=args.capture_dir,
        downscale=args.downscale,
        rig_model=dict(
            fx_ds_px=mdl.fx,
            fx_fullres_px=mdl.fx * args.downscale,
            k1=mdl.k1,
            cx=mdl.cx,
            cy=mdl.cy,
            rx_sign=mdl.sign,
            lever_mm=mdl.lever.tolist(),
            Rm=mdl.Rm.tolist(),
            dz0_mm=mdl.dz0,
            base_R=mdl.base_R.tolist(),
            base_t=mdl.base_t.tolist(),
        ),
        ba=dict(
            rms_px=ba["rms"],
            rms_mm=rms_mm,
            history=ba["history"],
            n_landmarks=int(len(X)),
            n_obs=int(len(ba["obs_uv"])),
        ),
        surface=dict(
            pitch_mm=args.surface_pitch, smooth=args.surface_smooth, landmark_rms_mm=surf_rms
        ),
        # Texture canvas gauge -- serialized so cross-scan tools can place two
        # maps on one common (u,v)-mm frame without re-deriving bounds from the
        # mesh (u recoverable, but v=arc-length depends on the unserialized
        # surface grid). The texture image is LINEAR in (u_mm, v_arclen_mm) at
        # ppmm, so pixel (i,j) <-> (umin+i/ppmm, vmin+j/ppmm). arclen_* maps
        # gantry-y -> arc-length-v for mapping texels back to 3D/gantry.
        texture=dict(
            umin_mm=float(tex_bounds[0]),
            vmin_mm=float(tex_bounds[1]),
            umax_mm=float(tex_bounds[2]),
            vmax_mm=float(tex_bounds[3]),
            ppmm=float(args.texture_ppmm),
            W=int(tex.shape[1]),
            H=int(tex.shape[0]),
            arclen_gy_mm=tp.gy.tolist(),
            arclen_s_mm=tp.s.tolist(),
        ),
        frames=[
            dict(
                idx=f.idx,
                station=f.station,
                row=f.row,
                col=f.col,
                image=os.path.basename(f.image_path),
                C_mm=C[f.idx].tolist(),
                R_cam2world=R[f.idx].tolist(),
                gantry=dict(
                    x=f.g[0],
                    y=f.g[1],
                    z=f.g[2],
                    rx_deg=math.degrees(f.rx),
                    standoff_mm=f.standoff,
                    settled=f.settled,
                ),
                prior_dev_mm=float(dt[f.idx]),
                prior_dev_deg=float(dr[f.idx]),
                median_reproj_px=(None if np.isnan(med_err[f.idx]) else float(med_err[f.idx])),
            )
            for f in frames
        ],
    )
    with open(os.path.join(out_dir, "placements3d.json"), "w") as fh:
        json.dump(placements, fh, indent=1)

    pxmm_full = mdl.fx * args.downscale / np.mean([mdl.depth(f) for f in frames])
    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write(f"3D registration report for {args.capture_dir}\n")
        fh.write(
            f"frames: {len(frames)}  pairs: {len(pairs)}  "
            f"landmarks: {len(X)}  obs: {len(ba['obs_uv'])}\n\n"
        )
        fh.write("-- accuracy --------------------------------------------------\n")
        fh.write(f"rig pre-fit rms (consecutive pairs):  {prefit_rms:.2f} px (ds)\n")
        fh.write(
            f"BA reprojection rms:                  {ba['rms']:.2f} px (ds) "
            f"= {rms_mm * 1000:.0f} um on skin\n"
        )
        fh.write(f"BA rms history: {' -> '.join(f'{h:.2f}' for h in ba['history'])}\n")
        fh.write(f"surface fit rms (landmark->surface):  {surf_rms:.3f} mm\n")
        fh.write(
            f"pose deviation from proprioception:   |dt| median "
            f"{np.median(dt):.2f} / max {dt.max():.2f} mm, |dr| median "
            f"{np.median(dr):.2f} / max {dr.max():.2f} deg\n"
        )
        fh.write(
            f"standoff sensors vs recovered depth:  offset {dzfit:+.1f} mm, rms {so_rms:.2f} mm\n"
        )
        if msg_settle:
            fh.write(f"{msg_settle}\n")
        fh.write("\n-- fitted rig model ------------------------------------------\n")
        fh.write(
            f"fx = {mdl.fx * args.downscale:.0f} px (full res) -> {pxmm_full:.1f} px/mm on skin\n"
        )
        fh.write(f"k1 = {mdl.k1:+.4f}\n")
        fh.write(
            f"rx sign = {mdl.sign:+.0f}, lever arm = {mdl.lever.round(1).tolist()} mm "
            f"(|{np.linalg.norm(mdl.lever):.1f}| mm)\n"
        )
        fh.write(f"sensor->optical-centre depth offset dz0 = {mdl.dz0:+.1f} mm\n")
        fh.write(RECOMMENDATIONS)
        fh.write("\nper-frame (station, row/col, settled, prior dev mm/deg, median reproj px):\n")
        for f in frames:
            me = med_err[f.idx]
            fh.write(
                f"  st{f.station:>3} r{f.row:>2}c{f.col:>2} "
                f"{'S' if f.settled else '.'} "
                f"{dt[f.idx]:6.2f}mm {dr[f.idx]:5.2f}deg "
                f"{me if not np.isnan(me) else float('nan'):6.2f}px\n"
            )
    print(f"      wrote {os.path.join(out_dir, 'report.txt')}")
    print("\ndone.")
