"""Write reconstruction artifacts and quality reports."""

from __future__ import annotations

import json
import math
import os
from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from ..registration_features import Frame, Pair
from ..registration_geometry import BundleAdjustmentResult, CenterArray, FloatArray, RigModel, RotationArray
from ..registration_surface import Surface, TexParam
from .parser import ScanCliArguments
from ..registration_export import (
    RECOMMENDATIONS,
    export_surface_landmarks_ply,
    export_surface_mesh_obj,
    export_surface_viewer_html,
    render_reconstruction_overview_png,
)


class ReconstructionProblemForExport(Protocol):
    out_dir: str
    frames: list[Frame]
    pairs: list[Pair]
    rig_model: RigModel
    prefit_rms: float


class ReconstructionSolutionForExport(Protocol):
    bundle_adjustment_result: BundleAdjustmentResult
    R: RotationArray
    C: CenterArray
    X: FloatArray
    dt: FloatArray
    dr: FloatArray
    med_err: FloatArray
    surf: Surface
    surf_rms: float
    texture_parameters: TexParam
    tex: NDArray[np.uint8]
    wacc: NDArray[np.float32] | None
    tex_bounds: tuple[float, float, float, float]
    pos: NDArray[np.float32]
    nrm: NDArray[np.float32]
    uvn: NDArray[np.float32]
    faces: NDArray[np.int64]
    msg_settle: str
    dzfit: float
    so_rms: float


class ViewerStats(TypedDict):
    capture: str
    n_frames: int
    n_landmarks: int
    rms_px: str
    rms_mm: str
    px_per_mm: str
    area_cm2: str


class GantryPlacement(TypedDict):
    x: float
    y: float
    z: float
    rx_deg: float
    standoff_mm: float
    settled: bool


class FramePlacement(TypedDict):
    idx: int
    station: int
    row: int
    col: int
    image: str
    C_mm: list[float]
    R_cam2world: list[list[float]]
    gantry: GantryPlacement
    prior_dev_mm: float
    prior_dev_deg: float
    median_reproj_px: float | None


class RigPlacement(TypedDict):
    fx_ds_px: float
    fx_fullres_px: float
    k1: float
    cx: float
    cy: float
    rx_sign: float
    lever_mm: list[float]
    Rm: list[list[float]]
    dz0_mm: float
    base_R: list[list[float]]
    base_t: list[float]


class BundleAdjustmentPlacement(TypedDict):
    rms_px: float
    rms_mm: float
    history: list[float]
    n_landmarks: int
    n_obs: int


class SurfacePlacement(TypedDict):
    pitch_mm: float
    smooth: float
    landmark_rms_mm: float


class TexturePlacement(TypedDict):
    umin_mm: float
    vmin_mm: float
    umax_mm: float
    vmax_mm: float
    ppmm: float
    W: int
    H: int
    arclen_gy_mm: list[float]
    arclen_s_mm: list[float]


class PlacementsDocument(TypedDict):
    capture_dir: str
    downscale: int
    rig_model: RigPlacement
    ba: BundleAdjustmentPlacement
    surface: SurfacePlacement
    texture: TexturePlacement
    frames: list[FramePlacement]


def export_scan_reconstruction_artifacts(
    args: ScanCliArguments,
    problem: ReconstructionProblemForExport,
    solution: ReconstructionSolutionForExport,
) -> None:
    """Write meshes, diagnostics, placements, and the human-readable report."""
    out_dir = problem.out_dir
    frames = problem.frames
    pairs = problem.pairs
    rig_model = problem.rig_model
    prefit_rms = problem.prefit_rms
    bundle_adjustment_result = solution.bundle_adjustment_result
    camera_rotations = solution.R
    camera_centers = solution.C
    landmark_points = solution.X
    dt = solution.dt
    dr = solution.dr
    med_err = solution.med_err
    surf = solution.surf
    surf_rms = solution.surf_rms
    texture_parameters = solution.texture_parameters
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
    export_surface_landmarks_ply(out_dir, landmark_points, bundle_adjustment_result["track_err"])
    render_reconstruction_overview_png(out_dir, frames, camera_rotations, camera_centers, rig_model, surf, landmark_points)

    rms_mm = bundle_adjustment_result["rms"] / (rig_model.fx / np.mean([rig_model.depth(frame) for frame in frames]))
    area_cm2 = (
        0.0
        if wacc is None
        else float((wacc > 0).sum()) / (args.texture_ppmm**2) / 100.0
    )
    stats: ViewerStats = ViewerStats(
        capture=os.path.basename(os.path.normpath(args.capture_dir)),
        n_frames=len(frames),
        n_landmarks=int(len(landmark_points)),
        rms_px=f"{bundle_adjustment_result['rms']:.2f}",
        rms_mm=f"{rms_mm:.3f}",
        px_per_mm=f"{rig_model.fx * args.downscale / np.mean([rig_model.depth(frame) for frame in frames]):.0f}",
        area_cm2=f"{area_cm2:.0f}",
    )
    export_surface_viewer_html(out_dir, frames, camera_rotations, camera_centers, rig_model, pos, uvn, faces, landmark_points, bundle_adjustment_result["track_err"], stats)

    placements: PlacementsDocument = PlacementsDocument(
        capture_dir=args.capture_dir,
        downscale=args.downscale,
        rig_model=RigPlacement(
            fx_ds_px=rig_model.fx,
            fx_fullres_px=rig_model.fx * args.downscale,
            k1=rig_model.k1,
            cx=rig_model.cx,
            cy=rig_model.cy,
            rx_sign=rig_model.sign,
            lever_mm=rig_model.lever.tolist(),
            Rm=rig_model.Rm.tolist(),
            dz0_mm=rig_model.dz0,
            base_R=rig_model.base_R.tolist(),
            base_t=rig_model.base_t.tolist(),
        ),
        ba=BundleAdjustmentPlacement(
            rms_px=bundle_adjustment_result["rms"],
            rms_mm=float(rms_mm),
            history=bundle_adjustment_result["history"],
            n_landmarks=int(len(landmark_points)),
            n_obs=int(len(bundle_adjustment_result["obs_uv"])),
        ),
        surface=SurfacePlacement(
            pitch_mm=args.surface_pitch, smooth=args.surface_smooth, landmark_rms_mm=surf_rms
        ),
        # Texture canvas gauge -- serialized so cross-scan tools can place two
        # maps on one common (u,v)-mm frame without re-deriving bounds from the
        # mesh (u recoverable, but v=arc-length depends on the unserialized
        # surface grid). The texture image is LINEAR in (u_mm, v_arclen_mm) at
        # ppmm, so pixel (i,j) <-> (umin+i/ppmm, vmin+j/ppmm). arclen_* maps
        # gantry-y -> arc-length-v for mapping texels back to 3D/gantry.
        texture=TexturePlacement(
            umin_mm=float(tex_bounds[0]),
            vmin_mm=float(tex_bounds[1]),
            umax_mm=float(tex_bounds[2]),
            vmax_mm=float(tex_bounds[3]),
            ppmm=float(args.texture_ppmm),
            W=int(tex.shape[1]),
            H=int(tex.shape[0]),
            arclen_gy_mm=texture_parameters.gy.tolist(),
            arclen_s_mm=texture_parameters.s.tolist(),
        ),
        frames=[
            FramePlacement(
                idx=frame.idx,
                station=frame.station,
                row=frame.row,
                col=frame.col,
                image=os.path.basename(frame.image_path),
                C_mm=camera_centers[frame.idx].tolist(),
                R_cam2world=camera_rotations[frame.idx].tolist(),
                gantry=GantryPlacement(
                    x=float(frame.gauge[0]),
                    y=float(frame.gauge[1]),
                    z=float(frame.gauge[2]),
                    rx_deg=math.degrees(frame.rx),
                    standoff_mm=frame.standoff,
                    settled=frame.settled,
                ),
                prior_dev_mm=float(dt[frame.idx]),
                prior_dev_deg=float(dr[frame.idx]),
                median_reproj_px=(None if np.isnan(med_err[frame.idx]) else float(med_err[frame.idx])),
            )
            for frame in frames
        ],
    )
    with open(os.path.join(out_dir, "placements3d.json"), "w") as fh:
        json.dump(placements, fh, indent=1)

    pxmm_full = rig_model.fx * args.downscale / np.mean([rig_model.depth(frame) for frame in frames])
    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write(f"3D registration report for {args.capture_dir}\n")
        fh.write(
            f"frames: {len(frames)}  pairs: {len(pairs)}  "
            f"landmarks: {len(landmark_points)}  obs: {len(bundle_adjustment_result['obs_uv'])}\n\n"
        )
        fh.write("-- accuracy --------------------------------------------------\n")
        fh.write(f"rig pre-fit rms (consecutive pairs):  {prefit_rms:.2f} px (ds)\n")
        fh.write(
            f"BA reprojection rms:                  {bundle_adjustment_result['rms']:.2f} px (ds) "
            f"= {rms_mm * 1000:.0f} um on skin\n"
        )
        fh.write(f"BA rms history: {' -> '.join(f'{image_height:.2f}' for image_height in bundle_adjustment_result['history'])}\n")
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
            f"fx = {rig_model.fx * args.downscale:.0f} px (full res) -> {pxmm_full:.1f} px/mm on skin\n"
        )
        fh.write(f"k1 = {rig_model.k1:+.4f}\n")
        fh.write(
            f"rx sign = {rig_model.sign:+.0f}, lever arm = {rig_model.lever.round(1).tolist()} mm "
            f"(|{np.linalg.norm(rig_model.lever):.1f}| mm)\n"
        )
        fh.write(f"sensor->optical-centre depth offset dz0 = {rig_model.dz0:+.1f} mm\n")
        fh.write(RECOMMENDATIONS)
        fh.write("\nper-frame (station, row/col, settled, prior dev mm/deg, median reproj px):\n")
        for frame in frames:
            me = med_err[frame.idx]
            fh.write(
                f"  st{frame.station:>3} r{frame.row:>2}c{frame.col:>2} "
                f"{'S' if frame.settled else '.'} "
                f"{dt[frame.idx]:6.2f}mm {dr[frame.idx]:5.2f}deg "
                f"{me if not np.isnan(me) else float('nan'):6.2f}px\n"
            )
    print(f"      wrote {os.path.join(out_dir, 'report.txt')}")
    print("\ndone.")
