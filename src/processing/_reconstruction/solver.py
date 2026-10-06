"""Bundle adjustment, surface fitting, and texture reconstruction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Protocol, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..registration_features import Frame, Pair
from ..registration_geometry import (
    BundleAdjustmentResult,
    CenterArray,
    FloatArray,
    IntArray,
    RegistrationArgs,
    RigModel,
    RotationArray,
    bundle_adjust_camera_poses_and_feature_tracks,
    bundle_adjust_grouped_camera_poses_and_feature_tracks,
    compute_feature_track_reprojection_errors,
    triangulate_3d_feature_tracks,
)
from ..registration_export import build_surface_mesh
from ..registration_surface import (
    Surface,
    TexParam,
    align_frames_with_deformable_surface_warps,
    fit_surface_heightfield,
)
from ..registration_texture import Bounds, Warp, estimate_camera_frame_texture_gains, render_surface_texture


class ReconstructionArgs(RegistrationArgs, Protocol):
    """Arguments consumed by the surface reconstruction stage."""

    @property
    def group_ba(self) -> str: ...

    @property
    def contour(self) -> str: ...

    @property
    def deformable_order(self) -> int: ...

    @property
    def lf_gain(self) -> str: ...

    @property
    def blend(self) -> str: ...

    @property
    def device(self) -> str: ...

    @property
    def mesh_smooth(self) -> Sequence[float]: ...

    group_by_row: bool
    reject_pose_mm: float
    reject_rot_deg: float
    surface_pitch: float
    surface_smooth: float
    contour_rx_thresh_deg: float
    contour_rms_thresh: float
    contour_smooth: float
    deformable: bool
    deformable_reg: float
    texture_ppmm: float
    blend_sharpness: float
    focus_weight: float
    hf_coherence_mm: float
    hf_cross_group: bool
    group_feather_mm: float
    max_incidence_deg: float
    mesh_pitch: float


class ReconstructionProblem(Protocol):
    out_dir: str
    frames: list[Frame]
    pairs: list[Pair]
    rig_model: RigModel
    R0: RotationArray
    C0: CenterArray


class ReconstructionSolution(Protocol):
    """Attributes shared with the reconstruction artifact exporter."""

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
    tex_bounds: Bounds
    pos: NDArray[np.float32]
    nrm: NDArray[np.float32]
    uvn: NDArray[np.float32]
    faces: NDArray[np.int64]
    msg_settle: str
    dzfit: float
    so_rms: float


def reconstruct_surface_from_camera_frames(
    args: ReconstructionArgs, problem: ReconstructionProblem
) -> ReconstructionSolution:
    """Run bundle adjustment, reject outliers, fit the surface, and render."""
    output_directory = problem.out_dir
    frames = problem.frames
    pairs = problem.pairs
    rig_model = problem.rig_model
    prior_camera_rotations = problem.R0
    prior_camera_centers = problem.C0
    frame_group: dict[int, int] | None = None
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
        if any(frame.phase in ("+x", "-x") for frame in frames):
            frame_group = {frame.idx: 2 * frame.col + (1 if frame.phase == "-x" else 0) for frame in frames}
        else:
            frame_group = {frame.idx: 2 * frame.row + (1 if frame.phase == "-y" else 0) for frame in frames}
    if frame_group is not None and args.group_ba == "group":
        if not args.rig_from:
            print(
                "      ! --group-by-row is best with --rig-from (a single row "
                "can't fit the rig model); proceeding with the global rig"
            )
        print("[6/9] 3D bundle adjustment (per-row groups + inter-group align)")
        bundle_adjustment_result = bundle_adjust_grouped_camera_poses_and_feature_tracks(frames, pairs, rig_model, prior_camera_rotations, prior_camera_centers, args, frame_group)
    else:
        # global BA; frame_group (if any) still drives the render-side group
        # ownership/LF gates, which don't need per-group pose solves
        print("[6/9] 3D bundle adjustment (alternating triangulate_3d_feature_tracks/resect)")
        bundle_adjustment_result = bundle_adjust_camera_poses_and_feature_tracks(frames, pairs, rig_model, prior_camera_rotations, prior_camera_centers, args)
    camera_rotations: RotationArray = bundle_adjustment_result["R"]
    camera_centers: CenterArray = bundle_adjustment_result["C"]
    landmark_points: FloatArray = bundle_adjustment_result["X"]

    # pose deviation from the (re-anchored) proprioception prior
    prior_camera_rotations, prior_camera_centers = rig_model.poses(frames)
    position_deviations_mm: FloatArray = np.linalg.norm(camera_centers - prior_camera_centers, axis=1)
    rotation_deviations_deg: FloatArray = np.array(
        [
            np.degrees(np.linalg.norm(Rotation.from_matrix(prior_camera_rotations[index].T @ camera_rotations[index]).as_rotvec()))
            for index in range(len(frames))
        ]
    )
    print(
        f"      pose deviation from prior: |dt| median {np.median(position_deviations_mm):.2f}mm "
        f"max {position_deviations_mm.max():.2f}mm; |dr| median {np.median(rotation_deviations_deg):.2f}deg max {rotation_deviations_deg.max():.2f}deg"
    )

    # --- reject gross-outlier frames -----------------------------------------
    # The gantry encoders locate the camera to ~mm, so a frame whose recovered
    # pose sits far from proprioception is a false feature-lock on repetitive/
    # low-texture skin, not real motion (breathing and stabilization motion are smaller).
    # Trust the gantry for those frames: snap the pose back to the prior (keeps
    # their texture coverage roughly right) and drop their observations, then
    # re-triangulate_3d_feature_tracks the surviving landmarks from the trustworthy rays only --
    # otherwise a couple of bad frames bend the subject surface into a spike
    # and smear the ortho-texture.
    if args.reject_pose_mm > 0 or args.reject_rot_deg > 0:
        outlier_frame_mask: NDArray[np.bool_] = np.zeros(len(frames), bool)
        if args.reject_pose_mm > 0:
            outlier_frame_mask |= position_deviations_mm > args.reject_pose_mm
        if args.reject_rot_deg > 0:
            outlier_frame_mask |= rotation_deviations_deg > args.reject_rot_deg
        if outlier_frame_mask.any():
            outlier_frame_indices: IntArray = np.where(outlier_frame_mask)[0].astype(np.int32)
            for index in outlier_frame_indices:
                camera_rotations[index], camera_centers[index] = prior_camera_rotations[index].copy(), prior_camera_centers[index].copy()  # trust the gantry
            outlier_frame_index_set: set[int] = set(int(index) for index in outlier_frame_indices)
            retained_observation_mask: NDArray[np.bool_] = np.array([int(observation_frame_index) not in outlier_frame_index_set for observation_frame_index in bundle_adjustment_result["obs_frame"]], dtype=bool)
            bundle_adjustment_result["obs_frame"] = bundle_adjustment_result["obs_frame"][retained_observation_mask]
            bundle_adjustment_result["obs_uv"] = bundle_adjustment_result["obs_uv"][retained_observation_mask]
            bundle_adjustment_result["obs_track"] = bundle_adjustment_result["obs_track"][retained_observation_mask]
            # re-triangulate_3d_feature_tracks landmarks that still have >=2 trustworthy views;
            # keep the old position for the rest (they are dropped later by the
            # >=3-obs surface-fit gate, but must stay finite for the bounds calc).
            retriangulated_landmark_points: FloatArray = triangulate_3d_feature_tracks(bundle_adjustment_result["obs_frame"], bundle_adjustment_result["obs_uv"], bundle_adjustment_result["obs_track"], len(landmark_points), camera_rotations, camera_centers, rig_model)
            observation_count_by_landmark: NDArray[np.int64] = np.bincount(bundle_adjustment_result["obs_track"], minlength=len(landmark_points))
            landmark_retriangulation_valid: NDArray[np.bool_] = (observation_count_by_landmark >= 2) & np.isfinite(retriangulated_landmark_points).all(1)
            landmark_points = np.where(landmark_retriangulation_valid[:, None], retriangulated_landmark_points, landmark_points)
            bundle_adjustment_result["X"] = landmark_points
            print(
                f"      ! rejected {len(outlier_frame_indices)} gross-outlier frame(s) "
                f"(>{args.reject_pose_mm:.0f}mm or >{args.reject_rot_deg:.0f}deg "
                f"off gantry): {sorted(outlier_frame_index_set)} -> snapped to proprioception, "
                f"{int((~retained_observation_mask).sum())} obs dropped, {int(landmark_retriangulation_valid.sum())} landmarks "
                f"re-triangulated"
            )

    # standoff sensor agreement
    reprojection_errors, camera_depths = compute_feature_track_reprojection_errors(landmark_points, bundle_adjustment_result["obs_frame"], bundle_adjustment_result["obs_uv"], bundle_adjustment_result["obs_track"], camera_rotations, camera_centers, rig_model)
    median_depth_by_frame: FloatArray = np.full(len(frames), np.nan)
    for frame in frames:
        mask = bundle_adjustment_result["obs_frame"] == frame.idx
        if mask.sum() > 10:
            median_depth_by_frame[frame.idx] = np.median(camera_depths[mask])
    standoff_readings: FloatArray = np.array([frame.standoff for frame in frames])
    landmark_retriangulation_valid: NDArray[np.bool_] = ~np.isnan(median_depth_by_frame)
    standoff_depth_offset = float(np.median(median_depth_by_frame[landmark_retriangulation_valid] - standoff_readings[landmark_retriangulation_valid]))
    standoff_depth_rmse = float(np.sqrt(np.mean((median_depth_by_frame[landmark_retriangulation_valid] - standoff_readings[landmark_retriangulation_valid] - standoff_depth_offset) ** 2)))
    print(f"      standoff sensors vs recovered depth: offset {standoff_depth_offset:+.1f}mm, rms {standoff_depth_rmse:.2f}mm")

    # settled vs unsettled
    median_reprojection_error_by_frame: FloatArray = np.full(len(frames), np.nan)
    for frame in frames:
        mask = bundle_adjustment_result["obs_frame"] == frame.idx
        if mask.sum():
            median_reprojection_error_by_frame[frame.idx] = np.median(reprojection_errors[mask])
    settled: NDArray[np.bool_] = np.array([frame.settled for frame in frames])
    msg_settle = ""
    if settled.any() and (~settled).any():
        settled_median_error = np.nanmedian(median_reprojection_error_by_frame[settled])
        unsettled_median_error = np.nanmedian(median_reprojection_error_by_frame[~settled])
        msg_settle = f"median reproj: settled {settled_median_error:.2f}px vs unsettled {unsettled_median_error:.2f}px"
        print(f"      {msg_settle}")

    print("[7/9] fitting surface heightfield")
    # orient: does +z normal face the cameras?
    surface_normal_sign = 1.0 if (camera_centers[:, 2].mean() > np.median(landmark_points[:, 2])) else -1.0
    # Auto contour metric: a strongly curved subject scanned with the rig tilting
    # to follow the surface has a wide RX spread. Surface RMS provides an
    # independent confirmation. The gate only changes surface smoothing, so the
    # bounds and landmark selection remain identical.
    frame_x_rotation_angles: FloatArray = np.array([frame.rx for frame in frames])
    frame_x_rotation_spread_deg = float(np.degrees(frame_x_rotation_angles.max() - frame_x_rotation_angles.min()))
    # robust bounds + well-supported landmarks only: a handful of blown-up
    # tracks must not inflate the grid (oscillating extrapolation wrecks the
    # arc-length parameterisation downstream)
    landmark_x_percentiles: FloatArray = np.percentile(landmark_points[:, 0], [0.5, 99.5])
    landmark_y_percentiles: FloatArray = np.percentile(landmark_points[:, 1], [0.5, 99.5])
    landmark_x_min, landmark_x_max = landmark_x_percentiles[0] - 8, landmark_x_percentiles[1] + 8
    landmark_y_min, landmark_y_max = landmark_y_percentiles[0] - 8, landmark_y_percentiles[1] + 8
    observation_count_by_landmark: NDArray[np.int64] = np.bincount(bundle_adjustment_result["obs_track"], minlength=len(landmark_points))
    landmark_surface_fit_weights: FloatArray = np.clip(observation_count_by_landmark - 1, 1, 8).astype(float)
    surface_fit_landmark_mask: NDArray[np.bool_] = (landmark_points[:, 0] >= landmark_x_min) & (landmark_points[:, 0] <= landmark_x_max) & (landmark_points[:, 1] >= landmark_y_min) & (landmark_points[:, 1] <= landmark_y_max)
    well_observed_landmark_mask: NDArray[np.bool_] = observation_count_by_landmark >= 3
    if (surface_fit_landmark_mask & well_observed_landmark_mask).mean() > 0.3:
        surface_fit_landmark_mask &= well_observed_landmark_mask
    print(
        f"      using {surface_fit_landmark_mask.sum()}/{len(landmark_points)} landmarks "
        f"(robust bounds x[{landmark_x_min:.0f},{landmark_x_max:.0f}] y[{landmark_y_min:.0f},{landmark_y_max:.0f}], "
        f">=3-obs tracks)"
    )
    fitted_surface: Surface
    fitted_surface, surface_fit_rms = fit_surface_heightfield(
        landmark_points[surface_fit_landmark_mask], (landmark_x_min, landmark_x_max, landmark_y_min, landmark_y_max), args.surface_pitch, args.surface_smooth, initial_landmark_weights=landmark_surface_fit_weights[surface_fit_landmark_mask]
    )
    # Finalize the contour gate from both RX spread and landmark-surface RMS so
    # isolated noisy landmarks cannot enable it on their own.
    contour_smoothing_enabled = (args.contour == "on") or (
        args.contour == "auto"
        and frame_x_rotation_spread_deg > args.contour_rx_thresh_deg
        and surface_fit_rms > args.contour_rms_thresh
    )
    if contour_smoothing_enabled:
        # Refit with stronger smoothing so depth noise and breathing do not turn
        # the broad measured contour into sharp peaks and valleys.
        fitted_surface, contour_fit_rms = fit_surface_heightfield(
            landmark_points[surface_fit_landmark_mask], (landmark_x_min, landmark_x_max, landmark_y_min, landmark_y_max), args.surface_pitch, args.contour_smooth, initial_landmark_weights=landmark_surface_fit_weights[surface_fit_landmark_mask]
        )
        print(
            f"      [contour] enabled (RX spread {frame_x_rotation_spread_deg:.0f}deg, surface RMS "
            f"{surface_fit_rms:.0f}mm): landmark surface smoothing="
            f"{args.contour_smooth:.0f}"
        )
    texture_parameters = TexParam(fitted_surface, rig_model.base_R, rig_model.base_t)

    frame_deformation_warps: Warp = None
    if args.deformable:
        print("      deformable alignment (smooth per-frame map warp)")
        frame_deformation_warps = align_frames_with_deformable_surface_warps(
            frames, pairs, camera_rotations, camera_centers, rig_model, fitted_surface, texture_parameters, reg=args.deformable_reg, order=args.deformable_order
        )

    frame_texture_gains: NDArray[np.float32] | None = None
    if args.lf_gain != "off":
        frame_texture_gains = estimate_camera_frame_texture_gains(
            frames,
            rig_model,
            bundle_adjustment_result["obs_frame"],
            bundle_adjustment_result["obs_uv"],
            bundle_adjustment_result["obs_track"],
            bundle_adjustment_result.get("err"),
            mode=args.lf_gain,
        )

    print("[8/9] rendering ortho-texture")
    rendered_texture: NDArray[np.uint8]
    texture_accumulation_weights: NDArray[np.float32] | None
    texture_bounds: Bounds
    rendered_texture, texture_accumulation_weights, texture_bounds = render_surface_texture(
        frames,
        camera_rotations,
        camera_centers,
        rig_model,
        fitted_surface,
        texture_parameters,
        args.texture_ppmm,
        surface_normal_sign,
        output_directory,
        args.blend_sharpness,
        args.focus_weight,
        args.blend,
        frame_group,
        frame_deformation_warps,
        args.hf_coherence_mm,
        args.hf_cross_group,
        args.group_feather_mm,
        args.max_incidence_deg,
        device="cpu" if any(frame.mask_path for frame in frames) else args.device,  # GPU geometry has no mask term
        frame_gain=frame_texture_gains,
    )
    if texture_accumulation_weights is None:
        raise RuntimeError("Texture renderer did not return accumulation weights")
    pos: NDArray[np.float32]
    nrm: NDArray[np.float32]
    uvn: NDArray[np.float32]
    faces: NDArray[np.int64]
    pos, nrm, uvn, faces = build_surface_mesh(
        fitted_surface,
        texture_parameters,
        texture_bounds,
        texture_accumulation_weights,
        args.texture_ppmm,
        args.mesh_pitch,
        surface_normal_sign,
        (args.mesh_smooth[0], args.mesh_smooth[1]),
    )
    return cast(ReconstructionSolution, SimpleNamespace(
        bundle_adjustment_result=bundle_adjustment_result,
        R=camera_rotations,
        C=camera_centers,
        X=landmark_points,
        dt=position_deviations_mm,
        dr=rotation_deviations_deg,
        med_err=median_reprojection_error_by_frame,
        surf=fitted_surface,
        surf_rms=surface_fit_rms,
        texture_parameters=texture_parameters,
        tex=rendered_texture,
        wacc=texture_accumulation_weights,
        tex_bounds=texture_bounds,
        pos=pos,
        nrm=nrm,
        uvn=uvn,
        faces=faces,
        msg_settle=msg_settle,
        dzfit=standoff_depth_offset,
        so_rms=standoff_depth_rmse,
    ))
