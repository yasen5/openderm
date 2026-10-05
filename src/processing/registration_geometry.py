"""Rig fitting, overlap discovery, tracking, and bundle adjustment."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .registration_features import Frame, Pair

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
RotationArray = NDArray[np.float64]
CenterArray = NDArray[np.float64]


class RegistrationArgs(Protocol):
    max_corr_per_pair: int
    sigma_r: float
    rounds: int
    rig_from: str | None
    sigma_t: float
    sigma_px: float
    @property
    def fx_full(self) -> float | None: ...
    fit_k1: bool
    @property
    def group_align(self) -> str: ...


class BundleAdjustmentResult(TypedDict):
    X: FloatArray
    track_err: FloatArray
    obs_frame: IntArray
    obs_uv: FloatArray
    obs_track: IntArray
    err: FloatArray
    R: RotationArray
    C: CenterArray
    rms: float
    history: list[float]


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
    lever: FloatArray  # camera centre offset from toolhead, rotates with rx (mm)
    Rm: RotationArray  # 3x3 camera mount rotation (cam->world at rx=0)
    dz0: float  # standoff sensor -> optical centre depth offset (mm)
    downscale: int = 1
    base_R: RotationArray = field(default_factory=lambda: np.eye(3))  # gauge fix
    base_t: CenterArray = field(default_factory=lambda: np.zeros(3))

    def poses(self, frames: list[Frame]) -> tuple[RotationArray, CenterArray]:
        """Nominal (R_cam2world (n,3,3), C (n,3)) for every frame."""
        frame_count = len(frames)
        camera_rotations = np.zeros((frame_count, 3, 3))
        camera_centers = np.zeros((frame_count, 3))
        for frame in frames:
            Rrx = Rotation.from_rotvec([self.sign * frame.rx, 0, 0]).as_matrix()
            camera_centers[frame.idx] = self.base_t + self.base_R @ (frame.g + Rrx @ self.lever)
            camera_rotations[frame.idx] = self.base_R @ Rrx @ self.Rm
        return camera_rotations, camera_centers

    def depth(self, frame: Frame) -> float:
        return frame.standoff + self.dz0


def project_world_points_into_camera(
    landmark_points: FloatArray,
    camera_to_world_rotation: RotationArray,
    camera_centers: FloatArray,
    focal_length_px: float,
    k1: float,
    principal_point_x: float,
    principal_point_y: float,
) -> tuple[FloatArray, FloatArray]:
    """World pts (N,3) -> pixel (N,2) + cam-z (N,). Rcw: cam->world 3x3."""
    camera_points = (landmark_points - camera_centers) @ camera_to_world_rotation  # = Rcw^T (X - C)
    camera_depth = np.maximum(camera_points[:, 2], 1e-6)
    normalized_image_points = camera_points[:, :2] / camera_depth[:, None]
    normalized_radius_squared = (normalized_image_points**2).sum(1)
    radial_distortion_factor = 1.0 + k1 * normalized_radius_squared
    image_points = np.empty_like(normalized_image_points)
    image_points[:, 0] = focal_length_px * normalized_image_points[:, 0] * radial_distortion_factor + principal_point_x
    image_points[:, 1] = focal_length_px * normalized_image_points[:, 1] * radial_distortion_factor + principal_point_y
    return image_points, camera_points[:, 2]


def undistort_image_points_to_normalized_camera(
    uv: FloatArray, fx: float, k1: float, cx: float, cy: float
) -> FloatArray:
    """Pixel (N,2) -> normalised undistorted (N,2)."""
    xn = (uv - [cx, cy]) / fx
    xu = xn.copy()
    for iteration_index in range(3):
        r2 = (xu**2).sum(1)
        xu = xn / (1.0 + k1 * r2)[:, None]
    return xu


def estimate_initial_rig_camera_model(
    frames: list[Frame], consecutive_pairs: list[Pair], image_width: int, image_height: int,
    downscale: int, initial_focal_length_px: float,
) -> tuple[RigModel, float]:
    """Fit the mechanical rig model from consecutive-pair affine transforms.

    The focal-length estimate here is an initialization for the mechanical fit;
    a supplied --fx-full value replaces it before bundle adjustment. Five sample
    points per pair capture translation, rotation, and scale of the measured
    affine.
    """
    cx, cy = image_width / 2.0, image_height / 2.0
    sample_image_points = np.array(
        [
            [cx, cy],
            [cx - image_width / 4, cy - image_height / 4],
            [cx + image_width / 4, cy - image_height / 4],
            [cx + image_width / 4, cy + image_height / 4],
            [cx - image_width / 4, cy + image_height / 4],
        ],
        float,
    )

    # measured destinations of the sample points under each pair's affine
    measured_image_positions = []
    for consecutive_pair in consecutive_pairs:
        pair_rotation_radians = math.radians(consecutive_pair.rot_deg)
        cosine_component = consecutive_pair.scale * math.cos(pair_rotation_radians)
        sine_component = consecutive_pair.scale * math.sin(pair_rotation_radians)
        pair_affine = np.array(
            [[cosine_component, -sine_component, consecutive_pair.tx], [sine_component, cosine_component, consecutive_pair.ty]]
        )
        measured_image_positions.append(sample_image_points @ pair_affine[:, :2].T + pair_affine[:, 2])
    measured_image_positions = np.array(measured_image_positions)  # (npair, 5, 2)

    gantry_positions = np.array([frame.gauge for frame in frames])
    frame_x_rotation_angles = np.array([frame.rx for frame in frames])
    frame_standoff_measurements = np.array([frame.standoff for frame in frames])
    pair_first_frame_indices = np.array([consecutive_pair.index for consecutive_pair in consecutive_pairs])
    pair_second_frame_indices = np.array([consecutive_pair.neighbor_index for consecutive_pair in consecutive_pairs])

    def build_initial_camera_mount_rotation(optical_axis_tilt_degrees: float, camera_roll_degrees: float) -> RotationArray:
        camera_forward_axis = np.array([0, 0, -1.0])
        camera_roll_radians = math.radians(camera_roll_degrees)
        camera_space_points = np.array([math.cos(camera_roll_radians), math.sin(camera_roll_radians), 0.0])
        camera_up_axis = np.cross(camera_forward_axis, camera_space_points)
        camera_mount_basis = np.stack([camera_space_points, camera_up_axis, camera_forward_axis], axis=1)
        return Rotation.from_rotvec([math.radians(optical_axis_tilt_degrees), 0, 0]).as_matrix() @ camera_mount_basis

    def build_pair_reprojection_residual_function(
        frame_rotation_sign: float, initial_mount_rotation: RotationArray
    ) -> Callable[[FloatArray], FloatArray]:
        frame_x_rotations = Rotation.from_rotvec(np.outer(frame_rotation_sign * frame_x_rotation_angles, [1, 0, 0])).as_matrix()

        def compute_reprojection_residuals(parameter_vector: FloatArray) -> FloatArray:
            focal_length_px, depth_offset_mm = parameter_vector[0], parameter_vector[1]
            lever = parameter_vector[2:5]
            camera_mount_rotation = initial_mount_rotation @ Rotation.from_rotvec(parameter_vector[5:8]).as_matrix()
            camera_centers = gantry_positions + frame_x_rotations @ lever  # (n,3)
            camera_rotations = np.einsum("nij,jk->nik", frame_x_rotations, camera_mount_rotation)  # cam->world
            pair_camera_depths = frame_standoff_measurements[pair_first_frame_indices] + depth_offset_mm  # (npair,)
            normalized_image_points = (sample_image_points - [cx, cy]) / focal_length_px  # (5,2)
            camera_space_points = np.empty((len(pair_first_frame_indices), 5, 3))
            camera_space_points[:, :, :2] = normalized_image_points[None] * pair_camera_depths[:, None, None]
            camera_space_points[:, :, 2] = pair_camera_depths[:, None]
            world_sample_points = np.einsum("pij,pkj->pki", camera_rotations[pair_first_frame_indices], camera_space_points) + camera_centers[pair_first_frame_indices][:, None, :]
            distance = world_sample_points - camera_centers[pair_second_frame_indices][:, None, :]
            neighbor_camera_points = np.einsum("pki,pij->pkj", distance, camera_rotations[pair_second_frame_indices])
            camera_depths = np.maximum(neighbor_camera_points[:, :, 2], 1e-6)
            projected_image_points = focal_length_px * neighbor_camera_points[:, :, :2] / camera_depths[..., None] + [cx, cy]
            reprojection_residuals = (projected_image_points - measured_image_positions).ravel() / 3.0
            parameter_prior_residuals = np.concatenate([[(depth_offset_mm - 50.0) / 100.0], lever / 200.0, parameter_vector[5:8] / 0.5])
            return np.concatenate([reprojection_residuals, parameter_prior_residuals])

        return compute_reprojection_residuals

    # dz0 can be large and positive: the pinhole centre (entrance pupil) of a
    # long macro lens sits far behind the standoff sensors' reference plane.
    parameter_lower_bounds = [initial_focal_length_px * 0.1, -20, -300, -300, -300, -1.5, -1.5, -1.5]
    parameter_upper_bounds = [initial_focal_length_px * 20, 400, 300, 300, 300, 1.5, 1.5, 1.5]
    initial_parameter_vector = np.zeros(8)
    initial_parameter_vector[0] = initial_focal_length_px
    fit_trials = []
    for frame_rotation_sign in (1.0, -1.0):
        for optical_axis_tilt_deg in (-70, -35, 0, 35, 70):
            for camera_roll_deg in (0, 90, 180, 270):
                residual_function = build_pair_reprojection_residual_function(frame_rotation_sign, build_initial_camera_mount_rotation(optical_axis_tilt_deg, camera_roll_deg))
                try:
                    fit_result = least_squares(
                        residual_function,
                        initial_parameter_vector,
                        method="trf",
                        loss="soft_l1",
                        f_scale=5.0,
                        max_nfev=25,
                        bounds=(parameter_lower_bounds, parameter_upper_bounds),
                    )
                except Exception:
                    continue
                fit_trials.append((fit_result.cost, fit_result.x, frame_rotation_sign, optical_axis_tilt_deg, camera_roll_deg))
    fit_trials.sort(key=lambda fit_trial: fit_trial[0])
    best = None
    for seed_cost, seed_parameters, frame_rotation_sign, optical_axis_tilt_deg, camera_roll_deg in fit_trials[:3]:  # polish the 3 best seeds
        residual_function = build_pair_reprojection_residual_function(frame_rotation_sign, build_initial_camera_mount_rotation(optical_axis_tilt_deg, camera_roll_deg))
        fit_result = least_squares(
            residual_function, seed_parameters, method="trf", loss="soft_l1", f_scale=5.0, max_nfev=150, bounds=(parameter_lower_bounds, parameter_upper_bounds)
        )
        if best is None or fit_result.cost < best[0]:
            best = (fit_result.cost, fit_result, frame_rotation_sign, build_initial_camera_mount_rotation(optical_axis_tilt_deg, camera_roll_deg), optical_axis_tilt_deg, camera_roll_deg)
    assert best is not None, "initial rig fit produced no valid optimizer trials"
    fit_cost, fit_result, frame_rotation_sign, initial_mount_rotation, optical_axis_tilt_deg, camera_roll_deg = best
    parameter_vector = fit_result.x
    camera_mount_rotation = initial_mount_rotation @ Rotation.from_rotvec(parameter_vector[5:8]).as_matrix()
    rig_model = RigModel(
        fx=parameter_vector[0],
        k1=0.0,
        cx=cx,
        cy=cy,
        sign=frame_rotation_sign,
        lever=parameter_vector[2:5].copy(),
        Rm=camera_mount_rotation,
        dz0=parameter_vector[1],
        downscale=downscale,
    )
    # robust-ish rms of the pixel part only
    pixel_residual_count = measured_image_positions.size
    residual_function = build_pair_reprojection_residual_function(frame_rotation_sign, initial_mount_rotation)
    pixel_reprojection_residuals = residual_function(parameter_vector)[:pixel_residual_count] * 3.0
    rms = float(np.sqrt(np.mean(pixel_reprojection_residuals**2)))
    print(
        f"      best init: frame_rotation_sign={frame_rotation_sign:+.0f} optical_axis_tilt_degrees={optical_axis_tilt_deg} camera_roll_degrees={camera_roll_deg}; "
        f"fit rms {rms:.2f}px over {len(consecutive_pairs)} consecutive pairs"
    )
    return rig_model, rms


# ----------------------------------------------------------------------------
# overlap prediction on the (curved) surface
# ----------------------------------------------------------------------------
def predict_frame_pair_translation(
    rig_model: RigModel, frames: list[Frame], camera_rotations: RotationArray,
    camera_centers: CenterArray, index: int, neighbor_index: int,
) -> FloatArray:
    """Predicted image translation i->j (mean over sample pts) in ds px."""
    image_width, image_height = rig_model.cx * 2, rig_model.cy * 2
    sample_image_points = np.array(
        [
            [rig_model.cx, rig_model.cy],
            [image_width * 0.3, image_height * 0.3],
            [image_width * 0.7, image_height * 0.7],
        ]
    )
    camera_depth_mm = rig_model.depth(frames[index])
    normalized_sample_points = undistort_image_points_to_normalized_camera(
        sample_image_points, rig_model.fx, rig_model.k1, rig_model.cx, rig_model.cy
    )
    camera_space_points = np.concatenate(
        [normalized_sample_points * camera_depth_mm, np.full((len(sample_image_points), 1), camera_depth_mm)],
        1,
    )
    world_sample_points = camera_space_points @ camera_rotations[index].T + camera_centers[index]
    projected_image_points, camera_depths = project_world_points_into_camera(
        world_sample_points,
        camera_rotations[neighbor_index],
        camera_centers[neighbor_index],
        rig_model.fx,
        rig_model.k1,
        rig_model.cx,
        rig_model.cy,
    )
    return (projected_image_points - sample_image_points).mean(0)


def find_overlapping_frame_pairs(
    rig_model: RigModel, frames: list[Frame], camera_rotations: RotationArray,
    camera_centers: CenterArray, overlap_frac: float, max_partners: int,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], float]]:
    """Predict pairwise overlap by projecting footprints onto the mean plane."""
    frame_count = len(frames)
    image_width, image_height = rig_model.cx * 2, rig_model.cy * 2
    image_corners = np.array(
        [[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]], float
    )
    centers = np.zeros((frame_count, 3))
    quads = np.zeros((frame_count, 4, 3))
    for frame in frames:
        camera_depth_mm = rig_model.depth(frame)
        normalized_corner_points = undistort_image_points_to_normalized_camera(
            image_corners, rig_model.fx, rig_model.k1, rig_model.cx, rig_model.cy
        )
        camera_space_corners = np.concatenate(
            [normalized_corner_points * camera_depth_mm, np.full((4, 1), camera_depth_mm)], 1
        )
        quads[frame.idx] = camera_space_corners @ camera_rotations[frame.idx].T + camera_centers[frame.idx]
        centers[frame.idx] = camera_centers[frame.idx] + camera_depth_mm * camera_rotations[frame.idx][:, 2]
    ctr = centers.mean(0)
    left_singular_vectors, singular_values, right_singular_vectors = np.linalg.svd(centers - ctr, full_matrices=False)
    e1, e2 = right_singular_vectors[0], right_singular_vectors[1]
    q2 = np.stack([(quads - ctr) @ e1, (quads - ctr) @ e2], axis=-1)  # (n,4,2)
    lo = q2.min(1)
    hi = q2.max(1)
    area = (hi - lo).prod(1)
    overlaps = {}
    for index in range(frame_count):
        ix = np.maximum(0.0, np.minimum(hi[index, 0], hi[:, 0]) - np.maximum(lo[index, 0], lo[:, 0]))
        iy = np.maximum(0.0, np.minimum(hi[index, 1], hi[:, 1]) - np.maximum(lo[index, 1], lo[:, 1]))
        frac = ix * iy / np.minimum(area[index], area)
        for neighbor_index in range(index + 1, frame_count):
            if frac[neighbor_index] >= overlap_frac:
                overlaps[(index, neighbor_index)] = float(frac[neighbor_index])
    partners = {index: [] for index in range(frame_count)}
    for (index, neighbor_index), fr in overlaps.items():
        partners[index].append((fr, neighbor_index))
        partners[neighbor_index].append((fr, index))
    keep = set()
    for item_index, lst in partners.items():
        for overlap_fraction, partner_index in sorted(lst, reverse=True)[:max_partners]:
            keep.add((min(item_index, partner_index), max(item_index, partner_index)))
    return keep, overlaps


# ----------------------------------------------------------------------------
# tracks (union-find over matched keypoints)
# ----------------------------------------------------------------------------
def build_3d_feature_tracks(
    frames: list[Frame], pairs: list[Pair], max_corr_per_pair: int
) -> tuple[IntArray, FloatArray, IntArray, int]:
    t0 = time.time()
    nkp: list[int] = []
    for frame in frames:
        if frame.kp is None:
            raise ValueError(f"Frame {frame.idx} has no extracted keypoints")
        nkp.append(len(frame.kp))
    off = np.concatenate([[0], np.cumsum(nkp)])
    parent = np.arange(off[-1], dtype=np.int64)

    def find(parent_node_index: int) -> int:
        root_node_index = parent_node_index
        while parent[root_node_index] != root_node_index:
            root_node_index = parent[root_node_index]
        while parent[parent_node_index] != root_node_index:
            parent[parent_node_index], parent_node_index = root_node_index, parent[parent_node_index]
        return root_node_index

    rng = np.random.default_rng(0)
    for point in pairs:
        correspondence_count = len(point.src_kp)
        selected_correspondence_indices = rng.permutation(correspondence_count)[:max_corr_per_pair]
        for source_keypoint_index, destination_keypoint_index in zip(
            point.src_kp[selected_correspondence_indices], point.dst_kp[selected_correspondence_indices]
        ):
            source_root = find(off[point.index] + source_keypoint_index)
            destination_root = find(off[point.neighbor_index] + destination_keypoint_index)
            if source_root != destination_root:
                parent[destination_root] = source_root

    # collect components
    groups = {}
    used = set()
    for point in pairs:
        for arr, fi in ((point.src_kp, point.index), (point.dst_kp, point.neighbor_index)):
            for first_value in arr:
                used.add(off[fi] + first_value)
    for node in used:
        groups.setdefault(find(node), []).append(node)

    obs_frame, obs_uv, obs_track = [], [], []
    tid = 0
    n_dup = 0
    fr_of = np.searchsorted(off, np.arange(off[-1]), side="right") - 1
    for root, nodes in groups.items():
        if len(nodes) < 2:
            continue
        frs = [int(fr_of[item_count]) for item_count in nodes]
        if len(set(frs)) != len(frs):  # same frame twice -> ambiguous
            n_dup += 1
            continue
        if len(set(frs)) < 2:
            continue
        for n_, fi in zip(nodes, frs):
            kpidx = int(n_ - off[fi])
            obs_frame.append(fi)
            keypoints = frames[fi].kp
            assert keypoints is not None
            obs_uv.append(keypoints[kpidx].pt)
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
def triangulate_3d_feature_tracks(
    obs_frame: IntArray, obs_uv: FloatArray, obs_track: IntArray, ntracks: int,
    camera_rotations: RotationArray, camera_centers: CenterArray, rig_model: RigModel,
) -> FloatArray:
    xu = undistort_image_points_to_normalized_camera(obs_uv, rig_model.fx, rig_model.k1, rig_model.cx, rig_model.cy)
    d_cam = np.concatenate([xu, np.ones((len(xu), 1))], 1)
    d_w = np.einsum("nij,nj->ni", camera_rotations[obs_frame], d_cam)
    d_w /= np.linalg.norm(d_w, axis=1, keepdims=True)
    matrix = np.eye(3)[None] - d_w[:, :, None] * d_w[:, None, :]
    triangulation_normal_matrices = np.zeros((ntracks, 3, 3))
    triangulation_right_hand_sides = np.zeros((ntracks, 3))
    np.add.at(triangulation_normal_matrices, obs_track, matrix)
    np.add.at(
        triangulation_right_hand_sides,
        obs_track,
        np.einsum("nij,nj->ni", matrix, camera_centers[obs_frame]),
    )
    triangulation_normal_matrices += np.eye(3)[None] * 1e-9
    # NumPy 2.x interprets a 2-D right-hand side as a stack of matrices,
    # which broadcasts ``(tracks, 3)`` into ``(tracks, tracks, 3)`` here.
    # Make the per-track vector dimension explicit across NumPy versions.
    return cast(
        FloatArray,
        np.linalg.solve(
            triangulation_normal_matrices, triangulation_right_hand_sides[..., None]
        )[..., 0],
    )


def compute_feature_track_reprojection_errors(
    landmark_points: FloatArray, obs_frame: IntArray, obs_uv: FloatArray,
    obs_track: IntArray, camera_rotations: RotationArray,
    camera_centers: CenterArray, rig_model: RigModel,
) -> tuple[FloatArray, FloatArray]:
    observed_landmark_points = landmark_points[obs_track]
    observed_camera_centers = camera_centers[obs_frame]
    camera_space_points = np.einsum(
        "ni,nij->nj", observed_landmark_points - observed_camera_centers, camera_rotations[obs_frame]
    )
    camera_depths = np.maximum(camera_space_points[:, 2], 1e-6)
    normalized_image_points = camera_space_points[:, :2] / camera_depths[:, None]
    normalized_radius_squared = (normalized_image_points**2).sum(1)
    projected_image_points = (
        rig_model.fx * normalized_image_points * (1 + rig_model.k1 * normalized_radius_squared)[:, None]
        + [rig_model.cx, rig_model.cy]
    )
    pixel_reprojection_errors = np.linalg.norm(projected_image_points - obs_uv, axis=1)
    return pixel_reprojection_errors, camera_space_points[:, 2]


def refine_all_camera_poses_from_tracks(
    frames: list[Frame], landmark_points: FloatArray, obs_frame: IntArray,
    obs_uv: FloatArray, obs_track: IntArray, camera_rotations: RotationArray,
    camera_centers: CenterArray, R0: RotationArray, C0: CenterArray,
    rig_model: RigModel, sigma_px: float, sigma_t: float, sigma_r: float,
) -> None:
    order = np.argsort(obs_frame, kind="stable")
    of, ou, ot = obs_frame[order], obs_uv[order], obs_track[order]
    bounds = np.searchsorted(of, np.arange(len(frames) + 1))
    n_small = 0
    for frame in frames:
        index = frame.idx
        first_observation_index, observation_end_index = bounds[index], bounds[index + 1]
        if observation_end_index - first_observation_index < 20:
            n_small += 1
            continue
        frame_landmark_points = landmark_points[ot[first_observation_index:observation_end_index]]
        observed_image_points = ou[first_observation_index:observation_end_index]
        # init delta from current pose relative to nominal anchor
        d0 = np.zeros(6)
        d0[:3] = camera_centers[index] - C0[index]
        d0[3:] = Rotation.from_matrix(R0[index].T @ camera_rotations[index]).as_rotvec()

        def residuals(pose_delta: FloatArray) -> FloatArray:
            camera_to_world_rotation = R0[index] @ Rotation.from_rotvec(pose_delta[3:]).as_matrix()
            camera_center = C0[index] + pose_delta[:3]
            projected_image_points, camera_depths = project_world_points_into_camera(
                frame_landmark_points,
                camera_to_world_rotation,
                camera_center,
                rig_model.fx,
                rig_model.k1,
                rig_model.cx,
                rig_model.cy,
            )
            pixel_residuals = ((projected_image_points - observed_image_points) / sigma_px).ravel()
            return np.concatenate([pixel_residuals, pose_delta[:3] / sigma_t, pose_delta[3:] / sigma_r])

        # robust loss: early rounds still contain large-residual cross-row
        # obs that the pose must converge towards, not be dragged by
        optimization_result = least_squares(residuals, d0, method="trf", loss="soft_l1", f_scale=4.0, max_nfev=40)
        camera_rotations[index] = R0[index] @ Rotation.from_rotvec(optimization_result.x[3:]).as_matrix()
        camera_centers[index] = C0[index] + optimization_result.x[:3]
    if n_small:
        print(f"        ({n_small} frames with <20 obs kept at prior pose)")


def optimize_camera_intrinsics_from_tracks(
    landmark_points: FloatArray, obs_frame: IntArray, obs_uv: FloatArray,
    obs_track: IntArray, camera_rotations: RotationArray,
    camera_centers: CenterArray, rig_model: RigModel, nsub: int = 80000,
    fit_k1: bool = False,
) -> None:
    """Closed-form refit of fx (and optionally k1): uv-c = [xn, xn*r2]@[fx, fx*k1].

    k1 fitting is off by default: Canon applies lens corrections to JPGs, and
    in practice k1 just absorbs IS systematics and pegs its cap unstably.
    """
    if len(obs_uv) < 2000:
        return
    rng = np.random.default_rng(1)
    selected_observation_indices = rng.permutation(len(obs_uv))[:nsub]
    observed_landmark_points = landmark_points[obs_track[selected_observation_indices]]
    observed_camera_rotations = camera_rotations[obs_frame[selected_observation_indices]]
    observed_camera_centers = camera_centers[obs_frame[selected_observation_indices]]
    observed_image_points = obs_uv[selected_observation_indices]
    camera_space_points = np.einsum(
        "ni,nij->nj", observed_landmark_points - observed_camera_centers, observed_camera_rotations
    )
    camera_depths = np.maximum(camera_space_points[:, 2], 1e-6)
    normalized_image_points = camera_space_points[:, :2] / camera_depths[:, None]
    normalized_radius_squared = (normalized_image_points**2).sum(1)
    if fit_k1:
        intrinsic_design_matrix = np.stack(
            [normalized_image_points.ravel(), (normalized_image_points * normalized_radius_squared[:, None]).ravel()], 1
        )
    else:
        intrinsic_design_matrix = normalized_image_points.ravel()[:, None]
    centered_image_coordinates = (observed_image_points - [rig_model.cx, rig_model.cy]).ravel()
    intrinsic_coefficients: FloatArray = np.zeros(2 if fit_k1 else 1)
    for iteration_index in range(3):  # MAD-robust outlier trimming
        solved_intrinsic_coefficients, residual_sums, matrix_rank, singular_values = np.linalg.lstsq(
            intrinsic_design_matrix, centered_image_coordinates, rcond=None
        )
        intrinsic_coefficients = cast(FloatArray, solved_intrinsic_coefficients)
        intrinsic_fit_residuals = intrinsic_design_matrix @ intrinsic_coefficients - centered_image_coordinates
        residual_median = np.median(intrinsic_fit_residuals)
        robust_residual_scale = 1.4826 * np.median(np.abs(intrinsic_fit_residuals - residual_median)) + 1e-9
        inlier_observation_mask = np.abs(intrinsic_fit_residuals - residual_median) < 4.0 * robust_residual_scale
        if inlier_observation_mask.all():
            break
        intrinsic_design_matrix = intrinsic_design_matrix[inlier_observation_mask]
        centered_image_coordinates = centered_image_coordinates[inlier_observation_mask]
    focal_length_px = float(intrinsic_coefficients[0])
    if focal_length_px > 100.0:  # sanity: never collapse
        rig_model.fx = focal_length_px
        if fit_k1:
            rig_model.k1 = float(np.clip(float(intrinsic_coefficients[1]) / focal_length_px, -0.15, 0.15))


def _project_matrix_to_rotation(matrix: FloatArray) -> RotationArray:
    left_singular_vectors, singular_values, right_singular_vectors = np.linalg.svd(matrix)
    rotation_correction = np.diag(
        [1, 1, np.linalg.det(left_singular_vectors @ right_singular_vectors)]
    )
    return left_singular_vectors @ rotation_correction @ right_singular_vectors


def refit_rig_model_from_camera_poses(
    frames: list[Frame], camera_rotations: RotationArray,
    camera_centers: CenterArray, rig_model: RigModel,
) -> tuple[RotationArray, CenterArray]:
    """Re-anchor the rig model on the current BA poses.

    The pre-fit (consecutive-pair affines only) carries systematic bias; once
    BA has settled, the lever arm, the mount rotation AND a global base
    transform (gauge: rotation Q + translation T of the whole gantry frame)
    are re-estimated from the optimised poses, so the proprioception prior
    stops fighting that bias:  C ~ T + Q (g + Rrx lever),  R ~ Q Rrx Rm.
    """
    frame_x_rotations = np.stack(
        [Rotation.from_rotvec([rig_model.sign * frame.rx, 0, 0]).as_matrix() for frame in frames]
    )
    gantry_positions = np.array([frame.gauge for frame in frames])
    base_rotation = np.eye(3)
    base_translation = np.zeros(3)
    lever = rig_model.lever.copy()
    mount_rotation: RotationArray = rig_model.Rm.copy()
    for iteration_index in range(4):
        # Solve the lever arm given the current base rotation and translation.
        lever_design_matrix = np.einsum("ij,njk->nik", base_rotation, frame_x_rotations).reshape(-1, 3)
        lever_targets = (
            camera_centers - base_translation - gantry_positions @ base_rotation.T
        ).ravel()
        lever, residual_sums, matrix_rank, singular_values = np.linalg.lstsq(
            lever_design_matrix, lever_targets, rcond=None
        )
        # Solve the camera mount rotation given the current base rotation.
        mount_rotation_matrix = np.einsum(
            "nji,njk->ik",
            np.einsum("ij,njk->nik", base_rotation, frame_x_rotations),
            camera_rotations,
        )
        mount_rotation = _project_matrix_to_rotation(mount_rotation_matrix)
        # Fit the base rotation and translation from camera centers (Kabsch).
        predicted_gantry_centers = gantry_positions + np.einsum(
            "nij,j->ni", frame_x_rotations, lever
        )
        mean_camera_center = camera_centers.mean(0)
        mean_predicted_center = predicted_gantry_centers.mean(0)
        center_covariance = (predicted_gantry_centers - mean_predicted_center).T @ (
            camera_centers - mean_camera_center
        )
        base_rotation = _project_matrix_to_rotation(center_covariance.T)
        base_translation = mean_camera_center - base_rotation @ mean_predicted_center
    rig_model.lever = lever
    rig_model.Rm = mount_rotation
    rig_model.base_R = base_rotation
    rig_model.base_t = base_translation
    return rig_model.poses(frames)


def classify_frames_by_capture_direction(
    frames: list[Frame], camera_rotations: RotationArray, camera_centers: CenterArray,
    R0: RotationArray, C0: CenterArray, Zs: FloatArray,
    sigma_t: float = 1.5, sigma_r_deg: float = 3.0,
) -> None:
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
    for frame in frames:
        index = frame.idx
        design_matrix = Zs[index]
        item_index = design_matrix * design_matrix / st2 + 1.0 / sr2
        dtc = R0[index].T @ (camera_centers[index] - C0[index])
        rv = Rotation.from_matrix(R0[index].T @ camera_rotations[index]).as_rotvec()
        # valley pair (t_x, th_y): t_x' = t_x - Z s, th_y' = th_y + s
        score = (design_matrix * dtc[0] / st2 - rv[1] / sr2) / item_index
        dtc[0] -= design_matrix * score
        rv[1] += score
        # valley pair (t_y, th_x): t_y' = t_y + Z s, th_x' = th_x + s
        score = -(design_matrix * dtc[1] / st2 + rv[0] / sr2) / item_index
        dtc[1] += design_matrix * score
        rv[0] += score
        camera_centers[index] = C0[index] + R0[index] @ dtc
        camera_rotations[index] = R0[index] @ Rotation.from_rotvec(rv).as_matrix()


def bundle_adjust_camera_poses_and_feature_tracks(
    frames: list[Frame], pairs: list[Pair], rig_model: RigModel,
    R0: RotationArray, C0: CenterArray, args: RegistrationArgs,
) -> BundleAdjustmentResult:
    obs_frame, obs_uv, obs_track, ntracks = build_3d_feature_tracks(frames, pairs, args.max_corr_per_pair)
    camera_rotations, camera_centers = R0.copy(), C0.copy()
    if ntracks == 0:
        # no usable tracks (e.g. a sparse revisit group with zero intra-group
        # matches): the poses stay at the proprioception prior, which is what
        # anchors every group anyway; cross-group alignment is regularised so
        # a track-less group simply gets t~0
        print("      ! 0 tracks -- keeping proprioception-prior poses")
        return BundleAdjustmentResult(
            X=np.zeros((0, 3)),
            track_err=np.zeros(0),
            obs_frame=np.zeros(0, np.int32),
            obs_uv=np.zeros((0, 2)),
            obs_track=np.zeros(0, np.int32),
            err=np.zeros(0),
            R=camera_rotations,
            C=camera_centers,
            rms=0.0,
            history=[],
        )
    R0, C0 = R0.copy(), C0.copy()
    sigma_r = math.radians(args.sigma_r)
    Zs = np.array([rig_model.depth(frame) for frame in frames])
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
            Xs_ = triangulate_3d_feature_tracks(obs_frame[good], obs_uv[good], obs_track[good], ntracks, camera_rotations, camera_centers, rig_model)
            reprojection_errors, camera_depths = compute_feature_track_reprojection_errors(Xs_, obs_frame[good], obs_uv[good], obs_track[good], camera_rotations, camera_centers, rig_model)
            so_ = np.array([frames[index].standoff for index in obs_frame[good]])
            rig_model.dz0 = float(np.clip(np.median(camera_depths - so_), -20, 600))
            R0, C0 = refit_rig_model_from_camera_poses(frames, camera_rotations, camera_centers, rig_model)
            Zs = np.array([rig_model.depth(frame) for frame in frames])
            print(
                f"        re-anchored rig model: lever="
                f"({rig_model.lever[0]:.1f},{rig_model.lever[1]:.1f},{rig_model.lever[2]:.1f})mm, "
                f"dz0={rig_model.dz0:+.1f}mm"
            )
        # loose prior while the anchor still carries pre-fit bias
        st = args.sigma_t * (2.0 if rnd < switch else 1.0)
        sr = sigma_r * (2.0 if rnd < switch else 1.0)
        landmark_points = triangulate_3d_feature_tracks(obs_frame[good], obs_uv[good], obs_track[good], ntracks, camera_rotations, camera_centers, rig_model)
        cnt = np.bincount(obs_track[good], minlength=ntracks)
        # reproject ALL obs so early aggressive prunes can be re-admitted
        err, zc = compute_feature_track_reprojection_errors(landmark_points, obs_frame, obs_uv, obs_track, camera_rotations, camera_centers, rig_model)
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
        refine_all_camera_poses_from_tracks(
            frames,
            landmark_points,
            obs_frame[good],
            obs_uv[good],
            obs_track[good],
            camera_rotations,
            camera_centers,
            R0,
            C0,
            rig_model,
            args.sigma_px,
            st,
            sr,
        )
        print(f"        resect {time.time() - t_rs:.0f}s")
        if rnd >= 1 and not args.fx_full:
            landmark_points = triangulate_3d_feature_tracks(obs_frame[good], obs_uv[good], obs_track[good], ntracks, camera_rotations, camera_centers, rig_model)
            optimize_camera_intrinsics_from_tracks(
                landmark_points, obs_frame[good], obs_uv[good], obs_track[good], camera_rotations, camera_centers, rig_model, fit_k1=args.fit_k1
            )
            print(
                f"        intrinsics: fx={rig_model.fx:.1f}ds-px "
                f"({rig_model.fx * rig_model.downscale:.0f} full-res px), k1={rig_model.k1:+.4f}"
            )
    # re-split the translation/rotation valley to physical values, then one
    # tight-prior polish pass and a second re-split
    err, camera_depths = compute_feature_track_reprojection_errors(
        triangulate_3d_feature_tracks(obs_frame[good], obs_uv[good], obs_track[good], ntracks, camera_rotations, camera_centers, rig_model),
        obs_frame[good],
        obs_uv[good],
        obs_track[good],
        camera_rotations,
        camera_centers,
        rig_model,
    )
    med0 = float(np.median(err))
    classify_frames_by_capture_direction(frames, camera_rotations, camera_centers, R0, C0, Zs)
    landmark_points = triangulate_3d_feature_tracks(obs_frame[good], obs_uv[good], obs_track[good], ntracks, camera_rotations, camera_centers, rig_model)
    refine_all_camera_poses_from_tracks(
        frames,
        landmark_points,
        obs_frame[good],
        obs_uv[good],
        obs_track[good],
        camera_rotations,
        camera_centers,
        R0,
        C0,
        rig_model,
        args.sigma_px,
        1.5,
        math.radians(3.0),
    )
    classify_frames_by_capture_direction(frames, camera_rotations, camera_centers, R0, C0, Zs)
    err, camera_depths = compute_feature_track_reprojection_errors(
        triangulate_3d_feature_tracks(obs_frame[good], obs_uv[good], obs_track[good], ntracks, camera_rotations, camera_centers, rig_model),
        obs_frame[good],
        obs_uv[good],
        obs_track[good],
        camera_rotations,
        camera_centers,
        rig_model,
    )
    print(
        f"      re-split soft valley (metric fix): median reproj "
        f"{med0:.2f} -> {float(np.median(err)):.2f}px"
    )
    # final
    landmark_points = triangulate_3d_feature_tracks(obs_frame[good], obs_uv[good], obs_track[good], ntracks, camera_rotations, camera_centers, rig_model)
    cnt = np.bincount(obs_track[good], minlength=ntracks)
    err, zc = compute_feature_track_reprojection_errors(landmark_points, obs_frame, obs_uv, obs_track, camera_rotations, camera_centers, rig_model)
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
    landmark_points = landmark_points[keep_tracks]
    track_err = np.zeros(len(keep_tracks))
    np.maximum.at(track_err, obs_track, err)
    rms = float(np.sqrt(np.mean(err**2)))
    print(
        f"      final: {len(landmark_points)} landmarks, {len(obs_uv)} obs, rms {rms:.2f}px "
        f"(median {np.median(err):.2f}px)"
    )
    return BundleAdjustmentResult(
        X=landmark_points,
        track_err=track_err,
        obs_frame=obs_frame,
        obs_uv=obs_uv,
        obs_track=obs_track,
        err=err,
        R=camera_rotations,
        C=camera_centers,
        rms=rms,
        history=history,
    )


def bundle_adjust_grouped_camera_poses_and_feature_tracks(
    frames: list[Frame], pairs: list[Pair], rig_model: RigModel,
    R0: RotationArray, C0: CenterArray, args: RegistrationArgs,
    group_of: dict[int, int],
) -> BundleAdjustmentResult:
    """Breathing-robust registration. The skin deforms between scan passes, so a
    single rigid bundle can't explain frames captured minutes apart -- it puts
    cross-pass features at compromise 3D positions, which then ghost in the
    texture. Instead we register each breathing-coherent GROUP (a scan row, whose
    frames are seconds apart) rigidly on its OWN tracks, then align the groups to
    each other with a per-group 3D translation solved from the cross-group
    feature matches (the deformable step). Within a group geometry is rigid and
    single; between groups, breathing becomes a small shift the alignment removes
    and the group-owned texture compositing renders once instead of doubling.

    group_of: dict frame.idx -> group id. Returns the bundle_adjust_camera_poses_and_feature_tracks dict shape."""
    gids = sorted(set(group_of.values()))
    intra = {gauge: [] for gauge in gids}
    cross = []
    for point in pairs:
        ga, gb = group_of[point.index], group_of[point.neighbor_index]
        if ga == gb:
            intra[ga].append(point)
        else:
            cross.append(point)

    camera_rotations = R0.copy()
    camera_centers = C0.copy()
    Xparts, ofr, ouv, otr, terr = [], [], [], [], []
    pt2track = {}  # (frame, x_round, y_round) -> global track
    grp_of_track = []  # group id per global track
    tbase = 0

    def key(fi: int, pt: FloatArray) -> tuple[int, float, float]:
        return (int(fi), round(float(pt[0]), 2), round(float(pt[1]), 2))

    for gauge in gids:
        ng = sum(1 for index in group_of if group_of[index] == gauge)
        print(f"      -- group {gauge}: {ng} frames, {len(intra[gauge])} intra-pairs")
        bundle_adjustment_result = bundle_adjust_camera_poses_and_feature_tracks(frames, intra[gauge], rig_model, R0, C0, args)
        for index in group_of:
            if group_of[index] == gauge:
                camera_rotations[index] = bundle_adjustment_result["R"][index]
                camera_centers[index] = bundle_adjustment_result["C"][index]
        tt = bundle_adjustment_result["obs_track"].astype(np.int64) + tbase
        for fi, pt, threshold in zip(bundle_adjustment_result["obs_frame"], bundle_adjustment_result["obs_uv"], tt):
            pt2track[key(fi, pt)] = int(threshold)
        Xparts.append(bundle_adjustment_result["X"])
        ofr.append(bundle_adjustment_result["obs_frame"])
        ouv.append(bundle_adjustment_result["obs_uv"])
        otr.append(tt)
        terr.append(bundle_adjustment_result["track_err"])
        grp_of_track.extend([gauge] * len(bundle_adjustment_result["X"]))
        tbase += len(bundle_adjustment_result["X"])

    Xg = np.concatenate(Xparts) if Xparts else np.zeros((0, 3))
    grp_of_track = np.asarray(grp_of_track, dtype=np.int64)

    # cross-group landmark correspondences (dedup by track pair)
    corr: set[tuple[int, int]] = set()
    for point in cross:
        for first_value, second_value in zip(point.src_kp, point.dst_kp):
            ta = pt2track.get(key(point.index, frames[point.index].kp[int(first_value)].pt))
            tb = pt2track.get(key(point.neighbor_index, frames[point.neighbor_index].kp[int(second_value)].pt))
            if ta is not None and tb is not None and grp_of_track[ta] != grp_of_track[tb]:
                corr.add((ta, tb))

    gidx = {gauge: item_index for item_index, gauge in enumerate(gids)}
    t_solved = np.zeros((len(gids), 3))
    if args.group_align != "none" and corr:
        # per-group translation t_g minimising  sum |(Xa+t_a)-(Xb+t_b)|^2
        #                                        + lam sum |t_g|^2  (proprioception anchor / gauge)
        matrix = np.zeros((len(gids), len(gids)))
        rhs = np.zeros((len(gids), 3))
        for ta, tb in corr:
            ga, gb = gidx[grp_of_track[ta]], gidx[grp_of_track[tb]]
            distance = Xg[ta] - Xg[tb]
            matrix[ga, ga] += 1
            matrix[gb, gb] += 1
            matrix[ga, gb] -= 1
            matrix[gb, ga] -= 1
            rhs[ga] -= distance
            rhs[gb] += distance
        lam = 1.0
        t_solved = np.linalg.solve(matrix + lam * np.eye(len(gids)), rhs)
        # The dot ghosting is an IN-PLANE displacement (a feature's ortho-texture
        # position depends on its x,y, not its depth). The along-optical-axis
        # component, by contrast, is the fx/depth-degenerate direction where each
        # small group floats unreliably -- fitting it drags weakly-constrained end
        # rows many mm. So keep only the tangential correction and leave depth to
        # proprioception (which the per-group BA already pins via the standoff).
        nhat = np.array([camera_rotations[index][:, 2] for index in group_of]).mean(0)
        nhat /= np.linalg.norm(nhat)
        t_solved -= (t_solved @ nhat)[:, None] * nhat
        for gauge in gids:
            for index in group_of:
                if group_of[index] == gauge:
                    camera_centers[index] = camera_centers[index] + t_solved[gidx[gauge]]
            Xg[grp_of_track == gauge] += t_solved[gidx[gauge]]
        mags = np.linalg.norm(t_solved, axis=1)
        print(
            f"      group alignment ({len(corr)} cross-corr): "
            f"|t| per group " + " ".join(f"{gauge}:{mags[gidx[gauge]]:.1f}" for gauge in gids) + " mm"
        )
    elif args.group_align != "none":
        print("      ! no cross-group correspondences -- groups left at proprioception")

    obs_frame = np.concatenate(ofr).astype(np.int32)
    obs_uv = np.concatenate(ouv)
    obs_track = np.concatenate(otr).astype(np.int32)
    track_err = np.concatenate(terr)
    err, camera_depths = compute_feature_track_reprojection_errors(Xg, obs_frame, obs_uv, obs_track, camera_rotations, camera_centers, rig_model)
    rms = float(np.sqrt(np.mean(err**2)))
    print(
        f"      grouped final: {len(gids)} groups, {len(Xg)} landmarks, "
        f"{len(obs_uv)} obs, rms {rms:.2f}px (median {np.median(err):.2f}px)"
    )
    return BundleAdjustmentResult(
        X=Xg,
        track_err=track_err,
        obs_frame=obs_frame,
        obs_uv=obs_uv,
        obs_track=obs_track,
        err=err,
        R=camera_rotations,
        C=camera_centers,
        rms=rms,
        history=[],
    )
