"""Photometric gain fitting and CPU/GPU ortho-texture rendering."""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any, Protocol, TypedDict, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .registration_features import Frame
from .registration_features import _get_available_system_memory_bytes
from .registration_geometry import RigModel, project_world_points_into_camera
from .registration_surface import Surface, TexParam, compute_camera_frame_surface_footprint

FloatArray = NDArray[np.float32]
NumericArray = NDArray[Any]
Bounds = tuple[float, float, float, float]
Rect = tuple[int, int, int, int]
Warp = tuple[NumericArray, NumericArray, float, float] | None
FrameGroups = Mapping[int, int] | None
Geometry = (
    tuple[Rect, NumericArray]
    | tuple[Rect, NumericArray, NumericArray, NumericArray, NumericArray]
)


class GpuTileRect(TypedDict):
    rect: Rect


class GpuTiles(GpuTileRect, total=False):
    """CPU arrays returned by the optional GPU renderer."""

    accumulated: bool
    col: FloatArray
    wgt2: FloatArray
    lo: FloatArray
    soft: FloatArray
    hf: FloatArray
    wown: FloatArray


class GpuSoftBlendTiles(TypedDict):
    rect: Rect
    col: FloatArray
    wgt2: FloatArray


class GpuTwoBandTiles(TypedDict):
    rect: Rect
    lo: FloatArray
    soft: FloatArray
    hf: FloatArray
    wown: FloatArray


class RenderState(Protocol):
    frames: Sequence[Frame]
    R: NumericArray
    C: NumericArray
    rig_model: RigModel
    surf: Surface
    texture_parameters: TexParam
    tp: TexParam
    pixels_per_mm: float
    ppmm: float
    up_sign: float
    blend_sharpness: float
    blend_mode: str
    max_incidence_deg: float
    warp: Any
    foot_uv: dict[int, NumericArray]
    bounds: Bounds
    texture_width: int
    texture_height: int
    W: int
    H: int
    umin: float
    vmin: float
    umax: float
    vmax: float
    gpu: Any
    gpu_oom: type[BaseException]
    gpu_canvas: bool
    gpu_geom: Any
    # Canvas arrays are allocated either here or by the GPU and copied back
    # before the CPU accumulation path uses them.
    acc: FloatArray
    wacc: FloatArray
    hf_best: FloatArray
    w_best: FloatArray
    fxf: float
    k1: float
    cxf: float
    cyf: float
    Wf: float
    Hf: float
    img_pxmm: float
    sscale: float
    geo_q: int


# ----------------------------------------------------------------------------
# ortho-texture rendering
# ----------------------------------------------------------------------------
def estimate_camera_frame_texture_gains(
    frames: Sequence[Frame],
    rig_model: RigModel,
    obs_frame: NDArray[np.integer[Any]],
    obs_uv: NumericArray,
    obs_track: NDArray[np.integer[Any]],
    err: NumericArray | None = None,
    mode: str = "on",
) -> FloatArray | None:
    """Per-frame BGR gain fitted from BA track colors (photometric seams).

    The lamp travels with the camera, so the same skin patch renders up to
    ~40 gray levels apart across frames (lamp angle follows rx/z, plus
    vignetting); the two-band LF cross-fade turns those offsets into blocky
    tone steps at feather and group-gate boundaries. Every BA track is one
    skin point observed from several frames, so per-frame log-gains fall out
    of an alternating least squares over the tracks' low-pass colors -- no
    extra geometry or canvas pass.

    mode 'on':    one scalar per channel per frame (frame-level lamp /
                  exposure offset). Returns (F,3) float32 BGR multipliers,
                  geometric mean 1 per channel (overall exposure preserved).
    mode 'field': an affine log-gain field per frame g(x^,y^) = exp(frame_center_gains +
                  gx*x^ + gy*y^) in normalised image coords (x^,y^ in
                  [-0.5,0.5]) -- also corrects the within-frame shading /
                  vignetting gradient that a scalar cannot. Returns (F,3,3)
                  float32 log-coeffs [channel, (g0,gx,gy)]; the render
                  exponentiates per texel.
    """
    reduced_decode_factor = 8  # IMREAD_REDUCED_COLOR_8 decode factor
    reprojection_valid_mask = np.ones(len(obs_frame), bool) if err is None else (err < 10.0)
    mean_camera_depth = float(np.mean([rig_model.depth(frame) for frame in frames]))
    # sample at the LF band's own scale (~2mm) so the gains describe exactly
    # the band they will correct
    blur_sigma_pixels = max(2.0, 2.0 * (rig_model.fx * rig_model.downscale / mean_camera_depth) / reduced_decode_factor)
    frame_count = len(frames)
    observed_bgr_samples = np.zeros((len(obs_frame), 3), np.float32)
    valid_sample_mask = np.zeros(len(obs_frame), bool)
    for frame in frames:
        frame_observation_indices = np.where(
            (obs_frame == frame.idx) & reprojection_valid_mask
        )[0]
        if not len(frame_observation_indices):
            continue
        im = cv2.imread(frame.image_path, cv2.IMREAD_REDUCED_COLOR_8 | cv2.IMREAD_IGNORE_ORIENTATION)
        if im is None:
            continue
        im = cv2.GaussianBlur(im.astype(np.float32), (0, 0), blur_sigma_pixels)
        reduced_observation_pixels = obs_uv[frame_observation_indices] * (
            rig_model.downscale / reduced_decode_factor
        )
        pixel_x = np.clip(np.round(reduced_observation_pixels[:, 0]).astype(int), 0, im.shape[1] - 1)
        pixel_y = np.clip(np.round(reduced_observation_pixels[:, 1]).astype(int), 0, im.shape[0] - 1)
        sampled_bgr_values = im[pixel_y, pixel_x]
        valid_color_mask = (sampled_bgr_values.min(1) > 5.0) & (sampled_bgr_values.max(1) < 250.0)  # not clipped
        observed_bgr_samples[frame_observation_indices[valid_color_mask]] = sampled_bgr_values[valid_color_mask]
        valid_sample_mask[frame_observation_indices[valid_color_mask]] = True
    observed_frame_indices = obs_frame[valid_sample_mask]
    observed_track_indices = obs_track[valid_sample_mask]
    if len(observed_frame_indices) < 10 * frame_count:
        print(f"      ! LF gain: only {len(observed_frame_indices)} usable track colors -- skipping compensation")
        return None
    log_samples = np.log(observed_bgr_samples[valid_sample_mask].clip(1.0, None))  # (M,3)
    # normalised image coords of each obs (downscale cancels: ds px / ds dims)
    normalized_image_x = obs_uv[valid_sample_mask, 0] / (2.0 * rig_model.cx) - 0.5
    normalized_image_y = obs_uv[valid_sample_mask, 1] / (2.0 * rig_model.cy) - 0.5
    image_gain_basis = np.stack([np.ones_like(normalized_image_x), normalized_image_x, normalized_image_y], 1)  # (M, 3) field basis
    basis_coefficient_count = 1 if mode == "on" else 3
    track_count = int(observed_track_indices.max()) + 1
    # model: log_samples + G_f(x^,y^) ~= mu_t, G = log gain the render will apply
    log_gain_coefficients = np.zeros((frame_count, 3, basis_coefficient_count))
    robust_observation_mask = np.ones(len(log_samples), bool)
    frame_observation_counts: NumericArray = np.zeros(frame_count, dtype=np.float64)
    for fit_iteration in range(16):
        log_gain_correction = np.einsum("mcb,mb->mc", log_gain_coefficients[observed_frame_indices], image_gain_basis[:, :basis_coefficient_count])
        track_mean_log_colors = np.zeros((track_count, 3))
        track_observation_counts = np.zeros(track_count)
        np.add.at(track_mean_log_colors, observed_track_indices[robust_observation_mask], (log_samples + log_gain_correction)[robust_observation_mask])
        np.add.at(track_observation_counts, observed_track_indices[robust_observation_mask], 1)
        track_mean_log_colors /= np.maximum(track_observation_counts, 1)[:, None]
        target_log_gains = track_mean_log_colors[observed_track_indices] - log_samples  # per-observation target log-gain
        frame_observation_counts: NumericArray = np.zeros(frame_count, dtype=np.float64)
        np.add.at(frame_observation_counts, observed_frame_indices[robust_observation_mask], 1)
        if basis_coefficient_count == 1:
            frame_log_gain_sums = np.zeros((frame_count, 3))
            np.add.at(frame_log_gain_sums, observed_frame_indices[robust_observation_mask], target_log_gains[robust_observation_mask])
            log_gain_coefficients[:, :, 0] = frame_log_gain_sums / np.maximum(frame_observation_counts, 1)[:, None]
        else:
            for frame_index in range(frame_count):
                frame_observation_mask = robust_observation_mask & (observed_frame_indices == frame_index)
                frame_observation_count = int(frame_observation_mask.sum())
                if frame_observation_count < 12:
                    continue
                frame_basis = image_gain_basis[frame_observation_mask]
                # ridge on the gradient terms only: a weak prior that wins
                # when a frame's obs are clustered, negligible otherwise
                ridge_strength = max(20.0, 0.02 * frame_observation_count)
                normal_matrix = frame_basis.T @ frame_basis + np.diag(
                    [1e-9, ridge_strength, ridge_strength]
                )
                log_gain_coefficients[frame_index] = np.linalg.solve(
                    normal_matrix, frame_basis.T @ target_log_gains[frame_observation_mask]
                ).T
        if fit_iteration == 7:  # one robust trim
            log_gain_residuals = (
                log_samples + np.einsum("mcb,mb->mc", log_gain_coefficients[observed_frame_indices], image_gain_basis[:, :basis_coefficient_count]) - track_mean_log_colors[observed_track_indices]
            )
            robust_observation_mask &= np.abs(log_gain_residuals).max(1) < 2.5 * (log_gain_residuals.std() + 1e-6)
    log_gain_coefficients[:, :, 0] -= log_gain_coefficients[:, :, 0].mean(0, keepdims=True)  # geometric mean 1
    if mode == "on":
        gains = np.exp(log_gain_coefficients[:, :, 0]).astype(np.float32)
        print(
            f"      LF gain compensation: {int(np.count_nonzero(frame_observation_counts > 0))}/{frame_count} frames "
            f"from {int(robust_observation_mask.sum())} track colors, gain range "
            f"{gains.min():.3f}..{gains.max():.3f}"
        )
        return gains
    frame_center_gains = np.exp(log_gain_coefficients[:, :, 0])
    max_gradient_magnitude = float(np.abs(log_gain_coefficients[:, :, 1:]).max())
    print(
        f"      LF gain field: {int(np.count_nonzero(frame_observation_counts > 0))}/{frame_count} frames from "
        f"{int(robust_observation_mask.sum())} track colors, centre gain {frame_center_gains.min():.3f}.."
        f"{frame_center_gains.max():.3f}, max |gradient| {max_gradient_magnitude:.3f}/half-image"
    )
    return log_gain_coefficients.astype(np.float32)


def compute_texture_bounds(
    frames: Sequence[Frame],
    camera_rotations: NumericArray,
    camera_centers: NumericArray,
    rig_model: RigModel,
    surf: Surface,
    texture_parameters: TexParam,
    pixels_per_mm: float,
) -> tuple[dict[int, NumericArray], Bounds, int, int]:
    """Return frame footprints and the landmark-supported texture canvas."""
    foot_uv: dict[int, NumericArray] = {}
    umin = vmin = np.inf
    umax = vmax = -np.inf
    for descriptive_frame in frames:
        descriptive_points = compute_camera_frame_surface_footprint(descriptive_frame, camera_rotations, camera_centers, rig_model, surf)
        texture_u, texture_v = texture_parameters.to_uv(descriptive_points[:, 0], descriptive_points[:, 1])
        foot_uv[descriptive_frame.idx] = np.stack([texture_u, texture_v], 1)
        umin, umax = min(umin, texture_u.min()), max(umax, texture_u.max())
        vmin, vmax = min(vmin, texture_v.min()), max(vmax, texture_v.max())
    pad = 1.0
    umin -= pad
    vmin -= pad
    umax += pad
    vmax += pad
    # clamp to the landmark-supported surface region (no textured extrapolation)
    sy_i, sx_i = np.where(surf.support)
    if len(sx_i):
        su, sv = texture_parameters.to_uv(surf.xs[sx_i], surf.ys[sy_i])
        umin = max(umin, su.min() - pad)
        umax = min(umax, su.max() + pad)
        vmin = max(vmin, sv.min() - pad)
        vmax = min(vmax, sv.max() + pad)
    texture_width = int(math.ceil((umax - umin) * pixels_per_mm))
    texture_height = int(math.ceil((vmax - vmin) * pixels_per_mm))
    print(
        f"      texture canvas {texture_width}x{texture_height} @ {pixels_per_mm}px/mm "
        f"(u: {umin:.0f}..{umax:.0f}, v: {vmin:.0f}..{vmax:.0f} mm)"
    )
    return foot_uv, (umin, vmin, umax, vmax), texture_width, texture_height


def prepare_texture_render_state(
    frames: Sequence[Frame],
    camera_rotations: NumericArray,
    camera_centers: NumericArray,
    rig_model: RigModel,
    surf: Surface,
    texture_parameters: TexParam,
    pixels_per_mm: float,
    up_sign: float,
    blend_sharpness: float,
    blend_mode: str,
    max_incidence_deg: float,
    warp: Warp,
    device: str,
    foot_uv: dict[int, NumericArray],
    bounds: Bounds,
    texture_width: int,
    texture_height: int,
) -> RenderState:
    """Allocate canvases and initialize the optional GPU geometry pipeline."""
    umin, vmin, umax, vmax = bounds
    # GPU fast path: the per-frame image pipeline (decode/remap/blurs) runs on
    # CUDA and returns finished tiles; the canvases stay in system RAM and the
    # accumulation below is shared with the CPU path (see render_gpu.py).
    gpu = None
    gpu_oom = RuntimeError
    if device != "cpu":
        from processing.render_gpu import (
            GpuFramePipe,
            GpuOom,
            texture_gpu_renderer_available,
        )

        gpu_oom = GpuOom
        if texture_gpu_renderer_available():
            gpu = GpuFramePipe(blend_mode)
            if os.environ.get("SKINMAP_NO_VRAM_CANVAS"):
                print("      render device: cuda (torch), canvases in RAM (SKINMAP_NO_VRAM_CANVAS)")
            elif gpu.canvas_fits(texture_height, texture_width):
                gpu.canvases_begin(texture_height, texture_width)
                print("      render device: cuda (torch), canvases in VRAM")
            else:
                print(
                    "      render device: cuda (torch), canvases in RAM "
                    "(canvas exceeds VRAM budget)"
                )
        elif device != "auto":
            print(f"      ! --device {device} unavailable (need torch+CUDA); falling back to CPU")
    gpu_canvas = gpu is not None and gpu.canvas
    # RAM budget check BEFORE allocating canvas-sized arrays: a degenerate
    # registration can inflate the arc-length canvas enormously; dying to the
    # kernel OOM killer takes innocent processes with it, so abort cleanly
    per_texel = 32 if blend_mode == "two-band" else 16
    need = float(texture_height) * texture_width * per_texel
    avail_ram = _get_available_system_memory_bytes()
    if need > 0.6 * avail_ram:
        sys.exit(
            f"      ! texture canvases for {texture_width}x{texture_height} need {need / 1e9:.1f} GB "
            f"but only {avail_ram / 1e9:.1f} GB RAM is available -- "
            "aborting before the OOM killer does it for us. A canvas "
            "this large usually means a degenerate registration "
            "(inflated arc length); otherwise lower --texture-ppmm."
        )
    acc = wacc = hf_best = w_best = None
    if not gpu_canvas:
        acc = np.zeros((texture_height, texture_width, 3), np.float32)
        wacc = np.zeros((texture_height, texture_width), np.float32)
    if blend_mode == "two-band" and not gpu_canvas:
        # high frequencies are never averaged: per texel the argmax-weight
        # frame contributes them alone; only the low-pass (illumination) band
        # is soft-blended, which hides photometric seams without losing detail
        hf_best = np.zeros((texture_height, texture_width, 3), np.float32)
        w_best = np.zeros((texture_height, texture_width), np.float32)

    fxf = rig_model.fx * rig_model.downscale
    k1 = rig_model.k1
    cxf, cyf = rig_model.cx * rig_model.downscale, rig_model.cy * rig_model.downscale
    Wf, Hf = cxf * 2, cyf * 2
    img_pxmm = fxf / np.mean([rig_model.depth(descriptive_frame) for descriptive_frame in frames])
    sscale = min(1.0, 1.4 * pixels_per_mm / img_pxmm)

    # geometry decimation: the per-texel fields geom() computes (projected
    # image coords, spline height, weights) are all smooth at the surface-grid
    # scale (2mm pitch), while scipy spline .ev is ~us/point single-threaded --
    # at 78px/mm a frame tile is ~10M texels and the splines dominate the whole
    # render (minutes/frame). Evaluate on a ~20px/mm grid (q texels) and
    # bilinearly upsample: placement error << the solve rms, ~16x faster.
    # q=1 at <=20px/mm uses the exact per-texel path.
    geo_q = max(1, int(round(pixels_per_mm / 20.0)))

    # geometry fields on CUDA: same math as geom() below (float64), evaluated
    # on the GPU and never leaving it -- geom dominated the GPU render's wall
    # clock (~3 s/frame of spline/projection numpy per pass)
    gpu_geom = None
    if gpu is not None:
        from processing.render_gpu import GpuGeom

        gpu_geom = GpuGeom(
            surf,
            texture_parameters,
            camera_rotations,
            camera_centers,
            fxf,
            k1,
            cxf,
            cyf,
            Wf,
            Hf,
            up_sign,
            blend_sharpness,
            max_incidence_deg,
            warp,
        )
    return cast(RenderState, SimpleNamespace(
        frames=frames,
        R=camera_rotations,
        C=camera_centers,
        rig_model=rig_model,
        surf=surf,
        texture_parameters=texture_parameters,
        pixels_per_mm=pixels_per_mm,
        up_sign=up_sign,
        blend_sharpness=blend_sharpness,
        blend_mode=blend_mode,
        max_incidence_deg=max_incidence_deg,
        warp=warp,
        foot_uv=foot_uv,
        bounds=bounds,
        texture_width=texture_width,
        texture_height=texture_height,
        umin=umin,
        vmin=vmin,
        umax=umax,
        vmax=vmax,
        gpu=gpu,
        gpu_oom=gpu_oom,
        gpu_canvas=gpu_canvas,
        gpu_geom=gpu_geom,
        acc=acc,
        wacc=wacc,
        hf_best=hf_best,
        w_best=w_best,
        fxf=fxf,
        k1=k1,
        cxf=cxf,
        cyf=cyf,
        Wf=Wf,
        Hf=Hf,
        img_pxmm=img_pxmm,
        sscale=sscale,
        geo_q=geo_q,
        # Short aliases used throughout the projection and accumulation stages.
        tp=texture_parameters,
        ppmm=pixels_per_mm,
        W=texture_width,
        H=texture_height,
    ))


def compute_camera_frame_surface_projection(
    state: RenderState, frame: Frame, upsample: bool = True
) -> Any:
    """Project one frame and calculate its geometric blend weights."""
    fp = state.foot_uv[frame.idx]
    umin, vmin, pixels_per_mm = state.umin, state.vmin, state.ppmm
    texture_width, texture_height, geometry_query_stride = state.W, state.H, state.geo_q
    u0 = max(0, int((fp[:, 0].min() - umin - 1) * pixels_per_mm))
    u1 = min(texture_width, int((fp[:, 0].max() - umin + 1) * pixels_per_mm) + 1)
    v0 = max(0, int((fp[:, 1].min() - vmin - 1) * pixels_per_mm))
    v1 = min(texture_height, int((fp[:, 1].max() - vmin + 1) * pixels_per_mm) + 1)
    if u1 <= u0 or v1 <= v0:
        return None
    if geometry_query_stride > 1:
        wd = max(2, -(-(u1 - u0) // geometry_query_stride))
        hd = max(2, -(-(v1 - v0) // geometry_query_stride))
        uu = umin + (u0 + (np.arange(wd) + 0.5) * geometry_query_stride) / pixels_per_mm
        vv = vmin + (v0 + (np.arange(hd) + 0.5) * geometry_query_stride) / pixels_per_mm
    else:
        wd, hd = u1 - u0, v1 - v0
        uu = umin + (np.arange(u0, u1) + 0.5) / pixels_per_mm
        vv = vmin + (np.arange(v0, v1) + 0.5) / pixels_per_mm
    if not upsample and state.gpu_geom is not None:
        try:
            dec = state.gpu_geom.fields(frame.idx, uu, vv)
            if dec is None:
                return None
            return (u0, u1, v0, v1), dec
        except Exception as exc:  # GpuOom etc: numpy fallback
            print(
                f"        ! gpu geom fell back to numpy for frame "
                f"{frame.idx} ({type(exc).__name__})"
            )
    Ug, Vg = np.meshgrid(uu, vv)
    Uw, Vw = Ug.ravel(), Vg.ravel()
    if state.warp is not None:
        Mu, Mv, umid, vmid = state.warp
        du, dv = Uw - umid, Vw - vmid
        horizontal_warp_coefficients = Mu[frame.idx]
        vertical_warp_coefficients = Mv[frame.idx]
        horizontal_warp = (
            horizontal_warp_coefficients[0] * du
            + horizontal_warp_coefficients[1] * dv
            + horizontal_warp_coefficients[2]
        )
        vertical_warp = (
            vertical_warp_coefficients[0] * du
            + vertical_warp_coefficients[1] * dv
            + vertical_warp_coefficients[2]
        )
        if len(horizontal_warp_coefficients) == 6:
            q0, q1, q2 = du * du / 50.0, du * dv / 50.0, dv * dv / 50.0
            horizontal_warp += (
                horizontal_warp_coefficients[3] * q0
                + horizontal_warp_coefficients[4] * q1
                + horizontal_warp_coefficients[5] * q2
            )
            vertical_warp += (
                vertical_warp_coefficients[3] * q0
                + vertical_warp_coefficients[4] * q1
                + vertical_warp_coefficients[5] * q2
            )
        Uw, Vw = Uw - horizontal_warp, Vw - vertical_warp
    Xg, Yg = state.tp.to_xy(
        np.asarray(Uw, dtype=np.float64), np.asarray(Vw, dtype=np.float64)
    )
    Zg = state.surf.height(Xg, Yg)
    descriptive_points = np.stack([Xg, Yg, Zg], 1)
    uv, zc = project_world_points_into_camera(
        descriptive_points,
        state.R[frame.idx],
        state.C[frame.idx],
        state.fxf,
        state.k1,
        state.cxf,
        state.cyf,
    )
    valid = (
        (zc > 10)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < state.Wf - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < state.Hf - 1)
    )
    if not valid.any():
        return None
    uv[:, 0] = np.clip(uv[:, 0], 0, state.Wf - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, state.Hf - 1)
    feather = np.clip(
        np.minimum.reduce(
            [
                uv[:, 0],
                state.Wf - 1 - uv[:, 0],
                uv[:, 1],
                state.Hf - 1 - uv[:, 1],
            ]
        )
        / (0.10 * min(state.Wf, state.Hf)),
        0,
        1,
    )
    r2c = ((uv[:, 0] - state.cxf) / state.cxf) ** 2 + ((uv[:, 1] - state.cyf) / state.cyf) ** 2
    center = np.maximum(
        np.exp(-state.blend_sharpness * 0.5 * r2c),
        0.02,
    )
    vs = valid * state.surf.supported(descriptive_points[:, 0], descriptive_points[:, 1])
    if state.max_incidence_deg > 0:
        nrm = state.surf.normal(descriptive_points[:, 0], descriptive_points[:, 1], state.up_sign)
        ray = descriptive_points - state.C[frame.idx][None, :]
        ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-9)
        cosi = np.abs(np.sum(nrm * ray, axis=1))
        c_hi = math.cos(math.radians(max(state.max_incidence_deg - 15.0, 1.0)))
        c_lo = math.cos(math.radians(state.max_incidence_deg))
        vs *= np.clip((cosi - c_lo) / max(c_hi - c_lo, 1e-6), 0.0, 1.0)
    wgt2d = (feather * center * vs).astype(np.float32).reshape(hd, wd)
    soft2d = (feather * vs).astype(np.float32).reshape(hd, wd)
    mapx = uv[:, 0].astype(np.float32).reshape(hd, wd)
    mapy = uv[:, 1].astype(np.float32).reshape(hd, wd)
    if not upsample:
        return (u0, u1, v0, v1), np.stack([mapx, mapy, wgt2d, soft2d])
    if geometry_query_stride > 1:
        size = (u1 - u0, v1 - v0)
        mapx = cv2.resize(mapx, size, interpolation=cv2.INTER_LINEAR)
        mapy = cv2.resize(mapy, size, interpolation=cv2.INTER_LINEAR)
        wgt2d = cv2.resize(wgt2d, size, interpolation=cv2.INTER_LINEAR)
        soft2d = cv2.resize(soft2d, size, interpolation=cv2.INTER_LINEAR)
    return (u0, u1, v0, v1), mapx, mapy, wgt2d, soft2d


def build_texture_ownership_by_frame_group(
    state: RenderState, frame_group: FrameGroups
) -> tuple[NumericArray | None, dict[int, Geometry], list[int]]:
    """Compute one owning capture group per texture texel."""
    if frame_group is None:
        return None, {}, []
    frames, gpu = state.frames, state.gpu
    texture_height, texture_width = state.H, state.W
    started = time.time()
    gids = sorted(set(frame_group.values()))
    geo_cache: dict[int, Geometry] = {}
    geo_bytes, geo_budget = 0, 0.25 * _get_available_system_memory_bytes()
    gpu_own = False
    if gpu is not None:
        try:
            gpu.ownership_begin(texture_height, texture_width)
            gpu_own = True
        except Exception as exc:
            print(f"      ! ownership fields don't fit VRAM ({exc}); streaming ownership on CPU")
    ordered = sorted(frames, key=lambda frame: frame_group[frame.idx])
    if gpu_own:
        assert gpu is not None
        for frame in ordered:
            result = compute_camera_frame_surface_projection(state, frame, upsample=False)
            if result is None:
                continue
            if state.gpu_geom is None and geo_bytes < geo_budget:
                geo_cache[frame.idx] = result
                geo_bytes += result[1].nbytes
            rect, decimated = cast(tuple[Rect, NumericArray], result)
            gpu.ownership_add(frame_group[frame.idx], rect, decimated[2])
        best_group = gpu.ownership_finish()
    else:
        own_max = np.zeros((texture_height, texture_width), np.float32)
        own_arg = np.full((texture_height, texture_width), -1, np.int32)
        own_sum = np.zeros((texture_height, texture_width), np.float32)
        current_group = None
        for frame in ordered:
            result = compute_camera_frame_surface_projection(state, frame)
            if result is None:
                continue
            (u0, u1, v0, v1), low_frequency_texture, high_frequency_texture, blend_weight, coverage_mask = cast(
                tuple[Rect, NumericArray, NumericArray, NumericArray, NumericArray], result
            )
            group = frame_group[frame.idx]
            if current_group is not None and group != current_group:
                wins = own_sum > own_max
                own_max[wins] = own_sum[wins]
                own_arg[wins] = current_group
                own_sum[...] = 0.0
            current_group = group
            own_sum[v0:v1, u0:u1] += blend_weight
        if current_group is not None:
            wins = own_sum > own_max
            own_arg[wins] = current_group
        best_group = own_arg
    print(f"      group-owned compositing across {len(gids)} groups ({time.time() - started:.0f}s)")
    return best_group, geo_cache, gids


def compute_low_frequency_group_blend_gates(
    best_group: NumericArray | None,
    gids: Sequence[int],
    group_feather_mm: float,
    pixels_per_mm: float,
    texture_width: int,
    texture_height: int,
) -> dict[int, NDArray[np.uint8]] | None:
    """Build feathered, partition-of-unity gates for low-frequency blending."""
    if best_group is None or group_feather_mm <= 0:
        return None
    sigma = group_feather_mm * pixels_per_mm
    reduction = max(1, int(round(sigma / 12.0)))
    small_w, small_h = max(1, texture_width // reduction), max(1, texture_height // reduction)
    gates: list[FloatArray] = []
    for group in gids:
        mask = (best_group == group).astype(np.float32)
        if reduction > 1:
            mask = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gates.append(cv2.GaussianBlur(mask, (0, 0), sigma / reduction))
    gate_sum = np.maximum(np.sum(gates, 0), 1e-6)
    result: dict[int, NDArray[np.uint8]] = {}
    for group, mask in zip(gids, gates):
        mask /= gate_sum
        if reduction > 1:
            mask = cv2.resize(mask, (texture_width, texture_height), interpolation=cv2.INTER_LINEAR)
        result[group] = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    print(f"      LF group gates feathered at {group_feather_mm:.0f}mm")
    return result


def log_texture_render_profile_tick(profile: dict[str, float], key: str, started: float) -> float:
    profile[key] = profile.get(key, 0.0) + time.perf_counter() - started
    return time.perf_counter()


def accumulate_camera_frame_into_cpu_texture(
    state: RenderState,
    frame: Frame,
    geometry: tuple[Rect, NumericArray, NumericArray, NumericArray, NumericArray],
    frame_group: FrameGroups,
    frame_gain: NumericArray | None,
    best_group: NumericArray | None,
    lf_gate: Mapping[int, NDArray[np.uint8]] | None,
    focus_weight: float,
    hf_cross_group: bool,
    hf_coherence_mm: float,
) -> None:
    """Decode and deposit one frame through the CPU rendering path."""
    (u0, u1, v0, v1), mapx, mapy, weight, soft = geometry
    image = cv2.imread(
        frame.image_path,
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    if image is None:
        return
    if state.sscale < 0.999:
        image = cv2.resize(
            image,
            None,
            fx=state.sscale,
            fy=state.sscale,
            interpolation=cv2.INTER_AREA,
        )
        mapx *= state.sscale
        mapy *= state.sscale
    color = cv2.remap(
        image,
        mapx,
        mapy,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(np.float32)
    if frame_gain is not None:
        gain = frame_gain[frame.idx]
        if gain.ndim == 1:
            color *= gain
        else:
            gx = mapx / image.shape[1] - 0.5
            gy = mapy / image.shape[0] - 0.5
            color *= np.exp(
                gain[:, 0] + gx[..., None] * gain[:, 1] + gy[..., None] * gain[:, 2]
            ).astype(np.float32)
    if best_group is not None and frame_group is not None and not hf_cross_group:
        weight *= best_group[v0:v1, u0:u1] == frame_group[frame.idx]
    if focus_weight > 0:
        pxmm_image = state.img_pxmm * state.sscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        high = gray - cv2.GaussianBlur(gray, (0, 0), max(1.0, 0.06 * pxmm_image))
        energy = cv2.GaussianBlur(high * high, (0, 0), max(3.0, 1.5 * pxmm_image))
        sharpness = cv2.remap(
            np.sqrt(np.maximum(energy, 0)) + 1e-3,
            mapx,
            mapy,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        weight *= sharpness**focus_weight
    if state.blend_mode != "two-band":
        state.acc[v0:v1, u0:u1] += color * weight[..., None]
        state.wacc[v0:v1, u0:u1] += weight
        return
    mask = (soft > 0).astype(np.float32)
    sigma = 2.0 * state.ppmm
    reduction = max(1, int(round(sigma / 12.0)))
    if reduction > 1:
        small_w = max(1, (u1 - u0) // reduction)
        small_h = max(1, (v1 - v0) // reduction)
        low = cv2.GaussianBlur(
            cv2.resize(
                color * mask[..., None],
                (small_w, small_h),
                interpolation=cv2.INTER_AREA,
            ),
            (0, 0),
            sigma / reduction,
        )
        low_mask = cv2.GaussianBlur(
            cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_AREA),
            (0, 0),
            sigma / reduction,
        )
        size = (u1 - u0, v1 - v0)
        low = cv2.resize(low, size, interpolation=cv2.INTER_LINEAR)
        low_mask = cv2.resize(low_mask, size, interpolation=cv2.INTER_LINEAR)
    else:
        low = cv2.GaussianBlur(color * mask[..., None], (0, 0), sigma)
        low_mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    low /= np.maximum(low_mask, 1e-6)[..., None]
    high = (color - low) * mask[..., None]
    if lf_gate is not None and frame_group is not None:
        soft *= lf_gate[frame_group[frame.idx]][v0:v1, u0:u1].astype(np.float32) / 255.0
    state.acc[v0:v1, u0:u1] += low * soft[..., None]
    state.wacc[v0:v1, u0:u1] += soft
    ownership_weight = (
        weight
        if hf_coherence_mm <= 0
        else cv2.GaussianBlur(weight, (0, 0), hf_coherence_mm * state.ppmm)
    )
    wins = ownership_weight > state.w_best[v0:v1, u0:u1]
    state.hf_best[v0:v1, u0:u1][wins] = high[wins]
    state.w_best[v0:v1, u0:u1][wins] = ownership_weight[wins]


def accumulate_camera_frames_into_texture(
    state: RenderState,
    frame_group: FrameGroups,
    frame_gain: NumericArray | None,
    best_group: NumericArray | None,
    geo_cache: dict[int, Geometry],
    lf_gate: Mapping[int, NDArray[np.uint8]] | None,
    focus_weight: float,
    hf_cross_group: bool,
    hf_coherence_mm: float,
) -> None:
    """Render all frames, using GPU tiles when available and CPU as fallback."""
    started = time.time()
    profile: dict[str, float] = {}
    for frame in state.frames:
        step_started = time.perf_counter()
        if state.gpu is not None:
            geometry = geo_cache.pop(frame.idx, None) or compute_camera_frame_surface_projection(
                state, frame, upsample=False
            )
            step_started = log_texture_render_profile_tick(profile, "geom", step_started)
            if geometry is None:
                continue
            rect, decimated = cast(tuple[Rect, NumericArray], geometry)
            owner = best_group if best_group is not None and not hf_cross_group else None
            gate = (
                lf_gate[frame_group[frame.idx]]
                if lf_gate is not None and frame_group is not None
                else None
            )
            try:
                tiles = state.gpu.frame(
                    frame.image_path,
                    rect,
                    decimated,
                    sscale=state.sscale,
                    img_pxmm=state.img_pxmm,
                    focus_weight=focus_weight,
                    best_g=owner,
                    gid=frame_group[frame.idx] if frame_group else 0,
                    lf_gate=gate,
                    pixels_per_mm=state.ppmm,
                    hf_coherence_mm=hf_coherence_mm,
                    gain=None if frame_gain is None else frame_gain[frame.idx],
                )
            except state.gpu_oom:
                print(
                    f"        ! frame {frame.idx}: tile exceeds VRAM even "
                    "after support crop -- rendering it on CPU"
                )
                if state.gpu_canvas:
                    (
                        state.acc,
                        state.wacc,
                        high_gpu,
                        best_gpu,
                    ) = state.gpu.canvases_take()
                    if high_gpu is not None:
                        state.hf_best, state.w_best = high_gpu, best_gpu
                    state.gpu_canvas = False
                tiles = "cpu-fallback"
            if tiles != "cpu-fallback":
                step_started = log_texture_render_profile_tick(profile, "gpu", step_started)
                if tiles is None:
                    continue
                if tiles.get("accumulated"):
                    log_texture_render_profile_tick(profile, "accum", step_started)
                    print_texture_render_progress(frame, state.frames, started, profile)
                    continue
                accumulate_gpu_texture_render_tiles(state, tiles)
                log_texture_render_profile_tick(profile, "accum", step_started)
                print_texture_render_progress(frame, state.frames, started, profile)
                continue
        geometry = compute_camera_frame_surface_projection(state, frame)
        step_started = log_texture_render_profile_tick(profile, "geom", step_started)
        if geometry is None:
            continue
        accumulate_camera_frame_into_cpu_texture(
            state,
            frame,
            cast(tuple[Rect, NumericArray, NumericArray, NumericArray, NumericArray], geometry),
            frame_group,
            frame_gain,
            best_group,
            lf_gate,
            focus_weight,
            hf_cross_group,
            hf_coherence_mm,
        )
        log_texture_render_profile_tick(profile, "cpu", step_started)
        print_texture_render_progress(frame, state.frames, started, profile)
    if state.gpu_canvas:
        assert state.gpu is not None
        state.acc, state.wacc, high_gpu, high_frequency_weight = state.gpu.canvases_take()
        if high_gpu is not None:
            state.hf_best = high_gpu


def accumulate_gpu_texture_render_tiles(state: RenderState, tiles: GpuTiles) -> None:
    u0, u1, v0, v1 = tiles["rect"]
    if state.blend_mode == "two-band":
        two_band_tiles = cast(GpuTwoBandTiles, tiles)
        soft, ownership_weight = two_band_tiles["soft"], two_band_tiles["wown"]
        state.acc[v0:v1, u0:u1] += two_band_tiles["lo"] * soft[..., None]
        state.wacc[v0:v1, u0:u1] += soft
        wins = ownership_weight > state.w_best[v0:v1, u0:u1]
        state.hf_best[v0:v1, u0:u1][wins] = two_band_tiles["hf"][wins]
        state.w_best[v0:v1, u0:u1][wins] = ownership_weight[wins]
    else:
        soft_blend_tiles = cast(GpuSoftBlendTiles, tiles)
        state.acc[v0:v1, u0:u1] += soft_blend_tiles["col"] * soft_blend_tiles["wgt2"][..., None]
        state.wacc[v0:v1, u0:u1] += soft_blend_tiles["wgt2"]


def print_texture_render_progress(
    frame: Frame, frames: Sequence[Frame], started: float, profile: Mapping[str, float]
) -> None:
    if frame.idx % 20 == 0 or frame.idx == len(frames) - 1:
        breakdown = " ".join(f"{key}:{value:.0f}s" for key, value in profile.items())
        print(
            f"        frame {frame.idx:>3}/{len(frames)} "
            f"({time.time() - started:.0f}s | {breakdown})"
        )


def write_texture_outputs(state: RenderState, out_dir: str) -> tuple[NDArray[np.uint8], FloatArray | None, Bounds]:
    """Normalize the canvases and write texture, coverage, and index images."""
    texture = state.acc / np.maximum(state.wacc[..., None], 1e-6)
    if state.blend_mode == "two-band":
        texture += state.hf_best
    texture = texture.clip(0, 255).astype(np.uint8)
    texture[state.wacc == 0] = 40
    cv2.imwrite(
        os.path.join(out_dir, "texture.jpg"),
        texture,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    cv2.imwrite(
        os.path.join(out_dir, "coverage.png"),
        (state.wacc > 0).astype(np.uint8) * 255,
    )
    index_image = texture.copy()
    for frame in state.frames:
        footprint = state.foot_uv[frame.idx]
        points = np.stack(
            [
                (footprint[:, 0] - state.umin) * state.ppmm,
                (footprint[:, 1] - state.vmin) * state.ppmm,
            ],
            1,
        )
        cv2.polylines(
            index_image,
            [points.astype(np.int32)],
            True,
            (0, 255, 0),
            2,
        )
        center = points.mean(0).astype(int)
        cv2.putText(
            index_image,
            str(frame.station),
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    cv2.imwrite(
        os.path.join(out_dir, "texture_index.jpg"),
        index_image,
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    return texture, state.wacc, state.bounds


def render_surface_texture(
    frames: Sequence[Frame],
    camera_rotations: NumericArray,
    camera_centers: NumericArray,
    rig_model: RigModel,
    surf: Surface,
    texture_parameters: TexParam,
    pixels_per_mm: float,
    up_sign: float,
    out_dir: str,
    blend_sharpness: float = 100.0,
    focus_weight: float = 0.0,
    blend_mode: str = "soft",
    frame_group: FrameGroups = None,
    warp: Warp = None,
    hf_coherence_mm: float = 0.0,
    hf_cross_group: bool = False,
    group_feather_mm: float = 0.0,
    max_incidence_deg: float = 0.0,
    device: str = "auto",
    frame_gain: NumericArray | None = None,
) -> tuple[NDArray[np.uint8], FloatArray | None, Bounds]:
    """Render the registered captures through bounded, independently testable stages."""
    foot_uv, bounds, width, height = compute_texture_bounds(frames, camera_rotations, camera_centers, rig_model, surf, texture_parameters, pixels_per_mm)
    state = prepare_texture_render_state(
        frames,
        camera_rotations,
        camera_centers,
        rig_model,
        surf,
        texture_parameters,
        pixels_per_mm,
        up_sign,
        blend_sharpness,
        blend_mode,
        max_incidence_deg,
        warp,
        device,
        foot_uv,
        bounds,
        width,
        height,
    )
    best_group, geometry_cache, group_ids = build_texture_ownership_by_frame_group(state, frame_group)
    low_frequency_gates = compute_low_frequency_group_blend_gates(
        best_group, group_ids, group_feather_mm, pixels_per_mm, width, height
    )
    accumulate_camera_frames_into_texture(
        state,
        frame_group,
        frame_gain,
        best_group,
        geometry_cache,
        low_frequency_gates,
        focus_weight,
        hf_cross_group,
        hf_coherence_mm,
    )
    return write_texture_outputs(state, out_dir)
