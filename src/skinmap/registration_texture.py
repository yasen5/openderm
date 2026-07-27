"""Photometric gain fitting and CPU/GPU ortho-texture rendering."""

from __future__ import annotations

import math
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

from .registration_features import _mem_available_bytes
from .registration_geometry import project
from .registration_surface import frame_footprint_world


# ----------------------------------------------------------------------------
# ortho-texture rendering
# ----------------------------------------------------------------------------
def fit_frame_gains(frames, mdl, obs_frame, obs_uv, obs_track, err=None, mode="on"):
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
    mode 'field': an affine log-gain field per frame g(x^,y^) = exp(g0 +
                  gx*x^ + gy*y^) in normalised image coords (x^,y^ in
                  [-0.5,0.5]) -- also corrects the within-frame shading /
                  vignetting gradient that a scalar cannot. Returns (F,3,3)
                  float32 log-coeffs [channel, (g0,gx,gy)]; the render
                  exponentiates per texel.
    """
    RED = 8  # IMREAD_REDUCED_COLOR_8 decode factor
    ok = np.ones(len(obs_frame), bool) if err is None else (err < 10.0)
    zmean = float(np.mean([mdl.depth(f) for f in frames]))
    # sample at the LF band's own scale (~2mm) so the gains describe exactly
    # the band they will correct
    sig = max(2.0, 2.0 * (mdl.fx * mdl.downscale / zmean) / RED)
    F = len(frames)
    samples = np.zeros((len(obs_frame), 3), np.float32)
    have = np.zeros(len(obs_frame), bool)
    for f in frames:
        sel = np.where((obs_frame == f.idx) & ok)[0]
        if not len(sel):
            continue
        im = cv2.imread(f.image_path, cv2.IMREAD_REDUCED_COLOR_8 | cv2.IMREAD_IGNORE_ORIENTATION)
        if im is None:
            continue
        im = cv2.GaussianBlur(im.astype(np.float32), (0, 0), sig)
        pt = obs_uv[sel] * (mdl.downscale / RED)
        xi = np.clip(np.round(pt[:, 0]).astype(int), 0, im.shape[1] - 1)
        yi = np.clip(np.round(pt[:, 1]).astype(int), 0, im.shape[0] - 1)
        v = im[yi, xi]
        good = (v.min(1) > 5.0) & (v.max(1) < 250.0)  # not clipped
        samples[sel[good]] = v[good]
        have[sel[good]] = True
    fi = obs_frame[have]
    ti = obs_track[have]
    if len(fi) < 10 * F:
        print(f"      ! LF gain: only {len(fi)} usable track colors -- skipping compensation")
        return None
    l = np.log(samples[have].clip(1.0, None))  # (M,3)
    # normalised image coords of each obs (downscale cancels: ds px / ds dims)
    xh = obs_uv[have, 0] / (2.0 * mdl.cx) - 0.5
    yh = obs_uv[have, 1] / (2.0 * mdl.cy) - 0.5
    B = np.stack([np.ones_like(xh), xh, yh], 1)  # (M, 3) field basis
    nb = 1 if mode == "on" else 3
    ntr = int(ti.max()) + 1
    # model: l + G_f(x^,y^) ~= mu_t, G = log gain the render will apply
    gam = np.zeros((F, 3, nb))
    keep = np.ones(len(l), bool)
    for it in range(16):
        G = np.einsum("mcb,mb->mc", gam[fi], B[:, :nb])
        mu = np.zeros((ntr, 3))
        cnt = np.zeros(ntr)
        np.add.at(mu, ti[keep], (l + G)[keep])
        np.add.at(cnt, ti[keep], 1)
        mu /= np.maximum(cnt, 1)[:, None]
        t = mu[ti] - l  # per-obs target log-gain
        gc = np.zeros(F)
        np.add.at(gc, fi[keep], 1)
        if nb == 1:
            gs = np.zeros((F, 3))
            np.add.at(gs, fi[keep], t[keep])
            gam[:, :, 0] = gs / np.maximum(gc, 1)[:, None]
        else:
            for f in range(F):
                s = keep & (fi == f)
                m = int(s.sum())
                if m < 12:
                    continue
                Bf = B[s]
                # ridge on the gradient terms only: a weak prior that wins
                # when a frame's obs are clustered, negligible otherwise
                lam = max(20.0, 0.02 * m)
                N = Bf.T @ Bf + np.diag([1e-9, lam, lam])
                gam[f] = np.linalg.solve(N, Bf.T @ t[s]).T
        if it == 7:  # one robust trim
            res = l + np.einsum("mcb,mb->mc", gam[fi], B[:, :nb]) - mu[ti]
            keep &= np.abs(res).max(1) < 2.5 * (res.std() + 1e-6)
    gam[:, :, 0] -= gam[:, :, 0].mean(0, keepdims=True)  # geometric mean 1
    if mode == "on":
        gains = np.exp(gam[:, :, 0]).astype(np.float32)
        print(
            f"      LF gain compensation: {int((gc > 0).sum())}/{F} frames "
            f"from {int(keep.sum())} track colors, gain range "
            f"{gains.min():.3f}..{gains.max():.3f}"
        )
        return gains
    g0 = np.exp(gam[:, :, 0])
    span = float(np.abs(gam[:, :, 1:]).max())
    print(
        f"      LF gain field: {int((gc > 0).sum())}/{F} frames from "
        f"{int(keep.sum())} track colors, centre gain {g0.min():.3f}.."
        f"{g0.max():.3f}, max |gradient| {span:.3f}/half-image"
    )
    return gam.astype(np.float32)


def _texture_bounds(frames, R, C, mdl, surf, tp, ppmm):
    """Return frame footprints and the landmark-supported texture canvas."""
    foot_uv = {}
    umin = vmin = np.inf
    umax = vmax = -np.inf
    for f in frames:
        P = frame_footprint_world(f, R, C, mdl, surf)
        u, v = tp.to_uv(P[:, 0], P[:, 1])
        foot_uv[f.idx] = np.stack([u, v], 1)
        umin, umax = min(umin, u.min()), max(umax, u.max())
        vmin, vmax = min(vmin, v.min()), max(vmax, v.max())
    pad = 1.0
    umin -= pad
    vmin -= pad
    umax += pad
    vmax += pad
    # clamp to the landmark-supported surface region (no textured extrapolation)
    sy_i, sx_i = np.where(surf.support)
    if len(sx_i):
        su, sv = tp.to_uv(surf.xs[sx_i], surf.ys[sy_i])
        umin = max(umin, su.min() - pad)
        umax = min(umax, su.max() + pad)
        vmin = max(vmin, sv.min() - pad)
        vmax = min(vmax, sv.max() + pad)
    W = int(math.ceil((umax - umin) * ppmm))
    H = int(math.ceil((vmax - vmin) * ppmm))
    print(
        f"      texture canvas {W}x{H} @ {ppmm}px/mm "
        f"(u: {umin:.0f}..{umax:.0f}, v: {vmin:.0f}..{vmax:.0f} mm)"
    )
    return foot_uv, (umin, vmin, umax, vmax), W, H


def _prepare_renderer(
    frames,
    R,
    C,
    mdl,
    surf,
    tp,
    ppmm,
    up_sign,
    blend_sharpness,
    blend_mode,
    max_incidence_deg,
    warp,
    device,
    foot_uv,
    bounds,
    W,
    H,
):
    """Allocate canvases and initialize the optional GPU geometry pipeline."""
    umin, vmin, umax, vmax = bounds
    # GPU fast path: the per-frame image pipeline (decode/remap/blurs) runs on
    # CUDA and returns finished tiles; the canvases stay in system RAM and the
    # accumulation below is shared with the CPU path (see render_gpu.py).
    gpu = None
    gpu_oom = RuntimeError
    if device != "cpu":
        from skinmap.render_gpu import GpuFramePipe, GpuOom, gpu_available

        gpu_oom = GpuOom
        if gpu_available():
            gpu = GpuFramePipe(blend_mode)
            if os.environ.get("SKINMAP_NO_VRAM_CANVAS"):
                print("      render device: cuda (torch), canvases in RAM (SKINMAP_NO_VRAM_CANVAS)")
            elif gpu.canvas_fits(H, W):
                gpu.canvases_begin(H, W)
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
    need = float(H) * W * per_texel
    avail_ram = _mem_available_bytes()
    if need > 0.6 * avail_ram:
        sys.exit(
            f"      ! texture canvases for {W}x{H} need {need / 1e9:.1f} GB "
            f"but only {avail_ram / 1e9:.1f} GB RAM is available -- "
            "aborting before the OOM killer does it for us. A canvas "
            "this large usually means a degenerate registration "
            "(inflated arc length); otherwise lower --texture-ppmm."
        )
    acc = wacc = hf_best = w_best = None
    if not gpu_canvas:
        acc = np.zeros((H, W, 3), np.float32)
        wacc = np.zeros((H, W), np.float32)
    if blend_mode == "two-band" and not gpu_canvas:
        # high frequencies are never averaged: per texel the argmax-weight
        # frame contributes them alone; only the low-pass (illumination) band
        # is soft-blended, which hides photometric seams without losing detail
        hf_best = np.zeros((H, W, 3), np.float32)
        w_best = np.zeros((H, W), np.float32)

    fxf = mdl.fx * mdl.downscale
    k1 = mdl.k1
    cxf, cyf = mdl.cx * mdl.downscale, mdl.cy * mdl.downscale
    Wf, Hf = cxf * 2, cyf * 2
    img_pxmm = fxf / np.mean([mdl.depth(f) for f in frames])
    sscale = min(1.0, 1.4 * ppmm / img_pxmm)

    # geometry decimation: the per-texel fields geom() computes (projected
    # image coords, spline height, weights) are all smooth at the surface-grid
    # scale (2mm pitch), while scipy spline .ev is ~us/point single-threaded --
    # at 78px/mm a frame tile is ~10M texels and the splines dominate the whole
    # render (minutes/frame). Evaluate on a ~20px/mm grid (q texels) and
    # bilinearly upsample: placement error << the solve rms, ~16x faster.
    # q=1 at <=20px/mm uses the exact per-texel path.
    geo_q = max(1, int(round(ppmm / 20.0)))

    # geometry fields on CUDA: same math as geom() below (float64), evaluated
    # on the GPU and never leaving it -- geom dominated the GPU render's wall
    # clock (~3 s/frame of spline/projection numpy per pass)
    gpu_geom = None
    if gpu is not None:
        from skinmap.render_gpu import GpuGeom

        gpu_geom = GpuGeom(
            surf,
            tp,
            R,
            C,
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
    return SimpleNamespace(
        frames=frames,
        R=R,
        C=C,
        mdl=mdl,
        surf=surf,
        tp=tp,
        ppmm=ppmm,
        up_sign=up_sign,
        blend_sharpness=blend_sharpness,
        blend_mode=blend_mode,
        max_incidence_deg=max_incidence_deg,
        warp=warp,
        foot_uv=foot_uv,
        bounds=bounds,
        W=W,
        H=H,
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
    )


def _frame_geometry(state, frame, upsample=True):
    """Project one frame and calculate its geometric blend weights."""
    fp = state.foot_uv[frame.idx]
    umin, vmin, ppmm = state.umin, state.vmin, state.ppmm
    W, H, q = state.W, state.H, state.geo_q
    u0 = max(0, int((fp[:, 0].min() - umin - 1) * ppmm))
    u1 = min(W, int((fp[:, 0].max() - umin + 1) * ppmm) + 1)
    v0 = max(0, int((fp[:, 1].min() - vmin - 1) * ppmm))
    v1 = min(H, int((fp[:, 1].max() - vmin + 1) * ppmm) + 1)
    if u1 <= u0 or v1 <= v0:
        return None
    if q > 1:
        wd = max(2, -(-(u1 - u0) // q))
        hd = max(2, -(-(v1 - v0) // q))
        uu = umin + (u0 + (np.arange(wd) + 0.5) * q) / ppmm
        vv = vmin + (v0 + (np.arange(hd) + 0.5) * q) / ppmm
    else:
        wd, hd = u1 - u0, v1 - v0
        uu = umin + (np.arange(u0, u1) + 0.5) / ppmm
        vv = vmin + (np.arange(v0, v1) + 0.5) / ppmm
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
        a = Mu[frame.idx]
        bvp = Mv[frame.idx]
        cu_ = a[0] * du + a[1] * dv + a[2]
        cv_ = bvp[0] * du + bvp[1] * dv + bvp[2]
        if len(a) == 6:
            q0, q1, q2 = du * du / 50.0, du * dv / 50.0, dv * dv / 50.0
            cu_ += a[3] * q0 + a[4] * q1 + a[5] * q2
            cv_ += bvp[3] * q0 + bvp[4] * q1 + bvp[5] * q2
        Uw, Vw = Uw - cu_, Vw - cv_
    Xg, Yg = state.tp.to_xy(Uw, Vw)
    Zg = state.surf.height(Xg, Yg)
    P = np.stack([Xg, Yg, Zg], 1)
    uv, zc = project(
        P,
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
    vs = valid * state.surf.supported(P[:, 0], P[:, 1])
    if state.max_incidence_deg > 0:
        nrm = state.surf.normal(P[:, 0], P[:, 1], state.up_sign)
        ray = P - state.C[frame.idx][None, :]
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
    if q > 1:
        size = (u1 - u0, v1 - v0)
        mapx = cv2.resize(mapx, size, interpolation=cv2.INTER_LINEAR)
        mapy = cv2.resize(mapy, size, interpolation=cv2.INTER_LINEAR)
        wgt2d = cv2.resize(wgt2d, size, interpolation=cv2.INTER_LINEAR)
        soft2d = cv2.resize(soft2d, size, interpolation=cv2.INTER_LINEAR)
    return (u0, u1, v0, v1), mapx, mapy, wgt2d, soft2d


def _group_ownership(state, frame_group):
    """Compute one owning capture group per texture texel."""
    if frame_group is None:
        return None, {}, []
    frames, gpu = state.frames, state.gpu
    H, W = state.H, state.W
    started = time.time()
    gids = sorted(set(frame_group.values()))
    geo_cache = {}
    geo_bytes, geo_budget = 0, 0.25 * _mem_available_bytes()
    gpu_own = False
    if gpu is not None:
        try:
            gpu.ownership_begin(H, W)
            gpu_own = True
        except Exception as exc:
            print(f"      ! ownership fields don't fit VRAM ({exc}); streaming ownership on CPU")
    ordered = sorted(frames, key=lambda frame: frame_group[frame.idx])
    if gpu_own:
        for frame in ordered:
            result = _frame_geometry(state, frame, upsample=False)
            if result is None:
                continue
            if state.gpu_geom is None and geo_bytes < geo_budget:
                geo_cache[frame.idx] = result
                geo_bytes += result[1].nbytes
            rect, decimated = result
            gpu.ownership_add(frame_group[frame.idx], rect, decimated[2])
        best_group = gpu.ownership_finish()
    else:
        own_max = np.zeros((H, W), np.float32)
        own_arg = np.full((H, W), -1, np.int32)
        own_sum = np.zeros((H, W), np.float32)
        current_group = None
        for frame in ordered:
            result = _frame_geometry(state, frame)
            if result is None:
                continue
            (u0, u1, v0, v1), _, _, weight, _ = result
            group = frame_group[frame.idx]
            if current_group is not None and group != current_group:
                wins = own_sum > own_max
                own_max[wins] = own_sum[wins]
                own_arg[wins] = current_group
                own_sum[...] = 0.0
            current_group = group
            own_sum[v0:v1, u0:u1] += weight
        if current_group is not None:
            wins = own_sum > own_max
            own_arg[wins] = current_group
        best_group = own_arg
    print(f"      group-owned compositing across {len(gids)} groups ({time.time() - started:.0f}s)")
    return best_group, geo_cache, gids


def _low_frequency_group_gates(
    best_group,
    gids,
    group_feather_mm,
    ppmm,
    W,
    H,
):
    """Build feathered, partition-of-unity gates for low-frequency blending."""
    if best_group is None or group_feather_mm <= 0:
        return None
    sigma = group_feather_mm * ppmm
    reduction = max(1, int(round(sigma / 12.0)))
    small_w, small_h = max(1, W // reduction), max(1, H // reduction)
    gates = []
    for group in gids:
        mask = (best_group == group).astype(np.float32)
        if reduction > 1:
            mask = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gates.append(cv2.GaussianBlur(mask, (0, 0), sigma / reduction))
    gate_sum = np.maximum(np.sum(gates, 0), 1e-6)
    result = {}
    for group, mask in zip(gids, gates):
        mask /= gate_sum
        if reduction > 1:
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
        result[group] = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    print(f"      LF group gates feathered at {group_feather_mm:.0f}mm")
    return result


def _tick(profile, key, started):
    profile[key] = profile.get(key, 0.0) + time.perf_counter() - started
    return time.perf_counter()


def _deposit_cpu_frame(
    state,
    frame,
    geometry,
    frame_group,
    frame_gain,
    best_group,
    lf_gate,
    focus_weight,
    hf_cross_group,
    hf_coherence_mm,
):
    """Decode and deposit one frame through the CPU rendering path."""
    (u0, u1, v0, v1), mapx, mapy, weight, soft = geometry
    image = cv2.imread(
        frame.image_path,
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
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
    if best_group is not None and not hf_cross_group:
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
    if lf_gate is not None:
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


def _deposit_frames(
    state,
    frame_group,
    frame_gain,
    best_group,
    geo_cache,
    lf_gate,
    focus_weight,
    hf_cross_group,
    hf_coherence_mm,
):
    """Render all frames, using GPU tiles when available and CPU as fallback."""
    started = time.time()
    profile = {}
    for frame in state.frames:
        step_started = time.perf_counter()
        if state.gpu is not None:
            geometry = geo_cache.pop(frame.idx, None) or _frame_geometry(
                state, frame, upsample=False
            )
            step_started = _tick(profile, "geom", step_started)
            if geometry is None:
                continue
            rect, decimated = geometry
            owner = best_group if best_group is not None and not hf_cross_group else None
            gate = lf_gate[frame_group[frame.idx]] if lf_gate is not None else None
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
                    ppmm=state.ppmm,
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
                step_started = _tick(profile, "gpu", step_started)
                if tiles is None:
                    continue
                if tiles.get("accumulated"):
                    _tick(profile, "accum", step_started)
                    _print_render_progress(frame, state.frames, started, profile)
                    continue
                _accumulate_gpu_tiles(state, tiles)
                _tick(profile, "accum", step_started)
                _print_render_progress(frame, state.frames, started, profile)
                continue
        geometry = _frame_geometry(state, frame)
        step_started = _tick(profile, "geom", step_started)
        if geometry is None:
            continue
        _deposit_cpu_frame(
            state,
            frame,
            geometry,
            frame_group,
            frame_gain,
            best_group,
            lf_gate,
            focus_weight,
            hf_cross_group,
            hf_coherence_mm,
        )
        _tick(profile, "cpu", step_started)
        _print_render_progress(frame, state.frames, started, profile)
    if state.gpu_canvas:
        state.acc, state.wacc, high_gpu, _ = state.gpu.canvases_take()
        if high_gpu is not None:
            state.hf_best = high_gpu


def _accumulate_gpu_tiles(state, tiles):
    u0, u1, v0, v1 = tiles["rect"]
    if state.blend_mode == "two-band":
        soft, ownership_weight = tiles["soft"], tiles["wown"]
        state.acc[v0:v1, u0:u1] += tiles["lo"] * soft[..., None]
        state.wacc[v0:v1, u0:u1] += soft
        wins = ownership_weight > state.w_best[v0:v1, u0:u1]
        state.hf_best[v0:v1, u0:u1][wins] = tiles["hf"][wins]
        state.w_best[v0:v1, u0:u1][wins] = ownership_weight[wins]
    else:
        state.acc[v0:v1, u0:u1] += tiles["col"] * tiles["wgt2"][..., None]
        state.wacc[v0:v1, u0:u1] += tiles["wgt2"]


def _print_render_progress(frame, frames, started, profile):
    if frame.idx % 20 == 0 or frame.idx == len(frames) - 1:
        breakdown = " ".join(f"{key}:{value:.0f}s" for key, value in profile.items())
        print(
            f"        frame {frame.idx:>3}/{len(frames)} "
            f"({time.time() - started:.0f}s | {breakdown})"
        )


def _write_texture_outputs(state, out_dir):
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


def render_texture(
    frames,
    R,
    C,
    mdl,
    surf,
    tp,
    ppmm,
    up_sign,
    out_dir,
    blend_sharpness=100.0,
    focus_weight=0.0,
    blend_mode="soft",
    frame_group=None,
    warp=None,
    hf_coherence_mm=0.0,
    hf_cross_group=False,
    group_feather_mm=0.0,
    max_incidence_deg=0.0,
    device="auto",
    frame_gain=None,
):
    """Render the registered captures through bounded, independently testable stages."""
    foot_uv, bounds, width, height = _texture_bounds(frames, R, C, mdl, surf, tp, ppmm)
    state = _prepare_renderer(
        frames,
        R,
        C,
        mdl,
        surf,
        tp,
        ppmm,
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
    best_group, geometry_cache, group_ids = _group_ownership(state, frame_group)
    low_frequency_gates = _low_frequency_group_gates(
        best_group, group_ids, group_feather_mm, ppmm, width, height
    )
    _deposit_frames(
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
    return _write_texture_outputs(state, out_dir)
