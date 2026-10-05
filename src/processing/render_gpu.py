"""GPU (torch/CUDA) fast path for render_surface_texture's per-frame image pipeline.

Division of labour: the geometry fields stay on CPU (decimated ~20px/mm grid,
cheap), the texture CANVASES stay in system RAM (VRAM must not bound canvas
size), and everything per-frame image-sized runs on the GPU:

  JPEG decode (nvJPEG) -> bilinear upsample of the geometry fields -> remap
  (grid_sample) -> focus/sharpness blurs -> two-band low-pass -> group/LF
  gates -> HF-ownership blur

frame() returns the finished per-frame TILES as numpy; the caller accumulates
them into the shared canvases with the exact same code the CPU path uses, so
the two paths cannot diverge at the compositing stage.  Per-frame transfer is
~0.3 GB over PCIe (~10 ms) against ~30 s of cv2 work replaced.

Every op mirrors the cv2 call it replaces -- identical Gaussian taps (taken
from cv2.getGaussianKernel), the same half-pixel bilinear convention
(align_corners=False), REFLECT_101 borders -- so a GPU render matches the CPU
render to float tolerance.  The one knowing approximation: cv2.INTER_AREA
box-resampling is torch's adaptive 'area' mode, whose fractional-window edges
differ slightly; it only feeds strongly low-passed decision/illumination
fields, never deposited detail.

The ownership pre-pass keeps a running (max, argmax) pair on the GPU instead
of one canvas per group, so its VRAM use is independent of group count.
"""

# Torch and torchvision are optional GPU-only dependencies.
# pyright: reportMissingImports=false

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, TYPE_CHECKING, TypeAlias, TypedDict, cast

if TYPE_CHECKING:
    import torch as torch_typing
    from .registration_surface import Surface, TexParam

    TorchTensor: TypeAlias = torch_typing.Tensor
else:
    TorchTensor: TypeAlias = Any

FloatArray = NDArray[np.float32]
NumericArray = NDArray[Any]
Rect = tuple[int, int, int, int]
Warp = tuple[NumericArray, NumericArray, float, float] | None


def _tensor_numpy(tensor: TorchTensor) -> NDArray[np.float32]:
    """Convert a float32 tensor to NumPy (torch's stubs leave numpy() unknown)."""
    return cast(NDArray[np.float32], tensor.cpu().numpy())  # type: ignore[reportUnknownMemberType]


class GpuFrameResult(TypedDict, total=False):
    rect: Rect
    accumulated: bool
    col: FloatArray
    wgt2: FloatArray
    lo: FloatArray
    soft: FloatArray
    hf: FloatArray
    wown: FloatArray


# Keep the CUDA path optional at runtime while giving the checker stable names.
torch: Any = None
torch_functional: Any = None
read_file: Any = None
decode_jpeg: Any = None
ImageReadMode: Any = None

try:
    import torch
    import torch.nn.functional as torch_functional
    from torchvision.io import read_file, decode_jpeg, ImageReadMode  # type: ignore[reportMissingTypeStubs]

    _torch_available = True
except Exception:  # torch not installed
    _torch_available = False

import cv2


def texture_gpu_renderer_available() -> bool:
    return _torch_available and torch.cuda.is_available()


class GpuOom(RuntimeError):
    """Raised when a frame doesn't fit in VRAM even after a retry; the caller
    renders that frame on the CPU path instead."""


class GpuGeom:
    """Per-frame geometry-field evaluation on CUDA, mirroring the numpy math
    of render_surface_texture's geom() (bilinear surface interps, the arc-length
    unwrap and its Newton inverse, projection, feather x centre x incidence
    weights). Coordinates run in float64 exactly like numpy; the returned
    field stack is float32 like the CPU path's final cast."""

    def __init__(
        self,
        surf: Surface,
        texture_parameters: TexParam,
        camera_rotations: NumericArray,
        camera_centers: NumericArray,
        fxf: float,
        k1: float,
        cxf: float,
        cyf: float,
        Wf: float,
        Hf: float,
        up_sign: float,
        blend_sharpness: float,
        max_incidence_deg: float,
        warp: Warp,
        device: str = "cuda",
    ) -> None:
        distance = torch.device(device)
        f64: dict[str, Any] = {"dtype": torch.float64, "device": distance}
        self.dev = distance
        self.zg = torch.tensor(surf.z, **f64)
        self.gxg = torch.tensor(surf.gx, **f64)
        self.gyg = torch.tensor(cast(NumericArray, surf.gy), **f64)
        self.supg = torch.tensor(surf.support.astype(np.uint8), device=distance)
        self.xs0, self.dxs, self.nx = (
            float(surf.xs[0]),
            float(surf.xs[1] - surf.xs[0]),
            len(surf.xs),
        )
        self.ys0, self.dys, self.ny = (
            float(surf.ys[0]),
            float(surf.ys[1] - surf.ys[0]),
            len(surf.ys),
        )
        self.bR = torch.tensor(np.asarray(texture_parameters.bR), **f64)
        self.bt = torch.tensor(np.asarray(texture_parameters.bt), **f64)
        self.arc_s = torch.tensor(texture_parameters.s, **f64)
        self.arc_gy = torch.tensor(texture_parameters.gy, **f64)  # type: ignore[reportUnknownMemberType]
        self.R = torch.tensor(np.stack(list(camera_rotations)), **f64)  # (F,3,3) cam->world
        self.C = torch.tensor(np.stack(list(camera_centers)), **f64)  # (F,3)
        self.fxf, self.k1 = float(fxf), float(k1)
        self.cxf, self.cyf = float(cxf), float(cyf)
        self.Wf, self.Hf = float(Wf), float(Hf)
        self.up_sign = float(up_sign)
        self.bs = float(blend_sharpness)
        self.max_inc = float(max_incidence_deg)
        if warp is None:
            self.warp = None
        else:
            Mu, Mv, umid, vmid = warp
            self.warp = (
                torch.tensor(np.stack(list(Mu)), **f64),
                torch.tensor(np.stack(list(Mv)), **f64),
                float(umid),
                float(vmid),
            )

    @staticmethod
    def _interp1(array: TorchTensor, xp: TorchTensor, fp: TorchTensor) -> TorchTensor:
        """np.interp for monotonically increasing xp (endpoint-clamped)."""
        index = torch.searchsorted(xp, array).clamp(1, len(xp) - 1)
        x0, x1 = xp[index - 1], xp[index]
        f0, f1 = fp[index - 1], fp[index]
        threshold = ((array - x0) / (x1 - x0)).clamp(0.0, 1.0)
        return f0 + threshold * (f1 - f0)

    def _bil(self, grid: TorchTensor, array: TorchTensor, world_y: TorchTensor) -> TorchTensor:
        """Surface-grid bilinear interp, identical clip/floor to Surface._interp."""
        xi = ((array - self.xs0) / self.dxs).clamp(0, self.nx - 1.001)
        yi = ((world_y - self.ys0) / self.dys).clamp(0, self.ny - 1.001)
        x0 = xi.floor().long()
        y0 = yi.floor().long()
        fx = xi - x0
        fy = yi - y0
        gauge = grid
        return (gauge[y0, x0] * (1 - fx) + gauge[y0, x0 + 1] * fx) * (1 - fy) + (
            gauge[y0 + 1, x0] * (1 - fx) + gauge[y0 + 1, x0 + 1] * fx
        ) * fy

    def _height(self, array: TorchTensor, world_y: TorchTensor) -> TorchTensor:
        return self._bil(self.zg, array, world_y)

    def _to_xy(self, texture_u: TorchTensor, texture_v: TorchTensor) -> tuple[TorchTensor, TorchTensor]:
        """TexParam.to_xy: arc-length -> gantry y, then 6 Newton steps."""
        gy = self._interp1(texture_v, self.arc_s, self.arc_gy)
        array = texture_u.clone()
        world_y = gy.clone()
        for _ in range(6):
            points = torch.stack([array, world_y, self._height(array, world_y)], -1)
            Pg = (points - self.bt) @ self.bR
            array = array - (Pg[..., 0] - texture_u)
            world_y = world_y - (Pg[..., 1] - gy)
        return array, world_y

    def fields(self, fidx: int, uu: NumericArray, vv: NumericArray) -> TorchTensor | None:
        """Decimated geometry fields for one frame's tile.
        uu (wd,), vv (hd,) numpy float64 texture-mm sample coords.
        -> torch float32 (4,hd,wd) [mapx,mapy,wgt,soft] on device, or None.
        Raises GpuOom when VRAM is exhausted (e.g. another process grabbed
        the GPU mid-run); the caller falls back to the numpy geom path."""
        try:
            return self._fields(fidx, uu, vv)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            try:
                return self._fields(fidx, uu, vv)
            except torch.OutOfMemoryError as caught_exception:
                torch.cuda.empty_cache()
                raise GpuOom(str(caught_exception)) from None

    def _fields(self, fidx: int, uu: NumericArray, vv: NumericArray) -> TorchTensor | None:
        f64 = dict(dtype=torch.float64, device=self.dev)
        uu_t = torch.tensor(uu, **f64)
        vv_t = torch.tensor(vv, **f64)
        hd, wd = len(vv), len(uu)
        Vw, Uw = torch.meshgrid(vv_t, uu_t, indexing="ij")
        Uw = Uw.reshape(-1)
        Vw = Vw.reshape(-1)
        if self.warp is not None:
            Mu, Mv, umid, vmid = self.warp
            du, dv = Uw - umid, Vw - vmid
            first_value, second_value = Mu[fidx], Mv[fidx]
            cu = first_value[0] * du + first_value[1] * dv + first_value[2]
            cv = second_value[0] * du + second_value[1] * dv + second_value[2]
            if first_value.shape[0] == 6:  # quadratic warp (solved as mm^2/50)
                q0, q1, q2 = du * du / 50.0, du * dv / 50.0, dv * dv / 50.0
                cu = cu + first_value[3] * q0 + first_value[4] * q1 + first_value[5] * q2
                cv = cv + second_value[3] * q0 + second_value[4] * q1 + second_value[5] * q2
            Uw = Uw - cu
            Vw = Vw - cv
        Xg, Yg = self._to_xy(Uw, Vw)
        Zg = self._height(Xg, Yg)
        points = torch.stack([Xg, Yg, Zg], 1)
        xc = (points - self.C[fidx]) @ self.R[fidx]
        zc = xc[:, 2]
        world_z = torch.clamp(zc, min=1e-6)
        xn = xc[:, :2] / world_z[:, None]
        fdist = 1.0 + self.k1 * (xn**2).sum(1)
        uvx = self.fxf * xn[:, 0] * fdist + self.cxf
        uvy = self.fxf * xn[:, 1] * fdist + self.cyf
        valid = (zc > 10) & (uvx >= 0) & (uvx < self.Wf - 1) & (uvy >= 0) & (uvy < self.Hf - 1)
        if not bool(valid.any()):
            return None
        uvx = uvx.clamp(0, self.Wf - 1)
        uvy = uvy.clamp(0, self.Hf - 1)
        feather = torch.minimum(
            torch.minimum(uvx, self.Wf - 1 - uvx), torch.minimum(uvy, self.Hf - 1 - uvy)
        )
        feather = (feather / (0.10 * min(self.Wf, self.Hf))).clamp(0, 1)
        r2c = ((uvx - self.cxf) / self.cxf) ** 2 + ((uvy - self.cyf) / self.cyf) ** 2
        center = torch.clamp(torch.exp(-self.bs * 0.5 * r2c), min=0.02)
        xi = ((Xg - self.xs0) / self.dxs).round().long().clamp(0, self.nx - 1)
        yi = ((Yg - self.ys0) / self.dys).round().long().clamp(0, self.ny - 1)
        sup = self.supg[yi, xi].to(torch.float64)
        vs = valid.to(torch.float64) * sup
        if self.max_inc > 0:
            gx = self._bil(self.gxg, Xg, Yg)
            gy = self._bil(self.gyg, Xg, Yg)
            nrm = torch.stack([-gx, -gy, torch.ones_like(gx)], 1) * self.up_sign
            nrm = nrm / nrm.norm(dim=1, keepdim=True)
            ray = points - self.C[fidx]
            ray = ray / torch.clamp(ray.norm(dim=1, keepdim=True), min=1e-9)
            cosi = (nrm * ray).sum(1).abs()
            import math as _m

            c_hi = _m.cos(_m.radians(max(self.max_inc - 15.0, 1.0)))
            c_lo = _m.cos(_m.radians(self.max_inc))
            inc = ((cosi - c_lo) / max(c_hi - c_lo, 1e-6)).clamp(0.0, 1.0)
            vs = vs * inc
        wgt = (feather * center * vs).float().reshape(hd, wd)
        soft = (feather * vs).float().reshape(hd, wd)
        mapx = uvx.float().reshape(hd, wd)
        mapy = uvy.float().reshape(hd, wd)
        return torch.stack([mapx, mapy, wgt, soft])


class GpuFramePipe:
    """Per-frame image pipeline on CUDA; holds no canvas-sized state between
    frames except the ownership pre-pass fields and, when they fit in VRAM,
    the texture canvases themselves (canvases_begin)."""

    def __init__(self, blend_mode: str, device: str = "cuda") -> None:
        self.dev = torch.device(device)
        self.blend_mode = blend_mode
        self.canvas = False
        self._kern: dict[tuple[int, float], TorchTensor] = {}  # (ksize, sigma) -> taps

    # -- VRAM-resident canvases ----------------------------------------------
    def canvas_fits(self, canvas_height: int, canvas_width: int) -> bool:
        """Would device canvases + per-frame transients fit comfortably?"""
        per_texel = 32 if self.blend_mode == "two-band" else 16
        free, _total = torch.cuda.mem_get_info(self.dev)
        return canvas_height * canvas_width * per_texel < 0.45 * free

    def canvases_begin(self, canvas_height: int, canvas_width: int) -> None:
        """Accumulate deposits on-device: kills the per-frame PCIe download
        and the CPU-side += (fp32 elementwise adds in the same frame order --
        results identical to the numpy accumulation)."""
        def zeros(*shape: int) -> TorchTensor:
            return cast(TorchTensor, torch.zeros(*shape, dtype=torch.float32, device=self.dev))

        world_z = zeros
        self.cacc = world_z(3, canvas_height, canvas_width)
        self.cwacc = world_z(canvas_height, canvas_width)
        if self.blend_mode == "two-band":
            self.chf = world_z(3, canvas_height, canvas_width)
            self.cwbest = world_z(canvas_height, canvas_width)
        self.canvas = True

    def canvases_take(self) -> tuple[FloatArray, NDArray[np.float32], FloatArray | None, NDArray[np.float32] | None]:
        """Download and free the device canvases.
        -> (acc (H,W,3), wacc, hf_best (H,W,3)|None, w_best|None) numpy."""
        acc = _tensor_numpy(self.cacc.permute(1, 2, 0).contiguous())
        wacc = _tensor_numpy(self.cwacc)
        hf = wb = None
        if self.blend_mode == "two-band":
            hf = _tensor_numpy(self.chf.permute(1, 2, 0).contiguous())
            wb = _tensor_numpy(self.cwbest)
            del self.chf, self.cwbest
        del self.cacc, self.cwacc
        self.canvas = False
        torch.cuda.empty_cache()
        return acc, wacc, hf, wb

    # -- op building blocks ---------------------------------------------------
    def _gauss1d(self, sigma: float) -> TorchTensor:
        """cv2.GaussianBlur(ksize=(0,0)) tap vector for float input."""
        ksize = int(round(sigma * 4 * 2 + 1)) | 1
        key = (ksize, float(sigma))
        if key not in self._kern:
            item_index = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F)[:, 0]
            self._kern[key] = torch.from_numpy(item_index).to(self.dev)
        return self._kern[key]

    def _blur(self, array: TorchTensor, sigma: float) -> TorchTensor:
        """Separable Gaussian, REFLECT_101 border, on (C,H,W) or (H,W)."""
        squeeze = array.dim() == 2
        if squeeze:
            array = array[None]
        item_index = self._gauss1d(sigma)
        camera_rotation = len(item_index) // 2
        candidate, image_height, image_width = array.shape
        if camera_rotation >= image_height or camera_rotation >= image_width:
            # torch reflect-pad needs pad < dim; tiny clipped edge tiles take
            # the exact cv2 path instead (identical taps, negligible size)
            arr = _tensor_numpy(array.permute(1, 2, 0))
            arr = cv2.GaussianBlur(arr, (0, 0), sigma)
            out = torch.from_numpy(arr.reshape(image_height, image_width, candidate)).to(self.dev)
            out = out.permute(2, 0, 1)
            return out[0] if squeeze else out
        array = array[None]  # (1,C,H,W)
        array = torch_functional.pad(array, (camera_rotation, camera_rotation, 0, 0), mode="reflect")
        array = torch_functional.conv2d(array, item_index.view(1, 1, 1, -1).expand(candidate, 1, 1, -1), groups=candidate)
        array = torch_functional.pad(array, (0, 0, camera_rotation, camera_rotation), mode="reflect")
        array = torch_functional.conv2d(array, item_index.view(1, 1, -1, 1).expand(candidate, 1, -1, 1), groups=candidate)
        array = array[0]
        return array[0] if squeeze else array

    def _up(self, fields: TorchTensor, image_height: int, image_width: int) -> TorchTensor:
        """Bilinear upsample (C,hd,wd) -> (C,h,w); cv2.INTER_LINEAR match."""
        if fields.shape[-2:] == (image_height, image_width):
            return fields
        return torch_functional.interpolate(fields[None], size=(image_height, image_width), mode="bilinear", align_corners=False)[0]

    def _area(self, array: TorchTensor, image_height: int, image_width: int) -> TorchTensor:
        """cv2.INTER_AREA-style box downsample on (C,H,W)."""
        return torch_functional.interpolate(array[None], size=(image_height, image_width), mode="area")[0]

    def decode(self, path: str, sscale: float) -> TorchTensor:
        """BGR float32 (3,H,W) on device, optionally INTER_AREA downscaled."""
        try:
            # nvJPEG decodes sensor pixels and never applies EXIF rotation --
            # exactly right here (the mount is fixed; the Canon's auto-rotate
            # tag flips with rx tilt and must be ignored)
            im = decode_jpeg(read_file(path), mode=ImageReadMode.RGB, device=self.dev)
        except Exception:  # non-JPEG / nvjpeg hiccup
            arr = cv2.imread(
                path,  # BGR HWC
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
            )
            im = torch.from_numpy(arr).to(self.dev).permute(2, 0, 1).flip(0)
        im = im.flip(0).float()  # RGB -> BGR, like imread
        if sscale < 0.999:
            h2 = int(round(im.shape[1] * sscale))
            w2 = int(round(im.shape[2] * sscale))
            im = self._area(im, h2, w2)
        return im

    def _sample(self, im: TorchTensor, mapx: TorchTensor, mapy: TorchTensor) -> TorchTensor:
        """cv2.remap(INTER_LINEAR, BORDER_CONSTANT=0) via grid_sample."""
        _channel_count, image_height, image_width = im.shape
        gx = (mapx + 0.5) * (2.0 / image_width) - 1.0
        gy = (mapy + 0.5) * (2.0 / image_height) - 1.0
        grid = torch.stack([gx, gy], -1)[None]
        return torch_functional.grid_sample(
            im[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )[0]

    # -- ownership pre-pass ----------------------------------------------------
    def ownership_begin(self, canvas_height: int, canvas_width: int) -> None:
        """Raises GpuOom when the three (H,W) fields don't fit (e.g. another
        process holds the GPU); the caller streams ownership on CPU instead."""
        free, _total = torch.cuda.mem_get_info(self.dev)
        if canvas_height * canvas_width * 12 > 0.6 * free:
            raise GpuOom(
                f"ownership fields need {canvas_height * canvas_width * 12 / 1e9:.1f} GB, {free / 1e9:.1f} GB free"
            )
        try:
            self._own_max = torch.zeros(canvas_height, canvas_width, device=self.dev)
            self._own_arg = torch.full((canvas_height, canvas_width), -1, dtype=torch.int32, device=self.dev)
            self._own_sum = torch.zeros(canvas_height, canvas_width, device=self.dev)
        except torch.OutOfMemoryError as caught_exception:
            for first_value in ("_own_max", "_own_arg", "_own_sum"):
                if hasattr(self, first_value):
                    delattr(self, first_value)
            torch.cuda.empty_cache()
            raise GpuOom(str(caught_exception)) from None
        self._own_gid = None

    def ownership_add(self, gid: int, rect: Rect, wgt_dec: NumericArray | TorchTensor) -> None:
        """Accumulate one frame's weight; frames MUST arrive sorted by group
        (ascending), so each group's sum completes before the merge -- ties
        then resolve to the lowest gid exactly like np.stack(...).argmax(0)."""
        u0, u1, v0, v1 = rect
        if self._own_gid is not None and gid != self._own_gid:
            self._ownership_merge()
        self._own_gid = gid
        try:
            if torch.is_tensor(wgt_dec):
                image_width = wgt_dec
            else:
                image_width = torch.from_numpy(np.ascontiguousarray(wgt_dec)).to(self.dev)
            self._own_sum[v0:v1, u0:u1] += self._up(cast(TorchTensor, image_width[None]), v1 - v0, u1 - u0)[0]
        except torch.OutOfMemoryError:
            # monster (warp-inflated) tile: upsample on CPU (same bilinear
            # convention) and stream row-bands into the running sum
            torch.cuda.empty_cache()
            wnp: NDArray[np.float32] = (
                _tensor_numpy(cast(TorchTensor, wgt_dec))
                if torch.is_tensor(wgt_dec)
                else cast(NDArray[np.float32], wgt_dec)
            )
            arr = cv2.resize(wnp, (u1 - u0, v1 - v0), interpolation=cv2.INTER_LINEAR)
            step = max(1, (1 << 26) // max(1, u1 - u0))
            for world_y in range(0, v1 - v0, step):
                band = torch.from_numpy(arr[world_y : world_y + step]).to(self.dev)
                self._own_sum[v0 + world_y : v0 + world_y + band.shape[0], u0:u1] += band

    def _ownership_merge(self) -> None:
        win = self._own_sum > self._own_max
        self._own_max = torch.where(win, self._own_sum, self._own_max)
        self._own_arg = torch.where(
            win, torch.tensor(self._own_gid, dtype=torch.int32, device=self.dev), self._own_arg
        )
        self._own_sum.zero_()

    def ownership_finish(self) -> NDArray[np.int32]:
        """-> best_g (H,W) int32 numpy; texels no group touched stay -1."""
        if self._own_gid is not None:
            self._ownership_merge()
        best = self._own_arg.cpu().numpy()
        del self._own_max, self._own_arg, self._own_sum
        torch.cuda.empty_cache()
        return best

    # -- per-frame pipeline ------------------------------------------------------
    def frame(
        self,
        path: str,
        rect: Rect,
        dec_fields: NumericArray | TorchTensor,
        *,
        sscale: float,
        img_pxmm: float,
        focus_weight: float,
        best_g: NDArray[np.integer[Any]] | None = None,
        gid: int | None = None,
        lf_gate: NDArray[np.uint8] | None = None,
        pixels_per_mm: float = 0.0,
        hf_coherence_mm: float = 0.0,
        gain: NumericArray | None = None,
    ) -> GpuFrameResult | None:
        """Compute one frame's deposit tiles. dec_fields = np (4,hd,wd) from
        geom(upsample=False); best_g / lf_gate are the FULL-canvas arrays
        (sliced here after the support crop). Returns None if the frame
        deposits nothing, else numpy tiles + the (possibly cropped) rect,
        ready for the caller's (CPU-path-identical) canvas accumulation:
          two-band: dict(rect, lo (h,w,3), soft (h,w), hf (h,w,3), wown (h,w))
          soft:     dict(rect, col (h,w,3), wgt2 (h,w))
        Raises GpuOom if the frame doesn't fit in VRAM after cache-flush retry.
        """
        try:
            return self._frame(
                path, rect, dec_fields,
                sscale=sscale, img_pxmm=img_pxmm, focus_weight=focus_weight,
                best_g=best_g, gid=gid, lf_gate=lf_gate,
                pixels_per_mm=pixels_per_mm, hf_coherence_mm=hf_coherence_mm, gain=gain,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            try:
                return self._frame(
                    path, rect, dec_fields,
                    sscale=sscale, img_pxmm=img_pxmm, focus_weight=focus_weight,
                    best_g=best_g, gid=gid, lf_gate=lf_gate,
                    pixels_per_mm=pixels_per_mm, hf_coherence_mm=hf_coherence_mm, gain=gain,
                )
            except torch.OutOfMemoryError as caught_exception:
                torch.cuda.empty_cache()
                raise GpuOom(str(caught_exception)) from None

    def _frame(
        self,
        path: str,
        rect: Rect,
        dec_fields: NumericArray | TorchTensor,
        *,
        sscale: float,
        img_pxmm: float,
        focus_weight: float,
        best_g: NDArray[np.integer[Any]] | None,
        gid: int | None,
        lf_gate: NDArray[np.uint8] | None,
        pixels_per_mm: float,
        hf_coherence_mm: float,
        gain: NumericArray | None = None,
    ) -> GpuFrameResult | None:
        u0, u1, v0, v1 = rect
        image_height, image_width = v1 - v0, u1 - u0
        if torch.is_tensor(dec_fields):
            threshold = dec_fields  # already on device (GpuGeom)
        else:
            threshold = torch.from_numpy(np.ascontiguousarray(dec_fields)).to(self.dev)
        f4 = self._up(cast(TorchTensor, threshold), image_height, image_width)
        del threshold
        # SUPPORT CROP: a deformable-warped footprint can project a tile far
        # larger than the frame's actual deposit (everything outside the
        # feather x incidence x support weight is zero; wgt2>0 implies
        # soft>0). Crop to the soft-support bbox + the HF-ownership blur
        # kernel radius, so the wown tails that can win texels are kept.
        # Only fires when it saves >40% area: an uncropped tile stays
        # bit-identical to the validated path.
        sup = f4[3] > 0
        rows = torch.where(sup.any(dim=1))[0]
        if len(rows) == 0:
            return None
        cols = torch.where(sup.any(dim=0))[0]
        marg = int(round(4.0 * hf_coherence_mm * pixels_per_mm + pixels_per_mm))
        y0c = max(int(rows[0]) - marg, 0)
        y1c = min(int(rows[-1]) + 1 + marg, image_height)
        x0c = max(int(cols[0]) - marg, 0)
        x1c = min(int(cols[-1]) + 1 + marg, image_width)
        del sup, rows, cols
        if (y1c - y0c) * (x1c - x0c) < 0.6 * image_height * image_width:
            f4 = f4[:, y0c:y1c, x0c:x1c].contiguous()
            u0, v0 = u0 + x0c, v0 + y0c
            image_height, image_width = y1c - y0c, x1c - x0c
            u1, v1 = u0 + image_width, v0 + image_height
        mapx, mapy, wgt2, soft = f4
        rect = (u0, u1, v0, v1)

        im = self.decode(path, sscale)
        if sscale < 0.999:
            mapx = mapx * sscale
            mapy = mapy * sscale
        col = self._sample(im, mapx, mapy)  # (3,h,w) BGR
        if gain is not None:
            # per-frame photometric gain (estimate_camera_frame_texture_gains): (3,) scalar per
            # channel, or (3,3) affine log-gain field [ch,(g0,gx,gy)] in
            # normalised image coords. Scales the whole frame, so lo and hf
            # inherit it consistently below.
            gt = torch.as_tensor(gain, dtype=col.dtype, device=col.device)
            if gt.ndim == 1:
                col = col * gt[:, None, None]
            else:
                gxh = (mapx / im.shape[-1] - 0.5).to(col.dtype)
                gyh = (mapy / im.shape[-2] - 0.5).to(col.dtype)
                col = col * torch.exp(
                    gt[:, 0, None, None] + gt[:, 1, None, None] * gxh + gt[:, 2, None, None] * gyh
                )

        if best_g is not None:
            gm = torch.from_numpy(np.ascontiguousarray(best_g[v0:v1, u0:u1] == gid))
            wgt2 = wgt2 * gm.to(self.dev).float()
            del gm
        if focus_weight > 0:
            pxmm_im = img_pxmm * sscale
            second_value, gauge, camera_rotation = im[0], im[1], im[2]
            gray = 0.114 * second_value + 0.587 * gauge + 0.299 * camera_rotation
            hf_im = gray - self._blur(gray, max(1.0, 0.06 * pxmm_im))
            en = self._blur(hf_im * hf_im, max(3.0, 1.5 * pxmm_im))
            del hf_im
            sharp_im = torch.sqrt(torch.clamp(en, min=0)) + 1e-3
            del en
            sharp = self._sample(sharp_im[None], mapx, mapy)[0]
            del sharp_im
            wgt2 = wgt2 * sharp**focus_weight
            del sharp
        del im, f4, mapx, mapy

        def npy(array: TorchTensor) -> NDArray[np.float32]:
            return _tensor_numpy(array)

        def npy_hwc(array: TorchTensor) -> FloatArray:
            return _tensor_numpy(array.permute(1, 2, 0).contiguous())

        if self.blend_mode != "two-band":
            if self.canvas:
                self.cacc[:, v0:v1, u0:u1] += col * wgt2[None]
                self.cwacc[v0:v1, u0:u1] += wgt2
                return cast(GpuFrameResult, {"rect": rect, "accumulated": True})
            return cast(GpuFrameResult, {"rect": rect, "col": npy_hwc(col), "wgt2": npy(wgt2)})

        mask = (soft > 0).float()
        sig = 2.0 * pixels_per_mm
        r_ = max(1, int(round(sig / 12.0)))
        if r_ > 1:
            sw, sh_ = max(1, image_width // r_), max(1, image_height // r_)
            lo = self._blur(self._area(col * mask[None], sh_, sw), sig / r_)
            ml = self._blur(self._area(mask[None], sh_, sw), sig / r_)
            lo = self._up(lo, image_height, image_width)
            ml = self._up(ml, image_height, image_width)[0]
        else:
            lo = self._blur(col * mask[None], sig)
            ml = self._blur(mask, sig)
        lo = lo / torch.clamp(ml, min=1e-6)[None]
        del ml
        hf = (col - lo) * mask[None]
        del col, mask
        if lf_gate is not None:
            gate = torch.from_numpy(np.ascontiguousarray(lf_gate[v0:v1, u0:u1]))
            soft = soft * (gate.to(self.dev).float() / 255.0)
            del gate
        wown = wgt2 if hf_coherence_mm <= 0 else self._blur(wgt2, hf_coherence_mm * pixels_per_mm)
        if self.canvas:
            self.cacc[:, v0:v1, u0:u1] += lo * soft[None]
            self.cwacc[v0:v1, u0:u1] += soft
            win = wown > self.cwbest[v0:v1, u0:u1]
            self.chf[:, v0:v1, u0:u1] = torch.where(win[None], hf, self.chf[:, v0:v1, u0:u1])
            self.cwbest[v0:v1, u0:u1] = torch.where(win, wown, self.cwbest[v0:v1, u0:u1])
            return cast(GpuFrameResult, {"rect": rect, "accumulated": True})
        out: GpuFrameResult = {"rect": rect, "lo": npy_hwc(lo), "soft": npy(soft)}
        del lo
        out["hf"] = npy_hwc(hf)
        del hf
        out["wown"] = npy(wown)
        return out
