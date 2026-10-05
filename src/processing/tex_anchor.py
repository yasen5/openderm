"""Texture-map gauge for cross-scan registration.

A single scan's ortho-texture (register_scan_3d.py) is LINEAR in its own axes:
    pixel (i, j)  <->  (u_mm, v_mm) = (umin + i/ppmm, vmin + j/ppmm)
where u = gantry x and v = surface arc length along gantry y, both expressed
in the gantry/proprioception frame shared across scans (the rig gauge base_R,
base_t is byte-identical between scans rendered with the same --rig-from). So
two scans of the same site land in the SAME (u, v) mm frame up to a small
residual deformation -- the precondition for longitudinal lesion tracking.

The gauge is read from the ``texture`` block in ``placements3d.json``. The
authoritative covered-texel mask is read from ``coverage.png``. Both files are
written by ``openderm-process``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TypedDict, cast

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]


class RigModel(TypedDict):
    """Rig calibration values read from placements3d.json."""

    base_R: list[list[float]]
    base_t: list[float]
    fx_fullres_px: float


class RawTextureMetadata(TypedDict, total=False):
    """Texture calibration values before required fields have been checked."""

    umin_mm: float
    vmin_mm: float
    umax_mm: float
    vmax_mm: float
    ppmm: float
    W: int
    H: int


class TextureMetadata(TypedDict):
    """Complete texture calibration values from placements3d.json."""

    umin_mm: float
    vmin_mm: float
    umax_mm: float
    vmax_mm: float
    ppmm: float
    W: int
    H: int


class Placements3D(TypedDict):
    rig_model: RigModel
    texture: RawTextureMetadata


@dataclass
class Gauge:
    """pixel <-> (u, v) mm for one scan's texture, plus its gantry gauge."""

    umin: float
    vmin: float
    umax: float
    vmax: float
    pixels_per_mm: float
    image_width: int
    image_height: int
    base_R: FloatArray  # world -> gantry: Pg = (P_world - base_t) @ base_R
    base_t: FloatArray
    fx_fullres_px: float

    def __getattr__(self, field_name: str) -> float | int:
        # Preserve the established placements/texture gauge accessors.
        legacy_field_aliases = {
            "ppmm": "pixels_per_mm",
            "W": "image_width",
            "H": "image_height",
        }
        if field_name in legacy_field_aliases:
            return getattr(self, legacy_field_aliases[field_name])
        raise AttributeError(field_name)

    # --- linear pixel/mm maps (the texture image is linear in (u_mm, v_mm)) ---
    def px_to_uv(
        self, pixel_x: ArrayLike, pixel_y: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        return (
            self.umin + (np.asarray(pixel_x, dtype=np.float64) + 0.5) / self.pixels_per_mm,
            self.vmin + (np.asarray(pixel_y, dtype=np.float64) + 0.5) / self.pixels_per_mm,
        )

    def uv_to_px(
        self, texture_u: ArrayLike, texture_v: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        return (
            (np.asarray(texture_u, dtype=np.float64) - self.umin) * self.pixels_per_mm - 0.5,
            (np.asarray(texture_v, dtype=np.float64) - self.vmin) * self.pixels_per_mm - 0.5,
        )


def _load_placements(reg_dir: str) -> Placements3D:
    with open(os.path.join(reg_dir, "placements3d.json"), encoding="utf-8") as fh:
        return cast(Placements3D, json.load(fh))


def _parse_obj_v_vt(obj_path: str) -> tuple[FloatArray, FloatArray]:
    """Return (V[N,3] world xyz, VT[N,2] normalized uv) paired by face indices.

    OBJ faces reference v/vt indices that need not be 1:1, so we pair them
    through the faces (the renderer emits one vt per vertex but pairing via
    faces is robust to any ordering)."""
    verts: list[list[float]] = []
    texs: list[list[float]] = []
    pairs: dict[int, int] = {}
    with open(obj_path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith("v "):
                verts.append([float(world_x) for world_x in ln.split()[1:4]])
            elif ln.startswith("vt "):
                texs.append([float(world_x) for world_x in ln.split()[1:3]])
            elif ln.startswith("f "):
                for tok in ln.split()[1:]:
                    first_value = tok.split("/")
                    vi = int(first_value[0]) - 1
                    ti = int(first_value[1]) - 1 if len(first_value) > 1 and first_value[1] else vi
                    pairs[vi] = ti
    vertex_array: FloatArray = np.asarray(verts, dtype=np.float64)
    texture_array: FloatArray = np.asarray(texs, dtype=np.float64)
    vertex_indices = np.asarray(sorted(pairs), dtype=np.intp)
    texture_indices = np.asarray([pairs[int(index)] for index in vertex_indices], dtype=np.intp)
    return vertex_array[vertex_indices], texture_array[texture_indices]


def load_gauge(reg_dir: str) -> Gauge:
    pl = _load_placements(reg_dir)
    rm = pl["rig_model"]
    base_R: FloatArray = np.asarray(rm["base_R"], dtype=np.float64)
    base_t: FloatArray = np.asarray(rm["base_t"], dtype=np.float64)
    fx = float(rm["fx_fullres_px"])

    tx: object = pl.get("texture")
    if not isinstance(tx, dict):
        raise ValueError(
            f"{os.path.join(reg_dir, 'placements3d.json')} is missing required "
            "texture metadata; rebuild the scan with openderm-process."
        )
    required = ("umin_mm", "vmin_mm", "umax_mm", "vmax_mm", "ppmm", "W", "H")
    missing = [name for name in required if name not in tx]
    if missing:
        raise ValueError("texture metadata is missing required fields: " + ", ".join(missing))
    texture_metadata = cast(TextureMetadata, tx)
    return Gauge(
        texture_metadata["umin_mm"],
        texture_metadata["vmin_mm"],
        texture_metadata["umax_mm"],
        texture_metadata["vmax_mm"],
        texture_metadata["ppmm"],
        int(texture_metadata["W"]),
        int(texture_metadata["H"]),
        base_R,
        base_t,
        fx,
    )


def coverage_mask(reg_dir: str) -> UInt8Array:
    """Authoritative covered-texel mask (uint8 0/255), shape (H, W)."""
    cov_p = os.path.join(reg_dir, "coverage.png")
    if not os.path.exists(cov_p):
        raise FileNotFoundError(f"{cov_p} is missing; rebuild the scan with openderm-process.")
    mask = cast(UInt8Array | None, cv2.imread(cov_p, cv2.IMREAD_GRAYSCALE))
    if mask is None:
        raise ValueError(f"Could not read coverage mask: {cov_p}")
    return ((mask > 127).astype(np.uint8) * 255).astype(np.uint8)
