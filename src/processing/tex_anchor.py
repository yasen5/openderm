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

import cv2
import numpy as np


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
    base_R: np.ndarray  # world -> gantry: Pg = (P_world - base_t) @ base_R
    base_t: np.ndarray
    fx_fullres_px: float

    def __getattr__(self, field_name):
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
    def px_to_uv(self, pixel_x, pixel_y):
        return (
            self.umin + (np.asarray(pixel_x) + 0.5) / self.pixels_per_mm,
            self.vmin + (np.asarray(pixel_y) + 0.5) / self.pixels_per_mm,
        )

    def uv_to_px(self, texture_u, texture_v):
        return (
            (np.asarray(texture_u) - self.umin) * self.pixels_per_mm - 0.5,
            (np.asarray(texture_v) - self.vmin) * self.pixels_per_mm - 0.5,
        )


def _load_placements(reg_dir):
    with open(os.path.join(reg_dir, "placements3d.json")) as fh:
        return json.load(fh)


def _parse_obj_v_vt(obj_path):
    """Return (V[N,3] world xyz, VT[N,2] normalized uv) paired by face indices.

    OBJ faces reference v/vt indices that need not be 1:1, so we pair them
    through the faces (the renderer emits one vt per vertex but pairing via
    faces is robust to any ordering)."""
    verts, texs, pairs = [], [], {}
    with open(obj_path) as fh:
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
    verts = np.array(verts)
    texs = np.array(texs)
    vi = np.array(sorted(pairs))
    ti = np.array([pairs[item_index] for item_index in vi])
    return verts[vi], texs[ti]


def load_gauge(reg_dir) -> Gauge:
    pl = _load_placements(reg_dir)
    rm = pl["rig_model"]
    base_R = np.array(rm["base_R"])
    base_t = np.array(rm["base_t"])
    fx = float(rm["fx_fullres_px"])

    tx = pl.get("texture")
    if not isinstance(tx, dict):
        raise ValueError(
            f"{os.path.join(reg_dir, 'placements3d.json')} is missing required "
            "texture metadata; rebuild the scan with openderm-process."
        )
    required = ("umin_mm", "vmin_mm", "umax_mm", "vmax_mm", "ppmm", "W", "H")
    missing = [name for name in required if name not in tx]
    if missing:
        raise ValueError("texture metadata is missing required fields: " + ", ".join(missing))
    return Gauge(
        tx["umin_mm"],
        tx["vmin_mm"],
        tx["umax_mm"],
        tx["vmax_mm"],
        tx["ppmm"],
        int(tx["W"]),
        int(tx["H"]),
        base_R,
        base_t,
        fx,
    )


def coverage_mask(reg_dir) -> np.ndarray:
    """Authoritative covered-texel mask (uint8 0/255), shape (H, W)."""
    cov_p = os.path.join(reg_dir, "coverage.png")
    if not os.path.exists(cov_p):
        raise FileNotFoundError(f"{cov_p} is missing; rebuild the scan with openderm-process.")
    mask = cv2.imread(cov_p, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read coverage mask: {cov_p}")
    return (mask > 127).astype(np.uint8) * 255
