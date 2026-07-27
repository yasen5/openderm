"""Runtime self-collision guard (numpy-only, control host).

Loads the precomputed collision envelope (``cad/collision_envelope.npz``) and answers
whether a commanded ``(x, y, z, rx)`` pose clears the static frame by the configured
margin. This is a pure table lookup -- it never calls FCL/trimesh at runtime, so it runs
in the control-host venv (numpy only). The envelope is generated offline by
``scripts/collision/build_collision_envelope.py`` from the CAD model. See ``docs/collision_guard.md``.

Units are REAL controller units: x, y, z in mm; rx in rad.
"""

from __future__ import annotations

import math
import os
import sysconfig
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_REPO_ENVELOPE = Path(_REPO) / "cad" / "collision_envelope.npz"
_INSTALLED_ENVELOPE = (
    Path(sysconfig.get_path("data")) / "share" / "openderm" / "cad" / "collision_envelope.npz"
)
DEFAULT_ENVELOPE = str(_REPO_ENVELOPE if _REPO_ENVELOPE.exists() else _INSTALLED_ENVELOPE)


class GantryCollisionError(RuntimeError):
    """A move was refused because the target pose is within the self-collision margin."""

    def __init__(self, x: float, y: float, z: float, rx: float, clearance: float, margin: float):
        self.pose = (x, y, z, rx)
        self.clearance = clearance
        self.margin = margin
        super().__init__(
            f"self-collision guard: pose x={x:.1f} y={y:.1f} z={z:.1f} mm rx={rx:.3f} rad "
            f"has clearance {clearance:.1f} mm < margin {margin:.1f} mm"
        )


@dataclass
class CollisionGuard:
    """Config-dependent self-collision limits from the precomputed envelope."""

    rx: np.ndarray  # fine rx grid (rad), ascending
    z: np.ndarray  # z grid (mm), ascending
    x: np.ndarray  # x grid (mm), ascending
    y: np.ndarray  # y grid (mm), ascending
    clearance: np.ndarray  # robust min-clearance (mm), shape (rx,z,x,y)
    margin: float  # mm; a pose is safe iff min_clearance >= margin
    backlash_deg: float

    def __post_init__(self):
        # Pre-erode: eroded[i,j,k,l] = MIN robust-clearance over the 2x2x2x2 corner block
        # of grid cell (i,j,k,l). Then min_clearance is a single O(1) index (grids are
        # regular -> direct arithmetic, no searchsorted or slice) and stays conservative.
        c = self.clearance
        c = np.minimum(c[:-1], c[1:])
        c = np.minimum(c[:, :-1], c[:, 1:])
        c = np.minimum(c[:, :, :-1], c[:, :, 1:])
        c = np.minimum(c[:, :, :, :-1], c[:, :, :, 1:])
        self._eroded = np.ascontiguousarray(c, dtype=np.float64)
        grids = (self.rx, self.z, self.x, self.y)
        self._g0 = tuple(float(g[0]) for g in grids)
        self._inv = tuple((len(g) - 1) / (float(g[-1]) - float(g[0])) for g in grids)
        self._last = tuple(len(g) - 1 for g in grids)  # last grid index (range check)
        self._cmax = tuple(len(g) - 2 for g in grids)  # max cell index

    # --- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path: str = DEFAULT_ENVELOPE, margin: float | None = None) -> "CollisionGuard":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"collision envelope not found: {path}. Generate it with "
                f"`build_collision_envelope.py` before running with the collision guard."
            )
        d = np.load(path)
        return cls(
            rx=d["rx"].astype(float),
            z=d["z"].astype(float),
            x=d["x"].astype(float),
            y=d["y"].astype(float),
            clearance=d["clearance_robust"].astype(float),
            margin=float(d["margin"]) if margin is None else float(margin),
            backlash_deg=float(d["backlash_deg"]),
        )

    # --- core lookup -------------------------------------------------------
    def min_clearance(self, x: float, y: float, z: float, rx: float) -> float:
        """Conservative clearance (mm): the MIN robust-clearance over the grid cell
        surrounding the pose (fail-safe against the coarse grid). Out of the swept ranges
        returns -inf (fail closed). O(1) index into the pre-eroded grid."""
        g0 = self._g0
        inv = self._inv
        last = self._last
        cm = self._cmax
        fr = (rx - g0[0]) * inv[0]
        fz = (z - g0[1]) * inv[1]
        fx = (x - g0[2]) * inv[2]
        fy = (y - g0[3]) * inv[3]
        if (
            fr < -1e-6
            or fr > last[0] + 1e-6
            or fz < -1e-6
            or fz > last[1] + 1e-6
            or fx < -1e-6
            or fx > last[2] + 1e-6
            or fy < -1e-6
            or fy > last[3] + 1e-6
        ):
            return -math.inf
        ir = int(fr)
        ir = 0 if ir < 0 else cm[0] if ir > cm[0] else ir
        iz = int(fz)
        iz = 0 if iz < 0 else cm[1] if iz > cm[1] else iz
        ix = int(fx)
        ix = 0 if ix < 0 else cm[2] if ix > cm[2] else ix
        iy = int(fy)
        iy = 0 if iy < 0 else cm[3] if iy > cm[3] else iy
        return float(self._eroded[ir, iz, ix, iy])

    def is_safe(self, x: float, y: float, z: float, rx: float) -> bool:
        return self.min_clearance(x, y, z, rx) >= self.margin

    def check_pose(self, x: float, y: float, z: float, rx: float) -> None:
        """Raise GantryCollisionError if the pose is unsafe (for discrete moves)."""
        c = self.min_clearance(x, y, z, rx)
        if c < self.margin:
            raise GantryCollisionError(x, y, z, rx, c, self.margin)

    # --- paths / transients ------------------------------------------------
    def check_path(self, a: tuple, b: tuple, n: int = 8):
        """Check the swept path A->B (each a (x,y,z,rx) tuple). Also checks the
        rx-leads-xyz transient corner, since rx is dispatched fire-and-forget before the
        gantry move. Returns (ok, worst_pose, worst_clearance)."""
        poses = [
            tuple(a[k] + (b[k] - a[k]) * t for k in range(4))
            for t in np.linspace(0.0, 1.0, max(2, n))
        ]
        poses.append((a[0], a[1], a[2], b[3]))  # rx at target, x/y/z still at start
        worst_c, worst_p = math.inf, None
        for p in poses:
            c = self.min_clearance(*p)
            if c < worst_c:
                worst_c, worst_p = c, p
        return worst_c >= self.margin, worst_p, worst_c

    def check_move(self, target: tuple, current: tuple | None = None, n: int = 8) -> None:
        """Raise GantryCollisionError if the target pose (or the path to it, if `current`
        is given) is unsafe."""
        if current is not None:
            ok, wp, wc = self.check_path(current, target, n)
            if not ok:
                raise GantryCollisionError(*wp, wc, self.margin)
        self.check_pose(*target)
