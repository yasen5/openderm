"""Rotate the RX-axis motor while holding the viewed point fixed (RX pivot controller).

The rig is 4-DOF: an X/Y/Z gantry carrying a camera on a single RX rotational axis whose
axis is *nominally* parallel to X but in practice slightly tilted. When you rotate
rx the camera swings, so the point it looks at moves. To rotate the camera in
place you must move the gantry along the arc the camera centre traces about the
rx axis.

Geometry that makes this trivial to calibrate: the vector from the gantry
toolhead to the fixed viewed point is rigid and rotates with rx about a fixed
axis. Rotating a fixed vector about a fixed axis makes every gantry coordinate an
affine harmonic in the angle::

    x(rx) = ax + bx*cos(rx) + cx*sin(rx)
    y(rx) = ay + by*cos(rx) + cy*sin(rx)
    z(rx) = az + bz*cos(rx) + cz*sin(rx)

The (b, c) pair for each axis is the rig's arc shape (axis direction + radius);
the constant a only selects *which* point is viewed. So to pivot about whatever
point is currently under the camera we only need the harmonic amplitudes (b, c),
and the move is taken *relative* to the current pose -- the constants cancel::

    dq = b*(cos rx_target - cos rx_ref) + c*(sin rx_target - sin rx_ref)

This holds for x, y AND z, so a coordinated move keeps the standoff distance
constant along the whole arc (no separate z hunting needed, though the sensor
standoff loop can still trim residual surface tilt on top).

Calibrate the amplitudes from a handful of poses that all look at the *same*
fixed point at a *constant* standoff. Collect them with
``src/scripts/calibration/rx_pivot_capture.py`` and fit them with
``src/scripts/calibration/rx_pivot_fit.py``. Six well-spread poses should
cross-validate to less than 1 mm.

Evaluating the model (``predict`` / ``pivot``) is pure-Python so it runs on the
control host with no numpy; only ``fit_records`` needs numpy and is run offline.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


AXES: tuple[str, str, str] = ("x", "y", "z")


@dataclass(frozen=True)
class HarmonicFit:
    """One axis' coefficients: value(rx) = a + b*cos(rx) + c*sin(rx)."""

    a: float
    b: float
    c: float

    def value(self, rx_rad: float) -> float:
        return self.a + self.b * math.cos(rx_rad) + self.c * math.sin(rx_rad)

    @property
    def radius(self) -> float:
        """Arc radius this axis contributes (amplitude of the harmonic)."""
        return math.hypot(self.b, self.c)


@dataclass(frozen=True)
class RxPivotModel:
    """Calibrated rx-pivot arc for the gantry.

    ``axis_coeffs`` maps each gantry axis to its :class:`HarmonicFit`. Build one
    with :meth:`fit_records` (offline, needs numpy) or :meth:`load`.
    """

    axis_coeffs: dict[str, HarmonicFit]
    metadata: dict[str, Any]

    # -- evaluation (pure-Python, safe on the control host) -----------------

    def predict(self, rx_rad: float) -> dict[str, float]:
        """Absolute gantry pose that re-views the *calibration* point at ``rx``.

        Useful for replaying the calibration arc. To pivot about an arbitrary
        point currently in view, use :meth:`pivot` instead.
        """
        return {axis: self.axis_coeffs[axis].value(rx_rad) for axis in self.axis_coeffs}

    def pivot(
        self,
        ref_pose: Mapping[str, float],
        rx_target_rad: float,
        rx_ref_rad: float,
    ) -> dict[str, float]:
        """Gantry pose that keeps the currently-viewed point fixed at ``rx_target``.

        ``ref_pose`` is the current gantry pose (keys ``x``/``y``/``z``, mm) that
        is looking at the point of interest at ``rx_ref_rad``. The returned pose
        differs only by the arc displacement, so the camera ends up at
        ``rx_target_rad`` still looking at the same point. The constant term of
        the fit cancels, so this works for *any* point, not just the calibrated
        one.
        """
        dcos = math.cos(rx_target_rad) - math.cos(rx_ref_rad)
        dsin = math.sin(rx_target_rad) - math.sin(rx_ref_rad)
        out: dict[str, float] = {}
        for axis, fit in self.axis_coeffs.items():
            if axis not in ref_pose:
                continue
            out[axis] = float(ref_pose[axis]) + fit.b * dcos + fit.c * dsin
        return out

    def arc_radius_mm(self) -> float:
        """Radius of the camera-centre orbit about the rx axis (mm).

        Each axis harmonic is a projection of the same 3D circle, so the squared
        amplitudes sum to 2 R^2; hence the 1/sqrt(2). This is the distance from
        the rx rotation axis to the viewed point.
        """
        sum_sq = sum(fit.b**2 + fit.c**2 for fit in self.axis_coeffs.values())
        return math.sqrt(sum_sq / 2.0)

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "rx_pivot_harmonic_v1",
            "axis_coeffs": {
                axis: {"a": fit.a, "b": fit.b, "c": fit.c} for axis, fit in self.axis_coeffs.items()
            },
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RxPivotModel":
        coeffs = {
            axis: HarmonicFit(float(c["a"]), float(c["b"]), float(c["c"]))
            for axis, c in data["axis_coeffs"].items()
        }
        return cls(axis_coeffs=coeffs, metadata=dict(data.get("metadata", {})))

    @classmethod
    def load(cls, path: str | Path) -> "RxPivotModel":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # -- calibration (offline; needs numpy) ---------------------------------

    @classmethod
    def fit_records(
        cls,
        records: Sequence[Mapping[str, float]],
        *,
        source: str | None = None,
    ) -> tuple["RxPivotModel", dict[str, Any]]:
        """Least-squares fit the arc from poses that all view the same point.

        Each record needs ``rx_rad`` and ``x_mm``/``y_mm``/``z_mm``. Returns the
        model plus a diagnostics dict (per-axis fit residual and leave-one-out
        prediction error, both in mm). Requires >= 3 distinct rx angles.
        """
        import numpy as np  # local import: only calibration needs numpy

        rx = np.array([float(r["rx_rad"]) for r in records], float)
        coords = {
            "x": np.array([float(r["x_mm"]) for r in records], float),
            "y": np.array([float(r["y_mm"]) for r in records], float),
            "z": np.array([float(r["z_mm"]) for r in records], float),
        }
        n = len(rx)
        if n < 3 or len(np.unique(np.round(rx, 4))) < 3:
            raise ValueError("need >= 3 poses at >= 3 distinct rx angles to fit the arc")

        def design(angles: "np.ndarray") -> "np.ndarray":
            return np.stack([np.ones_like(angles), np.cos(angles), np.sin(angles)], 1)

        coeffs: dict[str, HarmonicFit] = {}
        fit_resid: dict[str, float] = {}
        for axis, vals in coords.items():
            sol, *_ = np.linalg.lstsq(design(rx), vals, rcond=None)
            coeffs[axis] = HarmonicFit(float(sol[0]), float(sol[1]), float(sol[2]))
            resid = design(rx) @ sol - vals
            fit_resid[axis] = float(np.sqrt(np.mean(resid**2)))

        # Leave-one-out: honest accuracy when predicting an unseen rx.
        loo: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        loo_max: dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        for axis, vals in coords.items():
            errs = []
            for k in range(n):
                keep = [i for i in range(n) if i != k]
                sol, *_ = np.linalg.lstsq(design(rx[keep]), vals[keep], rcond=None)
                pred = float((design(rx[k : k + 1]) @ sol)[0])
                errs.append(abs(pred - float(vals[k])))
            loo[axis] = float(np.mean(errs))
            loo_max[axis] = float(np.max(errs))

        metadata: dict[str, Any] = {
            "n_poses": int(n),
            "rx_min_rad": float(rx.min()),
            "rx_max_rad": float(rx.max()),
            "rx_min_deg": float(np.degrees(rx.min())),
            "rx_max_deg": float(np.degrees(rx.max())),
            "fit_residual_rms_mm": fit_resid,
            "loo_mean_err_mm": loo,
            "loo_max_err_mm": loo_max,
        }
        if source is not None:
            metadata["source"] = source
        model = cls(axis_coeffs=coeffs, metadata=metadata)
        diagnostics = {
            "fit_residual_rms_mm": fit_resid,
            "loo_mean_err_mm": loo,
            "loo_max_err_mm": loo_max,
            "arc_radius_mm": model.arc_radius_mm(),
        }
        return model, diagnostics


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Read pose records from a captures/rx_pivot_poses_*.jsonl file.

    Keeps only lines that have a numeric rx and all three gantry coordinates.
    """
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if any(row.get(k) is None for k in ("rx_rad", "x_mm", "y_mm", "z_mm")):
            continue
        records.append(row)
    return records
