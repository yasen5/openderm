"""Persistent settings for repository calibration and collision utilities.

The CLI supplies only run-specific inputs and output paths. All other settings
come from config/scripts.json, resolved relative to this checkout, not the CWD.
Missing or misspelled settings are errors; there are no fallback settings here.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import get_type_hints


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "scripts.json"


@dataclass(frozen=True)
class CalibrationConfig:
    gantry_server_url: str
    rx_server_url: str
    pico_port: str
    target_mm: float
    gain_mm_per_mm: float
    max_step_mm: float
    search_step_mm: float
    deadband_mm: float
    period_s: float
    samples: int


@dataclass(frozen=True)
class PivotCaptureConfig:
    degrees: bool
    jog_step_mm: float
    report_interval_s: float
    y_pico_vmax_mm_s: float
    z_pico_vmax_mm_s: float
    home_y: bool
    home_z: bool


@dataclass(frozen=True)
class FloorTareConfig:
    home_z: bool
    settle_iters: int
    tare_samples: int
    max_travel_mm: float
    timeout_s: float
    rx_speed_rad_s: float
    rx_settle_tol_rad: float
    debug: bool


@dataclass(frozen=True)
class CollisionConfig:
    margin_mm: float
    backlash_deg: float
    rx_points: int
    z_points: int
    x_points: int
    y_points: int
    fine_rx_points: int


_SECTIONS = {
    "calibration": CalibrationConfig,
    "rx_pivot_capture": PivotCaptureConfig,
    "floor_depth_tare": FloorTareConfig,
    "collision": CollisionConfig,
}
_ALLOW_ZERO = {"deadband_mm", "period_s", "margin_mm", "backlash_deg"}


def _section(document: dict, name: str):
    cls = _SECTIONS[name]
    values = document.get(name)
    if not isinstance(values, dict):
        raise ValueError(f"{CONFIG_PATH}: missing or invalid {name!r} section")
    expected = {field.name for field in fields(cls)}
    if missing := expected - values.keys():
        raise ValueError(f"{CONFIG_PATH}: {name}: missing settings: {', '.join(sorted(missing))}")
    if unknown := values.keys() - expected:
        raise ValueError(f"{CONFIG_PATH}: {name}: unknown settings: {', '.join(sorted(unknown))}")
    for key, annotation in get_type_hints(cls).items():
        value = values[key]
        valid = type(value) is annotation or (annotation is float and type(value) is int)
        label = f"{CONFIG_PATH}: {name}.{key}"
        if not valid:
            raise ValueError(f"{label}: expected {annotation}, got {value!r}")
        if type(value) in (int, float):
            allow_zero = key in _ALLOW_ZERO
            if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
                bound = "nonnegative" if allow_zero else "positive"
                raise ValueError(f"{label}: must be finite and {bound}")
            if key.endswith("_points") and value < 2:
                raise ValueError(f"{label}: needs at least 2 grid points")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{label}: must not be empty")
    if name == "floor_depth_tare" and values["tare_samples"] < 3:
        raise ValueError(f"{CONFIG_PATH}: floor_depth_tare.tare_samples: must be at least 3")
    return cls(**values)


def load_script_config(name: str) -> dict:
    """Load a utility's settings, including shared settings for calibration."""
    try:
        document = json.loads(CONFIG_PATH.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read script configuration {CONFIG_PATH}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{CONFIG_PATH}: expected an object of configuration sections")
    if unknown := document.keys() - _SECTIONS.keys():
        raise ValueError(f"{CONFIG_PATH}: unknown sections: {', '.join(sorted(unknown))}")
    settings = {}
    if name in ("rx_pivot_capture", "floor_depth_tare"):
        settings.update(asdict(_section(document, "calibration")))
    settings.update(asdict(_section(document, name)))
    return settings


def calibration_options(name: str, args: argparse.Namespace) -> argparse.Namespace:
    """Combine required run inputs with validated persistent settings."""
    return argparse.Namespace(**load_script_config(name), **vars(args))
