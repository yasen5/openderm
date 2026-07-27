from __future__ import annotations

"""Moonraker protocol errors and G-code formatting helpers."""


class MoonrakerError(RuntimeError):
    """Raised when the Moonraker API returns an error or invalid data."""


def format_gcode_value(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")
