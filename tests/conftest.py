"""Make repository-local packages and test support modules importable."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "pico",
    PROJECT_ROOT / "scripts" / "calibration",
    PROJECT_ROOT / "scripts" / "collision",
)

for import_root in reversed(IMPORT_ROOTS):
    path = str(import_root)
    if path not in sys.path:
        sys.path.insert(0, path)
