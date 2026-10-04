"""Make repository-local packages and test support modules importable."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPORT_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "src" / "tests",
    PROJECT_ROOT / "src" / "pico",
    PROJECT_ROOT / "src" / "scripts" / "calibration",
    PROJECT_ROOT / "src" / "scripts" / "collision",
    PROJECT_ROOT / "third_party" / "pico",
)

for import_root in reversed(IMPORT_ROOTS):
    path = str(import_root)
    if path not in sys.path:
        sys.path.insert(0, path)
