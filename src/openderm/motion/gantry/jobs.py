"""Motion queue jobs and coordinator errors."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


class MotionError(RuntimeError):
    """Raised when the coordinator rejects or fails a command."""


@dataclass
class MotionJob:
    move_id: str
    kind: str
    created_at: float
    wait_event: asyncio.Event = field(default_factory=asyncio.Event)
    target: dict[str, float] | None = None
    feed_mm_s: float | None = None
    tolerance_mm: float = 0.05
    axes: tuple[str, ...] = ()
    stop_mode: str | None = None
    commander_id: str | None = None
    result: dict[str, Any] | None = None
