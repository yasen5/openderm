"""Queued gantry homing controller."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from .jobs import MotionError, MotionJob
from .moonraker import MoonrakerError
from .state import AXES, MoveRecord

if TYPE_CHECKING:
    from .manager import MotionManager


class GantryHomingController:
    """Queue home jobs and monitor Moonraker until each axis is homed."""

    def __init__(self, manager: MotionManager) -> None:
        self.manager = manager

    async def enqueue(self, axes: tuple[str, ...]) -> MoveRecord:
        manager = self.manager
        manager.safety.require_emergency_reset()
        move_id = manager._next_move_id()
        job = MotionJob(
            move_id=move_id,
            kind="home",
            created_at=time.monotonic(),
            axes=axes,
        )
        record = MoveRecord(
            move_id=move_id,
            kind="home",
            status="queued",
            created_at=job.created_at,
            waiter=job.wait_event,
        )
        manager.jobs[move_id] = job
        manager.records[move_id] = record
        await manager.queue.put(job)
        await manager.state_store.update(queue_depth=manager.queue.qsize())
        return record

    async def wait_for_completion(
        self,
        axes: tuple[str, ...],
        home_task: asyncio.Task[dict[str, Any]],
    ) -> None:
        manager = self.manager
        deadline = time.monotonic() + manager.config.home_timeout_s
        revision = manager.state_store.get().revision
        axes_set = set(axes)
        last_refresh = 0.0
        while True:
            manager.safety.require_emergency_reset()
            if home_task.cancelled():
                raise MotionError("command_cancelled")
            if home_task.done():
                exception = home_task.exception()
                if exception is not None:
                    raise MotionError(str(exception))
            snapshot = manager.state_store.get()
            if axes_set.issubset(set(snapshot.homed_axes)):
                return
            if snapshot.fault is not None:
                raise MotionError(snapshot.fault)
            now = time.monotonic()
            if now - last_refresh >= 0.25:
                try:
                    await manager.moonraker.query_objects({"toolhead": None, "webhooks": None})
                except MoonrakerError as exc:
                    raise MotionError(str(exc)) from exc
                last_refresh = now
            if time.monotonic() >= deadline:
                raise MotionError("home_timeout")
            snapshot = await manager.state_store.wait_for_update(
                revision,
                timeout_s=0.1,
            )
            revision = snapshot.revision

    @staticmethod
    def build_gcode(axes: tuple[str, ...]) -> str:
        ordered_axes = [axis.upper() for axis in AXES if axis in axes]
        suffix = "" if not ordered_axes else " " + " ".join(ordered_axes)
        return f"G28{suffix}"
