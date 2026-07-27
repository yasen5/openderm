"""Live setpoint streaming controller for coordinated gantry motion."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from ...config import linear_axis_limits
from .jobs import MotionError
from .state import AXES, project_position

if TYPE_CHECKING:
    from .manager import MotionManager


class GantryStreamingController:
    """Own the live setpoint and emit non-blocking G-code updates."""

    def __init__(self, manager: MotionManager) -> None:
        self.manager = manager

    @property
    def active(self) -> bool:
        return self.manager._streaming

    async def start(
        self,
        *,
        feed_mm_s: float | None = None,
        tick_s: float = 0.05,
        min_step_mm: float = 0.01,
    ) -> dict[str, Any]:
        manager = self.manager
        manager.safety.require_emergency_reset()
        if manager._streaming:
            raise MotionError("stream_already_active")
        if manager._active_job is not None or not manager.queue.empty():
            raise MotionError("busy_blocking_move")
        snapshot = manager.state_store.get()
        if snapshot.fault is not None:
            raise MotionError(snapshot.fault)
        if snapshot.stale:
            raise MotionError("stale_state")
        if tick_s <= 0:
            raise MotionError("invalid_tick")
        await manager.safety.clear_soft_stop()
        resolved_feed = feed_mm_s or (manager.config.default_feed_mm_min / 60.0)
        hold = dict(project_position(snapshot))
        manager._stream_target = hold
        manager._streaming = True
        await manager.state_store.update(streaming=True)
        manager._stream_task = asyncio.create_task(
            self.run(
                feed_mm_s=resolved_feed,
                tick_s=tick_s,
                min_step_mm=max(0.0, min_step_mm),
                initial_last_sent=dict(hold),
            ),
            name="gantry-stream-worker",
        )
        return {
            "streaming": True,
            "feed_mm_s": resolved_feed,
            "tick_s": tick_s,
        }

    async def update_target(
        self,
        target: dict[str, float],
    ) -> dict[str, float]:
        manager = self.manager
        if not manager._streaming:
            raise MotionError("stream_not_active")
        resolved = self.resolve_target(target)
        manager._stream_target = {
            **(manager._stream_target or {}),
            **resolved,
        }
        return dict(manager._stream_target)

    async def stop(self) -> dict[str, Any]:
        manager = self.manager
        was_streaming = manager._streaming
        manager._streaming = False
        if manager._stream_task is not None:
            manager._stream_task.cancel()
            try:
                await manager._stream_task
            except asyncio.CancelledError:
                pass
            manager._stream_task = None
        manager._stream_target = None
        if was_streaming:
            await manager.state_store.update(
                streaming=False,
                is_moving=False,
                velocity_mm_s=0.0,
            )
        return {"streaming": False}

    def resolve_target(
        self,
        target: dict[str, float],
    ) -> dict[str, float]:
        manager = self.manager
        snapshot = manager.state_store.get()
        if snapshot.fault is not None:
            raise MotionError(snapshot.fault)
        resolved: dict[str, float] = {}
        for axis, value in target.items():
            if axis not in AXES:
                raise MotionError(f"unknown_axis:{axis}")
            if axis not in snapshot.homed_axes:
                raise MotionError("requires_homing")
            minimum, maximum = linear_axis_limits(axis)
            if axis == manager.config.axis:
                minimum = manager.config.travel_min_mm
                maximum = manager.config.travel_max_mm
            resolved[axis] = min(max(float(value), minimum), maximum)
        return resolved

    async def run(
        self,
        *,
        feed_mm_s: float,
        tick_s: float,
        min_step_mm: float,
        initial_last_sent: dict[str, float] | None = None,
    ) -> None:
        manager = self.manager
        last_sent = dict(initial_last_sent) if initial_last_sent else None
        try:
            while manager._streaming:
                snapshot = manager.state_store.get()
                if snapshot.fault is not None:
                    await manager.state_store.update(
                        streaming=False,
                        is_moving=False,
                        velocity_mm_s=0.0,
                    )
                    manager._streaming = False
                    break
                target = manager._stream_target
                if target is not None and (
                    last_sent is None
                    or self.target_changed(
                        target,
                        last_sent,
                        min_step_mm,
                    )
                ):
                    await manager.moonraker.gcode_script(
                        manager._build_move_gcode(target, feed_mm_s)
                    )
                    last_sent = dict(target)
                    await manager.state_store.update(
                        last_commanded_target=dict(target),
                        is_moving=True,
                        velocity_mm_s=feed_mm_s,
                        timestamp=time.monotonic(),
                    )
                await asyncio.sleep(tick_s)
        except asyncio.CancelledError:
            raise
        finally:
            manager._streaming = False

    @staticmethod
    def target_changed(
        target: dict[str, float],
        last_sent: dict[str, float],
        min_step_mm: float,
    ) -> bool:
        return any(
            axis not in last_sent or abs(value - last_sent[axis]) >= min_step_mm
            for axis, value in target.items()
        )
