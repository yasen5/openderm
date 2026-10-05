"""Emergency-stop, dead-man, and move safety monitoring."""

from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING, Any

from .jobs import MotionError, MotionJob
from .moonraker import MoonrakerError
from .state import StateSnapshot, project_position

if TYPE_CHECKING:
    from .manager import MotionManager


class GantrySafetyController:
    """Own stop latches and monitor active commands for safety failures."""

    def __init__(self, manager: MotionManager) -> None:
        self.manager = manager

    async def request_soft_stop(self) -> dict[str, Any]:
        manager = self.manager
        self.require_emergency_reset()
        await manager.streaming.stop()
        await manager.state_store.update(stop_requested=True)
        cancelled = self.cancel_queued(reason="soft_stop")
        await manager.state_store.update(queue_depth=manager.queue.qsize())
        snapshot = manager.state_store.get()
        return {
            "stopped": "soft",
            "cancelled_move_ids": cancelled,
            "active_move_id": snapshot.active_move_id,
        }

    async def request_emergency_stop(
        self,
        *,
        reason: str = "emergency_stop",
    ) -> dict[str, Any]:
        manager = self.manager
        async with manager._emergency_lock:
            already_latched = manager._emergency_latched
            manager._emergency_latched = True
            manager._emergency_generation += 1
            active_move_id = None if manager._active_job is None else manager._active_job.move_id
            cancelled = self.cancel_all_queued(reason=reason)
            await manager.state_store.update(
                fault=reason,
                stop_requested=True,
                is_moving=False,
                streaming=False,
                velocity_mm_s=0.0,
                queue_depth=manager.queue.qsize(),
                emergency_latched=True,
            )
            await self.cancel_active_command_task()
            await manager.streaming.stop()

            try:
                await manager.moonraker.emergency_stop()
            except Exception as exc:
                await manager.state_store.update(
                    fault=f"{reason}_delivery_failed",
                    is_moving=False,
                )
                raise MotionError(f"{reason}_delivery_failed: {exc}") from exc

            await manager.state_store.update(
                fault=reason,
                stop_requested=True,
                is_moving=False,
                streaming=False,
                velocity_mm_s=0.0,
                queue_depth=manager.queue.qsize(),
                emergency_latched=True,
            )
            return {
                "stopped": "emergency",
                "reason": reason,
                "active_move_id": active_move_id,
                "cancelled_move_ids": cancelled,
                "emergency_latched": True,
                "already_latched": already_latched,
            }

    async def clear_soft_stop(self) -> None:
        self.require_emergency_reset()
        manager = self.manager
        if manager.state_store.get().stop_requested:
            await manager.state_store.update(stop_requested=False)

    async def wait_for_move_completion(
        self,
        job: MotionJob,
        script_task: asyncio.Task[dict[str, Any]],
    ) -> None:
        manager = self.manager
        assert job.target is not None
        deadline = time.monotonic() + self.move_timeout(job)
        dwell_started: float | None = None
        revision = manager.state_store.get().revision
        last_refresh = time.monotonic()
        while True:
            self.require_emergency_reset()
            if script_task.cancelled():
                raise MotionError("command_cancelled")
            if script_task.done():
                exception = script_task.exception()
                if exception is not None:
                    raise MotionError(str(exception))
            snapshot = manager.state_store.get()
            if snapshot.fault is not None:
                raise MotionError(snapshot.fault)
            if job.commander_id is not None and self.deadman_expired(job.commander_id):
                await self.request_emergency_stop(reason="deadman_timeout")
                raise MotionError("deadman_timeout")
            if self.is_at_target(
                snapshot,
                job.target,
                job.tolerance_mm,
            ):
                if dwell_started is None:
                    dwell_started = time.monotonic()
                elif time.monotonic() - dwell_started >= 0.075:
                    await manager.state_store.update(
                        position=project_position(snapshot),
                        is_moving=False,
                    )
                    return
            else:
                dwell_started = None
            now = time.monotonic()
            if now - last_refresh >= 0.25:
                try:
                    await manager.moonraker.query_objects(
                        {
                            "toolhead": None,
                            "motion_report": None,
                            "webhooks": None,
                        }
                    )
                except MoonrakerError as exc:
                    raise MotionError(str(exc)) from exc
                last_refresh = now
            if time.monotonic() >= deadline:
                raise MotionError("timeout")
            snapshot = await manager.state_store.wait_for_update(
                revision,
                timeout_s=0.1,
            )
            revision = snapshot.revision

    async def wait_for_ready(self) -> None:
        manager = self.manager
        deadline = time.monotonic() + 30.0
        revision = manager.state_store.get().revision
        while True:
            snapshot = manager.state_store.get()
            if snapshot.webhooks_state.lower() == "ready" and not snapshot.stale:
                return
            if time.monotonic() >= deadline:
                raise MotionError("reset_timeout")
            snapshot = await manager.state_store.wait_for_update(
                revision,
                timeout_s=0.2,
            )
            revision = snapshot.revision

    def move_timeout(self, job: MotionJob) -> float:
        manager = self.manager
        start_position = manager.state_store.get().position
        distance = math.sqrt(
            sum((job.target[axis] - start_position[axis]) ** 2 for axis in job.target or {})
        )
        speed = max(
            1e-6,
            job.feed_mm_s or (manager.config.default_feed_mm_min / 60.0),
        )
        return max(5.0, (distance / speed) * 3.0 + 2.0)

    @staticmethod
    def is_at_target(
        snapshot: StateSnapshot,
        target: dict[str, float],
        tolerance_mm: float,
    ) -> bool:
        position = project_position(snapshot)
        within_target = all(abs(position[axis] - target[axis]) <= tolerance_mm for axis in target)
        return within_target and snapshot.velocity_mm_s <= 0.25

    def cancel_queued(self, *, reason: str) -> list[str]:
        manager = self.manager
        cancelled: list[str] = []
        drained: list[MotionJob] = []
        while True:
            try:
                queued_job = manager.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            drained.append(queued_job)
        for queued_job in drained:
            if queued_job.kind == "move":
                cancelled.append(queued_job.move_id)
                manager._finish_job(
                    queued_job,
                    status="cancelled",
                    error=reason,
                )
            else:
                manager.queue.put_nowait(queued_job)
        return cancelled

    def cancel_all_queued(self, *, reason: str) -> list[str]:
        manager = self.manager
        cancelled: list[str] = []
        while True:
            try:
                queued_job = manager.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            cancelled.append(queued_job.move_id)
            manager._finish_job(
                queued_job,
                status="cancelled",
                error=reason,
            )
        return cancelled

    def require_emergency_reset(self) -> None:
        if self.manager._emergency_latched:
            raise MotionError("emergency_stop_latched")

    async def cancel_active_command_task(self) -> None:
        manager = self.manager
        task = manager._active_command_task
        if task is None:
            return
        await self.settle_command_task(task)
        if manager._active_command_task is task:
            manager._active_command_task = None

    @staticmethod
    async def settle_command_task(
        task: asyncio.Task[dict[str, Any]],
    ) -> None:
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    def deadman_expired(self, commander_id: str) -> bool:
        last_ping = self.manager._active_commander_pings.get(commander_id)
        return last_ping is None or (time.monotonic() - last_ping) > 2.0
