from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from ...config import GantryConfig, linear_axis_limits
from .homing import GantryHomingController
from .jobs import MotionError, MotionJob
from .moonraker import format_gcode_value
from .moonraker_coordinator import MoonrakerCoordinatorClient
from .safety import GantrySafetyController
from .state import AXES, MoveRecord, StateStore
from .streaming import GantryStreamingController


class MotionManager:
    def __init__(
        self,
        config: GantryConfig,
        state_store: StateStore,
        moonraker: MoonrakerCoordinatorClient,
    ) -> None:
        self.config = config
        self.state_store = state_store
        self.moonraker = moonraker
        self.queue: asyncio.Queue[MotionJob] = asyncio.Queue()
        self.jobs: dict[str, MotionJob] = {}
        self.records: dict[str, MoveRecord] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._running = False
        self._active_job: MotionJob | None = None
        self._active_command_task: asyncio.Task[dict[str, Any]] | None = None
        self._active_commander_pings: dict[str, float] = {}
        self._emergency_lock = asyncio.Lock()
        self._emergency_latched = False
        self._emergency_generation = 0
        # Streaming mode (additive, independent of the blocking queue/worker).
        self._streaming = False
        self._stream_task: asyncio.Task[None] | None = None
        self._stream_target: dict[str, float] | None = None
        self.homing = GantryHomingController(self)
        self.safety = GantrySafetyController(self)
        self.streaming = GantryStreamingController(self)

    async def start(self) -> None:
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop(), name="gantry-motion-worker")

    async def stop(self) -> None:
        self._running = False
        await self.stop_stream()
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def enqueue_move(
        self,
        *,
        target: dict[str, float],
        feed_mm_s: float | None,
        tolerance_mm: float,
        commander_id: str | None,
    ) -> MoveRecord:
        self.safety.require_emergency_reset()
        # A NEW move request after a soft stop is the operator saying "go
        # again" -- clear the latch here, exactly as start_stream does. The
        # latch used to persist forever for the move path: request_soft_stop
        # cancels the queue and sets stop_requested, the worker loop cancels
        # any move dequeued while it is set (the race net), and the only
        # move-side clear lived inside _run_job -- which a move could never
        # reach past that dequeue check. Every move after an aborted scan was
        # cancelled 'soft_stop' until a restart/home/enable cleared it.
        await self.clear_soft_stop()
        move_id = self._next_move_id()
        axes = tuple(sorted(target.keys()))
        job = MotionJob(
            move_id=move_id,
            kind="move",
            created_at=time.monotonic(),
            target=target,
            feed_mm_s=feed_mm_s or (self.config.default_feed_mm_min / 60.0),
            tolerance_mm=tolerance_mm,
            axes=axes,
            commander_id=commander_id,
        )
        record = MoveRecord(
            move_id=move_id,
            kind="move",
            status="queued",
            created_at=job.created_at,
            target=target,
            feed_mm_s=job.feed_mm_s,
            tolerance_mm=tolerance_mm,
            waiter=job.wait_event,
        )
        self.jobs[move_id] = job
        self.records[move_id] = record
        await self.queue.put(job)
        await self.state_store.update(queue_depth=self.queue.qsize())
        return record

    async def enqueue_home(self, axes: tuple[str, ...]) -> MoveRecord:
        return await self.homing.enqueue(axes)

    async def enqueue_control(self, kind: str, *, stop_mode: str | None = None) -> MoveRecord:
        if kind != "reset":
            self.safety.require_emergency_reset()
        move_id = self._next_move_id()
        job = MotionJob(
            move_id=move_id,
            kind=kind,
            created_at=time.monotonic(),
            stop_mode=stop_mode,
        )
        record = MoveRecord(
            move_id=move_id,
            kind=kind,
            status="queued",
            created_at=job.created_at,
            waiter=job.wait_event,
        )
        self.jobs[move_id] = job
        self.records[move_id] = record
        await self.queue.put(job)
        await self.state_store.update(queue_depth=self.queue.qsize())
        return record

    def get_record(self, move_id: str) -> MoveRecord | None:
        return self.records.get(move_id)

    def register_commander_ping(self, commander_id: str) -> None:
        self._active_commander_pings[commander_id] = time.monotonic()

    @property
    def emergency_latched(self) -> bool:
        return self._emergency_latched

    @property
    def worker_running(self) -> bool:
        return self._running and self._worker_task is not None and not self._worker_task.done()

    async def request_soft_stop(self) -> dict[str, Any]:
        return await self.safety.request_soft_stop()

    async def request_emergency_stop(
        self,
        *,
        reason: str = "emergency_stop",
    ) -> dict[str, Any]:
        return await self.safety.request_emergency_stop(reason=reason)

    # ------------------------------------------------------------------
    # Streaming mode
    #
    # An alternative to the blocking enqueue_move/queue path: a single live
    # setpoint is continuously chased by emitting G1 moves WITHOUT waiting for
    # each to settle, so Klipper's look-ahead blends them into continuous
    # motion. The blocking path above is left fully intact; callers pick a mode
    # by either enqueuing moves or by starting a stream (not both at once).
    # ------------------------------------------------------------------

    @property
    def is_streaming(self) -> bool:
        return self.streaming.active

    async def start_stream(
        self,
        *,
        feed_mm_s: float | None = None,
        tick_s: float = 0.05,
        min_step_mm: float = 0.01,
    ) -> dict[str, Any]:
        return await self.streaming.start(
            feed_mm_s=feed_mm_s,
            tick_s=tick_s,
            min_step_mm=min_step_mm,
        )

    async def update_stream_target(self, target: dict[str, float]) -> dict[str, float]:
        return await self.streaming.update_target(target)

    async def stop_stream(self) -> dict[str, Any]:
        return await self.streaming.stop()

    async def clear_soft_stop(self) -> None:
        await self.safety.clear_soft_stop()

    async def _worker_loop(self) -> None:
        while self._running:
            job = await self.queue.get()
            await self.state_store.update(queue_depth=self.queue.qsize())
            if self._emergency_latched and job.kind != "reset":
                self._finish_job(job, status="cancelled", error="emergency_stop")
                continue
            if job.kind == "move" and self.state_store.get().stop_requested:
                self._finish_job(job, status="cancelled", error="soft_stop")
                continue
            try:
                await self._run_job(job)
            except asyncio.CancelledError:
                if not (self._running and self._emergency_latched):
                    raise
                self._finish_job(job, status="cancelled", error="emergency_stop")
            except Exception as exc:  # noqa: BLE001
                status = "cancelled" if self._emergency_latched else "failed"
                error_message = "emergency_stop" if status == "cancelled" else str(exc)
                self._finish_job(job, status=status, error=error_message)
            finally:
                self._active_job = None
                current = self.state_store.get()
                await self.state_store.update(
                    active_move_id=None
                    if current.active_move_id == job.move_id
                    else current.active_move_id,
                    is_moving=current.is_moving if job.kind != "move" else False,
                )

    async def _run_job(self, job: MotionJob) -> None:
        self._active_job = job
        self._set_record_status(job.move_id, "active")
        if job.kind == "move":
            self.safety.require_emergency_reset()
            await self.clear_soft_stop()
            await self._validate_move(job)
            now = time.monotonic()
            await self.state_store.update(
                active_move_id=job.move_id,
                last_commanded_target=job.target,
                is_moving=True,
                velocity_mm_s=job.feed_mm_s or 0.0,
                timestamp=now,
            )
            script_task = asyncio.create_task(
                self.moonraker.gcode_script(
                    self._build_move_gcode(job.target or {}, job.feed_mm_s or 0.0)
                )
            )
            self._active_command_task = script_task
            try:
                await self.safety.wait_for_move_completion(job, script_task)
                await script_task
                self.safety.require_emergency_reset()
                self._finish_job(job, status="completed")
            finally:
                await self.safety.settle_command_task(script_task)
                if self._active_command_task is script_task:
                    self._active_command_task = None
            return
        if job.kind == "home":
            self.safety.require_emergency_reset()
            await self.clear_soft_stop()
            home_task = asyncio.create_task(
                self.moonraker.gcode_script(self.homing.build_gcode(job.axes))
            )
            self._active_command_task = home_task
            try:
                await self.homing.wait_for_completion(job.axes, home_task)
                await home_task
                self.safety.require_emergency_reset()
                self._finish_job(job, status="completed")
            finally:
                await self.safety.settle_command_task(home_task)
                if self._active_command_task is home_task:
                    self._active_command_task = None
            return
        if job.kind == "enable":
            self.safety.require_emergency_reset()
            await self.clear_soft_stop()
            for stepper in self.config.controlled_steppers:
                self.safety.require_emergency_reset()
                await self.moonraker.gcode_script(f"SET_STEPPER_ENABLE STEPPER={stepper} ENABLE=1")
            self.safety.require_emergency_reset()
            await self.state_store.update(enabled=True)
            self._finish_job(
                job,
                status="completed",
                result={"enabled": True, "homed_axes": self.state_store.get().homed_axes},
            )
            return
        if job.kind == "disable":
            self.safety.require_emergency_reset()
            snapshot = self.state_store.get()
            if snapshot.is_moving:
                raise MotionError("disable_conflict")
            await self.moonraker.gcode_script("M84")
            await self.state_store.update(enabled=False, homed_axes="", is_moving=False)
            self._finish_job(job, status="completed", result={"enabled": False, "homed_axes": ""})
            return
        if job.kind == "reset":
            emergency_generation = self._emergency_generation
            # Do not accept the pre-reset ready state as proof that Klipper
            # restarted. Moonraker must publish a fresh ready/non-stale state.
            await self.state_store.update(
                stale=True,
                webhooks_state="restarting",
                webhooks_message=None,
            )
            await self.moonraker.firmware_restart()
            await self.safety.wait_for_ready()
            if emergency_generation != self._emergency_generation:
                raise MotionError("emergency_stop_during_reset")
            await self.state_store.update(
                fault=None,
                homed_axes="",
                enabled=False,
                stop_requested=False,
                emergency_latched=False,
            )
            self._emergency_latched = False
            self._finish_job(job, status="completed")
            return
        raise MotionError(f"unsupported_job:{job.kind}")

    async def _validate_move(self, job: MotionJob) -> None:
        snapshot = self.state_store.get()
        if snapshot.fault is not None:
            raise MotionError(snapshot.fault)
        if snapshot.stale:
            raise MotionError("stale_state")
        if snapshot.webhooks_state.lower() != "ready":
            raise MotionError(
                snapshot.webhooks_message or f"printer_not_ready:{snapshot.webhooks_state}"
            )
        requested_axes = set(job.axes)
        if not requested_axes.issubset(AXES):
            raise MotionError("unknown_axis")
        for axis in job.axes:
            if axis not in snapshot.homed_axes:
                raise MotionError("requires_homing")
        for axis, position_mm in (job.target or {}).items():
            minimum, maximum = linear_axis_limits(axis)
            if axis == self.config.axis:
                minimum, maximum = self.config.travel_min_mm, self.config.travel_max_mm
            if not minimum <= position_mm <= maximum:
                raise MotionError(f"out_of_bounds:{axis}")

    @staticmethod
    def _build_move_gcode(target: dict[str, float], feed_mm_s: float) -> str:
        terms = [f"{axis.upper()}{format_gcode_value(target[axis])}" for axis in sorted(target)]
        feed_mm_min = feed_mm_s * 60.0
        return "G90\nG1 " + " ".join(terms) + f" F{format_gcode_value(feed_mm_min)}"

    def _finish_job(
        self,
        job: MotionJob,
        *,
        status: str,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        record = self.records[job.move_id]
        record.status = status
        record.error = error
        record.completed_at = time.monotonic()
        job.result = result
        job.wait_event.set()

    def _set_record_status(self, move_id: str, status: str) -> None:
        self.records[move_id].status = status

    @staticmethod
    def _next_move_id() -> str:
        return f"mv_{uuid.uuid4().hex[:12]}"
