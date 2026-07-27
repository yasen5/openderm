"""Shared state models for coordinated gantry motion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import math
import time
from typing import Any


AXES = ("x",)


def empty_position() -> dict[str, float]:
    return {axis: 0.0 for axis in AXES}


@dataclass(frozen=True)
class StateSnapshot:
    position: dict[str, float] = field(default_factory=empty_position)
    velocity_mm_s: float = 0.0
    timestamp: float = 0.0
    is_moving: bool = False
    last_commanded_target: dict[str, float] | None = None
    homed_axes: str = ""
    fault: str | None = None
    stale: bool = True
    enabled: bool = False
    active_move_id: str | None = None
    queue_depth: int = 0
    webhooks_state: str = "disconnected"
    webhooks_message: str | None = None
    stop_requested: bool = False
    streaming: bool = False
    emergency_latched: bool = False
    revision: int = 0


@dataclass
class MoveRecord:
    move_id: str
    kind: str
    status: str
    created_at: float
    target: dict[str, float] | None = None
    feed_mm_s: float | None = None
    tolerance_mm: float = 0.05
    error: str | None = None
    completed_at: float | None = None
    waiter: asyncio.Event = field(default_factory=asyncio.Event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "move_id": self.move_id,
            "kind": self.kind,
            "status": self.status,
            "target": self.target,
            "feed_mm_s": self.feed_mm_s,
            "tolerance_mm": self.tolerance_mm,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class StateStore:
    def __init__(self):
        self._snapshot = StateSnapshot(timestamp=time.monotonic())
        self._condition: asyncio.Condition | None = None
        self._stream_subscribers: set[asyncio.Queue[StateSnapshot]] = set()

    def _condition_obj(self) -> asyncio.Condition:
        if self._condition is None:
            self._condition = asyncio.Condition()
        return self._condition

    def get(self) -> StateSnapshot:
        return self._snapshot

    async def publish(self, snapshot: StateSnapshot) -> StateSnapshot:
        condition = self._condition_obj()
        async with condition:
            next_snapshot = replace(snapshot, revision=self._snapshot.revision + 1)
            self._snapshot = next_snapshot
            condition.notify_all()
        for queue in list(self._stream_subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(next_snapshot)
            except asyncio.QueueFull:
                continue
        return next_snapshot

    async def update(self, **changes: Any) -> StateSnapshot:
        return await self.publish(replace(self._snapshot, **changes))

    async def wait_for_update(self, revision: int, timeout_s: float | None = None) -> StateSnapshot:
        condition = self._condition_obj()
        async with condition:
            if timeout_s is None:
                while self._snapshot.revision <= revision:
                    await condition.wait()
            else:
                deadline = time.monotonic() + timeout_s
                while self._snapshot.revision <= revision:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait_for(condition.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
            return self._snapshot

    def subscribe(self, maxsize: int = 1) -> asyncio.Queue[StateSnapshot]:
        queue: asyncio.Queue[StateSnapshot] = asyncio.Queue(maxsize=maxsize)
        self._stream_subscribers.add(queue)
        try:
            queue.put_nowait(self._snapshot)
        except asyncio.QueueFull:
            pass
        return queue

    def unsubscribe(self, queue: asyncio.Queue[StateSnapshot]) -> None:
        self._stream_subscribers.discard(queue)


def merge_position(current: dict[str, float], updates: dict[str, float] | None) -> dict[str, float]:
    merged = dict(current)
    if updates is not None:
        merged.update(updates)
    return merged


def project_position(snapshot: StateSnapshot, now: float | None = None) -> dict[str, float]:
    if now is None:
        now = time.monotonic()
    position = dict(snapshot.position)
    if (
        not snapshot.is_moving
        or snapshot.last_commanded_target is None
        or snapshot.velocity_mm_s <= 0
        or snapshot.stale
    ):
        return position
    dt = max(0.0, now - snapshot.timestamp)
    remaining = {
        axis: snapshot.last_commanded_target[axis] - position[axis]
        for axis in snapshot.last_commanded_target
    }
    distance = math.sqrt(sum(delta * delta for delta in remaining.values()))
    if distance <= 1e-9:
        return position
    step = min(distance, snapshot.velocity_mm_s * dt)
    for axis, delta in remaining.items():
        position[axis] += delta * (step / distance)
    return position
