"""RX-axis limit-switch homing controller."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from ...sensors.limit_switches import (
    LimitSwitchConfig,
    LimitSwitchError,
    LimitSwitchReader,
)
from ..cubemars import SET_ORIGIN_TEMP_PAYLOAD, CubeMarsFeedback
from .types import HomingRecord, RxAxisServerError

if TYPE_CHECKING:
    from .service import RxAxisService


class RxHomingController:
    """Run the two-pass switch homing sequence and track its state."""

    def __init__(self, service: RxAxisService) -> None:
        self.service = service

    async def home_axis(self) -> HomingRecord:
        service = self.service
        service.safety.require_motion_allowed()
        service.safety.require_velocity_allowed(service.server_config.homing_fast_speed_rad_s)
        service.safety.require_velocity_allowed(service.server_config.homing_slow_speed_rad_s)
        if service.server_config.homing_backoff_rad <= 0:
            raise RxAxisServerError("homing_backoff_rad must be positive.")
        if service.server_config.homing_search_distance_rad <= 0:
            raise RxAxisServerError("homing_search_distance_rad must be positive.")
        reader = self.get_limit_switch_reader()
        left_pressed = self.axis_left_pressed_fn(reader)
        record = HomingRecord(
            status="running",
            axis=service.config.axis,
            started_at=time.monotonic(),
            completed_at=None,
        )
        trigger_label = f"first_{service.config.axis}_left_trigger"
        release_label = f"{service.config.axis}_left_release"
        second_trigger_label = f"second_{service.config.axis}_left_trigger"
        final_release_label = f"final_{service.config.axis}_left_release"
        self.set_status("starting")
        service._homing_active = True
        try:
            self.set_status("initial_feedback")
            await service.poll_once()
            if left_pressed():
                self.set_status("initial_backoff", left_pressed=True)
                await self.move_relative_and_wait(
                    service.server_config.homing_backoff_rad,
                    service.server_config.homing_fast_speed_rad_s,
                    timeout_s=service.server_config.homing_timeout_s,
                    stop_when=lambda: not left_pressed(),
                    stop_label=release_label,
                )
            self.set_status("fast_negative_approach")
            first_feedback = await self.move_relative_and_wait(
                -service.server_config.homing_search_distance_rad,
                service.server_config.homing_fast_speed_rad_s,
                timeout_s=service.server_config.homing_timeout_s,
                stop_when=left_pressed,
                stop_label=trigger_label,
            )
            record.first_trigger_position_rad = (
                None if first_feedback is None else first_feedback.position_rad
            )
            self.set_status(
                "first_triggered",
                position_rad=record.first_trigger_position_rad,
            )
            await service.stop_motor()
            self.set_status("backoff_after_first_trigger")
            await self.move_relative_and_wait(
                service.server_config.homing_backoff_rad,
                service.server_config.homing_fast_speed_rad_s,
                timeout_s=service.server_config.homing_timeout_s,
                stop_when=lambda: not left_pressed(),
                stop_label=release_label,
            )
            self.set_status("slow_negative_approach")
            second_feedback = await self.move_relative_and_wait(
                -service.server_config.homing_search_distance_rad,
                service.server_config.homing_slow_speed_rad_s,
                timeout_s=service.server_config.homing_timeout_s,
                stop_when=left_pressed,
                stop_label=second_trigger_label,
            )
            record.second_trigger_position_rad = (
                None if second_feedback is None else second_feedback.position_rad
            )
            self.set_status(
                "second_triggered",
                position_rad=record.second_trigger_position_rad,
            )
            await service.stop_motor()
            self.set_status("setting_zero")
            await service._send(
                service.set_origin_can_id,
                bytes.fromhex(SET_ORIGIN_TEMP_PAYLOAD),
            )
            record.zeroed = True
            await asyncio.sleep(0.1)
            self.set_status("moving_to_final_position")
            await service.move_to(
                service.server_config.homing_final_position_rad,
                service.server_config.homing_fast_speed_rad_s,
                require_homed=False,
                require_nonnegative=True,
            )
            self.set_status("waiting_for_final_switch_release")
            await self.wait_for_condition(
                lambda: not left_pressed(),
                timeout_s=service.server_config.homing_timeout_s,
                label=final_release_label,
            )
            record.final_position_rad = service.server_config.homing_final_position_rad
            record.status = "completed"
            record.completed_at = time.monotonic()
            service._homed = True
            self.set_status(
                "completed",
                final_position_rad=record.final_position_rad,
            )
            service._last_command = {
                "type": "home",
                "final_position_rad": (service.server_config.homing_final_position_rad),
                "can_id": f"0x{service.command_can_id:08X}",
                "timestamp": time.monotonic(),
            }
            return record
        except Exception as exc:
            record.status = "failed"
            record.completed_at = time.monotonic()
            record.error = str(exc)
            self.set_status("failed", error=str(exc))
            try:
                await service.stop_motor()
            except Exception:
                pass
            raise RxAxisServerError(f"RX-axis homing failed: {exc}") from exc
        finally:
            service._homing_active = False

    async def move_relative_and_wait(
        self,
        delta_rad: float,
        velocity_rad_s: float,
        *,
        timeout_s: float,
        stop_when: Any,
        stop_label: str,
    ) -> CubeMarsFeedback | None:
        service = self.service
        start_feedback = service._feedback or await service.poll_once()
        if start_feedback is None:
            raise RxAxisServerError("No CAN feedback is available for homing.")
        target = start_feedback.position_rad + delta_rad
        service.safety.require_within_limits(target)
        deadline = time.monotonic() + timeout_s
        latest_feedback: CubeMarsFeedback | None = start_feedback
        self.set_status(
            stop_label,
            target_position_rad=target,
            velocity_rad_s=velocity_rad_s,
            start_position_rad=start_feedback.position_rad,
        )
        await service.move_to(
            target,
            velocity_rad_s,
            require_homed=False,
            require_nonnegative=False,
        )
        while True:
            await service.safety.check_limit_switch_shutdown(
                allow_left=stop_label.endswith("trigger")
            )
            if stop_when():
                return latest_feedback
            if time.monotonic() >= deadline:
                raise RxAxisServerError(f"Homing timed out waiting for {stop_label}.")
            latest_feedback = await service.poll_once()
            await asyncio.sleep(service.server_config.poll_interval_s)

    async def wait_for_condition(
        self,
        condition: Any,
        *,
        timeout_s: float,
        label: str,
    ) -> None:
        service = self.service
        deadline = time.monotonic() + timeout_s
        while True:
            if condition():
                return
            if time.monotonic() >= deadline:
                raise RxAxisServerError(f"Homing timed out waiting for {label}.")
            await service.poll_once()
            await asyncio.sleep(service.server_config.poll_interval_s)

    def get_limit_switch_reader(self) -> Any:
        service = self.service
        if service._limit_switch_reader is None:
            try:
                service._limit_switch_reader = LimitSwitchReader(LimitSwitchConfig.from_env())
            except LimitSwitchError as exc:
                raise RxAxisServerError(str(exc)) from exc
        return service._limit_switch_reader

    @staticmethod
    def axis_left_pressed_fn(reader: Any):
        return reader.rx_left_pressed

    @staticmethod
    def axis_right_pressed_fn(reader: Any):
        return reader.rx_right_pressed

    def limit_switch_payload(self) -> dict[str, Any]:
        service = self.service
        reader = self.get_limit_switch_reader()
        left_key = f"{service.config.axis}_left_pressed"
        right_key = f"{service.config.axis}_right_pressed"
        return {
            left_key: self.axis_left_pressed_fn(reader)(),
            right_key: self.axis_right_pressed_fn(reader)(),
        }

    def require_homed(self) -> None:
        if not self.service._homed:
            raise RxAxisServerError("RX axis must be homed before accepting motion commands.")

    def set_status(self, phase: str, **extra: Any) -> None:
        payload = {
            "phase": phase,
            "timestamp": time.monotonic(),
        }
        payload.update(extra)
        self.service._homing_status = payload
