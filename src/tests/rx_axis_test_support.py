"""Reusable fakes and fixtures for the focused RX-axis server test modules."""

from __future__ import annotations

import argparse
import threading
import unittest

from capture.config import GantryConfig
from capture.motion.cubemars import (
    DUTY_CYCLE_ZERO_PAYLOAD,
    SET_ORIGIN_TEMP_PAYLOAD,
    encode_position_velocity_payload,
    encode_rpm_payload,
    rpm_can_id,
)
from capture.motion.rx_axis.server import (
    CanFrame,
    RxAxisServerClient,
    RxAxisServerError,
    RxAxisService,
    config_from_args,
)

__all__ = [
    "BlockingCanTransport",
    "DUTY_CYCLE_ZERO_PAYLOAD",
    "FailingReceiveTransport",
    "FakeCanTransport",
    "FakeLimitSwitchReader",
    "GantryConfig",
    "NOMINAL_FEEDBACK",
    "OVERCURRENT_FEEDBACK",
    "OVERTEMP_FEEDBACK",
    "RX_COMMAND_ID",
    "RX_DUTY_ID",
    "RX_FEEDBACK_ID",
    "RX_ORIGIN_ID",
    "SET_ORIGIN_TEMP_PAYLOAD",
    "TEST_GEAR",
    "TEST_POLES",
    "RxAxisServerClient",
    "RxAxisServerError",
    "RxAxisService",
    "RxAxisServiceTestCase",
    "config_from_args",
    "encode_position_velocity_payload",
    "encode_rpm_payload",
    "make_config",
    "make_server_args",
    "rpm_can_id",
]

TEST_GEAR = 1.0
TEST_POLES = 1


def make_config() -> GantryConfig:
    # Most tests verify the un-inverted wire protocol so they're easier to read;
    # the dedicated motor_position_inverted test opts back in explicitly.
    return GantryConfig(
        axis="rx",
        motor_pole_pairs=TEST_POLES,
        motor_gear_ratio=TEST_GEAR,
        rx_motor_position_inverted=False,
    )


# A canonical "nominal" feedback frame: pos π rad, speed 0, current 0, temp 30, no error.
NOMINAL_FEEDBACK = bytes.fromhex("070800000000 1E 00".replace(" ", ""))

# Over-current feedback: current 10 A (well above the 5 A test threshold).
OVERCURRENT_FEEDBACK = bytes.fromhex("0708000003E81E00")

# Over-temperature feedback: motor temp 85°C (above the 80°C test threshold).
OVERTEMP_FEEDBACK = bytes.fromhex("0000000000005500")

# The CAN ID for the rx-axis (node 1) feedback frame: (0x29 << 8) | 1 = 0x2901.
RX_FEEDBACK_ID = 0x2901
RX_COMMAND_ID = 0x0601  # packet 6 << 8 | 1
RX_ORIGIN_ID = 0x0501  # packet 5 << 8 | 1
RX_DUTY_ID = 0x0001  # packet 0 << 8 | 1


class FakeCanTransport:
    """Records sends and serves pre-canned feedback frames."""

    def __init__(self, feedback: bytes = NOMINAL_FEEDBACK) -> None:
        self.sends: list[tuple[int, bytes, bool]] = []
        self.feedback_queue: list[bytes] = []
        self.default_feedback = feedback
        self.feedback_enabled = True
        self.closed = False

    def send(self, can_id: int, payload: bytes, *, extended: bool = True) -> None:
        self.sends.append((can_id, payload, extended))

    def transact_receive(self, *, feedback_can_id: int, timeout_s: float) -> CanFrame | None:
        if not self.feedback_enabled:
            return None
        data = self.feedback_queue.pop(0) if self.feedback_queue else self.default_feedback
        return CanFrame(feedback_can_id, data)

    def close(self) -> None:
        self.closed = True


class FailingReceiveTransport(FakeCanTransport):
    def transact_receive(self, *, feedback_can_id: int, timeout_s: float) -> CanFrame | None:
        raise OSError("CAN receive failed")


class BlockingCanTransport(FakeCanTransport):
    """Can pause one send to make command/refresh ordering deterministic."""

    def __init__(self) -> None:
        super().__init__()
        self.block_next_send = False
        self.send_started = threading.Event()
        self.release_send = threading.Event()

    def send(self, can_id: int, payload: bytes, *, extended: bool = True) -> None:
        if self.block_next_send:
            self.block_next_send = False
            self.send_started.set()
            if not self.release_send.wait(timeout=1):
                raise TimeoutError("Test timed out waiting to release the blocked CAN send.")
        super().send(can_id, payload, extended=extended)


class FakeLimitSwitchReader:
    def __init__(
        self,
        rx_left_sequence: list[bool] | None = None,
        *,
        rx_right_sequence: list[bool] | None = None,
    ) -> None:
        self.rx_left_sequence = list(rx_left_sequence or [])
        self.rx_right_sequence = list(rx_right_sequence or [])
        self.closed = False

    def _pop(self, seq: list[bool]) -> bool:
        return seq.pop(0) if seq else False

    def rx_left_pressed(self) -> bool:
        return self._pop(self.rx_left_sequence)

    def rx_right_pressed(self) -> bool:
        return self._pop(self.rx_right_sequence)

    def close(self) -> None:
        self.closed = True


def make_server_args(**overrides):
    base = dict(
        host="127.0.0.1",
        port=8091,
        axis="rx",
        poll_interval_s=0.05,
        feedback_timeout_s=0.05,
        max_speed_rad_s=5.0,
        default_accel_rad_s2=10.0,
        current_shutdown_threshold_a=5.0,
        motor_temperature_shutdown_c=80.0,
        homing_fast_speed_rad_s=0.3,
        homing_slow_speed_rad_s=0.05,
        homing_backoff_rad=0.2,
        homing_search_distance_rad=6.0,
        homing_timeout_s=45.0,
        homing_final_position_rad=0.56,
        min_command_position_rad=0.05,
        max_command_position_rad=4.0,
        velocity_deadman_s=0.4,
        velocity_window_margin_rad=0.01,
        velocity_window_brake_s=0.1,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class RxAxisServiceTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.transport = FakeCanTransport()
        self.service = RxAxisService(
            make_config(),
            config_from_args(make_server_args()),
            transport=self.transport,
            limit_switch_reader=FakeLimitSwitchReader([]),
        )
        self.service._homed = True
