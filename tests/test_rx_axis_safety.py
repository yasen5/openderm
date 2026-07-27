from __future__ import annotations

import asyncio
from dataclasses import replace

from rx_axis_test_support import (
    BlockingCanTransport,
    DUTY_CYCLE_ZERO_PAYLOAD,
    FailingReceiveTransport,
    FakeCanTransport,
    FakeLimitSwitchReader,
    NOMINAL_FEEDBACK,
    OVERCURRENT_FEEDBACK,
    OVERTEMP_FEEDBACK,
    RX_DUTY_ID,
    TEST_GEAR,
    TEST_POLES,
    RxAxisServerError,
    RxAxisService,
    RxAxisServiceTestCase,
    config_from_args,
    encode_rpm_payload,
    make_config,
    make_server_args,
    rpm_can_id,
)


class RxAxisSafetyTests(RxAxisServiceTestCase):
    async def test_health_reports_loop_state_feedback_and_safety(self) -> None:
        initial = self.service.health_payload()
        self.assertFalse(initial["ok"])
        self.assertIn("feedback_poll_loop_not_running", initial["issues"])
        self.assertIn("feedback_stale", initial["issues"])

        await self.service.start()
        try:
            for _ in range(100):
                health = self.service.health_payload()
                if health["ok"]:
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(health["ok"])
            self.assertTrue(health["ready_for_motion"])
            self.assertTrue(health["loops"]["feedback_poll"]["running"])
            self.assertTrue(health["loops"]["command_refresh"]["running"])

            self.transport.default_feedback = OVERCURRENT_FEEDBACK
            for _ in range(100):
                health = self.service.health_payload()
                if "safety_shutdown_latched" in health["issues"]:
                    break
                await asyncio.sleep(0.01)
            self.assertFalse(health["ok"])
            self.assertIn("safety_shutdown_latched", health["issues"])
        finally:
            await self.service.stop()

    async def test_health_exposes_background_poll_failure(self) -> None:
        service = RxAxisService(
            make_config(),
            self.service.server_config,
            transport=FailingReceiveTransport(),
            limit_switch_reader=FakeLimitSwitchReader([]),
        )
        service._homed = True
        await service.start()
        try:
            for _ in range(100):
                health = service.health_payload()
                if health["loops"]["feedback_poll"]["error"] is not None:
                    break
                await asyncio.sleep(0.01)
            self.assertFalse(health["ok"])
            self.assertIn("feedback_poll_error", health["issues"])
            self.assertIn(
                "CAN receive failed",
                health["loops"]["feedback_poll"]["error"],
            )
        finally:
            await service.stop()

    async def test_poll_once_disables_drive_when_current_exceeds_threshold(self) -> None:
        self.transport.default_feedback = OVERCURRENT_FEEDBACK
        with self.assertRaises(RxAxisServerError):
            await self.service.poll_once()
        # On safety shutdown, the service sends duty-cycle = 0.
        self.assertEqual(len(self.transport.sends), 1)
        can_id, payload, _ = self.transport.sends[0]
        self.assertEqual(can_id, RX_DUTY_ID)
        self.assertEqual(payload.hex().upper(), DUTY_CYCLE_ZERO_PAYLOAD)
        state = self.service.state_payload()
        self.assertEqual(state["safety_shutdown"]["reason"], "current_threshold_exceeded")

    async def test_poll_once_disables_drive_when_motor_temperature_exceeds_threshold(self) -> None:
        self.transport.default_feedback = OVERTEMP_FEEDBACK
        with self.assertRaises(RxAxisServerError):
            await self.service.poll_once()
        self.assertEqual(len(self.transport.sends), 1)
        can_id, payload, _ = self.transport.sends[0]
        self.assertEqual(can_id, RX_DUTY_ID)
        self.assertEqual(payload.hex().upper(), DUTY_CYCLE_ZERO_PAYLOAD)
        shutdown = self.service.state_payload()["safety_shutdown"]
        self.assertEqual(shutdown["reason"], "motor_temperature_exceeded")
        self.assertEqual(shutdown["motor_temperature_c"], 85)
        self.assertEqual(shutdown["threshold_c"], 80.0)

    async def test_motor_temperature_shutdown_disabled_when_threshold_nonpositive(self) -> None:
        service = RxAxisService(
            make_config(),
            config_from_args(make_server_args(motor_temperature_shutdown_c=0.0)),
            transport=FakeCanTransport(OVERTEMP_FEEDBACK),
            limit_switch_reader=FakeLimitSwitchReader([]),
        )
        service._homed = True
        # Should NOT raise even with 85°C feedback, because threshold=0 disables the check.
        await service.poll_once()
        self.assertIsNone(service.state_payload()["safety_shutdown"])

    async def test_clear_errors_releases_safety_latch_without_wire_command(self) -> None:
        self.transport.default_feedback = OVERCURRENT_FEEDBACK
        with self.assertRaises(RxAxisServerError):
            await self.service.poll_once()
        with self.assertRaises(RxAxisServerError):
            await self.service.move_to(1.0, 0.5)
        sends_before = len(self.transport.sends)
        await self.service.clear_errors()
        # AK servo mode has no clear-errors CAN command — the latch reset is purely software-side.
        self.assertEqual(len(self.transport.sends), sends_before)
        # Reset feedback so the next poll-driven check no longer trips.
        self.transport.default_feedback = NOMINAL_FEEDBACK
        await self.service.move_to(1.0, 0.5)
        self.assertIsNone(self.service.state_payload()["safety_shutdown"])

    async def test_limit_switch_after_homing_triggers_duty_cycle_zero(self) -> None:
        service = RxAxisService(
            make_config(),
            self.service.server_config,
            transport=self.transport,
            limit_switch_reader=FakeLimitSwitchReader([True]),
        )
        service._homed = True
        with self.assertRaises(RxAxisServerError):
            await service.poll_once()
        can_id, payload, _ = self.transport.sends[-1]
        self.assertEqual(can_id, RX_DUTY_ID)
        self.assertEqual(payload.hex().upper(), DUTY_CYCLE_ZERO_PAYLOAD)
        self.assertEqual(
            service.state_payload()["safety_shutdown"]["reason"], "limit_switch_triggered"
        )

    async def test_velocity_guard_zeroes_on_deadman(self) -> None:
        service = RxAxisService(
            make_config(),
            config_from_args(make_server_args(velocity_deadman_s=0.01)),
            transport=self.transport,
            limit_switch_reader=FakeLimitSwitchReader([]),
        )
        service._homed = True
        await service.poll_once()
        await service.set_velocity(0.5)
        await asyncio.sleep(0.03)
        await service.poll_once()
        self.assertEqual(service._last_command["type"], "velocity_zeroed")
        self.assertEqual(service._last_command["reason"], "deadman")
        self.assertEqual(service._velocity_cmd["velocity_rad_s"], 0.0)

    async def test_velocity_deadman_zeroes_when_feedback_disappears(self) -> None:
        server_config = replace(
            config_from_args(make_server_args(velocity_deadman_s=0.03)),
            # Deliberately longer than the dead-man. The watchdog must still
            # wake on its own deadline instead of waiting for the next refresh.
            command_refresh_interval_s=0.5,
            feedback_timeout_s=0.001,
        )
        transport = FakeCanTransport()
        service = RxAxisService(
            make_config(),
            server_config,
            transport=transport,
            limit_switch_reader=FakeLimitSwitchReader([]),
        )
        service._homed = True
        await service.poll_once()
        await service.set_velocity(0.5)
        transport.feedback_enabled = False

        await service.start()
        try:
            await asyncio.sleep(0.1)
        finally:
            await service.stop()

        zero_payload = bytes.fromhex(
            encode_rpm_payload(0.0, gear_ratio=TEST_GEAR, pole_pairs=TEST_POLES)
        )
        rpm_payloads = [
            payload
            for can_id, payload, _ in transport.sends
            if can_id == rpm_can_id(service.config.selected_can_node_id)
        ]
        first_zero = rpm_payloads.index(zero_payload)
        self.assertTrue(rpm_payloads[:first_zero])
        self.assertTrue(all(payload == zero_payload for payload in rpm_payloads[first_zero:]))
        self.assertEqual(service._last_command["type"], "velocity_zeroed")
        self.assertEqual(service._last_command["reason"], "deadman")
        self.assertEqual(service._velocity_cmd["velocity_rad_s"], 0.0)

    async def test_stop_cannot_be_followed_by_stale_velocity_refresh(self) -> None:
        transport = BlockingCanTransport()
        service = RxAxisService(
            make_config(),
            replace(
                config_from_args(make_server_args()),
                command_refresh_interval_s=0.01,
            ),
            transport=transport,
            limit_switch_reader=FakeLimitSwitchReader([]),
        )
        service._homed = True
        await service.poll_once()
        await service.set_velocity(0.5)

        transport.block_next_send = True
        await service.start()
        try:
            send_started = await asyncio.to_thread(transport.send_started.wait, 1)
            self.assertTrue(send_started, "stream refresh did not reach the transport")
            stop_task = asyncio.create_task(service.stop_motor())
            await asyncio.sleep(0)
            self.assertFalse(stop_task.done(), "stop should wait for the in-flight refresh")
            transport.release_send.set()
            await asyncio.wait_for(stop_task, timeout=1)
        finally:
            transport.release_send.set()
            await service.stop()

        stop_index = next(
            index for index, (can_id, _, _) in enumerate(transport.sends) if can_id == RX_DUTY_ID
        )
        self.assertFalse(
            any(
                can_id == rpm_can_id(service.config.selected_can_node_id)
                for can_id, _, _ in transport.sends[stop_index + 1 :]
            ),
            "a stale nonzero velocity refresh was sent after stop",
        )

    async def test_velocity_guard_zeroes_at_window_edge_mid_flight(self) -> None:
        await self.service.poll_once()
        await self.service.set_velocity(0.5)
        # Next feedback frame reports 3.98 rad (228.03 deg -> 2280 = 0x08E8),
        # inside the 0.05 margin below the 4.0 max window.
        self.transport.feedback_queue.append(bytes.fromhex("08E8000000001E00".replace(" ", "")))
        await self.service.poll_once()
        self.assertEqual(self.service._last_command["type"], "velocity_zeroed")
        self.assertEqual(self.service._last_command["reason"], "max_window")

    async def test_safety_shutdown_stops_refreshing_active_command(self) -> None:
        # After a current shutdown the refresh loop must go quiet: a re-sent
        # position payload would re-engage the servo loop against the disable,
        # and a re-sent velocity payload would keep the motor SPINNING.
        await self.service.poll_once()
        await self.service.set_velocity(0.5)
        overcurrent = bytes.fromhex("07080000030C1E00")  # 7.8A > 5A threshold
        self.transport.feedback_queue.append(overcurrent)
        with self.assertRaises(RxAxisServerError):
            await self.service.poll_once()
        self.assertIsNone(self.service._active_command_payload)
        self.assertIsNone(self.service._velocity_cmd)

    async def test_clear_errors_drops_active_streaming_target(self) -> None:
        await self.service.move_to(1.0, 0.5)
        self.assertIsNotNone(self.service._active_command_payload)
        await self.service.clear_errors()
        self.assertIsNone(self.service._active_command_payload)
