from __future__ import annotations

import math

from rx_axis_test_support import (
    DUTY_CYCLE_ZERO_PAYLOAD,
    FakeCanTransport,
    FakeLimitSwitchReader,
    GantryConfig,
    RX_COMMAND_ID,
    RX_DUTY_ID,
    TEST_GEAR,
    TEST_POLES,
    RxAxisServerError,
    RxAxisService,
    RxAxisServiceTestCase,
    config_from_args,
    encode_position_velocity_payload,
    encode_rpm_payload,
    make_config,
    make_server_args,
    rpm_can_id,
)


class RxAxisCommandTests(RxAxisServiceTestCase):
    async def test_poll_once_decodes_feedback_without_sending_anything(self) -> None:
        feedback = await self.service.poll_once()
        # AK servo mode auto-publishes feedback; poll_once is purely a receive.
        self.assertEqual(self.transport.sends, [])
        self.assertAlmostEqual(feedback.position_rad, math.pi, places=3)
        self.assertEqual(feedback.error_code, 0)

    async def test_move_to_sends_single_servo_command(self) -> None:
        await self.service.move_to(1.0, 0.5)
        # No enter-mode preamble — just one extended-frame position-velocity packet.
        self.assertEqual(len(self.transport.sends), 1)
        can_id, payload, extended = self.transport.sends[0]
        self.assertEqual(can_id, RX_COMMAND_ID)
        self.assertTrue(extended)
        self.assertEqual(
            payload.hex().upper(),
            encode_position_velocity_payload(
                1.0,
                0.5,
                self.service.server_config.default_accel_rad_s2,
                gear_ratio=TEST_GEAR,
                pole_pairs=TEST_POLES,
            ),
        )
        state = self.service.state_payload()
        self.assertEqual(state["last_command"]["type"], "move_to")
        self.assertEqual(state["last_command"]["accel_rad_s2"], 10.0)

    async def test_move_to_accepts_explicit_accel(self) -> None:
        await self.service.move_to(1.0, 0.5, 25.0)
        payload = self.transport.sends[0][1]
        expected = encode_position_velocity_payload(
            1.0, 0.5, 25.0, gear_ratio=TEST_GEAR, pole_pairs=TEST_POLES
        )
        self.assertEqual(payload.hex().upper(), expected)

    async def test_move_to_rejects_out_of_range_position(self) -> None:
        with self.assertRaises(RxAxisServerError):
            await self.service.move_to(20.0, 0.5)

    async def test_move_to_rejects_before_homing(self) -> None:
        self.service._homed = False
        with self.assertRaises(RxAxisServerError):
            await self.service.move_to(1.0, 0.5)

    async def test_move_to_rejects_negative_target_after_homing(self) -> None:
        with self.assertRaises(RxAxisServerError):
            await self.service.move_to(-0.1, 0.5)

    async def test_move_to_rejects_target_below_post_home_minimum(self) -> None:
        with self.assertRaises(RxAxisServerError):
            await self.service.move_to(0.04, 0.5)

    async def test_move_to_rejects_target_above_post_home_maximum(self) -> None:
        with self.assertRaises(RxAxisServerError):
            await self.service.move_to(4.5, 0.5)

    async def test_move_to_rejects_speed_above_server_limit(self) -> None:
        with self.assertRaises(RxAxisServerError):
            await self.service.move_to(1.0, 10.0)
        self.assertEqual(self.transport.sends, [])

    async def test_stop_motor_sends_duty_cycle_zero(self) -> None:
        await self.service.stop_motor()
        self.assertEqual(len(self.transport.sends), 1)
        can_id, payload, _ = self.transport.sends[0]
        self.assertEqual(can_id, RX_DUTY_ID)
        self.assertEqual(payload.hex().upper(), DUTY_CYCLE_ZERO_PAYLOAD)

    async def test_set_velocity_sends_rpm_packet_and_arms_refresh(self) -> None:
        await self.service.poll_once()  # feedback (pos = pi) for the window guard
        await self.service.set_velocity(0.5)
        can_id, payload, extended = self.transport.sends[-1]
        self.assertEqual(can_id, rpm_can_id(self.service.config.selected_can_node_id))
        self.assertEqual(
            payload.hex().upper(),
            encode_rpm_payload(0.5, gear_ratio=1.0, pole_pairs=1),
        )
        self.assertTrue(extended)
        # The velocity payload is the ACTIVE command (refresh keeps it alive).
        self.assertEqual(self.service._active_command_payload, payload)
        self.assertEqual(self.service._velocity_cmd["velocity_rad_s"], 0.5)

    async def test_set_velocity_zero_needs_no_feedback(self) -> None:
        # Zero is the safe direction: accepted even before the first feedback.
        out = await self.service.set_velocity(0.0)
        self.assertEqual(self.service._velocity_cmd["velocity_rad_s"], 0.0)
        self.assertNotIn("held", out)

    async def test_set_velocity_rejects_speed_above_server_limit(self) -> None:
        await self.service.poll_once()
        with self.assertRaises(RxAxisServerError):
            await self.service.set_velocity(6.0)  # server max is 5.0

    async def test_set_velocity_rejects_before_homing(self) -> None:
        self.service._homed = False
        with self.assertRaises(RxAxisServerError):
            await self.service.set_velocity(0.5)

    async def test_set_velocity_holds_at_window_edge(self) -> None:
        # Feedback position pi (~3.14) sits above a 3.0 rad max window: a
        # positive velocity must be refused down to ZERO, not run outward.
        service = RxAxisService(
            make_config(),
            config_from_args(make_server_args(max_command_position_rad=3.0)),
            transport=self.transport,
            limit_switch_reader=FakeLimitSwitchReader([]),
        )
        service._homed = True
        await service.poll_once()
        out = await service.set_velocity(0.5)
        self.assertEqual(out["held"], "max_window")
        self.assertEqual(service._velocity_cmd["velocity_rad_s"], 0.0)
        _, payload, _ = self.transport.sends[-1]
        self.assertEqual(
            payload.hex().upper(), encode_rpm_payload(0.0, gear_ratio=1.0, pole_pairs=1)
        )

    async def test_move_to_cancels_velocity_mode(self) -> None:
        await self.service.poll_once()
        await self.service.set_velocity(0.5)
        await self.service.move_to(1.0, 0.5)
        self.assertIsNone(self.service._velocity_cmd)

    async def test_stop_closes_transport(self) -> None:
        await self.service.stop()
        self.assertTrue(self.transport.closed)

    async def test_move_to_arms_streaming_and_stop_disarms(self) -> None:
        # move_to should latch the position-velocity payload as the active
        # streaming command; stop_motor should clear it so we don't keep
        # re-sending stale targets.
        await self.service.move_to(1.0, 0.5)
        self.assertIsNotNone(self.service._active_command_payload)
        self.assertEqual(self.service._active_command_can_id, RX_COMMAND_ID)
        await self.service.stop_motor()
        self.assertIsNone(self.service._active_command_payload)

    async def test_motor_position_inverted_flips_only_the_wire_layer(self) -> None:
        # When the motor's "+ direction" is toward the home switch, set
        # rx_motor_position_inverted=True in GantryConfig. The higher-level API
        # still works in positive-operating-envelope units; only the encoded
        # position on the wire is sign-flipped, and feedback positions are
        # flipped back.
        inverted_config = GantryConfig(
            axis="rx",
            motor_pole_pairs=TEST_POLES,
            motor_gear_ratio=TEST_GEAR,
            rx_motor_position_inverted=True,
        )
        transport = FakeCanTransport()
        service = RxAxisService(
            inverted_config,
            config_from_args(make_server_args()),
            transport=transport,
            limit_switch_reader=FakeLimitSwitchReader([]),
        )
        service._homed = True
        await service.move_to(1.0, 0.5)
        # The wire payload should reflect the FLIPPED position (-1.0 rad), even
        # though the API was called with +1.0 rad.
        expected = encode_position_velocity_payload(
            1.0,
            0.5,
            service.server_config.default_accel_rad_s2,
            gear_ratio=TEST_GEAR,
            pole_pairs=TEST_POLES,
            position_inverted=True,
        )
        self.assertEqual(transport.sends[0][1].hex().upper(), expected)
