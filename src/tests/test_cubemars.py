from __future__ import annotations

import math
import unittest


from capture.motion.cubemars import (
    CAN_PACKET_SET_DUTY,
    CAN_PACKET_SET_ORIGIN_HERE,
    CAN_PACKET_SET_POS_SPD,
    CAN_PACKET_SET_RPM,
    CAN_PACKET_FEEDBACK,
    CubeMarsError,
    decode_feedback_payload,
    duty_cycle_can_id,
    encode_position_velocity_payload,
    encode_rpm_payload,
    feedback_can_id,
    position_velocity_can_id,
    rpm_can_id,
    set_origin_can_id,
)


# Tests deliberately set gear_ratio=1.0 and pole_pairs=1 so the wire arithmetic stays
# easy to verify by hand. The AK45-36 KV80 production defaults are validated separately.
TEST_GEAR = 1.0
TEST_POLES = 1


class CanIdTests(unittest.TestCase):
    def test_position_velocity_can_id_packs_packet6_and_node(self) -> None:
        # AK servo position-velocity-loop packet ID is 6; encoded as (6 << 8) | node.
        self.assertEqual(position_velocity_can_id(0x01), 0x0601)
        self.assertEqual(CAN_PACKET_SET_POS_SPD, 0x06)

    def test_set_origin_can_id_uses_packet5(self) -> None:
        self.assertEqual(set_origin_can_id(0x02), 0x0502)
        self.assertEqual(CAN_PACKET_SET_ORIGIN_HERE, 0x05)

    def test_duty_cycle_can_id_uses_packet0(self) -> None:
        self.assertEqual(duty_cycle_can_id(0x07), 0x0007)
        self.assertEqual(CAN_PACKET_SET_DUTY, 0x00)

    def test_rpm_can_id_uses_packet3(self) -> None:
        self.assertEqual(rpm_can_id(0x01), 0x0301)
        self.assertEqual(CAN_PACKET_SET_RPM, 0x03)

    def test_feedback_can_id_uses_packet0x29(self) -> None:
        self.assertEqual(feedback_can_id(0x01), 0x2901)
        self.assertEqual(CAN_PACKET_FEEDBACK, 0x29)

    def test_can_id_rejects_oversized_node(self) -> None:
        with self.assertRaises(CubeMarsError):
            position_velocity_can_id(0x100)


class EncodingTests(unittest.TestCase):
    def test_encode_packs_int32_be_position_int16_be_speed_int16_be_accel(self) -> None:
        # Position is output-shaft degrees × 10000 (no gear correction by us;
        # the driver firmware applies its configured reduction internally).
        # π rad → 180° × 10000 = 1,800,000 = 0x001B7740.
        # Speed/accel: 10 rad/s × 1 × 60/(2π) ≈ 95.49 motor RPM × 1 pole pair = 95.49 ERPM → /10 ≈ 10 = 0x000A.
        payload = encode_position_velocity_payload(
            math.pi, 10.0, 10.0, gear_ratio=TEST_GEAR, pole_pairs=TEST_POLES
        )
        self.assertEqual(payload, "001B7740000A000A")

    def test_encode_with_ak45_36_defaults(self) -> None:
        # Realistic AK45-36 KV80 calibration: gear 36, pole pairs 14.
        # 0.1 rad output → 5.7296° × 10000 = 57296 = 0x0000DFD0.
        # 0.5 rad/s output → 0.5 × 36 × 9.549 × 14 / 10 ≈ 241 = 0x00F1 ERPM-wire units.
        # 1.0 rad/s² output → ≈ 481 = 0x01E1 ERPM/s²-wire units.
        payload = encode_position_velocity_payload(0.1, 0.5, 1.0, gear_ratio=36.0, pole_pairs=14)
        self.assertEqual(payload, "0000DFD000F101E1")

    def test_encode_rpm_packs_int32_be_full_erpm(self) -> None:
        # Speed loop is FULL ERPM in one int32 (not the pos-vel int16 ERPM/10):
        # 10 rad/s x 1 x 60/(2pi) = 95.49 motor RPM x 1 pole pair -> 95 ERPM = 0x0000005F.
        payload = encode_rpm_payload(10.0, gear_ratio=TEST_GEAR, pole_pairs=TEST_POLES)
        self.assertEqual(payload, "0000005F")

    def test_encode_rpm_with_ak45_36_defaults(self) -> None:
        # 0.5 rad/s output x 36 x 9.549 = 171.89 motor RPM x 14 pole pairs
        # -> 2406 ERPM = 0x00000966.
        payload = encode_rpm_payload(0.5, gear_ratio=36.0, pole_pairs=14)
        self.assertEqual(payload, "00000966")

    def test_encode_rpm_is_signed_and_inversion_flips_direction(self) -> None:
        # Speed mode is DIRECTIONAL: negative velocity is a negative int32, and
        # position_inverted flips the wire sign exactly like positions.
        neg = encode_rpm_payload(-0.5, gear_ratio=36.0, pole_pairs=14)
        self.assertEqual(neg, "FFFFF69A")  # -2406 two's complement
        inverted = encode_rpm_payload(0.5, gear_ratio=36.0, pole_pairs=14, position_inverted=True)
        self.assertEqual(inverted, neg)

    def test_encode_rpm_rejects_out_of_ak_range(self) -> None:
        # The AK speed loop accepts -100000..100000 ERPM; reject, never truncate.
        with self.assertRaises(CubeMarsError):
            encode_rpm_payload(20000.0, gear_ratio=TEST_GEAR, pole_pairs=TEST_POLES)

    def test_encode_rejects_position_out_of_int32_range(self) -> None:
        with self.assertRaises(CubeMarsError):
            encode_position_velocity_payload(
                1.0e6, 1.0, 1.0, gear_ratio=TEST_GEAR, pole_pairs=TEST_POLES
            )

    def test_encode_rejects_negative_acceleration(self) -> None:
        with self.assertRaises(CubeMarsError):
            encode_position_velocity_payload(
                0.1, 0.5, -1.0, gear_ratio=TEST_GEAR, pole_pairs=TEST_POLES
            )

    def test_encode_with_position_inverted_flips_position_only(self) -> None:
        # Position is sign-flipped on the wire; speed and accel are unchanged
        # (they're magnitudes here).
        upright = encode_position_velocity_payload(
            0.1, 0.5, 1.0, gear_ratio=TEST_GEAR, pole_pairs=TEST_POLES
        )
        inverted = encode_position_velocity_payload(
            0.1,
            0.5,
            1.0,
            gear_ratio=TEST_GEAR,
            pole_pairs=TEST_POLES,
            position_inverted=True,
        )
        # The position halves should be negatives of each other; the speed/accel
        # halves (last 4 bytes) should be identical.
        self.assertEqual(upright[8:], inverted[8:])
        upright_pos = int(upright[:8], 16)
        inverted_pos = int(inverted[:8], 16)
        if upright_pos >= 0x80000000:
            upright_pos -= 0x100000000
        if inverted_pos >= 0x80000000:
            inverted_pos -= 0x100000000
        self.assertEqual(upright_pos, -inverted_pos)


class DecodingTests(unittest.TestCase):
    def test_decode_unpacks_servo_feedback_layout(self) -> None:
        # Feedback layout: int16 output_deg×10, int16 motor_ERPM/10, int16 current×100, int8 temp, uint8 error.
        # pos_wire=1800 → 180° output → π rad (no gear correction; firmware reports post-gearbox).
        # cur_wire=50 → 0.5 A. temp=30. error=0.
        payload = bytes.fromhex("0708000000321E00")
        fb = decode_feedback_payload(payload, gear_ratio=TEST_GEAR, pole_pairs=TEST_POLES)
        self.assertAlmostEqual(fb.position_rad, math.pi, places=3)
        self.assertEqual(fb.speed_rad_s, 0.0)
        self.assertAlmostEqual(fb.current_a, 0.5)
        self.assertEqual(fb.motor_temperature_c, 30)
        self.assertEqual(fb.error_code, 0)

    def test_decode_position_ignores_gear_ratio(self) -> None:
        # Position decoding is gear-free; the same feedback bytes decode identically
        # whether we pass gear=1 or gear=36 (only speed differs).
        payload = bytes.fromhex("0708000000321E00")
        fb1 = decode_feedback_payload(payload, gear_ratio=1.0, pole_pairs=1)
        fb36 = decode_feedback_payload(payload, gear_ratio=36.0, pole_pairs=14)
        self.assertEqual(fb1.position_rad, fb36.position_rad)

    def test_decode_with_position_inverted_flips_position_and_speed(self) -> None:
        # A non-zero speed payload so we can verify both fields flip.
        payload = bytes.fromhex("070800640000 1E 00".replace(" ", ""))
        upright = decode_feedback_payload(payload, gear_ratio=TEST_GEAR, pole_pairs=TEST_POLES)
        inverted = decode_feedback_payload(
            payload,
            gear_ratio=TEST_GEAR,
            pole_pairs=TEST_POLES,
            position_inverted=True,
        )
        self.assertEqual(upright.position_rad, -inverted.position_rad)
        self.assertEqual(upright.speed_rad_s, -inverted.speed_rad_s)
        # Current and temperature are unaffected.
        self.assertEqual(upright.current_a, inverted.current_a)
        self.assertEqual(upright.motor_temperature_c, inverted.motor_temperature_c)

    def test_decode_handles_signed_negatives_and_error_codes(self) -> None:
        payload = bytes.fromhex("FFFFFFFFFFFFFF07")  # motor stall error code = 7
        fb = decode_feedback_payload(payload, gear_ratio=TEST_GEAR, pole_pairs=TEST_POLES)
        # int16 0xFFFF = -1 → -0.1° → -0.001745 rad
        self.assertAlmostEqual(fb.position_rad, -0.1 * math.pi / 180.0, places=6)
        self.assertEqual(fb.error_code, 7)

    def test_decode_rejects_wrong_payload_length(self) -> None:
        with self.assertRaises(CubeMarsError):
            decode_feedback_payload(b"\x00" * 7, gear_ratio=TEST_GEAR, pole_pairs=TEST_POLES)


if __name__ == "__main__":
    unittest.main()
