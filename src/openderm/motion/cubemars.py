from __future__ import annotations

from dataclasses import dataclass
import math
import struct


class CubeMarsError(RuntimeError):
    """Raised when a CubeMars CAN command cannot be encoded or sent."""


# AK series servo-mode packet IDs (AK driver manual section 5.1).
CAN_PACKET_SET_DUTY = 0x00
CAN_PACKET_SET_CURRENT = 0x01
CAN_PACKET_SET_CURRENT_BRAKE = 0x02
CAN_PACKET_SET_RPM = 0x03
CAN_PACKET_SET_POS = 0x04
CAN_PACKET_SET_ORIGIN_HERE = 0x05
CAN_PACKET_SET_POS_SPD = 0x06
CAN_PACKET_FEEDBACK = 0x29

CAN_EFF_FLAG = 0x80000000
CAN_EFF_MASK = 0x1FFFFFFF

# Set-origin payloads (1 byte): 0x00 sets a temporary zero, 0x01 a permanent zero (dual-encoder models only).
SET_ORIGIN_TEMP_PAYLOAD = "00"
SET_ORIGIN_PERMANENT_PAYLOAD = "01"

# 4-byte int32 duty = 0 — sent as a "disable drive" stop command in servo mode.
DUTY_CYCLE_ZERO_PAYLOAD = "00000000"

SERVO_ERROR_CODES: dict[int, str] = {
    0: "no_fault",
    1: "motor_over_temperature",
    2: "over_current",
    3: "over_voltage",
    4: "under_voltage",
    5: "encoder_fault",
    6: "mosfet_over_temperature",
    7: "motor_stall",
}


@dataclass(frozen=True)
class CubeMarsFeedback:
    position_rad: float
    speed_rad_s: float
    current_a: float
    motor_temperature_c: int
    error_code: int


# The AK45-36 driver firmware applies the configured reduction ratio internally:
# position commands and feedback are in OUTPUT-shaft degrees. The speed/accel fields,
# however, are in ERPM (electrical RPM of the rotor), so we still have to convert
# output rad/s → motor mechanical RPM (× gear) → ERPM (× pole pairs) ourselves.


def _rad_to_output_deg_x10000(rad: float) -> int:
    output_deg = rad * (180.0 / math.pi)
    return int(round(output_deg * 10000.0))


def _output_deg_x10_to_rad(wire: int) -> float:
    output_deg = wire * 0.1
    return output_deg * (math.pi / 180.0)


def _rad_s_to_motor_erpm_div10(rad_s: float, gear_ratio: float, pole_pairs: int) -> int:
    motor_rpm = rad_s * gear_ratio * (60.0 / (2.0 * math.pi))
    motor_erpm = motor_rpm * pole_pairs
    return int(round(motor_erpm / 10.0))


def _motor_erpm_x10_to_rad_s(wire: int, gear_ratio: float, pole_pairs: int) -> float:
    motor_erpm = wire * 10.0
    motor_rpm = motor_erpm / pole_pairs
    return (motor_rpm / gear_ratio) * (2.0 * math.pi / 60.0)


def encode_position_velocity_payload(
    position_rad: float,
    velocity_rad_s: float,
    accel_rad_s2: float,
    *,
    gear_ratio: float,
    pole_pairs: int,
    position_inverted: bool = False,
) -> str:
    """Pack an AK servo-mode position-velocity-loop payload (8 bytes, big-endian).

    Layout: int32 output-deg×10000, int16 motor-ERPM/10, int16 motor-ERPM/s²/10.
    Position is post-gearbox; speed and accel are rotor electrical RPM.

    When `position_inverted` is True, the position sign is flipped on the wire so
    callers can treat positive as "away from the home switch" regardless of the
    motor's physical wiring. Speed and accel are positive magnitudes here, so
    they are unaffected.
    """
    pos_wire = _rad_to_output_deg_x10000(-position_rad if position_inverted else position_rad)
    spd_wire = _rad_s_to_motor_erpm_div10(velocity_rad_s, gear_ratio, pole_pairs)
    acc_wire = _rad_s_to_motor_erpm_div10(accel_rad_s2, gear_ratio, pole_pairs)
    if not -(1 << 31) <= pos_wire <= (1 << 31) - 1:
        raise CubeMarsError(
            f"Position {position_rad:.4f}rad maps to {pos_wire} which exceeds the int32 range."
        )
    if not -(1 << 15) <= spd_wire <= (1 << 15) - 1:
        raise CubeMarsError(
            f"Velocity {velocity_rad_s:.4f}rad/s maps to {spd_wire} which exceeds the int16 range."
        )
    if not 0 <= acc_wire <= (1 << 15) - 1:
        raise CubeMarsError(
            f"Acceleration {accel_rad_s2:.4f}rad/s² maps to {acc_wire} which must be a non-negative int16."
        )
    return (
        (struct.pack(">i", pos_wire) + struct.pack(">h", spd_wire) + struct.pack(">h", acc_wire))
        .hex()
        .upper()
    )


def position_velocity_can_id(node_id: int) -> int:
    """Extended CAN ID for AK servo-mode position-velocity-loop command (packet ID 6)."""
    if not 0 <= node_id <= 0xFF:
        raise CubeMarsError(f"CAN node ID must fit in 8 bits. Received {node_id}.")
    return (CAN_PACKET_SET_POS_SPD << 8) | node_id


def _rad_s_to_motor_erpm(rad_s: float, gear_ratio: float, pole_pairs: int) -> int:
    motor_rpm = rad_s * gear_ratio * (60.0 / (2.0 * math.pi))
    return int(round(motor_rpm * pole_pairs))


def encode_rpm_payload(
    velocity_rad_s: float,
    *,
    gear_ratio: float,
    pole_pairs: int,
    position_inverted: bool = False,
) -> str:
    """Pack an AK servo-mode speed-loop (SET_RPM, packet ID 3) payload.

    Layout: one big-endian int32 of motor ERPM (electrical RPM of the rotor) --
    FULL ERPM, unlike the position-velocity payload's int16 ERPM/10 fields.
    The value is SIGNED (direction matters in speed mode), so with
    `position_inverted` the sign flips on the wire exactly like positions do
    (the feedback decoder flips speed the same way), keeping "positive = away
    from the home switch" for callers regardless of the motor's wiring. The
    AK servo firmware accepts -100000..100000 ERPM; a command mapping outside
    that is rejected here rather than silently truncated.
    """
    wire = _rad_s_to_motor_erpm(
        -velocity_rad_s if position_inverted else velocity_rad_s,
        gear_ratio,
        pole_pairs,
    )
    if not -100000 <= wire <= 100000:
        raise CubeMarsError(
            f"Velocity {velocity_rad_s:.4f}rad/s maps to {wire} ERPM, outside "
            "the AK speed-mode range -100000..100000."
        )
    return struct.pack(">i", wire).hex().upper()


def rpm_can_id(node_id: int) -> int:
    """Extended CAN ID for AK servo-mode speed-loop command (packet ID 3)."""
    if not 0 <= node_id <= 0xFF:
        raise CubeMarsError(f"CAN node ID must fit in 8 bits. Received {node_id}.")
    return (CAN_PACKET_SET_RPM << 8) | node_id


def set_origin_can_id(node_id: int) -> int:
    """Extended CAN ID for AK servo-mode set-origin command (packet ID 5)."""
    if not 0 <= node_id <= 0xFF:
        raise CubeMarsError(f"CAN node ID must fit in 8 bits. Received {node_id}.")
    return (CAN_PACKET_SET_ORIGIN_HERE << 8) | node_id


def duty_cycle_can_id(node_id: int) -> int:
    """Extended CAN ID for AK servo-mode duty-cycle command (packet ID 0)."""
    if not 0 <= node_id <= 0xFF:
        raise CubeMarsError(f"CAN node ID must fit in 8 bits. Received {node_id}.")
    return (CAN_PACKET_SET_DUTY << 8) | node_id


def feedback_can_id(node_id: int) -> int:
    """Extended CAN ID for AK servo-mode timed feedback frame (packet ID 0x29)."""
    if not 0 <= node_id <= 0xFF:
        raise CubeMarsError(f"CAN node ID must fit in 8 bits. Received {node_id}.")
    return (CAN_PACKET_FEEDBACK << 8) | node_id


def decode_feedback_payload(
    payload: bytes,
    *,
    gear_ratio: float,
    pole_pairs: int,
    position_inverted: bool = False,
) -> CubeMarsFeedback:
    """Decode an AK servo-mode timed feedback frame (8 bytes, big-endian).

    Layout: int16 output-deg×10, int16 motor-ERPM/10, int16 current×100, int8 motor_temp_c, uint8 error.

    When `position_inverted` is True, the position and speed signs are flipped so
    callers see "positive = away from the home switch" regardless of the motor's
    physical wiring. (Velocity sign tracks position-direction sign, so they flip
    together. Current is unsigned in meaning, so it's left alone.)
    """
    if len(payload) != 8:
        raise CubeMarsError(
            f"CubeMars servo-mode feedback payload must be 8 bytes. Received {len(payload)}."
        )
    pos_raw = struct.unpack(">h", payload[0:2])[0]
    spd_raw = struct.unpack(">h", payload[2:4])[0]
    cur_raw = struct.unpack(">h", payload[4:6])[0]
    motor_temp = struct.unpack("b", payload[6:7])[0]
    error_code = payload[7]
    position_rad = _output_deg_x10_to_rad(pos_raw)
    speed_rad_s = _motor_erpm_x10_to_rad_s(spd_raw, gear_ratio, pole_pairs)
    if position_inverted:
        position_rad = -position_rad
        speed_rad_s = -speed_rad_s
    return CubeMarsFeedback(
        position_rad=position_rad,
        speed_rad_s=speed_rad_s,
        current_a=cur_raw * 0.01,
        motor_temperature_c=motor_temp,
        error_code=error_code,
    )
