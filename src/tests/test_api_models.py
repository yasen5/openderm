from __future__ import annotations

import math
import unittest

from pydantic import ValidationError


from capture.motion.api_models import (
    GantryHomeRequest,
    GantryMoveRequest,
    GantryStopRequest,
    GantryStreamStartRequest,
    GantryStreamTargetRequest,
    RxMoveToRequest,
    RxVelocityRequest,
)


class GantryRequestModelTests(unittest.TestCase):
    def test_move_accepts_the_documented_shape(self) -> None:
        request = GantryMoveRequest(
            x=12,
            feed_mm_s=5,
            tolerance_mm=0.1,
            commander_id="scan-1",
            blocking=True,
        )
        self.assertEqual(request.x, 12.0)
        self.assertTrue(request.blocking)

    def test_move_rejects_missing_unknown_and_coerced_fields(self) -> None:
        invalid_payloads = (
            {},
            {"x": 1.0, "y": 2.0},
            {"x": "1.0"},
            {"x": True},
            {"x": 1.0, "blocking": "false"},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                GantryMoveRequest.model_validate(payload)

    def test_motion_numbers_must_be_finite_and_in_range(self) -> None:
        invalid_payloads = (
            {"x": math.nan},
            {"x": math.inf},
            {"x": 1.0, "feed_mm_s": 0.0},
            {"x": 1.0, "tolerance_mm": -0.1},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                GantryMoveRequest.model_validate(payload)

        with self.assertRaises(ValidationError):
            GantryStreamTargetRequest(x=-math.inf)

    def test_home_stop_and_stream_enforce_closed_contracts(self) -> None:
        self.assertEqual(GantryHomeRequest().axes, ("x",))
        self.assertEqual(GantryStopRequest().mode, "emergency")
        with self.assertRaises(ValidationError):
            GantryHomeRequest(axes=("y",))
        with self.assertRaises(ValidationError):
            GantryStopRequest(mode="pause")
        with self.assertRaises(ValidationError):
            GantryStreamStartRequest(feed_mm_s=1.0, feed_mm_min=60.0)
        with self.assertRaises(ValidationError):
            GantryStreamStartRequest(tick_s=0.0)

    def test_stream_feed_units_resolve_consistently(self) -> None:
        request = GantryStreamStartRequest(feed_mm_min=120)
        self.assertEqual(request.resolved_feed_mm_s, 2.0)


class RxRequestModelTests(unittest.TestCase):
    def test_move_requires_finite_position_and_positive_limits(self) -> None:
        request = RxMoveToRequest(
            position_rad=1,
            speed_rad_s=0.5,
            accel_rad_s2=10,
        )
        self.assertEqual(request.position_rad, 1.0)
        invalid_payloads = (
            {},
            {"position_rad": "1.0"},
            {"position_rad": math.nan},
            {"position_rad": 1.0, "speed_rad_s": 0.0},
            {"position_rad": 1.0, "accel_rad_s2": -1.0},
            {"position_rad": 1.0, "unexpected": 1},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                RxMoveToRequest.model_validate(payload)

    def test_velocity_requires_one_finite_numeric_value(self) -> None:
        self.assertEqual(RxVelocityRequest(velocity_rad_s=-0.5).velocity_rad_s, -0.5)
        for payload in (
            {},
            {"velocity_rad_s": "0.5"},
            {"velocity_rad_s": math.inf},
            {"velocity_rad_s": 0.5, "extra": True},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                RxVelocityRequest.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
