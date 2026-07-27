from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from openderm.scanning._contour.recovery import attempt_edge_recovery


class ContourRecoveryTests(unittest.TestCase):
    def test_exhausted_probe_returns_last_commanded_rx_position(self) -> None:
        commanded: list[float] = []
        out_of_range = SimpleNamespace(in_range=False, distance_mm=None)

        def set_rx_absolute(target: float) -> float:
            commanded.append(target)
            return target

        context = SimpleNamespace(
            args=SimpleNamespace(
                edge_tilt_sign=1.0,
                edge_tilt_max_deg=10.0,
                edge_tilt_step_deg=5.0,
                edge_recover_window_mm=10.0,
                target_mm=110.0,
                debug=False,
            ),
            read_exclusive=lambda _sensor_name: out_of_range,
            read_rx_rad=lambda: 0.0,
            rx_move_fail={"reason": None},
            set_rx_absolute=set_rx_absolute,
            stop_state={"requested": False},
        )

        recovered, rx_used, limited = attempt_edge_recovery(
            context,
            level_ref=0.0,
            phase_dir=1,
        )

        self.assertFalse(recovered)
        self.assertFalse(limited)
        self.assertEqual(len(commanded), 2)
        self.assertAlmostEqual(commanded[-1], math.radians(10.0))
        self.assertAlmostEqual(rx_used, commanded[-1])
