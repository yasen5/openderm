from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from openderm.app.cli import build_parser as build_motion_parser
from openderm.config import default_pico_port, linear_axis_limits
from openderm.scanning.cli import build_parser, expanded_args
from openderm.scanning.regulate import build_parser as build_regulator_parser


class CaptureCliTests(unittest.TestCase):
    def test_scan_profile_matches_default_rig_settings(self) -> None:
        args = build_parser().parse_args(["captures/example"])
        scanner, values = expanded_args(args, [])
        self.assertEqual(scanner.__name__, "openderm.scanning.contour")
        expected_pairs = {
            "--x-travel-mm": "300",
            "--x-step-mm": "20",
            "--y-step-mm": "15",
            "--target-mm": "110",
            "--z-max-mm": "392",
            "--z-min-mm": "5",
            "--rx-filter-alpha": "0.6",
            "--rx-speed-rad-s": "0.3",
            "--rx-accel-rad-s2": "2.0",
            "--rx-gain-rad-per-mm": "-0.015",
            "--floor-depth-mm": "507",
            "--floor-margin-mm": "3",
            "--band-miss-stop-frac": "0.30",
        }
        for flag, value in expected_pairs.items():
            self.assertEqual(values[values.index(flag) + 1], value)
        for flag in (
            "--record",
            "--band-edge-recovery",
            "--simultaneous-sensors",
        ):
            self.assertIn(flag, values)

    def test_expert_options_are_forwarded_last_as_overrides(self) -> None:
        args = build_parser().parse_args(["captures/example"])
        _, values = expanded_args(args, ["--x-step-mm", "12"])
        self.assertEqual(values[-2:], ["--x-step-mm", "12"])

    def test_scan_profile_parses_in_the_detailed_cli(self) -> None:
        args = build_parser().parse_args(["captures/example"])
        scanner, values = expanded_args(args, [])
        parsed = scanner.build_parser().parse_args(values)
        self.assertEqual(parsed.capture_dir, "captures/example")
        self.assertFalse(hasattr(parsed, "y_backend"))
        self.assertFalse(hasattr(parsed, "z_backend"))

    def test_scanner_exposes_one_motion_and_traversal_path(self) -> None:
        args = build_parser().parse_args(["captures/example"])
        scanner, _values = expanded_args(args, [])
        parser = scanner.build_parser()
        options = {option for action in parser._actions for option in action.option_strings}
        for removed in (
            "--scan-order",
            "--no-z",
            "--no-rx",
            "--no-park",
            "--no-pivot-compensate",
            "--stream",
            "--rx-velocity-servo",
            "--pursuit",
            "--smooth-return",
            "--seek-max-steps",
        ):
            self.assertNotIn(removed, options)

    def test_pivot_model_default_is_portable(self) -> None:
        args = build_parser().parse_args(["captures/example"])
        scanner, _values = expanded_args(args, [])
        parsed = scanner.build_parser().parse_args([])
        self.assertEqual(parsed.rx_pivot_model, "captures/rx_pivot_model.json")

    def test_capture_command_has_no_anatomy_mode(self) -> None:
        parser = build_parser()
        self.assertFalse(any(action.dest == "mode" for action in parser._actions))


class NetworkDefaultsTests(unittest.TestCase):
    def test_motion_cli_exposes_exactly_four_robot_axes(self) -> None:
        parser = build_motion_parser()
        axis_action = next(action for action in parser._actions if action.dest == "axis")
        self.assertEqual(tuple(axis_action.choices), ("x", "y", "z", "rx"))

    def test_pico_bridge_is_derived_from_gantry_host(self) -> None:
        env = {"GANTRY_SERVER_URL": "http://10.0.0.8:8090"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(default_pico_port(), "socket://10.0.0.8:8095")

    def test_explicit_pico_port_wins(self) -> None:
        env = {
            "GANTRY_SERVER_URL": "http://10.0.0.8:8090",
            "PICO_PORT": "/dev/ttyACM1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(default_pico_port(), "/dev/ttyACM1")

    def test_x_limit_is_800_mm(self) -> None:
        self.assertEqual(linear_axis_limits("x"), (0.0, 800.0))


class RegulatorDefaultsTests(unittest.TestCase):
    def test_defaults_match_the_documented_command(self) -> None:
        args = build_regulator_parser().parse_args([])
        self.assertEqual(args.target_mm, 110.0)
        self.assertEqual(args.rx_gain_rad_per_mm, -0.005)
        self.assertEqual(args.rx_max_step_rad, 0.004)
        self.assertEqual(args.rx_filter_alpha, 0.2)
        self.assertEqual(args.samples, 1)
        self.assertEqual(args.rx_speed_rad_s, 0.02)
        self.assertEqual(args.rx_deadband_mm, 0.5)


class KlipperExampleTests(unittest.TestCase):
    def test_x_travel_and_mcu_identifier_are_portable(self) -> None:
        path = Path(__file__).resolve().parents[1] / "klipper" / "printer.cfg"
        contents = path.read_text(encoding="utf-8")
        x_section = contents.split("[stepper_x]", 1)[1].split("[stepper_x1]", 1)[0]
        self.assertIn("position_max: 800", x_section)
        self.assertIn("serial: /dev/serial/by-id/<your-klipper-mcu-id>", contents)

    def test_yz_sections_are_inert_klipper_requirements(self) -> None:
        path = Path(__file__).resolve().parents[1] / "klipper" / "printer.cfg"
        contents = path.read_text(encoding="utf-8").lower()
        self.assertIn("inert required sections", contents)
        self.assertIn("pico exclusively", contents)


if __name__ == "__main__":
    unittest.main()
