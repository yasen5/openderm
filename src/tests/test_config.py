from __future__ import annotations

import os
import unittest
from unittest import mock


from openderm.config import GantryConfig


class GantryConfigTests(unittest.TestCase):
    def test_with_overrides_uses_axis_specific_default_limits(self) -> None:
        config = GantryConfig()

        y_config = config.with_overrides(axis="y")
        z_config = config.with_overrides(axis="z")

        self.assertEqual((y_config.travel_min_mm, y_config.travel_max_mm), (0.0, 665.0))
        self.assertEqual((z_config.travel_min_mm, z_config.travel_max_mm), (0.0, 392.0))

    def test_rx_uses_configured_default_interface(self) -> None:
        config = GantryConfig()

        self.assertEqual(config.with_overrides(axis="rx").selected_can_interface, "can1")

    def test_rx_can_interface_from_env(self) -> None:
        env = os.environ | {"GANTRY_RX_CAN_INTERFACE": "can2"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = GantryConfig.from_env()
        self.assertEqual(config.with_overrides(axis="rx").selected_can_interface, "can2")

    def test_rx_axis_server_url_dispatches_to_rx(self) -> None:
        config = GantryConfig(rx_axis_server_url="http://rx-host:1234")
        self.assertEqual(
            config.with_overrides(axis="rx").selected_rx_axis_server_url,
            "http://rx-host:1234",
        )

    def test_rx_axis_server_url_from_env(self) -> None:
        env = os.environ | {"RX_AXIS_SERVER_URL": "http://rx:1"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = GantryConfig.from_env()
        self.assertEqual(config.rx_axis_server_url, "http://rx:1")

    def test_motor_position_inverted_dispatches_to_rx(self) -> None:
        config = GantryConfig(rx_motor_position_inverted=True)
        self.assertTrue(config.with_overrides(axis="rx").motor_position_inverted)

    def test_motor_position_inverted_reads_rx_env(self) -> None:
        env = os.environ | {"MOTOR_POSITION_INVERTED_RX": "false"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = GantryConfig.from_env()
        self.assertFalse(config.rx_motor_position_inverted)

    def test_non_rx_axis_is_not_a_can_axis(self) -> None:
        config = GantryConfig().with_overrides(axis="x")
        self.assertFalse(config.is_can_axis)
        with self.assertRaisesRegex(ValueError, "RX CAN axis"):
            _ = config.selected_can_interface


if __name__ == "__main__":
    unittest.main()
