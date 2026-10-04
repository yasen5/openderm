from __future__ import annotations

import argparse
import io
import unittest
from unittest import mock


from openderm.sensors.hg_c import (
    HgCSensorConfig,
    HgCSensorController,
    SensorChannelConfig,
    build_parser,
    run,
)


class FakeVoltageReader:
    def __init__(self, voltages_by_channel: dict[int, list[float]]) -> None:
        self._voltages_by_channel = {
            channel: list(values) for channel, values in voltages_by_channel.items()
        }

    def read_voltage(self, channel: int) -> float:
        values = self._voltages_by_channel[channel]
        if len(values) > 1:
            return values.pop(0)
        return values[0]


class FakePinController:
    def __init__(self) -> None:
        self.values: dict[int, bool] = {}

    def setup_output(self, pin: int, *, initial_high: bool = False) -> None:
        self.values[pin] = initial_high

    def write(self, pin: int, high: bool) -> None:
        self.values[pin] = high

    def close(self) -> None:
        return


class HgCSensorTests(unittest.TestCase):
    def make_controller(
        self,
        *,
        voltages_by_channel: dict[int, list[float]] | None = None,
        config: HgCSensorConfig | None = None,
    ) -> HgCSensorController:
        resolved_config = config or HgCSensorConfig(
            settle_time_s=0.0,
            sensors=(
                SensorChannelConfig(name="sensor1", adc_channel=2, gpio_pin=5),
                SensorChannelConfig(name="sensor2", adc_channel=1, gpio_pin=6),
            ),
        )
        return HgCSensorController(
            config=resolved_config,
            voltage_reader=FakeVoltageReader(
                voltages_by_channel
                or {
                    1: [1.0],
                    2: [4.0],
                }
            ),
            pin_controller=FakePinController(),
        )

    def test_distance_mapping_uses_default_hg_c_range(self) -> None:
        controller = self.make_controller()
        self.assertAlmostEqual(controller.current_from_voltage(0.6), 4.0)
        self.assertAlmostEqual(controller.current_from_voltage(1.8), 12.0)
        self.assertAlmostEqual(controller.current_from_voltage(3.0), 20.0)
        self.assertEqual(controller.distance_from_current(4.0), 135.0)
        self.assertEqual(controller.distance_from_current(12.0), 100.0)
        self.assertEqual(controller.distance_from_current(20.0), 65.0)

    def test_distance_mapping_returns_none_outside_sensor_output_range(self) -> None:
        controller = self.make_controller()
        self.assertIsNone(controller.distance_from_current(3.99))
        self.assertIsNone(controller.distance_from_current(20.01))

    def test_read_sensor_averages_samples_and_tracks_enabled_state(self) -> None:
        controller = self.make_controller(
            voltages_by_channel={0: [2.0, 3.0], 1: [1.0], 2: [2.0, 3.0]}
        )
        controller.set_enabled("sensor1", True)
        reading = controller.read_sensor("sensor1", samples=2)
        self.assertTrue(reading.enabled)
        self.assertEqual(reading.voltage_v, 2.5)
        self.assertAlmostEqual(reading.current_ma, 16.666667, places=5)
        self.assertAlmostEqual(reading.distance_mm, 79.583333, places=5)

    def test_default_enable_polarity_is_active_low(self) -> None:
        controller = self.make_controller()
        fake_pins = controller._pin_controller.values  # noqa: SLF001
        self.assertTrue(fake_pins[5])
        controller.set_enabled("sensor1", True)
        self.assertFalse(fake_pins[5])
        controller.set_enabled("sensor1", False)
        self.assertTrue(fake_pins[5])

    def test_run_read_enables_then_disables_selected_sensor(self) -> None:
        controller = self.make_controller()
        args = argparse.Namespace(command="read", sensor="sensor2", samples=None)
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = run(args, controller)
        self.assertEqual(result, 0)
        rendered = stdout.getvalue()
        self.assertIn('"name": "sensor2"', rendered)
        self.assertFalse(controller._enabled_by_name["sensor2"])  # noqa: SLF001

    def test_run_enable_leaves_selected_sensor_enabled(self) -> None:
        controller = self.make_controller()
        args = argparse.Namespace(command="enable", sensor="sensor1")
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = run(args, controller)
        self.assertEqual(result, 0)
        self.assertIn('"enabled": true', stdout.getvalue())
        self.assertTrue(controller._enabled_by_name["sensor1"])  # noqa: SLF001

    def test_run_disable_drives_selected_sensor_low(self) -> None:
        controller = self.make_controller()
        controller.set_enabled("sensor1", True)
        args = argparse.Namespace(command="disable", sensor="sensor1")
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = run(args, controller)
        self.assertEqual(result, 0)
        self.assertIn('"enabled": false', stdout.getvalue())
        self.assertFalse(controller._enabled_by_name["sensor1"])  # noqa: SLF001

    def test_parser_supports_watch_interval(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["watch", "--sensor", "sensor2", "--interval-s", "0.5"])
        self.assertEqual(args.command, "watch")
        self.assertEqual(args.sensor, "sensor2")
        self.assertEqual(args.interval_s, 0.5)

    def test_parser_supports_enable_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["enable", "--sensor", "sensor1"])
        self.assertEqual(args.command, "enable")
        self.assertEqual(args.sensor, "sensor1")


if __name__ == "__main__":
    unittest.main()
