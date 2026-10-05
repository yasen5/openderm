from __future__ import annotations

import sys
import types
import unittest
from unittest import mock


from capture.sensors.limit_switches import (
    GpiodLimitSwitchReader,
    LimitSwitchConfig,
    LimitSwitchReader,
    pressed_message,
    watch_limit_switches,
)


class FakeButton:
    instances: list["FakeButton"] = []

    def __init__(self, pin: int, *, pull_up: bool, bounce_time: float) -> None:
        self.pin = pin
        self.pull_up = pull_up
        self.bounce_time = bounce_time
        self.when_pressed = None
        self.closed = False
        self.is_pressed = False
        self.instances.append(self)

    def close(self) -> None:
        self.closed = True

    def press(self) -> None:
        self.is_pressed = True
        if self.when_pressed is not None:
            self.when_pressed()


class FakeLineValue:
    ACTIVE = object()
    INACTIVE = object()


class FakeLine:
    class Bias:
        PULL_UP = "pull-up"
        PULL_DOWN = "pull-down"

    class Direction:
        INPUT = "input"

    Value = FakeLineValue


class FakeLineRequest:
    def __init__(self) -> None:
        self.values: dict[int, object] = {}
        self.released = False

    def get_value(self, pin: int) -> object:
        return self.values.get(pin, FakeLineValue.INACTIVE)

    def release(self) -> None:
        self.released = True


class FakeGpiod(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(line=FakeLine)
        self.request = FakeLineRequest()
        self.requested_chip = None
        self.requested_config = None

    def LineSettings(self, **kwargs):
        return kwargs

    def request_lines(self, chip, *, consumer, config):
        self.requested_chip = chip
        self.requested_config = config
        return self.request


class LimitSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeButton.instances = []

    def test_pressed_message_includes_switch_name_and_pin(self) -> None:
        self.assertIn("RX left limit switch pressed on GPIO 26", pressed_message("RX left", 26))

    def test_watch_limit_switches_prints_for_each_pressed_switch(self) -> None:
        printed: list[str] = []

        def pause_once() -> None:
            for button in FakeButton.instances:
                button.press()

        result = watch_limit_switches(
            LimitSwitchConfig(
                rx_left_pin=26,
                rx_right_pin=16,
                pull_up=True,
                bounce_time_s=0.05,
            ),
            button_factory=FakeButton,
            pause_fn=pause_once,
            print_fn=printed.append,
        )

        self.assertEqual(result, 0)
        self.assertEqual([button.pin for button in FakeButton.instances], [26, 16])
        self.assertTrue(all(button.pull_up for button in FakeButton.instances))
        self.assertEqual([button.bounce_time for button in FakeButton.instances], [0.05, 0.05])
        self.assertIn("RX left limit switch pressed on GPIO 26", printed[1])
        self.assertIn("RX right limit switch pressed on GPIO 16", printed[2])
        self.assertTrue(all(button.closed for button in FakeButton.instances))

    def test_limit_switch_reader_reports_left_pressed(self) -> None:
        reader = LimitSwitchReader(
            LimitSwitchConfig(rx_left_pin=26, rx_right_pin=16),
            button_factory=FakeButton,
        )
        FakeButton.instances[0].is_pressed = True
        self.assertTrue(reader.rx_left_pressed())
        self.assertFalse(reader.rx_right_pressed())
        reader.close()
        self.assertTrue(all(button.closed for button in FakeButton.instances))

    def test_gpiod_limit_switch_reader_reports_active_press(self) -> None:
        fake_gpiod = FakeGpiod()
        with mock.patch.dict(sys.modules, {"gpiod": fake_gpiod}):
            reader = GpiodLimitSwitchReader(LimitSwitchConfig(rx_left_pin=26, rx_right_pin=16))
        fake_gpiod.request.values[26] = FakeLineValue.ACTIVE
        fake_gpiod.request.values[16] = FakeLineValue.INACTIVE
        self.assertTrue(reader.rx_left_pressed())
        self.assertFalse(reader.rx_right_pressed())
        reader.close()
        self.assertTrue(fake_gpiod.request.released)


if __name__ == "__main__":
    unittest.main()
