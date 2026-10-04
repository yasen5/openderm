from __future__ import annotations

from types import SimpleNamespace
import unittest

from openderm.calibration import AxisSetup, CalibrationSession


class FakeLink:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeAxis:
    def __init__(self, axis: str) -> None:
        self.axis = axis
        self.position = 10.0
        self.homed = False
        self.moves: list[tuple[str, float]] = []

    def home(self) -> None:
        self.homed = True

    def status(self) -> dict:
        return {
            "homed_axes": [self.axis] if self.homed else [],
            "position": {self.axis: self.position},
        }

    def move_to(self, target: float, **_kwargs) -> None:
        self.position = target
        self.moves.append(("to", target))

    def move_by(self, delta: float, **_kwargs) -> None:
        self.position += delta
        self.moves.append(("by", delta))


class FakeSensors:
    def __init__(self) -> None:
        self.enabled = {"sensor1": False, "sensor2": False}
        self.closed = False
        self.fail = False

    def set_enabled_for_selection(self, _selection: str, enabled: bool) -> None:
        for sensor_name in self.enabled:
            self.enabled[sensor_name] = enabled

    def set_enabled(self, sensor_name: str, enabled: bool) -> None:
        self.enabled[sensor_name] = enabled

    def read_sensor(self, sensor_name: str, *, samples: int):
        if self.fail:
            raise RuntimeError("sensor failed")
        return SimpleNamespace(
            name=sensor_name,
            samples=samples,
            distance_mm=100.0,
            in_range=True,
        )

    def close(self) -> None:
        self.closed = True


class CalibrationSessionTests(unittest.TestCase):
    def test_connects_shared_pico_link_and_homes_selected_axes(self) -> None:
        link = FakeLink()
        created: dict[str, FakeAxis] = {}

        def make_axis(_link, axis: str, **_kwargs):
            created[axis] = FakeAxis(axis)
            return created[axis]

        session = CalibrationSession()
        session.connect_pico_axes(
            "fake-port",
            {
                "y": AxisSetup(home=True),
                "z": AxisSetup(home=False),
            },
            open_link=lambda _port, **_kwargs: link,
            axis_client_factory=make_axis,
            timeout_s=1.0,
        )

        self.assertTrue(created["y"].homed)
        self.assertFalse(created["z"].homed)
        self.assertTrue(session.is_homed("y"))
        session.close()
        self.assertTrue(link.closed)

    def test_reads_sensors_exclusively_and_computes_average(self) -> None:
        sensors = FakeSensors()
        session = CalibrationSession()
        session.connect_sensors(lambda: sensors)

        sensor1, sensor2 = session.read_pair(samples=4)

        self.assertEqual(sensor1.samples, 4)
        self.assertEqual(sensor2.samples, 4)
        self.assertEqual(session.average_distance((sensor1, sensor2)), 100.0)
        self.assertFalse(any(sensors.enabled.values()))

    def test_sensor_is_disabled_when_read_raises(self) -> None:
        sensors = FakeSensors()
        sensors.fail = True
        session = CalibrationSession()
        session.connect_sensors(lambda: sensors)

        with self.assertRaisesRegex(RuntimeError, "sensor failed"):
            session.read_exclusive("sensor1", samples=1)

        self.assertFalse(sensors.enabled["sensor1"])

    def test_relative_move_supports_absolute_and_relative_clients(self) -> None:
        session = CalibrationSession()
        session.add_axis("z", FakeAxis("z"))
        session.add_axis("x", FakeAxis("x"))

        self.assertTrue(session.move_axis_relative("z", 2.5))
        self.assertTrue(session.move_axis_relative("x", -3.0, prefer_move_by=True))

        self.assertEqual(session.axes["z"].moves, [("to", 12.5)])
        self.assertEqual(session.axes["x"].moves, [("by", -3.0)])

    def test_close_leaves_lasers_off_and_is_idempotent(self) -> None:
        sensors = FakeSensors()
        session = CalibrationSession()
        session.connect_sensors(lambda: sensors)
        sensors.set_enabled("sensor1", True)

        session.close()
        session.close()

        self.assertFalse(any(sensors.enabled.values()))
        self.assertTrue(sensors.closed)
