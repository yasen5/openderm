from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import time
from typing import Any, Protocol


class HgCSensorError(RuntimeError):
    """Raised when HG-C sensor control or acquisition fails."""


@dataclass(frozen=True)
class SensorChannelConfig:
    name: str
    adc_channel: int
    gpio_pin: int


@dataclass(frozen=True)
class HgCSensorConfig:
    ads1115_address: int = 0x48
    i2c_bus_number: int = 1
    samples_per_read: int = 5
    settle_time_s: float = 0.02
    sensor_enable_active_high: bool = False
    shunt_resistance_ohms: float = 150.0
    current_min_ma: float = 4.0
    current_max_ma: float = 20.0
    distance_min_mm: float = 65.0
    distance_max_mm: float = 135.0
    sensors: tuple[SensorChannelConfig, ...] = (
        SensorChannelConfig(name="sensor1", adc_channel=2, gpio_pin=5),
        SensorChannelConfig(name="sensor2", adc_channel=1, gpio_pin=6),
    )

    @classmethod
    def from_env(cls) -> "HgCSensorConfig":
        default = cls()
        sensors = []
        for index, sensor in enumerate(default.sensors, start=1):
            sensors.append(
                SensorChannelConfig(
                    name=os.getenv(f"HG_C_SENSOR_{index}_NAME", sensor.name),
                    adc_channel=int(
                        os.getenv(f"HG_C_SENSOR_{index}_ADC_CHANNEL", str(sensor.adc_channel))
                    ),
                    gpio_pin=int(os.getenv(f"HG_C_SENSOR_{index}_GPIO_PIN", str(sensor.gpio_pin))),
                )
            )
        return cls(
            ads1115_address=int(
                os.getenv("HG_C_ADS1115_ADDRESS", hex(default.ads1115_address)),
                0,
            ),
            i2c_bus_number=int(os.getenv("HG_C_I2C_BUS_NUMBER", str(default.i2c_bus_number))),
            samples_per_read=max(
                1,
                int(os.getenv("HG_C_SAMPLES_PER_READ", str(default.samples_per_read))),
            ),
            settle_time_s=max(
                0.0,
                float(os.getenv("HG_C_SETTLE_TIME_S", str(default.settle_time_s))),
            ),
            sensor_enable_active_high=os.getenv(
                "HG_C_SENSOR_ENABLE_ACTIVE_HIGH",
                "1" if default.sensor_enable_active_high else "0",
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            shunt_resistance_ohms=float(
                os.getenv(
                    "HG_C_SHUNT_RESISTANCE_OHMS",
                    str(default.shunt_resistance_ohms),
                )
            ),
            current_min_ma=float(os.getenv("HG_C_CURRENT_MIN_MA", str(default.current_min_ma))),
            current_max_ma=float(os.getenv("HG_C_CURRENT_MAX_MA", str(default.current_max_ma))),
            distance_min_mm=float(os.getenv("HG_C_DISTANCE_MIN_MM", str(default.distance_min_mm))),
            distance_max_mm=float(os.getenv("HG_C_DISTANCE_MAX_MM", str(default.distance_max_mm))),
            sensors=tuple(sensors),
        )


@dataclass(frozen=True)
class HgCSensorReading:
    name: str
    adc_channel: int
    gpio_pin: int
    enabled: bool
    voltage_v: float
    current_ma: float
    distance_mm: float | None
    distance_offset_mm: float | None
    in_range: bool
    signal_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adc_channel": self.adc_channel,
            "gpio_pin": self.gpio_pin,
            "enabled": self.enabled,
            "voltage_v": round(self.voltage_v, 6),
            "current_ma": round(self.current_ma, 6),
            "distance_mm": None if self.distance_mm is None else round(self.distance_mm, 6),
            "distance_offset_mm": None
            if self.distance_offset_mm is None
            else round(self.distance_offset_mm, 6),
            "in_range": self.in_range,
            "signal_status": self.signal_status,
        }


class VoltageReader(Protocol):
    def read_voltage(self, channel: int) -> float: ...


class PinController(Protocol):
    def setup_output(self, pin: int, *, initial_high: bool = False) -> None: ...

    def write(self, pin: int, high: bool) -> None: ...

    def close(self) -> None: ...


class HgCSensorController:
    def __init__(
        self,
        config: HgCSensorConfig,
        voltage_reader: VoltageReader,
        pin_controller: PinController,
    ) -> None:
        self.config = config
        self._voltage_reader = voltage_reader
        self._pin_controller = pin_controller
        self._sensors_by_name = {sensor.name: sensor for sensor in config.sensors}
        self._enabled_by_name: dict[str, bool] = {}
        for sensor in config.sensors:
            self._pin_controller.setup_output(
                sensor.gpio_pin,
                initial_high=self._pin_level_for_enabled(False),
            )
            self._enabled_by_name[sensor.name] = False

    def close(self) -> None:
        self._pin_controller.close()

    def resolve_sensors(self, selected: str) -> tuple[SensorChannelConfig, ...]:
        if selected == "all":
            return self.config.sensors
        sensor = self._sensors_by_name.get(selected)
        if sensor is None:
            raise HgCSensorError(f"Unknown sensor {selected!r}.")
        return (sensor,)

    def set_enabled(self, sensor_name: str, enabled: bool, *, settle: bool = True) -> None:
        """Switch a sensor's laser. ``settle=False`` skips the post-switch settle
        sleep -- safe when disabling in an alternating exclusive-read pattern,
        where the next sensor's enable settle already covers this turn-off."""
        sensor = self._sensors_by_name.get(sensor_name)
        if sensor is None:
            raise HgCSensorError(f"Unknown sensor {sensor_name!r}.")
        self._pin_controller.write(sensor.gpio_pin, self._pin_level_for_enabled(enabled))
        self._enabled_by_name[sensor_name] = enabled
        if settle and self.config.settle_time_s > 0:
            time.sleep(self.config.settle_time_s)

    def _pin_level_for_enabled(self, enabled: bool) -> bool:
        return enabled if self.config.sensor_enable_active_high else not enabled

    def set_enabled_for_selection(self, selected: str, enabled: bool) -> None:
        for sensor in self.resolve_sensors(selected):
            self.set_enabled(sensor.name, enabled)

    def read_sensor(self, sensor_name: str, *, samples: int | None = None) -> HgCSensorReading:
        sensor = self._sensors_by_name.get(sensor_name)
        if sensor is None:
            raise HgCSensorError(f"Unknown sensor {sensor_name!r}.")
        sample_count = max(1, samples if samples is not None else self.config.samples_per_read)
        voltage_v = (
            sum(self._voltage_reader.read_voltage(sensor.adc_channel) for _ in range(sample_count))
            / sample_count
        )
        current_ma = self.current_from_voltage(voltage_v)
        distance_mm = self.distance_from_current(current_ma)
        return HgCSensorReading(
            name=sensor.name,
            adc_channel=sensor.adc_channel,
            gpio_pin=sensor.gpio_pin,
            enabled=self._enabled_by_name[sensor.name],
            voltage_v=voltage_v,
            current_ma=current_ma,
            distance_mm=distance_mm,
            distance_offset_mm=None
            if distance_mm is None
            else distance_mm - ((self.config.distance_min_mm + self.config.distance_max_mm) / 2.0),
            in_range=distance_mm is not None,
            signal_status=self.signal_status(current_ma),
        )

    def read_selection(
        self, selected: str, *, samples: int | None = None
    ) -> list[HgCSensorReading]:
        return [
            self.read_sensor(sensor.name, samples=samples)
            for sensor in self.resolve_sensors(selected)
        ]

    def current_from_voltage(self, voltage_v: float) -> float:
        if self.config.shunt_resistance_ohms <= 0:
            raise HgCSensorError("HG-C shunt resistance must be positive.")
        return (voltage_v / self.config.shunt_resistance_ohms) * 1000.0

    def distance_from_current(self, current_ma: float) -> float | None:
        current_span = self.config.current_max_ma - self.config.current_min_ma
        if current_span <= 0:
            raise HgCSensorError("HG-C current range must be positive.")
        if current_ma < self.config.current_min_ma or current_ma > self.config.current_max_ma:
            return None
        distance_span = self.config.distance_max_mm - self.config.distance_min_mm
        proportion = (current_ma - self.config.current_min_ma) / current_span
        # The HG-C sources its MAXIMUM current at its MINIMUM (near) distance,
        # so distance decreases as current rises: current_min -> distance_max
        # (far), current_max -> distance_min (near).
        return self.config.distance_max_mm - (proportion * distance_span)

    def signal_status(self, current_ma: float) -> str:
        if current_ma < self.config.current_min_ma:
            return "below_range"
        if current_ma > self.config.current_max_ma:
            return "above_range"
        return "ok"


class Ads1115VoltageReader:
    def __init__(self, address: int, *, i2c_bus_number: int = 1) -> None:
        try:
            import board
            import busio
            import adafruit_ads1x15.ads1115 as ADS
            from adafruit_ads1x15.analog_in import AnalogIn
        except ImportError:
            self._fallback = _SmbusAds1115VoltageReader(
                address=address,
                i2c_bus_number=i2c_bus_number,
            )
            self._channels = None
        else:
            self._fallback = None
            i2c = busio.I2C(board.SCL, board.SDA)
            ads = ADS.ADS1115(i2c, address=address)
            ads.gain = 2 / 3
            self._channels = {
                0: AnalogIn(ads, ADS.P0),
                1: AnalogIn(ads, ADS.P1),
                2: AnalogIn(ads, ADS.P2),
                3: AnalogIn(ads, ADS.P3),
            }

    def read_voltage(self, channel: int) -> float:
        if self._fallback is not None:
            return self._fallback.read_voltage(channel)
        analog_in = self._channels.get(channel)
        if analog_in is None:
            raise HgCSensorError(f"Unsupported ADS1115 channel {channel}.")
        return float(analog_in.voltage)


class _SmbusAds1115VoltageReader:
    _POINTER_CONVERSION = 0x00
    _POINTER_CONFIG = 0x01
    _OS_SINGLE = 0x8000
    _MUX_BY_CHANNEL = {
        0: 0x4000,
        1: 0x5000,
        2: 0x6000,
        3: 0x7000,
    }
    _PGA_6_144V = 0x0000
    _MODE_SINGLE = 0x0100
    _DR_860SPS = 0x00E0
    _COMP_DISABLE = 0x0003
    _FULL_SCALE_V = 6.144

    def __init__(self, address: int, *, i2c_bus_number: int) -> None:
        try:
            from smbus2 import SMBus
        except ImportError as exc:
            raise HgCSensorError(
                "ADS1115 support requires either board/busio/adafruit-circuitpython-ads1x15 "
                "or the smbus2 package for direct Linux I2C access."
            ) from exc
        self._address = address
        self._bus = SMBus(i2c_bus_number)

    def read_voltage(self, channel: int) -> float:
        mux = self._MUX_BY_CHANNEL.get(channel)
        if mux is None:
            raise HgCSensorError(f"Unsupported ADS1115 channel {channel}.")
        config = (
            self._OS_SINGLE
            | mux
            | self._PGA_6_144V
            | self._MODE_SINGLE
            | self._DR_860SPS
            | self._COMP_DISABLE
        )
        self._bus.write_i2c_block_data(
            self._address,
            self._POINTER_CONFIG,
            [(config >> 8) & 0xFF, config & 0xFF],
        )
        time.sleep(0.002)
        raw = self._bus.read_i2c_block_data(
            self._address,
            self._POINTER_CONVERSION,
            2,
        )
        value = int.from_bytes(bytes(raw), byteorder="big", signed=True)
        return (value / 32767.0) * self._FULL_SCALE_V


class GpioZeroPinController:
    def __init__(self) -> None:
        try:
            from gpiozero import OutputDevice
        except ImportError as exc:
            raise HgCSensorError(
                "GPIO output control requires the gpiozero package on Raspberry Pi #2."
            ) from exc
        self._output_device = OutputDevice
        self._pins: dict[int, Any] = {}

    def setup_output(self, pin: int, *, initial_high: bool = False) -> None:
        try:
            if pin not in self._pins:
                self._pins[pin] = self._output_device(
                    pin=pin, active_high=True, initial_value=initial_high
                )
            else:
                self.write(pin, initial_high)
        except Exception as exc:
            raise HgCSensorError(
                f"Failed to initialize GPIO pin {pin} via gpiozero: {exc}"
            ) from exc

    def write(self, pin: int, high: bool) -> None:
        device = self._pins.get(pin)
        if device is None:
            raise HgCSensorError(f"GPIO pin {pin} has not been initialized.")
        if high:
            device.on()
            return
        device.off()

    def close(self) -> None:
        for device in self._pins.values():
            device.close()
        self._pins.clear()


class GpiodPinController:
    def __init__(self) -> None:
        try:
            import gpiod
        except ImportError as exc:
            raise HgCSensorError(
                "GPIO output control requires either gpiozero with a working pin factory or the gpiod Python package."
            ) from exc
        self._gpiod = gpiod
        self._pins: dict[int, Any] = {}
        self._chip_path = self._resolve_gpiochip(gpiod)

    @staticmethod
    def _resolve_gpiochip(gpiod: Any) -> str:
        """Find the gpiochip that owns the 40-pin header.

        The kernel renumbers gpiochips across firmware/kernel updates (e.g. on
        the Pi 5 the header moved from gpiochip4 -> gpiochip0), so we never
        hardcode a number. Order of preference: the HG_C_GPIOCHIP override, then
        the chip whose label identifies it as the SoC pin controller, then the
        lowest-numbered chip that exists."""
        override = os.getenv("HG_C_GPIOCHIP")
        if override:
            return override if override.startswith("/dev/") else f"/dev/gpiochip{override}"
        chips = sorted(f"/dev/{name}" for name in os.listdir("/dev") if name.startswith("gpiochip"))
        for path in chips:
            try:
                with gpiod.Chip(path) as chip:
                    label = chip.get_info().label
            except Exception:
                continue
            # "pinctrl-rp1" (Pi 5), "pinctrl-bcm2835"/"-bcm2711" (Pi 4 and older).
            if "pinctrl" in label or "rp1" in label:
                return path
        return chips[0] if chips else "/dev/gpiochip0"

    def setup_output(self, pin: int, *, initial_high: bool = False) -> None:
        if pin in self._pins:
            self.write(pin, initial_high)
            return
        try:
            line = self._gpiod.request_lines(
                self._chip_path,
                consumer="openderm-sensors",
                config={
                    pin: self._gpiod.LineSettings(
                        direction=self._gpiod.line.Direction.OUTPUT,
                        output_value=self._gpiod.line.Value.ACTIVE
                        if initial_high
                        else self._gpiod.line.Value.INACTIVE,
                    )
                },
            )
        except Exception as exc:
            raise HgCSensorError(f"Failed to request GPIO pin {pin} via gpiod: {exc}") from exc
        self._pins[pin] = line

    def write(self, pin: int, high: bool) -> None:
        line = self._pins.get(pin)
        if line is None:
            raise HgCSensorError(f"GPIO pin {pin} has not been initialized.")
        value = self._gpiod.line.Value.ACTIVE if high else self._gpiod.line.Value.INACTIVE
        line.set_value(pin, value)

    def close(self) -> None:
        for line in self._pins.values():
            line.release()
        self._pins.clear()


def build_sensor_controller(config: HgCSensorConfig | None = None) -> HgCSensorController:
    resolved = config or HgCSensorConfig.from_env()
    voltage_reader = Ads1115VoltageReader(
        resolved.ads1115_address,
        i2c_bus_number=resolved.i2c_bus_number,
    )
    try:
        return HgCSensorController(
            config=resolved,
            voltage_reader=voltage_reader,
            pin_controller=GpioZeroPinController(),
        )
    except HgCSensorError:
        return HgCSensorController(
            config=resolved,
            voltage_reader=voltage_reader,
            pin_controller=GpiodPinController(),
        )


def build_parser() -> argparse.ArgumentParser:
    # Derive the valid --sensor values from the configured sensors so the CLI
    # stays in sync with HgCSensorConfig (and never offers an unwired sensor).
    sensor_choices = tuple(sensor.name for sensor in HgCSensorConfig().sensors) + ("all",)

    parser = argparse.ArgumentParser(
        description="Control and read the configured HG-C1100-P distance sensors on Raspberry Pi #2."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    read_parser = subparsers.add_parser(
        "read", help="Enable sensors, read them once, and optionally disable them."
    )
    read_parser.add_argument(
        "--sensor",
        choices=sensor_choices,
        default="all",
        help="Select one sensor or read all configured sensors.",
    )
    read_parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override the number of ADC samples to average per reading.",
    )

    watch_parser = subparsers.add_parser(
        "watch", help="Continuously stream readings until interrupted."
    )
    watch_parser.add_argument(
        "--sensor",
        choices=sensor_choices,
        default="all",
        help="Select one sensor or watch all configured sensors.",
    )
    watch_parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override the number of ADC samples to average per reading.",
    )
    watch_parser.add_argument(
        "--interval-s",
        type=float,
        default=0.25,
        help="Delay between readings while watching.",
    )

    for command_name, help_text in (
        ("enable", "Enable the selected sensor or sensors and exit."),
        ("disable", "Disable the selected sensor or sensors and exit."),
    ):
        command_parser = subparsers.add_parser(command_name, help=help_text)
        command_parser.add_argument(
            "--sensor",
            choices=sensor_choices,
            default="all",
            help="Select one sensor or apply to all configured sensors.",
        )
    return parser


def _render_readings(readings: list[HgCSensorReading]) -> str:
    payload = {"sensors": [reading.to_dict() for reading in readings]}
    return json.dumps(payload, indent=2, sort_keys=True)


def run(args: argparse.Namespace, controller: HgCSensorController) -> int:
    selection = args.sensor
    if args.command == "enable":
        controller.set_enabled_for_selection(selection, True)
        print(
            json.dumps({"ok": True, "sensor": selection, "enabled": True}, indent=2, sort_keys=True)
        )
        return 0
    if args.command == "disable":
        controller.set_enabled_for_selection(selection, False)
        print(
            json.dumps(
                {"ok": True, "sensor": selection, "enabled": False}, indent=2, sort_keys=True
            )
        )
        return 0

    controller.set_enabled_for_selection(selection, True)
    try:
        if args.command == "read":
            readings = controller.read_selection(selection, samples=args.samples)
            print(_render_readings(readings))
            return 0
        if args.command == "watch":
            try:
                while True:
                    print(
                        _render_readings(controller.read_selection(selection, samples=args.samples))
                    )
                    time.sleep(max(0.0, args.interval_s))
            except KeyboardInterrupt:
                return 0
        raise ValueError(f"Unhandled command: {args.command}")
    finally:
        controller.set_enabled_for_selection(selection, False)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    controller: HgCSensorController | None = None
    try:
        controller = build_sensor_controller()
        return run(args, controller)
    except HgCSensorError as exc:
        parser.exit(status=1, message=f"{exc}\n")
    finally:
        if controller is not None:
            controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
