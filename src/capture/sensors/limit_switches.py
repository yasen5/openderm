from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
import os
from signal import pause
import time
from typing import Any, Callable


class LimitSwitchError(RuntimeError):
    """Raised when limit switch GPIO monitoring cannot start."""


@dataclass(frozen=True)
class LimitSwitchConfig:
    rx_left_pin: int = 26
    rx_right_pin: int = 16
    pull_up: bool = True
    bounce_time_s: float = 0.02
    gpio_chip: str = "/dev/gpiochip4"

    @classmethod
    def from_env(cls) -> "LimitSwitchConfig":
        default = cls()
        return cls(
            rx_left_pin=int(os.getenv("LIMIT_SWITCH_RX_LEFT_GPIO_PIN", str(default.rx_left_pin))),
            rx_right_pin=int(
                os.getenv(
                    "LIMIT_SWITCH_RX_RIGHT_GPIO_PIN",
                    str(default.rx_right_pin),
                )
            ),
            pull_up=os.getenv("LIMIT_SWITCH_PULL_UP", "1" if default.pull_up else "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            bounce_time_s=float(
                os.getenv("LIMIT_SWITCH_BOUNCE_TIME_S", str(default.bounce_time_s))
            ),
            gpio_chip=os.getenv("LIMIT_SWITCH_GPIO_CHIP", default.gpio_chip),
        )


def pressed_message(name: str, pin: int, *, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now()).isoformat(timespec="seconds")
    return f"{timestamp} {name} limit switch pressed on GPIO {pin}"


def _build_button_factory() -> Callable[..., Any]:
    try:
        from gpiozero import Button
    except ImportError as exc:
        raise LimitSwitchError(
            "Limit switch monitoring requires gpiozero. Install project dependencies with "
            '`pip install -e ".[hardware]"`.'
        ) from exc
    return Button


class LimitSwitchReader:
    def __init__(
        self,
        config: LimitSwitchConfig,
        *,
        button_factory: Callable[..., Any] | None = None,
    ) -> None:
        if button_factory is None:
            self._impl = GpiodLimitSwitchReader(config)
            return
        self._impl = GpiozeroLimitSwitchReader(config, button_factory=button_factory)

    def close(self) -> None:
        self._impl.close()

    def rx_left_pressed(self) -> bool:
        return self._impl.rx_left_pressed()

    def rx_right_pressed(self) -> bool:
        return self._impl.rx_right_pressed()


class GpiozeroLimitSwitchReader:
    def __init__(
        self,
        config: LimitSwitchConfig,
        *,
        button_factory: Callable[..., Any] | None = None,
    ) -> None:
        button_cls = button_factory or _build_button_factory()
        self.config = config
        self.rx_left: Any | None = None
        self.rx_right: Any | None = None
        try:
            self.rx_left = button_cls(
                config.rx_left_pin,
                pull_up=config.pull_up,
                bounce_time=max(0.0, config.bounce_time_s),
            )
            self.rx_right = button_cls(
                config.rx_right_pin,
                pull_up=config.pull_up,
                bounce_time=max(0.0, config.bounce_time_s),
            )
        except Exception as exc:
            raise LimitSwitchError(f"Failed to initialize limit switch reader: {exc}") from exc

    def close(self) -> None:
        for button in (self.rx_left, self.rx_right):
            if button is not None:
                button.close()

    def rx_left_pressed(self) -> bool:
        return False if self.rx_left is None else bool(self.rx_left.is_pressed)

    def rx_right_pressed(self) -> bool:
        return False if self.rx_right is None else bool(self.rx_right.is_pressed)


class GpiodLimitSwitchReader:
    def __init__(
        self,
        config: LimitSwitchConfig,
    ) -> None:
        try:
            import gpiod
        except ImportError as exc:
            raise LimitSwitchError(
                "Limit switch reading requires gpiod or gpiozero. Install with "
                '`pip install -e ".[hardware]"` '
                "or `sudo apt install python3-gpiod`."
            ) from exc
        self.gpiod = gpiod
        self.config = config
        self.pins = (config.rx_left_pin, config.rx_right_pin)
        self.request = self._request_lines()

    def close(self) -> None:
        self.request.release()

    def rx_left_pressed(self) -> bool:
        return self._is_pressed(self.config.rx_left_pin)

    def rx_right_pressed(self) -> bool:
        return self._is_pressed(self.config.rx_right_pin)

    def _request_lines(self) -> Any:
        gpiod = self.gpiod
        bias = gpiod.line.Bias.PULL_UP if self.config.pull_up else gpiod.line.Bias.PULL_DOWN
        settings = gpiod.LineSettings(
            direction=gpiod.line.Direction.INPUT,
            bias=bias,
        )
        line_config = {pin: settings for pin in self.pins}
        chips = [self.config.gpio_chip]
        for fallback in ("/dev/gpiochip4", "/dev/gpiochip0", "/dev/gpiochip1"):
            if fallback not in chips:
                chips.append(fallback)
        errors: list[str] = []
        for chip in chips:
            try:
                return gpiod.request_lines(
                    chip,
                    consumer="openderm-rx-axis-homing",
                    config=line_config,
                )
            except Exception as exc:
                errors.append(f"{chip}: {exc}")
        raise LimitSwitchError(
            "Failed to request limit switch GPIO lines via gpiod: " + "; ".join(errors)
        )

    def _is_pressed(self, pin: int) -> bool:
        value = self.request.get_value(pin)
        active = getattr(self.gpiod.line.Value, "ACTIVE", None)
        if active is not None:
            return value == active
        return bool(value)


def _switch_specs(config: LimitSwitchConfig) -> tuple[tuple[str, int], ...]:
    return (
        ("RX left", config.rx_left_pin),
        ("RX right", config.rx_right_pin),
    )


def _startup_message(config: LimitSwitchConfig) -> str:
    pins = ", ".join(f"{name}=GPIO{pin}" for name, pin in _switch_specs(config))
    wiring = (
        "pull-up, pressed when shorted to ground"
        if config.pull_up
        else "pull-down, pressed when driven high"
    )
    return f"Watching limit switches: {pins} ({wiring}). Press Ctrl+C to exit."


def _watch_with_gpiozero(
    config: LimitSwitchConfig,
    *,
    pause_fn: Callable[[], object] = pause,
    print_fn: Callable[[str], object] = print,
    button_factory: Callable[..., Any] | None = None,
) -> int:
    button_cls = button_factory or _build_button_factory()
    switches: list[tuple[str, int, Any]] = []
    for name, pin in _switch_specs(config):
        try:
            button = button_cls(
                pin,
                pull_up=config.pull_up,
                bounce_time=max(0.0, config.bounce_time_s),
            )
        except Exception as exc:
            raise LimitSwitchError(
                f"Failed to initialize {name} limit switch on GPIO {pin}: {exc}"
            ) from exc
        switches.append((name, pin, button))

    for name, pin, button in switches:
        button.when_pressed = lambda name=name, pin=pin: print_fn(pressed_message(name, pin))

    print_fn(_startup_message(config))

    try:
        pause_fn()
    except KeyboardInterrupt:
        return 0
    finally:
        for _name, _pin, button in switches:
            button.close()
    return 0


def _request_gpiod_lines(gpiod: Any, config: LimitSwitchConfig) -> Any:
    edge = gpiod.line.Edge.FALLING if config.pull_up else gpiod.line.Edge.RISING
    bias = gpiod.line.Bias.PULL_UP if config.pull_up else gpiod.line.Bias.PULL_DOWN
    settings = gpiod.LineSettings(
        direction=gpiod.line.Direction.INPUT,
        bias=bias,
        edge_detection=edge,
        debounce_period=timedelta(seconds=max(0.0, config.bounce_time_s)),
    )
    line_config = {pin: settings for _name, pin in _switch_specs(config)}
    chips = [config.gpio_chip]
    for fallback in ("/dev/gpiochip4", "/dev/gpiochip0", "/dev/gpiochip1"):
        if fallback not in chips:
            chips.append(fallback)

    errors: list[str] = []
    for chip in chips:
        try:
            return gpiod.request_lines(
                chip,
                consumer="openderm-limit-switches",
                config=line_config,
            )
        except Exception as exc:
            errors.append(f"{chip}: {exc}")
    raise LimitSwitchError(
        "Failed to request limit switch GPIO lines via gpiod: " + "; ".join(errors)
    )


def _watch_with_gpiod(
    config: LimitSwitchConfig,
    *,
    print_fn: Callable[[str], object] = print,
) -> int:
    try:
        import gpiod
    except ImportError as exc:
        raise LimitSwitchError(
            "Limit switch monitoring requires gpiod or gpiozero. Install with "
            '`pip install -e ".[hardware]"` '
            "or `sudo apt install python3-gpiod`."
        ) from exc

    request = _request_gpiod_lines(gpiod, config)
    names_by_pin = {pin: name for name, pin in _switch_specs(config)}
    print_fn(_startup_message(config))
    try:
        while True:
            if not request.wait_edge_events(timeout=1.0):
                continue
            for event in request.read_edge_events():
                pin = int(event.line_offset)
                name = names_by_pin.get(pin, f"GPIO{pin}")
                print_fn(pressed_message(name, pin))
                if config.bounce_time_s > 0:
                    time.sleep(config.bounce_time_s)
    except KeyboardInterrupt:
        return 0
    finally:
        request.release()


def watch_limit_switches(
    config: LimitSwitchConfig,
    *,
    button_factory: Callable[..., Any] | None = None,
    pause_fn: Callable[[], object] = pause,
    print_fn: Callable[[str], object] = print,
) -> int:
    if button_factory is not None:
        return _watch_with_gpiozero(
            config,
            button_factory=button_factory,
            pause_fn=pause_fn,
            print_fn=print_fn,
        )
    try:
        return _watch_with_gpiod(config, print_fn=print_fn)
    except LimitSwitchError as gpiod_error:
        try:
            return _watch_with_gpiozero(config, pause_fn=pause_fn, print_fn=print_fn)
        except LimitSwitchError as gpiozero_error:
            raise LimitSwitchError(
                f"{gpiod_error}; gpiozero fallback also failed: {gpiozero_error}"
            ) from gpiozero_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a message when an RX Raspberry Pi GPIO limit switch is pressed."
    )
    parser.add_argument(
        "--rx-left-pin",
        type=int,
        default=None,
        help="BCM GPIO pin for the RX left limit switch. Defaults to LIMIT_SWITCH_RX_LEFT_GPIO_PIN or 26.",
    )
    parser.add_argument(
        "--rx-right-pin",
        type=int,
        default=None,
        help="BCM GPIO pin for the RX right limit switch. Defaults to LIMIT_SWITCH_RX_RIGHT_GPIO_PIN or 16.",
    )
    parser.add_argument(
        "--active-high",
        action="store_true",
        help="Use pull-down inputs and treat high as pressed. Default is pull-up, pressed-to-ground.",
    )
    parser.add_argument(
        "--bounce-time-s",
        type=float,
        default=None,
        help="Debounce interval in seconds. Defaults to LIMIT_SWITCH_BOUNCE_TIME_S or 0.02.",
    )
    parser.add_argument(
        "--gpio-chip",
        default=None,
        help="GPIO chip path for gpiod. Defaults to LIMIT_SWITCH_GPIO_CHIP or /dev/gpiochip4, with fallback probes.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> LimitSwitchConfig:
    config = LimitSwitchConfig.from_env()
    return LimitSwitchConfig(
        rx_left_pin=config.rx_left_pin if args.rx_left_pin is None else args.rx_left_pin,
        rx_right_pin=config.rx_right_pin if args.rx_right_pin is None else args.rx_right_pin,
        pull_up=False if args.active_high else config.pull_up,
        bounce_time_s=config.bounce_time_s if args.bounce_time_s is None else args.bounce_time_s,
        gpio_chip=config.gpio_chip if args.gpio_chip is None else args.gpio_chip,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return watch_limit_switches(config_from_args(args))
    except LimitSwitchError as exc:
        parser.exit(status=1, message=f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
