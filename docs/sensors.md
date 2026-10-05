# Sensors

## HG-C Sensor Control

The `openderm-sensors` command controls and reads the two HG-C1100-P sensors on Raspberry Pi #2.

Default wiring assumptions:

```text
sensor1 -> ADS1115 A2, GPIO 5
sensor2 -> ADS1115 A1, GPIO 6
ADS1115 I2C address -> 0x48
I2C bus -> /dev/i2c-1
```

Example commands:

```bash
# enable only sensor1 (A2 / GPIO 5)
openderm-sensors enable --sensor sensor1

# disable only sensor1 again
openderm-sensors disable --sensor sensor1

# enable all configured sensors, take one averaged reading, then disable them again
openderm-sensors read --sensor all

# read only sensor2 and average 10 ADC samples
openderm-sensors read --sensor sensor2 --samples 10

# keep sensor1 enabled and continuously print JSON readings until Ctrl-C
openderm-sensors watch --sensor sensor1 --interval-s 0.25
```

Each reading reports:

- measured analog voltage from the ADS1115
- inferred loop current in mA using the configured shunt resistor
- converted distance in mm
- distance offset from the configured center point
- whether the sensor output is inside the configured 4-20 mA range

The HG-C1100-P sensors are assumed to be wired in current-output mode. The code reads the ADS1115 voltage across a shunt resistor, converts that to loop current, then linearly maps `4-20 mA` to `65-135 mm`.

Default calibration:

```bash
export HG_C_SHUNT_RESISTANCE_OHMS=150
export HG_C_CURRENT_MIN_MA=4
export HG_C_CURRENT_MAX_MA=20
export HG_C_DISTANCE_MIN_MM=65
export HG_C_DISTANCE_MAX_MM=135
export HG_C_SENSOR_1_ADC_CHANNEL=2
export HG_C_SENSOR_1_GPIO_PIN=5
export HG_C_SENSOR_ENABLE_ACTIVE_HIGH=0
export HG_C_I2C_BUS_NUMBER=1
```

The ADS1115 code prefers the Adafruit/Blinka stack when it works, and falls back to direct Linux I2C access through `smbus2`.

For sensor enable GPIO pins, the code prefers `gpiozero` when its backend works on the target Pi, and falls back to the Linux `gpiod` Python package.

## RX Limit Switches

The `openderm-limit-switches` command prints when either RX limit switch is pressed.

Defaults assume BCM GPIO numbering and normally-open switches wired from GPIO to ground:

```text
RX left limit switch: GPIO 26
RX right limit switch: GPIO 16
Input mode: pull-up, pressed when shorted to ground
```

Run:

```bash
openderm-limit-switches
```

Use `--active-high` for pull-down wiring, or override pins with `--rx-left-pin` and `--rx-right-pin`. The watcher uses Linux `gpiod` first, with `gpiozero` as a fallback.
