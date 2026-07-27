# Fresh Start: Raspberry Pi 5 + Octopus Pro V1.1

This guide assumes the Raspberry Pi 5, the BigTreeTech Octopus Pro V1.1, and the X-axis CL57Y driver are all new and unconfigured. The Pi runs Klipper host services and Moonraker, while the Octopus Pro generates real-time step pulses for X only. Y and Z are connected to the Pico described in the root README.

## 1. Gather parts

You need:

- Raspberry Pi 5 with power supply, microSD card, and network access
- Octopus Pro V1.1
- USB cable from Pi to Octopus
- 24V or other appropriate board power supply for the Octopus
- CL57Y driver, NEMA 23 motor, and motor power supply
- Two Omron SS-5GL2 limit switches
- Small screwdriver, multimeter, and ferrules or crimped wire ends

Do the first software steps with the Pi and Octopus only. Leave the motor disconnected until Klipper can see endstops correctly.

## 2. Prepare the Raspberry Pi

Recommended path: flash `MainsailOS` to the Pi. 

1. Install Raspberry Pi Imager on your computer.
2. In the imager, choose `MainsailOS` for Raspberry Pi.
3. Open the imager's advanced options before writing the card.
4. Set:
   - hostname, for example `openderm-gantry`
   - a username and password
   - your Wi-Fi SSID and password if not using Ethernet
   - locale and timezone
   - SSH enabled
5. Write the image to the microSD card.
6. Insert the card into the Pi, connect network, then power it on.
7. Wait a few minutes for first boot and filesystem expansion.

After boot, connect by SSH:

```bash
ssh <pi-user>@openderm-gantry.local
```

Then update packages:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

## 3. Setup the Octopus Pro

Before applying board power:

1. Confirm the board is actually `Octopus Pro V1.1`.
2. Read the voltage-jumper warning in the BTT manual. The board supports both `MOTOR_POWER` and `Main Power`, and using the wrong side can damage drivers.
3. Because you are using an external CL57Y driver, you do not need to install a plug-in stepper driver on the Octopus for this axis.
4. For first bring-up, connect only:
   - Octopus board power
   - USB from Pi to Octopus
   - one home switch
   - optionally the far-end switch
5. Leave the motor phases and high-current motor supply disconnected until endstops and step/dir polarity are confirmed.

## 4. Wire the axis as a simple step/dir system

For this repo, treat one Octopus stepper socket as a signal source only.

- Octopus `STEP` -> CL57Y `PUL`
- Octopus `DIR` -> CL57Y `DIR`
- Octopus `EN` -> CL57Y `ENA` if you want software enable control
- Octopus signal ground -> CL57Y logic ground
- Home switch -> the configured `X` endstop input
- Far-end limit switch -> a spare input used for emergency stop

The baseline config in this repo assumes:

- `stepper_x.step_pin = PF13`
- `stepper_x.dir_pin = PF12`
- `stepper_x.enable_pin = PF14`
- `stepper_x.endstop_pin = ^PG6`
- far-end limit on `PG9`

Those values come from Klipper's generic Octopus Pro v1.1 pin mapping and this repo's starter config. Verify them against your exact wiring before continuing.

## 5. Flash Klipper to the Octopus Pro

SSH into the Pi and find the Klipper source. On MainsailOS it is typically already installed in `~/klipper`. If it is missing, install Klipper through Mainsail/KIAUH first.

Build firmware:

```bash
cd ~/klipper
make menuconfig
```

Use these settings for the common Octopus Pro V1.1 STM32H723 build:

- enable low-level configuration options
- micro-controller: `STM32H723`
- bootloader offset: `128KiB bootloader`
- clock reference: `25 MHz crystal`
- communication interface: `USB (on PA11/PA12)`

Then compile:

```bash
make
```

Flash using the method supported by your board revision. The common SD-card method is:

1. Copy `out/klipper.bin` to a microSD card as `firmware.bin`.
2. Power the Octopus off.
3. Insert the card.
4. Power the Octopus on.
5. Wait for the flash to complete.
6. Check whether the file was renamed to `FIRMWARE.CUR` or similar. That usually indicates a successful flash.

After flashing, reconnect USB to the Pi and find the board's serial path:

```bash
ls /dev/serial/by-id/*
```

## 6. Install this repo's Klipper config

Clone this repository on the Pi if you have not already:

```bash
cd ~
git clone <your-repo-url> openderm
cd ~/openderm
```

Copy the baseline Klipper files into the active config directory:

```bash
mkdir -p ~/printer_data/config
cp klipper/printer.cfg ~/printer_data/config/printer.cfg
cp klipper/macros.cfg ~/printer_data/config/macros.cfg
```

Edit `~/printer_data/config/printer.cfg` and set:

- the exact `serial:` path from `ls /dev/serial/by-id/*`
- correct endstop pins if your wiring differs
- `position_max` to your usable travel
- `rotation_distance` if your mechanics require a different conversion

For the RM1605 ballscrew, `rotation_distance: 5`.

Restart Klipper from the Mainsail web UI or with:

```bash
sudo systemctl restart klipper
sudo systemctl restart moonraker
```

## 7. Verify board communication before connecting the motor

Open the Mainsail web UI in a browser. The hostname is usually your Pi hostname with `.local`, for example:

```text
http://openderm-gantry.local
```

In the console:

1. Run `STATUS`. Klipper should report ready.
2. Run `QUERY_ENDSTOPS`.
3. Trigger the home switch by hand and run `QUERY_ENDSTOPS` again.
4. Confirm the switch state changes correctly.

If the logic is reversed, invert the pin in `printer.cfg`, for example `PG6` to `!PG6` or `^PG6` to `^!PG6`, then save and restart.

Do not attempt homing until both switch polarity and travel direction make sense.

## 8. Connect and validate the CL57Y and motor

With board communication working:

1. Power everything off.
2. Wire the CL57Y to the motor exactly per its manual.
3. Wire the CL57Y logic inputs to the Octopus pins chosen in `printer.cfg`.
4. Set the CL57Y current and microstep switches conservatively for the first run.
5. Power the driver and board back on.

Before issuing a real move:

1. Keep the axis mechanically clear.
2. Confirm the ballscrew can rotate freely.
3. Be ready to cut power.

From Mainsail, send a tiny move:

```text
G91
G1 X5 F300
G90
```

If direction is wrong, invert `dir_pin` by adding or removing `!`.

If motion scale is wrong, re-check:

- `rotation_distance`
- CL57Y microstep switch settings
- whether the motor is really 200 full steps per revolution

## 9. Set up safe homing

After manual jogging works:

1. Move the carriage near the middle of the axis.
2. Run `QUERY_ENDSTOPS` one last time.
3. Issue `G28 X`.
4. Confirm the carriage moves toward the home switch.
5. Confirm it stops when the switch triggers, retracts, and re-homes slowly.

If the carriage moves away from home, stop immediately and fix `dir_pin`.

If the carriage hits the switch and keeps pushing, stop immediately and fix `endstop_pin` polarity.

## 10. Install the Pi-side control client from this repo

From the repository root on the Pi:

```bash
cd ~/openderm
python3 -m pip install -e ".[hardware]"
```

If Moonraker uses a non-default local address, set it before starting the X-axis service:

```bash
export MOONRAKER_WS_URL=ws://127.0.0.1:7125/websocket
export OPENDERM_CONTROL_TOKEN="replace-with-the-same-random-token-used-on-pi-2"
openderm-gantry-server --host 0.0.0.0 --port 8090
```

The gantry server refuses a non-loopback bind without the shared token. Keep
port 8090 restricted to the private control network.

In another shell, verify status before homing and moving X:

```bash
openderm --axis x status
openderm --axis x home
openderm --axis x move-to 250 --feed 900
openderm --axis x move-by -10 --feed 900
```

Klipper controls X only. Follow the root README to install
`pico/gantry_firmware.py` as `main.py` and start `openderm-pico-bridge` for Y/Z
control.

## 11. Calibrate before use

Before treating the axis as production-ready:

1. Measure actual travel over a commanded move and fine-tune `rotation_distance`.
2. Set conservative `max_velocity` and `max_accel`.
3. Test both limit switches.
4. Verify emergency stop behavior.
5. Confirm the carriage cannot crash into either end due to soft-limit errors.

## References

- Raspberry Pi Imager: <https://www.raspberrypi.com/software/>
- Klipper installation guide: <https://www.klipper3d.org/Installation.html>
- Moonraker installation guide: <https://moonraker.readthedocs.io/en/latest/installation/>
- MainsailOS docs: <https://docs.mainsail.xyz/mainsailos/>
- Klipper generic Octopus Pro v1.1 config: <https://raw.githubusercontent.com/Klipper3d/klipper/master/config/generic-bigtreetech-octopus-pro-v1.1.cfg>
- BigTreeTech Octopus Pro repo: <https://github.com/bigtreetech/BIGTREETECH-OCTOPUS-Pro>
