# Motion

OpenDerm has four controlled degrees of freedom:

- X is controlled by Klipper through `openderm-gantry-server` on Raspberry Pi #1.
- Y and Z are controlled by the Raspberry Pi Pico through `openderm-pico-bridge` on Raspberry Pi #1.
- RX is controlled through `openderm-rx-axis-server` on Raspberry Pi #2.

The Klipper service controls X only. OpenDerm always sends Y and Z commands to the Pico.

## Services

Create one random control token, store the same value in the untracked
OpenDerm environment file on both Pis, and load that file before starting
services or clients:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
export OPENDERM_CONTROL_TOKEN="replace-with-the-generated-token"
```

Start the X-axis command server and Pico bridge on Raspberry Pi #1:

```bash
openderm-gantry-server --host 0.0.0.0 --port 8090
openderm-pico-bridge --host 0.0.0.0 --port 8095
```

Start the RX command server on Raspberry Pi #2:

```bash
openderm-rx-axis-server --axis rx --host 127.0.0.1 --port 8091
```

Set the service addresses on Raspberry Pi #2 and in operator shells:

```bash
export GANTRY_SERVER_URL=http://openderm-gantry.local:8090
export PICO_PORT=socket://openderm-gantry.local:8095
export RX_AXIS_SERVER_URL=http://127.0.0.1:8091
```

Motion services bind to loopback by default. A non-loopback bind is rejected
unless `OPENDERM_CONTROL_TOKEN` is set. HTTP clients add it as a bearer token,
and the Pico socket client authenticates before serial bytes are relayed.
Restrict ports 8090 and 8095 to a private, firewalled control network as well.

## Operator commands

Home each axis before commanding motion:

```bash
openderm --axis x home
openderm --axis y home
openderm --axis z home
openderm --axis rx home
```

Move to absolute positions:

```bash
openderm --axis x move-to 250 --feed 900
openderm --axis y move-to 175 --feed 900
openderm --axis z move-to 200 --feed 900
openderm --axis rx move-to 0.95 --speed 0.2
```

Relative moves, status, and stops use the same axis selection:

```bash
openderm --axis y move-by -10 --feed 900
openderm --axis rx status
openderm --axis rx stop
openderm --axis rx clear-errors
```

When `--axis` is omitted, `openderm` controls X.

## X-axis command server

`openderm-gantry-server` maintains the Moonraker WebSocket connection, serializes X commands, enforces the 0–800 mm travel range, and supports the streamed X setpoints used during scanning.

The primary HTTP endpoints are:

```text
GET  /healthz
GET  /state
POST /move
POST /home
POST /stop
POST /stream/start
POST /stream/target
POST /stream/stop
```

`POST /stop` is an out-of-band, latched emergency stop by default: it calls
Moonraker immediately instead of waiting behind an active move. Recovery
requires `POST /reset` followed by homing. Callers that only intend to cancel
queued and streamed commands may explicitly request `{"mode": "soft"}`; a
soft stop does not interrupt G-code already executing in Klipper.

The server reads `MOONRAKER_WS_URL`, which defaults to `ws://localhost:7125/websocket`.

## Pico bridge

`openderm-pico-bridge` exposes the Pico serial protocol over an authenticated
TCP connection so Raspberry Pi #2 can control Y and Z through one shared
connection. The Pico firmware performs homing, real-time motion generation,
and physical-travel limit enforcement for both axes.

The default serial device is `/dev/ttyACM0`. Override it when necessary:

```bash
openderm-pico-bridge --serial /dev/ttyACM1 --host 0.0.0.0 --port 8095
```

## RX command server

`openderm-rx-axis-server` owns the SocketCAN connection, reads motor feedback, homes RX against its limit switch, enforces its verified position and speed ranges, and stops the motor when a limit switch, current limit, temperature limit, or velocity dead-man guard trips.

The primary HTTP endpoints are:

```text
GET  /healthz
GET  /state
GET  /limit-switches
POST /poll
POST /home
POST /move-to
POST /velocity
POST /stop
POST /clear-errors
```

The default CAN interface is `can1`. Override it with `GANTRY_RX_CAN_INTERFACE`.

Configure SocketCAN before starting the server:

```bash
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can1 up
```

The RX server rejects motion until homing succeeds. A safety shutdown remains latched until the cause is corrected and `openderm --axis rx clear-errors` succeeds.
