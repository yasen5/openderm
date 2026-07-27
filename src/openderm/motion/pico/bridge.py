#!/usr/bin/env python3
"""Expose the local Pico Y/Z gantry controller over TCP.

Run this on the host the Pico is plugged into (Pi #1). A remote host (Pi #2, running the
contour scan) then reaches the Pico as a pyserial URL:

    openderm-scan captures/example --pico-port socket://<pi1-ip>:8095 ...

After an initial shared-token handshake it is a transparent, byte-for-byte
relay (one TCP client at a time, bidirectional). The line-based gantry-firmware
protocol is then forwarded unchanged. The bridge owns /dev/ttyACM0 while it
runs, so don't also attach mpremote to the Pico on Pi #1 at the same time.

Requires pyserial (pip install pyserial).

Usage:
    openderm-pico-bridge                 # defaults: /dev/ttyACM0 <-> 127.0.0.1:8095
    OPENDERM_CONTROL_TOKEN=... openderm-pico-bridge --host 0.0.0.0 --port 8095
"""

from __future__ import annotations

import argparse
import select
import socket
import sys
import time

try:
    import serial  # pyserial
except ModuleNotFoundError:  # pragma: no cover - only base installs omit hardware extras.
    serial = None

from ..security import (
    MotionSecurityError,
    bridge_auth_line_matches,
    control_token_from_env,
    require_secure_bind,
)


AUTH_TIMEOUT_S = 5.0
AUTH_MAX_BYTES = 512


def open_serial(dev: str, baud: int) -> serial.Serial:
    if serial is None:
        raise RuntimeError("pyserial is required; install OpenDerm with the hardware extra.")
    ser = serial.Serial(dev, baud, timeout=0)  # non-blocking reads (driven by select)
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    return ser


def relay(ser: serial.Serial, conn: socket.socket) -> None:
    """Pump bytes between the serial port and the TCP client until either side closes."""
    conn.setblocking(False)
    ser_fd = ser.fileno()
    while True:
        try:
            readable, _, _ = select.select([ser_fd, conn], [], [], 1.0)
        except (OSError, ValueError):
            return
        if conn in readable:
            try:
                data = conn.recv(4096)
            except (BlockingIOError, InterruptedError):
                continue  # spurious wakeup; nothing actually available yet
            except OSError:
                return
            if data == b"":
                return  # client closed cleanly
            try:
                ser.write(data)
            except serial.SerialException:
                return
        if ser_fd in readable:
            try:
                data = ser.read(4096)
            except serial.SerialException:
                return  # device went away -> caller reopens
            if data:
                try:
                    conn.sendall(data)
                except OSError:
                    return


def authenticate_connection(
    conn: socket.socket,
    token: str | None,
    *,
    timeout_s: float = AUTH_TIMEOUT_S,
) -> bool:
    """Authenticate one bridge client before exposing the raw serial stream."""
    if token is None:
        return True
    previous_timeout = conn.gettimeout()
    line = bytearray()
    try:
        conn.settimeout(timeout_s)
        while len(line) < AUTH_MAX_BYTES:
            chunk = conn.recv(1)
            if not chunk:
                return False
            line.extend(chunk)
            if chunk == b"\n":
                break
        if not bridge_auth_line_matches(bytes(line), token):
            conn.sendall(b"ERR AUTH\n")
            return False
        conn.sendall(b"OK AUTH\n")
        return True
    except (OSError, TimeoutError):
        return False
    finally:
        try:
            conn.settimeout(previous_timeout)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="/dev/ttyACM0", help="local serial device of the Pico")
    ap.add_argument("--baud", type=int, default=115200, help="ignored by USB-CDC, but required")
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help=("interface to listen on; non-loopback binds require OPENDERM_CONTROL_TOKEN"),
    )
    ap.add_argument("--port", type=int, default=8095, help="TCP port to expose")
    args = ap.parse_args()

    if serial is None:
        ap.error("pyserial is required; install OpenDerm with the hardware extra.")
    try:
        control_token = control_token_from_env()
        require_secure_bind(args.host, control_token)
    except MotionSecurityError as exc:
        ap.error(str(exc))

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(
        f"serial bridge: {args.serial} <-> tcp {args.host}:{args.port} "
        f"(connect with socket://<this-host>:{args.port})",
        file=sys.stderr,
    )

    ser = None
    try:
        while True:
            if ser is None:
                try:
                    ser = open_serial(args.serial, args.baud)
                except serial.SerialException as exc:
                    print(f"cannot open {args.serial}: {exc}; retrying in 2s", file=sys.stderr)
                    time.sleep(2.0)
                    continue
            print("waiting for a client...", file=sys.stderr)
            conn, addr = srv.accept()
            # Disable Nagle: the gantry protocol is tiny newline-framed request/reply
            # lines, so Nagle + delayed-ACK would add ~40ms per round trip. Pair with the
            # client-side TCP_NODELAY in client.py (both ends must set it).
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            print(f"client connected: {addr}", file=sys.stderr)
            if not authenticate_connection(conn, control_token):
                print(f"client authentication failed: {addr}", file=sys.stderr)
                conn.close()
                continue
            try:
                ser.reset_input_buffer()  # start each client on a clean stream
            except Exception:
                pass
            try:
                relay(ser, conn)
            finally:
                conn.close()
                print(f"client disconnected: {addr}", file=sys.stderr)
            if ser is not None and not ser.is_open:
                ser = None  # reopen on next loop
    except KeyboardInterrupt:
        pass
    finally:
        if ser is not None:
            ser.close()
        srv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
