#!/usr/bin/env python3
"""Host-side client for the multi-axis Pico gantry firmware.

The Pico serves MULTIPLE axes (Y, Z) over ONE serial port, so the connection is shared:

  PicoLink   -- owns the serial connection + the line protocol (axis-agnostic cmd()).
  AxisClient -- a thin per-axis convenience built on a PicoLink (default Y), for interactive
                use. The contour/pivot scripts use the GantryServerClient-compatible adapter
                in openderm.motion.pico.adapter (PicoAxisClient), which also shares a
                single PicoLink across Y and Z.

A pyserial URL is accepted as the port (e.g. socket://pi1-ip:8095) so the Pico can be bridged
over the network by ``openderm-pico-bridge``. Requires pyserial.

Quick demo:  python3 third_party/pico/gantry_client.py            # ping, home Y, then move Y
             python3 third_party/pico/gantry_client.py --axis z   # same on Z
"""

import argparse
import socket
import time

import serial  # pyserial

from ..security import control_token_from_env


class PicoClientError(RuntimeError):
    pass


class PicoLink:
    """Owns the serial connection to the Pico and the newline-framed request/reply protocol.
    Shared by every axis (the Pico multiplexes axes over one port)."""

    def __init__(self, port="/dev/ttyACM0", baud=115200, timeout=2.0, on_event=None):
        # serial_for_url accepts a device path OR a URL (socket://host:port, ...).
        # write_timeout is CRITICAL for the network bridge: if the bridge stalls or has a
        # stale client, the TCP send buffer fills and a plain blocking write() would hang
        # FOREVER. With write_timeout, a stuck write raises (translated to
        # PicoClientError in cmd())
        # so callers report "link stalled" instead of freezing.
        self.ser = serial.serial_for_url(
            port, baudrate=baud, timeout=timeout, write_timeout=min(timeout, 5.0)
        )
        # Disable Nagle on a bridged socket:// transport. The protocol is tiny
        # newline-framed request/reply lines -- exactly the Nagle + delayed-ACK
        # pathological case (~40ms stall per round trip; measured ~61ms before this).
        # No effect on a real serial port (no underlying socket). Pyserial's socket://
        # handler stores the socket as `_socket`. Pair with the bridge's TCP_NODELAY.
        _sock = getattr(self.ser, "_socket", None)
        if _sock is not None:
            try:
                _sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        self._read_base = timeout
        self.events = []
        self.on_event = on_event
        # serial_for_url() OPENS the port/socket immediately, so any failure in the setup
        # below (notably the liveness check raising for a silent Pico or a REPL echo) must
        # release it -- otherwise the handle leaks (a lingering TCP connection over the
        # bridge). Close-and-re-raise on ANY failure, including KeyboardInterrupt.
        try:
            control_token = control_token_from_env()
            if _sock is not None and control_token is not None:
                self._authenticate_bridge(control_token, timeout)
            time.sleep(0.2)
            self.ser.reset_input_buffer()
            # A freshly-opened connection (especially socket:// over the bridge, or USB-CDC)
            # can drop the first byte and/or leave a partial line in the firmware's input
            # buffer. Flush with newlines and absorb a dropped byte with a throwaway PING, so
            # the first REAL command isn't corrupted (e.g. SOFT -> OFT).
            try:
                self.ser.write(b"\n\n")
                self.ser.flush()
                time.sleep(0.05)
                self.cmd("PING", timeout=1.5)
            except (PicoClientError, serial.SerialException):
                pass
            self.ser.reset_input_buffer()
            # Liveness check: the gantry firmware answers PING with "OK PONG". If the reply is
            # the command echoed back (the Pico is at the MicroPython REPL, not running the
            # server) or there is no reply, fail NOW with a clear message -- otherwise every
            # later command silently reads an echo and the protocol is meaningless.
            try:
                reply = self.cmd("PING", timeout=1.5)
            except PicoClientError as exc:
                raise PicoClientError(
                    "no PING reply from the Pico on %s (%s). Is the gantry firmware running? "
                    "The "
                    "Pico boots to the REPL unless main.py is the server." % (port, exc)
                )
            if not reply.startswith("OK"):
                raise PicoClientError(
                    "Pico answered PING with %r, not 'OK PONG' -- the REPL is echoing, not the "
                    "gantry firmware. Deploy it as main.py (mpremote fs cp "
                    "src/pico/gantry_firmware.py :main.py), reset the Pico, then restart the "
                    "bridge." % reply
                )
        except BaseException:
            self.close()
            raise

    def _authenticate_bridge(self, token, timeout):
        """Authenticate a socket:// connection before any Pico bytes are relayed."""
        previous_timeout = self.ser.timeout
        try:
            self.ser.timeout = min(max(0.1, timeout), 5.0)
            self.ser.write(("AUTH " + token + "\n").encode())
            self.ser.flush()
            reply = self.ser.readline().decode(errors="replace").strip()
        except serial.SerialException as exc:
            raise PicoClientError("Pico bridge authentication failed: %s" % exc)
        finally:
            self.ser.timeout = previous_timeout
        if reply != "OK AUTH":
            raise PicoClientError(
                "Pico bridge rejected OPENDERM_CONTROL_TOKEN (expected 'OK AUTH', got %r)." % reply
            )

    def _stash(self, ln):
        self.events.append(ln)
        if self.on_event:
            self.on_event(ln)

    def _readline(self, deadline):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PicoClientError("timeout waiting for reply")
            self.ser.timeout = max(0.05, min(self._read_base, remaining))
            try:
                raw = self.ser.readline()
            except serial.SerialException as exc:
                raise PicoClientError("read failed (link dropped?): %s" % exc)
            if not raw:
                continue
            ln = raw.decode(errors="replace").strip()
            if not ln:
                continue
            if ln.startswith("EVT"):
                self._stash(ln)
                continue
            return ln

    def _drain(self):
        # Consume ALREADY-BUFFERED async EVT lines. Runs before every cmd(), so it must
        # NEVER block: read only what has already arrived (timeout 0). Otherwise a peer
        # that left half a line in the buffer (e.g. a Pico sitting at the REPL echoing
        # input, which has no newline framing) blocks readline() for the full base
        # timeout (~30s) and the whole client appears frozen.
        prev = self.ser.timeout
        try:
            self.ser.timeout = 0
            while self.ser.in_waiting:
                raw = self.ser.readline()
                if not raw:
                    break
                ln = raw.decode(errors="replace").strip()
                if ln.startswith("EVT"):
                    self._stash(ln)
        finally:
            self.ser.timeout = prev

    def drain_events(self, axis=None):
        """Return queued async EVT lines. With axis=None return+clear all; with an axis,
        return+remove only that axis's events and LEAVE the others queued (so a Y wait can
        never consume Z's CLAMP/ABORT, and vice-versa, over the shared connection)."""
        self._drain()
        if axis is None:
            ev, self.events = self.events, []
            return ev
        axis = axis.lower()
        keep, out = [], []
        for ev in self.events:
            (out if _evt_axis(ev) == axis else keep).append(ev)
        self.events = keep
        return out

    def cmd(self, line, timeout=5.0):
        self._drain()
        try:
            self.ser.write((line.strip() + "\n").encode())
            self.ser.flush()
        except serial.SerialException as exc:
            # write_timeout fired (stalled bridge) or the link dropped: surface it as a
            # PicoClientError so callers fail cleanly instead of the write blocking forever.
            raise PicoClientError("%s -> write failed (link stalled?): %s" % (line, exc))
        reply = self._readline(time.monotonic() + timeout)
        if reply.startswith("ERR"):
            raise PicoClientError("%s -> %s" % (line, reply))
        return reply

    def status_all(self, timeout=5.0):
        """Read EVERY axis in ONE round trip via the firmware's STATUS? command. Returns
        {name: {"pos_mm": float, "homed": bool, "moving": bool}}. Raises
        PicoClientError whose message contains 'UNKNOWN' if the firmware lacks STATUS?
        (only very old single-axis builds), so callers can fall back to per-axis POS?."""
        reply = self.cmd("STATUS?", timeout=timeout)
        toks = reply.split()
        if len(toks) < 2 or toks[0] != "OK" or toks[1] != "STATUS":
            raise PicoClientError("unexpected STATUS reply: %r" % reply)
        axes = {}
        cur = None
        for t in toks[2:]:
            if "=" not in t:
                cur = t.lower()
                axes[cur] = {}
            elif cur is not None:
                k, v = t.split("=", 1)
                axes[cur][k] = v
        out = {}
        for name, kv in axes.items():
            try:
                out[name] = {
                    "pos_mm": float(kv["pos"]),
                    "homed": kv.get("homed") == "1",
                    "moving": float(kv.get("vel", "0")) != 0.0,
                }
            except (KeyError, ValueError):
                raise PicoClientError("bad STATUS field for %s: %r" % (name, kv))
        return out

    def move_multi(self, targets_mm, continuous=False, timeout=5.0):
        """Command several axes' absolute targets (mm) in ONE round trip via MOVEM.
        ``targets_mm`` maps axis name/letter -> mm; ``continuous`` selects cruise (MOVEC)
        look-ahead. Raises PicoClientError whose message contains 'UNKNOWN' on firmware
        that predates MOVEM, so callers can fall back to per-axis MOVE/MOVEC."""
        if not targets_mm:
            return None
        line = "MOVEM" + (" C" if continuous else "")
        for ax, mm in targets_mm.items():
            line += " %s %.4f" % (ax.upper(), float(mm))
        return self.cmd(line, timeout=timeout)

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


def _parse_pos(reply):
    """Parse 'OK POS [axis] <mm> <homed> <moving>' -> (mm, homed, moving)."""
    f = reply.split()
    i = 2
    if i < len(f) and f[i].upper() in ("Y", "Z"):
        i += 1
    return float(f[i]), bool(int(f[i + 1])), bool(int(f[i + 2]))


def _evt_axis(ev):
    """Axis name (lowercase) of an 'EVT <KIND> <axis> ...' line, or None."""
    f = ev.split()
    return f[2].lower() if len(f) >= 3 and f[2].lower() in ("y", "z") else None


class AxisClient:
    """Per-axis convenience (default Y). Pass link= to share a connection across axes."""

    def __init__(
        self, port="/dev/ttyACM0", baud=115200, timeout=2.0, on_event=None, axis="y", link=None
    ):
        self.link = link if link is not None else PicoLink(port, baud, timeout, on_event)
        self.axis = axis.lower()
        self.A = self.axis.upper()

    def cmd(self, line, **kw):
        return self.link.cmd(line, **kw)

    def drain_events(self):
        return self.link.drain_events(self.axis)

    def close(self):
        self.link.close()

    def ping(self):
        return self.link.cmd("PING")

    def enable(self, on=True):
        return self.link.cmd("EN %s %d" % (self.A, 1 if on else 0))

    def home(self, fast_mm_s=None, slow_mm_s=None, timeout=180.0):
        line = "HOME %s" % self.A
        if fast_mm_s is not None:
            line += " %.4f" % fast_mm_s
            if slow_mm_s is not None:
                line += " %.4f" % slow_mm_s
        return self.link.cmd(line, timeout=timeout)

    def set_vmax(self, mm_s):
        return self.link.cmd("VMAX %s %.4f" % (self.A, mm_s))

    def set_acc(self, mm_s2):
        return self.link.cmd("ACC %s %.4f" % (self.A, mm_s2))

    def soft_limits(self, on=True):
        return self.link.cmd("SOFT %s %d" % (self.A, 1 if on else 0))

    def move(self, mm):
        return self.link.cmd("MOVE %s %.4f" % (self.A, mm))

    def move_continuous(self, mm):
        """Cruise THROUGH the target (look-ahead) instead of stopping at it -- for streamed
        waypoints. End a sequence with move() so the axis lands at rest."""
        return self.link.cmd("MOVEC %s %.4f" % (self.A, mm))

    def jog(self, mm_s):
        return self.link.cmd("JOG %s %.4f" % (self.A, mm_s))

    def stop(self):
        return self.link.cmd("STOP %s" % self.A)

    def position(self):
        return _parse_pos(self.link.cmd("POS? %s" % self.A))

    def status(self):
        return self.link.cmd("STATUS?")

    def bye(self):
        return self.link.cmd("BYE")

    def wait_idle(self, settle=0.0, poll=0.03, timeout=120.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for ev in self.drain_events():
                if ev.startswith("EVT CLAMP " + self.axis) or ev.startswith(
                    "EVT ABORT " + self.axis
                ):
                    raise PicoClientError(ev)
            _, _, moving = self.position()
            if not moving:
                if settle:
                    time.sleep(settle)
                return
            time.sleep(poll)
        raise PicoClientError("wait_idle timeout")

    def move_blocking(self, mm, **kw):
        self.move(mm)
        self.wait_idle(**kw)


def _demo(c):
    print("ping ->", c.ping())
    print("home ->", c.home())
    for tgt in (100, 300, 100, 60):
        c.move_blocking(tgt)
        print("at", c.position()[0], "mm")
    print("status ->", c.status())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--axis", default="y", choices=["y", "z"])
    args = ap.parse_args()
    c = AxisClient(port=args.port, axis=args.axis)
    try:
        _demo(c)
    finally:
        c.stop()
        c.close()


if __name__ == "__main__":
    main()
