"""Off-hardware tests for the Pico gantry-client adapters.

Uses a fake PicoLink that speaks the line protocol, so no serial port / Pico is needed.
Verifies the adapter is a faithful per-axis drop-in for the GantryServerClient subset:
status() shape, firmware-clamp (EVT CLAMP) -> GantrySoftLimitError, and
that one shared link serves both Y and Z.
"""

import pytest

pytest.importorskip("serial")  # pyserial; the adapter's lazy Pico client import needs it

from gantry_client import PicoClientError
from capture.motion.gantry.server import (
    GantryServerError,
    GantrySoftLimitError,
)
from capture.motion.pico.adapter import PicoAxisClient


class FakeLink:
    """Minimal stand-in for PicoLink that answers the gantry-firmware protocol."""

    def __init__(self):
        self.pos = 100.0
        self.homed = True
        self.moving = False
        self.sent = []
        self.vmaxes = []
        self.events = []
        self.raise_on_move = None  # message -> raise PicoClientError on the next MOVE
        self.raise_on_pos = None  # message -> raise PicoClientError on POS?
        self.clamp_on_move = False  # queue an EVT CLAMP after the next MOVE
        self.stay_moving_after_move = False

    @staticmethod
    def _ax(p):
        if len(p) > 1 and p[1].upper() in ("Y", "Z"):
            return p[1].lower(), p[2:]
        return "y", p[1:]

    def cmd(self, line, timeout=5.0):
        self.sent.append(line)
        p = line.split()
        c = p[0].upper()
        ax, args = self._ax(p)
        A = ax.upper()
        if c == "VMAX":
            self.vmaxes.append(float(args[0]))
            return "OK VMAX %s %.3f" % (A, float(args[0]))
        if c == "ACC":
            return "OK ACC %s %.3f" % (A, float(args[0]))
        if c == "SOFT":
            return "OK SOFT %s %s" % (A, args[0])
        if c == "POS?":
            if self.raise_on_pos:
                raise PicoClientError(self.raise_on_pos)
            return "OK POS %s %.3f %d %d" % (
                A,
                self.pos,
                1 if self.homed else 0,
                1 if self.moving else 0,
            )
        if c == "MOVE":
            if self.raise_on_move:
                raise PicoClientError(self.raise_on_move)
            self.pos = float(args[0])
            self.moving = self.stay_moving_after_move
            if self.clamp_on_move:
                self.events.append("EVT CLAMP %s pos=%.3f tgt=%s" % (ax, self.pos, args[0]))
            return "OK MOVE %s %.3f" % (A, self.pos)
        if c == "HOME":
            self.pos = 0.0
            self.homed = True
            return "OK HOMED %s 0.000" % A
        if c == "STOP":
            return "OK STOP %s" % A
        return "OK"

    def drain_events(self, axis=None):
        if axis is None:
            ev, self.events = self.events, []
            return ev
        keep, out = [], []
        for e in self.events:
            f = e.split()
            a = f[2].lower() if len(f) >= 3 else None
            (out if a == axis.lower() else keep).append(e)
        self.events = keep
        return out

    def close(self):
        pass


def mk(link=None, axis="y", **kw):
    return PicoAxisClient(link or FakeLink(), axis, **kw)


def test_status_shape():
    a = mk(FakeLink())
    s = a.status()
    assert s["position"]["y"] == 100.0
    assert s["homed_axes"] == ["y"]
    assert s["moving"] is False


def test_constructor_applies_speed_and_softlimits():
    f = FakeLink()
    mk(f, vmax_mm_s=20.0, acc_mm_s2=1000.0)
    assert 20.0 in f.vmaxes
    assert any(s.startswith("ACC Y") for s in f.sent)
    assert any(s.startswith("SOFT Y 1") for s in f.sent)


def test_move_to_ok_maps_feed_to_vmax_transiently():
    f = FakeLink()
    a = mk(f)  # no vmax_mm_s -> restore to firmware default
    r = a.move_to(120.0, feed_mm_min=600.0)
    assert r["status"] == "completed"
    assert f.pos == 120.0
    assert 10.0 in f.vmaxes  # 600 mm/min -> 10 mm/s applied during the move
    assert f.vmaxes[-1] == 20.0  # ...restored to DEFAULT_VMAX_MM_S after


def test_move_to_feed_restores_configured_vmax():
    f = FakeLink()
    a = mk(f, vmax_mm_s=15.0)
    a.move_to(120.0, feed_mm_min=600.0)
    assert 10.0 in f.vmaxes
    assert f.vmaxes[-1] == 15.0


def test_firmware_clamp_translates_to_soft_limit():
    f = FakeLink()
    f.clamp_on_move = True
    a = mk(f)
    with pytest.raises(GantrySoftLimitError):
        a.move_to(560.0)  # in-band target, but the firmware clamps -> soft-limit edge


def test_other_firmware_error_translates_to_server_error():
    f = FakeLink()
    f.raise_on_move = "MOVE Y 200 -> ERR BADARG"
    a = mk(f)
    with pytest.raises(GantryServerError):
        a.move_to(200.0)


def test_move_timeout_requests_stop_and_raises():
    f = FakeLink()
    f.stay_moving_after_move = True
    a = mk(f, timeout_s=0.0)

    with pytest.raises(GantryServerError, match="wait_idle timeout"):
        a.move_to(200.0)

    assert "STOP Y" in f.sent


def test_status_read_failure_is_server_error():
    f = FakeLink()
    f.raise_on_pos = "read timeout"
    with pytest.raises(GantryServerError):
        mk(f).status()


def test_z_axis_uses_z_protocol_commands():
    f = FakeLink()
    f.pos = 200.0
    z = mk(f, axis="z")
    assert z.axis == "z"
    assert z.status()["position"]["z"] == 200.0
    assert any(s.startswith("POS? Z") for s in f.sent)
    z.move_to(380.0)
    assert any(s.startswith("MOVE Z") for s in f.sent)


def test_shared_link_serves_both_axes():
    f = FakeLink()
    y = PicoAxisClient(f, "y")
    z = PicoAxisClient(f, "z")
    assert y._link is z._link  # one connection, two axes


def test_events_are_axis_scoped():
    f = FakeLink()
    y = PicoAxisClient(f, "y")
    z = PicoAxisClient(f, "z")
    f.events = ["EVT CLAMP y pos=569 tgt=600", "EVT ABORT z end-switch"]
    assert y.drain_events() == ["EVT CLAMP y pos=569 tgt=600"]  # y gets only its own event
    assert f.events == ["EVT ABORT z end-switch"]  # z's event stays queued
    assert z.drain_events() == ["EVT ABORT z end-switch"]  # z gets its own later
