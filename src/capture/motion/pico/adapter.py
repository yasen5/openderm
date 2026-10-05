"""Drop-in gantry-client adapters backed by the Raspberry Pi Pico real-time controller.

The Pico gantry firmware serves multiple axes (Y, Z) over ONE serial port, so a single
``PicoLink`` connection is shared:

  open_pico_link(port)          -> a shared PicoLink (one serial connection)
  PicoAxisClient(link, "y"/"z") -> a GantryServerClient-compatible client for that axis

``PicoAxisClient`` exposes the subset of :class:`GantryServerClient` used by the
scan and regulation controllers. It raises the same error types as the X-axis client
(:class:`GantrySoftLimitError` / :class:`GantryServerError`) so existing soft-limit edge
handling works unchanged.

The Pico client module is imported lazily so commands that use only X
or RX do not need pyserial.
"""

from __future__ import annotations

import time
from typing import Any

from ..gantry.server import (
    GantryServerError,
    GantrySoftLimitError,
)

# Must match gantry_firmware.py's per-axis VMAX (used to restore the firmware vmax after a per-move
# feed override, which the firmware applies persistently).
DEFAULT_VMAX_MM_S = 20.0


def _import_link():
    """Lazily import the Pico client (needs pyserial)."""
    from .client import PicoClientError, PicoLink

    return PicoLink, PicoClientError


def open_pico_link(port: str = "/dev/ttyACM0", timeout_s: float = 30.0):
    """Open one shared connection to the Pico (use for both Y and Z)."""
    PicoLink, _ = _import_link()
    return PicoLink(port=port, timeout=timeout_s)


def _parse_pos(reply: str):
    f = reply.split()
    i = 2
    if i < len(f) and f[i].upper() in ("Y", "Z"):
        i += 1
    return float(f[i]), bool(int(f[i + 1])), bool(int(f[i + 2]))


class PicoAxisClient:
    """GantryServerClient-compatible client for ONE axis on the Pico, over a shared PicoLink."""

    def __init__(
        self,
        link,
        axis: str,
        *,
        enforce_limits: bool = True,
        vmax_mm_s: float | None = None,
        acc_mm_s2: float | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        _, self._client_error = _import_link()
        self._link = link
        self.axis = axis.lower()
        self._A = self.axis.upper()
        self.timeout_s = timeout_s
        self._vmax_mm_s = vmax_mm_s
        if vmax_mm_s is not None:
            self._link.cmd("VMAX %s %.4f" % (self._A, vmax_mm_s))
        if acc_mm_s2 is not None:
            self._link.cmd("ACC %s %.4f" % (self._A, acc_mm_s2))
        self._link.cmd("SOFT %s %d" % (self._A, 1 if enforce_limits else 0))

    # --- error translation ---
    def _translate(self, exc: Exception) -> GantryServerError:
        msg = str(exc)
        if "CLAMP" in msg or "SOFTLIMIT" in msg or "soft" in msg.lower():
            return GantrySoftLimitError("Pico %s clamped at a soft limit: %s" % (self.axis, msg))
        return GantryServerError("Pico %s error: %s" % (self.axis, msg))

    def _pos(self):
        return _parse_pos(self._link.cmd("POS? %s" % self._A))

    def drain_events(self):
        """This axis's pending async EVT lines (others stay queued)."""
        return self._link.drain_events(self.axis)

    # --- GantryServerClient-compatible API ---
    def status(self) -> dict[str, Any]:
        try:
            mm, homed, moving = self._pos()
        except self._client_error as exc:
            raise GantryServerError("Pico %s status read failed: %s" % (self.axis, exc)) from exc
        return {
            "position": {self.axis: mm},
            "homed_axes": ([self.axis] if homed else []),
            "moving": moving,
        }

    def _wait_idle(self, poll: float = 0.03) -> None:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            for ev in self._link.drain_events(self.axis):
                if ev.startswith("EVT CLAMP " + self.axis) or ev.startswith(
                    "EVT ABORT " + self.axis
                ):
                    raise self._client_error(ev)
            _, _, moving = self._pos()
            if not moving:
                return
            time.sleep(poll)
        # Never report a timed-out axis as completed: a coordinated caller
        # could otherwise start the next move while this axis is still moving
        # or stalled. Request a firmware stop before surfacing the failure.
        try:
            self._link.cmd("STOP %s" % self._A)
        except self._client_error as stop_exc:
            raise self._client_error(
                "wait_idle timeout on %s; STOP failed: %s" % (self._A, stop_exc)
            ) from stop_exc
        raise self._client_error("wait_idle timeout on %s; STOP requested" % self._A)

    def move_to(
        self,
        position_mm: float,
        feed_mm_min: float | None = None,
        tolerance_mm: float | None = None,
    ) -> dict[str, Any]:
        if feed_mm_min is not None:
            self._link.cmd("VMAX %s %.4f" % (self._A, feed_mm_min / 60.0))
        try:
            self._link.cmd("MOVE %s %.4f" % (self._A, position_mm))
            self._wait_idle()
        except self._client_error as exc:
            raise self._translate(exc) from exc
        finally:
            if feed_mm_min is not None:
                restore = self._vmax_mm_s if self._vmax_mm_s is not None else DEFAULT_VMAX_MM_S
                try:
                    self._link.cmd("VMAX %s %.4f" % (self._A, restore))
                except Exception:
                    pass
        return {"status": "completed"}

    def stream_to(self, position_mm: float, continuous: bool = False) -> dict[str, Any]:
        """Push a position setpoint without waiting. ``continuous=True`` uses the firmware
        look-ahead (MOVEC): the axis cruises THROUGH this target instead of decelerating to
        a stop, so a stream of same-direction waypoints flows smoothly. The LAST waypoint of
        a continuous sequence should be a normal (continuous=False) stop so it lands at rest
        rather than hard-stopping; overshoot is impossible either way (firmware clamps)."""
        verb = "MOVEC" if continuous else "MOVE"
        try:
            self._link.cmd("%s %s %.4f" % (verb, self._A, position_mm))
        except self._client_error as exc:
            raise self._translate(exc) from exc
        return {"status": "queued"}

    def set_vmax(self, mm_s: float) -> float:
        """Set this axis's max speed (mm/s) on the firmware. Persists for later moves and
        streams (the firmware applies VMAX until it is changed again), so it doubles as the
        restore target for a per-move ``move_to`` feed override."""
        try:
            self._link.cmd("VMAX %s %.4f" % (self._A, mm_s))
        except self._client_error as exc:
            raise self._translate(exc) from exc
        self._vmax_mm_s = mm_s
        return mm_s

    # --- lifecycle ---
    def home(
        self, fast_mm_s: float | None = None, slow_mm_s: float | None = None
    ) -> dict[str, Any]:
        line = "HOME %s" % self._A
        if fast_mm_s is not None:
            line += " %.4f" % fast_mm_s
            if slow_mm_s is not None:
                line += " %.4f" % slow_mm_s
        try:
            self._link.cmd(line, timeout=180.0)
        except self._client_error as exc:
            raise self._translate(exc) from exc
        return {"status": "homed"}

    def stop(self, mode: str = "soft") -> dict[str, Any]:
        try:
            self._link.cmd("STOP %s" % self._A)
        except self._client_error as exc:
            raise self._translate(exc) from exc
        return {"status": "stopped"}

    def close(self) -> None:
        self._link.close()


class PicoMultiAxis:
    """Batched reads/streams across several PicoAxisClients that SHARE one PicoLink, so Y and
    Z cross the (possibly network-bridged) serial link in ONE round trip instead of one per
    axis. Reads use the firmware's ``STATUS?`` (all axes), streams use ``MOVEM`` (all axes).
    Falls back to per-axis -- permanently -- the first time the firmware reports the batched
    verb is unknown, so it is safe against an un-updated Pico."""

    def __init__(self, clients: dict[str, PicoAxisClient]) -> None:
        # clients: {axis_name: PicoAxisClient}; all MUST share a single PicoLink.
        self._clients = dict(clients)
        if not self._clients:
            raise ValueError("PicoMultiAxis needs at least one client")
        links = {id(c._link): c._link for c in self._clients.values()}
        if len(links) != 1:
            raise ValueError("PicoMultiAxis requires all clients to share one PicoLink")
        self._link = next(iter(links.values()))
        _, self._client_error = _import_link()
        self._batch_read = True
        self._batch_move = True

    def _any(self) -> PicoAxisClient:
        return next(iter(self._clients.values()))

    def positions(self) -> dict[str, float]:
        """{axis: mm} for all clients. ONE STATUS? round trip when supported, else per-axis."""
        if self._batch_read:
            try:
                allst = self._link.status_all()
                return {a: float(allst[a]["pos_mm"]) for a in self._clients if a in allst}
            except self._client_error as exc:
                if "UNKNOWN" in str(exc):
                    self._batch_read = False  # old firmware: stop trying STATUS?
                else:
                    raise GantryServerError("Pico STATUS? failed: %s" % exc) from exc
        out: dict[str, float] = {}
        for a, cl in self._clients.items():
            out[a] = float(cl.status()["position"][a])
        return out

    def stream_to(self, targets_mm: dict[str, float], continuous: bool = False) -> dict[str, Any]:
        """Stream several absolute setpoints (mm) in ONE MOVEM round trip when supported, else
        per-axis. Applies each axis's soft-limit guard first (same as PicoAxisClient.stream_to),
        so an out-of-range target raises GantrySoftLimitError before anything is commanded."""
        if self._batch_move:
            try:
                self._link.move_multi(
                    {self._clients[a]._A: mm for a, mm in targets_mm.items()},
                    continuous=continuous,
                )
                return {"status": "queued"}
            except self._client_error as exc:
                if "UNKNOWN" in str(exc):
                    self._batch_move = False  # old firmware: stop trying MOVEM
                else:
                    raise self._any()._translate(exc) from exc
        for a, mm in targets_mm.items():
            self._clients[a].stream_to(mm, continuous=continuous)
        return {"status": "queued"}
