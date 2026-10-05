"""SocketCAN transport for the RX-axis motor."""

from __future__ import annotations

import socket
import struct
import time

from ..cubemars import CAN_EFF_FLAG, CAN_EFF_MASK
from .types import CanFrame, RxAxisServerError


class SocketCanTransport:
    CAN_FRAME_FORMAT = "=IB3x8s"
    CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FORMAT)

    def __init__(self, interface: str, timeout_s: float = 0.05) -> None:
        if not hasattr(socket, "AF_CAN"):
            raise RxAxisServerError("SocketCAN is not available on this platform.")
        self.interface = interface
        self.timeout_s = timeout_s
        self.socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.socket.settimeout(timeout_s)
        self.socket.bind((interface,))

    def close(self) -> None:
        self.socket.close()

    def send(self, can_id: int, payload: bytes, *, extended: bool = True) -> None:
        if len(payload) > 8:
            raise RxAxisServerError(
                f"CAN payload must be at most 8 bytes. Received {len(payload)}."
            )
        if extended:
            wire_id = (can_id & CAN_EFF_MASK) | CAN_EFF_FLAG
        else:
            wire_id = can_id & 0x7FF
        frame = struct.pack(
            self.CAN_FRAME_FORMAT,
            wire_id,
            len(payload),
            payload.ljust(8, b"\x00"),
        )
        self.socket.send(frame)

    def receive(self, timeout_s: float | None = None) -> CanFrame | None:
        previous_timeout = self.socket.gettimeout()
        if timeout_s is not None:
            self.socket.settimeout(timeout_s)
        try:
            raw = self.socket.recv(self.CAN_FRAME_SIZE)
        except TimeoutError:
            return None
        except socket.timeout:
            return None
        finally:
            if timeout_s is not None:
                self.socket.settimeout(previous_timeout)
        raw_id, dlc, data = struct.unpack(self.CAN_FRAME_FORMAT, raw)
        if raw_id & CAN_EFF_FLAG:
            can_id = raw_id & CAN_EFF_MASK
        else:
            can_id = raw_id & 0x7FF
        return CanFrame(can_id=can_id, data=data[:dlc])

    def transact(
        self,
        can_id: int,
        payload: bytes,
        *,
        feedback_can_id: int,
        timeout_s: float,
        extended: bool = True,
    ) -> CanFrame | None:
        deadline = time.monotonic() + timeout_s
        self.send(can_id, payload, extended=extended)
        return self.transact_receive(
            feedback_can_id=feedback_can_id, timeout_s=max(0.0, deadline - time.monotonic())
        )

    def transact_receive(
        self,
        *,
        feedback_can_id: int,
        timeout_s: float,
    ) -> CanFrame | None:
        # The motor publishes feedback continuously; if the caller polls slower than
        # the publish rate, the kernel buffer accumulates frames and a plain recv()
        # returns the *oldest*. Drain everything currently queued and return the
        # most recent matching frame so the caller always sees fresh state. If the
        # buffer is empty, block up to timeout_s for the next frame.
        latest: CanFrame | None = None
        while True:
            frame = self.receive(timeout_s=0.0001)
            if frame is None:
                break
            if frame.can_id == feedback_can_id and len(frame.data) == 8:
                latest = frame
        if latest is not None:
            return latest
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            frame = self.receive(timeout_s=remaining)
            if frame is None:
                return None
            if frame.can_id == feedback_can_id and len(frame.data) == 8:
                return frame
