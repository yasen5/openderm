"""Synchronous HTTP client for the RX-axis service."""

from __future__ import annotations

import json
import socket
from typing import Any
from urllib import error, request

from ..security import control_token_from_env
from .types import RxAxisServerError


class RxAxisServerClient:
    def __init__(
        self,
        base_url: str,
        timeout_s: float = 5.0,
        control_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.control_token = (
            control_token if control_token is not None else control_token_from_env()
        )

    def status(self) -> dict[str, Any]:
        return self._request_json("/state")

    def limit_switches(self) -> dict[str, Any]:
        return self._request_json("/limit-switches")

    def home(self) -> dict[str, Any]:
        return self._request_json("/home", method="POST", payload={})

    def move_to(
        self,
        position_rad: float,
        speed_rad_s: float | None = None,
        accel_rad_s2: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"position_rad": position_rad}
        if speed_rad_s is not None:
            payload["speed_rad_s"] = speed_rad_s
        if accel_rad_s2 is not None:
            payload["accel_rad_s2"] = accel_rad_s2
        return self._request_json("/move-to", method="POST", payload=payload)

    def set_velocity(self, velocity_rad_s: float) -> dict[str, Any]:
        """AK speed-loop servoing: command a signed angular velocity (rad/s).
        The server auto-zeroes at the command-position window edge and on the
        dead-man timeout, so callers MUST keep refreshing this within the
        server's velocity_deadman_s while motion is wanted; send 0.0 to stop."""
        return self._request_json(
            "/velocity", method="POST", payload={"velocity_rad_s": velocity_rad_s}
        )

    def move_by(
        self,
        delta_rad: float,
        speed_rad_s: float | None = None,
        accel_rad_s2: float | None = None,
    ) -> dict[str, Any]:
        status = self.status()
        current_position = status.get("position_rad")
        if current_position is None:
            raise RxAxisServerError("RX-axis server has no measured position yet.")
        return self.move_to(
            float(current_position) + delta_rad,
            speed_rad_s=speed_rad_s,
            accel_rad_s2=accel_rad_s2,
        )

    def stop(self) -> dict[str, Any]:
        return self._request_json("/stop", method="POST", payload={})

    def clear_errors(self) -> dict[str, Any]:
        return self._request_json("/clear-errors", method="POST", payload={})

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        if self.control_token is not None:
            headers["Authorization"] = f"Bearer {self.control_token}"
        req = request.Request(
            f"{self.base_url}{path}",
            method=method,
            data=data,
            headers=headers,
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise RxAxisServerError(f"RX-axis server request failed: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise RxAxisServerError(f"RX-axis server request failed: {exc}") from exc
        except TimeoutError as exc:
            raise RxAxisServerError("RX-axis server request timed out.") from exc
        except socket.timeout as exc:
            raise RxAxisServerError("RX-axis server request timed out.") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RxAxisServerError(f"Invalid JSON from RX-axis server: {body!r}") from exc
