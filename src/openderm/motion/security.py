"""Shared authentication and bind-safety helpers for motion services."""

from __future__ import annotations

import hmac
import ipaddress
import os


CONTROL_TOKEN_ENV = "OPENDERM_CONTROL_TOKEN"


class MotionSecurityError(ValueError):
    """Raised when a motion service would be exposed insecurely."""


def control_token_from_env() -> str | None:
    """Return the shared control token, rejecting values unsafe for wire protocols."""
    token = os.getenv(CONTROL_TOKEN_ENV, "").strip()
    if not token:
        return None
    if "\r" in token or "\n" in token:
        raise MotionSecurityError(f"{CONTROL_TOKEN_ENV} must be a single line.")
    return token


def is_loopback_host(host: str) -> bool:
    """Whether a bind host is restricted to the local machine."""
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def require_secure_bind(host: str, token: str | None) -> None:
    """Refuse network-exposed motion control unless authentication is configured."""
    if not is_loopback_host(host) and token is None:
        raise MotionSecurityError(
            f"Refusing to bind motion control to {host!r} without "
            f"{CONTROL_TOKEN_ENV}. Bind to 127.0.0.1 or configure a shared token."
        )


def bearer_token_matches(authorization: str | None, token: str) -> bool:
    """Constant-time validation for an HTTP ``Authorization: Bearer`` header."""
    if authorization is None:
        return False
    scheme, separator, supplied = authorization.partition(" ")
    return separator == " " and scheme.lower() == "bearer" and hmac.compare_digest(supplied, token)


def bridge_auth_line_matches(line: bytes, token: str) -> bool:
    """Constant-time validation for the Pico bridge's initial AUTH line."""
    try:
        supplied = line.decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError:
        return False
    return hmac.compare_digest(supplied, f"AUTH {token}")
