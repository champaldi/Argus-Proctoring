"""Drop-in module for core.security.SecurityAdapter; import has no OS side effects."""

from __future__ import annotations

import atexit
import os
import threading

from .controller import Protection, ProtectionConfig

__all__ = ["Protection", "ProtectionConfig", "enable", "disable", "status", "drain_events"]

ALLOW_CAPTURE_ENV = "PROCTOR_ALLOW_SCREEN_CAPTURE"
_protection: Protection | None = None
_lock = threading.RLock()


def enable(callback=None, *, hwnd=None, config: ProtectionConfig | None = None):
    """Protect this application's window, or an explicit HWND, until disable().

    Compatible with the team's enable(callback) adapter. With no callback, use
    drain_events(). Ctrl+Alt+F12 always requests immediate release.
    """
    global _protection
    if config is None and os.getenv(ALLOW_CAPTURE_ENV, "").strip() == "1":
        # Demo recording only: screen recorders cannot see a protected window.
        config = ProtectionConfig(protect_capture=False)
    with _lock:
        if _protection is None:
            from .windows import WindowsBackend

            _protection = Protection(WindowsBackend(), config)
        elif config is not None:
            if _protection.status()["enabled"]:
                raise RuntimeError("Disable protection before changing its configuration")
            _protection.config = config
        _protection.enable(callback, hwnd=hwnd)


def disable():
    """Release protection; also registered for normal interpreter shutdown."""
    if _protection is not None:
        _protection.disable()


def status():
    if _protection is None:
        return {
            "enabled": False,
            "reason": "not_started",
            "last_error": None,
            "hwnd": None,
            "capture_protected": False,
            "window_locked": False,
            "monitor_count": None,
        }
    return _protection.status()


def drain_events():
    return _protection.drain_events() if _protection is not None else []


atexit.register(disable)
