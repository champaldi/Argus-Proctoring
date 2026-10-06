"""Session-scoped protection logic. OS operations live in windows.py."""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

EventSink = Callable[[dict[str, Any]], None]
DEFAULT_PROCESS_NAMES = frozenset(
    {
        "chrome.exe",
        "msedge.exe",
        "firefox.exe",
        "opera.exe",
        "brave.exe",
        "vivaldi.exe",
        "browser.exe",
        "telegram.exe",
        "discord.exe",
        "slack.exe",
        "whatsapp.exe",
        "teams.exe",
        "ms-teams.exe",
        "skype.exe",
        "anydesk.exe",
        "rustdesk.exe",
        "parsec.exe",
        "parsecd.exe",
        "pservice.exe",
        "teamviewer.exe",
        "teamviewer_service.exe",
        "winvnc.exe",
        "vncserver.exe",
        "tvnserver.exe",
        "uvnc_service.exe",
        "remoting_host.exe",
        "chromoting_host.exe",
        "quickassist.exe",
        "msra.exe",
    }
)
DEFAULT_SERVICE_NAMES = frozenset(
    {
        "anydesk",
        "rustdesk",
        "parsec",
        "teamviewer",
        "winvnc",
        "vncserver",
        "tvnserver",
        "uvnc_service",
        "chromoting",
        "chrome remote desktop",
    }
)

# Alt combinations that leave or close the test window, by physical scan code:
# Tab (switch), F4 (close), Esc (cycle windows), Space (window menu).
_ALT_HOTKEYS = {15: "alt+tab", 62: "alt+f4", 1: "alt+esc", 57: "alt+space"}


@dataclass(frozen=True)
class ProtectionConfig:
    poll_interval: float = 0.2
    process_interval: float = 2.0
    # Safety limit after which the hooks release themselves. It must exceed the
    # longest real exam, otherwise protection silently ends mid-test.
    max_seconds: float = 4 * 60 * 60.0
    injected_input_idle_seconds: float = 1.0
    # Keep the test window maximized and above every other window.
    # Hide the test window from screen capture. Turn off only to record a demo.
    protect_capture: bool = True
    lock_window: bool = True
    # Empty the clipboard when the test starts and ends, so text can neither be
    # brought into the test nor carried out of it.
    clear_clipboard: bool = True
    blocked_process_names: frozenset[str] = DEFAULT_PROCESS_NAMES
    blocked_service_names: frozenset[str] = DEFAULT_SERVICE_NAMES

    def __post_init__(self):
        for name in (
            "poll_interval",
            "process_interval",
            "max_seconds",
            "injected_input_idle_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(
            self,
            "blocked_process_names",
            frozenset(name.casefold() for name in self.blocked_process_names),
        )
        object.__setattr__(
            self,
            "blocked_service_names",
            frozenset(name.casefold() for name in self.blocked_service_names),
        )


@dataclass
class _Session:
    hwnd: int
    pid: int
    callback: EventSink | None
    deadline: float
    stopped: threading.Event = field(default_factory=threading.Event)
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=256))
    modifiers: set[str] = field(default_factory=set)
    blocked_keys: set[int] = field(default_factory=set)
    seen_processes: set[tuple[int, float | None]] = field(default_factory=set)
    seen_services: set[str] = field(default_factory=set)
    monitor_count: int | None = None
    multiple_monitors_active: bool = False
    remove_hook: Callable | None = None
    remove_input_hook: Callable | None = None
    last_injected_input: float | None = None
    capture_protected: bool = False
    window_locked: bool = False
    clipboard_pending: bool = False
    cleanup_lock: threading.Lock = field(default_factory=threading.Lock)
    threads: list[threading.Thread] = field(default_factory=list)
    reason: str = "enabled"
    error: str | None = None

    def stop(self, reason: str, error: Exception | None = None):
        if error is not None:
            self.error = f"{type(error).__name__}: {error}"
        if not self.stopped.is_set():
            self.reason = reason
            self.stopped.set()


class Protection:
    """One session at a time; callbacks and OS monitoring run independently.

    Keyboard callbacks only classify a key and enqueue an event. They never
    call application code, perform process scans or wait for worker threads.
    """

    def __init__(self, backend: Any, config: ProtectionConfig | None = None):
        self.backend = backend
        self.config = config or ProtectionConfig()
        self._session: _Session | None = None
        self._lifecycle_lock = threading.RLock()

    def enable(self, callback: EventSink | None = None, *, hwnd: int | None = None):
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        with self._lifecycle_lock:
            previous = self._session
            if previous is not None:
                if not previous.stopped.is_set():
                    return
                self._cleanup(previous)
                if (
                    previous.remove_hook
                    or previous.remove_input_hook
                    or previous.capture_protected
                    or previous.window_locked
                    or any(t.is_alive() for t in previous.threads)
                ):
                    raise RuntimeError("Previous protection session is still shutting down")
            target, pid = self.backend.resolve_target(hwnd)
            session = _Session(target, pid, callback, time.monotonic() + self.config.max_seconds)
            self._session = session
            try:
                if self.config.protect_capture:
                    session.capture_protected = self.backend.protect_capture(target)
                    if not session.capture_protected:
                        self._emit(session, "capture_protection_failed", target_hwnd=target)
                if self.config.clear_clipboard:
                    self._clear_clipboard()
                    session.clipboard_pending = True
                if self.config.lock_window:
                    lock = getattr(self.backend, "lock_window", None)
                    session.window_locked = bool(lock(target)) if callable(lock) else False
                if self.backend.is_remote_session():
                    self._emit(session, "remote_session", protocol="rdp")
                session.modifiers = {
                    f"left {name}" if name in {"ctrl", "alt", "shift"} else name
                    for name in self.backend.initial_modifiers()
                }
                session.remove_hook = self.backend.install_hook(
                    lambda event: self._handle_key(session, event)
                )
                session.remove_input_hook = self.backend.install_input_monitor(
                    lambda details: self._handle_injected_input(session, details)
                )
                workers = [
                    ("security-watchdog", self._watchdog),
                    ("security-monitor", self._monitor),
                ]
                if callback is not None:
                    workers.append(("security-events", self._dispatch))
                for name, worker in workers:
                    thread = threading.Thread(
                        target=worker, args=(session,), name=name, daemon=True
                    )
                    thread.start()
                    session.threads.append(thread)
            except BaseException as exc:
                session.stop("startup_error", exc)
                self._cleanup(session)
                raise

    def disable(self):
        """Release blocking first, then remove our hook; safe to call repeatedly."""
        with self._lifecycle_lock:
            session = self._session
            if session is None:
                return
            session.stop("disabled")
            self._cleanup(session)
        # A stuck event consumer must never hold the keyboard or this caller.
        for thread in session.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=0.25)

    def status(self) -> dict[str, Any]:
        session = self._session
        return {
            "enabled": session is not None and not session.stopped.is_set(),
            "reason": session.reason if session else "not_started",
            "last_error": session.error if session else None,
            "hwnd": session.hwnd if session else None,
            "capture_protected": bool(session and session.capture_protected),
            "window_locked": bool(session and session.window_locked),
            "monitor_count": session.monitor_count if session else None,
        }

    def drain_events(self) -> list[dict[str, Any]]:
        """Retrieve events when enable() was called without a callback."""
        session = self._session
        if session is None:
            return []
        if session.callback is not None:
            raise RuntimeError("Events are already delivered to the callback")
        result = []
        while True:
            try:
                result.append(session.events.get_nowait())
            except queue.Empty:
                return result

    def _cleanup(self, session: _Session):
        with session.cleanup_lock:
            if session.remove_input_hook is not None:
                try:
                    session.remove_input_hook()
                except Exception as exc:
                    session.error = f"{type(exc).__name__}: {exc}"
                else:
                    session.remove_input_hook = None
            if session.clipboard_pending:
                session.clipboard_pending = False
                self._clear_clipboard()
            if session.window_locked:
                try:
                    unlocked = self.backend.unlock_window(session.hwnd)
                except Exception as exc:
                    session.error = f"{type(exc).__name__}: {exc}"
                else:
                    if unlocked:
                        session.window_locked = False
                    else:
                        session.error = "RuntimeError: Could not release window lock"
            if session.capture_protected:
                try:
                    released = self.backend.release_capture(session.hwnd)
                except Exception as exc:
                    session.error = f"{type(exc).__name__}: {exc}"
                else:
                    if released:
                        session.capture_protected = False
                    else:
                        session.error = "RuntimeError: Could not release capture protection"
            if session.remove_hook is not None:
                try:
                    session.remove_hook()
                except Exception as exc:
                    # Retain the handle for a later disable() retry. Its callback
                    # already passes every key because stopped is set first.
                    session.error = f"{type(exc).__name__}: {exc}"
                else:
                    session.remove_hook = None

    def _clear_clipboard(self) -> None:
        clear = getattr(self.backend, "clear_clipboard", None)
        if not callable(clear):
            return
        try:
            clear()
        except Exception:
            # A busy clipboard must never stop protection or its cleanup.
            pass

    def _emit(self, session: _Session, kind: str, **details):
        if session.stopped.is_set():
            return
        event = {
            "type": kind,
            "source": "security",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "details": details,
        }
        try:
            session.events.put_nowait(event)
        except queue.Full:
            session.stop("event_queue_full", RuntimeError("Security event queue is full"))

    def _handle_key(self, session: _Session, event: Any) -> bool:
        if session.stopped.is_set():
            return True
        if time.monotonic() >= session.deadline:
            session.stop("timeout")
            return True
        try:
            name = (event.name or "").casefold()
            scan = event.scan_code
            down = event.event_type == "down"
            base = name.removeprefix("left ").removeprefix("right ")
            if base in {"ctrl", "alt", "shift"}:
                identity = f"left {base}" if name == base else name
                if down:
                    session.modifiers.add(identity)
                else:
                    session.modifiers.discard(identity)
            modifiers = {m.removeprefix("left ").removeprefix("right ") for m in session.modifiers}
            if down and scan == 88 and {"ctrl", "alt"} <= modifiers:
                session.stop("emergency_hotkey")
                return True
            if not down:
                if scan in session.blocked_keys:
                    session.blocked_keys.discard(scan)
                    return False
                return True
            if scan in session.blocked_keys:
                return False

            hotkey = None
            if name in {"windows", "left windows", "right windows"} or scan in {91, 92}:
                hotkey = "win"
            elif name in {"print screen", "printscreen", "snapshot"}:
                hotkey = "print screen"
            elif "alt" in modifiers and scan in _ALT_HOTKEYS:
                hotkey = _ALT_HOTKEYS[scan]
            elif "ctrl" in modifiers and scan == 1:
                # Start menu, or Task Manager together with Shift.
                hotkey = "ctrl+shift+esc" if "shift" in modifiers else "ctrl+esc"
            elif "ctrl" in modifiers:
                # Physical Windows scan codes work in both English and Russian layouts.
                if scan in {45, 46, 47}:
                    hotkey = {45: "ctrl+x", 46: "ctrl+c", 47: "ctrl+v"}[scan]
                elif scan == 15:
                    hotkey = "ctrl+shift+tab" if "shift" in modifiers else "ctrl+tab"
                elif name in {"page up", "page down"}:
                    hotkey = f"ctrl+{name}"
                elif 2 <= scan <= 10:
                    hotkey = f"ctrl+{scan - 1}"
            if hotkey is None:
                return True
            session.blocked_keys.add(scan)
            self._emit(session, "hotkey_blocked", hotkey=hotkey)
            return session.stopped.is_set()
        except Exception as exc:
            session.stop("keyboard_error", exc)
            return True

    def _handle_injected_input(self, session: _Session, details: dict[str, Any]) -> None:
        if session.stopped.is_set() or not details.get("injected"):
            return
        try:
            now = time.monotonic()
            previous = session.last_injected_input
            session.last_injected_input = now
            if (
                previous is None
                or now - previous > self.config.injected_input_idle_seconds
            ):
                self._emit(session, "injected_input", **details)
        except Exception as exc:
            session.stop("input_monitor_error", exc)

    def _watchdog(self, session: _Session):
        if not session.stopped.wait(max(0, session.deadline - time.monotonic())):
            session.stop("timeout")
        self._cleanup(session)

    def _dispatch(self, session: _Session):
        while not session.stopped.is_set() or not session.events.empty():
            try:
                event = session.events.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                session.callback(event)
            except Exception as exc:
                session.stop("callback_error", exc)
                return

    def _monitor(self, session: _Session):
        outside = False
        next_scan = 0.0
        next_restore = 0.0
        try:
            while not session.stopped.is_set():
                if not self.backend.target_exists(session.hwnd, session.pid):
                    session.stop("target_closed")
                    break
                foreground = self.backend.foreground_window()
                inside = self.backend.belongs_to_target(foreground, session.hwnd, session.pid)
                now = time.monotonic()
                monitor_count = self.backend.monitor_count()
                session.monitor_count = monitor_count
                if monitor_count > 1:
                    if not session.multiple_monitors_active:
                        self._emit(
                            session,
                            "multiple_monitors",
                            monitor_count=monitor_count,
                        )
                    session.multiple_monitors_active = True
                else:
                    session.multiple_monitors_active = False
                if inside:
                    outside = False
                    next_restore = 0.0
                else:
                    restored = False
                    if now >= next_restore and not session.stopped.is_set():
                        restored = self.backend.restore(session.hwnd)
                        next_restore = now + 0.5
                    if not outside:
                        self._emit(
                            session,
                            "window_switched",
                            previous_hwnd=foreground,
                            target_hwnd=session.hwnd,
                            focus_restored=restored,
                        )
                    outside = not restored
                    if restored:
                        next_restore = 0.0
                if now >= next_scan and not session.stopped.is_set():
                    self._scan_processes(session)
                    next_scan = now + self.config.process_interval
                session.stopped.wait(self.config.poll_interval)
        except Exception as exc:
            session.stop("monitor_error", exc)
        finally:
            self._cleanup(session)

    def _scan_processes(self, session: _Session):
        processes = {p["pid"]: p for p in self.backend.processes()}
        seen = set()
        newly_detected = []
        for pid, process in processes.items():
            name = (process.get("name") or "").casefold()
            if name not in self.config.blocked_process_names:
                continue
            current = pid
            ancestry = set()
            allowed = False
            duplicate = False
            while current in processes and current not in ancestry:
                if current in {session.pid, os.getpid()}:
                    allowed = True
                    break
                ancestry.add(current)
                current = processes[current].get("ppid")
                if (
                    current in processes
                    and (processes[current].get("name") or "").casefold() == name
                ):
                    duplicate = True
            if allowed or duplicate:
                continue
            identity = (pid, process.get("create_time"))
            seen.add(identity)
            if identity not in session.seen_processes:
                newly_detected.append(
                    {"pid": pid, "name": process["name"], "create_time": process.get("create_time")}
                )
        session.seen_processes = seen
        running_services = []
        for service in self.backend.services():
            if (service.get("status") or "").casefold() != "running":
                continue
            name = (service.get("name") or "").casefold()
            display_name = (service.get("display_name") or "").casefold()
            if (
                name not in self.config.blocked_service_names
                and display_name not in self.config.blocked_service_names
            ):
                continue
            running_services.append(service)
        service_identities = {
            (service.get("name") or service.get("display_name") or "").casefold()
            for service in running_services
        }
        new_services = [
            service
            for service in running_services
            if (service.get("name") or service.get("display_name") or "").casefold()
            not in session.seen_services
        ]
        session.seen_services = service_identities
        if newly_detected or new_services:
            # Report new identities together; repeats are filtered above.
            self._emit(
                session,
                "suspicious_process",
                processes=newly_detected,
                services=new_services,
                count=len(newly_detected) + len(new_services),
                action="reported",
            )
