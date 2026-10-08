"""Windows boundary: keyboard hooks, foreground windows and process snapshots."""

from __future__ import annotations

import os
import sys
from ctypes import WinDLL, wintypes

from .input_hooks import (
    LowLevelInputMonitor,
    decode_keyboard_input,
    decode_mouse_input,
)


WDA_NONE = 0x00
WDA_EXCLUDEFROMCAPTURE = 0x11
SM_REMOTESESSION = 0x1000
SM_CMONITORS = 80


class WindowsBackend:
    def __init__(self):
        if sys.platform != "win32":
            raise OSError("Environment protection requires Windows")
        import keyboard
        import psutil
        import win32api
        import win32con
        import win32gui
        import win32process

        self.keyboard = keyboard
        self.psutil = psutil
        self.api = win32api
        self.con = win32con
        self.gui = win32gui
        self.process = win32process
        self.user32 = WinDLL("user32", use_last_error=True)
        self.kernel32 = WinDLL("kernel32", use_last_error=True)
        self.input_monitor_factory = LowLevelInputMonitor

    def resolve_target(self, hwnd):
        if hwnd is None:
            foreground = self.gui.GetForegroundWindow()
            if foreground and self.process.GetWindowThreadProcessId(foreground)[1] == os.getpid():
                hwnd = self.gui.GetAncestor(foreground, self.con.GA_ROOTOWNER)
            else:
                windows = []

                def collect(candidate, _):
                    if (
                        self.gui.IsWindowVisible(candidate)
                        and not self.gui.GetWindow(candidate, self.con.GW_OWNER)
                        and self.process.GetWindowThreadProcessId(candidate)[1] == os.getpid()
                    ):
                        windows.append(candidate)

                self.gui.EnumWindows(collect, None)
                if len(windows) != 1:
                    raise RuntimeError("Cannot identify test window; pass enable(..., hwnd=...)")
                hwnd = windows[0]
        if not hwnd or not self.gui.IsWindow(hwnd) or not self.gui.IsWindowVisible(hwnd):
            raise ValueError("Target must be a visible, existing Windows window")
        return hwnd, self.process.GetWindowThreadProcessId(hwnd)[1]

    def initial_modifiers(self):
        return {
            name
            for name, vk in (
                ("left ctrl", self.con.VK_LCONTROL),
                ("right ctrl", self.con.VK_RCONTROL),
                ("left alt", self.con.VK_LMENU),
                ("right alt", self.con.VK_RMENU),
                ("left shift", self.con.VK_LSHIFT),
                ("right shift", self.con.VK_RSHIFT),
            )
            if self.api.GetAsyncKeyState(vk) & 0x8000
        }

    def _set_capture_affinity(self, hwnd, affinity):
        function = self.user32.SetWindowDisplayAffinity
        function.argtypes = (wintypes.HWND, wintypes.DWORD)
        function.restype = wintypes.BOOL
        return bool(function(hwnd, affinity))

    def protect_capture(self, hwnd):
        return self._set_capture_affinity(hwnd, WDA_EXCLUDEFROMCAPTURE)

    def release_capture(self, hwnd):
        return self._set_capture_affinity(hwnd, WDA_NONE)

    def _covers_monitor(self, hwnd):
        left, top, right, bottom = self.gui.GetWindowRect(hwnd)
        monitor = self.api.MonitorFromWindow(hwnd, self.con.MONITOR_DEFAULTTONEAREST)
        m_left, m_top, m_right, m_bottom = self.api.GetMonitorInfo(monitor)["Monitor"]
        return left <= m_left and top <= m_top and right >= m_right and bottom >= m_bottom

    def lock_window(self, hwnd):
        """Keep the test window above all others and make it fill the screen.

        Requests are asynchronous so a caller on a worker thread is
        never blocked by the window's own thread.
        """
        try:
            # Complete checks before pinning: on failure the controller must
            # never lose track of an already-applied topmost flag.
            if not self._covers_monitor(hwnd):
                show = self.user32.ShowWindowAsync
                show.argtypes = (wintypes.HWND, wintypes.INT)
                show.restype = wintypes.BOOL
                if not show(hwnd, self.con.SW_MAXIMIZE):
                    return False
            self.gui.SetWindowPos(
                hwnd,
                self.con.HWND_TOPMOST,
                0,
                0,
                0,
                0,
                self.con.SWP_NOMOVE | self.con.SWP_NOSIZE | self.con.SWP_ASYNCWINDOWPOS,
            )
        except Exception:
            return False
        return True

    def unlock_window(self, hwnd):
        """Drop the always-on-top flag; the window size is left to the app."""
        try:
            if not self.gui.IsWindow(hwnd):
                return True
            self.gui.SetWindowPos(
                hwnd,
                self.con.HWND_NOTOPMOST,
                0,
                0,
                0,
                0,
                self.con.SWP_NOMOVE
                | self.con.SWP_NOSIZE
                | self.con.SWP_NOACTIVATE
                | self.con.SWP_ASYNCWINDOWPOS,
            )
        except Exception:
            return False
        return True

    def clear_clipboard(self):
        """Empty the system clipboard; returns False when another app holds it."""
        try:
            import win32clipboard

            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
            finally:
                win32clipboard.CloseClipboard()
        except Exception:
            return False
        return True

    def is_remote_session(self):
        return bool(self.user32.GetSystemMetrics(SM_REMOTESESSION))

    def monitor_count(self):
        return int(self.user32.GetSystemMetrics(SM_CMONITORS))

    def install_hook(self, callback):
        return self.keyboard.hook(callback, suppress=True)

    def install_input_monitor(self, callback):
        monitor = self.input_monitor_factory(self.user32, self.kernel32, callback)
        try:
            monitor.start()
        except BaseException:
            # A timeout can leave initialization running before the caller has
            # received its cleanup handle. Request shutdown even in that case.
            try:
                monitor.stop()
            except Exception:
                pass
            raise
        return monitor.stop

    def target_exists(self, hwnd, pid):
        return (
            self.gui.IsWindow(hwnd)
            and self.process.GetWindowThreadProcessId(hwnd)[1] == pid
            and self.gui.IsWindowVisible(hwnd)
        )

    def foreground_window(self):
        return self.gui.GetForegroundWindow()

    def belongs_to_target(self, hwnd, target, pid):
        if not hwnd or not self.gui.IsWindow(hwnd):
            return False
        return self.process.GetWindowThreadProcessId(hwnd)[1] == pid and (
            hwnd == target or self.gui.GetAncestor(hwnd, self.con.GA_ROOTOWNER) == target
        )

    def restore(self, hwnd):
        try:
            if self.gui.IsIconic(hwnd):
                self.gui.ShowWindow(hwnd, self.con.SW_RESTORE)
            self.gui.SetForegroundWindow(hwnd)
        except self.gui.error:
            return False
        return self.gui.GetForegroundWindow() == hwnd

    def processes(self):
        result = []
        for process in self.psutil.process_iter(
            ["pid", "ppid", "name", "create_time"], ad_value=None
        ):
            try:
                result.append(dict(process.info))
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
        return result

    def services(self):
        result = []
        for service in self.psutil.win_service_iter():
            try:
                info = service.as_dict()
                result.append(
                    {
                        "name": info.get("name"),
                        "display_name": info.get("display_name"),
                        "status": info.get("status"),
                    }
                )
            except (self.psutil.Error, OSError):
                continue
        return result
