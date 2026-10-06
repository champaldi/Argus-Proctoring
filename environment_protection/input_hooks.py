"""Low-level Windows hooks that report software-injected input without blocking it."""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes
from typing import Any, Callable


WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
WM_QUIT = 0x0012
LLKHF_LOWER_IL_INJECTED = 0x02
LLKHF_INJECTED = 0x10
LLMHF_INJECTED = 0x01
LLMHF_LOWER_IL_INJECTED = 0x02

KEYBOARD_ACTIONS = {
    0x0100: "key_down",
    0x0101: "key_up",
    0x0104: "system_key_down",
    0x0105: "system_key_up",
}
MOUSE_ACTIONS = {
    0x0200: "move",
    0x0201: "left_down",
    0x0202: "left_up",
    0x0204: "right_down",
    0x0205: "right_up",
    0x0207: "middle_down",
    0x0208: "middle_up",
    0x020A: "wheel",
    0x020B: "x_down",
    0x020C: "x_up",
    0x020E: "horizontal_wheel",
}


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = (
        ("vk_code", wintypes.DWORD),
        ("scan_code", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("extra_info", ctypes.c_size_t),
    )


class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = (
        ("point", wintypes.POINT),
        ("mouse_data", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("extra_info", ctypes.c_size_t),
    )


def decode_keyboard_input(
    message: int,
    vk_code: int,
    scan_code: int,
    flags: int,
) -> dict[str, Any] | None:
    """Return details only when Windows marks the keyboard event as injected."""
    injected = bool(flags & (LLKHF_INJECTED | LLKHF_LOWER_IL_INJECTED))
    if not injected:
        return None
    return {
        "device": "keyboard",
        "action": KEYBOARD_ACTIONS.get(message, f"message_0x{message:04x}"),
        "injected": True,
        "lower_integrity": bool(flags & LLKHF_LOWER_IL_INJECTED),
        "vk_code": int(vk_code),
        "scan_code": int(scan_code),
    }


def decode_mouse_input(
    message: int,
    x: int,
    y: int,
    mouse_data: int,
    flags: int,
) -> dict[str, Any] | None:
    """Return details only when Windows marks the mouse event as injected."""
    injected = bool(flags & (LLMHF_INJECTED | LLMHF_LOWER_IL_INJECTED))
    if not injected:
        return None
    return {
        "device": "mouse",
        "action": MOUSE_ACTIONS.get(message, f"message_0x{message:04x}"),
        "injected": True,
        "lower_integrity": bool(flags & LLMHF_LOWER_IL_INJECTED),
        "x": int(x),
        "y": int(y),
        "mouse_data": int(mouse_data),
    }


LowLevelProc = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(
    ctypes.c_ssize_t,
    ctypes.c_int,
    wintypes.WPARAM,
    wintypes.LPARAM,
)


class LowLevelInputMonitor:
    """Owns keyboard/mouse LL hooks and their Windows message-loop thread."""

    def __init__(self, user32: Any, kernel32: Any, callback: Callable[[dict], None]):
        self.user32 = user32
        self.kernel32 = kernel32
        self.callback = callback
        self._ready = threading.Event()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._error: BaseException | None = None
        self._keyboard_proc: Any = None
        self._mouse_proc: Any = None
        self._hooks: list[Any] = []
        self._configure_apis()

    def _configure_apis(self) -> None:
        self.kernel32.GetCurrentThreadId.argtypes = ()
        self.kernel32.GetCurrentThreadId.restype = wintypes.DWORD
        self.kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
        self.kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        self.user32.SetWindowsHookExW.argtypes = (
            ctypes.c_int,
            LowLevelProc,
            wintypes.HINSTANCE,
            wintypes.DWORD,
        )
        self.user32.SetWindowsHookExW.restype = wintypes.HHOOK
        self.user32.CallNextHookEx.argtypes = (
            wintypes.HHOOK,
            ctypes.c_int,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        self.user32.CallNextHookEx.restype = ctypes.c_ssize_t
        self.user32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
        self.user32.UnhookWindowsHookEx.restype = wintypes.BOOL
        self.user32.PostThreadMessageW.argtypes = (
            wintypes.DWORD,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        self.user32.PostThreadMessageW.restype = wintypes.BOOL
        self.user32.GetMessageW.argtypes = (
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
        )
        self.user32.GetMessageW.restype = wintypes.BOOL
        self.user32.TranslateMessage.argtypes = (ctypes.POINTER(wintypes.MSG),)
        self.user32.TranslateMessage.restype = wintypes.BOOL
        self.user32.DispatchMessageW.argtypes = (ctypes.POINTER(wintypes.MSG),)
        self.user32.DispatchMessageW.restype = ctypes.c_ssize_t

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="injected-input-monitor",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(2.0):
            raise TimeoutError("Low-level input hooks did not start in time")
        if self._error is not None:
            raise RuntimeError("Could not install low-level input hooks") from self._error

    def stop(self) -> None:
        self._stop_requested.set()
        if self._thread_id is not None:
            self.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
            if thread.is_alive():
                raise RuntimeError("Low-level input hook thread did not stop")

    def _report(self, details: dict[str, Any] | None) -> None:
        if details is not None and not self._stop_requested.is_set():
            self.callback(details)

    def _run(self) -> None:
        try:
            self._thread_id = int(self.kernel32.GetCurrentThreadId())

            def keyboard_proc(code, message, data_pointer):
                try:
                    if code >= 0:
                        data = ctypes.cast(
                            data_pointer,
                            ctypes.POINTER(KBDLLHOOKSTRUCT),
                        ).contents
                        self._report(
                            decode_keyboard_input(
                                int(message),
                                data.vk_code,
                                data.scan_code,
                                data.flags,
                            )
                        )
                finally:
                    return self.user32.CallNextHookEx(None, code, message, data_pointer)

            def mouse_proc(code, message, data_pointer):
                try:
                    if code >= 0:
                        data = ctypes.cast(
                            data_pointer,
                            ctypes.POINTER(MSLLHOOKSTRUCT),
                        ).contents
                        self._report(
                            decode_mouse_input(
                                int(message),
                                data.point.x,
                                data.point.y,
                                data.mouse_data,
                                data.flags,
                            )
                        )
                finally:
                    return self.user32.CallNextHookEx(None, code, message, data_pointer)

            self._keyboard_proc = LowLevelProc(keyboard_proc)
            self._mouse_proc = LowLevelProc(mouse_proc)
            module = self.kernel32.GetModuleHandleW(None)
            keyboard_hook = self.user32.SetWindowsHookExW(
                WH_KEYBOARD_LL,
                self._keyboard_proc,
                module,
                0,
            )
            if not keyboard_hook:
                raise ctypes.WinError(ctypes.get_last_error())
            self._hooks.append(keyboard_hook)
            mouse_hook = self.user32.SetWindowsHookExW(
                WH_MOUSE_LL,
                self._mouse_proc,
                module,
                0,
            )
            if not mouse_hook:
                raise ctypes.WinError(ctypes.get_last_error())
            self._hooks.append(mouse_hook)
            self._ready.set()

            message = wintypes.MSG()
            while not self._stop_requested.is_set():
                result = self.user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result <= 0:
                    break
                self.user32.TranslateMessage(ctypes.byref(message))
                self.user32.DispatchMessageW(ctypes.byref(message))
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            for hook in reversed(self._hooks):
                self.user32.UnhookWindowsHookEx(hook)
            self._hooks.clear()
            self._ready.set()
