"""Windows adapter checks without installing hooks or changing desktop focus."""

import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


@unittest.skipUnless(sys.platform == "win32", "Windows adapter")
class WindowsBackendTests(unittest.TestCase):
    def setUp(self):
        from environment_protection.windows import WindowsBackend

        self.backend = WindowsBackend()

    def test_automatic_target_never_chooses_foreign_foreground(self):
        backend = self.backend
        with patch.object(backend, "gui") as gui, patch.object(backend, "process") as process:
            gui.GetForegroundWindow.return_value = 200
            process.GetWindowThreadProcessId.return_value = (1, os.getpid() + 100)
            gui.EnumWindows.side_effect = lambda callback, arg: None
            with self.assertRaisesRegex(RuntimeError, "identify test window"):
                backend.resolve_target(None)

    def test_own_visible_window_selected_even_when_terminal_has_focus(self):
        backend = self.backend
        with patch.object(backend, "gui") as gui, patch.object(backend, "process") as process:
            gui.GetForegroundWindow.return_value = 200
            process.GetWindowThreadProcessId.side_effect = lambda hwnd: (
                1,
                os.getpid() if hwnd == 100 else os.getpid() + 100,
            )
            gui.EnumWindows.side_effect = lambda callback, arg: callback(100, arg)
            gui.IsWindowVisible.return_value = True
            gui.IsWindow.return_value = True
            gui.GetWindow.return_value = 0
            self.assertEqual(backend.resolve_target(None), (100, os.getpid()))

    def test_owned_dialog_is_allowed_but_other_window_same_pid_is_not(self):
        backend = self.backend
        with patch.object(backend, "gui") as gui, patch.object(backend, "process") as process:
            gui.IsWindow.return_value = True
            gui.GetAncestor.side_effect = lambda hwnd, flag: 100 if hwnd == 101 else hwnd
            process.GetWindowThreadProcessId.return_value = (1, 10)
            self.assertTrue(backend.belongs_to_target(101, 100, 10))
            self.assertFalse(backend.belongs_to_target(200, 100, 10))
            process.GetWindowThreadProcessId.return_value = (1, 20)
            self.assertFalse(backend.belongs_to_target(101, 100, 10))

    def test_windows_focus_denial_is_reported_as_false(self):
        import pywintypes

        self.backend.gui = SimpleNamespace(
            IsIconic=lambda hwnd: False,
            SetForegroundWindow=Mock(side_effect=pywintypes.error(0, "SetForegroundWindow", "")),
            error=pywintypes.error,
        )
        self.assertFalse(self.backend.restore(100))

    def test_initial_modifiers_keep_each_side_separate(self):
        backend = self.backend
        with patch.object(backend, "api") as api:
            api.GetAsyncKeyState.side_effect = lambda vk: (
                0x8000 if vk in {backend.con.VK_LCONTROL, backend.con.VK_RCONTROL} else 0
            )
            self.assertEqual(backend.initial_modifiers(), {"left ctrl", "right ctrl"})

    def test_hook_cleanup_removes_only_our_registration(self):
        backend = self.backend
        registrations = [object()]
        unrelated = registrations[0]

        def hook(callback, suppress):
            self.assertTrue(suppress)
            registrations.append(callback)
            return lambda: registrations.remove(callback)

        with patch.object(backend, "keyboard", SimpleNamespace(hook=hook)):
            remove = backend.install_hook(lambda event: True)
            remove()
        self.assertEqual(registrations, [unrelated])

    def test_capture_affinity_uses_exclusion_and_restores_normal_rendering(self):
        user32 = Mock()
        user32.SetWindowDisplayAffinity.return_value = 1
        self.backend.user32 = user32

        self.assertTrue(self.backend.protect_capture(100))
        self.assertTrue(self.backend.release_capture(100))

        self.assertEqual(
            [call.args for call in user32.SetWindowDisplayAffinity.call_args_list],
            [(100, 0x11), (100, 0x00)],
        )

    def test_service_snapshot_returns_complete_windows_service_identity(self):
        services = [
            SimpleNamespace(
                as_dict=lambda: {
                    "name": "TeamViewer",
                    "display_name": "TeamViewer Remote",
                    "status": "running",
                }
            )
        ]
        self.backend.psutil.win_service_iter = Mock(return_value=services)

        self.assertEqual(
            self.backend.services(),
            [
                {
                    "name": "TeamViewer",
                    "display_name": "TeamViewer Remote",
                    "status": "running",
                }
            ],
        )

    def test_remote_session_uses_windows_remote_session_metric(self):
        self.backend.user32 = Mock()
        self.backend.user32.GetSystemMetrics.return_value = 1

        self.assertTrue(self.backend.is_remote_session())
        self.backend.user32.GetSystemMetrics.assert_called_once_with(0x1000)

    def test_monitor_count_uses_windows_display_monitor_metric(self):
        self.backend.user32 = Mock()
        self.backend.user32.GetSystemMetrics.return_value = 2

        self.assertEqual(self.backend.monitor_count(), 2)
        self.backend.user32.GetSystemMetrics.assert_called_once_with(80)

    def test_injected_keyboard_flags_and_actions_are_decoded(self):
        from environment_protection.windows import decode_keyboard_input

        self.assertEqual(
            decode_keyboard_input(0x0100, 65, 30, 0x12),
            {
                "device": "keyboard",
                "action": "key_down",
                "injected": True,
                "lower_integrity": True,
                "vk_code": 65,
                "scan_code": 30,
            },
        )
        self.assertIsNone(decode_keyboard_input(0x0101, 65, 30, 0x00))

    def test_injected_mouse_flags_and_actions_are_decoded(self):
        from environment_protection.windows import decode_mouse_input

        self.assertEqual(
            decode_mouse_input(0x0200, 120, 80, 0, 0x03),
            {
                "device": "mouse",
                "action": "move",
                "injected": True,
                "lower_integrity": True,
                "x": 120,
                "y": 80,
                "mouse_data": 0,
            },
        )
        self.assertIsNone(decode_mouse_input(0x0201, 120, 80, 0, 0x00))

    def test_input_monitor_is_started_and_its_cleanup_is_returned(self):
        monitor = Mock()
        self.backend.input_monitor_factory = Mock(return_value=monitor)
        callback = Mock()

        remove = self.backend.install_input_monitor(callback)
        remove()

        self.backend.input_monitor_factory.assert_called_once_with(
            self.backend.user32,
            self.backend.kernel32,
            callback,
        )
        monitor.start.assert_called_once_with()
        monitor.stop.assert_called_once_with()

    def test_real_low_level_input_monitor_starts_and_stops_without_blocking(self):
        remove = self.backend.install_input_monitor(lambda details: None)
        try:
            self.assertTrue(callable(remove))
        finally:
            remove()

    def test_input_monitor_startup_timeout_stops_its_worker(self):
        from environment_protection.input_hooks import LowLevelInputMonitor

        monitor = LowLevelInputMonitor(Mock(), Mock(), lambda event: None)
        entered = threading.Event()

        def delayed_run():
            entered.set()
            monitor._stop_requested.wait(2)

        self.backend.input_monitor_factory = lambda *args: monitor
        with (
            patch.object(monitor, "_run", delayed_run),
            patch.object(monitor._ready, "wait", return_value=False),
        ):
            try:
                with self.assertRaises(TimeoutError):
                    self.backend.install_input_monitor(lambda details: None)
                self.assertTrue(entered.wait(1))
                self.assertTrue(monitor._stop_requested.is_set())
                self.assertFalse(monitor._thread.is_alive())
            finally:
                monitor.stop()

    def test_input_monitor_cleanup_failure_keeps_original_startup_error(self):
        monitor = Mock()
        monitor.start.side_effect = TimeoutError("startup timed out")
        monitor.stop.side_effect = RuntimeError("cleanup timed out")
        self.backend.input_monitor_factory = lambda *args: monitor
        with self.assertRaisesRegex(TimeoutError, "startup timed out"):
            self.backend.install_input_monitor(lambda details: None)
