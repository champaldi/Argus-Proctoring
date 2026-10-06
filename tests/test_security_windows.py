"""Windows adapter checks without installing hooks or changing desktop focus."""

import os
import sys
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
