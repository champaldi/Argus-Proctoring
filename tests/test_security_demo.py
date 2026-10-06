"""Demo UI lifecycle without real keyboard suppression."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from environment_protection import demo


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_stop_and_close_call_disable_and_refresh_the_status(self):
        state = {"enabled": False, "reason": "not_started", "last_error": None}

        def start(callback, **kwargs):
            state.update(enabled=True, reason="enabled")
            callback({"type": "hotkey_blocked", "details": {"hotkey": "win"}})

        def stop():
            state.update(enabled=False, reason="disabled")

        with (
            patch.object(demo, "enable", side_effect=start),
            patch.object(demo, "disable", side_effect=stop) as disable,
            patch.object(demo, "status", side_effect=lambda: state),
        ):
            window = demo.DemoWindow(60)
            try:
                window.start_protection()
                self.assertFalse(window.start_button.isEnabled())
                self.assertIn("hotkey_blocked", window.log.toPlainText())
                window.stop_protection()
                self.assertTrue(window.start_button.isEnabled())
                window.close()
                self.assertGreaterEqual(disable.call_count, 2)
                self.assertFalse(window.timer.isActive())
            finally:
                window.close()
