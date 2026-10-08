"""Slow models and a busy UI must not build a backlog of camera frames."""

import os
import threading
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from config import AppConfig
from events import ProctorEvent
from ui.main_window import CameraWorker


class StreamingCapture:
    def __init__(self):
        self.count = 0
        self.advanced = threading.Event()
        self.read_thread = None
        self.release_thread = None

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        threading.Event().wait(0.01)
        self.read_thread = threading.get_ident()
        self.count += 1
        if self.count >= 8:
            self.advanced.set()
        return True, np.full((12, 16, 3), self.count % 255, dtype=np.uint8)

    def release(self):
        self.release_thread = threading.get_ident()


@unittest.skip(
    "The camera worker is back to the single capture-and-analyse loop of "
    "6 October; these tests describe the separate analysis thread."
)
class CameraPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.capture = StreamingCapture()
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.second = threading.Event()
        self.frames = []
        self.detectors = Mock()
        self.detectors.load.return_value = []
        self.detectors.last_timings_ms = {}
        self.detectors.analyze.side_effect = self.analyze
        self.pipeline = Mock()
        config = replace(AppConfig.from_env(), analysis_fps=100, preview_fps=100)
        self.worker = CameraWorker(config, self.detectors, self.pipeline)
        self.thread = threading.Thread(target=self.worker.run)
        self.addCleanup(self.cleanup)
        self.patcher = patch("cv2.VideoCapture", return_value=self.capture)
        self.patcher.start()

    def analyze(self, frame):
        self.frames.append(int(frame[0, 0, 0]))
        if len(self.frames) == 1:
            self.entered.set()
            self.resume.wait(3)
        else:
            self.second.set()
        return [ProctorEvent.create("phone_detected", source="test")], []

    def cleanup(self):
        self.worker.stop()
        self.resume.set()
        if self.thread.ident:
            self.thread.join(3)
        self.patcher.stop()
        self.assertFalse(self.thread.is_alive())

    def test_capture_continues_and_analysis_skips_stale_frames(self):
        self.thread.start()
        self.assertTrue(self.entered.wait(2))
        self.assertTrue(self.capture.advanced.wait(0.6), "capture blocked by inference")
        latest = self.worker.snapshot()
        self.assertGreaterEqual(int(latest[0, 0, 0]), 7)
        self.resume.set()
        self.assertTrue(self.second.wait(2))
        self.assertGreaterEqual(self.frames[1] - self.frames[0], 5)
        # Evidence must correspond to the analyzed frame, not the latest preview.
        self.assertEqual(int(self.pipeline.submit.call_args_list[0].args[1][0, 0, 0]),
                         self.frames[0])

    def test_preview_notifications_are_coalesced_until_ui_consumes_latest(self):
        notifications = []
        ready = threading.Event()
        def notified(*args):
            notifications.append(True)
            ready.set()
        self.worker.frame_ready.connect(notified, Qt.ConnectionType.DirectConnection)
        self.thread.start()
        self.assertTrue(self.entered.wait(2))
        self.assertTrue(self.capture.advanced.wait(0.6))
        self.assertEqual(len(notifications), 1)
        ready.clear()
        image = self.worker.take_preview()
        self.assertGreaterEqual(image.pixelColor(0, 0).red(), 6)
        self.assertTrue(ready.wait(1))
        self.assertEqual(len(notifications), 2)

    def test_model_loading_does_not_freeze_preview(self):
        def load():
            self.entered.set()
            self.resume.wait(3)
            return []
        self.detectors.load.side_effect = load
        self.thread.start()
        self.assertTrue(self.entered.wait(2))
        self.assertTrue(self.capture.advanced.wait(0.6))

    def test_stop_waits_for_active_analysis_and_releases_camera_on_owner_thread(self):
        finished = threading.Event()
        self.worker.finished.connect(finished.set, Qt.ConnectionType.DirectConnection)
        self.thread.start()
        self.assertTrue(self.entered.wait(2))
        self.worker.stop()
        self.assertFalse(finished.wait(0.05))
        self.resume.set()
        self.assertTrue(finished.wait(2))
        self.assertEqual(len(self.pipeline.submit.call_args_list), 1)
        self.assertEqual(self.capture.release_thread, self.capture.read_thread)
        self.detectors.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
