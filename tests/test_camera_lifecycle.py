"""Camera shutdown stays responsive and preserves pending evidence."""

import os
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from config import AppConfig
from events import ProctorEvent
from ui.main_window import CameraWorker, MainWindow, QUESTIONS


class GatedWorker(QObject):
    frame_ready = Signal(object)
    camera_status = Signal(bool, str)
    module_status = Signal(object)
    analysis_error = Signal(str)
    performance_ready = Signal(float, float, float, float)
    finished = Signal()

    def __init__(self, config, detectors, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.entered = threading.Event()
        self.release = threading.Event()

    def run(self):
        self.entered.set()
        self.release.wait(5)
        self.pipeline.submit(ProctorEvent.create("phone_detected", source="test"))
        self.finished.emit()

    def stop(self):
        pass  # Models already running finish before they can observe cancellation.


class CameraLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config = replace(AppConfig.from_env(), database_path=root / "events.db",
                              screenshots_dir=root / "shots")

    def make_window(self):
        camera = patch("ui.main_window.CameraWorker", GatedWorker)
        security = patch("ui.main_window.SecurityAdapter")
        camera.start()
        adapter = security.start()
        adapter.return_value.enable.return_value = (True, "test")
        self.addCleanup(camera.stop)
        self.addCleanup(security.stop)
        window = MainWindow(self.config)
        self.app.processEvents()
        self.assertTrue(window.camera_worker.entered.wait(2))
        self.addCleanup(self.clean_window, window)
        return window

    def clean_window(self, window):
        window.camera_worker.release.set()
        window.camera_thread.quit()
        window.camera_thread.wait(2000)
        window.pipeline.stop()
        window.close()
        self.app.processEvents()

    def pump_until(self, predicate):
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            QThread.msleep(5)
        self.assertTrue(predicate())

    def test_close_waits_for_busy_camera_without_blocking_ui(self):
        window = self.make_window()
        event = QCloseEvent()
        # Old shutdown waits four seconds; shorten only that wait in the red run.
        with patch.object(window.camera_thread, "wait", return_value=False):
            started = time.monotonic()
            window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertFalse(window._shutdown_done)
        window.camera_worker.release.set()
        self.pump_until(lambda: window._shutdown_done)
        self.assertFalse(window.camera_thread.isRunning())
        self.assertEqual(window.pipeline.peak_risk().total, 25)

    def test_results_wait_for_camera_and_pending_event(self):
        window = self.make_window()
        window.question_index = len(QUESTIONS) - 1
        window._render_question()
        window.answer_group.button(0).setChecked(True)
        with patch("ui.main_window.TeacherReviewDialog") as review:
            with patch.object(window.camera_thread, "wait", return_value=False):
                window._next_question()
            self.assertFalse(review.called)
            window.camera_worker.release.set()
            self.pump_until(lambda: review.called)
            values = review.call_args.kwargs
            self.assertEqual(values["final_risk"], 25)
            self.assertEqual(len(values["events"]), 1)

    def test_cancel_before_run_does_not_open_camera(self):
        detectors = Mock()
        detectors.load.return_value = []
        worker = CameraWorker(self.config, detectors, Mock())
        worker.stop()
        with patch("cv2.VideoCapture") as capture:
            worker.run()
        capture.assert_not_called()

    def test_camera_setup_failure_still_finishes_worker(self):
        detectors = Mock()
        detectors.load.return_value = []
        worker = CameraWorker(self.config, detectors, Mock())
        finished = []
        worker.finished.connect(lambda: finished.append(True))
        with patch("cv2.VideoCapture", side_effect=RuntimeError("camera setup failed")):
            worker.run()
        self.assertEqual(finished, [True])
        detectors.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
