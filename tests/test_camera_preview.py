"""Preview optimizations must preserve the original detector input and timing."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Slot
from shiboken6 import delete

from config import AppConfig
from core.detectors import DetectorCollection
from gaze.analyzer import APP_CONFIG, FaceMetrics, GazeAnalyzer, HeadPose
from ui.main_window import CameraWorker
from ui.preview import make_preview


class CameraPreviewTests(unittest.TestCase):
    def make_worker(self, frames, detectors=None):
        clock = SimpleNamespace(now=10.0)
        source = iter(frames)
        capture = SimpleNamespace(
            isOpened=lambda: True, set=lambda *args: None, release=lambda: None,
        )

        def read():
            clock.now += 0.25
            return (True, frame) if (frame := next(source, None)) is not None else (False, None)

        capture.read = read
        received = []
        if detectors is None:
            detectors = SimpleNamespace(
                load=lambda: [], analyze=lambda frame: ([], []), close=lambda: None,
                last_timings_ms={},
            )
        pipeline = SimpleNamespace(submit=lambda event, frame: received.append((event, frame)))
        with patch.dict(os.environ, {"PROCTOR_ANALYSIS_FPS": "6", "PROCTOR_PREVIEW_FPS": "20"}):
            worker = CameraWorker(AppConfig.from_env(), detectors, pipeline)
        self.addCleanup(delete, worker)
        return worker, capture, clock, received

    def run_worker(self, worker, capture, clock):
        with (
            patch.object(cv2, "VideoCapture", return_value=capture),
            patch("ui.main_window.time.monotonic", side_effect=lambda: clock.now),
        ):
            worker.run()

    def test_first_preview_precedes_loading_and_first_inference(self):
        order = []
        detector = SimpleNamespace(
            load=lambda: order.append("load") or [],
            analyze=lambda frame: (order.append("analyze") or [], []),
            close=lambda: None, last_timings_ms={},
        )
        worker, capture, clock, _ = self.make_worker([np.zeros((720, 1280, 3), np.uint8)], detector)
        worker.frame_ready.connect(lambda *args: order.append("preview"), Qt.ConnectionType.DirectConnection)
        self.run_worker(worker, capture, clock)
        self.assertEqual(order[:3], ["preview", "load", "analyze"])

    def test_slow_ui_gets_one_notification_and_the_latest_small_frame(self):
        frames = [np.full((720, 1280, 3), (10, 20, red), np.uint8) for red in (30, 60, 90)]
        worker, capture, clock, _ = self.make_worker(frames)
        notifications = []
        worker.frame_ready.connect(lambda *args: notifications.append(args), Qt.ConnectionType.DirectConnection)
        self.run_worker(worker, capture, clock)
        self.assertEqual(len(notifications), 1, "UI backlog must not retain every camera image")
        image = worker.take_preview()
        self.assertEqual((image.width(), image.height()), (640, 360))
        self.assertEqual(image.pixelColor(0, 0).getRgb(), (90, 20, 10, 255))
        frames[-1][:] = 0
        self.assertEqual(image.pixelColor(0, 0).getRgb(), (90, 20, 10, 255))
        self.assertIsNone(worker.take_preview())

    def test_side_gaze_in_both_directions_survives_a_busy_preview(self):
        for yaw, iris_x, expected in (
            (-32.0, 0.5, ["gaze_side"]),
            (32.0, 0.5, ["gaze_side"]),
            (0.0, 0.2, ["gaze_side"]),
            (0.0, 0.8, ["gaze_side"]),
            (0.0, 0.5, []),
        ):
            with self.subTest(yaw=yaw, iris_x=iris_x):
                # Full BGR input is deliberately asymmetric: resizing, mirroring
                # or swapping channels before inference must break this check.
                frame = np.zeros((720, 1280, 3), np.uint8)
                frame[:, :640] = (11, 22, 33)
                frames = [frame.copy() for _ in range(23)]
                mesh = SimpleNamespace(
                    process=lambda rgb: SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=[])]),
                    close=lambda: None,
                )
                detector = DetectorCollection("unused_phone", "unused_gaze")
                worker, capture, clock, received = self.make_worker(frames, detector)
                gaze = GazeAnalyzer(APP_CONFIG, face_mesh=mesh, clock=lambda: clock.now)
                self.addCleanup(gaze.close)
                inputs = []
                detector.adapters[0]._function = lambda value: inputs.append(value) or []
                detector.adapters[1]._function = gaze.analyze
                metric = FaceMetrics(HeadPose(0, yaw, 0), iris_x, 0.5)
                with (
                    patch.object(detector, "load", return_value=[]),
                    patch("gaze.analyzer.measure_face", return_value=metric),
                ):
                    self.run_worker(worker, capture, clock)
                self.assertEqual(len(inputs), len(frames))
                for actual, original in zip(inputs, frames):
                    self.assertIs(actual, original)
                    np.testing.assert_array_equal(actual, frame)
                self.assertEqual([event.type.value for event, _ in received], expected)
                if expected:
                    self.assertIs(received[0][1], frames[8])

    def test_consuming_preview_allows_the_next_notification(self):
        frames = [np.full((40, 60, 3), (0, 0, red), np.uint8) for red in (30, 60, 90)]
        worker, capture, clock, _ = self.make_worker(frames)
        colors = []
        worker.frame_ready.connect(
            lambda: colors.append(worker.take_preview().pixelColor(0, 0).red()),
            Qt.ConnectionType.DirectConnection,
        )
        self.run_worker(worker, capture, clock)
        self.assertEqual(colors, [30, 60, 90])

    def test_stop_after_first_preview_does_not_start_models(self):
        calls = []
        detector = SimpleNamespace(
            load=lambda: calls.append("load") or [],
            analyze=lambda frame: (calls.append("analyze") or [], []),
            close=lambda: None, last_timings_ms={},
        )
        worker, capture, clock, _ = self.make_worker([np.zeros((40, 60, 3), np.uint8)], detector)
        worker.frame_ready.connect(worker.stop, Qt.ConnectionType.DirectConnection)
        self.run_worker(worker, capture, clock)
        self.assertEqual(calls, [])

    def test_preview_keeps_aspect_ratio_and_handles_noncontiguous_bgr(self):
        frame = np.full((800, 400, 3), (10, 20, 30), np.uint8)
        image = make_preview(frame[:, ::-1], 300, 300)
        self.assertEqual((image.width(), image.height()), (150, 300))
        self.assertEqual(image.pixelColor(0, 0).getRgb(), (30, 20, 10, 255))

    def test_original_snapshot_is_not_resized_or_shared_with_preview(self):
        original = np.full((720, 1280, 3), (11, 22, 33), np.uint8)
        worker, capture, clock, _ = self.make_worker([original])
        self.run_worker(worker, capture, clock)
        snapshot = worker.snapshot()
        self.assertEqual(snapshot.shape, (720, 1280, 3))
        np.testing.assert_array_equal(snapshot, original)
        snapshot[:] = 0
        self.assertEqual(tuple(original[0, 0]), (11, 22, 33))


class PreviewUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_consumes_preview_and_displays_correct_colors(self):
        from core.pipeline import EventPipeline
        from ui.main_window import MainWindow
        with (
            patch.object(MainWindow, "start_monitoring"),
            patch.object(EventPipeline, "start"),
            patch.dict(os.environ, {"PROCTOR_WATERMARK": "0"}),
        ):
            window = MainWindow(AppConfig.from_env())
            worker = CameraWorker(window.config, window.detectors, window.pipeline)
            window.camera_worker = worker
            try:
                window.resize(1280, 780)
                window.show()
                self.app.processEvents()
                window._set_preview_size()
                worker._publish_preview(np.full((720, 1280, 3), (10, 20, 30), np.uint8))
                window._show_frame()
                pixmap = window.camera_label.pixmap()
                self.assertFalse(pixmap.isNull())
                self.assertEqual(pixmap.toImage().pixelColor(0, 0).getRgb(), (30, 20, 10, 255))
                # Fractional Windows scaling rounds logical sizes to physical pixels.
                self.assertLessEqual(
                    pixmap.deviceIndependentSize().width(),
                    window.camera_label.width() + 1 / pixmap.devicePixelRatioF(),
                )
                self.assertIsNone(worker.take_preview())
                window._show_frame()  # A redundant notification must be harmless.
                self.assertFalse(window.camera_label.pixmap().isNull())
            finally:
                window.close()
                delete(worker)
                delete(window)

    def test_queued_delivery_runs_on_gui_thread_and_coalesces_busy_ui(self):
        worker = CameraWorker(AppConfig.from_env(), SimpleNamespace(), SimpleNamespace())
        thread = QThread()
        callbacks = []

        class Receiver(QObject):
            @Slot()
            def ready(self):
                image = worker.take_preview()
                callbacks.append((QThread.currentThread(), image.pixelColor(0, 0).red()))

        receiver = Receiver()
        worker.moveToThread(thread)
        worker.frame_ready.connect(receiver.ready)

        def publish():
            try:
                for red in range(30):
                    worker._publish_preview(np.full((40, 60, 3), (0, 0, red), np.uint8))
            finally:
                worker.moveToThread(self.app.thread())
                thread.quit()

        thread.started.connect(publish, Qt.ConnectionType.DirectConnection)
        try:
            thread.start()
            self.assertTrue(thread.wait(5000))  # Hold the UI until publication finishes.
            self.assertEqual(callbacks, [])
            self.app.processEvents()
            self.assertEqual(callbacks, [(self.app.thread(), 29)])
        finally:
            thread.quit()
            thread.wait()
            delete(receiver)
            delete(worker)
            delete(thread)


if __name__ == "__main__":
    unittest.main()
