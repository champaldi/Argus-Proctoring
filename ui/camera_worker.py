"""Camera capture and a bounded, latest-frame-only analysis worker."""

from __future__ import annotations

import threading
import time
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from config import AppConfig
from core.detectors import DetectorCollection
from core.pipeline import EventPipeline


class CameraWorker(QObject):
    # A notification carries no pixels: a busy UI consumes only the latest image.
    frame_ready = Signal()
    module_status = Signal(object)
    camera_status = Signal(bool, str)
    analysis_error = Signal(str)
    performance_ready = Signal(float, float, float, float)
    finished = Signal()

    def __init__(self, config: AppConfig, detectors: DetectorCollection,
                 pipeline: EventPipeline) -> None:
        super().__init__()
        self.config = config
        self.detectors = detectors
        self.pipeline = pipeline
        self._stop_event = threading.Event()
        self._frames = threading.Condition()
        self._latest_frame: Any = None
        self._frame_number = 0
        self._preview: QImage | None = None
        self._preview_notified = False
        self._capture: Any = None
        self._analysis_thread: threading.Thread | None = None
        self._analyzed_frames = 0
        self._phone_time_total = 0.0
        self._gaze_time_total = 0.0

    def stop(self) -> None:
        self._stop_event.set()
        with self._frames:
            self._frames.notify_all()

    def snapshot(self) -> Any:
        with self._frames:
            frame = self._latest_frame
        return None if frame is None else frame.copy()

    def take_preview(self) -> QImage | None:
        with self._frames:
            image = self._preview
            self._preview = None
            self._preview_notified = False
        return image

    def run(self) -> None:
        try:
            self._run_camera()
        except Exception as exc:
            self.analysis_error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.stop()
            try:
                if self._capture is not None:
                    self._capture.release()
            except Exception as exc:
                self.analysis_error.emit(f"{type(exc).__name__}: {exc}")
            finally:
                self._capture = None
                # Keep the QObject alive until in-flight inference and evidence
                # submission finish. MainWindow waits asynchronously for us.
                try:
                    if self._analysis_thread is not None:
                        self._analysis_thread.join()
                    else:
                        self.detectors.close()
                finally:
                    self.camera_status.emit(False, "Камера остановлена")
                    self.finished.emit()

    def _run_camera(self) -> None:
        """Capture and analyse in one loop, as in the 6 October version.

        A separate analysis thread (added later) delivered frames to the
        detectors irregularly on the team's laptops, and gaze and phone events
        were missed. Here every analysis takes the frame just captured, at
        ``analysis_fps``, exactly as when detection was verified on camera.
        """
        if self._stop_event.is_set():
            return
        try:
            import cv2
        except ImportError:
            self.camera_status.emit(False, "Не установлен OpenCV")
            return

        for status in self.detectors.load():
            self.module_status.emit(status)
        if self._stop_event.is_set():
            return

        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
        capture = cv2.VideoCapture(self.config.camera_index, backend)
        self._capture = capture
        if not capture.isOpened():
            self.camera_status.emit(False, f"Камера {self.config.camera_index} недоступна")
            return
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera_status.emit(True, "Камера активна")

        analysis_interval = 1.0 / self.config.analysis_fps
        preview_interval = 1.0 / self.config.preview_fps
        last_analysis = 0.0
        last_preview = 0.0
        last_error: dict[str, float] = {}
        performance_started = time.monotonic()
        captured_frames = 0
        analyzed_frames = 0
        phone_time_total = 0.0
        gaze_time_total = 0.0

        while not self._stop_event.is_set():
            ok, frame = capture.read()
            if self._stop_event.is_set():
                break
            if not ok:
                self.camera_status.emit(False, "Не удалось получить кадр")
                break
            with self._frames:
                self._latest_frame = frame
                self._frame_number += 1
            captured_frames += 1

            now = time.monotonic()
            if now - last_analysis >= analysis_interval:
                events, errors = self.detectors.analyze(frame)
                analyzed_frames += 1
                phone_time_total += self.detectors.last_timings_ms.get("phone", 0.0)
                gaze_time_total += self.detectors.last_timings_ms.get("gaze", 0.0)
                for event in events:
                    self.pipeline.submit(event, frame)
                if not events:
                    # A calm frame from the start of the test becomes the
                    # control photo shown to the teacher next to the name.
                    self.pipeline.offer_reference_frame(frame)
                for message in errors:
                    if now - last_error.get(message, float("-inf")) >= 5.0:
                        self.analysis_error.emit(message)
                        last_error[message] = now
                last_analysis = now

            now = time.monotonic()
            if now - last_preview >= preview_interval:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = rgb.shape
                image = QImage(rgb.data, width, height, rgb.strides[0],
                               QImage.Format.Format_RGB888).copy()
                with self._frames:
                    self._preview = image
                    notify = not self._preview_notified
                    self._preview_notified = True
                if notify:
                    self.frame_ready.emit()
                last_preview = now

            elapsed = now - performance_started
            if elapsed >= 5.0:
                self.performance_ready.emit(
                    captured_frames / elapsed,
                    analyzed_frames / elapsed,
                    phone_time_total / max(1, analyzed_frames),
                    gaze_time_total / max(1, analyzed_frames),
                )
                performance_started = now
                captured_frames = analyzed_frames = 0
                phone_time_total = gaze_time_total = 0.0
