"""Measure the application's actual preview worker without hooks or saved images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QCoreApplication, QThread, QTimer  # noqa: E402
from config import AppConfig  # noqa: E402
from core.detectors import DetectorCollection  # noqa: E402
from ui.camera_worker import CameraWorker  # noqa: E402


class MeasuredDetectors(DetectorCollection):
    def __init__(self, config):
        super().__init__(config.phone_module, config.gaze_module)
        self.lock = threading.Lock()
        self.completed: list[tuple[float, float]] = []

    def analyze(self, frame):
        start = time.monotonic()
        result = super().analyze(frame)
        end = time.monotonic()
        with self.lock:
            self.completed.append((end, (end - start) * 1000))
        return result


class DiscardEvidence:
    def submit(self, event, frame):
        pass  # Benchmark never creates sessions, evidence files, or OS hooks.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--seconds", type=float, default=15)
    args = parser.parse_args()
    os.environ["PROCTOR_CAMERA_INDEX"] = str(args.camera)
    config = AppConfig.from_env()
    app = QCoreApplication.instance() or QCoreApplication([])
    detectors = MeasuredDetectors(config)
    worker = CameraWorker(config, detectors, DiscardEvidence())
    thread = QThread()
    worker.moveToThread(thread)
    previews: list[float] = []
    errors: list[str] = []
    started = time.monotonic()
    measured_start = None
    measured_end = None

    def preview(*images):
        # Also accepts the former image signal for a local before/after check.
        image = images[0] if images else worker.take_preview()
        if image is not None:
            previews.append(time.monotonic())

    def check():
        nonlocal measured_start, measured_end
        with detectors.lock:
            first = detectors.completed[0][0] if detectors.completed else None
        if first is not None:
            measured_start = first + max(0, args.warmup)
            measured_end = measured_start + max(1, args.seconds)
            if time.monotonic() >= measured_end:
                worker.stop()
        elif time.monotonic() - started > 120:
            errors.append("No completed analysis within 120 seconds")
            worker.stop()

    worker.frame_ready.connect(preview)
    worker.analysis_error.connect(errors.append)
    worker.module_status.connect(lambda status: errors.append(status.message)
                                 if not status.loaded else None)
    worker.camera_status.connect(lambda ok, message: print(message, flush=True))
    worker.finished.connect(thread.quit)
    thread.started.connect(worker.run)
    thread.finished.connect(app.quit)
    timer = QTimer()
    timer.timeout.connect(check)
    timer.start(20)
    thread.start()
    try:
        app.exec()
    finally:
        timer.stop()
        worker.stop()
        thread.quit()
        thread.wait()

    if measured_start is None or measured_end is None or time.monotonic() < measured_end:
        print("Camera run ended before measurement completed", file=sys.stderr)
        return 2
    times = [t for t in previews if measured_start <= t <= measured_end]
    analyses = [ms for t, ms in detectors.completed if measured_start <= t <= measured_end]
    gaps = sorted((b - a) * 1000 for a, b in zip(times, times[1:]))
    if not gaps or not analyses:
        print("Not enough preview/analysis samples", file=sys.stderr)
        return 2
    duration = measured_end - measured_start
    print(f"Preview: {len(times) / duration:.2f} FPS ({len(times)} frames)")
    print(f"Preview gap: p95 {gaps[round((len(gaps)-1)*0.95)]:.1f} ms, max {max(gaps):.1f} ms")
    print(f"Analysis: {len(analyses) / duration:.2f} FPS; mean {statistics.mean(analyses):.1f} ms")
    for message in sorted(set(errors)):
        print(message, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
