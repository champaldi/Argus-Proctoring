"""Measure the shared camera loop with YOLO and MediaPipe enabled together."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PROCTOR_PHONE_MODULE", "detection.phone_detector")
os.environ.setdefault("PROCTOR_GAZE_MODULE", "gaze")

from core.detectors import DetectorCollection  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("OpenCV is not installed. Install requirements.txt first.", file=sys.stderr)
        return 2

    detectors = DetectorCollection(
        os.environ["PROCTOR_PHONE_MODULE"],
        os.environ["PROCTOR_GAZE_MODULE"],
    )
    statuses = detectors.load()
    failed = [status for status in statuses if not status.loaded]
    if failed:
        for status in failed:
            print(f"{status.name}: {status.message}", file=sys.stderr)
        detectors.close()
        return 2

    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    # This is the only VideoCapture in the benchmark. Both modules receive the
    # exact same array returned by each read below.
    capture = cv2.VideoCapture(args.camera, backend)
    if not capture.isOpened():
        capture.release()
        detectors.close()
        print(f"Camera {args.camera} is unavailable.", file=sys.stderr)
        return 3

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_times: list[float] = []
    phone_times: list[float] = []
    gaze_times: list[float] = []
    errors: list[str] = []
    try:
        # Initialize both models before starting the warm-up clock; otherwise
        # model construction/download would pollute the FPS measurement.
        ok, first_frame = capture.read()
        if not ok:
            print("Camera stopped returning frames.", file=sys.stderr)
            return 4
        _, startup_errors = detectors.analyze(first_frame)
        if startup_errors:
            for message in startup_errors:
                print(message, file=sys.stderr)
            return 6

        warmup_until = time.perf_counter() + max(0.0, args.warmup)
        finish_at = warmup_until + max(1.0, args.seconds)
        while time.perf_counter() < finish_at:
            ok, frame = capture.read()
            if not ok:
                print("Camera stopped returning frames.", file=sys.stderr)
                return 4
            started = time.perf_counter()
            _, current_errors = detectors.analyze(frame)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if time.perf_counter() >= warmup_until:
                frame_times.append(elapsed_ms)
                phone_times.append(detectors.last_timings_ms.get("phone", 0.0))
                gaze_times.append(detectors.last_timings_ms.get("gaze", 0.0))
                errors.extend(current_errors)
        measured_seconds = max(0.001, time.perf_counter() - warmup_until)
    finally:
        capture.release()
        detectors.close()

    if not frame_times:
        print("No benchmark samples were collected.", file=sys.stderr)
        return 5
    total_seconds = sum(frame_times) / 1000.0
    effective_fps = len(frame_times) / total_seconds if total_seconds else 0.0
    print(f"Samples: {len(frame_times)}")
    print(f"Observed shared-loop FPS: {len(frame_times) / measured_seconds:.2f}")
    print(f"Analysis-only throughput: {effective_fps:.2f} FPS")
    print(
        f"Combined latency: mean {statistics.mean(frame_times):.1f} ms, "
        f"p95 {percentile(frame_times, 0.95):.1f} ms"
    )
    print(f"YOLO adapter: mean {statistics.mean(phone_times):.1f} ms per input frame")
    print(f"MediaPipe adapter: mean {statistics.mean(gaze_times):.1f} ms per input frame")
    if errors:
        print(f"Analyzer errors: {len(errors)}", file=sys.stderr)
        for message in sorted(set(errors)):
            print(message, file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
