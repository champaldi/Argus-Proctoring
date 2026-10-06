"""Interactive, local demo. A camera is opened only when this script is run."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np

from .analyzer import AnalyzerConfig, FaceMetrics, GazeAnalyzer

WINDOW_NAME = "Gaze and head demo"
CALIBRATION_SECONDS = 2.0


@dataclass
class CalibrationCapture:
    target: str
    started_at: float
    samples: list[FaceMetrics] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local face/gaze demo; press C, K, R, Q or Esc in the window."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    source.add_argument("--video", type=Path, help="Read an existing video instead of a camera")
    parser.add_argument(
        "--output", type=Path, help="Write events as UTF-8 JSONL (replaces the file)"
    )
    parser.add_argument(
        "--side-seconds", type=float, default=3.0, help="Gaze-side episode threshold"
    )
    parser.add_argument(
        "--down-seconds", type=float, default=5.0, help="Gaze-down episode threshold"
    )
    return parser


def write_events(events: list[dict], output: TextIO | None) -> None:
    for event in events:
        line = json.dumps(event, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        if output is not None:
            output.write(line + "\n")
            output.flush()


def draw_overlay(
    frame: np.ndarray,
    analyzer: GazeAnalyzer,
    message: str,
    calibration: CalibrationCapture | None,
    timestamp: float,
    screen_calibrated: bool,
    keyboard_calibrated: bool,
) -> np.ndarray:
    preview = frame.copy()
    result = analyzer.last_result
    lines = [
        f"Faces: {result.face_count}   Time: {timestamp:.2f}s",
        "Signals: " + (", ".join(result.signals) or "none"),
    ]
    ratio = analyzer.get_face_width_ratio()
    lines.append("Face width: unavailable" if ratio is None else f"Face width: {ratio:.1%}")
    metrics = result.metrics
    if metrics is not None:
        pose = metrics.head_pose
        lines.append(f"Head deg: pitch={pose.pitch:+.1f} yaw={pose.yaw:+.1f} roll={pose.roll:+.1f}")
        if metrics.iris_x is None or metrics.iris_y is None:
            lines.append("Iris: unavailable / blink")
        else:
            lines.append(f"Iris: x={metrics.iris_x:.2f} y={metrics.iris_y:.2f}")
        height, width = preview.shape[:2]
        for x, y in metrics.iris_centers:
            point = (round(x * width), round(y * height))
            cv2.circle(preview, point, 3, (0, 255, 255), -1)
    lines.append(
        "Timers: " + "  ".join(f"{key}={value:.1f}s" for key, value in result.elapsed.items())
    )
    lines.append(
        f"Calibration: screen={'yes' if screen_calibrated else 'no'} "
        f"keyboard={'yes' if keyboard_calibrated else 'no'}"
    )
    if calibration is not None:
        remaining = max(0.0, CALIBRATION_SECONDS - (timestamp - calibration.started_at))
        lines.append(
            f"Look at {calibration.target}: {remaining:.1f}s "
            f"({len(calibration.samples)} valid samples)"
        )
    else:
        lines.append(message)
    lines.append("C: screen  K: keyboard  R: reset timers  Q/Esc: quit")

    # OpenCV's built-in font is ASCII; controls and measurements stay in English.
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = max(0.35, min(0.55, preview.shape[1] / 1400))
    line_height = 23
    for index, line in enumerate(lines):
        position = (10, 24 + index * line_height)
        cv2.putText(preview, line, position, font, text_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, line, position, font, text_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def run(args: argparse.Namespace) -> int:
    config = AnalyzerConfig(
        gaze_side_seconds=args.side_seconds,
        gaze_down_seconds=args.down_seconds,
    )
    if args.video is not None and args.output is not None:
        if args.video.resolve() == args.output.resolve():
            raise ValueError("Event output must be different from the input video.")
    source = str(args.video) if args.video is not None else args.camera
    with ExitStack() as stack:
        capture = cv2.VideoCapture(source)
        stack.callback(capture.release)
        stack.callback(cv2.destroyAllWindows)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {'video' if args.video else 'camera'}: {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if args.video is not None and (not math.isfinite(fps) or fps <= 0):
            raise RuntimeError("Video has no valid FPS; reliable source timestamps are required.")
        output = (
            stack.enter_context(args.output.open("w", encoding="utf-8"))
            if args.output is not None
            else None
        )
        analyzer = stack.enter_context(GazeAnalyzer(config))
        started_at = time.monotonic()
        frame_index = 0
        calibration: CalibrationCapture | None = None
        screen_calibrated = False
        keyboard_calibrated = False
        message = "Press C and look at the screen for 2 seconds."

        while True:
            success, frame = capture.read()
            if not success:
                if args.video is None:
                    print(
                        "Camera read failed; stopping without creating a no_face event.",
                        file=sys.stderr,
                    )
                break  # EOF/read failure is not an observation of an empty scene.
            timestamp = (
                frame_index / fps if args.video is not None else time.monotonic() - started_at
            )
            frame_index += 1
            events = analyzer.analyze(frame, timestamp=timestamp)
            if calibration is not None:
                events = [
                    event for event in events if event["type"] not in ("gaze_down", "gaze_side")
                ]
            write_events(events, output)

            if calibration is not None:
                result = analyzer.last_result
                metrics = result.metrics
                if (
                    result.face_count == 1
                    and metrics is not None
                    and metrics.iris_x is not None
                    and metrics.iris_y is not None
                ):
                    calibration.samples.append(metrics)
                if timestamp - calibration.started_at >= CALIBRATION_SECONDS:
                    try:
                        analyzer.calibrate(calibration.samples, target=calibration.target)
                    except ValueError as error:
                        message = "Calibration failed: keep one face and eyes visible, then retry."
                        print(f"Calibration failed: {error}", file=sys.stderr)
                    else:
                        if calibration.target == "screen":
                            screen_calibrated = True
                            keyboard_calibrated = False
                            message = "Screen calibrated. Press K and look at the keyboard."
                        else:
                            keyboard_calibrated = True
                            message = "Keyboard calibrated. Validate gaze and face-count events."
                        print(f"{calibration.target} calibration complete.", file=sys.stderr)
                    calibration = None

            preview = draw_overlay(
                frame,
                analyzer,
                message,
                calibration,
                timestamp,
                screen_calibrated,
                keyboard_calibrated,
            )
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                calibration = CalibrationCapture("screen", timestamp)
            elif key in (ord("k"), ord("K")):
                if screen_calibrated:
                    calibration = CalibrationCapture("keyboard", timestamp)
                else:
                    message = "Calibrate the screen first: press C."
            elif key in (ord("r"), ord("R")):
                analyzer.reset()
                calibration = None
                message = "Timers reset. Screen/keyboard calibration retained."
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
