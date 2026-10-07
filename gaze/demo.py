"""Interactive, local demo. A camera is opened only when this script is run."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TextIO

import cv2
import numpy as np

from .analyzer import (
    CALIBRATION_MAX_SECONDS,
    CALIBRATION_MIN_SECONDS,
    DEFAULT_GAZE_GAP_SECONDS,
    DEFAULT_MIN_EXTRA_FACE_RATIO,
    EVENT_TYPES,
    AnalysisResult,
    AnalyzerConfig,
    FaceMetrics,
    GazeAnalyzer,
    is_calibration_sample_valid,
)

WINDOW_NAME = "Gaze and head demo"


@dataclass
class CalibrationCapture:
    target: str
    started_at: float
    samples: list[FaceMetrics] = field(default_factory=list)


def key_command(raw_code: int) -> str | None:
    """Match the full window key code in English and Russian keyboard layouts."""
    if raw_code == 27:
        return "quit"
    for command, characters in (
        ("screen", "CcСс"),
        ("keyboard", "KkЛл"),
        ("reset", "RrКк"),
        ("quit", "QqЙй"),
    ):
        if raw_code in map(ord, characters):
            return command
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Проверка взгляда и лица с камеры. Клавиши в окне: C, K, R, Q или Esc.",
        epilog=(
            "Запуск: python -m gaze.demo --camera 0. "
            "C — экран, K — клавиатура, R — сброс таймеров, Q/Esc — выход."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Номер камеры (по умолчанию: 0)")
    source.add_argument("--video", type=Path, help="Проверить существующий видеофайл")
    parser.add_argument(
        "--output", type=Path, help="Сохранить события в UTF-8 JSONL (файл будет перезаписан)"
    )
    parser.add_argument(
        "--telemetry", type=Path, help="Сохранить измерения каждого кадра в JSONL без изображений"
    )
    parser.add_argument(
        "--profile", type=Path, help="Загрузить профиль калибровки и сохранять его после C/K"
    )
    parser.add_argument(
        "--side-seconds", type=float, default=3.0, help="Время взгляда в сторону до события"
    )
    parser.add_argument(
        "--down-seconds", type=float, default=5.0, help="Время взгляда вниз до события"
    )
    return parser


def write_events(events: list[dict], output: TextIO | None) -> None:
    for event in events:
        line = json.dumps(event, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        if output is not None:
            output.write(line + "\n")
            output.flush()


def validate_output_paths(
    video: Path | None,
    events: Path | None,
    telemetry: Path | None,
    profile: Path | None = None,
) -> None:
    paths = [path.resolve() for path in (video, events, telemetry, profile) if path is not None]
    if len(paths) != len(set(paths)):
        raise ValueError("Пути видео, событий, телеметрии и профиля должны различаться.")


def load_profile(analyzer: GazeAnalyzer, path: Path) -> bool:
    if not path.exists():
        return False
    try:
        profile = json.loads(path.read_text(encoding="utf-8-sig"))
        analyzer.import_calibration(profile)
    except ValueError as error:
        raise ValueError(f"Некорректный профиль калибровки {path}: {error}") from error
    return True


def save_profile(analyzer: GazeAnalyzer, path: Path) -> None:
    """Replace only after the complete profile has been written and closed."""
    payload = json.dumps(
        analyzer.export_calibration(), ensure_ascii=False, allow_nan=False, indent=2
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload + "\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_telemetry(
    output: TextIO,
    result: AnalysisResult,
    diagnostics: dict,
    *,
    timestamp: float,
    calibration: CalibrationCapture | None,
    screen_calibrated: bool = False,
    keyboard_calibrated: bool = False,
) -> None:
    record = {
        "timestamp": timestamp,
        "face_count": result.face_count,
        "face_width_ratio": result.face_width_ratio,
        "metrics": asdict(result.metrics) if result.metrics is not None else None,
        "diagnostics": diagnostics,
        "signals": result.signals,
        "elapsed": result.elapsed,
        "calibrating": calibration is not None,
        "screen_calibrated": screen_calibrated,
        "keyboard_calibrated": keyboard_calibrated,
        "calibration": (
            {"target": calibration.target, "samples": len(calibration.samples)}
            if calibration is not None
            else None
        ),
    }
    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    output.flush()


def _diagnostic_number(value: float | None, *, signed: bool = False, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "?"
    return format(value, f"{'+' if signed else ''}.{digits}f")


def _pose_values(metrics: dict | None) -> tuple[float | None, float | None]:
    pose = metrics.get("head_pose") if metrics is not None else None
    return (pose.get("pitch"), pose.get("yaw")) if pose is not None else (None, None)


def _iris_y(metrics: dict | None) -> float | None:
    return metrics.get("iris_y") if metrics is not None else None


def format_diagnostics_lines(
    diagnostics: dict, config: AnalyzerConfig, *, calibrating: bool = False
) -> list[str]:
    """Show measurements and thresholds without duplicating detector decisions."""
    raw = diagnostics.get("raw_metrics")
    smooth = diagnostics.get("smoothed_metrics")
    screen = diagnostics.get("screen")
    keyboard = diagnostics.get("keyboard")
    raw_pitch, raw_yaw = _pose_values(raw)
    pitch, yaw = _pose_values(smooth)
    screen_pitch, screen_yaw = _pose_values(screen) if screen is not None else (0.0, 0.0)
    screen_iris = _iris_y(screen) if screen is not None else 0.5
    pitch_delta = pitch - screen_pitch if pitch is not None and screen_pitch is not None else None
    iris = _iris_y(smooth)
    iris_delta = iris - screen_iris if iris is not None and screen_iris is not None else None
    lines = [
        f"Head raw p/y={_diagnostic_number(raw_pitch, signed=True)}/"
        f"{_diagnostic_number(raw_yaw, signed=True)} | "
        f"smooth={_diagnostic_number(pitch, signed=True)}/{_diagnostic_number(yaw, signed=True)}",
        (
            f"Screen p/y={_diagnostic_number(screen_pitch, signed=True)}/"
            f"{_diagnostic_number(screen_yaw, signed=True)}"
            if screen is not None
            else "Screen: DEFAULT p/y=0/0 iy=0.50"
        )
        + f" | dp={_diagnostic_number(pitch_delta, signed=True)}",
        f"Iris y: raw={_diagnostic_number(_iris_y(raw), digits=2)} "
        f"smooth={_diagnostic_number(iris, digits=2)} "
        f"screen={_diagnostic_number(screen_iris, digits=2)} "
        f"dy={_diagnostic_number(iris_delta, signed=True, digits=2)}",
    ]
    if keyboard is None:
        lines.append("Keyboard: NOT SET")
    else:
        keyboard_pitch, keyboard_yaw = _pose_values(keyboard)
        zone = diagnostics.get("keyboard_zone")
        zone_text = "?" if zone is None else "YES" if zone else "NO"
        lines.append(
            f"Keyboard p/y={_diagnostic_number(keyboard_pitch, signed=True)}/"
            f"{_diagnostic_number(keyboard_yaw, signed=True)} "
            f"iy={_diagnostic_number(_iris_y(keyboard), digits=2)} zone={zone_text}"
        )
    head_unknown = diagnostics.get("head_unknown", raw_pitch is None)
    iris_unknown = diagnostics.get("iris_unknown", _iris_y(raw) is None)
    down_episode = diagnostics.get("episodes", {}).get("gaze_down", {})
    emitted = down_episode.get("emitted")
    emitted_text = "?" if emitted is None else "yes" if emitted else "no"
    lines.append(
        f"Unknown: head={'yes' if head_unknown else 'no'} iris={'yes' if iris_unknown else 'no'} "
        f"| Down deferred={'yes' if 'gaze_down' in diagnostics.get('deferred', ()) else 'no'} "
        f"emitted={emitted_text}"
    )
    thresholds = diagnostics.get("down_thresholds", {})
    head_threshold = thresholds.get("head_degrees", config.head_down_degrees)
    combined_threshold = thresholds.get("combined_degrees", config.combined_down_degrees)
    iris_threshold = thresholds.get("iris", config.iris_down_threshold)
    combined_iris_threshold = thresholds.get("combined_iris", config.combined_iris_down_threshold)
    lines.append(
        f"Down: dp>={head_threshold:.1f} OR dy>={iris_threshold:.2f} "
        f"OR (dp>={combined_threshold:.1f} AND dy>={combined_iris_threshold:.2f})"
    )
    if calibrating:
        lines.append("COLLECTING PROFILE: gaze_down/gaze_side events are hidden")
    return lines


def select_events(events: list[dict], *, calibrating: bool) -> list[dict]:
    """Do not report gaze changes intentionally made while learning a profile."""
    if not calibrating:
        return list(events)
    return [event for event in events if event["type"] not in ("gaze_down", "gaze_side")]


def update_event_counts(counts: dict[str, int], events: list[dict]) -> dict[str, int]:
    updated = {name: counts.get(name, 0) for name in EVENT_TYPES}
    for event in events:
        if event["type"] in updated:
            updated[event["type"]] += 1
    return updated


def calibration_capture_finished(
    capture: CalibrationCapture, timestamp: float, *, min_samples: int
) -> bool:
    elapsed = timestamp - capture.started_at
    return elapsed >= CALIBRATION_MAX_SECONDS or (
        elapsed >= CALIBRATION_MIN_SECONDS and len(capture.samples) >= min_samples
    )


def format_overlay_lines(
    *,
    result: AnalysisResult,
    config: AnalyzerConfig,
    timestamp: float,
    message: str,
    calibration: CalibrationCapture | None,
    screen_calibrated: bool,
    keyboard_calibrated: bool,
    event_counts: dict[str, int],
    diagnostics: dict | None = None,
) -> list[str]:
    """ASCII diagnostics: OpenCV's built-in font cannot render Russian text."""
    lines = [
        f"Faces: {result.face_count}   Time: {timestamp:.2f}s",
        "Signals: " + (", ".join(result.signals) or "none"),
    ]
    ratio = result.face_width_ratio
    width_text = "unavailable" if ratio is None else f"{ratio:.1%}"
    lines.append(
        f"Face width: {width_text} | alert >{config.face_too_close_ratio:.1%} "
        f"for {config.face_too_close_seconds:.1f}s"
    )
    metrics = result.metrics
    if diagnostics is not None:
        lines.extend(
            format_diagnostics_lines(diagnostics, config, calibrating=calibration is not None)
        )
    elif metrics is not None:
        pose = metrics.head_pose
        lines.append(
            "Head: unknown"
            if pose is None
            else f"Head deg: pitch={pose.pitch:+.1f} yaw={pose.yaw:+.1f} roll={pose.roll:+.1f}"
        )
        if metrics.iris_x is None or metrics.iris_y is None:
            lines.append("Iris: unavailable / blink")
        else:
            lines.append(f"Iris: x={metrics.iris_x:.2f} y={metrics.iris_y:.2f}")
    limits = {
        "gaze_down": config.gaze_down_seconds,
        "gaze_side": config.gaze_side_seconds,
        "no_face": config.no_face_seconds,
        "multiple_faces": config.multiple_faces_seconds,
        "too_close_to_camera": config.face_too_close_seconds,
    }
    timers = [
        f"Timer {name}: {result.elapsed[name]:.1f}/{limits[name]:.1f}s"
        for name in EVENT_TYPES
        if name in result.elapsed
    ]
    lines.extend(timers or ["Timers: none"])
    for names in (EVENT_TYPES[:2], EVENT_TYPES[2:4], EVENT_TYPES[4:]):
        lines.append(
            "Events: " + "  ".join(f"{name}={event_counts.get(name, 0)}" for name in names)
        )
    lines.append(
        f"Calibration: screen={'READY' if screen_calibrated else 'NOT SET'} "
        f"keyboard={'READY' if keyboard_calibrated else 'NOT SET'}"
    )
    if calibration is not None:
        elapsed = max(0.0, timestamp - calibration.started_at)
        lines.append(
            f"Calibrating {calibration.target}: {elapsed:.1f}/{CALIBRATION_MAX_SECONDS:.1f}s "
            f"samples={len(calibration.samples)}/{config.calibration_min_samples}"
        )
        lines.append(
            f"Look at {calibration.target}; keep still for at least {CALIBRATION_MIN_SECONDS:.1f}s."
        )
    else:
        lines.append(message)
    lines.append("C: screen  K: keyboard  R: reset timers  Q/Esc: quit")
    return lines


def draw_overlay(
    frame: np.ndarray,
    analyzer: GazeAnalyzer,
    message: str,
    calibration: CalibrationCapture | None,
    timestamp: float,
    screen_calibrated: bool,
    keyboard_calibrated: bool,
    event_counts: dict[str, int],
) -> np.ndarray:
    preview = frame.copy()
    result = analyzer.last_result
    lines = format_overlay_lines(
        result=result,
        config=analyzer.config,
        timestamp=timestamp,
        message=message,
        calibration=calibration,
        screen_calibrated=screen_calibrated,
        keyboard_calibrated=keyboard_calibrated,
        event_counts=event_counts,
        diagnostics=analyzer.get_diagnostics(),
    )
    height, width = preview.shape[:2]
    if result.metrics is not None:
        for x, y in result.metrics.iris_centers:
            point = (round(x * width), round(y * height))
            cv2.circle(preview, point, 3, (0, 255, 255), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = max(12, min(23, (height - 16) // len(lines)))
    text_scale = min(0.55, width / 1400, line_height / 35)
    longest_width = max(cv2.getTextSize(line, font, text_scale, 1)[0][0] for line in lines)
    if longest_width > width - 20:
        text_scale *= max(1, width - 20) / longest_width
    for index, line in enumerate(lines):
        position = (10, line_height + index * line_height)
        cv2.putText(preview, line, position, font, text_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, line, position, font, text_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return preview


def run(args: argparse.Namespace) -> int:
    config = AnalyzerConfig(
        gaze_side_seconds=args.side_seconds,
        gaze_down_seconds=args.down_seconds,
    )
    validate_output_paths(args.video, args.output, args.telemetry, args.profile)
    source = str(args.video) if args.video is not None else args.camera
    with ExitStack() as stack:
        capture = cv2.VideoCapture(source)
        stack.callback(capture.release)
        stack.callback(cv2.destroyAllWindows)
        if not capture.isOpened():
            if args.video is not None:
                raise RuntimeError(f"Не удалось открыть видео: {source}")
            raise RuntimeError(
                f"Не удалось открыть камеру №{source}. Закройте другие приложения с камерой, "
                "проверьте разрешение на доступ или попробуйте --camera 1."
            )
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if args.video is not None and (not math.isfinite(fps) or fps <= 0):
            raise RuntimeError("У видео неизвестна частота кадров; нельзя корректно считать время.")
        output = (
            stack.enter_context(args.output.open("w", encoding="utf-8"))
            if args.output is not None
            else None
        )
        telemetry = (
            stack.enter_context(args.telemetry.open("w", encoding="utf-8"))
            if args.telemetry is not None
            else None
        )
        # Same reliability settings as gaze.analyze(), so the demo shows exactly
        # what the application will report.
        analyzer = stack.enter_context(
            GazeAnalyzer(
                config,
                gaze_gap_seconds=DEFAULT_GAZE_GAP_SECONDS,
                min_extra_face_ratio=DEFAULT_MIN_EXTRA_FACE_RATIO,
            )
        )
        profile_loaded = args.profile is not None and load_profile(analyzer, args.profile)
        calibration_status = analyzer.get_calibration_status()
        started_at = time.monotonic()
        frame_index = 0
        calibration: CalibrationCapture | None = None
        screen_calibrated = calibration_status["screen"]
        keyboard_calibrated = calibration_status["keyboard"]
        event_counts = update_event_counts({}, [])
        message = "Press C and look at the screen to calibrate."
        if profile_loaded:
            message = (
                "Profile LOADED. Ready to check events; C/K to recalibrate."
                if keyboard_calibrated
                else "Screen profile LOADED. Press K to calibrate keyboard."
                if screen_calibrated
                else "Empty profile LOADED. Press C to calibrate screen."
            )
            print(
                f"Профиль калибровки загружен: {args.profile}. "
                f"Экран: {'готов' if screen_calibrated else 'не настроен'}; "
                f"клавиатура: {'готова' if keyboard_calibrated else 'не настроена'}.",
                file=sys.stderr,
            )
        elif args.profile is not None:
            print(f"Новый профиль будет сохранён после калибровки: {args.profile}", file=sys.stderr)
        print(
            "Проверка запущена. Нажимайте клавиши, когда активно окно с камерой:\n"
            "  C — смотрите на экран; K — смотрите на клавиатуру после калибровки экрана.\n"
            "  Сбор занимает 2–8 секунд; требуется минимум 10 подходящих кадров.\n"
            "  R — сбросить таймеры / отменить сбор; профили и счётчики сохранятся.\n"
            "  Q или Esc — выйти (также можно закрыть окно).\n"
            "  Русская раскладка: С — экран, Л — клавиатура, К — сброс, Й — выход.\n"
            "Ширина лица: событие при >45% кадра в течение 3 секунд.\n"
            "События выводятся в формате JSONL; пояснения — в этот поток сообщений.",
            file=sys.stderr,
        )
        if telemetry is not None:
            print(f"Измерения каждого кадра сохраняются: {args.telemetry}", file=sys.stderr)

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    print(
                        "Видео закончилось."
                        if args.video is not None
                        else "Камера перестала отдавать кадры. Проверка остановлена.",
                        file=sys.stderr,
                    )
                    break  # EOF/read failure is not an observation of an empty scene.
                timestamp = (
                    frame_index / fps if args.video is not None else time.monotonic() - started_at
                )
                frame_index += 1
                events = select_events(
                    analyzer.analyze(frame, timestamp=timestamp),
                    calibrating=calibration is not None,
                )
                event_counts = update_event_counts(event_counts, events)
                write_events(events, output)
                if telemetry is not None:
                    write_telemetry(
                        telemetry,
                        analyzer.last_result,
                        analyzer.get_diagnostics(),
                        timestamp=timestamp,
                        calibration=calibration,
                        screen_calibrated=screen_calibrated,
                        keyboard_calibrated=keyboard_calibrated,
                    )

                if calibration is not None:
                    result = analyzer.last_result
                    metrics = result.metrics
                    if result.face_count == 1 and is_calibration_sample_valid(
                        metrics, target=calibration.target
                    ):
                        calibration.samples.append(metrics)
                    if calibration_capture_finished(
                        calibration, timestamp, min_samples=config.calibration_min_samples
                    ):
                        target_name = "экрана" if calibration.target == "screen" else "клавиатуры"
                        try:
                            analyzer.calibrate(calibration.samples, target=calibration.target)
                        except ValueError:
                            sample_count = len(calibration.samples)
                            message = (
                                f"Calibration FAILED: {sample_count} samples. "
                                "Keep one face visible and retry C/K."
                            )
                            print(
                                f"Калибровка {target_name} не удалась: собрано "
                                f"{sample_count}/{config.calibration_min_samples} кадров. "
                                "Держите одно лицо в кадре; для экрана глаза должны быть открыты. "
                                "Повторите C или K.",
                                file=sys.stderr,
                            )
                        else:
                            if calibration.target == "screen":
                                screen_calibrated = True
                                keyboard_calibrated = False
                                message = "Screen READY. Press K and look at the keyboard."
                            else:
                                keyboard_calibrated = True
                                message = "Keyboard READY. Now check the five event types."
                            print(
                                f"Калибровка {target_name} готова: "
                                f"{len(calibration.samples)} кадров.",
                                file=sys.stderr,
                            )
                            if args.profile is not None:
                                try:
                                    save_profile(analyzer, args.profile)
                                except OSError as error:
                                    message = (
                                        "Calibration READY, but profile SAVE FAILED. See terminal."
                                    )
                                    print(
                                        "Калибровка готова, но профиль не удалось сохранить: "
                                        f"{error}",
                                        file=sys.stderr,
                                    )
                                else:
                                    print(
                                        f"Профиль калибровки сохранён: {args.profile}",
                                        file=sys.stderr,
                                    )
                        finally:
                            analyzer.reset()
                            calibration = None

                preview = draw_overlay(
                    frame,
                    analyzer,
                    message,
                    calibration,
                    timestamp,
                    screen_calibrated,
                    keyboard_calibrated,
                    event_counts,
                )
                cv2.imshow(WINDOW_NAME, preview)
                command = key_command(cv2.waitKeyEx(1))
                if command == "quit" or (
                    cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
                ):
                    break
                if command == "screen":
                    analyzer.reset()
                    calibration = CalibrationCapture("screen", timestamp)
                    print("Калибровка экрана: смотрите в центр экрана.", file=sys.stderr)
                elif command == "keyboard":
                    if screen_calibrated:
                        analyzer.reset()
                        calibration = CalibrationCapture("keyboard", timestamp)
                        print("Калибровка клавиатуры: смотрите на клавиатуру.", file=sys.stderr)
                    else:
                        message = "Calibrate the screen first: press C."
                        print("Сначала откалибруйте экран клавишей C.", file=sys.stderr)
                elif command == "reset":
                    analyzer.reset()
                    calibration = None
                    message = "Timers reset. Calibration and event counts retained."
                    print(
                        "Таймеры сброшены. Сбор отменён; готовые профили сохранены.",
                        file=sys.stderr,
                    )
        finally:
            if calibration is not None:
                analyzer.reset()
                print("Незавершённый сбор калибровки отменён.", file=sys.stderr)
            print(
                "Проверка завершена. События: "
                + ", ".join(f"{name}={count}" for name, count in event_counts.items()),
                file=sys.stderr,
            )
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Остановлено пользователем.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
