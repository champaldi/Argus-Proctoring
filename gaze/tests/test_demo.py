"""Manual-demo diagnostics checked without opening a camera or a native window."""

import io
import json
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

from gaze.analyzer import AnalysisResult, AnalyzerConfig, FaceMetrics, GazeAnalyzer, HeadPose
from gaze.demo import (
    CalibrationCapture,
    build_parser,
    calibration_capture_finished,
    format_diagnostics_lines,
    format_overlay_lines,
    key_command,
    load_profile,
    run,
    save_profile,
    select_events,
    update_event_counts,
    validate_output_paths,
    write_events,
    write_telemetry,
)


class DemoDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.config = AnalyzerConfig()
        self.result = AnalysisResult(1, None, (), [], {}, 0.45)

    def lines(self, **overrides):
        values = dict(
            result=self.result,
            config=self.config,
            timestamp=3.0,
            message="Press C to calibrate screen.",
            calibration=None,
            screen_calibrated=False,
            keyboard_calibrated=False,
            event_counts={},
        )
        values.update(overrides)
        return format_overlay_lines(**values)

    def test_width_and_threshold_use_percentage_and_strict_comparison(self):
        text = "\n".join(self.lines())
        self.assertIn("Face width: 45.0%", text)
        self.assertIn("alert >45.0% for 3.0s", text)
        missing = AnalysisResult(0, None, ("no_face",), [], {"no_face": 0.0})
        self.assertIn("Face width: unavailable", "\n".join(self.lines(result=missing)))

    def test_active_timers_include_zero_start_and_actual_configured_thresholds(self):
        config = AnalyzerConfig(gaze_side_seconds=7.0, face_too_close_seconds=4.0)
        result = AnalysisResult(
            1,
            None,
            ("gaze_side", "too_close_to_camera"),
            [],
            {"gaze_side": 0.0, "too_close_to_camera": 2.5},
            0.5,
        )
        text = "\n".join(self.lines(result=result, config=config))
        self.assertIn("gaze_side: 0.0/7.0s", text)
        self.assertIn("too_close_to_camera: 2.5/4.0s", text)
        self.assertNotIn("gaze_down: ", text)

    def test_counts_are_cumulative_for_all_five_events_and_do_not_mutate_input(self):
        original = {"gaze_side": 2}
        counts = update_event_counts(
            original, [{"type": "gaze_side"}, {"type": "no_face"}, {"type": "gaze_side"}]
        )
        self.assertEqual(original, {"gaze_side": 2})
        self.assertEqual(counts["gaze_side"], 4)
        self.assertEqual(counts["no_face"], 1)
        self.assertEqual(len(counts), 5)
        text = "\n".join(self.lines(event_counts=counts))
        for name, count in counts.items():
            self.assertIn(f"{name}={count}", text)

    def test_calibration_filters_gaze_but_preserves_presence_and_distance_events(self):
        events = [
            {"type": "gaze_down"},
            {"type": "gaze_side"},
            {"type": "no_face"},
            {"type": "multiple_faces"},
            {"type": "too_close_to_camera"},
        ]
        selected = select_events(events, calibrating=True)
        self.assertEqual(
            [event["type"] for event in selected],
            ["no_face", "multiple_faces", "too_close_to_camera"],
        )
        self.assertEqual(select_events(events, calibrating=False), events)
        self.assertEqual(len(events), 5)

    def test_calibration_waits_for_both_minimum_time_and_enough_samples(self):
        sample = FaceMetrics(HeadPose(0, 0, 0), 0.5, 0.5)
        capture = CalibrationCapture("screen", 10.0, [sample] * 10)
        self.assertFalse(calibration_capture_finished(capture, 11.99, min_samples=10))
        self.assertTrue(calibration_capture_finished(capture, 12.0, min_samples=10))
        capture.samples.pop()
        self.assertFalse(calibration_capture_finished(capture, 12.0, min_samples=10))
        self.assertFalse(calibration_capture_finished(capture, 17.99, min_samples=10))
        self.assertTrue(calibration_capture_finished(capture, 18.0, min_samples=10))

    def test_calibration_overlay_reports_profiles_and_slow_sample_collection(self):
        capture = CalibrationCapture("keyboard", 1.0)
        text = "\n".join(
            self.lines(
                timestamp=4.0,
                calibration=capture,
                screen_calibrated=True,
                keyboard_calibrated=False,
            )
        )
        self.assertIn("screen=READY", text)
        self.assertIn("keyboard=NOT SET", text)
        self.assertIn("keyboard: 3.0/8.0s", text)
        self.assertIn("samples=0/10", text)
        self.assertTrue(text.isascii())

    def test_parser_exposes_camera_and_keeps_sources_mutually_exclusive(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args([]).camera, 0)
        self.assertEqual(parser.parse_args(["--camera", "2"]).camera, 2)
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
            parser.parse_args(["--help"])
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            parser.parse_args(["--camera", "2", "--video", "test.mp4"])

    def test_event_output_remains_utf8_jsonl_without_status_messages(self):
        stdout = io.StringIO()
        output = io.StringIO()
        with redirect_stdout(stdout):
            write_events([{"type": "no_face", "comment": "студент ушёл"}], output)
        self.assertEqual(stdout.getvalue(), output.getvalue())
        self.assertIn("студент ушёл", output.getvalue())
        self.assertNotIn("\\u", output.getvalue())

    def test_key_commands_accept_english_and_russian_layouts_in_both_cases(self):
        for command, characters in (
            ("screen", "CcСс"),
            ("keyboard", "KkЛл"),
            ("reset", "RrКк"),
            ("quit", "QqЙй"),
        ):
            for character in characters:
                with self.subTest(character=character):
                    self.assertEqual(key_command(ord(character)), command)
        self.assertEqual(key_command(27), "quit")

    def test_unknown_keys_are_not_truncated_into_commands(self):
        for code in (-1, 0, 13, ord("x"), ord("я"), ord("c") + 0x10000):
            with self.subTest(code=code):
                self.assertIsNone(key_command(code))

    def test_down_diagnostics_show_real_deltas_profiles_and_configured_thresholds(self):
        diagnostics = {
            "raw_metrics": asdict(FaceMetrics(HeadPose(27, -4, 0), 0.5, 0.73)),
            "smoothed_metrics": asdict(FaceMetrics(HeadPose(25, -3, 0), 0.5, 0.69)),
            "screen": asdict(FaceMetrics(HeadPose(8, 1, 0), 0.5, 0.55)),
            "keyboard": asdict(FaceMetrics(HeadPose(25, -2, 0), 0.5, 0.72)),
            "keyboard_zone": True,
            "head_unknown": False,
            "iris_unknown": False,
        }
        config = AnalyzerConfig(
            head_down_degrees=30,
            combined_down_degrees=22,
            iris_down_threshold=0.28,
            combined_iris_down_threshold=0.11,
        )
        text = "\n".join(format_diagnostics_lines(diagnostics, config))
        self.assertIn("raw p/y=+27.0/-4.0", text)
        self.assertIn("smooth=+25.0/-3.0", text)
        self.assertIn("dp=+17.0", text)
        self.assertIn("dy=+0.14", text)
        self.assertIn("Keyboard p/y=+25.0/-2.0 iy=0.72 zone=YES", text)
        self.assertIn("dp>=30.0", text)
        self.assertIn("dy>=0.28", text)
        self.assertIn("dp>=22.0 AND dy>=0.11", text)
        self.assertTrue(text.isascii())

    def test_missing_head_and_iris_diagnostics_are_explicit(self):
        text = "\n".join(format_diagnostics_lines({}, self.config))
        self.assertIn("head=yes iris=yes", text)
        self.assertIn("raw p/y=?/?", text)
        self.assertIn("Keyboard: NOT SET", text)
        self.assertIn("Screen: DEFAULT", text)

    def test_down_diagnostics_use_effective_calibrated_thresholds(self):
        diagnostics = {
            "down_thresholds": {
                "head_degrees": 20.0,
                "combined_degrees": 20.0,
                "iris": 0.27,
                "combined_iris": 0.14,
            }
        }
        text = "\n".join(format_diagnostics_lines(diagnostics, self.config))
        self.assertIn("Down: dp>=20.0 OR dy>=0.27 OR (dp>=20.0 AND dy>=0.14)", text)
        self.assertNotIn("dp>=35.0", text)
        self.assertNotIn("dp>=25.0", text)

    def test_unknown_pose_preserves_iris_diagnostics_and_never_becomes_zero_angles(self):
        metrics = FaceMetrics(None, 0.5, 0.84)
        diagnostics = {
            "raw_metrics": asdict(metrics),
            "smoothed_metrics": asdict(metrics),
            "head_unknown": True,
            "iris_unknown": False,
            "deferred": ["gaze_down"],
            "episodes": {"gaze_down": {"started_at": 0.0, "emitted": False}},
        }
        text = "\n".join(format_diagnostics_lines(diagnostics, self.config, calibrating=True))
        self.assertIn("raw p/y=?/? | smooth=?/?", text)
        self.assertIn("raw=0.84 smooth=0.84", text)
        self.assertIn("dy=+0.34", text)
        self.assertIn("head=yes iris=no", text)
        self.assertIn("Down deferred=yes emitted=no", text)
        self.assertIn("COLLECTING PROFILE", text)
        result = AnalysisResult(1, metrics, ("gaze_down",), [], {"gaze_down": 5.0}, 0.4)
        self.assertIn("Head: unknown", "\n".join(self.lines(result=result)))
        output = io.StringIO()
        write_telemetry(output, result, diagnostics, timestamp=5.0, calibration=None)
        self.assertIsNone(json.loads(output.getvalue())["metrics"]["head_pose"])

    def test_telemetry_keeps_frame_measurements_without_images_or_stdout(self):
        metrics = FaceMetrics(HeadPose(18, 2, 0), 0.5, 0.7)
        result = AnalysisResult(1, metrics, ("gaze_down",), [], {"gaze_down": 1.2}, 0.4)
        diagnostics = {"screen": asdict(FaceMetrics(HeadPose(3, 0, 0), 0.5, 0.5))}
        output = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            write_telemetry(output, result, diagnostics, timestamp=3.0, calibration=None)
        record = json.loads(output.getvalue())
        self.assertEqual(record["metrics"]["head_pose"]["pitch"], 18)
        self.assertEqual(record["diagnostics"]["screen"]["head_pose"]["pitch"], 3)
        self.assertEqual(record["signals"], ["gaze_down"])
        self.assertEqual(record["elapsed"], {"gaze_down": 1.2})
        self.assertFalse(record["calibrating"])
        self.assertFalse(record["screen_calibrated"])
        self.assertFalse(record["keyboard_calibrated"])
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn("frame", record)
        self.assertNotIn("image", record)

    def test_telemetry_identifies_collection_and_existing_ready_profiles(self):
        output = io.StringIO()
        write_telemetry(
            output,
            self.result,
            {},
            timestamp=3.0,
            calibration=CalibrationCapture("keyboard", 1.0),
            screen_calibrated=True,
            keyboard_calibrated=False,
        )
        record = json.loads(output.getvalue())
        self.assertTrue(record["calibrating"])
        self.assertTrue(record["screen_calibrated"])
        self.assertFalse(record["keyboard_calibrated"])
        self.assertEqual(record["calibration"], {"target": "keyboard", "samples": 0})

    def test_telemetry_and_event_output_cannot_overwrite_each_other_or_video(self):
        validate_output_paths(None, Path("events.jsonl"), Path("trace.jsonl"))
        for video, events, telemetry in (
            (None, Path("trace.jsonl"), Path("trace.jsonl")),
            (Path("test.mp4"), None, Path("test.mp4")),
            (Path("test.mp4"), Path("test.mp4"), None),
        ):
            with self.subTest(video=video, events=events, telemetry=telemetry):
                with self.assertRaises(ValueError):
                    validate_output_paths(video, events, telemetry)
        for video, events, telemetry in (
            (Path("profile.json"), None, None),
            (None, Path("profile.json"), None),
            (None, None, Path("profile.json")),
        ):
            with self.subTest(video=video, events=events, telemetry=telemetry):
                with self.assertRaises(ValueError):
                    validate_output_paths(video, events, telemetry, Path("profile.json"))


class DemoProfileTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "profile.json"
        self.profile = {
            "version": 1,
            "screen": asdict(FaceMetrics(HeadPose(3, 1, 0), 0.5, 0.55)),
            "keyboard": asdict(FaceMetrics(HeadPose(7, 1, 0))),
        }
        for reference in (self.profile["screen"], self.profile["keyboard"]):
            reference.pop("iris_centers")
        self.analyzer = MagicMock()
        self.analyzer.export_calibration.return_value = self.profile

    def test_existing_profile_is_loaded_through_the_public_validation_api(self):
        self.path.write_text(json.dumps(self.profile), encoding="utf-8")
        self.assertTrue(load_profile(self.analyzer, self.path))
        self.analyzer.import_calibration.assert_called_once_with(self.profile)

    def test_missing_profile_does_not_apply_any_calibration(self):
        self.assertFalse(load_profile(self.analyzer, self.path))
        self.analyzer.import_calibration.assert_not_called()

    def test_invalid_json_and_rejected_profile_report_a_clear_load_error(self):
        self.path.write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Некорректный профиль калибровки"):
            load_profile(self.analyzer, self.path)
        self.analyzer.import_calibration.assert_not_called()
        self.path.write_text(json.dumps(self.profile), encoding="utf-8")
        self.analyzer.import_calibration.side_effect = ValueError("Invalid screen head pose")
        with self.assertRaisesRegex(ValueError, "Некорректный профиль калибровки"):
            load_profile(self.analyzer, self.path)

    def test_profile_replacement_persists_references_and_removes_temporary_file(self):
        self.path.write_text("old contents", encoding="utf-8")
        save_profile(self.analyzer, self.path)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), self.profile)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_failed_atomic_replacement_preserves_previous_profile(self):
        self.path.write_text("old contents", encoding="utf-8")
        with patch("gaze.demo.Path.replace", side_effect=OSError("File is read-only")):
            with self.assertRaises(OSError):
                save_profile(self.analyzer, self.path)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "old contents")
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_file_round_trip_uses_real_profile_validation_without_native_resources(self):
        self.path.write_text(json.dumps(self.profile), encoding="utf-8")
        with GazeAnalyzer(face_mesh=MagicMock()) as analyzer:
            self.assertTrue(load_profile(analyzer, self.path))
            self.assertEqual(analyzer.get_calibration_status(), {"screen": True, "keyboard": True})
            save_profile(analyzer, self.path)
            self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), self.profile)
            invalid = {"version": 1, "screen": None, "keyboard": self.profile["keyboard"]}
            self.path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Некорректный профиль калибровки"):
                load_profile(analyzer, self.path)
            self.assertEqual(analyzer.export_calibration(), self.profile)


class DemoResourceTests(unittest.TestCase):
    def test_camera_open_failure_releases_capture_without_starting_analyzer(self):
        capture = MagicMock()
        capture.isOpened.return_value = False
        with (
            patch("gaze.demo.cv2.VideoCapture", return_value=capture),
            patch("gaze.demo.cv2.destroyAllWindows") as destroy,
            patch("gaze.demo.GazeAnalyzer") as analyzer,
        ):
            with self.assertRaisesRegex(RuntimeError, "Не удалось открыть камеру №2"):
                run(build_parser().parse_args(["--camera", "2"]))
        capture.release.assert_called_once()
        destroy.assert_called_once()
        analyzer.assert_not_called()

    def fake_video(self, *, valid_metrics=True):
        """All native calls are substitutes, including capture and window creation."""
        stack = ExitStack()
        self.addCleanup(stack.close)
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 5.0
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        capture.read.side_effect = [(True, frame)] * 11 + [(False, None)]
        analyzer = MagicMock()
        analyzer.config = AnalyzerConfig()
        metrics = FaceMetrics(HeadPose(0, 0, 0), 0.5, 0.5) if valid_metrics else None
        analyzer.last_result = AnalysisResult(1, metrics, (), [], {}, 0.3)
        analyzer.analyze.return_value = []
        analyzer.get_diagnostics.return_value = {}
        analyzer.get_calibration_status.return_value = {"screen": False, "keyboard": False}
        analyzer.export_calibration.return_value = {"version": 1, "screen": None, "keyboard": None}
        analyzer.__enter__.return_value = analyzer
        stack.enter_context(patch("gaze.demo.cv2.VideoCapture", return_value=capture))
        stack.enter_context(patch("gaze.demo.GazeAnalyzer", return_value=analyzer))
        stack.enter_context(patch("gaze.demo.draw_overlay", return_value=frame))
        stack.enter_context(patch("gaze.demo.cv2.imshow"))
        stack.enter_context(patch("gaze.demo.cv2.getWindowProperty", return_value=1))
        stack.enter_context(patch("gaze.demo.cv2.destroyAllWindows"))
        keys = stack.enter_context(patch("gaze.demo.cv2.waitKeyEx"))
        return capture, analyzer, keys

    def test_successful_calibration_resets_timers_before_and_after_collection(self):
        capture, analyzer, keys = self.fake_video()
        keys.side_effect = [ord("c")] + [-1] * 9 + [ord("q")]
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(run(build_parser().parse_args(["--video", "test.mp4"])), 0)
        samples = analyzer.calibrate.call_args.args[0]
        self.assertEqual(len(samples), 10)
        self.assertEqual(analyzer.calibrate.call_args.kwargs, {"target": "screen"})
        self.assertEqual(analyzer.reset.call_count, 2)
        analyzer.__exit__.assert_called_once()
        capture.release.assert_called_once()

    def test_eof_cancels_collection_and_does_not_fabricate_absence_events(self):
        capture, analyzer, keys = self.fake_video(valid_metrics=False)
        keys.side_effect = [ord("c")] + [-1] * 10
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(stdout):
            self.assertEqual(run(build_parser().parse_args(["--video", "test.mp4"])), 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Незавершённый сбор калибровки отменён", stderr.getvalue())
        self.assertEqual(analyzer.reset.call_count, 2)
        analyzer.calibrate.assert_not_called()
        capture.release.assert_called_once()

    def test_calibration_timeout_reports_failure_and_resets_timers(self):
        capture, analyzer, keys = self.fake_video(valid_metrics=False)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        capture.read.side_effect = [(True, frame)] * 41
        keys.side_effect = [ord("c")] + [-1] * 39 + [ord("q")]
        analyzer.calibrate.side_effect = ValueError("Need at least 10 samples")
        stderr = io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            self.assertEqual(run(build_parser().parse_args(["--video", "test.mp4"])), 0)
        self.assertEqual(analyzer.reset.call_count, 2)
        self.assertIn("Калибровка экрана не удалась: собрано 0/10 кадров", stderr.getvalue())
        self.assertEqual(analyzer.calibrate.call_args.args[0], [])

    def test_keyboard_collection_uses_pose_samples_when_irises_are_not_visible(self):
        capture, analyzer, keys = self.fake_video()
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        capture.read.side_effect = [(True, frame)] * 21
        keys.side_effect = [ord("c")] + [-1] * 9 + [ord("k")] + [-1] * 9 + [ord("q")]
        frame_number = 0

        def sample_frame(*args, **kwargs):
            nonlocal frame_number
            frame_number += 1
            metrics = (
                FaceMetrics(HeadPose(0, 0, 0), 0.5, 0.5)
                if frame_number <= 11
                else FaceMetrics(HeadPose(35, 0, 0))
            )
            analyzer.last_result = AnalysisResult(1, metrics, (), [], {}, 0.3)
            return []

        analyzer.analyze.side_effect = sample_frame
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(run(build_parser().parse_args(["--video", "test.mp4"])), 0)
        self.assertEqual(analyzer.calibrate.call_count, 2)
        keyboard_call = analyzer.calibrate.call_args_list[1]
        self.assertEqual(keyboard_call.kwargs, {"target": "keyboard"})
        self.assertEqual(len(keyboard_call.args[0]), 10)
        self.assertTrue(all(sample.iris_y is None for sample in keyboard_call.args[0]))
        self.assertEqual(analyzer.reset.call_count, 4)

    def test_optional_telemetry_records_each_observed_frame_and_closes_file(self):
        _, _, keys = self.fake_video()
        keys.side_effect = [-1] * 10 + [ord("q")]
        with TemporaryDirectory() as directory:
            telemetry = Path(directory) / "trace.jsonl"
            args = build_parser().parse_args(["--video", "test.mp4", "--telemetry", str(telemetry)])
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)
            records = [
                json.loads(line) for line in telemetry.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(records), 11)
            self.assertEqual(records[0]["timestamp"], 0.0)
            self.assertEqual(records[-1]["timestamp"], 2.0)
            telemetry.unlink()  # Windows forbids this if the output is still open.

    def test_loaded_screen_profile_allows_keyboard_capture_and_saves_result(self):
        _, analyzer, keys = self.fake_video()
        keys.side_effect = [ord("k")] + [-1] * 9 + [ord("q")]
        screen = asdict(FaceMetrics(HeadPose(3, 0, 0), 0.5, 0.5))
        screen.pop("iris_centers")
        old_profile = {"version": 1, "screen": screen, "keyboard": None}
        new_profile = {"version": 1, "screen": screen, "keyboard": screen}
        analyzer.get_calibration_status.return_value = {"screen": True, "keyboard": False}
        analyzer.export_calibration.return_value = new_profile
        stderr = io.StringIO()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(old_profile), encoding="utf-8")
            args = build_parser().parse_args(["--video", "test.mp4", "--profile", str(path)])
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)
            analyzer.import_calibration.assert_called_once_with(old_profile)
            self.assertEqual(analyzer.calibrate.call_args.kwargs, {"target": "keyboard"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), new_profile)
            self.assertIn("Профиль калибровки загружен", stderr.getvalue())
            self.assertIn("Профиль калибровки сохранён", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
