"""Integration checks against the host application's actual shared events."""

from __future__ import annotations

import importlib
import sys
import unittest
from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import gaze
from gaze import analyzer as implementation


def _load_host_contract():
    try:
        events = importlib.import_module("events")
    except ModuleNotFoundError as exc:
        if exc.name != "events":
            raise
        application = Path(__file__).resolve().parents[2] / "parts" / "integration-ui"
        if not (application / "events.py").is_file():
            return None, None
        sys.path.insert(0, str(application))
        try:
            events = importlib.import_module("events")
            detectors = importlib.import_module("core.detectors")
        finally:
            sys.path.pop(0)
        return events, detectors

    try:
        detectors = importlib.import_module("core.detectors")
    except ModuleNotFoundError as exc:
        if exc.name not in {"core", "core.detectors"}:
            raise
        detectors = None
    return events, detectors


host_events, host_detectors = _load_host_contract()


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


class FakeMesh:
    def __init__(self, *, face_count: int = 1, width_ratio: float = 0.3) -> None:
        self.closed = False
        self.faces = []
        for _ in range(face_count):
            points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
            points[234].x = 0.5 - width_ratio / 2
            points[454].x = 0.5 + width_ratio / 2
            self.faces.append(SimpleNamespace(landmark=points))

    def process(self, image):
        return SimpleNamespace(multi_face_landmarks=self.faces)

    def close(self) -> None:
        self.closed = True


def metrics(*, pitch: float = 0, yaw: float = 0, iris_y: float = 0.5):
    return gaze.FaceMetrics(gaze.HeadPose(pitch, yaw, 0), 0.5, iris_y)


@unittest.skipIf(host_events is None, "Shared host events.py contract is missing")
class SharedContractTests(unittest.TestCase):
    def setUp(self) -> None:
        gaze.reset_default_analyzer()
        self.addCleanup(gaze.reset_default_analyzer)
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)

    def _session(self, *, face_count=1, width_ratio=0.3):
        mesh = FakeMesh(face_count=face_count, width_ratio=width_ratio)
        clock = FakeClock()
        analyzer = gaze.GazeAnalyzer(
            gaze.AnalyzerConfig(smoothing_window=1), face_mesh=mesh, clock=clock
        )
        implementation._default_analyzer = analyzer
        return analyzer, mesh, clock

    def _observe_until(self, clock, seconds, analyze=gaze.analyze):
        emitted = []
        start = clock.now
        for tick in range(round(seconds * 2) + 1):
            clock.now = start + tick / 2
            current = analyze(self.frame)
            if tick < seconds * 2:
                self.assertEqual(current, [], "An event was emitted before its threshold")
            emitted.extend(current)
        return emitted

    def test_all_five_events_use_the_host_class_enum_and_timing_details(self) -> None:
        cases = (
            ("gaze_down", 1, 0.3, metrics(pitch=40), 5.0),
            ("gaze_side", 1, 0.3, metrics(yaw=30), 3.0),
            ("no_face", 0, 0.3, metrics(), 2.0),
            ("multiple_faces", 2, 0.3, metrics(), 1.0),
            ("too_close_to_camera", 1, 0.5, metrics(), 3.0),
        )
        for name, count, ratio, measured, seconds in cases:
            with self.subTest(event=name):
                gaze.reset_default_analyzer()
                analyzer, _, clock = self._session(face_count=count, width_ratio=ratio)
                before = host_events.utc_now()
                with patch.object(implementation, "measure_face", return_value=measured):
                    emitted = self._observe_until(clock, seconds)
                    clock.now += 0.5
                    self.assertEqual(gaze.analyze(self.frame), [])
                self.assertEqual(len(emitted), 1)
                event = emitted[0]
                self.assertIs(type(event), host_events.ProctorEvent)
                self.assertIs(type(event.type), host_events.EventType)
                self.assertEqual(event.type, host_events.EventType(name))
                self.assertEqual(event.source, "gaze")
                self.assertEqual(event.occurred_at.utcoffset(), timedelta(0))
                self.assertIs(event.occurred_at.tzinfo, timezone.utc)
                self.assertLessEqual(before, event.occurred_at)
                self.assertLessEqual(event.occurred_at, host_events.utc_now())
                self.assertEqual(event.details["started_at"], 10.0)
                self.assertEqual(event.details["timestamp"], 10.0 + seconds)
                self.assertEqual(event.details["duration"], seconds)
                self.assertEqual(event.details["face_count"], count)
                if count == 1:
                    self.assertAlmostEqual(event.details["face_width_ratio"], ratio)
                    self.assertEqual(event.details["head_pose"]["yaw"], measured.head_pose.yaw)
                else:
                    self.assertNotIn("face_width_ratio", event.details)
                self.assertEqual(analyzer.last_result.face_count, count)

    @unittest.skipIf(host_detectors is None, "Host core.detectors adapter is missing")
    def test_host_adapter_preserves_event_identity_and_closes_shared_singleton(self) -> None:
        analyzer, mesh, clock = self._session(width_ratio=0.5)
        captured = []
        real_analyze = gaze.analyze

        def capture(frame):
            result = real_analyze(frame)
            captured.extend(result)
            return result

        adapter = host_detectors.DetectorAdapter(
            "gaze", "gaze", "analyze", "reset_default_analyzer"
        )
        with (
            patch.object(gaze, "analyze", capture),
            patch.object(implementation, "measure_face", return_value=metrics()),
        ):
            self.assertTrue(adapter.load().loaded)
            normalized = self._observe_until(clock, 3.0, adapter.analyze)
            self.assertEqual(len(normalized), 1)
            self.assertIs(normalized[0], captured[0])
            self.assertIs(type(normalized[0]), host_events.ProctorEvent)
            self.assertIs(implementation._default_analyzer, analyzer)
            adapter.close()
        self.assertTrue(mesh.closed)
        self.assertIsNone(implementation._default_analyzer)

    def test_helpers_are_lazy_before_the_first_frame(self) -> None:
        with patch.object(implementation, "GazeAnalyzer") as constructor:
            self.assertIsNone(gaze.get_last_result())
            self.assertIsNone(gaze.get_face_width_ratio())
            self.assertEqual(gaze.get_calibration_status(), {"screen": False, "keyboard": False})
            gaze.reset_default_timers()
            with self.assertRaisesRegex(RuntimeError, "before calibration"):
                gaze.calibrate_default_analyzer([metrics()] * 10)
            constructor.assert_not_called()

    def test_legacy_import_and_package_helpers_share_one_calibrated_instance(self) -> None:
        legacy = importlib.import_module("gaze_analyzer")
        self.assertIs(legacy, implementation)
        self.assertIs(gaze.GazeAnalyzer, legacy.GazeAnalyzer)
        analyzer, _, clock = self._session()
        screen = metrics()
        keyboard = metrics(pitch=40, iris_y=0.8)
        with patch.object(implementation, "GazeAnalyzer") as constructor:
            gaze.calibrate_default_analyzer([screen] * 10)
            legacy.calibrate_default_analyzer([keyboard] * 10, target="keyboard")
            self.assertEqual(gaze.get_calibration_status(), {"screen": True, "keyboard": True})
            with patch.object(implementation, "measure_face", return_value=keyboard):
                self.assertEqual(legacy.analyze(self.frame), [])
                self.assertIs(gaze.get_last_result(), analyzer.last_result)
                self.assertAlmostEqual(gaze.get_face_width_ratio(), 0.3)
                clock.now += 0.5
                self.assertEqual(gaze.analyze(self.frame), [])
            gaze.reset_default_timers()
            self.assertIs(gaze.get_last_result(), analyzer.last_result)
            self.assertIsNone(legacy.get_face_width_ratio())
            self.assertEqual(legacy.get_calibration_status(), {"screen": True, "keyboard": True})
            constructor.assert_not_called()

    def test_close_face_event_is_independent_of_keyboard_and_missing_pose(self) -> None:
        keyboard = metrics(pitch=40, iris_y=0.8)
        for measured in (keyboard, None):
            with self.subTest(valid_pose=measured is not None):
                gaze.reset_default_analyzer()
                _, _, clock = self._session(width_ratio=0.5)
                gaze.calibrate_default_analyzer([metrics()] * 10)
                gaze.calibrate_default_analyzer([keyboard] * 10, target="keyboard")
                with patch.object(implementation, "measure_face", return_value=measured):
                    emitted = self._observe_until(clock, 3.0)
                self.assertEqual(
                    [event.type for event in emitted],
                    [host_events.EventType.TOO_CLOSE_TO_CAMERA],
                )
                self.assertAlmostEqual(gaze.get_face_width_ratio(), 0.5)


if __name__ == "__main__":
    unittest.main()
