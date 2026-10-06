"""Face-size episode checks using normalized contours; no camera or model needed."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gaze import analyzer as module
from gaze.analyzer import (
    FACE_CONTOUR_IDS,
    FACE_TOO_CLOSE_RATIO,
    FACE_TOO_CLOSE_SECONDS,
    AnalyzerConfig,
    FaceMetrics,
    GazeAnalyzer,
    HeadPose,
)


def face_contour(left: float, right: float) -> list[SimpleNamespace]:
    """Provide a raw Face Mesh contour with a known horizontal image fraction."""
    center = (left + right) / 2
    landmarks = [SimpleNamespace(x=center, y=0.5, z=0.0) for _ in range(478)]
    landmarks[FACE_CONTOUR_IDS[0]].x = left
    landmarks[FACE_CONTOUR_IDS[1]].x = right
    return landmarks


class ContourDetector:
    def __init__(self):
        self.landmarks = face_contour(0.25, 0.75)
        self.face_count = 1
        self.calls = 0
        self.closed = False

    def process(self, frame):
        self.calls += 1
        faces = [SimpleNamespace(landmark=self.landmarks) for _ in range(self.face_count)]
        return SimpleNamespace(multi_face_landmarks=faces or None)

    def close(self):
        self.closed = True


class FaceDistanceTests(unittest.TestCase):
    def setUp(self):
        self.detector = ContourDetector()
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.analyzer = GazeAnalyzer(AnalyzerConfig(smoothing_window=1), face_mesh=self.detector)
        self.addCleanup(self.analyzer.close)
        self.measure_patch = patch.object(
            module, "measure_face", return_value=FaceMetrics(HeadPose(0, 0, 0), 0.5, 0.5)
        )
        self.measure = self.measure_patch.start()
        self.addCleanup(self.measure_patch.stop)

    def sample(self, timestamp, *, ratio=0.5, face_count=1):
        self.detector.landmarks = face_contour(0.0, ratio)
        self.detector.face_count = face_count
        return self.analyzer.analyze(self.frame, timestamp=timestamp)

    def interval(self, start, stop, *, ratio=0.5, face_count=1):
        events = []
        for timestamp in np.arange(start, stop + 0.01, 0.5):
            events.extend(self.sample(float(timestamp), ratio=ratio, face_count=face_count))
        return events

    def test_defaults_represent_image_fraction_and_three_seconds(self):
        self.assertEqual(FACE_TOO_CLOSE_RATIO, 0.45)
        self.assertEqual(FACE_TOO_CLOSE_SECONDS, 3.0)
        config = AnalyzerConfig()
        self.assertEqual(config.face_too_close_ratio, FACE_TOO_CLOSE_RATIO)
        self.assertEqual(config.face_too_close_seconds, FACE_TOO_CLOSE_SECONDS)

    def test_exact_width_boundary_does_not_create_close_event(self):
        events = self.interval(0, 8, ratio=FACE_TOO_CLOSE_RATIO)
        self.assertEqual(events, [])
        self.assertEqual(self.analyzer.get_face_width_ratio(), FACE_TOO_CLOSE_RATIO)
        self.assertNotIn("too_close_to_camera", self.analyzer.last_result.signals)

    def test_larger_face_requires_three_continuous_seconds_and_emits_once(self):
        self.assertEqual(self.interval(0, 2.5, ratio=0.46), [])
        events = self.sample(3, ratio=0.46)
        self.assertEqual([event["type"] for event in events], ["too_close_to_camera"])
        self.assertEqual(events[0]["face_count"], 1)
        self.assertAlmostEqual(events[0]["duration"], 3)
        self.assertAlmostEqual(events[0]["started_at"], 0)
        self.assertAlmostEqual(events[0]["timestamp"], 3)
        self.assertEqual(self.interval(3.5, 10, ratio=0.46), [])

    def test_brief_approach_and_return_does_not_create_an_event(self):
        self.assertEqual(self.interval(0, 2.5), [])
        self.assertEqual(self.interval(3, 8, ratio=0.3), [])

    def test_new_approach_after_normal_distance_rearms_the_event(self):
        first = self.interval(0, 3)
        self.assertEqual([event["type"] for event in first], ["too_close_to_camera"])
        self.assertEqual(self.sample(3.5, ratio=0.3), [])
        second = self.interval(4, 7)
        self.assertEqual([event["type"] for event in second], ["too_close_to_camera"])
        self.assertAlmostEqual(second[0]["started_at"], 4)

    def test_width_fraction_is_independent_of_frame_resolution(self):
        for timestamp, (width, height) in enumerate(((320, 240), (1920, 1080), (480, 640))):
            with self.subTest(size=(width, height)):
                self.frame = np.zeros((height, width, 3), dtype=np.uint8)
                self.sample(float(timestamp), ratio=0.6)
                self.assertAlmostEqual(self.analyzer.get_face_width_ratio(), 0.6)
                self.assertAlmostEqual(self.analyzer.last_result.face_width_ratio, 0.6)

    def test_iris_landmarks_do_not_expand_the_face_contour_width(self):
        self.detector.landmarks = face_contour(0.25, 0.55)
        self.detector.landmarks[468].x = 0.0
        self.detector.landmarks[473].x = 1.0
        self.analyzer.analyze(self.frame, timestamp=0)
        self.assertAlmostEqual(self.analyzer.get_face_width_ratio(), 0.3)

    def test_partial_horizontal_truncation_uses_visible_frame_intersection(self):
        self.detector.landmarks = face_contour(-0.2, 1.2)
        events = []
        for timestamp in np.arange(0, 3.01, 0.5):
            events.extend(self.analyzer.analyze(self.frame, timestamp=float(timestamp)))
        self.assertEqual(self.analyzer.get_face_width_ratio(), 1.0)
        self.assertEqual([event["type"] for event in events], ["too_close_to_camera"])

    def test_zero_width_or_wholly_off_frame_contour_is_unknown(self):
        for left, right in ((0.5, 0.5), (1.1, 1.8), (-0.8, -0.1)):
            with self.subTest(contour=(left, right)):
                self.analyzer.reset()
                self.detector.landmarks = face_contour(left, right)
                events = []
                for timestamp in np.arange(0, 4.01, 0.5):
                    events.extend(self.analyzer.analyze(self.frame, timestamp=float(timestamp)))
                self.assertIsNone(self.analyzer.get_face_width_ratio())
                self.assertNotIn("too_close_to_camera", self.analyzer.last_result.signals)
                self.assertEqual(events, [])

    def test_nonfinite_contour_and_missing_landmarks_are_unknown(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                self.analyzer.reset()
                self.detector.landmarks = face_contour(0.2, 0.8)
                self.detector.landmarks[FACE_CONTOUR_IDS[0]].x = invalid
                self.assertEqual(self.analyzer.analyze(self.frame, timestamp=0), [])
                self.assertIsNone(self.analyzer.get_face_width_ratio())
        self.analyzer.reset()
        self.detector.landmarks = []
        self.assertEqual(self.analyzer.analyze(self.frame, timestamp=0), [])
        self.assertIsNone(self.analyzer.get_face_width_ratio())

    def test_width_detection_does_not_require_a_solved_head_pose(self):
        self.measure.return_value = None
        events = self.interval(0, 3, ratio=0.5)
        self.assertEqual([event["type"] for event in events], ["too_close_to_camera"])
        self.assertIsNone(self.analyzer.last_result.metrics)
        self.assertAlmostEqual(self.analyzer.get_face_width_ratio(), 0.5)
        self.assertNotIn("no_face", self.analyzer.last_result.signals)

    def test_no_face_or_multiple_faces_interrupt_the_close_episode(self):
        for face_count in (0, 2):
            with self.subTest(face_count=face_count):
                self.analyzer.reset()
                self.assertEqual(self.interval(0, 2.5), [])
                self.sample(3, face_count=face_count)
                self.assertIsNone(self.analyzer.get_face_width_ratio())
                self.assertNotIn("too_close_to_camera", self.analyzer.last_result.signals)
                self.assertEqual(self.interval(3.5, 6), [])
                events = self.sample(6.5)
                self.assertEqual([event["type"] for event in events], ["too_close_to_camera"])
                self.assertAlmostEqual(events[0]["started_at"], 3.5)

    def test_unknown_width_interrupts_the_pending_close_episode(self):
        self.assertEqual(self.interval(0, 2.5), [])
        self.detector.landmarks = face_contour(0.5, 0.5)
        self.assertEqual(self.analyzer.analyze(self.frame, timestamp=3), [])
        self.assertEqual(self.interval(3.5, 6), [])
        events = self.sample(6.5)
        self.assertEqual([event["type"] for event in events], ["too_close_to_camera"])
        self.assertAlmostEqual(events[0]["started_at"], 3.5)

    def test_long_sample_gap_starts_a_new_close_episode(self):
        self.assertEqual(self.interval(0, 2.5), [])
        self.assertEqual(self.interval(5, 7.5), [])
        events = self.sample(8)
        self.assertEqual([event["type"] for event in events], ["too_close_to_camera"])
        self.assertAlmostEqual(events[0]["started_at"], 5)

    def test_reset_discards_width_diagnostic_and_pending_episode(self):
        self.interval(0, 2.5)
        self.analyzer.reset()
        self.assertIsNone(self.analyzer.get_face_width_ratio())
        self.assertIsNone(self.analyzer.last_result.face_width_ratio)
        self.assertEqual(self.interval(0, 2.5), [])
        self.assertEqual(len(self.sample(3)), 1)

    def test_invalid_frame_discards_width_diagnostic_and_pending_episode(self):
        self.interval(0, 2.5)
        with self.assertRaises(ValueError):
            self.analyzer.analyze(None, timestamp=3)
        self.assertIsNone(self.analyzer.get_face_width_ratio())
        self.assertEqual(self.interval(3.5, 6), [])
        self.assertEqual(len(self.sample(6.5)), 1)

    def test_detector_error_discards_width_diagnostic_and_pending_episode(self):
        self.interval(0, 2.5)
        with patch.object(self.detector, "process", side_effect=RuntimeError("inference failed")):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                self.analyzer.analyze(self.frame, timestamp=3)
        self.assertIsNone(self.analyzer.get_face_width_ratio())
        self.assertEqual(self.interval(3.5, 6), [])
        self.assertEqual(len(self.sample(6.5)), 1)

    def test_ratio_getters_do_not_start_the_detector(self):
        self.assertIsNone(self.analyzer.get_face_width_ratio())
        self.assertEqual(self.detector.calls, 0)
        module.reset_default_analyzer()
        with patch.object(module, "GazeAnalyzer", side_effect=AssertionError("unexpected model")):
            self.assertIsNone(module.get_face_width_ratio())

    def test_distance_duration_and_ratio_can_be_tuned(self):
        self.analyzer.close()
        self.detector = ContourDetector()
        self.analyzer = GazeAnalyzer(
            AnalyzerConfig(
                smoothing_window=1, face_too_close_ratio=0.6, face_too_close_seconds=1.0
            ),
            face_mesh=self.detector,
        )
        self.addCleanup(self.analyzer.close)
        self.assertEqual(self.interval(0, 2, ratio=0.55), [])
        events = self.interval(2.5, 3.5, ratio=0.65)
        self.assertEqual([event["type"] for event in events], ["too_close_to_camera"])


if __name__ == "__main__":
    unittest.main()
