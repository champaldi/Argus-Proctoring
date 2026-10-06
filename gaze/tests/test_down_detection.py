"""Downward gaze remains observable when only head-pose estimation fails.

The eye landmarks are measured by production geometry. Only the independent
head solver and camera detector are substituted; no neutral angle is invented.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gaze import analyzer as module
from gaze.analyzer import AnalyzerConfig, FaceMetrics, GazeAnalyzer, HeadPose, measure_face


def eye_landmarks(iris_y=0.95, *, eye_height=20.0, left_iris_y=None):
    """Create two open 60-pixel-wide eyes in a 640 by 480 frame."""
    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    for center_x, indices, vertical_ratio in (
        (224.0, (33, 133, 159, 145, 468), iris_y),
        (416.0, (362, 263, 386, 374, 473), iris_y if left_iris_y is None else left_iris_y),
    ):
        a, b, top, bottom, iris = indices
        points = (
            (a, center_x - 30, 216),
            (b, center_x + 30, 216),
            (top, center_x, 216 - eye_height / 2),
            (bottom, center_x, 216 + eye_height / 2),
            (iris, center_x, 216 + eye_height * (vertical_ratio - 0.5)),
        )
        for index, x, y in points:
            landmarks[index] = SimpleNamespace(x=x / 640, y=y / 480, z=0.0)
    return landmarks


class LandmarkDetector:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def process(self, rgb):
        return SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=self.landmarks)])

    def close(self):
        pass


class DownDetectionTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector = LandmarkDetector(eye_landmarks())
        self.analyzer = GazeAnalyzer(AnalyzerConfig(smoothing_window=1), face_mesh=self.detector)
        self.addCleanup(self.analyzer.close)
        self.pose_patch = patch.object(module, "estimate_head_pose", return_value=None)
        self.pose = self.pose_patch.start()
        self.addCleanup(self.pose_patch.stop)

    def interval(self, start, stop):
        events = []
        for timestamp in np.arange(start, stop + 0.01, 0.5):
            events.extend(self.analyzer.analyze(self.frame, timestamp=float(timestamp)))
        return events

    def calibrate_profile(self, screen_pitch, keyboard_pitch, screen_iris=0.5, keyboard_iris=0.8):
        self.analyzer.calibrate([FaceMetrics(HeadPose(screen_pitch, 0, 0), 0.5, screen_iris)] * 10)
        self.analyzer.calibrate(
            [FaceMetrics(HeadPose(keyboard_pitch, 0, 0), 0.5, keyboard_iris)] * 10,
            target="keyboard",
        )

    def calibrate_recorded_profile(self):
        self.calibrate_profile(
            3.1452797175525693, 6.462476093106856, 0.41226878843306397, 0.40061765093947815
        )

    def test_reliable_irises_survive_a_rejected_head_pose(self):
        measured = measure_face(self.detector.landmarks, 640, 480)
        self.assertIsNotNone(measured)
        self.assertIsNone(measured.head_pose)
        self.assertAlmostEqual(measured.iris_x, 0.5)
        self.assertAlmostEqual(measured.iris_y, 0.95)

    def test_iris_only_down_emits_at_five_seconds_without_a_fabricated_head_pose(self):
        self.assertEqual(self.interval(0, 4.5), [])
        events = self.analyzer.analyze(self.frame, timestamp=5.0)
        self.assertEqual([event["type"] for event in events], ["gaze_down"])
        self.assertAlmostEqual(events[0]["started_at"], 0)
        self.assertNotIn("head_pose", events[0])
        self.assertEqual(self.interval(5.5, 8), [])

    def test_valid_head_pose_control_emits_the_same_down_event(self):
        self.pose.return_value = HeadPose(0, 0, 0)
        events = self.interval(0, 5)
        self.assertEqual([event["type"] for event in events], ["gaze_down"])
        self.assertEqual(events[0]["head_pose"], {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})

    def test_missing_pose_and_closed_eyes_are_unknown(self):
        self.detector.landmarks = eye_landmarks(eye_height=2)
        self.assertEqual(self.interval(0, 6), [])
        self.assertIsNone(self.analyzer.last_result.metrics)
        self.assertEqual(self.analyzer.last_result.signals, ())

    def test_missing_pose_and_disagreeing_eyes_are_unknown(self):
        self.detector.landmarks = eye_landmarks(iris_y=0.95, left_iris_y=0.4)
        self.assertEqual(self.interval(0, 6), [])
        self.assertIsNone(self.analyzer.last_result.metrics)

    def test_learned_iris_keyboard_profile_can_exempt_a_frame_without_pose(self):
        self.analyzer.calibrate([FaceMetrics(HeadPose(0, 0, 0), 0.5, 0.5)] * 10)
        self.analyzer.calibrate([FaceMetrics(HeadPose(42, 0, 0), 0.5, 0.9)] * 10, target="keyboard")
        self.assertEqual(self.interval(0, 8), [])

    def test_iris_outside_keyboard_profile_is_detected_even_without_pose(self):
        self.analyzer.calibrate([FaceMetrics(HeadPose(0, 0, 0), 0.5, 0.5)] * 10)
        self.analyzer.calibrate([FaceMetrics(HeadPose(42, 0, 0), 0.5, 0.7)] * 10, target="keyboard")
        events = self.interval(0, 5)
        self.assertEqual([event["type"] for event in events], ["gaze_down"])
        self.assertNotIn("head_pose", events[0])

    def test_head_only_keyboard_profile_cannot_exempt_an_unknown_head_pose(self):
        self.analyzer.calibrate([FaceMetrics(HeadPose(0, 0, 0), 0.5, 0.5)] * 10)
        self.analyzer.calibrate([FaceMetrics(HeadPose(42, 0, 0))] * 10, target="keyboard")
        events = self.interval(0, 5)
        self.assertEqual([event["type"] for event in events], ["gaze_down"])

    def test_recorded_profile_detects_held_moderate_down_when_irises_are_hidden(self):
        self.calibrate_recorded_profile()
        self.pose.return_value = HeadPose(28, 0, 0)
        self.detector.landmarks = eye_landmarks(eye_height=2)
        self.assertEqual(self.interval(0, 4.5), [])
        events = self.analyzer.analyze(self.frame, timestamp=5.0)
        self.assertEqual([event["type"] for event in events], ["gaze_down"])
        self.assertAlmostEqual(events[0]["duration"], 5.0)
        self.assertIsNone(events[0]["iris_y"])
        self.assertEqual(self.interval(5.5, 8), [])

    def test_recorded_keyboard_profile_stays_quiet_for_ten_seconds(self):
        self.calibrate_recorded_profile()
        self.pose.return_value = HeadPose(6.462476093106856, 0, 0)
        self.detector.landmarks = eye_landmarks(iris_y=0.40061765093947815)
        self.assertEqual(self.interval(0, 10), [])
        self.assertNotIn("gaze_down", self.analyzer.last_result.signals)

    def test_recorded_profile_neutral_and_short_downward_turn_stay_quiet(self):
        self.calibrate_recorded_profile()
        self.detector.landmarks = eye_landmarks(eye_height=2)
        self.pose.return_value = HeadPose(3.1452797175525693, 0, 0)
        self.assertEqual(self.interval(0, 6), [])
        self.pose.return_value = HeadPose(28, 0, 0)
        self.assertEqual(self.interval(6.5, 11), [])
        self.pose.return_value = HeadPose(3.1452797175525693, 0, 0)
        self.assertEqual(self.analyzer.analyze(self.frame, timestamp=11.5), [])
        self.assertNotIn("gaze_down", self.analyzer.last_result.elapsed)

    def test_deep_keyboard_profile_keeps_conservative_threshold_and_exemption(self):
        self.calibrate_profile(15, 42)
        self.detector.landmarks = eye_landmarks(eye_height=2)
        self.pose.return_value = HeadPose(42, 0, 0)
        self.assertEqual(self.interval(0, 10), [])
        # This smaller relative tilt is outside the keyboard zone but does not
        # satisfy the conservative 35-degree threshold of this camera profile.
        self.pose.return_value = HeadPose(45, 15, 0)
        self.assertEqual(self.interval(10.5, 16), [])
        self.pose.return_value = HeadPose(54, 0, 0)
        self.assertEqual(self.interval(16.5, 21), [])
        events = self.analyzer.analyze(self.frame, timestamp=21.5)
        self.assertEqual([event["type"] for event in events], ["gaze_down"])
        self.assertAlmostEqual(events[0]["started_at"], 16.5)

    def test_uncalibrated_stream_keeps_conservative_head_down_threshold(self):
        self.pose.return_value = HeadPose(28, 0, 0)
        self.detector.landmarks = eye_landmarks(eye_height=2)
        self.assertEqual(self.interval(0, 8), [])
        self.assertEqual(self.analyzer.get_diagnostics()["down_thresholds"]["head_degrees"], 35)

    def test_effective_down_thresholds_are_exposed_for_each_calibration_profile(self):
        self.calibrate_recorded_profile()
        limits = self.analyzer.get_diagnostics()["down_thresholds"]
        self.assertEqual(limits["head_degrees"], 20)
        self.assertEqual(limits["combined_degrees"], 20)
        self.assertEqual(limits["iris"], 0.3)
        self.assertEqual(limits["combined_iris"], 0.16)
        self.calibrate_profile(15, 42)
        limits = self.analyzer.get_diagnostics()["down_thresholds"]
        self.assertEqual(limits["head_degrees"], 35)
        self.assertEqual(limits["combined_degrees"], 25)

    def test_adaptive_threshold_configuration_rejects_nonpositive_or_nonfinite_values(self):
        self.assertEqual(AnalyzerConfig().calibrated_head_down_degrees, 20)
        self.assertEqual(AnalyzerConfig().keyboard_down_margin_degrees, 3)
        for name in ("calibrated_head_down_degrees", "keyboard_down_margin_degrees"):
            for value in (0, -1, float("nan"), float("inf"), True, "20"):
                with self.subTest(field=name, value=value):
                    with self.assertRaises(ValueError):
                        AnalyzerConfig(**{name: value})


if __name__ == "__main__":
    unittest.main()
