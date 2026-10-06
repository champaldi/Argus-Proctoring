"""Regressions for transitions that can otherwise produce false gaze episodes.

Only measurements, camera detection and the clock are substitutes. Each test
exercises the analyzer's public per-frame API and externally visible events.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gaze import analyzer as module
from gaze.analyzer import AnalyzerConfig, FaceMetrics, GazeAnalyzer, HeadPose


def observation(pitch=0.0, yaw=0.0, iris_x=0.5, iris_y=0.5):
    return FaceMetrics(HeadPose(pitch, yaw, 0.0), iris_x, iris_y)


class OneFaceDetector:
    def process(self, rgb):
        return SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=[])])

    def close(self):
        pass


class GazeReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.analyzer = self.make_analyzer()
        self.measure_patch = patch.object(module, "measure_face", return_value=observation())
        self.measure = self.measure_patch.start()
        self.addCleanup(self.measure_patch.stop)

    def make_analyzer(self, smoothing_window=1):
        analyzer = GazeAnalyzer(
            AnalyzerConfig(smoothing_window=smoothing_window), face_mesh=OneFaceDetector()
        )
        self.addCleanup(analyzer.close)
        return analyzer

    def sample(self, timestamp, measured):
        self.measure.return_value = measured
        return self.analyzer.analyze(self.frame, timestamp=timestamp)

    def interval(self, start, stop, measured):
        events = []
        for timestamp in np.arange(start, stop + 0.01, 0.5):
            events.extend(self.sample(float(timestamp), measured))
        return events

    def test_reopening_after_long_iris_loss_requires_a_new_observed_episode(self):
        side = observation(iris_x=0.75)
        self.assertEqual(self.interval(0, 2.5, side), [])
        self.assertEqual(self.sample(3.0, observation(iris_x=None, iris_y=None)), [])
        # No intermediate frame arrived, but the 0.8 s iris loss still exceeds
        # the 0.3 s blink allowance and must not complete the old 3 s episode.
        self.assertEqual(self.sample(3.8, side), [])
        self.assertEqual(self.interval(4.3, 6.3, side), [])
        events = self.sample(6.8, side)
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 3.8)

    def test_reopening_after_short_blink_preserves_the_pending_episode(self):
        side = observation(iris_x=0.75)
        self.interval(0, 2.5, side)
        self.assertEqual(self.sample(3.0, observation(iris_x=None, iris_y=None)), [])
        events = self.sample(3.1, side)
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 0)

    def test_new_head_turn_cannot_confirm_an_old_iris_episode_after_long_loss(self):
        self.assertEqual(self.interval(0, 2.5, observation(iris_x=0.75)), [])
        self.assertEqual(self.sample(3.0, observation(iris_x=None, iris_y=None)), [])
        # The first head turn appears after an unobserved 0.8 s iris gap. Its
        # timer must begin now, rather than inherit the earlier eye-only time.
        head_side = observation(yaw=32)
        self.assertEqual(self.sample(3.8, head_side), [])
        self.assertEqual(self.interval(4.3, 6.3, head_side), [])
        events = self.sample(6.8, head_side)
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 3.8)

    def test_smoothing_cannot_extend_head_turn_past_a_return_to_screen(self):
        self.analyzer = self.make_analyzer(smoothing_window=3)
        side = observation(yaw=32)
        self.assertEqual(self.interval(0, 2.5, side), [])
        # The latest valid observation is neutral, even though the history's
        # median still contains a side turn. It cannot prove a 3 s violation.
        self.assertEqual(self.sample(3.0, observation()), [])
        self.assertEqual(self.interval(3.5, 6.0, side), [])
        events = self.sample(6.5, side)
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 3.5)

    def test_reopened_neutral_iris_is_not_replaced_by_a_preblink_side_median(self):
        self.analyzer = self.make_analyzer(smoothing_window=3)
        self.assertEqual(self.interval(0, 2.5, observation(iris_x=0.95)), [])
        self.assertEqual(self.sample(3.0, observation(iris_x=None, iris_y=None)), [])
        self.assertEqual(self.sample(3.1, observation()), [])
        self.assertEqual(self.interval(3.6, 6.1, observation()), [])

    def test_keyboard_head_position_does_not_become_a_violation_when_iris_is_lost(self):
        self.analyzer.calibrate([observation()] * 10)
        keyboard = observation(pitch=42, iris_x=0.75, iris_y=0.8)
        self.analyzer.calibrate([keyboard] * 10, target="keyboard")
        self.assertEqual(self.interval(0, 1, keyboard), [])
        hidden_eyes = observation(pitch=42, iris_x=None, iris_y=None)
        self.assertEqual(self.interval(1.5, 8, hidden_eyes), [])
        self.assertEqual(self.interval(8.5, 10, keyboard), [])

    def test_missing_iris_still_allows_a_head_turn_outside_keyboard_zone(self):
        self.analyzer.calibrate([observation()] * 10)
        self.analyzer.calibrate(
            [observation(pitch=42, iris_x=0.75, iris_y=0.8)] * 10, target="keyboard"
        )
        events = self.interval(0, 3, observation(pitch=42, yaw=32, iris_x=None, iris_y=None))
        self.assertIn("gaze_side", [event["type"] for event in events])

    def test_unconvertible_timestamp_cannot_leave_a_pending_violation(self):
        side = observation(yaw=32)
        self.assertEqual(self.interval(0, 2.5, side), [])
        with self.assertRaises((ValueError, TypeError)):
            self.analyzer.analyze(self.frame, timestamp="not-a-timestamp")
        self.assertEqual(self.analyzer.last_result.signals, ())
        self.assertEqual(self.interval(3.0, 5.5, side), [])
        events = self.sample(6.0, side)
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 3.0)

    def test_measurement_error_cannot_leave_a_pending_violation(self):
        side = observation(yaw=32)
        self.assertEqual(self.interval(0, 2.5, side), [])
        self.measure.side_effect = RuntimeError("landmark conversion failed")
        with self.assertRaisesRegex(RuntimeError, "landmark conversion failed"):
            self.analyzer.analyze(self.frame, timestamp=3.0)
        self.measure.side_effect = None
        self.assertEqual(self.analyzer.last_result.signals, ())
        self.assertEqual(self.interval(3.5, 6.0, side), [])
        events = self.sample(6.5, side)
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 3.5)

    def test_keyboard_calibration_accepts_finite_head_pose_without_irises(self):
        self.analyzer.calibrate([observation()] * 10)
        keyboard = observation(pitch=42, iris_x=None, iris_y=None)
        self.analyzer.calibrate([keyboard] * 10, target="keyboard")
        self.assertEqual(self.interval(0, 8, keyboard), [])
        self.assertNotIn("gaze_down", self.analyzer.last_result.signals)

    def test_keyboard_calibration_rejects_nonfinite_pose_or_optional_iris(self):
        self.analyzer.calibrate([observation()] * 10)
        invalid_samples = (
            observation(pitch=float("nan"), iris_x=None, iris_y=None),
            observation(yaw=float("inf"), iris_x=None, iris_y=None),
            FaceMetrics(HeadPose(42, 0, float("nan")), None, None),
            observation(pitch=42, iris_x=float("nan"), iris_y=None),
            observation(pitch=42, iris_x=None, iris_y=float("inf")),
        )
        for sample in invalid_samples:
            with self.subTest(sample=sample):
                with self.assertRaises(ValueError):
                    self.analyzer.calibrate([sample] * 10, target="keyboard")

    def test_screen_calibration_still_requires_both_iris_coordinates(self):
        for iris_x, iris_y in ((None, None), (None, 0.5), (0.5, None)):
            with self.subTest(iris=(iris_x, iris_y)):
                with self.assertRaises(ValueError):
                    self.analyzer.calibrate([observation(iris_x=iris_x, iris_y=iris_y)] * 10)

    def test_side_turn_outside_head_only_keyboard_zone_remains_detectable(self):
        self.analyzer.calibrate([observation()] * 10)
        self.analyzer.calibrate(
            [observation(pitch=42, iris_x=None, iris_y=None)] * 10, target="keyboard"
        )
        events = self.interval(0, 3, observation(pitch=42, yaw=32, iris_x=None, iris_y=None))
        self.assertEqual([event["type"] for event in events], ["gaze_side"])

    def test_proximity_detection_remains_active_in_head_only_keyboard_zone(self):
        self.analyzer.calibrate([observation()] * 10)
        keyboard = observation(pitch=42, iris_x=None, iris_y=None)
        self.analyzer.calibrate([keyboard] * 10, target="keyboard")
        with patch.object(module, "measure_face_width_ratio", return_value=0.6):
            events = self.interval(0, 3, keyboard)
        self.assertEqual([event["type"] for event in events], ["too_close_to_camera"])
        self.assertAlmostEqual(self.analyzer.get_face_width_ratio(), 0.6)

    def test_final_keyboard_blink_keeps_the_supported_iris_reference(self):
        self.analyzer.calibrate([observation()] * 10)
        keyboard = observation(pitch=42, iris_x=0.75, iris_y=0.8)
        blink = observation(pitch=42, iris_x=None, iris_y=None)
        self.analyzer.calibrate([keyboard] * 10 + [blink], target="keyboard")
        self.assertEqual(self.interval(0, 8, keyboard), [])
        # A head-only reference would incorrectly suppress this fresh iris
        # movement. Ten valid eye measurements must survive the final blink.
        events = self.interval(8.5, 11.5, observation(pitch=42, iris_x=0.95, iris_y=0.8))
        self.assertEqual([event["type"] for event in events], ["gaze_side"])

    def test_failed_detector_cleanup_clears_analyzer_diagnostics(self):
        class FailingCleanupDetector(OneFaceDetector):
            def close(self):
                raise RuntimeError("mesh cleanup failed")

        analyzer = GazeAnalyzer(face_mesh=FailingCleanupDetector())
        self.addCleanup(analyzer.close)
        with patch.object(module, "measure_face_width_ratio", return_value=0.6):
            analyzer.analyze(self.frame, timestamp=0)
        self.assertAlmostEqual(analyzer.get_face_width_ratio(), 0.6)
        with self.assertRaisesRegex(RuntimeError, "mesh cleanup failed"):
            analyzer.close()
        self.assertIsNone(analyzer.get_face_width_ratio())
        self.assertEqual(analyzer.last_result.signals, ())
        self.assertEqual(analyzer.last_result.events, [])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            analyzer.analyze(self.frame, timestamp=0.5)
        analyzer.close()  # Cleanup remains idempotent after the first failure.

    def test_failed_singleton_cleanup_allows_a_fresh_detector_next_time(self):
        class FailingCleanupDetector(OneFaceDetector):
            def close(self):
                raise RuntimeError("mesh cleanup failed")

        module.reset_default_analyzer()
        self.addCleanup(module.reset_default_analyzer)
        broken = GazeAnalyzer(face_mesh=FailingCleanupDetector())
        replacement = GazeAnalyzer(face_mesh=OneFaceDetector())
        self.addCleanup(broken.close)
        self.addCleanup(replacement.close)
        with patch.object(module, "GazeAnalyzer", side_effect=[broken, replacement]) as factory:
            module.analyze(self.frame)
            self.assertIsNotNone(module.get_last_result())
            with self.assertRaisesRegex(RuntimeError, "mesh cleanup failed"):
                module.reset_default_analyzer()
            self.assertIsNone(module.get_last_result())
            self.assertIsNone(module.get_face_width_ratio())
            module.analyze(self.frame)
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(module.get_last_result().face_count, 1)
            module.reset_default_analyzer()


if __name__ == "__main__":
    unittest.main()
