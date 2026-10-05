"""Deterministic behavior checks for the proctoring face and gaze analyzer.

The detector and clock are substitutes; image geometry is tested independently
with projected landmarks. No camera, model download, or network is required.
"""

import math
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import gaze_analyzer
from gaze_analyzer import (
    HEAD_LANDMARK_IDS,
    HEAD_MODEL_POINTS,
    AnalyzerConfig,
    FaceMetrics,
    GazeAnalyzer,
    HeadPose,
    estimate_head_pose,
    measure_face,
)


class FakeFaceMesh:
    def __init__(self, face_count=1):
        self.face_count = face_count
        self.frames = []
        self.closed = False
        self.close_calls = 0

    def process(self, rgb):
        self.frames.append(rgb.copy())
        faces = [SimpleNamespace(landmark=[]) for _ in range(self.face_count)]
        return SimpleNamespace(multi_face_landmarks=faces or None)

    def close(self):
        self.closed = True
        self.close_calls += 1


class ManualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def metrics(pitch=0.0, yaw=0.0, roll=0.0, iris_x=0.5, iris_y=0.5):
    return FaceMetrics(HeadPose(pitch, yaw, roll), iris_x, iris_y)


class AnalyzerBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.mesh = FakeFaceMesh()
        self.clock = ManualClock()
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.config = AnalyzerConfig(smoothing_window=1)
        self.analyzer = GazeAnalyzer(self.config, face_mesh=self.mesh, clock=self.clock)
        self.metric_patch = patch("gaze_analyzer.measure_face", return_value=metrics())
        self.measure = self.metric_patch.start()
        self.addCleanup(self.metric_patch.stop)
        self.addCleanup(self.analyzer.close)

    def sample(self, timestamp, metric=None, face_count=1):
        self.clock.now = timestamp
        self.mesh.face_count = face_count
        self.measure.return_value = metrics() if metric is None else metric
        return self.analyzer.analyze(self.frame)

    def interval(self, start, stop, metric=None, face_count=1):
        result = []
        for timestamp in np.arange(start, stop + 0.01, 0.5):
            result.extend(self.sample(float(timestamp), metric, face_count))
        return result

    def test_default_durations_and_config_are_immutable(self):
        config = AnalyzerConfig()
        self.assertEqual(config.gaze_side_seconds, 3)
        self.assertEqual(config.gaze_down_seconds, 5)
        self.assertEqual(config.no_face_seconds, 2)
        self.assertEqual(config.multiple_faces_seconds, 1)
        with self.assertRaises(FrozenInstanceError):
            config.gaze_side_seconds = 0

    def test_screen_gaze_has_no_events(self):
        self.assertEqual(self.interval(0, 7), [])
        self.assertEqual(self.analyzer.last_result.face_count, 1)
        self.assertEqual(self.analyzer.last_result.signals, ())

    def test_side_turn_emits_once_at_continuous_threshold(self):
        side = metrics(yaw=32)
        self.assertEqual(self.interval(0, 2.5, side), [])
        events = self.sample(3, side)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "gaze_side")
        self.assertEqual(events[0]["face_count"], 1)
        self.assertAlmostEqual(events[0]["duration"], 3)
        self.assertAlmostEqual(events[0]["timestamp"], 3)
        self.assertAlmostEqual(events[0]["started_at"], 0)
        self.assertEqual(self.interval(3.5, 8, side), [])

    def test_side_turn_in_each_direction_is_detected(self):
        for yaw in (-32, 32):
            with self.subTest(yaw=yaw):
                self.analyzer.reset()
                events = self.interval(0, 3, metrics(yaw=yaw))
                self.assertEqual([event["type"] for event in events], ["gaze_side"])

    def test_return_to_screen_restarts_side_timer(self):
        side = metrics(yaw=32)
        self.assertEqual(self.interval(0, 2.5, side), [])
        self.sample(3, metrics())
        self.assertEqual(self.interval(3.5, 6, side), [])
        events = self.sample(6.5, side)
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 3.5)

    def test_second_side_episode_produces_a_new_event(self):
        side = metrics(yaw=32)
        self.assertEqual(len(self.interval(0, 4, side)), 1)
        self.sample(4.5)
        self.assertEqual(len(self.interval(5, 8, side)), 1)

    def test_long_downward_head_turn_produces_down_event(self):
        down = metrics(pitch=42)
        self.assertEqual(self.interval(0, 4.5, down), [])
        events = self.sample(5, down)
        self.assertEqual([event["type"] for event in events], ["gaze_down"])

    def test_brief_keyboard_glance_does_not_create_event(self):
        self.assertEqual(self.interval(0, 2, metrics(pitch=42, iris_y=0.8)), [])
        self.assertEqual(self.interval(2.5, 8), [])

    def test_moderate_keyboard_angle_has_no_down_signal(self):
        self.assertEqual(self.interval(0, 8, metrics(pitch=20, iris_y=0.6)), [])
        self.assertNotIn("gaze_down", self.analyzer.last_result.signals)

    def test_iris_side_motion_can_trigger_event_without_head_turn(self):
        for iris_x in (0.25, 0.75):
            with self.subTest(iris_x=iris_x):
                self.analyzer.reset()
                events = self.interval(0, 3, metrics(iris_x=iris_x))
                self.assertEqual([event["type"] for event in events], ["gaze_side"])

    def test_iris_down_motion_can_trigger_event_without_head_turn(self):
        events = self.interval(0, 5, metrics(iris_y=0.85))
        self.assertEqual([event["type"] for event in events], ["gaze_down"])

    def test_combined_head_and_iris_motion_can_trigger_down_event(self):
        events = self.interval(0, 5, metrics(pitch=29, iris_y=0.7))
        self.assertEqual([event["type"] for event in events], ["gaze_down"])

    def test_blink_missing_iris_does_not_trigger_iris_event(self):
        self.assertEqual(self.interval(0, 8, metrics(iris_x=None, iris_y=None)), [])

    def test_no_face_event_is_delayed_and_emitted_once(self):
        self.assertEqual(self.interval(0, 1.5, face_count=0), [])
        events = self.sample(2, face_count=0)
        self.assertEqual([event["type"] for event in events], ["no_face"])
        self.assertEqual(events[0]["face_count"], 0)
        self.assertEqual(self.interval(2.5, 6, face_count=0), [])

    def test_multiple_faces_event_is_delayed_and_emitted_once(self):
        self.assertEqual(self.interval(0, 0.5, face_count=2), [])
        events = self.sample(1, face_count=2)
        self.assertEqual([event["type"] for event in events], ["multiple_faces"])
        self.assertEqual(events[0]["face_count"], 2)
        self.assertEqual(self.interval(1.5, 5, face_count=3), [])

    def test_zero_or_multiple_faces_interrupt_gaze_episodes(self):
        for face_count in (0, 2):
            with self.subTest(face_count=face_count):
                self.analyzer.reset()
                side = metrics(yaw=32, pitch=42)
                self.assertEqual(self.interval(0, 2.5, side), [])
                self.sample(3, side, face_count)
                expected = "no_face" if face_count == 0 else "multiple_faces"
                self.assertEqual(self.analyzer.last_result.signals, (expected,))
                self.assertEqual(self.interval(3.5, 6, side), [])
                events = self.sample(6.5, side)
                self.assertEqual([event["type"] for event in events], ["gaze_side"])

    def test_switching_from_absence_to_multiple_faces_restarts_timing(self):
        self.assertEqual(self.interval(0, 1.5, face_count=0), [])
        self.assertEqual(self.sample(2, face_count=2), [])
        self.assertEqual(self.sample(2.5, face_count=2), [])
        events = self.sample(3, face_count=2)
        self.assertEqual([event["type"] for event in events], ["multiple_faces"])
        self.assertAlmostEqual(events[0]["started_at"], 2)

    def test_returning_single_face_with_closed_eyes_clears_presence_signals(self):
        for face_count, presence_signal in ((0, "no_face"), (2, "multiple_faces")):
            with self.subTest(face_count=face_count):
                self.analyzer.reset()
                events = self.interval(0, 2, face_count=face_count)
                self.assertEqual([event["type"] for event in events], [presence_signal])
                blink = metrics(iris_x=None, iris_y=None)
                self.assertEqual(self.sample(2.5, blink), [])
                self.assertEqual(self.analyzer.last_result.face_count, 1)
                self.assertEqual(self.analyzer.last_result.signals, ())
                self.assertEqual(self.interval(3, 6, blink), [])

    def test_long_sample_gap_cannot_count_as_continuous_gaze(self):
        side = metrics(yaw=32)
        self.assertEqual(self.interval(0, 1, side), [])
        self.assertEqual(self.interval(4, 6.5, side), [])
        events = self.sample(7, side)
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 4)

    def test_screen_calibration_offsets_camera_position(self):
        baseline = metrics(pitch=15, yaw=20, iris_x=0.65, iris_y=0.7)
        self.analyzer.calibrate([baseline] * 10)
        self.assertEqual(self.interval(0, 8, baseline), [])
        events = self.interval(8.5, 11.5, metrics(pitch=15, yaw=52, iris_x=0.65, iris_y=0.7))
        self.assertEqual([event["type"] for event in events], ["gaze_side"])

    def test_screen_calibration_uses_medians_for_outliers(self):
        baseline = metrics(pitch=4, yaw=12, iris_x=0.55, iris_y=0.6)
        outlier = metrics(pitch=85, yaw=85, iris_x=0.99, iris_y=0.99)
        self.analyzer.calibrate([baseline] * 9 + [outlier])
        self.assertEqual(self.interval(0, 8, baseline), [])

    def test_keyboard_calibration_exempts_the_learned_downward_zone(self):
        self.analyzer.calibrate([metrics()] * 10)
        keyboard = metrics(pitch=42, iris_y=0.8)
        self.analyzer.calibrate([keyboard] * 10, target="keyboard")
        self.assertEqual(self.interval(0, 8, keyboard), [])
        self.assertNotIn("gaze_down", self.analyzer.last_result.signals)

    def test_side_turn_still_detected_at_calibrated_keyboard_pitch(self):
        self.analyzer.calibrate([metrics()] * 10)
        self.analyzer.calibrate([metrics(pitch=42, iris_y=0.8)] * 10, target="keyboard")
        events = self.interval(0, 3, metrics(pitch=42, yaw=32, iris_y=0.8))
        self.assertIn("gaze_side", [event["type"] for event in events])

    def test_keyboard_calibration_can_include_a_sideways_iris_offset(self):
        self.analyzer.calibrate([metrics()] * 10)
        keyboard = metrics(pitch=42, iris_x=0.75, iris_y=0.8)
        self.analyzer.calibrate([keyboard] * 10, target="keyboard")
        self.assertEqual(self.interval(0, 10, keyboard), [])
        self.assertEqual(self.analyzer.last_result.signals, ())

    def test_iris_leaving_learned_keyboard_zone_restores_side_detection(self):
        self.analyzer.calibrate([metrics()] * 10)
        self.analyzer.calibrate(
            [metrics(pitch=42, iris_x=0.75, iris_y=0.8)] * 10, target="keyboard"
        )
        events = self.interval(0, 3, metrics(pitch=42, iris_x=0.95, iris_y=0.8))
        self.assertIn("gaze_side", [event["type"] for event in events])

    def test_keyboard_requires_screen_calibration_first(self):
        with self.assertRaises(ValueError):
            self.analyzer.calibrate([metrics(pitch=42)] * 10, target="keyboard")

    def test_calibration_rejects_insufficient_or_nonfinite_samples(self):
        with self.assertRaises(ValueError):
            self.analyzer.calibrate([metrics()] * 9)
        with self.assertRaises(ValueError):
            self.analyzer.calibrate([metrics(yaw=float("nan"))] * 10)
        with self.assertRaises(ValueError):
            self.analyzer.calibrate([metrics()] * 10, target="desk")

    def test_reset_clears_timers_and_retains_calibration(self):
        baseline = metrics(pitch=15, yaw=20, iris_x=0.65, iris_y=0.7)
        self.analyzer.calibrate([baseline] * 10)
        self.interval(0, 2.5, metrics(yaw=60))
        self.analyzer.reset()
        self.assertEqual(self.interval(0, 8, baseline), [])

    def test_explicit_timestamp_overrides_the_clock(self):
        self.clock.now = 1000
        self.measure.return_value = metrics(yaw=32)
        events = []
        for timestamp in np.arange(0, 3.01, 0.5):
            events.extend(self.analyzer.analyze(self.frame, timestamp=float(timestamp)))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["timestamp"], 3)

    def test_bgr_frame_is_converted_to_rgb_without_mutating_input(self):
        self.frame[:] = [10, 20, 30]
        original = self.frame.copy()
        self.sample(0)
        np.testing.assert_array_equal(self.mesh.frames[0][0, 0], [30, 20, 10])
        np.testing.assert_array_equal(self.frame, original)

    def test_invalid_frames_are_rejected_before_detector_call(self):
        bad_frames = [
            None,
            np.zeros((0, 64, 3), dtype=np.uint8),
            np.zeros((48, 64), dtype=np.uint8),
            np.zeros((48, 64, 4), dtype=np.uint8),
            np.zeros((48, 64, 3), dtype=np.float32),
        ]
        for frame in bad_frames:
            with self.subTest(shape=getattr(frame, "shape", None)):
                with self.assertRaises((ValueError, TypeError)):
                    self.analyzer.analyze(frame, timestamp=0)
        self.assertEqual(self.mesh.frames, [])

    def test_invalid_frame_clears_pending_episode_and_diagnostics(self):
        self.interval(0, 2.5, metrics(yaw=32))
        with self.assertRaises(ValueError):
            self.analyzer.analyze(None, timestamp=3)
        self.assertEqual(self.analyzer.last_result.signals, ())
        self.assertEqual(self.analyzer.last_result.events, [])
        self.assertEqual(self.analyzer.last_result.elapsed, {})
        self.assertEqual(self.interval(3.5, 6, metrics(yaw=32)), [])
        events = self.sample(6.5, metrics(yaw=32))
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 3.5)

    def test_nonfinite_and_backwards_timestamps_are_rejected(self):
        for timestamp in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(ValueError):
                    self.analyzer.analyze(self.frame, timestamp=timestamp)
        self.sample(2)
        with self.assertRaises(ValueError):
            self.analyzer.analyze(self.frame, timestamp=1)

    def test_context_manager_closes_detector(self):
        mesh = FakeFaceMesh()
        with GazeAnalyzer(self.config, face_mesh=mesh) as analyzer:
            analyzer.analyze(self.frame, timestamp=0)
        self.assertTrue(mesh.closed)

    def test_closed_analyzer_rejects_frames_and_close_is_idempotent(self):
        self.analyzer.close()
        self.analyzer.close()
        self.assertEqual(self.mesh.close_calls, 1)
        with self.assertRaises(RuntimeError):
            self.analyzer.analyze(self.frame, timestamp=0)

    def test_duplicate_timestamps_do_not_advance_timer(self):
        self.assertEqual(self.sample(0, metrics(yaw=32)), [])
        for _ in range(20):
            self.assertEqual(self.sample(0, metrics(yaw=32)), [])
        self.assertEqual(len(self.interval(0.5, 3, metrics(yaw=32))), 1)

    def test_invalid_geometry_is_unknown_and_interrupts_gaze_timer(self):
        self.interval(0, 2.5, metrics(yaw=32))
        self.measure.return_value = None
        self.assertEqual(self.analyzer.analyze(self.frame, timestamp=3), [])
        self.assertEqual(self.analyzer.last_result.face_count, 1)
        self.assertEqual(self.analyzer.last_result.signals, ())
        self.assertEqual(self.interval(3.5, 6, metrics(yaw=32)), [])
        self.assertEqual(len(self.sample(6.5, metrics(yaw=32))), 1)

    def test_detector_failure_propagates_and_resets_pending_timer(self):
        self.interval(0, 2.5, metrics(yaw=32))
        with patch.object(self.mesh, "process", side_effect=RuntimeError("detector failed")):
            with self.assertRaisesRegex(RuntimeError, "detector failed"):
                self.analyzer.analyze(self.frame, timestamp=3)
        self.assertEqual(self.interval(3.5, 6, metrics(yaw=32)), [])
        events = self.sample(6.5, metrics(yaw=32))
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 3.5)

    def test_smoothing_suppresses_isolated_iris_and_head_spikes(self):
        self.analyzer.close()
        self.mesh = FakeFaceMesh()
        self.analyzer = GazeAnalyzer(
            AnalyzerConfig(smoothing_window=3), face_mesh=self.mesh, clock=self.clock
        )
        self.addCleanup(self.analyzer.close)
        self.interval(0, 1)
        self.sample(1.5, metrics(yaw=70, iris_x=0.99))
        self.assertEqual(self.analyzer.last_result.signals, ())
        self.assertEqual(self.interval(2, 7), [])

    def test_smoothing_does_not_reuse_iris_position_during_blink(self):
        self.analyzer.close()
        self.mesh = FakeFaceMesh()
        self.analyzer = GazeAnalyzer(
            AnalyzerConfig(smoothing_window=3), face_mesh=self.mesh, clock=self.clock
        )
        self.addCleanup(self.analyzer.close)
        self.interval(0, 1, metrics(iris_x=0.75))
        self.assertEqual(self.sample(1.5, metrics(iris_x=None, iris_y=None)), [])
        self.assertIsNone(self.analyzer.last_result.metrics.iris_x)

    def test_short_regular_blinks_preserve_a_continuous_iris_episode(self):
        self.analyzer.close()
        self.mesh = FakeFaceMesh()
        self.analyzer = GazeAnalyzer(
            AnalyzerConfig(smoothing_window=1, iris_missing_grace_seconds=0.3),
            face_mesh=self.mesh,
            clock=self.clock,
        )
        self.addCleanup(self.analyzer.close)
        timestamps = sorted(set(np.arange(0, 10.01, 0.5)) | {2.6, 5.1, 7.6})
        events = []
        for timestamp in timestamps:
            blink = timestamp in (2.5, 5.0, 7.5)
            result = self.sample(
                float(timestamp),
                metrics(iris_x=None, iris_y=None) if blink else metrics(iris_x=0.75),
            )
            if blink:
                self.assertEqual(result, [])
            events.extend(result)
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 0)

    def test_iris_event_due_during_blink_waits_until_eyes_reopen(self):
        self.interval(0, 2.5, metrics(iris_x=0.75))
        self.assertEqual(self.sample(3, metrics(iris_x=None, iris_y=None)), [])
        events = self.sample(3.1, metrics(iris_x=0.75))
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 0)

    def test_prolonged_missing_iris_resets_the_iris_episode(self):
        self.interval(0, 2.5, metrics(iris_x=0.75))
        self.assertEqual(self.sample(3, metrics(iris_x=None, iris_y=None)), [])
        self.assertEqual(self.sample(3.4, metrics(iris_x=None, iris_y=None)), [])
        self.assertEqual(self.interval(3.5, 6, metrics(iris_x=0.75)), [])
        events = self.sample(6.5, metrics(iris_x=0.75))
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 3.5)

    def test_missing_iris_cannot_start_an_iris_episode(self):
        self.assertEqual(self.interval(0, 8, metrics(iris_x=None, iris_y=None)), [])
        self.assertEqual(self.interval(8.5, 11, metrics(iris_x=0.75)), [])
        events = self.sample(11.5, metrics(iris_x=0.75))
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 8.5)

    def test_blinks_do_not_interrupt_head_driven_episodes(self):
        side = metrics(yaw=32, iris_x=None, iris_y=None)
        events = self.interval(0, 3, side)
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        down = metrics(pitch=42, iris_x=None, iris_y=None)
        self.sample(3.5)
        events = self.interval(4, 9, down)
        self.assertEqual([event["type"] for event in events], ["gaze_down"])

    def test_screen_recalibration_invalidates_previous_keyboard_zone(self):
        self.analyzer.calibrate([metrics()] * 10)
        self.analyzer.calibrate([metrics(pitch=42, iris_y=0.8)] * 10, target="keyboard")
        self.analyzer.calibrate([metrics()] * 10)
        events = self.interval(0, 5, metrics(pitch=42, iris_y=0.8))
        self.assertEqual([event["type"] for event in events], ["gaze_down"])

    def test_convenience_function_keeps_one_session_and_releases_it(self):
        gaze_analyzer.reset_default_analyzer()
        with patch("gaze_analyzer.GazeAnalyzer", return_value=self.analyzer) as constructor:
            self.measure.return_value = metrics(yaw=32)
            events = []
            for timestamp in np.arange(0, 3.01, 0.5):
                self.clock.now = float(timestamp)
                events.extend(gaze_analyzer.analyze(self.frame))
            self.assertEqual([event["type"] for event in events], ["gaze_side"])
            constructor.assert_called_once()
            gaze_analyzer.reset_default_analyzer()
            self.assertTrue(self.mesh.closed)


class AnalyzerConfigTests(unittest.TestCase):
    def test_invalid_thresholds_and_face_capacity_are_rejected(self):
        invalid = [
            {"gaze_side_seconds": 0},
            {"gaze_down_seconds": -1},
            {"max_sample_gap_seconds": float("inf")},
            {"iris_side_threshold": float("nan")},
            {"no_face_seconds": True},
            {"max_num_faces": 1},
            {"max_num_faces": 2.5},
            {"smoothing_window": 2.5},
            {"calibration_min_samples": 2.5},
            {"combined_down_degrees": 36},
            {"combined_iris_down_threshold": 0.31},
        ]
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    AnalyzerConfig(**values)


class IrisGeometryTests(unittest.TestCase):
    @staticmethod
    def eye_landmarks(
        width, height, iris_x=0.5, iris_y=0.5, roll=0, eye_height=20, left_iris_x=None
    ):
        """Place both eyes in pixel geometry before normalizing for MediaPipe."""
        angle = math.radians(roll)
        horizontal = np.array([math.cos(angle), math.sin(angle)])
        vertical = np.array([-math.sin(angle), math.cos(angle)])
        landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]

        def put(index, point):
            landmarks[index] = SimpleNamespace(
                x=float(point[0] / width), y=float(point[1] / height), z=0.0
            )

        for center, indices, horizontal_ratio in (
            (np.array([width * 0.35, height * 0.45]), (33, 133, 159, 145, 468), iris_x),
            (
                np.array([width * 0.65, height * 0.45]),
                (362, 263, 386, 374, 473),
                iris_x if left_iris_x is None else left_iris_x,
            ),
        ):
            a, b, top, bottom, iris = indices
            put(a, center - horizontal * 30)
            put(b, center + horizontal * 30)
            put(top, center - vertical * eye_height / 2)
            put(bottom, center + vertical * eye_height / 2)
            put(
                iris,
                center
                + horizontal * 60 * (horizontal_ratio - 0.5)
                + vertical * eye_height * (iris_y - 0.5),
            )
        return landmarks

    def test_normalized_iris_coordinates_are_stable_for_roll_and_aspect_ratio(self):
        for width, height in ((640, 480), (1280, 720), (480, 640)):
            for roll in (-25, 0, 25):
                for iris_x, iris_y in ((0.5, 0.5), (0.25, 0.65), (0.75, 0.85)):
                    with self.subTest(size=(width, height), roll=roll, iris=(iris_x, iris_y)):
                        landmarks = self.eye_landmarks(width, height, iris_x, iris_y, roll)
                        with patch(
                            "gaze_analyzer.estimate_head_pose", return_value=HeadPose(0, 0, roll)
                        ):
                            result = measure_face(landmarks, width, height)
                        self.assertIsNotNone(result)
                        self.assertAlmostEqual(result.iris_x, iris_x, places=6)
                        self.assertAlmostEqual(result.iris_y, iris_y, places=6)
                        self.assertEqual(len(result.iris_centers), 2)

    def test_closed_eyes_keep_pose_but_do_not_supply_iris_gaze(self):
        landmarks = self.eye_landmarks(640, 480, eye_height=2)
        with patch("gaze_analyzer.estimate_head_pose", return_value=HeadPose(0, 0, 0)):
            result = measure_face(landmarks, 640, 480)
        self.assertIsNotNone(result)
        self.assertIsNone(result.iris_x)
        self.assertIsNone(result.iris_y)

    def test_opposing_iris_estimates_are_treated_as_unreliable(self):
        landmarks = self.eye_landmarks(640, 480, iris_x=0.2, left_iris_x=0.8)
        with patch("gaze_analyzer.estimate_head_pose", return_value=HeadPose(0, 0, 0)):
            result = measure_face(landmarks, 640, 480)
        self.assertIsNone(result.iris_x)
        self.assertIsNone(result.iris_y)

    def test_missing_refined_landmarks_still_supply_head_pose(self):
        landmarks = self.eye_landmarks(640, 480)[:468]
        with patch("gaze_analyzer.estimate_head_pose", return_value=HeadPose(12, 18, 0)):
            result = measure_face(landmarks, 640, 480)
        self.assertEqual(result.head_pose, HeadPose(12, 18, 0))
        self.assertIsNone(result.iris_x)
        self.assertIsNone(result.iris_y)


class HeadGeometryTests(unittest.TestCase):
    @staticmethod
    def project_landmarks(pitch, yaw, roll, width, height):
        """Project an upright face independently of the solver's Euler extraction.

        The canonical model is oriented in camera coordinates (y down).
        Positive pitch turns down, positive yaw turns
        toward image-left, and positive roll tilts clockwise in the image.
        """
        pitch, yaw, roll = np.deg2rad([pitch, yaw, roll])
        rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(pitch), -math.sin(pitch)],
                [0, math.sin(pitch), math.cos(pitch)],
            ]
        )
        ry = np.array(
            [[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]]
        )
        rz = np.array(
            [[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]]
        )
        rotation = rz @ ry @ rx
        rotation_vector, _ = cv2.Rodrigues(rotation)
        camera = np.array(
            [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64
        )
        points, _ = cv2.projectPoints(
            np.asarray(HEAD_MODEL_POINTS, dtype=np.float64),
            rotation_vector,
            np.array([0.0, 0.0, 60.0]),
            camera,
            np.zeros((4, 1)),
        )
        landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
        for index, point in zip(HEAD_LANDMARK_IDS, points.reshape(-1, 2), strict=True):
            landmarks[index] = SimpleNamespace(
                x=float(point[0] / width), y=float(point[1] / height), z=0.0
            )
        return landmarks

    def test_projected_pose_angles_use_degrees_and_expected_signs(self):
        poses = [
            (0, 0, 0),
            (20, 0, 0),
            (-20, 0, 0),
            (0, 28, 0),
            (0, -28, 0),
            (0, 0, 17),
            (12, -18, 9),
        ]
        for width, height in ((640, 480), (1280, 720), (480, 640)):
            for pitch, yaw, roll in poses:
                with self.subTest(size=(width, height), pose=(pitch, yaw, roll)):
                    landmarks = self.project_landmarks(pitch, yaw, roll, width, height)
                    result = estimate_head_pose(landmarks, width, height)
                    self.assertIsNotNone(result)
                    self.assertAlmostEqual(result.pitch, pitch, delta=0.5)
                    self.assertAlmostEqual(result.yaw, yaw, delta=0.5)
                    self.assertAlmostEqual(result.roll, roll, delta=0.5)

    def test_invalid_geometry_has_no_pose(self):
        self.assertIsNone(estimate_head_pose([], 640, 480))
        landmarks = [SimpleNamespace(x=float("nan"), y=0.5, z=0) for _ in range(478)]
        self.assertIsNone(estimate_head_pose(landmarks, 640, 480))


if __name__ == "__main__":
    unittest.main()
