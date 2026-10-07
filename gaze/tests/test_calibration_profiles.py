"""Saved calibration preserves behavior and rejects corrupt profiles atomically."""

from __future__ import annotations

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import gaze
from gaze import analyzer as module
from gaze.analyzer import AnalyzerConfig, FaceMetrics, GazeAnalyzer, HeadPose


class OneFaceDetector:
    def process(self, rgb):
        return SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=[])])

    def close(self):
        pass


class CalibrationProfileTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.analyzer = GazeAnalyzer(
            AnalyzerConfig(smoothing_window=1), face_mesh=OneFaceDetector()
        )
        self.addCleanup(self.analyzer.close)
        self.measure_patch = patch.object(module, "measure_face")
        self.measure = self.measure_patch.start()
        self.addCleanup(self.measure_patch.stop)
        self.screen = FaceMetrics(HeadPose(3.145, 0, 0), 0.534, 0.412, ((0.4, 0.4), (0.6, 0.4)))
        self.keyboard = FaceMetrics(HeadPose(6.462, 0, 0), 0.555, 0.401)
        self.analyzer.calibrate([self.screen] * 10)
        self.analyzer.calibrate([self.keyboard] * 10, target="keyboard")

    def interval(self, start, stop, measured):
        self.measure.return_value = measured
        events = []
        for timestamp in np.arange(start, stop + 0.01, 0.5):
            events.extend(self.analyzer.analyze(self.frame, timestamp=float(timestamp)))
        return events

    def test_json_roundtrip_preserves_keyboard_behavior_and_adaptive_threshold(self):
        profile = json.loads(json.dumps(self.analyzer.export_calibration(), allow_nan=False))
        with GazeAnalyzer(
            AnalyzerConfig(smoothing_window=1), face_mesh=OneFaceDetector()
        ) as restored:
            restored.import_calibration(profile)
            self.assertEqual(restored.get_calibration_status(), {"screen": True, "keyboard": True})
            self.assertEqual(restored.export_calibration(), profile)
            self.assertEqual(restored.get_diagnostics()["down_thresholds"]["head_degrees"], 20)
            self.measure.return_value = self.keyboard
            for timestamp in np.arange(0, 10.01, 0.5):
                self.assertEqual(restored.analyze(self.frame, timestamp=float(timestamp)), [])
            self.measure.return_value = FaceMetrics(HeadPose(28, 0, 0))
            for timestamp in np.arange(10.5, 15.01, 0.5):
                self.assertEqual(restored.analyze(self.frame, timestamp=float(timestamp)), [])
            events = restored.analyze(self.frame, timestamp=15.5)
            self.assertEqual([event["type"] for event in events], ["gaze_down"])
            self.assertAlmostEqual(events[0]["started_at"], 10.5)

    def test_export_contains_only_reference_numbers_without_frame_or_episode_data(self):
        self.interval(0, 2.5, FaceMetrics(HeadPose(3.145, 32, 0), 0.534, 0.412))
        profile = self.analyzer.export_calibration()
        self.assertEqual(set(profile), {"version", "screen", "keyboard"})
        for target in ("screen", "keyboard"):
            self.assertEqual(set(profile[target]), {"head_pose", "iris_x", "iris_y", "eye_open"})
            self.assertEqual(set(profile[target]["head_pose"]), {"pitch", "yaw", "roll"})
        json.dumps(profile, allow_nan=False)
        profile["screen"]["head_pose"]["pitch"] = 999
        self.assertAlmostEqual(
            self.analyzer.export_calibration()["screen"]["head_pose"]["pitch"], 3.145
        )

    def test_successful_import_discards_pending_episodes_and_previous_timestamps(self):
        side = FaceMetrics(HeadPose(3.145, 32, 0), 0.534, 0.412)
        self.assertEqual(self.interval(0, 2.5, side), [])
        profile = self.analyzer.export_calibration()
        self.analyzer.import_calibration(profile)
        self.assertEqual(self.analyzer.last_result.signals, ())
        self.assertIsNone(self.analyzer.get_diagnostics()["episodes"]["gaze_side"]["started_at"])
        self.assertEqual(self.interval(0, 2.5, side), [])
        events = self.interval(3, 3, side)
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 0)

    def test_invalid_imports_preserve_profiles_diagnostics_and_pending_episode(self):
        valid = self.analyzer.export_calibration()
        invalid = [
            None,
            [],
            "profile",
            {},
            {"version": 1},
            {"version": 2, "screen": None, "keyboard": None},
        ]
        for version in (True, 1.0, "1", None):
            candidate = copy.deepcopy(valid)
            candidate["version"] = version
            invalid.append(candidate)
        for key in ("screen", "keyboard"):
            candidate = copy.deepcopy(valid)
            del candidate[key]
            invalid.append(candidate)
        for target in ("screen", "keyboard"):
            for replacement in ([], "metrics", {"head_pose": None}, None):
                candidate = copy.deepcopy(valid)
                candidate[target] = replacement
                if replacement is not None:
                    invalid.append(candidate)
            for axis in ("pitch", "yaw", "roll"):
                for value in (float("nan"), float("inf"), True, "3", None):
                    candidate = copy.deepcopy(valid)
                    candidate[target]["head_pose"][axis] = value
                    invalid.append(candidate)
            for iris in ("iris_x", "iris_y"):
                for value in (float("nan"), float("inf"), True, "0.5"):
                    candidate = copy.deepcopy(valid)
                    candidate[target][iris] = value
                    invalid.append(candidate)
            candidate = copy.deepcopy(valid)
            candidate[target]["head_pose"] = None
            invalid.append(candidate)
            for missing in ("head_pose", "iris_x", "iris_y"):
                candidate = copy.deepcopy(valid)
                del candidate[target][missing]
                invalid.append(candidate)
        for iris in ("iris_x", "iris_y"):
            candidate = copy.deepcopy(valid)
            candidate["screen"][iris] = None
            invalid.append(candidate)
        candidate = copy.deepcopy(valid)
        candidate["screen"] = None
        invalid.append(candidate)  # A keyboard reference cannot stand alone.
        candidate = copy.deepcopy(valid)
        candidate["screen"]["head_pose"]["pitch"] = 99
        candidate["keyboard"]["iris_y"] = float("nan")
        invalid.append(candidate)  # Valid first reference must not be committed early.

        side = FaceMetrics(HeadPose(3.145, 32, 0), 0.534, 0.412)
        self.assertEqual(self.interval(0, 2.5, side), [])
        diagnostics = copy.deepcopy(self.analyzer.get_diagnostics())
        for index, profile in enumerate(invalid):
            with self.subTest(case=index, profile=profile):
                with self.assertRaises(ValueError):
                    self.analyzer.import_calibration(profile)
                self.assertEqual(self.analyzer.export_calibration(), valid)
                self.assertEqual(self.analyzer.get_diagnostics(), diagnostics)
        events = self.interval(3, 3, side)
        self.assertEqual([event["type"] for event in events], ["gaze_side"])
        self.assertAlmostEqual(events[0]["started_at"], 0)

    def test_empty_profile_clears_references_and_timers(self):
        self.interval(0, 2.5, FaceMetrics(HeadPose(3.145, 32, 0), 0.534, 0.412))
        empty = {"version": 1, "screen": None, "keyboard": None}
        self.analyzer.import_calibration(empty)
        self.assertEqual(
            self.analyzer.get_calibration_status(), {"screen": False, "keyboard": False}
        )
        self.assertEqual(self.analyzer.export_calibration(), empty)
        self.assertEqual(self.analyzer.last_result.elapsed, {})
        self.assertEqual(self.analyzer.get_diagnostics()["down_thresholds"]["head_degrees"], 35)
        self.assertEqual(self.interval(0, 6, FaceMetrics(HeadPose(28, 0, 0))), [])

    def test_screen_only_profile_restores_without_adaptive_keyboard_threshold(self):
        profile = self.analyzer.export_calibration()
        profile["keyboard"] = None
        self.analyzer.import_calibration(profile)
        self.assertEqual(
            self.analyzer.get_calibration_status(), {"screen": True, "keyboard": False}
        )
        self.assertEqual(self.analyzer.get_diagnostics()["down_thresholds"]["head_degrees"], 35)

    def test_pose_only_keyboard_profile_is_valid_and_retains_keyboard_exemption(self):
        profile = self.analyzer.export_calibration()
        profile["keyboard"]["iris_x"] = None
        profile["keyboard"]["iris_y"] = None
        self.analyzer.import_calibration(profile)
        self.assertEqual(self.analyzer.get_calibration_status(), {"screen": True, "keyboard": True})
        self.assertEqual(self.analyzer.export_calibration(), profile)
        self.assertEqual(self.interval(0, 10, FaceMetrics(self.keyboard.head_pose)), [])

    def test_import_into_a_closed_analyzer_is_rejected(self):
        profile = self.analyzer.export_calibration()
        self.analyzer.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.analyzer.import_calibration(profile)

    def test_module_diagnostics_read_does_not_construct_a_detector(self):
        module.reset_default_analyzer()
        self.addCleanup(module.reset_default_analyzer)
        with patch.object(module, "GazeAnalyzer", side_effect=AssertionError("model started")):
            self.assertIsNone(module.get_diagnostics())
            self.assertIsNone(gaze.get_diagnostics())


if __name__ == "__main__":
    unittest.main()
