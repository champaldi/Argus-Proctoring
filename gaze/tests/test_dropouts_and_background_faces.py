"""Brief signal dropouts and small background faces, with a substitute detector."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gaze import analyzer as gaze_analyzer
from gaze.analyzer import (
    DEFAULT_GAZE_GAP_SECONDS,
    DEFAULT_MIN_EXTRA_FACE_RATIO,
    FACE_CONTOUR_IDS,
    AnalyzerConfig,
    FaceMetrics,
    GazeAnalyzer,
    HeadPose,
)


def face(width, center=0.5):
    """Landmarks whose face oval spans ``width`` of the frame."""
    points = [SimpleNamespace(x=center, y=0.5) for _ in range(max(FACE_CONTOUR_IDS) + 1)]
    for position, index in enumerate(FACE_CONTOUR_IDS):
        points[index] = SimpleNamespace(
            x=center + (width / 2 if position % 2 else -width / 2), y=0.5
        )
    return SimpleNamespace(landmark=points)


class WidthFaceMesh:
    def __init__(self):
        self.faces = [face(0.30)]

    def process(self, rgb):
        return SimpleNamespace(multi_face_landmarks=list(self.faces) or None)

    def close(self):
        pass


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def metrics(pitch=0.0, iris_y=0.5, yaw=0.0):
    return FaceMetrics(HeadPose(pitch, yaw, 0.0), 0.5, iris_y)


DOWN = metrics(pitch=40.0)
SIDE = metrics(yaw=30.0)
SCREEN = metrics()


class Base(unittest.TestCase):
    gap = DEFAULT_GAZE_GAP_SECONDS
    ratio = DEFAULT_MIN_EXTRA_FACE_RATIO

    def setUp(self):
        self.mesh = WidthFaceMesh()
        self.clock = Clock()
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.analyzer = GazeAnalyzer(
            AnalyzerConfig(smoothing_window=1),
            face_mesh=self.mesh,
            clock=self.clock,
            gaze_gap_seconds=self.gap,
            min_extra_face_ratio=self.ratio,
        )
        patcher = patch("gaze.analyzer.measure_face", return_value=SCREEN)
        self.measure = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.analyzer.close)

    def at(self, timestamp, metric=SCREEN):
        self.clock.now = timestamp
        self.measure.return_value = metric
        return [event["type"] for event in self.analyzer.analyze(self.frame)]

    def run_timeline(self, timeline):
        events = []
        for timestamp, metric in timeline:
            events.extend(self.at(timestamp, metric))
        return events


class GazeDropoutTests(Base):
    def test_single_frames_under_threshold_do_not_restart_a_held_look_down(self):
        timeline = []
        for step in range(0, 33):  # 0.0 .. 5.33 s at six frames a second
            timestamp = step / 6
            # Every fifth frame slips under the threshold, as near-threshold poses do.
            timeline.append((timestamp, SCREEN if step % 5 == 4 else DOWN))
        self.assertEqual(self.run_timeline(timeline), ["gaze_down"])

    def test_strict_analyzer_still_restarts_on_every_dropout(self):
        strict = GazeAnalyzer(
            AnalyzerConfig(smoothing_window=1), face_mesh=self.mesh, clock=self.clock
        )
        self.addCleanup(strict.close)
        events = []
        for step in range(0, 33):
            self.clock.now = step / 6
            self.measure.return_value = SCREEN if step % 5 == 4 else DOWN
            events.extend(event["type"] for event in strict.analyze(self.frame))
        self.assertEqual(events, [])

    def test_return_to_the_screen_longer_than_the_gap_restarts_the_episode(self):
        timeline = [(step / 6, DOWN) for step in range(0, 24)]            # 0 .. 3.8 s down
        timeline += [(4.0 + step / 6, SCREEN) for step in range(0, 6)]    # 1 s at the screen
        timeline += [(5.0 + step / 6, DOWN) for step in range(0, 12)]     # 2 s down again
        self.assertEqual(self.run_timeline(timeline), [])

    def test_event_is_never_emitted_on_a_frame_without_the_signal(self):
        timeline = [(step / 6, DOWN) for step in range(0, 30)]            # 0 .. 4.83 s
        self.assertEqual(self.run_timeline(timeline), [])
        self.assertEqual(self.at(5.0, SCREEN), [])                         # dropout at the 5 s mark
        self.assertEqual(self.at(5.17, DOWN), ["gaze_down"])

    def test_a_held_episode_is_reported_once_despite_later_dropouts(self):
        timeline = [(step / 6, SIDE) for step in range(0, 20)]            # 3.2 s aside
        timeline += [(3.4, SCREEN), (3.6, SIDE), (3.8, SCREEN), (4.0, SIDE)]
        self.assertEqual(self.run_timeline(timeline), ["gaze_side"])

    def test_dropout_tolerance_applies_to_gaze_only(self):
        self.mesh.faces = []
        self.assertEqual(self.run_timeline([(0.0, SCREEN), (1.0, SCREEN)]), [])
        self.mesh.faces = [face(0.30)]
        self.assertEqual(self.at(1.2), [])           # the student is back for one frame
        self.mesh.faces = []
        # Absence starts again from here; the two seconds are not carried over.
        self.assertEqual(self.run_timeline([(1.4, SCREEN), (2.4, SCREEN), (3.2, SCREEN)]), [])
        self.assertEqual(self.at(3.5), ["no_face"])

    def test_invalid_settings_are_rejected(self):
        for kwargs in (
            {"gaze_gap_seconds": -0.1},
            {"gaze_gap_seconds": float("nan")},
            {"gaze_gap_seconds": True},
            {"gaze_gap_seconds": 5.0},          # would swallow the sample-gap reset
            {"min_extra_face_ratio": -1},
            {"min_extra_face_ratio": 1.5},
        ):
            with self.assertRaises(ValueError, msg=kwargs):
                GazeAnalyzer(face_mesh=self.mesh, clock=self.clock, **kwargs)


class BackgroundFaceTests(Base):
    def test_small_face_on_a_poster_is_not_a_second_person(self):
        self.mesh.faces = [face(0.30), face(0.06, center=0.1)]
        self.assertEqual(self.run_timeline([(t / 2, SCREEN) for t in range(0, 8)]), [])
        self.assertEqual(self.analyzer.last_result.face_count, 1)

    def test_comparable_second_face_is_still_reported(self):
        self.mesh.faces = [face(0.30), face(0.20, center=0.8)]
        events = self.run_timeline([(t / 2, SCREEN) for t in range(0, 4)])
        self.assertEqual(events, ["multiple_faces"])
        self.assertEqual(self.analyzer.last_result.face_count, 2)

    def test_the_student_is_the_largest_face_whatever_the_detector_order(self):
        self.mesh.faces = [face(0.05, center=0.1), face(0.50)]
        events = self.run_timeline([(t / 2, SCREEN) for t in range(0, 8)])
        # Width 0.50 exceeds the too-close ratio, proving the large face is used.
        self.assertEqual(events, ["too_close_to_camera"])
        self.assertAlmostEqual(self.analyzer.last_result.face_width_ratio, 0.50)

    def test_unmeasurable_faces_are_never_dropped(self):
        self.mesh.faces = [SimpleNamespace(landmark=[]), SimpleNamespace(landmark=[])]
        events = self.run_timeline([(t / 2, SCREEN) for t in range(0, 4)])
        self.assertEqual(events, ["multiple_faces"])

    def test_strict_analyzer_counts_every_face(self):
        strict = GazeAnalyzer(
            AnalyzerConfig(smoothing_window=1), face_mesh=self.mesh, clock=self.clock
        )
        self.addCleanup(strict.close)
        self.mesh.faces = [face(0.30), face(0.06, center=0.1)]
        events = []
        for step in range(0, 4):
            self.clock.now = step / 2
            events.extend(event["type"] for event in strict.analyze(self.frame))
        self.assertEqual(events, ["multiple_faces"])


class ConvenienceAnalyzerTests(unittest.TestCase):
    def test_application_entry_point_uses_the_tolerant_settings(self):
        created = {}

        class Recorder:
            def __init__(self, *args, **kwargs):
                created.update(kwargs)

            def analyze(self, frame):
                return []

            def close(self):
                pass

        gaze_analyzer.reset_default_analyzer()
        with patch.object(gaze_analyzer, "GazeAnalyzer", Recorder):
            try:
                gaze_analyzer.analyze(np.zeros((4, 4, 3), dtype=np.uint8))
            finally:
                gaze_analyzer.reset_default_analyzer()
        self.assertEqual(
            created,
            {
                "gaze_gap_seconds": DEFAULT_GAZE_GAP_SECONDS,
                "min_extra_face_ratio": DEFAULT_MIN_EXTRA_FACE_RATIO,
                "down_budget_seconds": gaze_analyzer.DEFAULT_DOWN_BUDGET_SECONDS,
            },
        )


if __name__ == "__main__":
    unittest.main()
