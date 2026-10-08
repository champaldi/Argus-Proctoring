"""The classic profile: the 6 October gaze behaviour with shorter times."""

import unittest
from unittest.mock import patch

from gaze import analyzer as module
from gaze.tests.test_auto_baseline import AT_SCREEN, look
from gaze.tests.test_dropouts_and_background_faces import Base, face

STRAIGHT = look(0.0, 0.30, iris_y=0.5, yaw=0.0)
TURNED = look(0.0, 0.30, iris_y=0.5, yaw=30.0)
HEAD_DOWN = look(40.0, 0.30, iris_y=0.5, yaw=0.0)


class ClassicProfileTests(Base):
    def setUp(self):
        super().setUp()
        self.analyzer.close()
        with patch.dict("os.environ", {"PROCTOR_DETECTION_PROFILE": "classic"}):
            self.analyzer = module.build_application_analyzer(
                face_mesh=self.mesh, clock=self.clock
            )
        self.addCleanup(self.analyzer.close)
        self.now = 0.0

    def hold(self, seconds, metric, step=0.5):
        events = []
        for _ in range(round(seconds / step)):
            events.extend(self.at(self.now, metric))
            self.now += step
        return events

    def test_turn_aside_is_reported_after_two_seconds(self):
        self.assertEqual(self.hold(1.5, TURNED), [])
        self.assertEqual(self.hold(1.5, TURNED), ["gaze_side"])

    def test_head_down_is_reported_after_three_seconds(self):
        self.assertEqual(self.hold(2.5, HEAD_DOWN), [])
        self.assertEqual(self.hold(1.5, HEAD_DOWN), ["gaze_down"])

    def test_reference_is_fixed_not_learned(self):
        self.hold(10.0, AT_SCREEN)
        self.assertEqual(self.analyzer.get_diagnostics()["baseline_source"], "default")

    def test_looking_straight_is_quiet(self):
        self.assertEqual(self.hold(60.0, STRAIGHT), [])

    def test_slow_frames_still_count(self):
        self.assertEqual(self.hold(3.9, TURNED, step=1.3), ["gaze_side"])

    def test_poster_face_is_still_ignored(self):
        self.mesh.faces = [face(0.30), face(0.08, center=0.85)]
        self.assertEqual(self.hold(5.0, STRAIGHT), [])


class ProfileSwitchTests(unittest.TestCase):
    def test_classic_is_the_default_and_adaptive_is_opt_in(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("PROCTOR_DETECTION_PROFILE", None)
            self.assertEqual(module.detection_profile(), "classic")
        with patch.dict("os.environ", {"PROCTOR_DETECTION_PROFILE": " Adaptive "}):
            self.assertEqual(module.detection_profile(), "adaptive")
        with patch.dict("os.environ", {"PROCTOR_DETECTION_PROFILE": "whatever"}):
            self.assertEqual(module.detection_profile(), "classic")
