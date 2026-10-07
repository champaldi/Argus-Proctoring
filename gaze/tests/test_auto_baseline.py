"""The first seconds of a session become the screen reference.

The numbers come from a real laptop camera: facing the screen the head pitch
was about +5 degrees, and looking at the keyboard it was about +14 with the
eyelids visibly lowered. Fixed thresholds (35 degrees) never saw that look.
"""

from types import SimpleNamespace

from gaze.analyzer import (
    DEFAULT_AUTO_BASELINE_SECONDS,
    AnalyzerConfig,
    FaceMetrics,
    GazeAnalyzer,
    HeadPose,
    measure_eye_opening,
)
from gaze.tests.test_dropouts_and_background_faces import Base

STEP = 0.2


def look(pitch, eye_open, iris_y=0.37, yaw=-4.0):
    return FaceMetrics(HeadPose(pitch, yaw, 0.0), 0.5, iris_y, eye_open=eye_open)


AT_SCREEN = look(4.9, 0.30)
AT_KEYBOARD = look(13.8, 0.17, iris_y=0.23)
BOTTOM_OF_SCREEN = look(7.5, 0.27, iris_y=0.33)
SLOUCHED = look(13.0, 0.30)
HEAD_WELL_DOWN = look(21.0, 0.30)
EYES_HIDDEN = FaceMetrics(HeadPose(13.8, -4.0, 0.0), eye_open=0.05)


class AutoBase(Base):
    seconds = DEFAULT_AUTO_BASELINE_SECONDS

    def setUp(self):
        super().setUp()
        self.analyzer.close()
        self.analyzer = GazeAnalyzer(
            AnalyzerConfig(smoothing_window=1),
            face_mesh=self.mesh,
            clock=self.clock,
            gaze_gap_seconds=self.gap,
            auto_baseline_seconds=self.seconds,
        )
        self.addCleanup(self.analyzer.close)
        self.now = 0.0

    def hold(self, seconds, metric):
        events = []
        for _ in range(round(seconds / STEP)):
            events.extend(self.at(self.now, metric))
            self.now += STEP
        return events

    def source(self):
        return self.analyzer.get_diagnostics()["baseline_source"]


class AutoBaselineTests(AutoBase):
    def test_reference_is_learned_from_the_first_seconds(self):
        self.hold(2.0, AT_SCREEN)
        self.assertEqual(self.source(), "default")
        self.hold(2.0, AT_SCREEN)
        self.assertEqual(self.source(), "auto")
        screen = self.analyzer.get_diagnostics()["screen"]
        self.assertAlmostEqual(screen["head_pose"]["pitch"], 4.9)
        self.assertAlmostEqual(screen["eye_open"], 0.30)

    def test_look_at_the_keyboard_is_reported_after_five_seconds(self):
        self.hold(4.0, AT_SCREEN)
        self.assertEqual(self.hold(4.8, AT_KEYBOARD), [])
        self.assertEqual(self.hold(1.0, AT_KEYBOARD), ["gaze_down"])

    def test_reading_the_bottom_of_the_screen_is_not_reported(self):
        self.hold(4.0, AT_SCREEN)
        self.assertEqual(self.hold(20.0, BOTTOM_OF_SCREEN), [])

    def test_slouching_with_open_eyes_is_not_reported(self):
        self.hold(4.0, AT_SCREEN)
        self.assertEqual(self.hold(20.0, SLOUCHED), [])

    def test_head_well_down_is_reported_even_with_open_eyes(self):
        self.hold(4.0, AT_SCREEN)
        self.assertEqual(self.hold(6.0, HEAD_WELL_DOWN), ["gaze_down"])

    def test_nearly_closed_eyes_with_a_tilted_head_count_as_down(self):
        self.hold(4.0, AT_SCREEN)
        self.assertEqual(self.hold(6.0, EYES_HIDDEN), ["gaze_down"])

    def test_facing_the_screen_after_learning_is_quiet(self):
        self.assertEqual(self.hold(60.0, AT_SCREEN), [])

    def test_frames_without_open_eyes_do_not_feed_the_reference(self):
        self.hold(6.0, EYES_HIDDEN)
        self.assertEqual(self.source(), "default")

    def test_reference_survives_a_timer_reset(self):
        self.hold(4.0, AT_SCREEN)
        self.analyzer.reset()
        self.assertEqual(self.source(), "auto")
        self.assertEqual(self.hold(6.0, AT_KEYBOARD), ["gaze_down"])

    def test_manual_calibration_takes_precedence(self):
        self.hold(4.0, AT_SCREEN)
        self.analyzer.calibrate([AT_KEYBOARD] * 10)
        self.assertEqual(self.source(), "calibrated")
        self.assertEqual(self.hold(8.0, AT_KEYBOARD), [])


class StrictTests(Base):
    def test_without_the_setting_the_keyboard_look_stays_unreported(self):
        events = []
        for step in range(60):
            events.extend(self.at(step * STEP, AT_SCREEN if step < 20 else AT_KEYBOARD))
        self.assertEqual(events, [])
        self.assertEqual(self.analyzer.get_diagnostics()["baseline_source"], "default")

    def test_invalid_ratio_is_rejected(self):
        with self.assertRaises(ValueError):
            AnalyzerConfig(eyelid_down_open_ratio=1.0)
        with self.assertRaises(ValueError):
            AnalyzerConfig(eyelid_down_degrees=20.0)


class EyeOpeningTests(Base):
    def landmarks(self, height):
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        eyes = ((33, 133, 159, 145, 0.4), (362, 263, 386, 374, 0.6))
        for left, right, top, bottom, center in eyes:
            points[left] = SimpleNamespace(x=center - 0.05, y=0.5)
            points[right] = SimpleNamespace(x=center + 0.05, y=0.5)
            points[top] = SimpleNamespace(x=center, y=0.5 - height / 2)
            points[bottom] = SimpleNamespace(x=center, y=0.5 + height / 2)
        return points

    def test_ratio_is_eye_height_over_width(self):
        self.assertAlmostEqual(measure_eye_opening(self.landmarks(0.03), 1000, 1000), 0.30)

    def test_closed_eye_is_zero_not_unknown(self):
        self.assertAlmostEqual(measure_eye_opening(self.landmarks(0.0), 1000, 1000), 0.0)

    def test_too_few_landmarks_are_unknown(self):
        self.assertIsNone(measure_eye_opening(self.landmarks(0.03)[:100], 1000, 1000))
