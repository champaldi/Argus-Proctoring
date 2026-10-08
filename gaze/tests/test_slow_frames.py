"""The application's analyzer keeps working when frames arrive slowly.

With YOLO, CLIP and a screen recorder on one laptop processor, frames reached
the gaze module less than once a second; a one-second limit reset everything.
"""

from gaze.analyzer import (
    DEFAULT_APP_GAZE_GAP_SECONDS,
    DEFAULT_AUTO_BASELINE_SECONDS,
    DEFAULT_DOWN_BUDGET_SECONDS,
    DEFAULT_GAZE_DOWN_SECONDS,
    DEFAULT_MAX_SAMPLE_GAP_SECONDS,
    DEFAULT_MIN_EXTRA_FACE_RATIO,
    AnalyzerConfig,
    GazeAnalyzer,
)
from gaze.tests.test_auto_baseline import AT_KEYBOARD, AT_SCREEN, look
from gaze.tests.test_dropouts_and_background_faces import Base

TURNED = look(4.9, 0.30, yaw=-34.0)


class SlowFramesTests(Base):
    def setUp(self):
        super().setUp()
        self.analyzer.close()
        self.analyzer = GazeAnalyzer(
            AnalyzerConfig(
                smoothing_window=1,
                gaze_down_seconds=DEFAULT_GAZE_DOWN_SECONDS,
                max_sample_gap_seconds=DEFAULT_MAX_SAMPLE_GAP_SECONDS,
            ),
            face_mesh=self.mesh,
            clock=self.clock,
            gaze_gap_seconds=DEFAULT_APP_GAZE_GAP_SECONDS,
            min_extra_face_ratio=DEFAULT_MIN_EXTRA_FACE_RATIO,
            down_budget_seconds=DEFAULT_DOWN_BUDGET_SECONDS,
            auto_baseline_seconds=DEFAULT_AUTO_BASELINE_SECONDS,
        )
        self.addCleanup(self.analyzer.close)
        self.now = 0.0

    def hold(self, seconds, metric, step):
        events = []
        for _ in range(round(seconds / step)):
            events.extend(self.at(self.now, metric))
            self.now += step
        return events

    def test_reference_is_learned_at_one_frame_per_1_3_seconds(self):
        self.hold(16.0, AT_SCREEN, 1.3)
        self.assertEqual(self.analyzer.get_diagnostics()["baseline_source"], "auto")

    def test_look_down_is_reported_at_slow_frames(self):
        self.hold(16.0, AT_SCREEN, 1.3)
        self.assertEqual(self.hold(7.8, AT_KEYBOARD, 1.3), ["gaze_down"])

    def test_turn_aside_is_reported_at_slow_frames(self):
        self.hold(16.0, AT_SCREEN, 1.3)
        self.assertEqual(self.hold(5.2, TURNED, 1.3), ["gaze_side"])

    def test_one_missed_frame_does_not_restart_a_look(self):
        self.hold(16.0, AT_SCREEN, 0.8)
        events = []
        for index in range(8):
            events.extend(self.hold(0.8, AT_SCREEN if index == 3 else AT_KEYBOARD, 0.8))
        self.assertEqual(events, ["gaze_down"])

    def test_honest_session_at_slow_frames_is_quiet(self):
        self.assertEqual(self.hold(120.0, AT_SCREEN, 1.3), [])

    def test_a_real_pause_still_resets(self):
        self.hold(16.0, AT_SCREEN, 1.0)
        self.hold(3.0, AT_KEYBOARD, 1.0)
        self.now += 4.0  # camera stalled longer than the limit
        self.assertEqual(self.hold(2.0, AT_KEYBOARD, 1.0), [])
