"""Short repeated glances down add up to one gaze_down event."""

from gaze.analyzer import (
    DEFAULT_DOWN_BUDGET_SECONDS,
    DEFAULT_DOWN_WINDOW_SECONDS,
    AnalyzerConfig,
    GazeAnalyzer,
)
from gaze.tests.test_dropouts_and_background_faces import DOWN, SCREEN, Base, np, patch

STEP = 0.25


class AccumulatedBase(Base):
    budget = DEFAULT_DOWN_BUDGET_SECONDS

    def setUp(self):
        super().setUp()
        self.analyzer.close()
        self.analyzer = GazeAnalyzer(
            AnalyzerConfig(smoothing_window=1),
            face_mesh=self.mesh,
            clock=self.clock,
            gaze_gap_seconds=self.gap,
            min_extra_face_ratio=self.ratio,
            down_budget_seconds=self.budget,
        )
        self.addCleanup(self.analyzer.close)
        self.now = 0.0

    def hold(self, seconds, metric):
        """Feed frames for ``seconds``; return the full events of that stretch."""
        events = []
        for _ in range(round(seconds / STEP)):
            self.clock.now = self.now
            self.measure.return_value = metric
            events.extend(self.analyzer.analyze(self.frame))
            self.now += STEP
        return events

    def glances(self, count, down=3.0, up=3.0):
        events = []
        for _ in range(count):
            events.extend(self.hold(down, DOWN))
            events.extend(self.hold(up, SCREEN))
        return events


class AccumulatedDownTests(AccumulatedBase):
    def test_short_glances_add_up_to_one_event(self):
        # Each glance holds 2.75 s of credited time, under the 5 s threshold.
        events = self.glances(6)
        self.assertEqual([event["type"] for event in events], ["gaze_down"])
        event = events[0]
        self.assertTrue(event["accumulated"])
        self.assertGreaterEqual(event["duration"], DEFAULT_DOWN_BUDGET_SECONDS)
        self.assertEqual(event["window_seconds"], DEFAULT_DOWN_WINDOW_SECONDS)
        self.assertLess(event["started_at"], 1.0)

    def test_budget_starts_again_after_a_report(self):
        events = self.glances(12)
        self.assertEqual(len(events), 2)

    def test_a_few_glances_are_not_reported(self):
        self.assertEqual(self.glances(4), [])

    def test_rare_glances_fall_out_of_the_window(self):
        self.assertEqual(self.glances(12, down=3.0, up=20.0), [])

    def test_continuous_look_reports_once_and_spends_the_budget(self):
        events = self.hold(6.0, DOWN)
        self.assertEqual(len(events), 1)
        self.assertNotIn("accumulated", events[0])
        self.hold(1.0, SCREEN)
        # The 6 s already reported do not count towards the budget again.
        self.assertEqual(self.glances(4), [])

    def test_long_look_is_not_reported_twice(self):
        self.assertEqual(len(self.hold(40.0, DOWN)), 1)

    def test_accumulated_report_silences_the_continuous_timer_for_that_look(self):
        self.glances(5)  # 13.75 s credited
        events = self.hold(8.0, DOWN)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["accumulated"])

    def test_reset_clears_the_budget(self):
        self.glances(5)
        self.analyzer.reset()
        self.assertEqual(self.glances(4), [])

    def test_diagnostics_show_the_running_total(self):
        self.hold(3.0, DOWN)
        diagnostics = self.analyzer.get_diagnostics()
        self.assertAlmostEqual(diagnostics["down_accumulated_seconds"], 2.75)
        self.assertEqual(diagnostics["down_budget_seconds"], DEFAULT_DOWN_BUDGET_SECONDS)


class StrictAnalyzerTests(Base):
    def test_budget_is_off_unless_requested(self):
        events = []
        now = 0.0
        for _ in range(12):
            for metric in (DOWN, SCREEN):
                for _ in range(12):
                    events.extend(self.at(now, metric))
                    now += STEP
        self.assertEqual(events, [])

    def test_invalid_settings_are_rejected(self):
        for kwargs in (
            {"down_budget_seconds": -1},
            {"down_budget_seconds": float("nan")},
            {"down_budget_seconds": 30, "down_window_seconds": 10},
        ):
            with self.assertRaises(ValueError):
                GazeAnalyzer(face_mesh=self.mesh, **kwargs)

    def test_convenience_analyzer_enables_the_budget(self):
        from gaze import analyzer as module

        module.reset_default_analyzer()
        with patch.object(module, "GazeAnalyzer") as factory:
            factory.return_value.analyze.return_value = []
            module.analyze(np.zeros((48, 64, 3), dtype=np.uint8))
            module._default_analyzer = None
        self.assertEqual(
            factory.call_args.kwargs["down_budget_seconds"], DEFAULT_DOWN_BUDGET_SECONDS
        )
