import unittest

from core.risk import RiskScorer
from events import ProctorEvent


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class RiskScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = ManualClock()
        self.scorer = RiskScorer(clock=self.clock)

    @staticmethod
    def event(name: str) -> ProctorEvent:
        return ProctorEvent.create(name, source="test")

    def test_configured_weights_and_combination_multiplier(self) -> None:
        first = self.scorer.add(self.event("phone_detected"))
        self.clock.now = 5.0
        second = self.scorer.add(self.event("gaze_side"))

        self.assertEqual(first.weight, 25.0)
        self.assertEqual(first.multiplier, 1.0)
        self.assertEqual(second.weight, 15.0)
        self.assertEqual(second.multiplier, 1.5)
        # Five seconds of quiet time decays one point before the second signal.
        self.assertEqual(second.total, 39.0)

    def test_same_signal_does_not_get_combination_multiplier(self) -> None:
        self.scorer.add(self.event("gaze_down"))
        second = self.scorer.add(self.event("gaze_down"))
        self.assertEqual(second.weight, 10.0)
        self.assertEqual(second.multiplier, 1.0)

    def test_signal_outside_window_has_no_multiplier(self) -> None:
        self.scorer.add(self.event("phone_detected"))
        self.clock.now = 11.0
        second = self.scorer.add(self.event("gaze_side"))
        self.assertEqual(second.weight, 10.0)
        self.assertEqual(second.total, 32.8)

    def test_risk_gradually_decays_and_uses_requested_zones(self) -> None:
        self.scorer.add(self.event("phone_aimed_at_screen"))
        self.assertEqual(self.scorer.current().level, "medium")
        self.clock.now = 50.0
        snapshot = self.scorer.current()
        self.assertEqual(snapshot.total, 30.0)
        self.assertEqual(snapshot.level, "medium")
        self.clock.now = 55.0
        self.assertEqual(self.scorer.current().level, "low")

    def test_capture_protection_failure_has_security_weight(self) -> None:
        update = self.scorer.add(self.event("capture_protection_failed"))
        self.assertEqual(update.weight, 15.0)

    def test_remote_session_has_strong_security_weight(self) -> None:
        update = self.scorer.add(self.event("remote_session"))
        self.assertEqual(update.weight, 30.0)

    def test_multiple_monitors_has_presence_weight(self) -> None:
        update = self.scorer.add(self.event("multiple_monitors"))
        self.assertEqual(update.weight, 20.0)

    def test_injected_input_has_security_weight(self) -> None:
        update = self.scorer.add(self.event("injected_input"))
        self.assertEqual(update.weight, 15.0)


if __name__ == "__main__":
    unittest.main()
