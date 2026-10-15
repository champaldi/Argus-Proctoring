"""Правила корреляции по журналу событий: без Qt и без камеры."""

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.storage import EventStore  # noqa: E402
from events import ProctorEvent  # noqa: E402
from teacher.conclusion import build_conclusion  # noqa: E402
from teacher.data import load_sessions  # noqa: E402
from teacher.report import render_session_report  # noqa: E402
from teacher.rules import classify, evaluate_rules  # noqa: E402


class RulesFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.database = self.root / "proctoring.db"
        self.started = datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc)
        self.ids: list[str] = []

    def session(self, *rows: tuple[str, float]):
        """rows: (тип события, секунда от начала теста)."""
        store = EventStore(self.database, self.root / "screenshots")
        store.start_session("s1", {"student_name": "Тест"})
        for name, offset in rows:
            event = ProctorEvent.create(
                name, source="test", occurred_at=self.started + timedelta(seconds=offset)
            )
            self.ids.append(event.event_id)
            store.record_event(event, session_id="s1", weight=10, risk_total=10)
        store.finish_session("s1", final_risk=10)
        store.connection.execute(
            "UPDATE sessions SET started_at=?, ended_at=? WHERE id='s1'",
            (self.started.isoformat(), (self.started + timedelta(minutes=20)).isoformat()),
        )
        store.connection.commit()
        store.close()
        return load_sessions(self.database)[0]

    def fired(self, *rows, excluded=()):
        return [finding.rule_id for finding in evaluate_rules(self.session(*rows), excluded)]


class RuleTests(RulesFixture):
    def test_clean_session_fires_nothing(self):
        session = self.session()
        self.assertEqual(evaluate_rules(session), ())
        self.assertEqual(classify(evaluate_rules(session)), "clean")

    def test_single_look_down_is_only_an_observation(self):
        self.assertEqual(self.fired(("gaze_down", 60), ("gaze_side", 200)), [])

    def test_phone_alone_is_suspicious(self):
        self.assertEqual(self.fired(("phone_detected", 60)), ["S1"])

    def test_phone_with_look_down_nearby_is_cheating_from_a_phone(self):
        self.assertEqual(self.fired(("gaze_down", 50), ("phone_detected", 70)), ["V2"])

    def test_phone_and_look_down_far_apart_do_not_correlate(self):
        self.assertEqual(self.fired(("gaze_down", 50), ("phone_detected", 400)), ["S1"])

    def test_phone_aimed_at_screen_is_a_violation_and_covers_the_phone(self):
        self.assertEqual(
            self.fired(("phone_detected", 60), ("phone_aimed_at_screen", 63)), ["V1"]
        )

    def test_repeated_looks_down_are_suspicious_only_within_the_window(self):
        self.assertEqual(
            self.fired(("gaze_down", 10), ("gaze_down", 100), ("gaze_down", 250)), ["S2"]
        )

    def test_spread_out_looks_down_are_not_reported(self):
        self.assertEqual(
            self.fired(("gaze_down", 10), ("gaze_down", 400), ("gaze_down", 900)), []
        )

    def test_remote_session_is_a_violation(self):
        self.assertEqual(self.fired(("remote_session", 5)), ["V3"])

    def test_injected_input_alone_is_suspicious_and_with_a_program_a_violation(self):
        self.assertEqual(self.fired(("injected_input", 30)), ["S8"])

    def test_injected_input_with_forbidden_program_is_outside_control(self):
        self.assertEqual(
            self.fired(("suspicious_process", 20), ("injected_input", 60)), ["V4"]
        )

    def test_emergency_release_is_a_violation_and_covers_exit_attempts(self):
        self.assertEqual(
            self.fired(("hotkey_blocked", 10), ("hotkey_blocked", 20),
                       ("hotkey_blocked", 30), ("protection_disabled", 40)),
            ["V7"],
        )

    def test_second_face_with_side_look_is_a_prompt(self):
        self.assertEqual(self.fired(("multiple_faces", 100), ("gaze_side", 110)), ["V5"])

    def test_second_face_alone_is_suspicious(self):
        self.assertEqual(self.fired(("multiple_faces", 100)), ["S3"])

    def test_hotkeys_need_three_within_two_minutes(self):
        self.assertEqual(self.fired(("hotkey_blocked", 10), ("hotkey_blocked", 20)), [])

    def test_three_exit_attempts_are_suspicious(self):
        self.assertEqual(
            self.fired(("hotkey_blocked", 10), ("window_switched", 20), ("hotkey_blocked", 90)),
            ["S7"],
        )

    def test_violations_come_first_then_by_time(self):
        self.assertEqual(
            self.fired(
                ("suspicious_process", 5), ("gaze_down", 300), ("phone_detected", 310),
                ("multiple_faces", 700),
            ),
            ["V2", "S6", "S3"],
        )

    def test_event_removed_by_the_teacher_breaks_the_correlation(self):
        session = self.session(("gaze_down", 50), ("phone_detected", 70))
        phone_id = self.ids[1]
        self.assertEqual(evaluate_rules(session, [phone_id]), ())
        down_id = self.ids[0]
        self.assertEqual(
            [finding.rule_id for finding in evaluate_rules(session, [down_id])], ["S1"]
        )

    def test_finding_names_rule_minute_and_events(self):
        session = self.session(("gaze_down", 125), ("phone_detected", 140))
        finding = evaluate_rules(session)[0]
        self.assertEqual(finding.event_ids, tuple(self.ids))
        text = finding.as_text(session)
        self.assertIn("Признаки нарушения: Списывание с телефона (V2, 3-я минута)", text)


class ConclusionWithRulesTests(RulesFixture):
    def test_violation_changes_headline_even_at_low_risk(self):
        conclusion = build_conclusion(self.session(("gaze_down", 50), ("phone_detected", 70)))
        self.assertNotEqual(conclusion.zone, "high")
        self.assertEqual(conclusion.classification, "violation")
        self.assertIn("признаки нарушения", conclusion.headline)
        self.assertIn("Сработавшие правила:", conclusion.as_text())

    def test_suspicious_finding_asks_to_look_at_frames(self):
        conclusion = build_conclusion(
            self.session(("gaze_down", 10), ("gaze_down", 100), ("gaze_down", 250))
        )
        self.assertEqual(conclusion.classification, "suspicious")
        self.assertIn("подозрительные", conclusion.headline)

    def test_observations_keep_the_calm_wording(self):
        conclusion = build_conclusion(self.session(("gaze_down", 60)))
        self.assertEqual(conclusion.classification, "clean")
        self.assertEqual(conclusion.findings, ())
        self.assertNotIn("Сработавшие правила", conclusion.as_text())

    def test_report_lists_fired_rules(self):
        session = self.session(("gaze_down", 50), ("phone_detected", 70))
        html = render_session_report(session, None, self.database)
        self.assertIn("Сработавшие правила", html)
        self.assertIn("Списывание с телефона (V2", html)


if __name__ == "__main__":
    unittest.main()
