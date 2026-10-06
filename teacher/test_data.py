"""Проверки чтения журнала и независимых решений преподавателя."""

from __future__ import annotations

import csv
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.storage import EventStore
from events import ProctorEvent

from teacher.data import (
    ReviewStore,
    current_verdict,
    export_roster,
    load_sessions,
    recalculate_risk,
    summarize_events,
)


class TeacherDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.database = self.root / "proctoring.db"
        self.started = datetime(2026, 10, 6, 10, 0, tzinfo=timezone.utc)
        store = EventStore(self.database, self.root / "screenshots")
        store.start_session("session-1", {"student_name": "Аня", "application": "proctoring"})
        phone = ProctorEvent.create(
            "phone_detected", source="test", occurred_at=self.started + timedelta(seconds=10),
            details={"duration_seconds": 3.5},
        )
        gaze = ProctorEvent.create(
            "gaze_side", source="test", occurred_at=self.started + timedelta(seconds=12),
            details={"duration": 2.0},
        )
        self.phone_id = phone.event_id
        self.gaze_id = gaze.event_id
        store.record_event(phone, session_id="session-1", weight=25, risk_total=25)
        store.record_event(gaze, session_id="session-1", weight=15, risk_total=40)
        store.finish_session("session-1", final_risk=39.4)
        store.connection.execute(
            "UPDATE sessions SET started_at=?, ended_at=? WHERE id=?",
            (self.started.isoformat(), (self.started + timedelta(seconds=15)).isoformat(),
             "session-1"),
        )
        store.connection.commit()
        store.close()

    def test_loads_student_session_events_and_duration(self) -> None:
        sessions = load_sessions(self.database)
        self.assertEqual(len(sessions), 1)
        session = sessions[0]
        self.assertEqual(session.student_name, "Аня")
        self.assertEqual(session.duration_seconds, 15)
        self.assertEqual([item.event.type.value for item in session.events],
                         ["phone_detected", "gaze_side"])
        self.assertEqual(session.final_risk, 39.4)

    def test_recalculation_excludes_false_positive_and_its_combination(self) -> None:
        session = load_sessions(self.database)[0]
        # Пересчёт возвращает максимум за сессию, без убывания к её концу.
        self.assertAlmostEqual(recalculate_risk(session), 39.6)
        self.assertAlmostEqual(recalculate_risk(session, {self.phone_id}), 10.0)
        self.assertAlmostEqual(recalculate_risk(session, {self.gaze_id}), 25.0)
        summary = summarize_events(session, {self.phone_id})
        self.assertEqual(summary["gaze_side"].count, 1)
        self.assertEqual(summary["gaze_side"].duration_seconds, 2.0)
        self.assertNotIn("phone_detected", summary)

    def test_early_violation_stays_visible_after_a_long_quiet_test(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute(
                    "UPDATE sessions SET ended_at=? WHERE id='session-1'",
                    ((self.started + timedelta(minutes=40)).isoformat(),),
                )
        session = load_sessions(self.database)[0]
        self.assertAlmostEqual(recalculate_risk(session), 39.6)

    def test_verdict_comment_and_flags_survive_reopening_without_changing_journal(self) -> None:
        reviews_path = self.root / "reviews.db"
        original_journal = self.database.read_bytes()
        reviews = ReviewStore(reviews_path)
        reviews.set_false_positive("session-1", self.phone_id, True)
        reviews.save_review("session-1", "questionable", "Нужно проверить кадр")
        reopened = ReviewStore(reviews_path).load_review("session-1")
        self.assertEqual(reopened.verdict, "questionable")
        self.assertEqual(reopened.comment, "Нужно проверить кадр")
        self.assertEqual(reopened.false_positive_ids, frozenset({self.phone_id}))
        self.assertEqual(len(load_sessions(self.database)[0].events), 2)
        self.assertEqual(self.database.read_bytes(), original_journal)

    def test_csv_exports_recalculated_risk_and_comment(self) -> None:
        session = load_sessions(self.database)[0]
        reviews = ReviewStore(self.root / "reviews.db")
        reviews.set_false_positive(session.id, self.phone_id, True)
        reviews.save_review(session.id, "not_cheated", "Ошибочный телефон")
        output = self.root / "roster.csv"
        export_roster(output, [session], reviews.load_all())
        with output.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["студент"], "Аня")
        self.assertEqual(rows[0]["риск"], "10.0")
        self.assertEqual(rows[0]["вердикт"], "не списывал")
        self.assertEqual(rows[0]["комментарий"], "Ошибочный телефон")

    def test_legacy_verdict_remains_visible_after_flag_only(self) -> None:
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute(
                    "UPDATE sessions SET teacher_verdict='cheated' WHERE id='session-1'"
                )
        session = load_sessions(self.database)[0]
        reviews = ReviewStore(self.root / "reviews.db")
        reviews.set_false_positive(session.id, self.phone_id, True)
        self.assertEqual(current_verdict(session, reviews.load_review(session.id)), "cheated")


if __name__ == "__main__":
    unittest.main()
