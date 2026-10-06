"""Заключение, контрольный кадр и HTML-отчёт: без Qt и без камеры."""

import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.pipeline import EventPipeline  # noqa: E402
from core.storage import REFERENCE_PHOTO_NAME, EventStore  # noqa: E402
from events import ProctorEvent  # noqa: E402
from teacher.conclusion import build_conclusion  # noqa: E402
from teacher.data import Review, load_sessions, resolve_evidence_path  # noqa: E402
from teacher.report import (  # noqa: E402
    default_report_name,
    render_session_report,
    write_session_report,
)


def frame(value: int = 90) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


class ReportFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "proctoring.db"
        self.started = datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc)
        store = EventStore(self.database, self.root / "screenshots")
        store.start_session("s1", {"student_name": "Аня <Ким>", "test_score": 7,
                                   "test_total": 10})
        self.ids = {}
        rows = (
            ("gaze_side", 40, 10, 10),
            ("phone_detected", 130, 25, 35),
            ("phone_aimed_at_screen", 135, 60, 95),
            ("suspicious_process", 200, 15, 100),
        )
        for name, offset, weight, total in rows:
            details = {"duration_seconds": 2.5}
            if name == "suspicious_process":
                details = {"processes": [{"name": "AnyDesk.exe", "pid": 77}]}
            event = ProctorEvent.create(
                name, source="test", details=details,
                occurred_at=self.started + timedelta(seconds=offset),
            )
            self.ids[name] = event.event_id
            store.record_event(event, session_id="s1", weight=weight, risk_total=total,
                               frame=frame() if name.startswith("phone") else None)
        store.save_reference_photo("s1", frame(140))
        store.finish_session("s1", final_risk=95)
        store.connection.execute(
            "UPDATE sessions SET started_at=?, ended_at=? WHERE id='s1'",
            (self.started.isoformat(), (self.started + timedelta(minutes=5)).isoformat()),
        )
        store.connection.commit()
        store.close()
        self.session = load_sessions(self.database)[0]

    def tearDown(self) -> None:
        self.temporary.cleanup()


class ReferencePhotoTests(ReportFixture):
    def test_reference_photo_is_stored_next_to_evidence_and_in_metadata(self) -> None:
        path = resolve_evidence_path(self.database, self.session.reference_photo)
        self.assertIsNotNone(path)
        self.assertEqual(path.name, REFERENCE_PHOTO_NAME)
        self.assertEqual(path.parent.name, "s1")
        self.assertEqual(self.session.student_name, "Аня <Ким>")

    def test_missing_or_relative_evidence_path_is_resolved_safely(self) -> None:
        self.assertIsNone(resolve_evidence_path(self.database, None))
        self.assertIsNone(resolve_evidence_path(self.database, "нет/такого.jpg"))
        relative = Path("screenshots") / "s1" / REFERENCE_PHOTO_NAME
        self.assertEqual(
            resolve_evidence_path(self.database, str(relative)),
            self.database.parent / relative,
        )

    def test_pipeline_keeps_one_calm_frame_after_the_start_delay(self) -> None:
        pipeline = EventPipeline(self.root / "live.db", self.root / "live_shots",
                                 metadata={"student_name": "Иван"})
        pipeline.start()
        self.assertFalse(pipeline.offer_reference_frame(frame()))  # камера ещё настраивается
        pipeline._started_at = time.monotonic() - 10
        self.assertFalse(pipeline.offer_reference_frame(None))
        self.assertTrue(pipeline.offer_reference_frame(frame()))
        self.assertFalse(pipeline.offer_reference_frame(frame(200)))
        pipeline.stop()
        session = load_sessions(self.root / "live.db")[0]
        self.assertEqual(session.student_name, "Иван")
        self.assertTrue(Path(session.reference_photo).is_file())
        self.assertFalse(pipeline.offer_reference_frame(frame()))


class ConclusionTests(ReportFixture):
    def test_high_risk_conclusion_names_strongest_reasons_first(self) -> None:
        conclusion = build_conclusion(self.session)
        self.assertEqual(conclusion.zone, "high")
        self.assertIn("Высокий риск", conclusion.headline)
        self.assertTrue(conclusion.reasons[0].startswith("телефон наведён на экран — 1 раз"))
        self.assertIn("на 3-й минуте", conclusion.reasons[0])
        self.assertIn("проверка преподавателем", conclusion.recommendation)

    def test_dismissed_events_change_the_conclusion(self) -> None:
        dismissed = {self.ids["phone_detected"], self.ids["phone_aimed_at_screen"]}
        conclusion = build_conclusion(self.session, dismissed)
        self.assertNotEqual(conclusion.zone, "high")
        self.assertFalse(any("телефон" in reason for reason in conclusion.reasons))

    def test_single_phone_episode_is_never_called_clean(self) -> None:
        keep = self.ids["phone_detected"]
        dismissed = {value for value in self.ids.values() if value != keep}
        conclusion = build_conclusion(self.session, dismissed)
        self.assertEqual(conclusion.zone, "low")
        self.assertIn("подозрительные", conclusion.headline)
        self.assertIn("просмотреть", conclusion.recommendation)

    def test_session_without_events_reports_no_violations(self) -> None:
        conclusion = build_conclusion(self.session, set(self.ids.values()))
        self.assertEqual(conclusion.reasons, ())
        self.assertIn("нарушений не осталось", conclusion.headline)

    def test_russian_plural_of_times(self) -> None:
        from teacher.conclusion import _times

        self.assertEqual([_times(n) for n in (1, 2, 5, 11, 12, 21, 22, 104)],
                         ["1 раз", "2 раза", "5 раз", "11 раз", "12 раз", "21 раз",
                          "22 раза", "104 раза"])


class SessionReportTests(ReportFixture):
    def test_report_is_self_contained_and_escapes_student_text(self) -> None:
        review = Review("questionable", "Проверить <b>кадр</b>",
                        frozenset({self.ids["gaze_side"]}))
        html = render_session_report(self.session, review, self.database)
        self.assertIn("Аня &lt;Ким&gt;", html)
        self.assertNotIn("Аня <Ким>", html)
        self.assertIn("Проверить &lt;b&gt;кадр&lt;/b&gt;", html)
        self.assertIn("7 из 10", html)
        self.assertIn("Приложение: AnyDesk.exe", html)
        self.assertIn("Под вопросом", html)
        self.assertIn("снято преподавателем", html)
        # Контрольный кадр и два кадра с телефоном встроены, внешних ссылок нет.
        self.assertEqual(html.count("data:image/jpeg;base64,"), 3)
        for external in ("http://", "https://", "<script", "<link"):
            self.assertNotIn(external, html)

    def test_report_without_frames_or_review_still_renders(self) -> None:
        for path in (self.root / "screenshots").rglob("*.jpg"):
            path.unlink()
        html = render_session_report(self.session, None, self.database)
        self.assertNotIn("data:image", html)
        self.assertIn("Нет кадра", html)
        self.assertIn("Не проверено", html)

    def test_report_file_and_default_name(self) -> None:
        target = self.root / default_report_name(self.session)
        self.assertTrue(target.name.endswith(".html"))
        self.assertNotIn("<", target.name)
        write_session_report(target, self.session, None, self.database)
        self.assertIn("<!doctype html>", target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
