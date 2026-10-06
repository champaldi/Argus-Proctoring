import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from core.pipeline import EventPipeline
from core.risk import RiskScorer
from core.storage import load_session_events, record_teacher_verdict
from events import EventType, ProctorEvent


class PipelineTests(unittest.TestCase):
    def test_different_applications_detected_in_quick_succession_are_both_saved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = EventPipeline(root / "events.db", root / "screenshots")
            pipeline.start()
            try:
                for pid, name in ((20, "AnyDesk.exe"), (30, "Discord.exe")):
                    pipeline.submit(ProctorEvent.create(
                        "suspicious_process", source="security",
                        details={"processes": [{"pid": pid, "name": name}]},
                    ))
            finally:
                pipeline.stop()
            stored = load_session_events(root / "events.db", pipeline.session_id)
            names = [item.event.details["processes"][0]["name"] for item in stored]
            self.assertEqual(names, ["AnyDesk.exe", "Discord.exe"])

    def test_records_event_and_suppresses_immediate_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorded = []
            pipeline = EventPipeline(
                root / "events.db",
                root / "screenshots",
                on_recorded=recorded.append,
            )
            pipeline.start()
            first = pipeline.submit(
                ProctorEvent.create(EventType.WINDOW_SWITCHED, source="test")
            )
            second = pipeline.submit(
                ProctorEvent.create(EventType.WINDOW_SWITCHED, source="test")
            )
            pipeline.stop()

            self.assertTrue(first)
            self.assertFalse(second)
            self.assertEqual(len(recorded), 1)
            connection = sqlite3.connect(root / "events.db")
            try:
                count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                status = connection.execute(
                    "SELECT status FROM sessions"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(count, 1)
            self.assertEqual(status, "completed")

    def test_risk_is_cumulative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorded = []
            pipeline = EventPipeline(
                root / "events.db",
                root / "screenshots",
                on_recorded=recorded.append,
                cooldowns={event_type: 0.0 for event_type in EventType},
                scorer=RiskScorer(clock=lambda: 0.0),
            )
            pipeline.start()
            pipeline.submit(ProctorEvent.create("gaze_side", source="test"))
            pipeline.submit(ProctorEvent.create("window_switched", source="test"))
            pipeline.stop()

            self.assertEqual([item.risk_total for item in recorded], [10.0, 32.5])

    def test_final_risk_and_teacher_verdict_are_stored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = EventPipeline(
                root / "events.db",
                root / "screenshots",
                scorer=RiskScorer(clock=lambda: 0.0),
            )
            pipeline.start()
            pipeline.submit(ProctorEvent.create("phone_detected", source="test"))
            session_id = pipeline.session_id
            pipeline.stop()
            record_teacher_verdict(root / "events.db", session_id, "not_cheated")

            connection = sqlite3.connect(root / "events.db")
            try:
                row = connection.execute(
                    "SELECT final_risk, teacher_verdict FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(row, (25.0, "not_cheated"))

    def test_stored_final_risk_is_the_session_peak(self) -> None:
        now = [0.0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = EventPipeline(
                root / "events.db",
                root / "screenshots",
                scorer=RiskScorer(clock=lambda: now[0]),
            )
            pipeline.start()
            pipeline.submit(ProctorEvent.create("phone_detected", source="test"))
            session_id = pipeline.session_id
            deadline = time.monotonic() + 5
            while pipeline.peak_risk().total < 25.0 and time.monotonic() < deadline:
                time.sleep(0.01)
            now[0] = 30 * 60.0
            self.assertEqual(pipeline.current_risk().total, 0.0)
            pipeline.stop()

            connection = sqlite3.connect(root / "events.db")
            try:
                final_risk = connection.execute(
                    "SELECT final_risk FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(final_risk, 25.0)

    def test_second_process_report_is_not_dropped_by_cooldown(self) -> None:
        recorded = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = EventPipeline(
                root / "events.db",
                root / "screenshots",
                on_recorded=recorded.append,
                scorer=RiskScorer(clock=lambda: 0.0),
            )
            pipeline.start()
            first = pipeline.submit(ProctorEvent.create("suspicious_process", source="test"))
            second = pipeline.submit(ProctorEvent.create("suspicious_process", source="test"))
            pipeline.stop()

            self.assertTrue(first and second)
            self.assertEqual(len(recorded), 2)


if __name__ == "__main__":
    unittest.main()

