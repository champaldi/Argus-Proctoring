import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from core.pipeline import EventPipeline
from events import EventType, ProctorEvent


class PipelineTests(unittest.TestCase):
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
                ProctorEvent.create(EventType.PHONE_DETECTED, source="test")
            )
            second = pipeline.submit(
                ProctorEvent.create(EventType.PHONE_DETECTED, source="test")
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
            )
            pipeline.start()
            pipeline.submit(ProctorEvent.create("gaze_side", source="test"))
            pipeline.submit(ProctorEvent.create("window_switched", source="test"))
            pipeline.stop()

            self.assertEqual([item.risk_total for item in recorded], [6, 21])


if __name__ == "__main__":
    unittest.main()

