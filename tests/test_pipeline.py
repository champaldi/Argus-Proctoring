import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.pipeline import EventPipeline
from core.risk import RiskScorer
from core.storage import record_teacher_verdict
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


if __name__ == "__main__":
    unittest.main()



class ReferencePhotoTests(unittest.TestCase):
    def test_first_calm_frame_after_the_delay_becomes_the_control_photo(self) -> None:
        import json
        from unittest.mock import patch

        import numpy as np

        from core import pipeline as pipeline_module

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = EventPipeline(root / "events.db", root / "screenshots")
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            with patch.object(pipeline_module, "REFERENCE_PHOTO_DELAY_SECONDS", 0.0):
                pipeline.start()
                self.assertTrue(pipeline.offer_reference_frame(frame))
                self.assertFalse(pipeline.offer_reference_frame(frame))
                pipeline.stop()
            connection = sqlite3.connect(root / "events.db")
            try:
                metadata = json.loads(
                    connection.execute("SELECT metadata_json FROM sessions").fetchone()[0]
                )
            finally:
                connection.close()
            self.assertTrue(Path(metadata["reference_photo"]).is_file())

    def test_no_photo_before_the_camera_settles(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = EventPipeline(root / "events.db", root / "screenshots")
            pipeline.start()
            self.assertFalse(pipeline.offer_reference_frame(np.zeros((4, 4, 3), np.uint8)))
            pipeline.stop()
