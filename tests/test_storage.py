import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.storage import EventStore
from events import ProctorEvent


class ScreenshotTests(unittest.TestCase):
    def test_event_screenshot_survives_a_cyrillic_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Снимки теста"
            store = EventStore(root / "events.db", root / "кадры")
            try:
                store.start_session("session")
                stored = store.record_event(
                    ProctorEvent.create("phone_detected", source="test"),
                    session_id="session", weight=25, risk_total=25,
                    frame=np.zeros((32, 32, 3), dtype=np.uint8),
                )
                self.assertIsNotNone(stored.screenshot_path)
                saved = Path(stored.screenshot_path)
                decoded = cv2.imdecode(np.frombuffer(saved.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape, (32, 32, 3))
                self.assertNotIn("screenshot_error", stored.event.details)
            finally:
                store.close()
