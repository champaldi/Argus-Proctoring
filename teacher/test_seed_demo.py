"""Проверка демонстрационных сессий в формате общего журнала."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from teacher.data import load_sessions, recalculate_risk, risk_zone
from teacher.seed_demo import seed_demo


class SeedDemoTests(unittest.TestCase):
    def test_creates_six_sessions_and_placeholder_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "proctoring.db"
            seed_demo(database, root / "screenshots")
            sessions = load_sessions(database)
            self.assertEqual(len(sessions), 6)
            self.assertEqual(Counter(risk_zone(recalculate_risk(item)) for item in sessions),
                             {"low": 2, "medium": 2, "high": 2})
            self.assertTrue(all("demo" in item.student_name.lower() for item in sessions))
            self.assertTrue(all(item.events for item in sessions))
            self.assertTrue(all(
                Path(stored.screenshot_path).is_file()
                for item in sessions for stored in item.events
            ))
            seed_demo(database, root / "screenshots")
            self.assertEqual(len(load_sessions(database)), 6)


if __name__ == "__main__":
    unittest.main()
