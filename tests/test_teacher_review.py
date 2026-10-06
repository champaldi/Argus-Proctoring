import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QLabel

from config import AppConfig
from core.storage import EventStore, StoredEvent, load_session_events
from events import ProctorEvent
from ui.main_window import MainWindow, TeacherReviewDialog


class TeacherReviewDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_review_shows_process_and_service_names_loaded_from_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "events.db"
            store = EventStore(database, root / "screenshots")
            store.start_session("review-session")
            store.record_event(
                ProctorEvent.create(
                    "suspicious_process", source="security", details={
                        "processes": [
                            {"name": "AnyDesk.exe", "pid": 1234},
                            {"name": "Discord.exe", "pid": 5678},
                        ],
                        "services": [{
                            "name": "chromoting", "display_name": "Chrome Remote Desktop",
                            "status": "running",
                        }],
                    },
                ),
                session_id="review-session", weight=15, risk_total=15,
            )
            store.close()
            dialog = TeacherReviewDialog(
                database_path=database, session_id="review-session",
                events=load_session_events(database, "review-session"),
                final_risk=15, test_score=(3, 4),
            )
            self.addCleanup(dialog.close)
            text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
            for expected in ("AnyDesk.exe", "1234", "Discord.exe", "5678",
                             "Chrome Remote Desktop", "chromoting"):
                self.assertIn(expected, text)

    def test_review_handles_service_only_and_old_events_without_names(self) -> None:
        for details, expected in (
            ({"services": [{"name": "TeamViewer"}]}, "TeamViewer"),
            ({}, "Обнаружен запрещённый процесс"),
        ):
            with self.subTest(details=details):
                event = ProctorEvent.create("suspicious_process", source="security", details=details)
                dialog = TeacherReviewDialog(
                    database_path=Path("unused.db"), session_id="review-session",
                    events=[StoredEvent(event, 15, 15, None)],
                    final_risk=15, test_score=(3, 4),
                )
                self.addCleanup(dialog.close)
                text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
                self.assertIn(expected, text)
                self.assertNotIn("None", text)

    def test_live_event_feed_identifies_the_application(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(AppConfig.from_env(), data_dir=root,
                             database_path=root / "events.db", screenshots_dir=root / "shots")
            # The feed can be tested without starting the camera or global OS hooks.
            with patch.object(MainWindow, "start_monitoring", lambda window: None):
                window = MainWindow(config)
                try:
                    self.app.processEvents()
                    event = ProctorEvent.create("suspicious_process", source="security", details={
                        "processes": [{"name": "AnyDesk.exe", "pid": 1234}],
                    })
                    window._on_event_recorded(StoredEvent(event, 15, 15, None))
                    self.assertIn("AnyDesk.exe", window.event_list.item(0).text())
                    self.assertIn("1234", window.event_list.item(0).text())
                finally:
                    window.close()

    def test_teacher_verdict_is_saved_from_review_screen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "events.db"
            store = EventStore(database, root / "screenshots")
            session_id = "review-session"
            store.start_session(session_id)
            stored = store.record_event(
                ProctorEvent.create("gaze_side", source="test"),
                session_id=session_id,
                weight=10.0,
                risk_total=10.0,
            )
            store.finish_session(session_id, final_risk=10.0)
            store.close()

            dialog = TeacherReviewDialog(
                database_path=database,
                session_id=session_id,
                events=[stored],
                final_risk=10.0,
                test_score=(3, 4),
            )
            QTimer.singleShot(0, lambda: dialog._save_verdict("cheated"))
            result = dialog.exec()

            self.assertEqual(result, QDialog.DialogCode.Accepted)
            connection = sqlite3.connect(database)
            try:
                verdict = connection.execute(
                    "SELECT teacher_verdict FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(verdict, "cheated")


if __name__ == "__main__":
    unittest.main()
