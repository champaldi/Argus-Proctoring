import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog
from shiboken6 import delete

from core.storage import EventStore
from events import ProctorEvent
from ui.main_window import TeacherReviewDialog


class TeacherReviewDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

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
            # Signal callbacks keep the dialog alive after accept(). Destroy it
            # before QApplication is torn down, including on assertion failure.
            self.addCleanup(delete, dialog)
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
