import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPushButton
from shiboken6 import delete

from core.storage import EventStore
from events import ProctorEvent
from ui.main_window import StudentFinishDialog, TeacherReviewDialog


class StudentFinishDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_student_sees_only_completion_without_verdict_or_risk(self) -> None:
        dialog = StudentFinishDialog(student_name="Карим", answered=4, total=4)
        self.addCleanup(delete, dialog)
        texts = " ".join(label.text() for label in dialog.findChildren(QLabel))
        buttons = [button.text() for button in dialog.findChildren(QPushButton)]
        self.assertIn("Тест завершён", texts)
        self.assertIn("4 из 4", texts)
        self.assertNotIn("риск", texts.lower())
        self.assertEqual(buttons, ["Закрыть"])

    def test_close_button_accepts(self) -> None:
        dialog = StudentFinishDialog(student_name=None, answered=2, total=4)
        self.addCleanup(delete, dialog)
        button = dialog.findChild(QPushButton, "finishClose")
        QTimer.singleShot(0, button.click)
        self.assertEqual(dialog.exec(), QDialog.DialogCode.Accepted)


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
