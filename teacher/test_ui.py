"""Быстрая проверка основных экранов без открытия настоящего окна."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QTableWidget  # noqa: E402

from teacher.data import recalculate_risk  # noqa: E402
from teacher.seed_demo import seed_demo  # noqa: E402
from teacher.ui import TeacherWindow  # noqa: E402


class TeacherUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_sessions_filter_and_detail_screen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "proctoring.db"
            seed_demo(database, root / "screenshots")
            window = TeacherWindow(database, root / "reviews.db")
            try:
                self.assertEqual(window.table.rowCount(), 6)
                self.assertIn("Всего 6, требуют проверки 6", window.counter_label.text())
                window._open_row(0, 0)
                self.assertEqual(window.pages.currentIndex(), 1)
                self.assertIsNotNone(window.detail_session)
                session = window.detail_session
                events = window.pages.widget(1).findChild(QTableWidget)
                marked_id = session.events[-1].event.event_id
                events.item(len(session.events) - 1, 5).setCheckState(Qt.CheckState.Checked)
                review = window.reviews.load_review(session.id)
                self.assertIn(marked_id, review.false_positive_ids)
                self.assertLess(recalculate_risk(session, review.false_positive_ids), 60)
                window.comment_edit.setPlainText("Проверено")
                window._save_review(session, "cheated")
                window._back()
                self.assertIn("требуют проверки 5", window.counter_label.text())
                window.review_filter.setChecked(True)
                self.assertEqual(window.table.rowCount(), 5)
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
