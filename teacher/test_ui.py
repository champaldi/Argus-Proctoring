"""Быстрая проверка основных экранов без открытия настоящего окна."""

import os
import tempfile
import unittest
from contextlib import chdir
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QTableWidget  # noqa: E402

from teacher.data import recalculate_risk  # noqa: E402
from teacher.seed_demo import seed_demo  # noqa: E402
from teacher.ui import TeacherWindow  # noqa: E402
from core.storage import EventStore
from events import ProctorEvent


class TeacherUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_legacy_cwd_relative_screenshot_is_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "proctoring.db"
            seed_demo(database, root / "screenshots")
            window = TeacherWindow(database, root / "reviews.db")
            try:
                stored = window.sessions[0].events[0]
                original = window._frame_pixmap(stored).toImage()
                # Old storage persisted paths relative to the launching directory.
                # The checkout and system temp directory can be on different
                # Windows drives. Launch from the fixture's drive so a relative
                # path exists, independently of where the repository lives.
                with chdir(root.parent):
                    relative = os.path.relpath(stored.screenshot_path, Path.cwd())
                    legacy = replace(stored, screenshot_path=relative)
                    self.assertEqual(window._frame_pixmap(legacy).toImage(), original)
            finally:
                window.close()

    def test_event_table_shows_detected_application_and_service(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = EventStore(root / "events.db", root / "shots")
            store.start_session("apps")
            store.record_event(
                ProctorEvent.create("suspicious_process", source="security", details={
                    "processes": [{"name": "AnyDesk.exe", "pid": 1234}],
                    "services": [{"name": "chromoting", "display_name": "Chrome Remote Desktop"}],
                }),
                session_id="apps", weight=15, risk_total=15,
            )
            store.finish_session("apps", final_risk=15)
            store.close()
            window = TeacherWindow(root / "events.db", root / "reviews.db")
            try:
                window._open_row(0, 0)
                table = window.pages.widget(1).findChild(QTableWidget)
                text = table.item(0, 1).text()
                for expected in ("AnyDesk.exe", "1234", "Chrome Remote Desktop", "chromoting"):
                    self.assertIn(expected, text)
                window.show()
                self.app.processEvents()
                self.assertGreaterEqual(
                    table.rowHeight(0), table.fontMetrics().lineSpacing() * len(text.splitlines())
                )
            finally:
                window.close()

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

    def test_live_alert_appears_in_table_and_toast(self) -> None:
        from teacher.live import LiveBoard, LiveServer

        server = LiveServer(LiveBoard(), host="127.0.0.1", port=0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window = TeacherWindow(root / "missing.db", root / "reviews.db", live_server=server)
            try:
                window.show()
                self.assertFalse(window.live_table.isVisible())
                self.assertIn(str(server.port), window.live_address_label.text())
                base = {"session_id": "s1", "student": "Карим",
                        "started_at": "2026-10-16T10:00:00+00:00"}
                event = ProctorEvent.create("phone_aimed_at_screen", source="test")
                for message in (
                    {**base, "kind": "start"},
                    {**base, "kind": "event", "event": event.to_dict(), "weight": 45,
                     "risk_total": 45, "thumbnail": "bm90IGEganBlZw=="},
                ):
                    for alert in server.board.apply(message):
                        server.alerts.put(alert)
                server.changed.set()
                window._poll_live()
                self.app.processEvents()
                self.assertEqual(window.live_table.rowCount(), 1)
                self.assertEqual(window.live_table.item(0, 0).text(), "Карим")
                self.assertEqual(window.live_table.item(0, 2).text(), "45")
                self.assertIn("V1", window.live_table.item(0, 3).text())
                self.assertEqual(len(window._toasts), 1)  # «Начал тест» без всплывашки
                self.assertIn("Фотографирование экрана", window._toasts[0].text_label.text())
                for alert in server.board.apply({**base, "kind": "finish", "final_risk": 45}):
                    server.alerts.put(alert)
                window._poll_live()
                self.assertEqual(window.live_table.item(0, 1).text(), "Завершил")
                self.assertEqual(len(window._toasts), 2)
            finally:
                window.close()
                server.server_close()
                # Toasts and timers are Qt children: free them while QApplication lives.
                from shiboken6 import delete

                delete(window)


if __name__ == "__main__":
    unittest.main()
