"""Центральный журнал преподавателя и веб-страница с отчётами."""

from __future__ import annotations

import base64
import json
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

from core.live import LiveReporter, RejectedMessage
from core.pipeline import EventPipeline
from core.storage import EventStore, StoredEvent
from events import ProctorEvent
from teacher.archive import SessionArchive
from teacher.data import load_sessions
from teacher.live import LiveBoard, LiveServer, _is_student_reachable, _is_tailscale

# Самый маленький допустимый для архива «JPEG»: архив проверяет только сигнатуру.
FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"argus" * 10 + b"\xff\xd9"
FAKE_JPEG_B64 = base64.b64encode(FAKE_JPEG).decode("ascii")


def _wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _messages(session: str = "s1", student: str = "Карим"):
    base = {"session_id": session, "student": student,
            "started_at": "2026-10-16T10:00:00+00:00", "sent_at": "2026-10-16T10:05:00+00:00"}
    phone = ProctorEvent.create("phone_aimed_at_screen", source="test")
    gaze = ProctorEvent.create("gaze_down", source="test")
    return [
        {**base, "kind": "start"},
        {**base, "kind": "reference", "image": FAKE_JPEG_B64},
        {**base, "kind": "event", "event": phone.to_dict(), "weight": 45,
         "risk_total": 45, "thumbnail": FAKE_JPEG_B64},
        {**base, "kind": "event", "event": gaze.to_dict(), "weight": 5,
         "risk_total": 50, "thumbnail": "not base64 at all"},
        {**base, "kind": "finish", "final_risk": 50},
    ]


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "proctoring.db"
        self.archive = SessionArchive(self.database)

    def tearDown(self):
        self.temporary.cleanup()

    def test_remote_session_becomes_a_full_session(self):
        for message in _messages():
            self.archive.record(message)
        sessions = load_sessions(self.database)
        self.assertEqual(len(sessions), 1)
        session = sessions[0]
        self.assertEqual(session.student_name, "Карим")
        self.assertEqual(session.status, "completed")
        self.assertIsNotNone(session.ended_at)
        self.assertEqual(session.final_risk, 50)
        self.assertEqual(session.metadata.get("source"), "live")
        self.assertTrue(Path(session.reference_photo).is_file())
        self.assertEqual(
            [stored.event.type.value for stored in session.events],
            ["phone_aimed_at_screen", "gaze_down"],
        )
        phone, gaze = session.events
        self.assertEqual(Path(phone.screenshot_path).read_bytes(), FAKE_JPEG)
        self.assertIsNone(gaze.screenshot_path)  # испорченный кадр не записан

    def test_repeated_delivery_does_not_duplicate(self):
        messages = _messages()
        for message in messages + messages:
            self.archive.record(message)
        session = load_sessions(self.database)[0]
        self.assertEqual(len(session.events), 2)
        self.assertEqual(len(list((self.root / "screenshots" / "s1").glob("*.jpg"))), 2)

    def test_session_from_this_computer_is_kept_as_is(self):
        store = EventStore(self.database, self.root / "screenshots")
        store.start_session("s1", {"student_name": "Карим"})
        messages = _messages()
        event = ProctorEvent(
            type="phone_aimed_at_screen", source="test",
            event_id=messages[2]["event"]["event_id"],
        )
        store.record_event(event, session_id="s1", weight=45, risk_total=45)
        store.finish_session("s1", final_risk=45)
        store.close()
        for message in messages:
            self.archive.record(message)
        session = load_sessions(self.database)[0]
        self.assertEqual(session.final_risk, 45)  # завершена локально, не перезаписана
        self.assertEqual(len(session.events), 2)
        self.assertIsNone(session.events[0].screenshot_path)

    def test_bad_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            self.archive.record({"kind": "start", "session_id": "../../etc"})

    def test_only_jpeg_is_written(self):
        message = _messages()[1]
        message["image"] = base64.b64encode(b"MZ\x90\x00 not an image").decode()
        self.archive.record(_messages()[0])
        self.archive.record(message)
        self.assertIsNone(load_sessions(self.database)[0].reference_photo)


class LateFrameTests(unittest.TestCase):
    def test_frame_sent_after_the_event_is_attached(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "proctoring.db"
            archive = SessionArchive(database)
            event = _messages()[2]
            text_only = {key: value for key, value in event.items() if key != "thumbnail"}
            archive.record(_messages()[0])
            archive.record(text_only)
            self.assertIsNone(load_sessions(database)[0].events[0].screenshot_path)
            archive.record(event)
            stored = load_sessions(database)[0].events[0]
            self.assertEqual(Path(stored.screenshot_path).read_bytes(), FAKE_JPEG)


class ResyncTests(unittest.TestCase):
    def _student_session(self, root: Path) -> Path:
        database = root / "student" / "events.db"
        store = EventStore(database, root / "student" / "shots")
        store.start_session("s9", {"student_name": "Нурлыбек"})
        shot = root / "frame.jpg"
        shot.write_bytes(FAKE_JPEG)
        store.connection.execute(
            "UPDATE sessions SET metadata_json = ? WHERE id = 's9'",
            (json.dumps({"student_name": "Нурлыбек", "reference_photo": str(shot)}),),
        )
        store.connection.commit()
        for kind in ("phone_aimed_at_screen", "gaze_down"):
            stored = store.record_event(
                ProctorEvent.create(kind, source="test"), session_id="s9",
                weight=10, risk_total=10,
            )
            store.connection.execute(
                "UPDATE events SET screenshot_path = ? WHERE id = ?",
                (str(shot), stored.event.event_id),
            )
        store.connection.commit()
        store.finish_session("s9", final_risk=20)
        store.close()
        return database

    def test_finished_session_is_resent_whole_and_marked(self):
        from core.live import resync_finished_sessions

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            student_db = self._student_session(root)
            teacher_db = root / "teacher" / "proctoring.db"
            board = LiveBoard()
            server = LiveServer(board, host="127.0.0.1", port=0,
                                archive=SessionArchive(teacher_db))
            server.start()
            try:
                base = f"http://127.0.0.1:{server.port}"
                encode = lambda path: FAKE_JPEG_B64 if path else None  # noqa: E731
                synced, error = resync_finished_sessions(student_db, base, thumbnailer=encode)
                self.assertEqual((synced, error), (1, ""))
                session = load_sessions(teacher_db)[0]
                self.assertEqual(session.student_name, "Нурлыбек")
                self.assertEqual(len(session.events), 2)
                self.assertTrue(all(
                    Path(stored.screenshot_path).is_file() for stored in session.events
                ))
                self.assertTrue(Path(session.reference_photo).is_file())
                self.assertIsNotNone(session.ended_at)
                self.assertEqual(board.sessions(), [])  # досылка не трогает активные
                self.assertEqual(server.drain(), [])
                # Второй запуск: уже отмечено, ничего не шлётся.
                self.assertEqual(resync_finished_sessions(student_db, base), (0, ""))
            finally:
                server.stop()

    def test_network_failure_leaves_session_for_next_start(self):
        from core.live import resync_finished_sessions

        with tempfile.TemporaryDirectory() as temporary:
            student_db = self._student_session(Path(temporary))

            def offline(*_):
                raise OSError("offline")

            synced, error = resync_finished_sessions(
                student_db, "127.0.0.1", sender=offline, thumbnailer=lambda _: None
            )
            self.assertEqual(synced, 0)
            self.assertIn("offline", error)
            sent = []
            synced, _ = resync_finished_sessions(
                student_db, "127.0.0.1",
                sender=lambda url, payload, token: sent.append(payload["kind"]),
                thumbnailer=lambda _: None,
            )
            self.assertEqual(synced, 1)
            self.assertEqual(sent[0], "start")
            self.assertEqual(sent[-1], "finish")


class RuleSummaryTests(unittest.TestCase):
    def test_violation_stays_main_rule_and_others_are_counted(self):
        board = LiveBoard()
        base = {"session_id": "s1", "student": "Карим"}

        def send(kind_name):
            event = ProctorEvent.create(kind_name, source="test")
            board.apply({**base, "kind": "event", "event": event.to_dict(),
                         "weight": 1, "risk_total": 10})

        send("suspicious_process")       # S6
        send("phone_aimed_at_screen")    # V1
        send("multiple_monitors")        # S9
        summary = board.sessions()[0].rule_summary
        self.assertTrue(summary.startswith("V1"), summary)
        self.assertTrue(summary.endswith("(+2)"), summary)


class AddressTests(unittest.TestCase):
    def test_virtual_adapters_are_hidden(self):
        self.assertTrue(_is_student_reachable("192.168.1.16"))
        for address in ("192.168.229.0", "10.5.0.0", "192.168.56.1", "169.254.3.4", "127.0.0.1"):
            self.assertFalse(_is_student_reachable(address), address)

    def test_tailscale_range(self):
        self.assertTrue(_is_tailscale("100.97.174.108"))
        self.assertFalse(_is_tailscale("100.20.1.1"))


class RejectedMessageTests(unittest.TestCase):
    def test_rejected_message_is_dropped_and_queue_moves_on(self):
        sent = []

        def sender(url, payload, token):
            if payload["kind"] == "event":
                raise RejectedMessage("HTTP 400")
            sent.append(payload["kind"])

        reporter = LiveReporter("127.0.0.1", sender=sender, thumbnailer=lambda _: None)
        reporter.start()
        reporter.session_started("s1", {})
        reporter.event_recorded("s1", StoredEvent(
            ProctorEvent.create("gaze_down", source="t"), 1, 1, None))
        reporter.session_finished("s1", 1)
        self.assertTrue(_wait(lambda: sent == ["start", "finish"]))
        reporter.close()
        self.assertEqual(reporter.dropped, 1)


class JournalServerTests(unittest.TestCase):
    def test_remote_pipeline_lands_in_teacher_journal_and_web_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher_db = root / "teacher" / "proctoring.db"
            server = LiveServer(
                LiveBoard(), host="127.0.0.1", port=0,
                archive=SessionArchive(teacher_db), reviews_path=root / "reviews.db",
            )
            server.start()
            try:
                base = f"http://127.0.0.1:{server.port}"
                reporter = LiveReporter(base, thumbnailer=lambda _: FAKE_JPEG_B64)
                pipeline = EventPipeline(
                    root / "student" / "events.db", root / "student" / "shots",
                    metadata={"student_name": "Артём"}, live=reporter,
                )
                pipeline.start()
                pipeline.submit(ProctorEvent.create("phone_aimed_at_screen", source="test"))
                self.assertTrue(_wait(lambda: server.archived.is_set()))
                pipeline.stop()
                self.assertTrue(_wait(
                    lambda: (s := load_sessions(teacher_db)) and s[0].ended_at is not None
                ))
                session = load_sessions(teacher_db)[0]
                self.assertEqual(session.student_name, "Артём")
                self.assertEqual(len(session.events), 1)

                with urllib.request.urlopen(base + "/", timeout=5) as response:
                    index = response.read().decode("utf-8")
                self.assertIn("Журнал сессий", index)
                self.assertIn(f"/report/{session.id}", index)
                with urllib.request.urlopen(base + f"/report/{session.id}", timeout=10) as response:
                    report = response.read().decode("utf-8")
                self.assertIn("Артём", report)
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(base + "/report/missing", timeout=5)
                self.assertEqual(caught.exception.code, 404)
            finally:
                server.stop()

    def test_token_protects_journal_and_links_carry_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = SessionArchive(root / "proctoring.db")
            for message in _messages():
                archive.record(message)
            server = LiveServer(LiveBoard(), host="127.0.0.1", port=0, token="k e y",
                                archive=archive)
            server.start()
            try:
                base = f"http://127.0.0.1:{server.port}"
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(base + "/report/s1", timeout=5)
                with urllib.request.urlopen(base + "/?token=k%20e%20y", timeout=5) as response:
                    index = response.read().decode("utf-8")
                self.assertIn("/report/s1?token=k%20e%20y", index)
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
