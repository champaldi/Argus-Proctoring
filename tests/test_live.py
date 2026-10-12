"""Живая отправка событий студента и приём на панели преподавателя."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from core import live as live_module
from core.live import LiveReporter, normalize_url, teacher_url_from_env
from core.pipeline import EventPipeline
from core.storage import StoredEvent
from events import ProctorEvent
from teacher.live import INFO, LiveBoard, LiveServer, render_page
from teacher.rules import SUSPICIOUS, VIOLATION


def _stored(kind: str, risk: float = 10.0, weight: float = 10.0, **kwargs) -> StoredEvent:
    return StoredEvent(
        ProctorEvent.create(kind, source="test", **kwargs),
        weight=weight, risk_total=risk, screenshot_path=None,
    )


def _message(kind: str, session: str = "s1", **extra) -> dict:
    return {"kind": kind, "session_id": session, "student": "Карим",
            "started_at": "2026-10-16T10:00:00+00:00", **extra}


def _event_message(stored: StoredEvent, session: str = "s1", **extra) -> dict:
    return _message("event", session, event=stored.event.to_dict(),
                    weight=stored.weight, risk_total=stored.risk_total, **extra)


def _wait(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class NormalizeUrlTests(unittest.TestCase):
    def test_bare_address_gets_scheme_and_port(self):
        self.assertEqual(normalize_url(" 192.168.1.5 "), "http://192.168.1.5:8765")

    def test_explicit_port_and_trailing_slash(self):
        self.assertEqual(normalize_url("http://10.0.0.2:9000/"), "http://10.0.0.2:9000")

    def test_empty_means_disabled(self):
        self.assertEqual(normalize_url("   "), "")

    def test_address_file_is_used_when_env_is_empty(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(os.environ, {"PROCTOR_TEACHER_URL": ""}):
            data = Path(temporary)
            self.assertEqual(teacher_url_from_env(data), "")
            (data / "teacher_url.txt").write_text("100.64.0.7\n", encoding="utf-8")
            self.assertEqual(teacher_url_from_env(data), "http://100.64.0.7:8765")


class LiveBoardTests(unittest.TestCase):
    def test_single_glance_is_not_an_alert(self):
        board = LiveBoard()
        board.apply(_message("start"))
        alerts = board.apply(_event_message(_stored("gaze_side")))
        self.assertEqual(alerts, [])
        self.assertEqual(board.sessions()[0].risk, 10.0)

    def test_phone_aimed_at_screen_raises_violation_once(self):
        board = LiveBoard()
        board.apply(_message("start"))
        first = board.apply(_event_message(_stored("phone_aimed_at_screen", 45), thumbnail="AAA"))
        self.assertEqual([alert.rule_id for alert in first], ["V1"])
        self.assertEqual(first[0].level, VIOLATION)
        self.assertEqual(first[0].thumbnail, "AAA")
        self.assertIn("Карим", first[0].text)
        again = board.apply(_event_message(_stored("phone_aimed_at_screen", 90)))
        self.assertEqual(again, [])
        session = board.sessions()[0]
        self.assertEqual(session.risk, 90)
        self.assertEqual(session.worst_level, VIOLATION)
        self.assertIn("V1", session.rule_summary)

    def test_repeated_delivery_is_ignored(self):
        board = LiveBoard()
        stored = _stored("phone_detected")
        self.assertEqual([a.rule_id for a in board.apply(_event_message(stored))], ["S1"])
        self.assertEqual(board.apply(_event_message(stored)), [])

    def test_suspicious_then_violation_from_combination(self):
        board = LiveBoard()
        phone = board.apply(_event_message(_stored("phone_detected")))
        self.assertEqual([(a.rule_id, a.level) for a in phone], [("S1", SUSPICIOUS)])
        combined = board.apply(_event_message(_stored("gaze_down", 30)))
        self.assertEqual([a.rule_id for a in combined], ["V2"])

    def test_events_before_start_create_the_session(self):
        board = LiveBoard()
        board.apply(_event_message(_stored("gaze_down")))
        self.assertEqual(board.sessions()[0].display_name, "Карим")

    def test_finish_and_status(self):
        board = LiveBoard()
        board.apply(_message("start"))
        alerts = board.apply(_message("finish", final_risk=72))
        self.assertEqual(alerts[0].level, INFO)
        self.assertIn("72", alerts[0].text)
        self.assertEqual(board.sessions()[0].status(), "Завершил")

    def test_unknown_event_type_keeps_connection(self):
        board = LiveBoard()
        message = _message("event", event={"type": "from_the_future", "source": "x"})
        self.assertEqual(board.apply(message), [])
        self.assertEqual(len(board.sessions()), 1)

    def test_bad_message_is_rejected(self):
        with self.assertRaises(ValueError):
            LiveBoard().apply({"kind": "hack", "session_id": "x"})

    def test_page_lists_students_and_escapes_names(self):
        board = LiveBoard()
        board.apply({**_message("start"), "student": "<b>Карим</b>"})
        board.apply(_event_message(_stored("phone_aimed_at_screen", 45)))
        page = render_page(board)
        self.assertIn("&lt;b&gt;Карим&lt;/b&gt;", page)
        self.assertIn("Фотографирование экрана", page)


class ReporterTests(unittest.TestCase):
    def test_messages_are_retried_in_order_after_network_failure(self):
        sent: list[str] = []
        failures = {"left": 2}

        def sender(url, payload, token):
            if failures["left"]:
                failures["left"] -= 1
                raise OSError("offline")
            sent.append(payload["kind"])

        with patch.object(live_module, "RETRY_MIN", 0.01), \
                patch.object(live_module, "RETRY_MAX", 0.02):
            reporter = LiveReporter("127.0.0.1", sender=sender, thumbnailer=lambda _: None)
            reporter.start()
            reporter.session_started("s1", {"student_name": "Карим"})
            reporter.event_recorded("s1", _stored("gaze_down"))
            reporter.session_finished("s1", 10)
            self.assertTrue(_wait(lambda: len(sent) == 3))
            reporter.close()
        self.assertEqual(sent, ["start", "event", "finish"])
        self.assertTrue(reporter.connected)

    def test_overflow_drops_events_not_session_boundaries(self):
        reporter = LiveReporter("127.0.0.1", sender=lambda *_: None)
        with patch.object(live_module, "MAX_QUEUE", 3):
            reporter.session_started("s1", {})
            for _ in range(5):
                reporter.event_recorded("s1", _stored("gaze_down"))
            reporter.session_finished("s1", 1)
        kinds = [message["kind"] for message in reporter._queue]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "finish")
        self.assertEqual(len(kinds), 3)
        self.assertGreater(reporter.dropped, 0)

    def test_close_does_not_hang_when_teacher_is_unreachable(self):
        def sender(*_):
            raise OSError("offline")

        reporter = LiveReporter("127.0.0.1", sender=sender)
        reporter.start()
        reporter.session_started("s1", {})
        self.assertTrue(_wait(lambda: reporter.connected is False))
        started = time.monotonic()
        reporter.close(timeout=1.0)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertFalse(reporter.connected)


class ImageFallbackTests(unittest.TestCase):
    def test_frame_is_dropped_after_repeated_failures(self):
        attempts = []

        def sender(url, payload, token):
            attempts.append("thumbnail" in payload)
            if "thumbnail" in payload:
                raise TimeoutError("timed out")

        with patch.object(live_module, "RETRY_MIN", 0.01), \
                patch.object(live_module, "RETRY_MAX", 0.02):
            reporter = LiveReporter("127.0.0.1", sender=sender, thumbnailer=lambda _: "AAAA")
            reporter.start()
            reporter.event_recorded("s1", StoredEvent(
                ProctorEvent.create("phone_aimed_at_screen", source="t"), 1, 1, "shot.jpg"))
            self.assertTrue(_wait(lambda: reporter.pending() == 0))
            reporter.close()
        self.assertEqual(attempts, [True, True, False])
        self.assertTrue(reporter.connected)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.board = LiveBoard()
        self.server = LiveServer(self.board, host="127.0.0.1", port=0, token="secret")
        self.server.start()
        self.base = f"http://127.0.0.1:{self.server.port}"

    def tearDown(self):
        self.server.stop()

    def test_health_and_token(self):
        with urllib.request.urlopen(self.base + "/api/health", timeout=3) as response:
            self.assertTrue(json.load(response)["ok"])
        request = urllib.request.Request(self.base + "/api/live", data=b"{}", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=3)
        self.assertEqual(caught.exception.code, 403)

    def test_pipeline_sends_session_to_teacher(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reporter = LiveReporter(self.base, token="secret", thumbnailer=lambda _: None)
            pipeline = EventPipeline(
                root / "events.db", root / "shots",
                metadata={"student_name": "Артём"}, live=reporter,
            )
            pipeline.start()
            pipeline.submit(ProctorEvent.create("phone_aimed_at_screen", source="test"))
            self.assertTrue(_wait(lambda: any(
                alert.rule_id == "V1" for alert in self.board.feed()
            )))
            pipeline.stop()
        self.assertTrue(_wait(lambda: self.board.sessions()[0].finished))
        alerts = self.server.drain()
        self.assertEqual(
            [alert.title for alert in alerts],
            ["Начал тест", "Фотографирование экрана", "Завершил тест"],
        )
        self.assertEqual(alerts[1].student, "Артём")
        with urllib.request.urlopen(self.base + "/?token=secret", timeout=3) as response:
            self.assertIn("Артём", response.read().decode("utf-8"))

    def test_concurrent_students(self):
        reporters = [
            LiveReporter(self.base, token="secret", thumbnailer=lambda _: None)
            for _ in range(5)
        ]
        threads = []
        for index, reporter in enumerate(reporters):
            reporter.start()

            def run(rep=reporter, number=index):
                rep.session_started(f"s{number}", {"student_name": f"Студент {number}"})
                for _ in range(10):
                    rep.event_recorded(f"s{number}", _stored("gaze_side"))

            threads.append(threading.Thread(target=run))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertTrue(_wait(lambda: all(r.pending() == 0 for r in reporters)))
        for reporter in reporters:
            reporter.close()
        self.assertEqual(len(self.board.sessions()), 5)


if __name__ == "__main__":
    unittest.main()
