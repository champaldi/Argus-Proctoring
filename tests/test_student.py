import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.pipeline import EventPipeline
from core.student import (
    STUDENT_NAME_ENV,
    STUDENT_NAME_MAX_LENGTH,
    normalize_student_name,
    session_metadata,
    student_name_from_env,
)
from teacher.data import load_sessions


class StudentNameTests(unittest.TestCase):
    def test_name_is_trimmed_and_inner_whitespace_collapsed(self) -> None:
        self.assertEqual(normalize_student_name("  Иванов   Иван \n"), "Иванов Иван")

    def test_empty_or_too_short_name_is_rejected(self) -> None:
        for value in (None, "", "   ", "А"):
            self.assertIsNone(normalize_student_name(value))

    def test_long_name_is_limited(self) -> None:
        name = normalize_student_name("я" * 500)
        self.assertEqual(len(name), STUDENT_NAME_MAX_LENGTH)

    def test_metadata_contains_only_a_valid_name(self) -> None:
        self.assertEqual(session_metadata(" Аня  Ким "), {"student_name": "Аня Ким"})
        self.assertEqual(session_metadata(None), {})
        self.assertEqual(session_metadata(" "), {})

    def test_name_can_be_preset_for_automated_runs(self) -> None:
        with patch.dict(os.environ, {STUDENT_NAME_ENV: "  Демо Студент "}):
            self.assertEqual(student_name_from_env(), "Демо Студент")
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(student_name_from_env())


class SessionNameStorageTests(unittest.TestCase):
    def test_session_carries_the_student_name_to_the_teacher_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "events.db"
            pipeline = EventPipeline(
                database,
                root / "screenshots",
                metadata=session_metadata("Иванов Иван"),
            )
            pipeline.start()
            session_id = pipeline.session_id
            pipeline.stop()

            connection = sqlite3.connect(database)
            try:
                raw = connection.execute(
                    "SELECT metadata_json FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(
                json.loads(raw),
                {"application": "proctoring", "student_name": "Иванов Иван"},
            )
            self.assertEqual(load_sessions(database)[0].student_name, "Иванов Иван")

    def test_session_without_a_name_keeps_the_previous_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = EventPipeline(root / "events.db", root / "screenshots")
            pipeline.start()
            pipeline.stop()
            session = load_sessions(root / "events.db")[0]
            self.assertEqual(dict(session.metadata), {"application": "proctoring"})


if __name__ == "__main__":
    unittest.main()


class StudentViewTests(unittest.TestCase):
    def test_student_view_is_the_default_and_zero_turns_it_off(self):
        from core.student import student_view_enabled

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROCTOR_STUDENT_VIEW", None)
            self.assertTrue(student_view_enabled())
        with patch.dict(os.environ, {"PROCTOR_STUDENT_VIEW": " 0 "}):
            self.assertFalse(student_view_enabled())
        with patch.dict(os.environ, {"PROCTOR_STUDENT_VIEW": "1"}):
            self.assertTrue(student_view_enabled())

