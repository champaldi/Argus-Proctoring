import unittest
from datetime import datetime

from core.watermark import WATERMARK_FALLBACK_NAME, watermark_text


class WatermarkTextTests(unittest.TestCase):
    def test_text_names_student_session_and_time(self) -> None:
        text = watermark_text(
            "  Иванов   Иван ", "0123456789abcdef", datetime(2026, 10, 7, 9, 5)
        )
        self.assertEqual(text, "Иванов Иван · 01234567 · 07.10.2026 09:05")

    def test_missing_name_uses_a_neutral_word(self) -> None:
        for name in (None, "", "   "):
            text = watermark_text(name, "abcdef0123", datetime(2026, 10, 7, 9, 5))
            self.assertTrue(text.startswith(f"{WATERMARK_FALLBACK_NAME} · abcdef01"))

    def test_current_time_is_used_by_default(self) -> None:
        self.assertIn(datetime.now().strftime("%d.%m.%Y"), watermark_text("Аня", "s1"))


if __name__ == "__main__":
    unittest.main()
