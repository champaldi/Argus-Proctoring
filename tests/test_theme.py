import re
import unittest

from ui import theme


class ThemeTests(unittest.TestCase):
    def test_titles_carry_the_product_name(self):
        self.assertEqual(theme.window_title("Начало теста"), f"{theme.APP_NAME} · Начало теста")
        self.assertTrue(theme.APP_NAME.strip())

    def test_style_sheets_are_well_formed(self):
        for sheet in (theme.MAIN_WINDOW_STYLE, theme.START_DIALOG_STYLE):
            self.assertEqual(sheet.count("{"), sheet.count("}"))
            self.assertNotIn("{{", sheet)
            for body in re.findall(r"\{([^{}]*)\}", sheet):
                for declaration in filter(str.strip, body.split(";")):
                    name, _, value = declaration.partition(":")
                    self.assertRegex(name.strip(), r"^[a-z-]+$")
                    self.assertTrue(value.strip(), declaration)
                    for colour in re.findall(r"#\w+", value):
                        self.assertRegex(colour, r"^#[0-9a-fA-F]{6}$")

    def test_main_window_styles_every_named_widget(self):
        for selector in ("QWidget#page", "QFrame#card", "QLabel#eyebrow", "QLabel#heading",
                         "QLabel#question", "QLabel#status", "QRadioButton:checked",
                         "QPushButton#secondary"):
            self.assertIn(selector, theme.MAIN_WINDOW_STYLE)

    def test_watermark_colour_is_a_dark_rgb_triplet(self):
        self.assertEqual(len(theme.WATERMARK_RGB), 3)
        self.assertTrue(all(0 <= channel <= 90 for channel in theme.WATERMARK_RGB))


if __name__ == "__main__":
    unittest.main()
