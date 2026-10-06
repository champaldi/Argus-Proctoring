import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog

from config import AppConfig
from ui import main_window
from ui.student_start import StudentStartDialog


class StudentStartDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_start_requires_both_name_and_consent(self) -> None:
        dialog = StudentStartDialog()
        try:
            self.assertFalse(dialog.start_button.isEnabled())
            dialog.name_edit.setText("  Иванов   Иван ")
            self.assertFalse(dialog.start_button.isEnabled())
            dialog.consent_box.setChecked(True)
            self.assertTrue(dialog.start_button.isEnabled())
            self.assertEqual(dialog.student_name(), "Иванов Иван")
            dialog.name_edit.setText(" ")
            self.assertFalse(dialog.start_button.isEnabled())
        finally:
            dialog.deleteLater()

    def test_start_button_accepts_only_when_ready(self) -> None:
        dialog = StudentStartDialog()
        try:
            dialog._accept_if_ready()
            self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
            dialog.name_edit.setText("Аня Ким")
            dialog.consent_box.setChecked(True)
            dialog._accept_if_ready()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        finally:
            dialog.deleteLater()

    def test_declined_dialog_starts_no_test_window(self) -> None:
        with (
            patch.dict(os.environ, {"PROCTOR_STUDENT_NAME": ""}),
            patch("ui.student_start.StudentStartDialog.exec",
                  return_value=QDialog.DialogCode.Rejected),
            patch.object(main_window, "MainWindow") as window,
        ):
            self.assertEqual(main_window.run_application(AppConfig.from_env()), 0)
        window.assert_not_called()


if __name__ == "__main__":
    unittest.main()
