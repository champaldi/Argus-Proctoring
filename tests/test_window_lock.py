import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from ui.window_lock import WindowLock  # noqa: E402


def _settle() -> None:
    for _ in range(5):
        QCoreApplication.processEvents()


class WindowLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = QWidget()
        self.window.show()
        self.lock = WindowLock(self.window)

    def tearDown(self) -> None:
        self.lock.release()
        self.window.close()
        self.window.deleteLater()
        _settle()

    def test_engage_makes_window_fullscreen(self) -> None:
        self.lock.engage()
        _settle()
        self.assertTrue(self.window.isFullScreen())

    def test_student_cannot_restore_or_minimize_while_locked(self) -> None:
        self.lock.engage()
        _settle()
        for leave in (self.window.showNormal, self.window.showMaximized, self.window.showMinimized):
            leave()
            _settle()
            self.assertTrue(self.window.isFullScreen(), leave.__name__)
            self.assertFalse(self.window.isMinimized(), leave.__name__)
        self.assertEqual(self.lock.restored, 3)

    def test_release_gives_the_window_back(self) -> None:
        self.lock.engage()
        _settle()
        self.lock.release()
        _settle()
        self.assertFalse(self.window.isFullScreen())
        self.window.showMaximized()
        _settle()
        self.assertFalse(self.window.isFullScreen())
        self.assertEqual(self.lock.restored, 0)

    def test_lock_is_off_until_engaged(self) -> None:
        self.window.showMaximized()
        _settle()
        self.assertFalse(self.window.isFullScreen())


if __name__ == "__main__":
    unittest.main()
