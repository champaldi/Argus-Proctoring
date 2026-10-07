import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

from ui.watermark import WatermarkOverlay


class WatermarkOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.host = QWidget()
        self.host.resize(800, 500)
        layout = QVBoxLayout(self.host)
        self.button = QPushButton("Далее")
        layout.addWidget(self.button)
        self.clicks = 0
        self.button.clicked.connect(self._clicked)
        self.overlay = WatermarkOverlay(self.host, lambda: "Иванов Иван · 01234567")
        self.host.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.host.close()
        self.host.deleteLater()

    def _clicked(self) -> None:
        self.clicks += 1

    def test_overlay_covers_the_host_and_follows_its_size(self) -> None:
        self.assertEqual(self.overlay.size(), self.host.size())
        self.host.resize(1000, 640)
        self.app.processEvents()
        self.assertEqual(self.overlay.size(), self.host.size())

    def test_clicks_reach_the_controls_under_the_overlay(self) -> None:
        centre = self.button.mapTo(self.host, self.button.rect().center())
        target = self.host.childAt(centre)
        self.assertIs(target, self.button)
        QTest.mouseClick(self.button, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
        self.assertEqual(self.clicks, 1)

    def test_overlay_paints_text_and_survives_empty_text(self) -> None:
        image = QImage(self.overlay.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        self.overlay.render(image)
        painted = any(
            image.pixelColor(x, y).alpha() > 0
            for x in range(0, image.width(), 4)
            for y in range(0, image.height(), 4)
        )
        self.assertTrue(painted)
        empty = WatermarkOverlay(self.host, lambda: "")
        blank = QImage(empty.size(), QImage.Format.Format_ARGB32)
        blank.fill(Qt.GlobalColor.transparent)
        empty.render(blank)


if __name__ == "__main__":
    unittest.main()
