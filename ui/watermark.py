"""Semi-transparent watermark repeated across the test window."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget


class WatermarkOverlay(QWidget):
    """Draws the student's name, session and time over everything in ``parent``.

    The overlay ignores the mouse, so the test underneath works as before. It is
    faint enough to read through, yet visible on any photo of the screen.
    """

    def __init__(
        self,
        parent: QWidget,
        text_source: Callable[[], str],
        *,
        opacity: float = 0.18,
        angle: float = -24.0,
        refresh_ms: int = 30_000,
    ) -> None:
        super().__init__(parent)
        self._text_source = text_source
        # Mid grey stays readable on both dark and light themes.
        self._color = QColor(140, 150, 165)
        self._color.setAlphaF(max(0.02, min(0.5, opacity)))
        self._angle = angle
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setStyleSheet("background: transparent;")
        parent.installEventFilter(self)
        self.setGeometry(parent.rect())
        self.raise_()
        self.show()
        # The time in the text changes, so repaint from time to time.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(refresh_ms)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.parent() and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.parentWidget().rect())
            self.raise_()
        return False

    def paintEvent(self, event: QPaintEvent) -> None:
        text = self._text_source()
        if not text:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        font = QFont(self.font())
        font.setPixelSize(20)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(self._color)
        metrics = QFontMetrics(font)
        step_x = metrics.horizontalAdvance(text) + 140
        step_y = 150
        # Rotate around the centre and cover the whole diagonal with rows.
        reach = int((self.width() ** 2 + self.height() ** 2) ** 0.5)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self._angle)
        row = 0
        y = -reach
        while y <= reach:
            x = -reach - (step_x // 2 if row % 2 else 0)
            while x <= reach:
                painter.drawText(x, y, text)
                x += step_x
            y += step_y
            row += 1
        painter.end()
