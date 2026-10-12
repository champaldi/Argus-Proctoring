"""Окно теста на весь экран без рамки, пока идёт экзамен.

Защита окружения разворачивает окно один раз, но у развёрнутого окна
остаются рамка и кнопка «Восстановить»: студент мог уменьшить окно и
открыть рядом что-то ещё. Пока замок включён, окно показано в полноэкранном
режиме (без рамки и кнопок), а любая смена состояния — свернуть, восстановить,
Win+стрелка — сразу возвращает его на весь экран.

Выключить для отладки: ``PROCTOR_FULLSCREEN=0``.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import QWidget


def fullscreen_enabled() -> bool:
    return os.getenv("PROCTOR_FULLSCREEN", "1").strip() != "0"


class WindowLock(QObject):
    """Держит окно на весь экран, пока замок включён."""

    def __init__(self, window: QWidget) -> None:
        super().__init__(window)
        self.window = window
        self.active = False
        self.restored = 0  # сколько раз окно пришлось вернуть на весь экран
        window.installEventFilter(self)

    def engage(self) -> None:
        self.active = True
        self.window.showFullScreen()

    def release(self) -> None:
        self.active = False
        if self.window.isFullScreen():
            self.window.showNormal()

    def _is_fullscreen(self) -> bool:
        state = self.window.windowState()
        return bool(state & Qt.WindowState.WindowFullScreen) and not bool(
            state & Qt.WindowState.WindowMinimized
        )

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if (
            watched is self.window
            and self.active
            and event.type() == QEvent.Type.WindowStateChange
            and not self._is_fullscreen()
        ):
            # Не внутри обработчика: Qt ещё применяет новое состояние.
            QTimer.singleShot(0, self._restore)
        return False

    def _restore(self) -> None:
        if not self.active or self._is_fullscreen():
            return
        self.restored += 1
        self.window.showFullScreen()
        self.window.raise_()
        self.window.activateWindow()
