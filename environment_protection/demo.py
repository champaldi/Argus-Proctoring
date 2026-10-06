"""Manual Windows check: python -m environment_protection.demo --seconds 60."""

from __future__ import annotations

import argparse
import json
import queue

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import ProtectionConfig, disable, enable, status


class DemoWindow(QWidget):
    def __init__(self, seconds=60.0):
        super().__init__()
        self.config = ProtectionConfig(max_seconds=seconds)
        self.incoming = queue.Queue()
        self.setWindowTitle("Проверка защиты окружения")
        self.resize(780, 500)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Нажмите «Включить», затем проверьте Alt+Tab, Win, Ctrl+C/V и PrtScn.\n"
            f"Аварийное отключение: Ctrl+Alt+F12. Автоотключение: {seconds:g} с.\n"
            "Обычный ввод и кнопка выключения остаются доступны."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.start_button = QPushButton("Включить защиту")
        self.start_button.clicked.connect(self.start_protection)
        layout.addWidget(self.start_button)
        stop_button = QPushButton("ВЫКЛЮЧИТЬ ЗАЩИТУ")
        stop_button.setMinimumHeight(48)
        stop_button.clicked.connect(self.stop_protection)
        layout.addWidget(stop_button)
        self.status_label = QLabel("Защита выключена")
        layout.addWidget(self.status_label)
        self.typing = QPlainTextEdit()
        self.typing.setPlaceholderText("Поле для проверки обычного ввода и Ctrl+C/V")
        layout.addWidget(self.typing)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(200)
        layout.addWidget(self.log)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

    def start_protection(self):
        try:
            enable(self.incoming.put_nowait, hwnd=int(self.winId()), config=self.config)
        except Exception as exc:
            self.log.appendPlainText(f"Не удалось включить защиту: {exc}")
        self.refresh()

    def stop_protection(self):
        disable()
        self.refresh()

    def refresh(self):
        state = status()
        self.start_button.setEnabled(not state["enabled"])
        label = "Защита включена" if state["enabled"] else "Защита выключена"
        self.status_label.setText(f"{label} · {state['reason']}")
        if state["last_error"]:
            self.status_label.setText(self.status_label.text() + f" · {state['last_error']}")
        while True:
            try:
                event = self.incoming.get_nowait()
            except queue.Empty:
                break
            self.log.appendPlainText(json.dumps(event, ensure_ascii=False))

    def closeEvent(self, event):
        self.timer.stop()
        disable()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    window = DemoWindow(args.seconds)
    window.show()
    app.aboutToQuit.connect(disable)
    try:
        return app.exec()
    finally:
        disable()


if __name__ == "__main__":
    raise SystemExit(main())
