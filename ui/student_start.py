"""Consent and student name dialog shown before the test starts."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.student import STUDENT_NAME_MAX_LENGTH, normalize_student_name


CONSENT_TEXT = (
    "Во время теста работает прокторинг с камерой. Видео обрабатывается на этом "
    "компьютере и никуда не передаётся. Сохраняются только отдельные кадры "
    "для преподавателя."
)
RULES_TEXT = (
    "Во время теста нельзя пользоваться телефоном и другими устройствами, "
    "открывать другие программы и переключать окна. В кадре должен быть "
    "только один человек."
)


class StudentStartDialog(QDialog):
    """Collects the student's name and consent; the test opens only after both."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Proctoring · Начало теста")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.setStyleSheet(
            """
            QDialog { background: #0b1020; color: #e8edf7; }
            QLabel { color: #e8edf7; font-size: 14px; }
            QLabel#heading { font-size: 22px; font-weight: 700; }
            QLabel#hint { color: #9aa8c4; font-size: 13px; }
            QLineEdit { background: #17223a; color: #e8edf7; border: 1px solid #2a3a5b;
                        border-radius: 8px; padding: 10px; font-size: 15px; }
            QLineEdit:focus { border-color: #5d7df6; }
            QCheckBox { color: #e8edf7; font-size: 14px; }
            QPushButton { background: #5d7df6; color: white; border: 0;
                          border-radius: 9px; padding: 11px 18px; font-weight: 700; }
            QPushButton:disabled { background: #34415e; color: #8290aa; }
            QPushButton#secondary { background: #202d49; }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(14)

        heading = QLabel("Перед началом теста")
        heading.setObjectName("heading")
        layout.addWidget(heading)

        consent = QLabel(CONSENT_TEXT)
        consent.setWordWrap(True)
        layout.addWidget(consent)

        rules = QLabel(RULES_TEXT)
        rules.setObjectName("hint")
        rules.setWordWrap(True)
        layout.addWidget(rules)

        layout.addWidget(QLabel("ФИО или ID студента"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Например: Иванов Иван, группа ИС-21")
        self.name_edit.setMaxLength(STUDENT_NAME_MAX_LENGTH)
        layout.addWidget(self.name_edit)

        self.consent_box = QCheckBox("Я ознакомлен(а) с условиями и согласен(на) на прокторинг")
        layout.addWidget(self.consent_box)

        buttons = QHBoxLayout()
        self.cancel_button = QPushButton("Выйти")
        self.cancel_button.setObjectName("secondary")
        self.cancel_button.clicked.connect(self.reject)
        self.start_button = QPushButton("Начать тест")
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self._accept_if_ready)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        buttons.addWidget(self.start_button)
        layout.addLayout(buttons)

        self.name_edit.textChanged.connect(self._refresh)
        self.consent_box.toggled.connect(self._refresh)
        self._refresh()
        self.name_edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def student_name(self) -> str | None:
        return normalize_student_name(self.name_edit.text())

    def is_ready(self) -> bool:
        return self.student_name() is not None and self.consent_box.isChecked()

    def _refresh(self, *_: object) -> None:
        self.start_button.setEnabled(self.is_ready())

    def _accept_if_ready(self) -> None:
        if self.is_ready():
            self.accept()
