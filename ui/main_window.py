"""Main test window and camera worker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QEventLoop, QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from config import AppConfig, RISK_RED_ABOVE, RISK_YELLOW_FROM
from core.detectors import DetectorCollection, ModuleStatus
from core.pipeline import EventPipeline
from core.event_presentation import EVENT_LABELS, application_details
from core.security import SecurityAdapter
from core.student import session_metadata, student_name_from_env
from core.watermark import watermark_text
from core.storage import (
    StoredEvent,
    load_session_events,
    record_teacher_verdict,
)
from events import ProctorEvent
from ui.camera_worker import CameraWorker


@dataclass(frozen=True, slots=True)
class Question:
    text: str
    answers: tuple[str, ...]
    correct_index: int


QUESTIONS = (
    Question(
        "Какой протокол используется для защищённой передачи веб-страниц?",
        ("FTP", "HTTPS", "SMTP", "SSH"),
        1,
    ),
    Question(
        "Какая структура данных работает по принципу LIFO?",
        ("Очередь", "Граф", "Стек", "Хеш-таблица"),
        2,
    ),
    Question(
        "Что вернёт len({1, 1, 2, 3}) в Python?",
        ("3", "4", "2", "Ошибка"),
        0,
    ),
    Question(
        "Какой компонент хранит события сессии в этом проекте?",
        ("YOLO", "SQLite", "MediaPipe", "Qt Style Sheets"),
        1,
    ),
)


class UiBridge(QObject):
    pipeline_error = Signal(str)


class TeacherReviewDialog(QDialog):
    """End-of-session evidence review for the teacher."""

    def __init__(
        self,
        *,
        database_path: Any,
        session_id: str,
        events: list[StoredEvent],
        final_risk: float,
        test_score: tuple[int, int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.database_path = database_path
        self.session_id = session_id
        self.setWindowTitle("Итоги сессии · Проверка преподавателем")
        self.resize(900, 720)
        self.setModal(True)
        self.setStyleSheet(
            """
            QDialog, QWidget { background: #0b1020; color: #e8edf7; }
            QFrame#reviewCard { background: #121a2d; border: 1px solid #26334f;
                                border-radius: 12px; }
            QLabel#reviewHeading { font-size: 24px; font-weight: 700; }
            QPushButton { border: 0; border-radius: 9px; padding: 12px 18px;
                          color: white; font-weight: 700; }
            QPushButton#cheated { background: #c94f5d; }
            QPushButton#notCheated { background: #279b78; }
            QScrollArea { border: 0; }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(14)
        heading = QLabel("Итоги сессии")
        heading.setObjectName("reviewHeading")
        root.addWidget(heading)

        score = QLabel(f"Результат теста: {test_score[0]} из {test_score[1]}")
        root.addWidget(score)
        risk = QLabel(f"Итоговый уровень риска: {final_risk:.1f} из 100")
        risk.setStyleSheet(
            f"font-size:18px; font-weight:700; color:{self._risk_color(final_risk)};"
        )
        root.addWidget(risk)

        timeline_title = QLabel(f"Таймлайн нарушений · {len(events)} событий")
        timeline_title.setStyleSheet("font-size:16px; font-weight:700;")
        root.addWidget(timeline_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        timeline = QWidget()
        timeline_layout = QVBoxLayout(timeline)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(10)
        if not events:
            empty = QLabel("Нарушения не зафиксированы")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            timeline_layout.addWidget(empty)
        for stored in events:
            timeline_layout.addWidget(self._event_card(stored))
        timeline_layout.addStretch(1)
        scroll.setWidget(timeline)
        root.addWidget(scroll, 1)

        actions = QHBoxLayout()
        cheated = QPushButton("Списывал")
        cheated.setObjectName("cheated")
        cheated.clicked.connect(lambda: self._save_verdict("cheated"))
        not_cheated = QPushButton("Не списывал")
        not_cheated.setObjectName("notCheated")
        not_cheated.clicked.connect(lambda: self._save_verdict("not_cheated"))
        actions.addWidget(cheated)
        actions.addWidget(not_cheated)
        root.addLayout(actions)

    @staticmethod
    def _risk_color(value: float) -> str:
        if value > RISK_RED_ABOVE:
            return "#ef6262"
        if value >= RISK_YELLOW_FROM:
            return "#f0c45d"
        return "#55d6a9"

    def _event_card(self, stored: StoredEvent) -> QFrame:
        card = QFrame()
        card.setObjectName("reviewCard")
        row = QHBoxLayout(card)
        row.setContentsMargins(12, 12, 12, 12)
        preview = QLabel("Без снимка")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setFixedSize(180, 105)
        preview.setStyleSheet("background:#070b15; border-radius:8px; color:#70809e;")
        if stored.screenshot_path:
            pixmap = QPixmap(stored.screenshot_path)
            if not pixmap.isNull():
                preview.setText("")
                preview.setPixmap(
                    pixmap.scaled(
                        preview.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        row.addWidget(preview)

        text = QVBoxLayout()
        event_name = stored.event.type.value
        title = QLabel(EVENT_LABELS.get(event_name, event_name))
        title.setStyleSheet("font-size:16px; font-weight:700;")
        text.addWidget(title)
        for line in application_details(stored.event):
            detail = QLabel(line)
            detail.setTextFormat(Qt.TextFormat.PlainText)
            detail.setWordWrap(True)
            detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            text.addWidget(detail)
        local_time = stored.event.occurred_at.astimezone().strftime("%H:%M:%S")
        text.addWidget(QLabel(f"Время: {local_time} · Источник: {stored.event.source}"))
        text.addWidget(
            QLabel(
                f"Добавлено: +{stored.weight:g} · "
                f"уровень после события: {stored.risk_total:.1f}"
            )
        )
        if stored.event.confidence is not None:
            text.addWidget(QLabel(f"Уверенность модели: {stored.event.confidence:.0%}"))
        text.addStretch(1)
        row.addLayout(text, 1)
        return card

    def _save_verdict(self, verdict: str) -> None:
        try:
            record_teacher_verdict(self.database_path, self.session_id, verdict)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Не удалось сохранить решение",
                f"{type(exc).__name__}: {exc}",
            )
            return
        self.accept()


class MainWindow(QMainWindow):
    shutdown_finished = Signal()

    def __init__(self, config: AppConfig, student_name: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.student_name = student_name
        self.answers: dict[int, int] = {}
        self.question_index = 0
        self._shutdown_done = False
        self._shutdown_started = False
        self._close_requested = False
        self._review_score: tuple[int, int] | None = None
        self._shutdown_timer = QTimer(self)
        self._shutdown_timer.setInterval(25)
        self._shutdown_timer.timeout.connect(self._continue_shutdown)

        self.bridge = UiBridge()
        self.bridge.pipeline_error.connect(self._show_runtime_error)
        self.pipeline = EventPipeline(
            config.database_path,
            config.screenshots_dir,
            on_error=self.bridge.pipeline_error.emit,
            metadata=session_metadata(student_name),
        )
        self.pipeline.start()

        self.detectors = DetectorCollection(config.phone_module, config.gaze_module)
        self.security = SecurityAdapter(config.security_module, self._submit_security_event)
        self.camera_thread: QThread | None = None
        self.camera_worker: CameraWorker | None = None

        self._build_ui()
        self._render_question()
        QTimer.singleShot(0, self.start_monitoring)

    def _build_ui(self) -> None:
        self.setWindowTitle("Proctoring · Контроль тестирования")
        self.resize(1280, 780)
        self.setMinimumSize(1050, 680)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b1020; color: #e8edf7; }
            QFrame#card { background: #121a2d; border: 1px solid #26334f;
                          border-radius: 14px; }
            QLabel#eyebrow { color: #7f8fae; font-size: 11px; font-weight: 700; }
            QLabel#heading { font-size: 25px; font-weight: 700; }
            QLabel#question { font-size: 19px; font-weight: 600; }
            QLabel#status { padding: 7px 10px; background: #1c2944;
                            border-radius: 8px; }
            QRadioButton { background: #17223a; border: 1px solid #2a3a5b;
                           border-radius: 10px; padding: 13px; font-size: 14px; }
            QRadioButton:hover { border-color: #5d7df6; }
            QPushButton { background: #5d7df6; color: white; border: 0;
                          border-radius: 9px; padding: 11px 18px; font-weight: 700; }
            QPushButton:hover { background: #7290ff; }
            QPushButton:disabled { background: #34415e; color: #8290aa; }
            QPushButton#secondary { background: #202d49; }
            """
        )

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(18)
        root.addWidget(self._build_test_panel(), 3)
        root.addWidget(self._build_monitor_panel(), 2)
        self.setCentralWidget(central)
        if os.getenv("PROCTOR_WATERMARK", "1").strip() != "0":
            # Imported here so the window still opens if the overlay is unavailable.
            from ui.watermark import WatermarkOverlay

            self.watermark = WatermarkOverlay(
                central,
                lambda: watermark_text(self.student_name, self.pipeline.session_id),
            )

    def _card(self) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)
        return card, layout

    def _build_test_panel(self) -> QWidget:
        card, layout = self._card()
        eyebrow = QLabel("ДЕМОНСТРАЦИОННЫЙ ТЕСТ")
        eyebrow.setObjectName("eyebrow")
        layout.addWidget(eyebrow)

        title = QLabel("Основы информационных технологий")
        title.setObjectName("heading")
        layout.addWidget(title)

        self.progress_label = QLabel()
        self.progress_label.setObjectName("status")
        layout.addWidget(self.progress_label)

        self.question_label = QLabel()
        self.question_label.setObjectName("question")
        self.question_label.setWordWrap(True)
        layout.addSpacing(12)
        layout.addWidget(self.question_label)

        self.answers_container = QWidget()
        self.answers_layout = QVBoxLayout(self.answers_container)
        self.answers_layout.setContentsMargins(0, 0, 0, 0)
        self.answers_layout.setSpacing(10)
        self.answer_group = QButtonGroup(self)
        layout.addWidget(self.answers_container)
        layout.addStretch(1)

        buttons = QHBoxLayout()
        self.previous_button = QPushButton("Назад")
        self.previous_button.setObjectName("secondary")
        self.previous_button.clicked.connect(self._previous_question)
        self.next_button = QPushButton("Далее")
        self.next_button.clicked.connect(self._next_question)
        buttons.addWidget(self.previous_button)
        buttons.addStretch(1)
        buttons.addWidget(self.next_button)
        layout.addLayout(buttons)
        return card

    def _build_monitor_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        camera_card, camera_layout = self._card()
        camera_heading = QLabel("Контроль сессии")
        camera_heading.setObjectName("heading")
        camera_layout.addWidget(camera_heading)
        self.camera_label = QLabel("Камера запускается…")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumHeight(250)
        self.camera_label.setStyleSheet(
            "background:#070b15; border-radius:10px; color:#70809e;"
        )
        camera_layout.addWidget(self.camera_label)

        self.camera_status_label = QLabel("○ Камера: ожидание")
        self.phone_status_label = QLabel("○ Телефон: ожидание")
        self.gaze_status_label = QLabel("○ Взгляд: ожидание")
        self.security_status_label = QLabel("○ Защита: ожидание")
        self.performance_status_label = QLabel("○ Производительность: замер…")
        for label in (
            self.camera_status_label,
            self.phone_status_label,
            self.gaze_status_label,
            self.security_status_label,
            self.performance_status_label,
        ):
            label.setObjectName("status")
            camera_layout.addWidget(label)
        layout.addWidget(camera_card)

        layout.addStretch(1)
        return container

    def _render_question(self) -> None:
        question = QUESTIONS[self.question_index]
        self.progress_label.setText(
            f"Вопрос {self.question_index + 1} из {len(QUESTIONS)}"
        )
        self.question_label.setText(question.text)

        for button in self.answer_group.buttons():
            self.answer_group.removeButton(button)
            button.deleteLater()
        for index, answer in enumerate(question.answers):
            button = QRadioButton(answer)
            self.answer_group.addButton(button, index)
            self.answers_layout.addWidget(button)
            if self.answers.get(self.question_index) == index:
                button.setChecked(True)

        self.previous_button.setEnabled(self.question_index > 0)
        is_last = self.question_index == len(QUESTIONS) - 1
        self.next_button.setText("Завершить тест" if is_last else "Далее")

    def _save_answer(self) -> None:
        checked = self.answer_group.checkedId()
        if checked >= 0:
            self.answers[self.question_index] = checked

    def _previous_question(self) -> None:
        self._save_answer()
        if self.question_index > 0:
            self.question_index -= 1
            self._render_question()

    def _next_question(self) -> None:
        if self.answer_group.checkedId() < 0:
            QMessageBox.information(self, "Выберите ответ", "Сначала выберите вариант ответа.")
            return
        self._save_answer()
        if self.question_index < len(QUESTIONS) - 1:
            self.question_index += 1
            self._render_question()
            return

        correct = sum(
            self.answers.get(index) == question.correct_index
            for index, question in enumerate(QUESTIONS)
        )
        self.next_button.setEnabled(False)
        self.previous_button.setEnabled(False)
        self._review_score = (correct, len(QUESTIONS))
        self.progress_label.setText("Тест завершён · сохраняем результаты…")
        if self.shutdown():
            self._show_review()

    def _show_review(self) -> None:
        score = self._review_score
        self._review_score = None
        if score is None:
            return
        final_risk = self.pipeline.peak_risk().total
        try:
            events = load_session_events(
                self.config.database_path,
                self.pipeline.session_id,
            )
        except Exception as exc:
            events = []
            self._show_runtime_error(
                f"Не удалось загрузить таймлайн: {type(exc).__name__}: {exc}"
            )
        self.hide()
        review = TeacherReviewDialog(
            database_path=self.config.database_path,
            session_id=self.pipeline.session_id,
            events=events,
            final_risk=final_risk,
            test_score=score,
            parent=self,
        )
        review.exec()
        self.close()

    def start_monitoring(self) -> None:
        if self._shutdown_started:
            return
        enabled, message = self.security.enable(hwnd=int(self.winId()))
        self._set_status(self.security_status_label, "Защита", enabled, message)

        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker(self.config, self.detectors, self.pipeline)
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.frame_ready.connect(self._show_frame)
        self.camera_worker.camera_status.connect(self._on_camera_status)
        self.camera_worker.module_status.connect(self._on_module_status)
        self.camera_worker.analysis_error.connect(self._show_runtime_error)
        self.camera_worker.performance_ready.connect(self._on_performance)
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_thread.start()

    def _submit_security_event(self, event: ProctorEvent) -> None:
        frame = self.camera_worker.snapshot() if self.camera_worker is not None else None
        self.pipeline.submit(event, frame)

    def _show_frame(self) -> None:
        image = self.camera_worker.take_preview() if self.camera_worker is not None else None
        if image is None or self._shutdown_started:
            return
        pixmap = QPixmap.fromImage(image)
        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _set_status(
        self,
        label: QLabel,
        title: str,
        ok: bool,
        message: str,
    ) -> None:
        icon = "●" if ok else "○"
        color = "#55d6a9" if ok else "#f0a45d"
        label.setText(f"{icon} {title}: {message}")
        label.setStyleSheet(f"color:{color};")

    def _on_camera_status(self, ok: bool, message: str) -> None:
        self._set_status(self.camera_status_label, "Камера", ok, message)
        if not ok and "остановлена" not in message:
            self.camera_label.setText(message)

    def _on_module_status(self, status: ModuleStatus) -> None:
        label = self.phone_status_label if status.name == "phone" else self.gaze_status_label
        title = "Телефон" if status.name == "phone" else "Взгляд"
        self._set_status(label, title, status.loaded, status.message)

    def _on_performance(
        self,
        camera_fps: float,
        analysis_fps: float,
        phone_ms: float,
        gaze_ms: float,
    ) -> None:
        total_ms = phone_ms + gaze_ms
        message = (
            f"камера {camera_fps:.1f} FPS · анализ {analysis_fps:.1f} FPS · "
            f"YOLO {phone_ms:.0f} мс · MediaPipe {gaze_ms:.0f} мс · "
            f"вместе {total_ms:.0f} мс"
        )
        self._set_status(self.performance_status_label, "Скорость", True, message)

    def _show_runtime_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 7000)

    def shutdown(self) -> bool:
        """Request shutdown and poll completion without blocking the Qt thread."""
        if self._shutdown_done:
            return True
        if not self._shutdown_started:
            self._shutdown_started = True
            self.next_button.setEnabled(False)
            self.previous_button.setEnabled(False)
            if self.camera_worker is not None:
                self.camera_worker.stop()
            if self.camera_thread is not None:
                self.camera_thread.quit()
            self.security.disable()
            self._shutdown_timer.start()
        if self.camera_thread is not None and self.camera_thread.isRunning():
            return False
        # Finish the writer only after the camera can no longer enqueue evidence.
        if not self.pipeline.stop(timeout=0):
            return False
        self._shutdown_done = True
        self._shutdown_timer.stop()
        self.shutdown_finished.emit()
        return True

    def _continue_shutdown(self) -> None:
        if self.shutdown():
            if self._close_requested:
                self.close()
            elif self._review_score is not None:
                self._show_review()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._close_requested = True
        self._review_score = None
        if self.shutdown():
            event.accept()
        else:
            event.ignore()


def run_application(config: AppConfig) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Proctoring")
    # Camera, protection and the session start only after consent and a name.
    student_name = student_name_from_env()
    if student_name is None:
        from ui.student_start import StudentStartDialog

        start = StudentStartDialog()
        if start.exec() != QDialog.DialogCode.Accepted:
            return 0
        student_name = start.student_name()
    window = MainWindow(config, student_name)
    window.show()
    try:
        return app.exec()
    finally:
        if not window.shutdown():
            # app.quit() can bypass closeEvent; retain the window and its QThread
            # until the same asynchronous cleanup finishes in this local loop.
            cleanup = QEventLoop()
            window.shutdown_finished.connect(cleanup.quit)
            cleanup.exec()

