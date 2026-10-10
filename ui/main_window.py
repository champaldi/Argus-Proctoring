"""Main test window and camera worker."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from config import AppConfig, RISK_RED_ABOVE, RISK_YELLOW_FROM
from core.watermark import watermark_text
from core.student import session_metadata, student_name_from_env
from ui.fonts import apply_brand_accent, load_app_fonts
from ui.theme import (
    ACCENT,
    ACCENT_SOFT,
    APP_NAME,
    BORDER,
    CARD,
    CRITICAL,
    CRITICAL_TEXT,
    HEADING,
    MAIN_WINDOW_STYLE,
    MUTED,
    PAGE,
    SAFE,
    SAFE_TEXT,
    TEXT,
    WARNING_TEXT,
    STATUS_OK_COLOR,
    STATUS_WARNING_COLOR,
    window_title,
)
from core.detectors import DetectorCollection, ModuleStatus
from core.pipeline import EventPipeline
from core.security import SecurityAdapter
from core.storage import (
    StoredEvent,
    load_session_events,
    record_teacher_verdict,
)
from events import ProctorEvent


EVENT_LABELS = {
    "phone_detected": "Обнаружен телефон",
    "phone_aimed_at_screen": "Телефон направлен на экран",
    "gaze_down": "Долгий взгляд вниз",
    "gaze_side": "Взгляд в сторону: возможен второй монитор или шпаргалка",
    "no_face": "Лицо не обнаружено",
    "multiple_faces": "В кадре несколько лиц",
    "too_close_to_camera": "Слишком близко к камере",
    "hotkey_blocked": "Заблокирована комбинация клавиш",
    "window_switched": "Переключение окна",
    "suspicious_process": "Обнаружен запрещённый процесс",
    "capture_protection_failed": "Не удалось скрыть окно от захвата",
    "remote_session": "Тест запущен через удалённую сессию",
    "multiple_monitors": "Подключено несколько мониторов",
    "injected_input": "Ввод не с физических устройств",
    "protection_disabled": "Защита отключена вручную (Ctrl+Alt+F12)",
}


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
    event_recorded = Signal(object)
    pipeline_error = Signal(str)


class CameraWorker(QObject):
    frame_ready = Signal(QImage)
    module_status = Signal(object)
    camera_status = Signal(bool, str)
    analysis_error = Signal(str)
    performance_ready = Signal(float, float, float, float)
    finished = Signal()

    def __init__(
        self,
        config: AppConfig,
        detectors: DetectorCollection,
        pipeline: EventPipeline,
    ) -> None:
        super().__init__()
        self.config = config
        self.detectors = detectors
        self.pipeline = pipeline
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Any = None
        self._capture: Any = None

    def stop(self) -> None:
        self._stop_event.set()
        capture = self._capture
        if capture is not None:
            capture.release()

    def snapshot(self) -> Any:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def run(self) -> None:
        try:
            import cv2
        except ImportError:
            self.camera_status.emit(False, "Не установлен OpenCV")
            self.finished.emit()
            return

        for status in self.detectors.load():
            self.module_status.emit(status)

        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
        capture = cv2.VideoCapture(self.config.camera_index, backend)
        self._capture = capture
        if not capture.isOpened():
            capture.release()
            self.camera_status.emit(
                False,
                f"Камера {self.config.camera_index} недоступна",
            )
            self.finished.emit()
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera_status.emit(True, "Камера активна")

        analysis_interval = 1.0 / self.config.analysis_fps
        preview_interval = 1.0 / self.config.preview_fps
        last_analysis = 0.0
        last_preview = 0.0
        last_error: dict[str, float] = {}
        performance_started = time.monotonic()
        captured_frames = 0
        analyzed_frames = 0
        phone_time_total = 0.0
        gaze_time_total = 0.0

        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    if self._stop_event.is_set():
                        break
                    self.camera_status.emit(False, "Не удалось получить кадр")
                    break

                with self._frame_lock:
                    self._latest_frame = frame
                captured_frames += 1

                now = time.monotonic()
                if now - last_analysis >= analysis_interval:
                    events, errors = self.detectors.analyze(frame)
                    analyzed_frames += 1
                    phone_time_total += self.detectors.last_timings_ms.get("phone", 0.0)
                    gaze_time_total += self.detectors.last_timings_ms.get("gaze", 0.0)
                    for event in events:
                        self.pipeline.submit(event, frame)
                    for message in errors:
                        if now - last_error.get(message, 0.0) >= 5.0:
                            self.analysis_error.emit(message)
                            last_error[message] = now
                    last_analysis = now

                performance_elapsed = now - performance_started
                if performance_elapsed >= 5.0:
                    self.performance_ready.emit(
                        captured_frames / performance_elapsed,
                        analyzed_frames / performance_elapsed,
                        phone_time_total / max(1, analyzed_frames),
                        gaze_time_total / max(1, analyzed_frames),
                    )
                    performance_started = now
                    captured_frames = 0
                    analyzed_frames = 0
                    phone_time_total = 0.0
                    gaze_time_total = 0.0

                if now - last_preview >= preview_interval:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channels = rgb.shape
                    image = QImage(
                        rgb.data,
                        width,
                        height,
                        channels * width,
                        QImage.Format.Format_RGB888,
                    ).copy()
                    self.frame_ready.emit(image)
                    last_preview = now
                QThread.msleep(2)
        finally:
            self.detectors.close()
            capture.release()
            self._capture = None
            self.camera_status.emit(False, "Камера остановлена")
            self.finished.emit()


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
        self.setWindowTitle(window_title("Итоги сессии"))
        self.resize(900, 720)
        self.setModal(True)
        self.setStyleSheet(
            f"""
            QDialog, QWidget {{ background: {PAGE}; color: {TEXT}; }}
            QFrame#reviewCard {{ background: {CARD}; border: 1px solid {BORDER};
                                border-radius: 12px; }}
            QFrame#reviewCard QLabel {{ background: transparent; }}
            QLabel#reviewHeading {{ font-size: 24px; font-weight: 700; color: {HEADING}; }}
            QPushButton {{ border: 0; border-radius: 9px; padding: 12px 18px;
                          color: white; font-weight: 700; }}
            QPushButton#cheated {{ background: {CRITICAL}; }}
            QPushButton#cheated:hover {{ background: {CRITICAL_TEXT}; }}
            QPushButton#notCheated {{ background: {SAFE}; }}
            QPushButton#notCheated:hover {{ background: {SAFE_TEXT}; }}
            QScrollArea {{ border: 0; }}
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
        # Darker shades keep the number readable on the light background.
        if value > RISK_RED_ABOVE:
            return CRITICAL_TEXT
        if value >= RISK_YELLOW_FROM:
            return WARNING_TEXT
        return SAFE_TEXT

    def _event_card(self, stored: StoredEvent) -> QFrame:
        card = QFrame()
        card.setObjectName("reviewCard")
        row = QHBoxLayout(card)
        row.setContentsMargins(12, 12, 12, 12)
        preview = QLabel("Без снимка")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setFixedSize(180, 105)
        preview.setStyleSheet(
            f"background:{ACCENT_SOFT}; border-radius:8px; color:{MUTED};"
        )
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
        title.setStyleSheet(f"font-size:16px; font-weight:700; color:{HEADING};")
        text.addWidget(title)
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
    def __init__(self, config: AppConfig, student_name: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.student_name = student_name
        self.answers: dict[int, int] = {}
        self.question_index = 0
        self._shutdown_done = False

        self.bridge = UiBridge()
        self.bridge.event_recorded.connect(self._on_event_recorded)
        self.bridge.pipeline_error.connect(self._show_runtime_error)
        self.pipeline = EventPipeline(
            config.database_path,
            config.screenshots_dir,
            on_recorded=self.bridge.event_recorded.emit,
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
        self.risk_timer = QTimer(self)
        self.risk_timer.timeout.connect(self._refresh_risk)
        self.risk_timer.start(1000)
        QTimer.singleShot(0, self.start_monitoring)

    def _build_ui(self) -> None:
        self.setWindowTitle(window_title("Тестирование"))
        self.resize(1280, 780)
        self.setMinimumSize(1050, 680)
        self.setStyleSheet(
            MAIN_WINDOW_STYLE
            + """
            QProgressBar { border: 0; border-radius: 7px; background: #E5E7EB;
                           height: 14px; text-align: center; color: #1F2937; }
            QProgressBar::chunk { background: #1E3A8A; border-radius: 7px; }
            QListWidget { background: transparent; border: 0; color: #374151; }
            """
        )

        central = QWidget()
        central.setObjectName("page")
        root = QHBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(18)
        root.addWidget(self._build_test_panel(), 3)
        root.addWidget(self._build_monitor_panel(), 2)
        self.setCentralWidget(central)
        if os.getenv("PROCTOR_WATERMARK", "1").strip() != "0":
            # One faint mark over the question: the student, the session and the
            # time stay on any phone photo of the screen.
            from ui.watermark import WatermarkOverlay

            self.watermark = WatermarkOverlay(
                central,
                lambda: watermark_text(self.student_name, self.pipeline.session_id),
                centre_x=0.3,
                opacity=0.10,
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
            f"background:{ACCENT_SOFT}; border-radius:10px; color:{MUTED};"
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

        risk_card, risk_layout = self._card()
        risk_title = QLabel("Уровень риска")
        risk_title.setObjectName("question")
        risk_layout.addWidget(risk_title)
        self.risk_bar = QProgressBar()
        self.risk_bar.setRange(0, 100)
        self.risk_bar.setValue(0)
        self.risk_bar.setFormat("0 / 100")
        risk_layout.addWidget(self.risk_bar)
        self.event_list = QListWidget()
        self.event_list.setMinimumHeight(120)
        risk_layout.addWidget(self.event_list)
        layout.addWidget(risk_card, 1)
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
        self.progress_label.setText("Тест завершён · контроль выключен")
        self.shutdown()
        final_risk = self.pipeline.current_risk().total
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
            test_score=(correct, len(QUESTIONS)),
            parent=self,
        )
        review.exec()
        self.close()

    def start_monitoring(self) -> None:
        enabled, message = self.security.enable(hwnd=int(self.winId()))
        self._set_status(
            self.security_status_label,
            "Защита",
            enabled,
            "включена" if enabled else "не включена",
            detail=message,
        )

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

    def _show_frame(self, image: QImage) -> None:
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
        detail: str = "",
    ) -> None:
        icon = "●" if ok else "○"
        color = STATUS_OK_COLOR if ok else STATUS_WARNING_COLOR
        label.setText(f"{icon} {title}: {message}")
        label.setStyleSheet(f"color:{color};")
        # Module names and errors stay available on hover for the team.
        label.setToolTip(detail)

    def _on_camera_status(self, ok: bool, message: str) -> None:
        icon = "●" if ok else "○"
        color = STATUS_OK_COLOR if ok else STATUS_WARNING_COLOR
        self.camera_status_label.setText(f"{icon} {message}")
        self.camera_status_label.setStyleSheet(f"color:{color};")
        if not ok and "остановлена" not in message:
            self.camera_label.setText(message)

    def _on_module_status(self, status: ModuleStatus) -> None:
        label = self.phone_status_label if status.name == "phone" else self.gaze_status_label
        title = "Телефон" if status.name == "phone" else "Взгляд"
        self._set_status(
            label,
            title,
            status.loaded,
            "работает" if status.loaded else "не загружен",
            detail=status.message,
        )

    def _on_performance(
        self,
        camera_fps: float,
        analysis_fps: float,
        phone_ms: float,
        gaze_ms: float,
    ) -> None:
        total_ms = phone_ms + gaze_ms
        message = f"камера {camera_fps:.0f} FPS · анализ {analysis_fps:.0f} в секунду"
        detail = (
            f"YOLO {phone_ms:.0f} мс · MediaPipe {gaze_ms:.0f} мс · "
            f"вместе {total_ms:.0f} мс"
        )
        self._set_status(
            self.performance_status_label, "Скорость", True, message, detail=detail
        )

    def _update_risk_display(self, risk: float) -> None:
        self.risk_bar.setValue(max(0, min(100, round(risk))))
        self.risk_bar.setFormat(f"{risk:.1f} / 100")
        if risk > RISK_RED_ABOVE:
            chunk = "#ef6262"
        elif risk >= RISK_YELLOW_FROM:
            chunk = "#f0c45d"
        else:
            chunk = "#55d6a9"
        self.risk_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background: {chunk}; border-radius: 7px; }}"
        )

    def _refresh_risk(self) -> None:
        if not self._shutdown_done:
            self._update_risk_display(self.pipeline.current_risk().total)

    def _on_event_recorded(self, stored: StoredEvent) -> None:
        data = stored.to_dict()
        risk = float(data["risk_total"])
        self._update_risk_display(risk)
        event_name = str(data["type"])
        label = EVENT_LABELS.get(event_name, event_name)
        time_label = stored.event.occurred_at.astimezone().strftime("%H:%M:%S")
        self.event_list.insertItem(0, f"{time_label}  {label}  +{stored.weight:g}")
        while self.event_list.count() > 8:
            self.event_list.takeItem(self.event_list.count() - 1)

    def _show_runtime_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 7000)

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        if hasattr(self, "risk_timer"):
            self.risk_timer.stop()
        try:
            if self.camera_worker is not None:
                self.camera_worker.stop()
            if self.camera_thread is not None:
                self.camera_thread.quit()
                self.camera_thread.wait(4000)
        finally:
            try:
                self.security.disable()
            finally:
                self.pipeline.stop()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.shutdown()
        event.accept()


def run_application(config: AppConfig) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(APP_NAME)
    load_app_fonts(app)
    apply_brand_accent(app, ACCENT)
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
        window.shutdown()

