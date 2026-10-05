"""Main test window and camera worker."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from config import AppConfig
from core.detectors import DetectorCollection, ModuleStatus
from core.pipeline import EventPipeline
from core.security import SecurityAdapter
from core.storage import StoredEvent
from events import ProctorEvent


EVENT_LABELS = {
    "phone_detected": "Обнаружен телефон",
    "phone_aimed_at_screen": "Телефон направлен на экран",
    "gaze_down": "Долгий взгляд вниз",
    "gaze_side": "Долгий взгляд в сторону",
    "no_face": "Лицо не обнаружено",
    "multiple_faces": "В кадре несколько лиц",
    "hotkey_blocked": "Заблокирована комбинация клавиш",
    "window_switched": "Переключение окна",
    "suspicious_process": "Обнаружен запрещённый процесс",
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
            self.camera_status.emit(False, "Не установлен opencv-python")
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

                now = time.monotonic()
                if now - last_analysis >= analysis_interval:
                    events, errors = self.detectors.analyze(frame)
                    for event in events:
                        self.pipeline.submit(event, frame)
                    for message in errors:
                        if now - last_error.get(message, 0.0) >= 5.0:
                            self.analysis_error.emit(message)
                            last_error[message] = now
                    last_analysis = now

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
            capture.release()
            self._capture = None
            self.camera_status.emit(False, "Камера остановлена")
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
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
            QProgressBar { border: 0; border-radius: 7px; background: #202a40;
                           height: 14px; text-align: center; }
            QProgressBar::chunk { background: #5d7df6; border-radius: 7px; }
            QListWidget { background: transparent; border: 0; color: #b9c4da; }
            """
        )

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(18)
        root.addWidget(self._build_test_panel(), 3)
        root.addWidget(self._build_monitor_panel(), 2)
        self.setCentralWidget(central)

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
        for label in (
            self.camera_status_label,
            self.phone_status_label,
            self.gaze_status_label,
            self.security_status_label,
        ):
            label.setObjectName("status")
            camera_layout.addWidget(label)
        layout.addWidget(camera_card)

        risk_card, risk_layout = self._card()
        risk_title = QLabel("Риск сессии")
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
        QMessageBox.information(
            self,
            "Тест завершён",
            f"Результат: {correct} из {len(QUESTIONS)}.\n"
            f"Риск сессии: {self.risk_bar.value()} из 100.",
        )
        self.next_button.setEnabled(False)
        self.progress_label.setText("Тест завершён · контроль выключен")
        self.shutdown()

    def start_monitoring(self) -> None:
        enabled, message = self.security.enable()
        self._set_status(self.security_status_label, "Защита", enabled, message)

        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker(self.config, self.detectors, self.pipeline)
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.frame_ready.connect(self._show_frame)
        self.camera_worker.camera_status.connect(self._on_camera_status)
        self.camera_worker.module_status.connect(self._on_module_status)
        self.camera_worker.analysis_error.connect(self._show_runtime_error)
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

    def _on_event_recorded(self, stored: StoredEvent) -> None:
        data = stored.to_dict()
        risk = int(data["risk_total"])
        self.risk_bar.setValue(risk)
        self.risk_bar.setFormat(f"{risk} / 100")
        if risk >= 60:
            chunk = "#ef6262"
        elif risk >= 25:
            chunk = "#f0a45d"
        else:
            chunk = "#55d6a9"
        self.risk_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background: {chunk}; border-radius: 7px; }}"
        )
        event_name = str(data["type"])
        label = EVENT_LABELS.get(event_name, event_name)
        time_label = stored.event.occurred_at.astimezone().strftime("%H:%M:%S")
        self.event_list.insertItem(0, f"{time_label}  {label}  +{stored.weight}")
        while self.event_list.count() > 8:
            self.event_list.takeItem(self.event_list.count() - 1)

    def _show_runtime_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 7000)

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
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
    app.setApplicationName("Proctoring")
    window = MainWindow(config)
    window.show()
    try:
        return app.exec()
    finally:
        window.shutdown()

