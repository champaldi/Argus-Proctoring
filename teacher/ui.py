"""Окна преподавателя в тёмной палитре основного приложения."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import base64

from PySide6.QtCore import QByteArray, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.storage import StoredEvent
from core.event_presentation import EVENT_LABELS, application_details
from .data import (
    EXAM_ALLOWANCES,
    Review,
    ReviewStore,
    Session,
    VERDICT_LABELS,
    allowed_event_ids,
    current_verdict,
    event_duration,
    excluded_event_ids,
    export_roster,
    load_sessions,
    recalculate_risk,
    resolve_evidence_path,
    risk_zone,
    summarize_events,
    weight_label,
)
from .live import INFO, Alert, LiveServer, local_addresses
from .rules import LEVEL_TITLES, SUSPICIOUS, VIOLATION, Finding, evaluate_rules
from ui.theme import (
    CRITICAL,
    CRITICAL_TEXT,
    MUTED,
    SAFE,
    SAFE_TEXT,
    TEACHER_STYLE,
    WARNING,
    WARNING_TEXT,
    window_title,
)

from .conclusion import Conclusion, build_conclusion
from .report import default_report_name, write_session_report


# Text shades of the palette's status colours: readable on the light page.
ZONE_COLORS = {"low": SAFE_TEXT, "medium": WARNING_TEXT, "high": CRITICAL_TEXT}
ZONE_LABELS = {"low": "зелёная", "medium": "жёлтая", "high": "красная"}
MOMENT_LABELS = {
    "phone_detected": "Телефон",
    "phone_aimed_at_screen": "Телефон у экрана",
    "multiple_faces": "Несколько лиц",
    "remote_session": "Удалённая сессия",
    "multiple_monitors": "Несколько мониторов",
    "capture_protection_failed": "Ошибка защиты",
    "too_close_to_camera": "Близко к камере",
    "protection_disabled": "Защита отключена",
    "gaze_down": "Взгляд вниз",
    "gaze_side": "Взгляд в сторону",
    "no_face": "Нет лица",
    "hotkey_blocked": "Заблокированные клавиши",
    "window_switched": "Смена окна",
    "suspicious_process": "Запрещённая программа",
    "injected_input": "Удалённый ввод",
}


def _duration_text(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _event_name(stored: StoredEvent) -> str:
    name = stored.event.type.value
    title = EVENT_LABELS.get(name, name)
    return "\n".join([title, *application_details(stored.event)])


def _risk_color(score: float) -> str:
    return ZONE_COLORS[risk_zone(score)]


def _card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(20, 18, 20, 18)
    layout.setSpacing(10)
    return frame, layout



def _conclusion_html(conclusion: Conclusion) -> str:
    """The conclusion with the headline and fired rules in bold."""
    parts = [f"<p><b>{escape(conclusion.headline)}</b></p>"]
    if conclusion.findings:
        parts.append(
            f"<p>Сработало правил: <b>{len(conclusion.findings)}</b> — "
            "подробно с кадрами в блоке «Сработавшие правила» ниже.</p><p>События:</p>"
        )
    if conclusion.reasons:
        parts.append("<ul>")
        parts.extend(f"<li>{escape(str(item))}</li>" for item in conclusion.reasons)
        parts.append("</ul>")
    parts.append(f"<p>{escape(conclusion.recommendation)}</p>")
    return "".join(parts)

class NumericItem(QTableWidgetItem):
    """Сортирует риск как число, а не как строку."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        left = self.data(Qt.ItemDataRole.UserRole)
        right = other.data(Qt.ItemDataRole.UserRole)
        if left is not None and right is not None:
            return float(left) < float(right)
        return super().__lt__(other)


class Timeline(QWidget):
    """Рисует события на оси времени с начала до конца теста."""

    def __init__(self, session: Session, excluded: frozenset[str]) -> None:
        super().__init__()
        self.session = session
        self.excluded = excluded
        self.setMinimumHeight(100)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left, right = 38, max(39, self.width() - 38)
        center = 48
        painter.setPen(QPen(QColor("#D1D5DB"), 3))
        painter.drawLine(left, center, right, center)
        started = self.session.started_at.timestamp()
        ended = (self.session.ended_at or datetime.now().astimezone()).timestamp()
        span = max(1.0, ended - started)
        for index, stored in enumerate(self.session.events):
            fraction = min(1.0, max(0.0, (stored.event.occurred_at.timestamp() - started) / span))
            x = left + round((right - left) * fraction)
            y = center + (-14 if index % 2 else 14)
            color = ("#9CA3AF" if stored.event.event_id in self.excluded else
                     CRITICAL if stored.weight >= 30 else
                     WARNING if stored.weight >= 15 else SAFE)
            painter.setPen(QPen(QColor(color), 2))
            painter.drawLine(x, center, x, y)
            painter.setBrush(QColor(color))
            painter.drawEllipse(x - 5, y - 5, 10, 10)
        painter.setPen(QColor(MUTED))
        painter.drawText(left, 89, self.session.started_at.astimezone().strftime("%H:%M"))
        if self.session.ended_at:
            label = self.session.ended_at.astimezone().strftime("%H:%M")
            painter.drawText(right - 38, 89, label)
        painter.end()


class ImageDialog(QDialog):
    """Показывает сохранённый кадр крупно по клику на момент."""

    def __init__(self, title: str, pixmap: QPixmap, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 760)
        root = QVBoxLayout(self)
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setPixmap(pixmap.scaled(
            QSize(1040, 700), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        root.addWidget(label)


TOAST_SECONDS = 12
MAX_TOASTS = 3
TOAST_COLORS = {VIOLATION: CRITICAL_TEXT, SUSPICIOUS: WARNING_TEXT, INFO: MUTED}


def _pixmap_from_base64(value: str | None) -> QPixmap | None:
    if not value:
        return None
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        return None
    pixmap = QPixmap()
    return pixmap if pixmap.loadFromData(QByteArray(data)) else None


class AlertToast(QFrame):
    """Всплывающее уведомление в правом верхнем углу; закрывается по щелчку."""

    def __init__(self, alert: Alert, parent: QWidget) -> None:
        super().__init__(parent)
        self.alert = alert
        color = TOAST_COLORS.get(alert.level, MUTED)
        self.setObjectName("toast")
        self.setStyleSheet(
            "QFrame#toast { background: #FFFFFF; border: 1px solid #E5E7EB;"
            f" border-left: 6px solid {color}; border-radius: 10px; }}"
            " QLabel { border: 0; background: transparent; }"
        )
        self.setFixedWidth(420)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)
        pixmap = _pixmap_from_base64(alert.thumbnail)
        if pixmap is not None:
            picture = QLabel()
            picture.setPixmap(pixmap.scaled(
                QSize(120, 90), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
            layout.addWidget(picture, 0, Qt.AlignmentFlag.AlignTop)
        when = alert.occurred_at.astimezone().strftime("%H:%M:%S")
        level = LEVEL_TITLES.get(alert.level)
        headline = f"{level} · {alert.rule_id}" if level else alert.title
        self.text_label = QLabel(
            f"<b style='color:{color}'>{escape(headline)}</b>"
            f" <span style='color:{MUTED}'>{when}</span><br>"
            + (f"<b>{escape(alert.title)}</b><br>" if level else "")
            + escape(alert.text)
        )
        self.text_label.setWordWrap(True)
        self.text_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.text_label, 1)
        self.setToolTip("Щёлкните, чтобы закрыть")
        # The timer is a child: it disappears with the toast if that is closed first.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.close)
        self._timer.start(TOAST_SECONDS * 1000)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        # Deleted later by the window, which still holds a reference to it.
        self.close()


class TeacherWindow(QMainWindow):
    """Список сессий и подробный просмотр в одном окне."""

    def __init__(
        self, database_path: Path, reviews_path: Path, live_server: LiveServer | None = None
    ) -> None:
        super().__init__()
        self.live_server = live_server
        self._toasts: list[AlertToast] = []
        # Qt offscreen на Windows иногда не видит системные шрифты сам.
        if not QFontDatabase.families():
            for path in (Path("C:/Windows/Fonts/segoeui.ttf"),
                         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
                if path.is_file():
                    font_id = QFontDatabase.addApplicationFont(str(path))
                    families = QFontDatabase.applicationFontFamilies(font_id)
                    if families:
                        QApplication.instance().setFont(QFont(families[0], 10))
                        break
        self.database_path = database_path
        # Frames are read from disk once; the session page is rebuilt on every
        # mark or condition change and would otherwise reload every image.
        self._pixmaps: dict[str, QPixmap] = {}
        self._thumbnails: dict[str, QIcon] = {}
        self.reviews = ReviewStore(reviews_path)
        self.sessions: list[Session] = []
        self.review_values: dict[str, Review] = {}
        self.detail_session: Session | None = None
        self.setWindowTitle(window_title("Панель преподавателя"))
        self.resize(1320, 830)
        self.setMinimumSize(980, 650)
        self.setStyleSheet(TEACHER_STYLE)
        self.pages = QStackedWidget()
        self.setCentralWidget(self.pages)
        self.sessions_page = self._build_sessions_page()
        self.pages.addWidget(self.sessions_page)
        self.reload_sessions()
        if self.live_server is not None:
            self._refresh_live_table()
            self.live_timer = QTimer(self)
            self.live_timer.timeout.connect(self._poll_live)
            self.live_timer.start(500)
            self._live_ticks = 0

    def _build_sessions_page(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(26, 24, 26, 24)
        root.setSpacing(16)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        eyebrow = QLabel("ПРОКТОРИНГ  /  ПРЕПОДАВАТЕЛЬ")
        eyebrow.setObjectName("eyebrow")
        titles.addWidget(eyebrow)
        heading = QLabel("Сессии")
        heading.setObjectName("heading")
        titles.addWidget(heading)
        header.addLayout(titles)
        header.addStretch()
        refresh = QPushButton("Обновить")
        refresh.setObjectName("secondary")
        refresh.clicked.connect(self.reload_sessions)
        header.addWidget(refresh)
        export = QPushButton("Экспорт ведомости")
        export.clicked.connect(self._export)
        header.addWidget(export)
        root.addLayout(header)

        summary_card, summary_layout = _card()
        self.counter_label = QLabel()
        self.counter_label.setObjectName("section")
        summary_layout.addWidget(self.counter_label)
        self.database_label = QLabel(str(self.database_path))
        self.database_label.setObjectName("eyebrow")
        self.database_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        summary_layout.addWidget(self.database_label)
        root.addWidget(summary_card)
        if self.live_server is not None:
            root.addWidget(self._build_live_card())

        filters = QHBoxLayout()
        self.review_filter = QCheckBox("Только требующие проверки")
        self.review_filter.toggled.connect(self._populate_table)
        filters.addWidget(self.review_filter)
        filters.addStretch()
        root.addLayout(filters)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels((
            "Студент", "Дата", "Длительность", "Риск", "Нарушения по типам", "Статус",
        ))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(self._open_row)
        root.addWidget(self.table, 1)
        hint = QLabel("Двойной щелчок по строке открывает сессию.")
        hint.setObjectName("eyebrow")
        root.addWidget(hint)
        return page

    def _build_live_card(self) -> QFrame:
        card, layout = _card()
        title = QLabel("Идёт экзамен")
        title.setObjectName("section")
        layout.addWidget(title)
        port = self.live_server.port
        addresses = ", ".join(f"{address}:{port}" for address in local_addresses())
        self.live_address_label = QLabel(
            f"Адрес для студентов: {addresses}  ·  в браузере: http://{local_addresses()[0]}:{port}"
        )
        self.live_address_label.setObjectName("eyebrow")
        self.live_address_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.live_address_label)
        self.live_table = QTableWidget(0, 4)
        self.live_table.setHorizontalHeaderLabels(
            ("Студент", "Статус", "Риск", "Последнее правило")
        )
        self.live_table.verticalHeader().setVisible(False)
        self.live_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.live_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self.live_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.live_table.setMaximumHeight(190)
        layout.addWidget(self.live_table)
        self.live_empty_label = QLabel("Пока никто не начал тест")
        self.live_empty_label.setStyleSheet(f"color:{MUTED};")
        layout.addWidget(self.live_empty_label)
        return card

    def _refresh_live_table(self) -> None:
        sessions = self.live_server.board.sessions()
        self.live_table.setRowCount(0)
        for live in sessions:
            row = self.live_table.rowCount()
            self.live_table.insertRow(row)
            status = live.status()
            values = (live.display_name, status, f"{live.risk:.0f}", live.last_alert or "—")
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if column == 2:
                    cell.setForeground(QColor(_risk_color(live.risk)))
                if column == 3 and live.worst_level:
                    cell.setForeground(QColor(TOAST_COLORS[live.worst_level]))
                if column == 1 and status == "Нет связи":
                    cell.setForeground(QColor(WARNING_TEXT))
                self.live_table.setItem(row, column, cell)
        self.live_table.setVisible(bool(sessions))
        self.live_empty_label.setVisible(not sessions)

    def _poll_live(self) -> None:
        """Забирает уведомления из сервера; статусы обновляет раз в 5 секунд."""
        alerts = self.live_server.drain()
        self._live_ticks += 1
        changed = self.live_server.changed.is_set()
        self.live_server.changed.clear()
        for alert in alerts:
            if alert.title == "Начал тест":
                continue
            self._show_toast(alert)
        if any(alert.level == VIOLATION for alert in alerts):
            QApplication.beep()
        if changed or alerts or self._live_ticks % 10 == 0:
            self._refresh_live_table()
        if any(alert.title == "Завершил тест" for alert in alerts):
            # Если студент сдавал на этом же компьютере, сессия уже в журнале.
            self.reload_sessions()

    def _drop_closed_toasts(self) -> None:
        for toast in self._toasts:
            if not toast.isVisible():
                toast.deleteLater()
        self._toasts = [toast for toast in self._toasts if toast.isVisible()]

    def _show_toast(self, alert: Alert) -> None:
        self._drop_closed_toasts()
        while len(self._toasts) >= MAX_TOASTS:
            oldest = self._toasts.pop(0)
            oldest.close()
            oldest.deleteLater()
        toast = AlertToast(alert, self)
        toast.show()
        toast.raise_()
        self._toasts.append(toast)
        self._place_toasts()

    def _place_toasts(self) -> None:
        self._drop_closed_toasts()
        y = 16
        for toast in reversed(self._toasts):
            toast.adjustSize()
            toast.move(self.width() - toast.width() - 20, y)
            y += toast.height() + 10

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._place_toasts()

    def reload_sessions(self) -> None:
        """Обновляет список, не меняя исходный журнал."""
        try:
            self.sessions = load_sessions(self.database_path)
            self.review_values = self.reviews.load_all()
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось загрузить сессии", str(exc))
            return
        needs_review = sum(
            current_verdict(item, self.review_values.get(item.id)) in
            {"unreviewed", "questionable"} for item in self.sessions
        )
        self.counter_label.setText(
            f"Всего {len(self.sessions)}, требуют проверки {needs_review}"
        )
        self._populate_table()

    def _populate_table(self) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for session in self.sessions:
            review = self.review_values.get(session.id, Review())
            verdict = current_verdict(session, self.review_values.get(session.id))
            if self.review_filter.isChecked() and verdict not in {"unreviewed", "questionable"}:
                continue
            excluded = excluded_event_ids(session, review)
            score = recalculate_risk(session, excluded)
            summary = summarize_events(session, excluded)
            count_text = ", ".join(
                f"{EVENT_LABELS.get(name, name)} ×{item.count}"
                for name, item in sorted(summary.items())
            ) or "Нет"
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = (
                session.student_name,
                session.started_at.astimezone().strftime("%d.%m.%Y %H:%M"),
                _duration_text(session.duration_seconds),
                f"{score:.1f} · {ZONE_LABELS[risk_zone(score)]}",
                count_text,
                VERDICT_LABELS[verdict].capitalize(),
            )
            for column, value in enumerate(values):
                cell = NumericItem(value) if column == 3 else QTableWidgetItem(value)
                if column == 0:
                    cell.setData(Qt.ItemDataRole.UserRole, session.id)
                if column == 3:
                    cell.setData(Qt.ItemDataRole.UserRole, score)
                    cell.setForeground(QColor(_risk_color(score)))
                self.table.setItem(row, column, cell)
            self.table.setRowHeight(row, 44)
        self.table.setSortingEnabled(True)
        self.table.sortItems(3, Qt.SortOrder.DescendingOrder)

    def _open_row(self, row: int, column: int) -> None:
        key = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        session = next((item for item in self.sessions if item.id == key), None)
        if session is not None:
            self._show_session(session)

    def _show_session(self, session: Session, draft_comment: str | None = None) -> None:
        self.detail_session = session
        review = self.reviews.load_review(session.id)
        self.review_values[session.id] = review
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(26, 24, 26, 24)
        root.setSpacing(18)

        top = QHBoxLayout()
        back = QPushButton("← К сессиям")
        back.setObjectName("secondary")
        back.clicked.connect(self._back)
        top.addWidget(back)
        top.addStretch()
        report_button = QPushButton("Сохранить отчёт")
        report_button.setObjectName("secondary")
        report_button.setToolTip("Отчёт по сессии одним HTML-файлом с кадрами")
        report_button.clicked.connect(lambda: self._export_report(session))
        top.addWidget(report_button)
        root.addLayout(top)
        heading = QLabel(session.student_name)
        heading.setObjectName("heading")
        root.addWidget(heading)
        reference_path = resolve_evidence_path(self.database_path, session.reference_photo)
        reference = QPixmap(str(reference_path)) if reference_path is not None else QPixmap()
        if not reference.isNull():
            photo = QLabel()
            photo.setPixmap(reference.scaledToWidth(
                220, Qt.TransformationMode.SmoothTransformation
            ))
            photo.setToolTip("Контрольный кадр в начале теста. Личность сверяет преподаватель.")
            root.addWidget(photo)
            root.addWidget(QLabel("Контрольный кадр в начале теста"))
        started = session.started_at.astimezone().strftime("%d.%m.%Y %H:%M")
        root.addWidget(QLabel(
            f"{started}  ·  {_duration_text(session.duration_seconds)}  ·  "
            f"{'идёт' if session.status == 'active' else 'завершена' if session.status == 'completed' else session.status}"
        ))
        allowed = allowed_event_ids(session, review)
        excluded = excluded_event_ids(session, review)
        score = recalculate_risk(session, excluded)
        risk_card, risk_layout = _card()
        risk_label = QLabel(
            f"Пересчитанный риск: {score:.1f} / 100  ·  {ZONE_LABELS[risk_zone(score)]} зона"
        )
        risk_label.setObjectName("section")
        risk_label.setStyleSheet(f"color: {_risk_color(score)};")
        risk_layout.addWidget(risk_label)
        risk_layout.addWidget(QLabel(f"Риск в исходном журнале: {session.final_risk:.1f}"))
        root.addWidget(risk_card)

        conditions_card, conditions_layout = _card()
        title = QLabel("Условия экзамена")
        title.setObjectName("section")
        conditions_layout.addWidget(title)
        conditions_layout.addWidget(QLabel(
            "Разрешённое действие не считается нарушением: события остаются в журнале "
            "с пометкой, риск и правила пересчитываются."
        ))
        for key, (label, _types) in EXAM_ALLOWANCES.items():
            box = QCheckBox(label)
            box.setChecked(key in review.allowances)
            box.toggled.connect(
                lambda checked, kind=key: self._allowance_changed(session, kind, checked)
            )
            conditions_layout.addWidget(box)
        root.addWidget(conditions_card)

        conclusion_card, conclusion_layout = _card()
        title = QLabel("Заключение системы")
        title.setObjectName("section")
        conclusion_layout.addWidget(title)
        conclusion_label = QLabel(
            _conclusion_html(build_conclusion(session, excluded))
        )
        conclusion_label.setTextFormat(Qt.TextFormat.RichText)
        conclusion_label.setWordWrap(True)
        # Readable from a projector: the rules are the main finding.
        conclusion_label.setStyleSheet("font-size: 16px;")
        conclusion_layout.addWidget(conclusion_label)
        root.addWidget(conclusion_card)

        findings = evaluate_rules(session, excluded)
        if findings:
            root.addWidget(self._rules_card(session, findings))

        summary_card, summary_layout = _card()
        title = QLabel("Сводка нарушений")
        title.setObjectName("section")
        summary_layout.addWidget(title)
        summary = summarize_events(session, excluded)
        if not summary:
            summary_layout.addWidget(QLabel("Подтверждённых нарушений нет."))
        for name, item in sorted(summary.items()):
            duration = (f"{item.duration_seconds:.1f} с" if item.measured_count else "нет данных")
            if 0 < item.measured_count < item.count:
                duration += f" (для {item.measured_count} из {item.count})"
            summary_layout.addWidget(QLabel(
                f"{EVENT_LABELS.get(name, name)}: {item.count}  ·  суммарная длительность {duration}"
            ))
        root.addWidget(summary_card)

        timeline_card, timeline_layout = _card()
        title = QLabel("Таймлайн теста")
        title.setObjectName("section")
        timeline_layout.addWidget(title)
        timeline_layout.addWidget(Timeline(session, excluded))
        timeline_layout.addWidget(QLabel("Зелёный — низкий вес · жёлтый — средний · красный — высокий · серый — ошибочное или разрешённое событие"))
        root.addWidget(timeline_card)

        moments_card, moments_layout = _card()
        title = QLabel("Ключевые моменты · до 5 событий")
        title.setObjectName("section")
        moments_layout.addWidget(title)
        moments_row = QHBoxLayout()
        moments = sorted(
            (item for item in session.events if item.event.event_id not in excluded),
            key=lambda item: (item.weight, item.event.occurred_at), reverse=True,
        )[:5]
        if not moments:
            moments_row.addWidget(QLabel("Подтверждённых моментов нет."))
        for stored in moments:
            pixmap = self._frame_pixmap(stored)
            button = QToolButton()
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setIcon(QIcon(pixmap))
            button.setIconSize(QSize(184, 108))
            button.setFixedSize(214, 168)
            button.setText(
                f"{MOMENT_LABELS.get(stored.event.type.value, _event_name(stored))}\n"
                f"{stored.event.occurred_at.astimezone():%H:%M:%S} · +{stored.weight:g}"
            )
            button.setToolTip(_event_name(stored))
            button.clicked.connect(
                lambda checked=False, item=stored, image=pixmap:
                ImageDialog(_event_name(item), image, self).exec()
            )
            moments_row.addWidget(button)
        moments_row.addStretch()
        moments_layout.addLayout(moments_row)
        root.addWidget(moments_card)

        events_card, events_layout = _card()
        title = QLabel(f"Все события · {len(session.events)}")
        title.setObjectName("section")
        events_layout.addWidget(title)
        event_table = QTableWidget(len(session.events), 7)
        event_table.setHorizontalHeaderLabels((
            "Время", "Нарушение", "Вес", "Длительность", "Источник", "Ложное срабатывание",
            "Кадр",
        ))
        event_table.setIconSize(QSize(112, 63))
        event_table.verticalHeader().setVisible(False)
        event_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        event_table.verticalHeader().setMinimumSectionSize(40)
        event_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        event_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        event_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        event_table.blockSignals(True)
        for row, stored in enumerate(session.events):
            duration = event_duration(stored)
            is_allowed = stored.event.event_id in allowed
            values = (
                stored.event.occurred_at.astimezone().strftime("%H:%M:%S"),
                _event_name(stored) + (" · разрешено" if is_allowed else ""),
                weight_label(stored),
                f"{duration:.1f} с" if duration is not None else "—",
                stored.event.source,
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setToolTip(value)
                if is_allowed:
                    cell.setForeground(QColor(MUTED))
                event_table.setItem(row, column, cell)
            flag = QTableWidgetItem()
            flag.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            flag.setData(Qt.ItemDataRole.UserRole, stored.event.event_id)
            flag.setCheckState(
                Qt.CheckState.Checked if stored.event.event_id in review.false_positive_ids
                else Qt.CheckState.Unchecked
            )
            event_table.setItem(row, 5, flag)
            frame_cell = QTableWidgetItem()
            frame_cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
            frame_cell.setToolTip("Нажмите, чтобы открыть кадр")
            if stored.screenshot_path:
                frame_cell.setIcon(self._thumbnail(stored))
            else:
                frame_cell.setText("нет кадра")
            event_table.setItem(row, 6, frame_cell)
            event_table.setRowHeight(row, 72)
        event_table.blockSignals(False)
        event_table.cellClicked.connect(
            lambda row, column: self._open_event_frame(session, row, column)
        )
        event_table.itemChanged.connect(
            lambda item: self._event_flag_changed(session, item)
        )
        event_table.setMinimumHeight(min(520, max(160, 76 * (len(session.events) + 1))))
        events_layout.addWidget(event_table)
        root.addWidget(events_card)

        review_card, review_layout = _card()
        title = QLabel("Решение преподавателя")
        title.setObjectName("section")
        review_layout.addWidget(title)
        current = current_verdict(session, self.review_values.get(session.id))
        review_layout.addWidget(QLabel(f"Статус: {VERDICT_LABELS[current]}"))
        self.comment_edit = QTextEdit()
        self.comment_edit.setPlaceholderText("Комментарий к сессии")
        self.comment_edit.setPlainText(review.comment if draft_comment is None else draft_comment)
        self.comment_edit.setFixedHeight(92)
        review_layout.addWidget(self.comment_edit)
        actions = QHBoxLayout()
        for label, verdict in (("Списывал", "cheated"),
                               ("Не списывал", "not_cheated"),
                               ("Под вопросом", "questionable")):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, value=verdict:
                                   self._save_review(session, value))
            actions.addWidget(button)
        save = QPushButton("Сохранить комментарий")
        save.setObjectName("secondary")
        save.clicked.connect(lambda: self._save_review(session, None))
        actions.addWidget(save)
        actions.addStretch()
        review_layout.addLayout(actions)
        root.addWidget(review_card)
        root.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

        if self.pages.count() > 1:
            old = self.pages.widget(1)
            self.pages.removeWidget(old)
            old.deleteLater()
        self.pages.addWidget(page)
        self.pages.setCurrentWidget(page)

    def _frame_pixmap(self, stored: StoredEvent) -> QPixmap:
        key = stored.event.event_id
        cached = self._pixmaps.get(key)
        if cached is None:
            cached = self._load_frame(stored)
            self._pixmaps[key] = cached
        return cached

    def _thumbnail(self, stored: StoredEvent) -> QIcon:
        key = stored.event.event_id
        icon = self._thumbnails.get(key)
        if icon is None:
            icon = QIcon(self._frame_pixmap(stored).scaled(
                QSize(224, 126), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
            self._thumbnails[key] = icon
        return icon

    def _load_frame(self, stored: StoredEvent) -> QPixmap:
        if stored.screenshot_path:
            path = Path(stored.screenshot_path)
            # Older EventStore versions wrote paths relative to the launch CWD;
            # imported journals may instead use paths relative to the database.
            if not path.is_absolute() and not path.is_file():
                path = self.database_path.parent / path
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                return pixmap
        pixmap = QPixmap(640, 360)
        pixmap.fill(QColor("#E5E7EB"))
        painter = QPainter(pixmap)
        painter.setPen(QColor(MUTED))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Нет кадра")
        painter.end()
        return pixmap

    def _rules_card(self, session: Session, findings: tuple[Finding, ...]) -> QFrame:
        """Each fired rule as a card: what it means and the frames it is built on."""
        card, layout = _card()
        title = QLabel(f"Сработавшие правила · {len(findings)}")
        title.setObjectName("section")
        layout.addWidget(title)
        hint = QLabel(
            "Одиночный сигнал — только наблюдение. Правило срабатывает на сочетание "
            "событий во времени; ниже — события, из которых оно сложилось."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        by_id = {stored.event.event_id: stored for stored in session.events}
        for finding in findings:
            box = QFrame()
            violation = finding.level == VIOLATION
            accent = CRITICAL_TEXT if violation else WARNING_TEXT
            box.setStyleSheet(
                f"QFrame {{ border-left: 4px solid {accent}; background: #F9FAFB;"
                " border-radius: 8px; } QLabel { border: 0; background: transparent; }"
            )
            inner = QVBoxLayout(box)
            inner.setContentsMargins(14, 10, 14, 10)
            level = "Признаки нарушения" if violation else "Подозрительно"
            name = QLabel(f"{finding.rule_id} · {finding.title}  —  {level}")
            name.setStyleSheet(f"font-size:16px; font-weight:700; color:{accent};")
            inner.addWidget(name)
            explanation = QLabel(finding.explanation.capitalize())
            explanation.setWordWrap(True)
            inner.addWidget(explanation)
            frames = QHBoxLayout()
            shown = [by_id[event_id] for event_id in finding.event_ids if event_id in by_id]
            for stored in shown[:4]:
                pixmap = self._frame_pixmap(stored)
                button = QToolButton()
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
                button.setIcon(self._thumbnail(stored))
                button.setIconSize(QSize(160, 90))
                button.setFixedSize(186, 140)
                button.setText(
                    f"{MOMENT_LABELS.get(stored.event.type.value, _event_name(stored))}\n"
                    f"{stored.event.occurred_at.astimezone():%H:%M:%S}"
                )
                button.setToolTip(_event_name(stored))
                button.clicked.connect(
                    lambda checked=False, item=stored, image=pixmap:
                    ImageDialog(_event_name(item), image, self).exec()
                )
                frames.addWidget(button)
            if len(shown) > 4:
                frames.addWidget(QLabel(f"и ещё {len(shown) - 4}"))
            frames.addStretch()
            inner.addLayout(frames)
            layout.addWidget(box)
        return card

    def _open_event_frame(self, session: Session, row: int, column: int) -> None:
        if column != 6 or not 0 <= row < len(session.events):
            return
        stored = session.events[row]
        ImageDialog(_event_name(stored), self._frame_pixmap(stored), self).exec()

    def _event_flag_changed(self, session: Session, item: QTableWidgetItem) -> None:
        if item.column() != 5:
            return
        event_id = item.data(Qt.ItemDataRole.UserRole)
        draft = self.comment_edit.toPlainText()
        try:
            self.reviews.set_false_positive(
                session.id, event_id, item.checkState() == Qt.CheckState.Checked
            )
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось сохранить отметку", str(exc))
            return
        self._show_session(session, draft_comment=draft)

    def _allowance_changed(self, session: Session, kind: str, enabled: bool) -> None:
        draft = self.comment_edit.toPlainText()
        try:
            self.reviews.set_allowance(session.id, kind, enabled)
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось сохранить условия экзамена", str(exc))
            return
        self._show_session(session, draft_comment=draft)

    def _save_review(self, session: Session, verdict: str | None) -> None:
        selected = verdict or current_verdict(session, self.review_values.get(session.id))
        try:
            self.reviews.save_review(session.id, selected, self.comment_edit.toPlainText())
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось сохранить решение", str(exc))
            return
        self.statusBar().showMessage("Решение сохранено", 4000)
        self._show_session(session)

    def _back(self) -> None:
        self.reload_sessions()
        self.pages.setCurrentIndex(0)

    def _export_report(self, session: Session) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчёт по сессии", default_report_name(session), "HTML (*.html)"
        )
        if not filename:
            return
        try:
            write_session_report(
                Path(filename), session, self.reviews.load_review(session.id),
                self.database_path,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось сохранить отчёт", str(exc))
            return
        self.statusBar().showMessage(f"Отчёт сохранён: {filename}", 6000)

    def _export(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self, "Экспорт ведомости", "vedomost.csv", "CSV (*.csv)"
        )
        if not filename:
            return
        try:
            export_roster(Path(filename), self.sessions, self.reviews.load_all())
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось экспортировать ведомость", str(exc))
            return
        self.statusBar().showMessage(f"Ведомость сохранена: {filename}", 6000)
