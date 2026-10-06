"""Окна преподавателя в тёмной палитре основного приложения."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QSize, Qt
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
    Review,
    ReviewStore,
    Session,
    VERDICT_LABELS,
    current_verdict,
    event_duration,
    export_roster,
    load_sessions,
    recalculate_risk,
    risk_zone,
    summarize_events,
)


ZONE_COLORS = {"low": "#55d6a9", "medium": "#f0c45d", "high": "#ef6262"}
ZONE_LABELS = {"low": "зелёная", "medium": "жёлтая", "high": "красная"}
MOMENT_LABELS = {
    "phone_detected": "Телефон",
    "phone_aimed_at_screen": "Телефон у экрана",
    "multiple_faces": "Несколько лиц",
    "remote_session": "Удалённая сессия",
    "multiple_monitors": "Несколько мониторов",
    "capture_protection_failed": "Ошибка защиты",
    "too_close_to_camera": "Близко к камере",
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
        painter.setPen(QPen(QColor("#44516b"), 3))
        painter.drawLine(left, center, right, center)
        started = self.session.started_at.timestamp()
        ended = (self.session.ended_at or datetime.now().astimezone()).timestamp()
        span = max(1.0, ended - started)
        for index, stored in enumerate(self.session.events):
            fraction = min(1.0, max(0.0, (stored.event.occurred_at.timestamp() - started) / span))
            x = left + round((right - left) * fraction)
            y = center + (-14 if index % 2 else 14)
            color = ("#70809e" if stored.event.event_id in self.excluded else
                     "#ef6262" if stored.weight >= 30 else
                     "#f0c45d" if stored.weight >= 15 else "#55d6a9")
            painter.setPen(QPen(QColor(color), 2))
            painter.drawLine(x, center, x, y)
            painter.setBrush(QColor(color))
            painter.drawEllipse(x - 5, y - 5, 10, 10)
        painter.setPen(QColor("#8f9cb5"))
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


class TeacherWindow(QMainWindow):
    """Список сессий и подробный просмотр в одном окне."""

    def __init__(self, database_path: Path, reviews_path: Path) -> None:
        super().__init__()
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
        self.reviews = ReviewStore(reviews_path)
        self.sessions: list[Session] = []
        self.review_values: dict[str, Review] = {}
        self.detail_session: Session | None = None
        self.setWindowTitle("Proctoring · Панель преподавателя")
        self.resize(1320, 830)
        self.setMinimumSize(980, 650)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0b1020; color: #e8edf7; }
            QLabel { background: transparent; }
            QFrame#card { background: #121a2d; border: 1px solid #26334f;
                          border-radius: 14px; }
            QLabel#eyebrow { color: #8d9bb8; font-size: 11px; font-weight: 700; }
            QLabel#heading { font-size: 27px; font-weight: 700; }
            QLabel#section { font-size: 17px; font-weight: 700; }
            QPushButton, QToolButton { background: #5d7df6; color: white; border: 0;
                         border-radius: 9px; padding: 9px 15px; font-weight: 700; }
            QPushButton:hover, QToolButton:hover { background: #7290ff; }
            QPushButton#secondary { background: #202d49; }
            QTableWidget, QTextEdit { background: #121a2d; border: 1px solid #26334f;
                                      border-radius: 10px; gridline-color: #26334f;
                                      selection-background-color: #294475; }
            QHeaderView::section { background: #202d49; color: #b9c4da;
                                   border: 0; padding: 9px; font-weight: 700; }
            QTableWidget::item { padding: 6px; }
            QCheckBox { spacing: 8px; }
            QScrollArea { border: 0; }
        """)
        self.pages = QStackedWidget()
        self.setCentralWidget(self.pages)
        self.sessions_page = self._build_sessions_page()
        self.pages.addWidget(self.sessions_page)
        self.reload_sessions()

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
            score = recalculate_risk(session, review.false_positive_ids)
            summary = summarize_events(session, review.false_positive_ids)
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
        root.addLayout(top)
        heading = QLabel(session.student_name)
        heading.setObjectName("heading")
        root.addWidget(heading)
        started = session.started_at.astimezone().strftime("%d.%m.%Y %H:%M")
        root.addWidget(QLabel(
            f"{started}  ·  {_duration_text(session.duration_seconds)}  ·  "
            f"{'идёт' if session.status == 'active' else 'завершена' if session.status == 'completed' else session.status}"
        ))
        score = recalculate_risk(session, review.false_positive_ids)
        risk_card, risk_layout = _card()
        risk_label = QLabel(
            f"Пересчитанный риск: {score:.1f} / 100  ·  {ZONE_LABELS[risk_zone(score)]} зона"
        )
        risk_label.setObjectName("section")
        risk_label.setStyleSheet(f"color: {_risk_color(score)};")
        risk_layout.addWidget(risk_label)
        risk_layout.addWidget(QLabel(f"Риск в исходном журнале: {session.final_risk:.1f}"))
        root.addWidget(risk_card)

        summary_card, summary_layout = _card()
        title = QLabel("Сводка нарушений")
        title.setObjectName("section")
        summary_layout.addWidget(title)
        summary = summarize_events(session, review.false_positive_ids)
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
        timeline_layout.addWidget(Timeline(session, review.false_positive_ids))
        timeline_layout.addWidget(QLabel("Зелёный — низкий вес · жёлтый — средний · красный — высокий · серый — ошибочное событие"))
        root.addWidget(timeline_card)

        moments_card, moments_layout = _card()
        title = QLabel("Ключевые моменты · до 5 событий")
        title.setObjectName("section")
        moments_layout.addWidget(title)
        moments_row = QHBoxLayout()
        moments = sorted(
            (item for item in session.events if item.event.event_id not in review.false_positive_ids),
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
        event_table = QTableWidget(len(session.events), 6)
        event_table.setHorizontalHeaderLabels((
            "Время", "Нарушение", "Вес", "Длительность", "Источник", "Ложное срабатывание",
        ))
        event_table.verticalHeader().setVisible(False)
        event_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        event_table.verticalHeader().setMinimumSectionSize(40)
        event_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        event_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        event_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        event_table.blockSignals(True)
        for row, stored in enumerate(session.events):
            duration = event_duration(stored)
            values = (
                stored.event.occurred_at.astimezone().strftime("%H:%M:%S"),
                _event_name(stored), f"+{stored.weight:g}",
                f"{duration:.1f} с" if duration is not None else "—",
                stored.event.source,
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setToolTip(value)
                event_table.setItem(row, column, cell)
            flag = QTableWidgetItem()
            flag.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            flag.setData(Qt.ItemDataRole.UserRole, stored.event.event_id)
            flag.setCheckState(
                Qt.CheckState.Checked if stored.event.event_id in review.false_positive_ids
                else Qt.CheckState.Unchecked
            )
            event_table.setItem(row, 5, flag)
        event_table.blockSignals(False)
        event_table.itemChanged.connect(
            lambda item: self._event_flag_changed(session, item)
        )
        event_table.setMinimumHeight(min(450, max(140, 60 * (len(session.events) + 1))))
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
        pixmap.fill(QColor("#18243b"))
        painter = QPainter(pixmap)
        painter.setPen(QColor("#91a2bf"))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Нет кадра")
        painter.end()
        return pixmap

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
