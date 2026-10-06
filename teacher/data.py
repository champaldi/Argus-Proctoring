"""Чтение сессий и отдельное хранение решений преподавателя."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from config import RISK_RED_ABOVE, RISK_YELLOW_FROM
from core.risk import RiskScorer
from core.storage import StoredEvent
from events import EventType, ProctorEvent


VERDICT_LABELS = {
    "unreviewed": "не проверено",
    "cheated": "списывал",
    "not_cheated": "не списывал",
    "questionable": "под вопросом",
}


def _parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result if result.tzinfo is not None else result.replace(tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    started_at: datetime
    ended_at: datetime | None
    status: str
    metadata: Mapping[str, object]
    final_risk: float
    legacy_verdict: str | None
    events: tuple[StoredEvent, ...]

    @property
    def student_name(self) -> str:
        """Берёт имя из метаданных; старые записи не содержат его."""
        for key in ("student_name", "student", "name", "student_id"):
            value = self.metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"Не указан · {self.id[:8]}"

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at or datetime.now(timezone.utc)
        return max(0.0, (end - self.started_at).total_seconds())


@dataclass(frozen=True, slots=True)
class EventSummary:
    count: int
    duration_seconds: float
    measured_count: int


@dataclass(frozen=True, slots=True)
class Review:
    verdict: str = "unreviewed"
    comment: str = ""
    false_positive_ids: frozenset[str] = frozenset()


def load_sessions(database_path: Path) -> list[Session]:
    """Читает существующий журнал SQLite, не создавая и не меняя его."""
    path = database_path.resolve()
    if not path.is_file():
        return []
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        session_rows = connection.execute(
            """SELECT id, started_at, ended_at, status, metadata_json,
                      final_risk, teacher_verdict FROM sessions
               ORDER BY started_at DESC"""
        ).fetchall()
        event_rows = connection.execute(
            """SELECT id, session_id, event_type, occurred_at, source,
                      confidence, weight, risk_total, details_json, screenshot_path
               FROM events ORDER BY occurred_at, rowid"""
        ).fetchall()
    finally:
        connection.close()

    by_session: dict[str, list[StoredEvent]] = {row["id"]: [] for row in session_rows}
    for row in event_rows:
        if row["session_id"] not in by_session:
            continue
        event = ProctorEvent(
            event_id=row["id"], type=EventType(row["event_type"]),
            occurred_at=_parse_time(row["occurred_at"]), source=row["source"],
            confidence=row["confidence"], details=json.loads(row["details_json"]),
        )
        by_session[row["session_id"]].append(
            StoredEvent(event, float(row["weight"]), float(row["risk_total"]),
                        row["screenshot_path"])
        )

    sessions = []
    for row in session_rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        if not isinstance(metadata, dict):
            metadata = {}
        sessions.append(Session(
            id=row["id"], started_at=_parse_time(row["started_at"]),
            ended_at=_parse_time(row["ended_at"]) if row["ended_at"] else None,
            status=row["status"], metadata=metadata,
            final_risk=float(row["final_risk"]), legacy_verdict=row["teacher_verdict"],
            events=tuple(by_session[row["id"]]),
        ))
    return sessions


def recalculate_risk(session: Session, false_positive_ids: Iterable[str] = ()) -> float:
    """Повторяет расчёт core.risk без событий, признанных ошибочными.

    Возвращает максимум за сессию, а не значение на момент окончания: живой
    уровень постепенно убывает, и раннее нарушение иначе исчезло бы к концу
    длинного теста.
    """
    excluded = set(false_positive_ids)
    clock = [session.started_at.timestamp()]
    scorer = RiskScorer(clock=lambda: clock[0])
    for stored in session.events:
        if stored.event.event_id in excluded:
            continue
        clock[0] = max(clock[0], stored.event.occurred_at.timestamp())
        scorer.add(stored.event)
    return scorer.peak().total


def risk_zone(score: float) -> str:
    if score > RISK_RED_ABOVE:
        return "high"
    if score >= RISK_YELLOW_FROM:
        return "medium"
    return "low"


def event_duration(stored: StoredEvent) -> float | None:
    details = stored.event.details
    value = details.get("duration_seconds", details.get("duration"))
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def summarize_events(
    session: Session, false_positive_ids: Iterable[str] = ()
) -> dict[str, EventSummary]:
    """Считает нарушения и известную длительность каждого типа."""
    excluded = set(false_positive_ids)
    result: dict[str, EventSummary] = {}
    for stored in session.events:
        if stored.event.event_id in excluded:
            continue
        name = stored.event.type.value
        previous = result.get(name, EventSummary(0, 0.0, 0))
        duration = event_duration(stored)
        result[name] = EventSummary(
            previous.count + 1,
            previous.duration_seconds + (duration or 0.0),
            previous.measured_count + int(duration is not None),
        )
    return result


def current_verdict(session: Session, review: Review | None) -> str:
    """Отдельное решение имеет приоритет над старым решением в журнале."""
    # Одна отметка ложного события ещё не означает нового вердикта.
    value = (review.verdict if review is not None and review.verdict != "unreviewed"
             else session.legacy_verdict)
    return value if value in VERDICT_LABELS else "unreviewed"


class ReviewStore:
    """Отдельная SQLite-база, не связанная записью с исходным журналом."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS reviews (
                    session_id TEXT PRIMARY KEY,
                    verdict TEXT NOT NULL,
                    comment TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS false_positives (
                    session_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    PRIMARY KEY (session_id, event_id)
                );
            """)

    @contextmanager
    def _connection(self):
        """Закрывает SQLite-дескриптор и на Windows, где он блокирует файл."""
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def load_all(self) -> dict[str, Review]:
        """Загружает решения и отметки для таблицы сессий."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT session_id, verdict, comment FROM reviews"
            ).fetchall()
            flags = connection.execute(
                "SELECT session_id, event_id FROM false_positives"
            ).fetchall()
        values = {session_id: (verdict, comment) for session_id, verdict, comment in rows}
        flagged: dict[str, set[str]] = {}
        for session_id, event_id in flags:
            flagged.setdefault(session_id, set()).add(event_id)
        return {
            session_id: Review(
                *(values.get(session_id, ("unreviewed", ""))),
                frozenset(flagged.get(session_id, set())),
            )
            for session_id in values.keys() | flagged.keys()
        }

    def load_review(self, session_id: str) -> Review:
        return self.load_all().get(session_id, Review())

    def save_review(self, session_id: str, verdict: str, comment: str) -> None:
        if verdict not in VERDICT_LABELS:
            raise ValueError(f"неизвестный вердикт: {verdict}")
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO reviews (session_id, verdict, comment, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     verdict=excluded.verdict, comment=excluded.comment,
                     updated_at=excluded.updated_at""",
                (session_id, verdict, comment, datetime.now(timezone.utc).isoformat()),
            )

    def set_false_positive(self, session_id: str, event_id: str, flagged: bool) -> None:
        with self._connection() as connection:
            if flagged:
                connection.execute(
                    "INSERT OR IGNORE INTO false_positives (session_id, event_id) VALUES (?, ?)",
                    (session_id, event_id),
                )
            else:
                connection.execute(
                    "DELETE FROM false_positives WHERE session_id=? AND event_id=?",
                    (session_id, event_id),
                )


def export_roster(
    path: Path, sessions: Iterable[Session], reviews: Mapping[str, Review]
) -> None:
    """Пишет CSV с пересчитанным риском в кодировке, понятной Excel."""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("студент", "риск", "вердикт", "комментарий"))
        for session in sessions:
            review = reviews.get(session.id, Review())
            writer.writerow((
                session.student_name,
                f"{recalculate_risk(session, review.false_positive_ids):.1f}",
                VERDICT_LABELS[current_verdict(session, reviews.get(session.id))],
                review.comment,
            ))
