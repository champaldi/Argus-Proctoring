"""SQLite persistence and evidence screenshot storage."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from events import EventType, ProctorEvent


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    final_risk REAL NOT NULL DEFAULT 0,
    teacher_verdict TEXT,
    reviewed_at TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL,
    weight REAL NOT NULL,
    risk_total REAL NOT NULL,
    details_json TEXT NOT NULL,
    screenshot_path TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_events_session_time
ON events(session_id, occurred_at);
"""


@dataclass(frozen=True, slots=True)
class StoredEvent:
    event: ProctorEvent
    weight: float
    risk_total: float
    screenshot_path: str | None

    def to_dict(self) -> dict[str, Any]:
        value = self.event.to_dict()
        value.update(
            {
                "weight": self.weight,
                "risk_total": self.risk_total,
                "screenshot_path": self.screenshot_path,
            }
        )
        return value


class EventStore:
    def __init__(self, database_path: Path, screenshots_dir: Path) -> None:
        self.database_path = database_path
        self.screenshots_dir = screenshots_dir
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.executescript(SCHEMA)
        self._migrate_sessions_table()

    def _migrate_sessions_table(self) -> None:
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(sessions)")
        }
        migrations = {
            "final_risk": "ALTER TABLE sessions ADD COLUMN final_risk REAL NOT NULL DEFAULT 0",
            "teacher_verdict": "ALTER TABLE sessions ADD COLUMN teacher_verdict TEXT",
            "reviewed_at": "ALTER TABLE sessions ADD COLUMN reviewed_at TEXT",
        }
        for column, statement in migrations.items():
            if column not in columns:
                self.connection.execute(statement)
        self.connection.commit()

    def start_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO sessions
                (id, started_at, ended_at, status, metadata_json)
            VALUES (?, ?, NULL, 'active', ?)
            """,
            (
                session_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(metadata or {}, ensure_ascii=False, default=str),
            ),
        )
        self.connection.commit()

    def finish_session(
        self,
        session_id: str,
        status: str = "completed",
        final_risk: float = 0.0,
    ) -> None:
        self.connection.execute(
            """
            UPDATE sessions
            SET ended_at = ?, status = ?, final_risk = ?
            WHERE id = ?
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                status,
                final_risk,
                session_id,
            ),
        )
        self.connection.commit()

    def _save_screenshot(self, event: ProctorEvent, session_id: str, frame: Any) -> str:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required to save screenshots") from exc

        session_dir = self.screenshots_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        timestamp = event.occurred_at.strftime("%Y%m%dT%H%M%S_%fZ")
        destination = session_dir / f"{timestamp}_{event.type.value}_{event.event_id[:8]}.jpg"
        if not cv2.imwrite(str(destination), frame):
            raise OSError(f"could not write screenshot to {destination}")
        return str(destination)

    def record_event(
        self,
        event: ProctorEvent,
        *,
        session_id: str,
        weight: float,
        risk_total: float,
        frame: Any = None,
    ) -> StoredEvent:
        screenshot_path: str | None = None
        if frame is not None:
            try:
                screenshot_path = self._save_screenshot(event, session_id, frame)
            except Exception as exc:
                details = dict(event.details)
                details["screenshot_error"] = f"{type(exc).__name__}: {exc}"
                event = ProctorEvent(
                    type=event.type,
                    source=event.source,
                    occurred_at=event.occurred_at,
                    confidence=event.confidence,
                    details=details,
                    event_id=event.event_id,
                )

        self.connection.execute(
            """
            INSERT INTO events (
                id, session_id, event_type, occurred_at, source, confidence,
                weight, risk_total, details_json, screenshot_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                session_id,
                event.type.value,
                event.occurred_at.isoformat(),
                event.source,
                event.confidence,
                weight,
                risk_total,
                json.dumps(dict(event.details), ensure_ascii=False, default=str),
                screenshot_path,
            ),
        )
        self.connection.commit()
        return StoredEvent(event, weight, risk_total, screenshot_path)

    def close(self) -> None:
        self.connection.close()


def record_teacher_verdict(
    database_path: Path,
    session_id: str,
    verdict: str,
) -> None:
    if verdict not in {"cheated", "not_cheated"}:
        raise ValueError("unsupported teacher verdict")
    connection = sqlite3.connect(database_path)
    try:
        cursor = connection.execute(
            """
            UPDATE sessions
            SET teacher_verdict = ?, reviewed_at = ?
            WHERE id = ?
            """,
            (verdict, datetime.now(timezone.utc).isoformat(), session_id),
        )
        if cursor.rowcount != 1:
            raise LookupError(f"session not found: {session_id}")
        connection.commit()
    finally:
        connection.close()


def load_session_events(database_path: Path, session_id: str) -> list[StoredEvent]:
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            """
            SELECT id, event_type, occurred_at, source, confidence,
                   details_json, weight, risk_total, screenshot_path
            FROM events
            WHERE session_id = ?
            ORDER BY occurred_at
            """,
            (session_id,),
        ).fetchall()
    finally:
        connection.close()

    result: list[StoredEvent] = []
    for row in rows:
        event = ProctorEvent(
            event_id=row[0],
            type=EventType(row[1]),
            occurred_at=datetime.fromisoformat(row[2]),
            source=row[3],
            confidence=row[4],
            details=json.loads(row[5]),
        )
        result.append(
            StoredEvent(
                event=event,
                weight=float(row[6]),
                risk_total=float(row[7]),
                screenshot_path=row[8],
            )
        )
    return result

