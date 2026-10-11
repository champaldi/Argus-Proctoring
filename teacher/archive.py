"""Центральный журнал: сессии студентов из живого режима — в базу преподавателя.

Сервер панели (``teacher/live.py``) передаёт сюда каждое сообщение студента.
Архив пишет его в ту же SQLite-схему, что и приложение студента: сессия,
события, кадры в ``screenshots/<id сессии>/`` и контрольное фото. Поэтому
сессия с другого ноутбука появляется в общей таблице панели, открывается с
правилами и кадрами, попадает в ведомость и HTML-отчёт и переживает перезапуск
панели.

Запись идемпотентна: сессия и события узнаются по своим ID, повторная доставка
после обрыва связи и тест на том же компьютере не создают дублей.
"""

from __future__ import annotations

import base64
import binascii
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.storage import REFERENCE_PHOTO_NAME, EventStore
from events import normalize_events

SOURCE_LABEL = "live"


def _iso(value: object) -> str:
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        except ValueError:
            pass
    return datetime.now(timezone.utc).isoformat()


def _decode_image(value: object) -> bytes | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    # Только JPEG: сервер не записывает на диск произвольные файлы.
    return data if data.startswith(b"\xff\xd8") else None


def _safe_id(value: object) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 64 or not all(char.isalnum() or char in "-_" for char in text):
        raise ValueError("bad id")
    return text


class SessionArchive:
    """Потокобезопасная запись живых сессий в базу панели."""

    def __init__(self, database_path: Path, screenshots_dir: Path | None = None) -> None:
        self.database_path = database_path
        self.screenshots_dir = screenshots_dir or database_path.parent / "screenshots"
        # Создаёт файл, схему и миграции тем же кодом, что у студента.
        EventStore(self.database_path, self.screenshots_dir).close()
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def record(self, message: Mapping[str, Any]) -> None:
        kind = message.get("kind")
        if kind == "ping":
            return
        session_id = _safe_id(message.get("session_id"))
        with self._lock:
            connection = self._connect()
            try:
                self._ensure_session(connection, session_id, message)
                if kind == "event":
                    self._record_event(connection, session_id, message)
                elif kind == "reference":
                    self._record_reference(connection, session_id, message)
                elif kind == "finish":
                    connection.execute(
                        """UPDATE sessions SET ended_at = ?, status = 'completed', final_risk = ?
                           WHERE id = ? AND ended_at IS NULL""",
                        (_iso(message.get("sent_at")),
                         float(message.get("final_risk") or 0.0), session_id),
                    )
                connection.commit()
            finally:
                connection.close()

    def _ensure_session(
        self, connection: sqlite3.Connection, session_id: str, message: Mapping[str, Any]
    ) -> None:
        metadata = {"source": SOURCE_LABEL}
        student = str(message.get("student") or "").strip()
        if student:
            metadata["student_name"] = student
        connection.execute(
            """INSERT OR IGNORE INTO sessions (id, started_at, ended_at, status, metadata_json)
               VALUES (?, ?, NULL, 'active', ?)""",
            (session_id, _iso(message.get("started_at") or message.get("sent_at")),
             json.dumps(metadata, ensure_ascii=False)),
        )

    def _session_dir(self, session_id: str) -> Path:
        folder = self.screenshots_dir / session_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _record_event(
        self, connection: sqlite3.Connection, session_id: str, message: Mapping[str, Any]
    ) -> None:
        raw = message.get("event")
        if not isinstance(raw, Mapping):
            raise ValueError("event payload is missing")
        try:
            event = normalize_events(raw, default_source=SOURCE_LABEL)[0]
        except (ValueError, TypeError, IndexError):
            return  # тип из более новой версии приложения
        event_id = _safe_id(event.event_id)
        exists = connection.execute(
            "SELECT 1 FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        if exists:
            return
        screenshot_path = None
        image = _decode_image(message.get("thumbnail"))
        if image is not None:
            stamp = event.occurred_at.strftime("%Y%m%dT%H%M%S_%fZ")
            path = self._session_dir(session_id) / (
                f"{stamp}_{event.type.value}_{event_id[:8]}.jpg"
            )
            path.write_bytes(image)
            screenshot_path = str(path)
        connection.execute(
            """INSERT OR IGNORE INTO events (
                   id, session_id, event_type, occurred_at, source, confidence,
                   weight, risk_total, details_json, screenshot_path
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id, session_id, event.type.value, event.occurred_at.isoformat(),
                event.source, event.confidence,
                float(message.get("weight") or 0.0), float(message.get("risk_total") or 0.0),
                json.dumps(dict(event.details), ensure_ascii=False, default=str),
                screenshot_path,
            ),
        )
        risk = float(message.get("risk_total") or 0.0)
        connection.execute(
            "UPDATE sessions SET final_risk = MAX(final_risk, ?) WHERE id = ? AND ended_at IS NULL",
            (risk, session_id),
        )

    def _record_reference(
        self, connection: sqlite3.Connection, session_id: str, message: Mapping[str, Any]
    ) -> None:
        image = _decode_image(message.get("image"))
        if image is None:
            return
        row = connection.execute(
            "SELECT metadata_json FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        try:
            metadata = json.loads(row[0] or "{}") if row else {}
        except json.JSONDecodeError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        existing = metadata.get("reference_photo")
        if isinstance(existing, str) and Path(existing).is_file():
            return  # студент сдавал на этом же компьютере
        path = self._session_dir(session_id) / REFERENCE_PHOTO_NAME
        path.write_bytes(image)
        metadata["reference_photo"] = str(path)
        connection.execute(
            "UPDATE sessions SET metadata_json = ? WHERE id = ?",
            (json.dumps(metadata, ensure_ascii=False, default=str), session_id),
        )
