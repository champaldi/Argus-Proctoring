"""Живой экзамен: приём событий от студентов по локальной сети.

Панель преподавателя поднимает небольшой HTTP-сервер (только стандартная
библиотека Python). Приложение студента присылает начало сессии, каждое
событие с уменьшенным кадром и конец сессии. ``LiveBoard`` собирает их по
студентам и прогоняет те же правила корреляции, что и разбор после теста:
уведомление приходит не на каждый взгляд, а когда сработало правило.

Полный журнал с кадрами по-прежнему лежит на ноутбуке студента; сюда приходит
только то, что нужно для оповещения.
"""

from __future__ import annotations

import html
import hmac
import json
import queue
import socket
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from core.event_presentation import EVENT_LABELS
from core.live import LIVE_PATH, TOKEN_HEADER
from core.storage import StoredEvent
from events import normalize_events

from .archive import SessionArchive
from .data import (
    VERDICT_LABELS,
    ReviewStore,
    Session,
    current_verdict,
    excluded_event_ids,
    load_sessions,
    recalculate_risk,
    risk_zone,
)
from .rules import LEVEL_TITLES, SUSPICIOUS, VIOLATION, evaluate_rules

DEFAULT_PORT = 8765
MAX_BODY = 2 * 1024 * 1024
OFFLINE_AFTER_SECONDS = 45.0
FEED_SIZE = 50
# Кадры в памяти для уведомлений; полный набор лежит в журнале на диске.
MAX_THUMBNAILS_PER_SESSION = 60
INFO = "info"


def _parse_time(value: object) -> datetime:
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Alert:
    session_id: str
    student: str
    level: str  # violation, suspicious или info
    title: str
    text: str
    occurred_at: datetime
    rule_id: str = ""
    thumbnail: str | None = None  # JPEG в base64


@dataclass(slots=True)
class LiveSession:
    session_id: str
    student: str
    started_at: datetime
    risk: float = 0.0
    finished: bool = False
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    events: list[StoredEvent] = field(default_factory=list)
    thumbnails: dict[str, str] = field(default_factory=dict)
    alerted_rules: set[str] = field(default_factory=set)
    main_rule: str = ""  # самое серьёзное из сработавших правил
    worst_level: str = ""
    rule_count: int = 0

    @property
    def rule_summary(self) -> str:
        """«V6 · Подмена… (+2)»: главное правило и сколько ещё сработало."""
        if not self.main_rule:
            return ""
        extra = self.rule_count - 1
        return f"{self.main_rule} (+{extra})" if extra > 0 else self.main_rule

    @property
    def display_name(self) -> str:
        return self.student or f"Не указан · {self.session_id[:8]}"

    def status(self, now: datetime | None = None) -> str:
        if self.finished:
            return "Завершил"
        now = now or datetime.now(timezone.utc)
        if (now - self.last_seen).total_seconds() > OFFLINE_AFTER_SECONDS:
            return "Нет связи"
        return "Идёт тест"

    def as_session(self) -> Session:
        return Session(
            id=self.session_id,
            started_at=self.started_at,
            ended_at=None,
            status="active",
            metadata={"student_name": self.student},
            final_risk=self.risk,
            legacy_verdict=None,
            events=tuple(self.events),
        )


class LiveBoard:
    """Состояние живого экзамена; безопасно вызывать из разных потоков."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, LiveSession] = {}
        self._feed: list[Alert] = []

    def apply(self, message: Mapping[str, Any]) -> list[Alert]:
        """Учитывает одно сообщение студента и возвращает новые уведомления."""
        kind = message.get("kind")
        session_id = str(message.get("session_id") or "").strip()
        if kind not in {"start", "event", "finish", "ping", "reference"} or not session_id:
            raise ValueError("unknown live message")
        now = datetime.now(timezone.utc)
        with self._lock:
            live = self._sessions.get(session_id)
            if live is None:
                live = LiveSession(
                    session_id=session_id,
                    student=str(message.get("student") or "").strip(),
                    started_at=_parse_time(message.get("started_at")),
                )
                self._sessions[session_id] = live
            elif message.get("student") and not live.student:
                live.student = str(message["student"]).strip()
            live.last_seen = now
            alerts: list[Alert] = []
            if kind == "start":
                alerts.append(Alert(
                    session_id, live.student, INFO, "Начал тест",
                    f"{live.display_name} начал тест", now,
                ))
            elif kind == "event":
                alerts.extend(self._apply_event(live, message))
            elif kind == "finish":
                live.finished = True
                live.risk = float(message.get("final_risk") or live.risk)
                alerts.append(Alert(
                    session_id, live.student, INFO, "Завершил тест",
                    f"{live.display_name} завершил тест, риск {live.risk:.0f}", now,
                ))
            elif kind == "ping" and message.get("risk_total") is not None:
                live.risk = float(message["risk_total"])
            for alert in alerts:
                if alert.level != INFO:
                    live.rule_count = len(live.alerted_rules)
                    upgrade = alert.level == VIOLATION and live.worst_level != VIOLATION
                    if upgrade or not live.main_rule:
                        live.main_rule = f"{alert.rule_id} · {alert.title}"
                        live.worst_level = alert.level
            self._feed = (alerts[::-1] + self._feed)[:FEED_SIZE]
            return alerts

    def _apply_event(self, live: LiveSession, message: Mapping[str, Any]) -> list[Alert]:
        raw = message.get("event")
        if not isinstance(raw, Mapping):
            raise ValueError("event payload is missing")
        try:
            event = normalize_events(raw, default_source="live")[0]
        except (ValueError, TypeError, IndexError):
            # Новый тип события из более свежей версии: связь жива, правило — нет.
            return []
        if any(stored.event.event_id == event.event_id for stored in live.events):
            return []  # повторная доставка после обрыва связи
        stored = StoredEvent(
            event=event,
            weight=float(message.get("weight") or 0.0),
            risk_total=float(message.get("risk_total") or 0.0),
            screenshot_path=None,
        )
        live.events.append(stored)
        live.risk = stored.risk_total
        thumbnail = message.get("thumbnail")
        if isinstance(thumbnail, str) and thumbnail:
            live.thumbnails[event.event_id] = thumbnail
            while len(live.thumbnails) > MAX_THUMBNAILS_PER_SESSION:
                live.thumbnails.pop(next(iter(live.thumbnails)))

        alerts = []
        for finding in evaluate_rules(live.as_session()):
            if finding.rule_id in live.alerted_rules:
                continue
            live.alerted_rules.add(finding.rule_id)
            picture = next(
                (live.thumbnails[event_id] for event_id in finding.event_ids
                 if event_id in live.thumbnails),
                None,
            )
            alerts.append(Alert(
                live.session_id, live.student, finding.level,
                finding.title,
                f"{live.display_name}: {finding.explanation}",
                finding.occurred_at, finding.rule_id, picture,
            ))
        return alerts

    def sessions(self) -> list[LiveSession]:
        """Копии сессий: сначала идущие, затем по риску."""
        with self._lock:
            items = [
                LiveSession(
                    session_id=item.session_id, student=item.student,
                    started_at=item.started_at, risk=item.risk, finished=item.finished,
                    last_seen=item.last_seen, main_rule=item.main_rule,
                    worst_level=item.worst_level, rule_count=item.rule_count,
                )
                for item in self._sessions.values()
            ]
        return sorted(items, key=lambda item: (item.finished, -item.risk))

    def feed(self) -> list[Alert]:
        with self._lock:
            return list(self._feed)


def event_name(event_type: str) -> str:
    return EVENT_LABELS.get(event_type, event_type)


def local_addresses() -> list[str]:
    """IPv4-адреса компьютера в локальной сети, которые нужно дать студентам."""
    found: list[str] = []
    try:
        # UDP connect ничего не отправляет, только выбирает основной интерфейс.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            found.append(probe.getsockname()[0])
    except OSError:
        pass
    try:
        for address in socket.gethostbyname_ex(socket.gethostname())[2]:
            if address not in found:
                found.append(address)
    except OSError:
        pass
    usable = [address for address in found if _is_student_reachable(address)]
    # Основной адрес сети первым, затем Tailscale; виртуальные адаптеры не нужны.
    primary = usable[:1]
    tailscale = [address for address in usable if _is_tailscale(address)]
    result = list(dict.fromkeys(primary + tailscale))
    return result or ["127.0.0.1"]


def _is_tailscale(address: str) -> bool:
    """Tailscale раздаёт адреса из 100.64.0.0/10."""
    parts = address.split(".")
    return len(parts) == 4 and parts[0] == "100" and 64 <= int(parts[1]) <= 127


def _is_student_reachable(address: str) -> bool:
    parts = address.split(".")
    if len(parts) != 4 or not all(part.isdigit() for part in parts):
        return False
    if parts[0] in {"127", "169"} or parts[3] in {"0", "255"}:
        return False  # петля, APIPA и адреса сетей виртуальных адаптеров
    # VirtualBox и VMware Host-Only: студенты сюда не подключатся.
    return not address.startswith(("192.168.56.", "192.168.99."))


_LEVEL_COLORS = {VIOLATION: "#B91C1C", SUSPICIOUS: "#B45309", INFO: "#374151"}
_ZONE_COLORS = {"high": "#B91C1C", "medium": "#B45309", "low": "#15803D"}


def _token_query(token: str | None) -> str:
    return f"?token={quote(token)}" if token else ""


def _journal_rows(
    sessions: list[Session], reviews: ReviewStore | None, token: str | None
) -> list[str]:
    values = reviews.load_all() if reviews is not None else {}
    rows = []
    for session in sessions:
        review = values.get(session.id)
        score = recalculate_risk(session, excluded_event_ids(session, review))
        color = _ZONE_COLORS[risk_zone(score)]
        verdict = VERDICT_LABELS[current_verdict(session, review)].capitalize()
        link = f"/report/{quote(session.id)}{_token_query(token)}"
        rows.append(
            "<tr>"
            f"<td><a href='{link}'>{html.escape(session.student_name)}</a></td>"
            f"<td>{session.started_at.astimezone():%d.%m %H:%M}</td>"
            f"<td style='color:{color};font-weight:700'>{score:.0f}</td>"
            f"<td>{html.escape(verdict)}</td>"
            "</tr>"
        )
    return rows


def render_page(
    board: LiveBoard,
    sessions: list[Session] | None = None,
    reviews: ReviewStore | None = None,
    token: str | None = None,
) -> str:
    """Страница для браузера или телефона: идёт экзамен, уведомления, журнал."""
    rows = []
    for item in board.sessions():
        color = _ZONE_COLORS[risk_zone(item.risk)]
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.display_name)}</td>"
            f"<td>{html.escape(item.status())}</td>"
            f"<td style='color:{color};font-weight:700'>{item.risk:.0f}</td>"
            f"<td>{html.escape(item.rule_summary or '—')}</td>"
            "</tr>"
        )
    alerts = []
    for alert in board.feed()[:10]:
        color = _LEVEL_COLORS.get(alert.level, "#374151")
        level = LEVEL_TITLES.get(alert.level, "")
        picture = (
            f"<img src='data:image/jpeg;base64,{alert.thumbnail}' alt=''>"
            if alert.thumbnail else ""
        )
        when = alert.occurred_at.astimezone().strftime("%H:%M:%S")
        alerts.append(
            f"<div class='alert' style='border-color:{color}'>{picture}<div>"
            f"<b style='color:{color}'>{html.escape(level or alert.title)}</b> "
            f"<span class='time'>{when}</span><br>"
            f"{html.escape(alert.title if level else '')}"
            f"{'<br>' if level else ''}{html.escape(alert.text)}</div></div>"
        )
    journal = ""
    if sessions is not None:
        journal_rows = _journal_rows(sessions, reviews, token)
        journal = (
            "<h2>Журнал сессий</h2><p class='time'>Имя открывает полный отчёт с кадрами</p>"
            "<table><tr><th>Студент</th><th>Начало</th><th>Риск</th><th>Решение</th></tr>"
            + ("".join(journal_rows) or "<tr><td colspan='4'>Сессий пока нет</td></tr>")
            + "</table>"
        )
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="5"><title>Argus · Идёт экзамен</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;margin:16px;background:#F8F9FA;color:#1F2937}}
h1{{font-size:20px;margin:0 0 12px}} h2{{font-size:16px;margin:20px 0 8px}}
table{{width:100%;border-collapse:collapse;background:#fff}}
td,th{{padding:8px;border-bottom:1px solid #E5E7EB;text-align:left;font-size:14px}}
a{{color:#1E3A8A}}
.alert{{display:flex;gap:10px;background:#fff;border-left:4px solid;padding:8px;margin:6px 0;font-size:14px}}
.alert img{{width:120px;height:auto;border-radius:6px}} .time{{color:#6B7280;font-size:13px}}
</style></head><body>
<h1>Argus · Идёт экзамен</h1>
<table><tr><th>Студент</th><th>Статус</th><th>Риск</th><th>Главное правило</th></tr>
{''.join(rows) or "<tr><td colspan='4'>Сейчас никто не сдаёт тест</td></tr>"}</table>
<h2>Уведомления</h2>{''.join(alerts) or '<p>Нарушений пока нет</p>'}
{journal}
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    server: "LiveServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return  # не засорять консоль преподавателя

    def _reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: Mapping[str, Any]) -> None:
        self._reply(status, json.dumps(value).encode("utf-8"), "application/json")

    def _authorized(self, supplied: str | None) -> bool:
        token = self.server.token
        return not token or hmac.compare_digest((supplied or "").encode(), token.encode())

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._json(200, {"ok": True, "app": "argus-teacher"})
            return
        if parsed.path == "/" or parsed.path.startswith("/report/"):
            supplied = parse_qs(parsed.query).get("token", [None])[0]
            if not self._authorized(supplied):
                self._reply(403, "Нужен ?token=…".encode("utf-8"), "text/plain; charset=utf-8")
                return
            try:
                if parsed.path == "/":
                    page = self.server.render_index()
                else:
                    page = self.server.render_report(parsed.path.removeprefix("/report/"))
            except Exception as exc:  # noqa: BLE001 — страница не должна ронять сервер
                self._reply(500, f"Ошибка: {exc}".encode("utf-8"), "text/plain; charset=utf-8")
                return
            if page is None:
                self._reply(404, "Сессия не найдена".encode("utf-8"), "text/plain; charset=utf-8")
                return
            self._reply(200, page.encode("utf-8"), "text/html; charset=utf-8")
            return
        self._json(404, {"ok": False})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != LIVE_PATH:
            self._json(404, {"ok": False})
            return
        if not self._authorized(self.headers.get(TOKEN_HEADER)):
            self._json(403, {"ok": False, "error": "token"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length <= 0 or length > MAX_BODY:
            self._json(413 if length > MAX_BODY else 400, {"ok": False})
            return
        try:
            message = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(message, dict):
                raise ValueError("object expected")
            if message.get("kind") not in {"start", "event", "finish", "ping", "reference"}:
                raise ValueError("unknown live message")
            if self.server.archive is not None:
                # Сначала журнал: если диск или база недоступны, студент повторит.
                self.server.archive.record(message)
            alerts = self.server.board.apply(message)
        except (ValueError, TypeError, UnicodeDecodeError):
            self._json(400, {"ok": False})
            return
        except Exception:  # noqa: BLE001 — база занята или нет места: повторить позже
            self._json(503, {"ok": False})
            return
        for alert in alerts:
            self.server.alerts.put(alert)
        self.server.changed.set()
        if self.server.archive is not None and message.get("kind") != "ping":
            self.server.archived.set()
        self._json(200, {"ok": True})


class LiveServer(ThreadingHTTPServer):
    """HTTP-сервер панели. ``alerts`` читает интерфейс по таймеру."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        board: LiveBoard,
        *,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        token: str | None = None,
        archive: SessionArchive | None = None,
        reviews_path: Path | None = None,
    ) -> None:
        super().__init__((host, port), _Handler)
        self.board = board
        self.token = token or None
        self.archive = archive
        self.reviews_path = reviews_path
        self.alerts: queue.Queue[Alert] = queue.Queue()
        self.changed = threading.Event()
        # Журнал на диске изменился: таблица сессий панели перечитывает базу.
        self.archived = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def _reviews(self) -> ReviewStore | None:
        return ReviewStore(self.reviews_path) if self.reviews_path is not None else None

    def render_index(self) -> str:
        sessions = None
        if self.archive is not None:
            sessions = load_sessions(self.archive.database_path)
        return render_page(self.board, sessions, self._reviews(), self.token)

    def render_report(self, session_id: str) -> str | None:
        if self.archive is None:
            return None
        from .report import render_session_report  # noqa: PLC0415

        database = self.archive.database_path
        session = next((item for item in load_sessions(database) if item.id == session_id), None)
        if session is None:
            return None
        reviews = self._reviews()
        review = reviews.load_review(session.id) if reviews is not None else None
        return render_session_report(session, review, database)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self.serve_forever, name="live-server", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def drain(self) -> list[Alert]:
        drained = []
        while True:
            try:
                drained.append(self.alerts.get_nowait())
            except queue.Empty:
                return drained
