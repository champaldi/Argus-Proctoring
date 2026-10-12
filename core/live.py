"""Живая отправка событий на панель преподавателя по локальной сети.

Студенческое приложение пишет каждое событие к себе в SQLite, как и раньше.
Если задан адрес панели (``PROCTOR_TEACHER_URL`` или файл ``data/teacher_url.txt``),
копия события уходит туда короткой JSON-записью: тип, время, баллы и один
уменьшенный кадр. Видео не передаётся.

Отправка идёт в отдельном потоке и никогда не тормозит камеру или тест. Если
сеть пропала, сообщения ждут в очереди и уходят, когда связь вернётся; при
переполнении очереди отбрасываются самые старые события, а не начало и конец
сессии.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .storage import StoredEvent

LIVE_PATH = "/api/live"
TOKEN_HEADER = "X-Argus-Token"
URL_FILE_NAME = "teacher_url.txt"
# Кадр события у преподавателя — это и уведомление, и доказательство в журнале.
THUMBNAIL_WIDTH = 640
THUMBNAIL_QUALITY = 75
MAX_QUEUE = 500
# Через Tailscale-ретранслятор кадр может идти несколько секунд.
REQUEST_TIMEOUT = 10.0
RETRY_MIN = 1.0
RETRY_MAX = 10.0
# Пока тест идёт, панель раз в столько секунд узнаёт, что студент на связи.
HEARTBEAT_SECONDS = 15.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_url(value: str) -> str:
    """``192.168.1.5`` → ``http://192.168.1.5:8765``; лишние пробелы и слэш убираются."""
    url = value.strip().rstrip("/")
    if not url:
        return ""
    if "://" not in url:
        url = f"http://{url}"
    host_part = url.split("://", 1)[1]
    if ":" not in host_part.split("/", 1)[0]:
        scheme, rest = url.split("://", 1)
        host, _, path = rest.partition("/")
        url = f"{scheme}://{host}:8765" + (f"/{path}" if path else "")
    return url


def teacher_url_from_env(data_dir: Path | None = None) -> str:
    value = os.getenv("PROCTOR_TEACHER_URL", "")
    if not value.strip() and data_dir is not None:
        path = data_dir / URL_FILE_NAME
        try:
            value = path.read_text(encoding="utf-8").strip().splitlines()[0]
        except (OSError, IndexError):
            value = ""
    return normalize_url(value)


def encode_thumbnail(path: str | None) -> str | None:
    """Уменьшенный JPEG кадра в base64 или ``None``, если кадра нет."""
    if not path:
        return None
    try:
        import cv2  # noqa: PLC0415 — тяжёлый импорт только при наличии кадра

        image = cv2.imread(path)
        if image is None:
            return None
        height, width = image.shape[:2]
        if width > THUMBNAIL_WIDTH:
            scale = THUMBNAIL_WIDTH / width
            image = cv2.resize(image, (THUMBNAIL_WIDTH, max(1, int(height * scale))))
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, THUMBNAIL_QUALITY])
        if not ok:
            return None
        return base64.b64encode(buffer.tobytes()).decode("ascii")
    except Exception:
        return None


class RejectedMessage(Exception):
    """Панель ответила, что сообщение неверное: повтор не поможет."""


def _post_json(url: str, payload: Mapping[str, Any], token: str | None) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json; charset=utf-8")
    if token:
        request.add_header(TOKEN_HEADER, token)
    # Системный прокси не должен перехватывать адрес в локальной сети.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=REQUEST_TIMEOUT) as response:
            if response.status >= 300:
                raise OSError(f"HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 413}:
            raise RejectedMessage(f"HTTP {exc.code}") from exc
        raise


class LiveReporter:
    """Очередь сообщений для панели преподавателя с повтором при обрыве связи."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        sender: Callable[[str, Mapping[str, Any], str | None], None] = _post_json,
        thumbnailer: Callable[[str | None], str | None] = encode_thumbnail,
    ) -> None:
        self.url = normalize_url(base_url) + LIVE_PATH
        self.token = token or None
        self._send = sender
        self._thumbnail = thumbnailer
        self._queue: deque[dict[str, Any]] = deque()
        self._condition = threading.Condition()
        self._closing = False
        self._thread: threading.Thread | None = None
        self.connected: bool | None = None  # None — ещё не пробовали
        self.sent = 0
        self.dropped = 0
        self.last_error = ""
        self._student = ""
        self._started_at = ""
        self._active_session: str | None = None
        self._last_risk = 0.0

    @classmethod
    def from_env(cls, data_dir: Path | None = None) -> "LiveReporter | None":
        url = teacher_url_from_env(data_dir)
        if not url:
            return None
        return cls(url, token=os.getenv("PROCTOR_LIVE_TOKEN") or None)

    # ---- публичный интерфейс, вызывается из потока записи событий ----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="live-reporter", daemon=True)
        self._thread.start()

    def session_started(self, session_id: str, metadata: Mapping[str, Any]) -> None:
        self._student = str(metadata.get("student_name") or "").strip()
        self._started_at = _now()
        self._active_session = session_id
        self._put({"kind": "start", "session_id": session_id})

    def event_recorded(self, session_id: str, stored: StoredEvent) -> None:
        event = stored.event.to_dict()
        self._last_risk = stored.risk_total
        self._put({
            "kind": "event",
            "session_id": session_id,
            "event": event,
            "weight": stored.weight,
            "risk_total": stored.risk_total,
            "_screenshot": stored.screenshot_path,
        })

    def reference_saved(self, session_id: str, path: str) -> None:
        """Контрольное фото студента — в журнал преподавателя."""
        self._put({"kind": "reference", "session_id": session_id, "_image_path": path})

    def session_finished(self, session_id: str, final_risk: float) -> None:
        self._active_session = None
        self._put({"kind": "finish", "session_id": session_id, "final_risk": final_risk})

    def close(self, timeout: float = 2.0) -> None:
        """Пытается дослать очередь за ``timeout`` секунд и останавливает поток."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._queue and self.connected is not False and time.monotonic() < deadline:
                self._condition.wait(timeout=max(0.0, deadline - time.monotonic()))
            self._closing = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, deadline - time.monotonic()))
            self._thread = None

    def pending(self) -> int:
        with self._condition:
            return len(self._queue)

    # ---- внутреннее ----

    def _put(self, message: dict[str, Any]) -> None:
        message["student"] = self._student
        message["started_at"] = self._started_at
        message["sent_at"] = _now()
        with self._condition:
            if len(self._queue) >= MAX_QUEUE:
                # Начало и конец сессии важнее отдельных событий.
                for index, queued in enumerate(self._queue):
                    if queued["kind"] == "event":
                        del self._queue[index]
                        self.dropped += 1
                        break
                else:
                    self._queue.popleft()
                    self.dropped += 1
            self._queue.append(message)
            self._condition.notify_all()

    def _run(self) -> None:
        delay = RETRY_MIN
        while True:
            with self._condition:
                while not self._queue and not self._closing:
                    if not self._condition.wait(timeout=HEARTBEAT_SECONDS):
                        if self._active_session is not None:
                            self._queue.append({
                                "kind": "ping", "session_id": self._active_session,
                                "risk_total": self._last_risk, "student": self._student,
                                "started_at": self._started_at, "sent_at": _now(),
                            })
                if self._closing and (not self._queue or self.connected is False):
                    return
                message = self._queue[0]
            payload = {key: value for key, value in message.items() if not key.startswith("_")}
            if "_screenshot" in message and "thumbnail" not in message:
                thumbnail = self._thumbnail(message.get("_screenshot"))
                if thumbnail:
                    message["thumbnail"] = thumbnail
            if "thumbnail" in message:
                payload["thumbnail"] = message["thumbnail"]
            if "_image_path" in message and "image" not in message:
                message["image"] = self._thumbnail(message.get("_image_path")) or ""
            if message.get("image"):
                payload["image"] = message["image"]
            try:
                self._send(self.url, payload, self.token)
            except RejectedMessage as exc:
                # Сломанное сообщение не должно навсегда заблокировать очередь.
                self.connected = True
                self.last_error = str(exc)
                self.dropped += 1
                with self._condition:
                    if self._queue and self._queue[0] is message:
                        self._queue.popleft()
                    self._condition.notify_all()
                continue
            except Exception as exc:
                self.connected = False
                self.last_error = f"{type(exc).__name__}: {exc}"
                with self._condition:
                    if self._closing:
                        return
                    self._condition.wait(timeout=delay)
                delay = min(RETRY_MAX, delay * 2)
                continue
            delay = RETRY_MIN
            self.connected = True
            self.last_error = ""
            self.sent += 1
            with self._condition:
                if self._queue and self._queue[0] is message:
                    self._queue.popleft()
                self._condition.notify_all()
