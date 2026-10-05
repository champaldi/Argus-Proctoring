"""Thread-safe event queue, deduplication, scoring and persistence."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from events import EventType, ProctorEvent

from .risk import RiskScorer
from .storage import EventStore, StoredEvent


DEFAULT_COOLDOWNS: dict[EventType, float] = {
    EventType.PHONE_DETECTED: 3.0,
    EventType.PHONE_AIMED_AT_SCREEN: 5.0,
    EventType.GAZE_DOWN: 3.0,
    EventType.GAZE_SIDE: 3.0,
    EventType.NO_FACE: 5.0,
    EventType.MULTIPLE_FACES: 5.0,
    EventType.HOTKEY_BLOCKED: 0.5,
    EventType.WINDOW_SWITCHED: 2.0,
    EventType.SUSPICIOUS_PROCESS: 10.0,
}


@dataclass(slots=True)
class _QueuedEvent:
    event: ProctorEvent
    frame: Any


class EventPipeline:
    def __init__(
        self,
        database_path: Path,
        screenshots_dir: Path,
        *,
        on_recorded: Callable[[StoredEvent], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        cooldowns: dict[EventType, float] | None = None,
        queue_size: int = 256,
    ) -> None:
        self.session_id = uuid4().hex
        self.database_path = database_path
        self.screenshots_dir = screenshots_dir
        self.on_recorded = on_recorded
        self.on_error = on_error
        self.cooldowns = dict(cooldowns or DEFAULT_COOLDOWNS)
        self._queue: queue.Queue[_QueuedEvent | None] = queue.Queue(queue_size)
        self._scorer = RiskScorer()
        self._last_seen: dict[EventType, float] = {}
        self._dedupe_lock = threading.Lock()
        self._ready = threading.Event()
        self._startup_error: Exception | None = None
        self._thread: threading.Thread | None = None
        self._stopping = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="event-writer",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise TimeoutError("event storage did not start in time")
        if self._startup_error is not None:
            raise RuntimeError("could not start event storage") from self._startup_error

    def submit(self, event: ProctorEvent, frame: Any = None) -> bool:
        if self._thread is None or self._stopping:
            return False
        now = time.monotonic()
        with self._dedupe_lock:
            last_seen = self._last_seen.get(event.type)
            cooldown = self.cooldowns.get(event.type, 0.0)
            if last_seen is not None and now - last_seen < cooldown:
                return False
            self._last_seen[event.type] = now

        safe_frame = frame.copy() if frame is not None and hasattr(frame, "copy") else frame
        try:
            self._queue.put_nowait(_QueuedEvent(event=event, frame=safe_frame))
        except queue.Full:
            with self._dedupe_lock:
                self._last_seen.pop(event.type, None)
            self._report_error("Очередь событий переполнена; событие пропущено")
            return False
        return True

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stopping = True
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            self._report_error("Не удалось корректно закрыть очередь событий")
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            self._report_error("Хранилище событий не завершилось вовремя")
        self._thread = None

    def _report_error(self, message: str) -> None:
        if self.on_error is not None:
            self.on_error(message)

    def _run(self) -> None:
        store: EventStore | None = None
        try:
            store = EventStore(self.database_path, self.screenshots_dir)
            store.start_session(self.session_id, {"application": "proctoring"})
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            return

        self._ready.set()
        try:
            while True:
                queued = self._queue.get()
                if queued is None:
                    self._queue.task_done()
                    break
                try:
                    update = self._scorer.add(queued.event)
                    stored = store.record_event(
                        queued.event,
                        session_id=self.session_id,
                        weight=update.weight,
                        risk_total=update.total,
                        frame=queued.frame,
                    )
                    if self.on_recorded is not None:
                        self.on_recorded(stored)
                except Exception as exc:
                    self._report_error(f"Ошибка записи события: {type(exc).__name__}: {exc}")
                finally:
                    self._queue.task_done()
        finally:
            store.finish_session(self.session_id)
            store.close()

