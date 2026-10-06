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

from .risk import RiskScorer, RiskSnapshot
from .storage import EventStore, StoredEvent


DEFAULT_COOLDOWNS: dict[EventType, float] = {
    # Stateful vision modules emit once per continuous episode, so a new
    # episode must not be hidden by an arbitrary time cooldown.
    EventType.PHONE_DETECTED: 0.0,
    EventType.PHONE_AIMED_AT_SCREEN: 0.0,
    EventType.GAZE_DOWN: 0.0,
    EventType.GAZE_SIDE: 0.0,
    EventType.NO_FACE: 0.0,
    EventType.MULTIPLE_FACES: 0.0,
    EventType.TOO_CLOSE_TO_CAMERA: 0.0,
    # Protection callbacks may report the same OS action more than once.
    EventType.HOTKEY_BLOCKED: 0.25,
    EventType.WINDOW_SWITCHED: 1.0,
    # Protection already reports only newly observed process/service identities.
    # A type-wide cooldown would discard a different app found on the next scan.
    EventType.SUSPICIOUS_PROCESS: 0.0,
    EventType.CAPTURE_PROTECTION_FAILED: 10.0,
    EventType.REMOTE_SESSION: 60.0,
    EventType.MULTIPLE_MONITORS: 0.0,
    EventType.INJECTED_INPUT: 0.0,
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
        scorer: RiskScorer | None = None,
        queue_size: int = 256,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = uuid4().hex
        # Stored with the session, e.g. the student name for the teacher panel.
        self.metadata: dict[str, Any] = {"application": "proctoring", **(metadata or {})}
        self.database_path = database_path
        self.screenshots_dir = screenshots_dir
        self.on_recorded = on_recorded
        self.on_error = on_error
        self.cooldowns = dict(DEFAULT_COOLDOWNS if cooldowns is None else cooldowns)
        self._queue: queue.Queue[_QueuedEvent | None] = queue.Queue(queue_size)
        self._scorer = scorer or RiskScorer()
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
        safe_frame = frame.copy() if frame is not None and hasattr(frame, "copy") else frame
        now = time.monotonic()
        with self._dedupe_lock:
            # Frame copying may overlap shutdown. Accept and enqueue atomically
            # with stop(), so a successful submit is always drained by the writer.
            if self._thread is None or self._stopping:
                return False
            last_seen = self._last_seen.get(event.type)
            cooldown = self.cooldowns.get(event.type, 0.0)
            if last_seen is not None and now - last_seen < cooldown:
                return False
            try:
                self._queue.put_nowait(_QueuedEvent(event=event, frame=safe_frame))
            except queue.Full:
                pass
            else:
                self._last_seen[event.type] = now
                return True
        self._report_error("Очередь событий переполнена; событие пропущено")
        return False

    def current_risk(self) -> RiskSnapshot:
        return self._scorer.current()

    def peak_risk(self) -> RiskSnapshot:
        """Highest risk reached in this session; this is what gets stored."""
        return self._scorer.peak()

    def stop(self, timeout: float = 5.0) -> bool:
        """Drain accepted events; return whether the writer has finished.

        A zero timeout requests shutdown and polls completion without reporting
        an error, so the UI can keep processing events while storage drains.
        """
        with self._dedupe_lock:
            thread = self._thread
            if thread is None:
                return True
            self._stopping = True
        thread.join(timeout=timeout)
        if thread.is_alive():
            if timeout > 0:
                self._report_error("Хранилище событий не завершилось вовремя")
            return False
        with self._dedupe_lock:
            if self._thread is thread:
                self._thread = None
        return True

    def _report_error(self, message: str) -> None:
        if self.on_error is not None:
            self.on_error(message)

    def _run(self) -> None:
        store: EventStore | None = None
        try:
            store = EventStore(self.database_path, self.screenshots_dir)
            store.start_session(self.session_id, dict(self.metadata))
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            return

        self._ready.set()
        try:
            while True:
                try:
                    queued = self._queue.get(timeout=0.1)
                except queue.Empty:
                    with self._dedupe_lock:
                        if self._stopping and self._queue.empty():
                            break
                    continue
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
            store.finish_session(
                self.session_id,
                final_risk=self._scorer.peak().total,
            )
            store.close()

