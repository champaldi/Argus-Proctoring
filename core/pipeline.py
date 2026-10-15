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

from .live import LiveReporter
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
    EventType.SUSPICIOUS_PROCESS: 10.0,
}


# Before the student presses «Начать тест» only lasting environment problems
# are recorded: a glance or a blocked key during the camera check is not a
# violation, but a remote session or a forbidden program already running is.
PRE_TEST_EVENT_TYPES: frozenset[EventType] = frozenset({
    EventType.SUSPICIOUS_PROCESS,
    EventType.REMOTE_SESSION,
    EventType.MULTIPLE_MONITORS,
    EventType.CAPTURE_PROTECTION_FAILED,
    EventType.PROTECTION_DISABLED,
})


@dataclass(slots=True)
class _QueuedEvent:
    event: ProctorEvent
    frame: Any


@dataclass(slots=True)
class _ReferenceFrame:
    frame: Any


# The camera needs a moment to settle exposure before the control photo. The
# delay counts from the first frame offered, not from the session start: the
# models load first, so the camera opens several seconds after the session.
REFERENCE_PHOTO_DELAY_SECONDS = 2.0
# Frames darker than this mean brightness (0-255) are a camera still waking up.
REFERENCE_PHOTO_MIN_BRIGHTNESS = 40.0


def _brightness(frame: Any) -> float:
    """Mean pixel value; frames that cannot be measured count as bright."""
    try:
        return float(frame.mean())
    except Exception:
        return 255.0


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
        live: LiveReporter | None = None,
        require_test_start: bool = False,
    ) -> None:
        self.session_id = uuid4().hex
        self.metadata: dict[str, Any] = {"application": "proctoring", **(metadata or {})}
        self.database_path = database_path
        self.screenshots_dir = screenshots_dir
        self.on_recorded = on_recorded
        self.on_error = on_error
        self.live = live
        self._test_started = not require_test_start
        self.cooldowns = dict(DEFAULT_COOLDOWNS if cooldowns is None else cooldowns)
        self._queue: queue.Queue[_QueuedEvent | _ReferenceFrame | None] = queue.Queue(
            queue_size
        )
        self._scorer = scorer or RiskScorer()
        self._last_seen: dict[EventType, float] = {}
        self._dedupe_lock = threading.Lock()
        self._ready = threading.Event()
        self._startup_error: Exception | None = None
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._started_at: float | None = None
        self._reference_requested = False
        self._first_offer_at: float | None = None

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
        self._started_at = time.monotonic()

    def offer_reference_frame(self, frame: Any) -> bool:
        """Keep one calm frame from the start of the test as the control photo.

        Call it with frames that produced no violation. The first frame offered
        after a short delay is stored; later calls return at once.
        """
        if frame is None or self._reference_requested or self._started_at is None:
            return False
        if self._thread is None or self._stopping:
            return False
        now = time.monotonic()
        if self._first_offer_at is None:
            self._first_offer_at = now
        if now - self._first_offer_at < REFERENCE_PHOTO_DELAY_SECONDS:
            return False
        if _brightness(frame) < REFERENCE_PHOTO_MIN_BRIGHTNESS:
            return False
        safe_frame = frame.copy() if hasattr(frame, "copy") else frame
        try:
            self._queue.put_nowait(_ReferenceFrame(frame=safe_frame))
        except queue.Full:
            return False
        self._reference_requested = True
        return True

    def begin_test(self) -> None:
        """From now on every event counts (see PRE_TEST_EVENT_TYPES)."""
        self._test_started = True

    @property
    def test_started(self) -> bool:
        return self._test_started

    def submit(self, event: ProctorEvent, frame: Any = None) -> bool:
        if self._thread is None or self._stopping:
            return False
        if not self._test_started and event.type not in PRE_TEST_EVENT_TYPES:
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

    def current_risk(self) -> RiskSnapshot:
        return self._scorer.current()

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
            store.start_session(self.session_id, dict(self.metadata))
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            return

        if self.live is not None:
            self.live.start()
            self.live.session_started(self.session_id, self.metadata)

        self._ready.set()
        try:
            while True:
                queued = self._queue.get()
                if queued is None:
                    self._queue.task_done()
                    break
                if isinstance(queued, _ReferenceFrame):
                    try:
                        path = store.save_reference_photo(self.session_id, queued.frame)
                        if self.live is not None:
                            self.live.reference_saved(self.session_id, path)
                    except Exception as exc:
                        # Allow another attempt with a later frame.
                        self._reference_requested = False
                        self._report_error(
                            f"Не удалось сохранить контрольный кадр: {type(exc).__name__}: {exc}"
                        )
                    finally:
                        self._queue.task_done()
                    continue
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
                    if self.live is not None:
                        self.live.event_recorded(self.session_id, stored)
                except Exception as exc:
                    self._report_error(f"Ошибка записи события: {type(exc).__name__}: {exc}")
                finally:
                    self._queue.task_done()
        finally:
            final_risk = self._scorer.current().total
            store.finish_session(self.session_id, final_risk=final_risk)
            store.close()
            if self.live is not None:
                self.live.session_finished(self.session_id, final_risk)
                self.live.close()

