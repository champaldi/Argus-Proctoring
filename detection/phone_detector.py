"""Поиск телефона и человека на последовательных BGR-кадрах камеры."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from events import EventType, ProctorEvent


# Пороги вынесены сюда, чтобы их было легко настроить под камеру.
PHONE_CLASS_ID = 67
PERSON_CLASS_ID = 0
MIN_CONFIDENCE = 0.5
PHONE_CONFIDENCE = 0.35
PHONE_WINDOW = 5
PHONE_MIN_HITS = 3
UPPER_FRAME_FRACTION = 0.6
AIMED_SECONDS = 1.5
AIMED_MISSING_GRACE_SECONDS = 0.7
NO_PERSON_SECONDS = 3.0
EVENT_COOLDOWN_SECONDS = 2.0
ENABLE_NO_PERSON = False
DETECT_EVERY_N_FRAMES = 3
MODEL_NAME = "yolov8s.pt"


@dataclass(frozen=True)
class Detection:
    """Рамка одного объекта в координатах исходного кадра."""

    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]


class PhoneDetector:
    """Хранит счётчики и таймеры для одного потока камеры."""

    def __init__(self, *, model: Any = None) -> None:
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(Path(__file__).resolve().parent / MODEL_NAME))
        self.model = model
        self.reset()

    def reset(self) -> None:
        """Начинает новую сессию без повторной загрузки весов."""
        self.phone_hits: deque[bool] = deque(maxlen=PHONE_WINDOW)
        self.aimed_since: float | None = None
        self.aimed_missing_since: float | None = None
        self.no_person_since: float | None = None
        self.last_emitted: dict[EventType, float] = {}
        self.last_timestamp: float | None = None
        self.last_detections: list[Detection] = []
        self.frames_seen = 0

    def _emit(
        self,
        event_type: EventType,
        now: float,
        *,
        confidence: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> ProctorEvent | None:
        """Создаёт событие, если с прошлого прошло хотя бы две секунды."""
        previous = self.last_emitted.get(event_type)
        if previous is not None and now - previous < EVENT_COOLDOWN_SECONDS:
            return None
        self.last_emitted[event_type] = now
        return ProctorEvent.create(
            event_type, source="phone_detector", confidence=confidence, details=details
        )

    def detect(self, frame: np.ndarray, *, timestamp: float | None = None) -> list[ProctorEvent]:
        """Обрабатывает один BGR-кадр и возвращает новые события."""
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.shape[0] == 0
            or frame.shape[1] == 0
        ):
            raise ValueError("frame must be a nonempty HxWx3 uint8 BGR image")
        now = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            raise ValueError("timestamp must be finite")
        if self.last_timestamp is not None and now < self.last_timestamp:
            raise ValueError("timestamps must be nondecreasing; call reset for a new session")

        # На пропущенном кадре сохраняем наблюдения и начало таймеров.
        # События появятся при следующем кадре, обработанном YOLO.
        next_frame = self.frames_seen + 1
        if (next_frame - 1) % DETECT_EVERY_N_FRAMES != 0:
            self.frames_seen = next_frame
            self.last_timestamp = now
            return []

        # YOLO получает BGR-массив OpenCV напрямую. Указываем CPU явно.
        results = self.model.predict(
            frame, device="cpu", classes=[PERSON_CLASS_ID, PHONE_CLASS_ID],
            conf=PHONE_CONFIDENCE, verbose=False,
        )
        detections: list[Detection] = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if class_id == PHONE_CLASS_ID and confidence < PHONE_CONFIDENCE:
                continue
            if class_id == PERSON_CLASS_ID and confidence < MIN_CONFIDENCE:
                continue
            if class_id not in (PERSON_CLASS_ID, PHONE_CLASS_ID):
                continue
            bbox = tuple(int(round(value)) for value in box.xyxy[0].tolist())
            detections.append(Detection(class_id, confidence, bbox))

        self.last_detections = detections
        self.frames_seen = next_frame
        self.last_timestamp = now
        phones = [item for item in detections if item.class_id == PHONE_CLASS_ID]
        people = [item for item in detections if item.class_id == PERSON_CLASS_ID]
        upper_phones = [
            item for item in phones
            if (item.bbox[1] + item.bbox[3]) / 2 < frame.shape[0] * UPPER_FRAME_FRACTION
        ]
        events: list[ProctorEvent] = []

        # Один промах модели больше не обнуляет накопленные наблюдения.
        self.phone_hits.append(bool(phones))
        hits = sum(self.phone_hits)
        if hits < PHONE_MIN_HITS:
            self.last_emitted.pop(EventType.PHONE_DETECTED, None)
        elif phones:
            phone = max(phones, key=lambda item: item.confidence)
            event = self._emit(
                EventType.PHONE_DETECTED, now, confidence=phone.confidence,
                details={"bbox": list(phone.bbox), "hits_in_window": hits},
            )
            if event is not None:
                events.append(event)

        # Положение оцениваем по центру рамки телефона.
        if upper_phones:
            if (self.aimed_missing_since is not None
                    and now - self.aimed_missing_since > AIMED_MISSING_GRACE_SECONDS):
                self.aimed_since = None
                self.last_emitted.pop(EventType.PHONE_AIMED_AT_SCREEN, None)
            self.aimed_missing_since = None
            if self.aimed_since is None:
                self.aimed_since = now
            duration = now - self.aimed_since
            if duration > AIMED_SECONDS:
                phone = max(upper_phones, key=lambda item: item.confidence)
                event = self._emit(
                    EventType.PHONE_AIMED_AT_SCREEN, now, confidence=phone.confidence,
                    details={"bbox": list(phone.bbox), "duration_seconds": duration},
                )
                if event is not None:
                    events.append(event)
        elif phones:
            # Телефон виден ниже верхней зоны: направление изменилось.
            self.aimed_since = None
            self.aimed_missing_since = None
            self.last_emitted.pop(EventType.PHONE_AIMED_AT_SCREEN, None)
        elif self.aimed_since is not None:
            if self.aimed_missing_since is None:
                self.aimed_missing_since = now
            elif now - self.aimed_missing_since > AIMED_MISSING_GRACE_SECONDS:
                self.aimed_since = None
                self.aimed_missing_since = None
                self.last_emitted.pop(EventType.PHONE_AIMED_AT_SCREEN, None)

        # В общем EventType нет no_person; при включении используем NO_FACE.
        # По умолчанию проверку делает модуль взгляда, поэтому дубль выключен.
        if ENABLE_NO_PERSON and not people:
            if self.no_person_since is None:
                self.no_person_since = now
            duration = now - self.no_person_since
            if duration > NO_PERSON_SECONDS:
                event = self._emit(
                    EventType.NO_FACE, now,
                    details={"observation": "no_person", "duration_seconds": duration},
                )
                if event is not None:
                    events.append(event)
        else:
            self.no_person_since = None
            self.last_emitted.pop(EventType.NO_FACE, None)
        return events


_default_detector: PhoneDetector | None = None


def detect(frame: np.ndarray) -> list[ProctorEvent]:
    """Точка входа, которую вызывает core.detectors.DetectorAdapter."""
    global _default_detector
    if _default_detector is None:
        _default_detector = PhoneDetector()
    return _default_detector.detect(frame)


def reset_default_detector() -> None:
    """Сбрасывает состояние функции detect перед новой сессией."""
    global _default_detector
    _default_detector = None
