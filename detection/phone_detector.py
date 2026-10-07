"""Поиск телефона и человека на последовательных BGR-кадрах камеры."""

from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from events import EventType, ProctorEvent


_PROJECT_DATA = Path(__file__).resolve().parent.parent / "data"
_YOLO_CONFIG_DIR = _PROJECT_DATA / "ultralytics"
_MATPLOTLIB_CONFIG_DIR = _PROJECT_DATA / "matplotlib"
_YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_YOLO_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CONFIG_DIR))


# Пороги вынесены сюда, чтобы их было легко настроить под камеру.
PHONE_CLASS_ID = 67
PERSON_CLASS_ID = 0
MIN_CONFIDENCE = 0.5
PHONE_CONFIDENCE = 0.35
PHONE_MAX_ASPECT: float | None = None
PHONE_WINDOW = 5
PHONE_MIN_HITS = 3
# Уверенный телефон, прошедший все фильтры, фиксируется с первого же кадра:
# ждать три наблюдения нужно только для сомнительных рамок.
PHONE_INSTANT_CONFIDENCE = 0.75
DISTRACTOR_CLASS_NAMES = {
    65: "remote",
    73: "book",
    64: "mouse",
    39: "bottle",
    41: "cup",
    76: "scissors",
    78: "hair drier",
    79: "toothbrush",
}
DISTRACTOR_IOU = 0.5
USE_DISTRACTOR_CLASSES = True
USE_VERIFIER = True
VERIFIER_CACHE_IOU = 0.6
VERIFIER_CACHE_SECONDS = 1.0
UPPER_FRAME_FRACTION = 0.5
LARGE_PHONE_AREA_FRACTION = 0.08
AIMED_SECONDS = 1.5
AIMED_MISSING_GRACE_SECONDS = 0.7
NO_PERSON_SECONDS = 3.0
ENABLE_NO_PERSON = False
# Приложение вызывает детектор 6 раз в секунду: каждый второй вызов даёт
# три прогона YOLO в секунду. При каждом третьем их было два, и телефон,
# показанный на секунду, мог не попасть ни в один.
DETECT_EVERY_N_FRAMES = 2
MODEL_NAME = "yolov8s.pt"
_verifier_warning_printed = False


def _warn_verifier_unavailable(error: Exception) -> None:
    """Сообщает об отключении CLIP только один раз за процесс."""
    global _verifier_warning_printed
    if not _verifier_warning_printed:
        print(f"Предупреждение: CLIP недоступен ({error}); работаю без проверки.",
              file=sys.stderr)
        _verifier_warning_printed = True


@dataclass(frozen=True)
class Detection:
    """Рамка одного объекта в координатах исходного кадра."""

    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]
    verifier_score: float | None = None

    @property
    def aspect(self) -> float:
        """Отношение длинной стороны рамки к короткой."""
        x1, y1, x2, y2 = self.bbox
        width, height = max(0, x2 - x1), max(0, y2 - y1)
        return max(width, height) / min(width, height) if min(width, height) else math.inf


@dataclass(frozen=True)
class PhoneCandidate:
    """Сырая рамка телефона и причина её отбрасывания, если есть."""

    detection: Detection
    rejected_reason: str | None = None
    verification_ms: float | None = None


@dataclass
class FrameAnalysis:
    """Принятые объекты и все кандидаты на телефон в одном кадре."""

    detections: list[Detection]
    phone_candidates: list[PhoneCandidate]


def load_model() -> Any:
    """Загружает веса рядом с этим файлом независимо от текущей папки."""
    from ultralytics import YOLO

    return YOLO(str(Path(__file__).resolve().parent / MODEL_NAME))


def intersection_over_union(first: Detection, second: Detection) -> float:
    """Возвращает долю пересечения двух рамок относительно их объединения."""
    ax1, ay1, ax2, ay2 = first.bbox
    bx1, by1, bx2, by2 = second.bbox
    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def analyze_frame(
    frame: np.ndarray,
    model: Any,
    *,
    phone_confidence: float = PHONE_CONFIDENCE,
    max_aspect: float | None = PHONE_MAX_ASPECT,
    use_distractors: bool = USE_DISTRACTOR_CLASSES,
    use_verifier: bool | None = None,
    verifier: Callable[[np.ndarray, tuple[int, int, int, int]], float | None] | None = None,
    query_confidence: float | None = None,
) -> FrameAnalysis:
    """Один запуск YOLO без временного окна; одинаков для демо и оценки."""
    if use_verifier is None:
        use_verifier = USE_VERIFIER
    classes = [PERSON_CLASS_ID, PHONE_CLASS_ID]
    if use_distractors:
        classes.extend(DISTRACTOR_CLASS_NAMES)
    results = model.predict(
        frame, device="cpu", classes=classes,
        conf=(min(phone_confidence, MIN_CONFIDENCE) if query_confidence is None
              else query_confidence),
        verbose=False,
    )
    people: list[Detection] = []
    phones: list[Detection] = []
    distractors: list[Detection] = []
    for box in results[0].boxes:
        item = Detection(
            int(box.cls[0]), float(box.conf[0]),
            tuple(int(round(value)) for value in box.xyxy[0].tolist()),
        )
        if item.class_id == PERSON_CLASS_ID and item.confidence >= MIN_CONFIDENCE:
            people.append(item)
        elif item.class_id == PHONE_CLASS_ID:
            phones.append(item)
        elif use_distractors and item.class_id in DISTRACTOR_CLASS_NAMES:
            distractors.append(item)

    candidates: list[PhoneCandidate] = []
    accepted = list(people)
    for phone in phones:
        reason: str | None = None
        verification_ms: float | None = None
        if not math.isfinite(phone.aspect):
            reason = "invalid bbox"
        elif phone.confidence < phone_confidence:
            reason = f"confidence {phone.confidence:.2f} < {phone_confidence:.2f}"
        elif max_aspect is not None and phone.aspect > max_aspect:
            reason = f"aspect {phone.aspect:.2f} > {max_aspect:.2f}"
        elif use_distractors:
            competitors = [
                item for item in distractors
                if item.confidence > phone.confidence
                and intersection_over_union(phone, item) > DISTRACTOR_IOU
            ]
            if competitors:
                rival = max(competitors, key=lambda item: item.confidence)
                reason = (
                    f"{DISTRACTOR_CLASS_NAMES[rival.class_id]} {rival.confidence:.2f}, "
                    f"IoU {intersection_over_union(phone, rival):.2f} > {DISTRACTOR_IOU:.2f}"
                )
        if reason is None and use_verifier:
            from .verifier import VERIFIER_MIN_SCORE, verify_phone

            check = verify_phone if verifier is None else verifier
            started = time.perf_counter()
            score = check(frame, phone.bbox)
            if score is not None:
                verification_ms = (time.perf_counter() - started) * 1000
            phone = replace(phone, verifier_score=score)
            if score is not None and score < VERIFIER_MIN_SCORE:
                reason = f"verifier_score {score:.2f} < {VERIFIER_MIN_SCORE:.2f}"
        candidates.append(PhoneCandidate(phone, reason, verification_ms))
        if reason is None:
            accepted.append(phone)
    return FrameAnalysis(accepted, candidates)


class PhoneDetector:
    """Хранит счётчики и таймеры для одного потока камеры."""

    def __init__(self, *, model: Any = None) -> None:
        if model is None:
            model = load_model()
        self.model = model
        self._closed = False
        self._verifier_acquired = False
        self.verifier_enabled = USE_VERIFIER
        if self.verifier_enabled:
            try:
                from .verifier import warmup

                warmup()
                self._verifier_acquired = True
            except Exception as error:
                self.verifier_enabled = False
                _warn_verifier_unavailable(error)
        self.reset()

    def close(self) -> None:
        """Освобождает модели и наблюдения; повторный вызов безопасен."""
        if self._closed:
            return
        self._closed = True
        model = self.model
        self.model = None
        self.verifier_enabled = False
        self.reset()
        try:
            close_model = getattr(model, "close", None)
            if callable(close_model):
                close_model()
        finally:
            if self._verifier_acquired:
                from .verifier import release

                release()
                self._verifier_acquired = False

    def reset(self) -> None:
        """Начинает новую сессию без повторной загрузки весов."""
        self.phone_hits: deque[bool] = deque(maxlen=PHONE_WINDOW)
        self.instant_episode = False
        self.aimed_since: float | None = None
        self.aimed_missing_since: float | None = None
        self.no_person_since: float | None = None
        self.last_emitted: dict[EventType, float] = {}
        self.last_timestamp: float | None = None
        self.last_detections: list[Detection] = []
        self.last_phone_candidates: list[PhoneCandidate] = []
        self.verifier_cache: list[tuple[float, Detection]] = []
        self.frames_seen = 0

    def _verify_cached(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int], now: float
    ) -> float | None:
        """Повторно использует свежую оценку близкой рамки или вызывает CLIP."""
        if not self.verifier_enabled:
            return None
        self.verifier_cache = [
            entry for entry in self.verifier_cache
            if 0 <= now - entry[0] < VERIFIER_CACHE_SECONDS
        ]
        current = Detection(PHONE_CLASS_ID, 0.0, bbox)
        for _, old in reversed(self.verifier_cache):
            if intersection_over_union(current, old) > VERIFIER_CACHE_IOU:
                return old.verifier_score
        try:
            from .verifier import verify_phone

            score = verify_phone(frame, bbox)
        except Exception as error:
            self.verifier_enabled = False
            _warn_verifier_unavailable(error)
            return None
        self.verifier_cache.append((now, replace(current, verifier_score=score)))
        return score

    def _emit(
        self,
        event_type: EventType,
        now: float,
        *,
        phone_count: int,
        aspect: float | None = None,
        confidence: float | None = None,
        verifier_score: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> ProctorEvent | None:
        """Создаёт событие один раз за непрерывный эпизод нарушения."""
        if event_type in self.last_emitted:
            return None
        self.last_emitted[event_type] = now
        return ProctorEvent.create(
            event_type, source="phone_detector", confidence=confidence,
            details={**(details or {}), "phone_count": phone_count,
                     "aspect": round(aspect, 2) if aspect is not None else None,
                     "verifier_score": verifier_score},
        )

    def detect(self, frame: np.ndarray, *, timestamp: float | None = None) -> list[ProctorEvent]:
        """Обрабатывает один BGR-кадр и возвращает новые события."""
        if self._closed:
            raise RuntimeError("PhoneDetector is closed")
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

        # Фильтры рамок одинаковы для камеры и покадровой оценки.
        analysis = analyze_frame(
            frame, self.model, use_verifier=self.verifier_enabled,
            verifier=lambda image, bbox: self._verify_cached(image, bbox, now),
        )
        detections = analysis.detections

        self.last_detections = detections
        self.last_phone_candidates = analysis.phone_candidates
        self.frames_seen = next_frame
        self.last_timestamp = now
        phones = [item for item in detections if item.class_id == PHONE_CLASS_ID]
        people = [item for item in detections if item.class_id == PERSON_CLASS_ID]
        phone_count = len(phones)
        upper_phones = [
            item for item in phones
            if (
                item.bbox[1] < frame.shape[0] * UPPER_FRAME_FRACTION
                or max(0, item.bbox[2] - item.bbox[0])
                * max(0, item.bbox[3] - item.bbox[1])
                > frame.shape[0] * frame.shape[1] * LARGE_PHONE_AREA_FRACTION
            )
        ]
        events: list[ProctorEvent] = []

        # Один промах модели больше не обнуляет накопленные наблюдения.
        self.phone_hits.append(bool(phones))
        hits = sum(self.phone_hits)
        best = max(phones, key=lambda item: item.confidence) if phones else None
        if best is not None and best.confidence >= PHONE_INSTANT_CONFIDENCE:
            self.instant_episode = True
        elif hits == 0:
            self.instant_episode = False
        if hits < PHONE_MIN_HITS and not self.instant_episode:
            self.last_emitted.pop(EventType.PHONE_DETECTED, None)
        elif best is not None:
            phone = best
            event = self._emit(
                EventType.PHONE_DETECTED, now, phone_count=phone_count,
                confidence=phone.confidence, aspect=phone.aspect,
                verifier_score=phone.verifier_score,
                details={"bbox": list(phone.bbox), "hits_in_window": hits},
            )
            if event is not None:
                events.append(event)

        # Считаем телефон направленным на экран по верхнему краю или площади.
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
                    EventType.PHONE_AIMED_AT_SCREEN, now, phone_count=phone_count,
                    confidence=phone.confidence, aspect=phone.aspect,
                    verifier_score=phone.verifier_score,
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
                    EventType.NO_FACE, now, phone_count=phone_count,
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
    """Закрывает модель функции detect перед новой сессией."""
    global _default_detector
    previous = _default_detector
    _default_detector = None
    if previous is not None:
        previous.close()
