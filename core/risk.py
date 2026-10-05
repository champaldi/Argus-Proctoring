"""Configurable risk level with combinations and gradual decay."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from config import (
    RISK_COMBINATION_MULTIPLIER,
    RISK_COMBINATION_WINDOW_SECONDS,
    RISK_DECAY_PER_SECOND,
    RISK_MAXIMUM,
    RISK_RED_ABOVE,
    RISK_WEIGHTS,
    RISK_YELLOW_FROM,
)
from events import EventType, ProctorEvent


DEFAULT_WEIGHTS: Mapping[EventType, float] = {
    EventType(name): weight for name, weight in RISK_WEIGHTS.items()
}


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    total: float
    level: str


@dataclass(frozen=True, slots=True)
class RiskUpdate(RiskSnapshot):
    weight: float
    multiplier: float


class RiskScorer:
    """Scores recorded episodes, not every frame where a signal stays active."""

    def __init__(
        self,
        weights: Mapping[EventType, float] | None = None,
        *,
        maximum: float = RISK_MAXIMUM,
        combination_window: float = RISK_COMBINATION_WINDOW_SECONDS,
        combination_multiplier: float = RISK_COMBINATION_MULTIPLIER,
        decay_per_second: float = RISK_DECAY_PER_SECOND,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.maximum = float(maximum)
        self.combination_window = float(combination_window)
        self.combination_multiplier = float(combination_multiplier)
        self.decay_per_second = float(decay_per_second)
        self._clock = clock
        self._total = 0.0
        self._last_update = float(clock())
        self._recent: deque[tuple[float, EventType]] = deque()
        self._lock = threading.RLock()

    @staticmethod
    def _level(total: float) -> str:
        if total < RISK_YELLOW_FROM:
            return "low"
        if total <= RISK_RED_ABOVE:
            return "medium"
        return "high"

    def _apply_decay(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_update)
        if elapsed:
            self._total = max(0.0, self._total - elapsed * self.decay_per_second)
            self._last_update = now

    def _prune_recent(self, now: float) -> None:
        cutoff = now - self.combination_window
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()

    def add(self, event: ProctorEvent) -> RiskUpdate:
        with self._lock:
            now = float(self._clock())
            self._apply_decay(now)
            self._prune_recent(now)
            combined = any(event_type != event.type for _, event_type in self._recent)
            multiplier = self.combination_multiplier if combined else 1.0
            weight = self.weights.get(event.type, 0.0) * multiplier
            self._total = min(self.maximum, self._total + weight)
            self._recent.append((now, event.type))
            total = round(self._total, 2)
            return RiskUpdate(
                total=total,
                level=self._level(total),
                weight=round(weight, 2),
                multiplier=multiplier,
            )

    def current(self) -> RiskSnapshot:
        with self._lock:
            now = float(self._clock())
            self._apply_decay(now)
            self._prune_recent(now)
            total = round(self._total, 2)
            return RiskSnapshot(total=total, level=self._level(total))
