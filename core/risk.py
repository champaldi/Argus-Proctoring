"""Simple cumulative risk score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from events import EventType, ProctorEvent


DEFAULT_WEIGHTS: Mapping[EventType, int] = {
    EventType.PHONE_DETECTED: 8,
    EventType.PHONE_AIMED_AT_SCREEN: 25,
    EventType.GAZE_DOWN: 3,
    EventType.GAZE_SIDE: 6,
    EventType.NO_FACE: 15,
    EventType.MULTIPLE_FACES: 20,
    EventType.HOTKEY_BLOCKED: 12,
    EventType.WINDOW_SWITCHED: 15,
    EventType.SUSPICIOUS_PROCESS: 10,
}


@dataclass(frozen=True, slots=True)
class RiskUpdate:
    weight: int
    total: int
    level: str


class RiskScorer:
    def __init__(
        self,
        weights: Mapping[EventType, int] | None = None,
        maximum: int = 100,
    ) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.maximum = maximum
        self.total = 0

    def add(self, event: ProctorEvent) -> RiskUpdate:
        weight = self.weights.get(event.type, 0)
        self.total = min(self.maximum, self.total + weight)
        if self.total < 25:
            level = "low"
        elif self.total < 60:
            level = "medium"
        else:
            level = "high"
        return RiskUpdate(weight=weight, total=self.total, level=level)

