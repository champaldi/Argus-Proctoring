"""Shared event contract for every proctoring module.

Detector authors should return ``list[ProctorEvent]``.  The integration layer
also accepts strings and mappings to make early prototypes easy to merge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from uuid import uuid4


class EventType(str, Enum):
    PHONE_DETECTED = "phone_detected"
    PHONE_AIMED_AT_SCREEN = "phone_aimed_at_screen"
    GAZE_DOWN = "gaze_down"
    GAZE_SIDE = "gaze_side"
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    TOO_CLOSE_TO_CAMERA = "too_close_to_camera"
    HOTKEY_BLOCKED = "hotkey_blocked"
    WINDOW_SWITCHED = "window_switched"
    SUSPICIOUS_PROCESS = "suspicious_process"
    CAPTURE_PROTECTION_FAILED = "capture_protection_failed"
    REMOTE_SESSION = "remote_session"
    MULTIPLE_MONITORS = "multiple_monitors"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class ProctorEvent:
    """One immutable observation produced by a detector or protection module."""

    type: EventType
    source: str
    occurred_at: datetime = field(default_factory=utc_now)
    confidence: float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            object.__setattr__(self, "type", EventType(self.type))
        if not self.source.strip():
            raise ValueError("event source must not be empty")
        if self.occurred_at.tzinfo is None:
            object.__setattr__(
                self,
                "occurred_at",
                self.occurred_at.replace(tzinfo=timezone.utc),
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @classmethod
    def create(
        cls,
        event_type: EventType | str,
        *,
        source: str,
        confidence: float | None = None,
        details: Mapping[str, Any] | None = None,
        occurred_at: datetime | None = None,
    ) -> "ProctorEvent":
        return cls(
            type=EventType(event_type),
            source=source,
            occurred_at=occurred_at or utc_now(),
            confidence=confidence,
            details=details or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type.value,
            "source": self.source,
            "occurred_at": self.occurred_at.isoformat(),
            "confidence": self.confidence,
            "details": dict(self.details),
        }


RawEvent = ProctorEvent | str | Mapping[str, Any]


def _from_mapping(value: Mapping[str, Any], default_source: str) -> ProctorEvent:
    event_type = value.get("type", value.get("event_type"))
    if event_type is None:
        raise ValueError("event mapping must contain 'type' or 'event_type'")

    occurred_at = value.get("occurred_at")
    if isinstance(occurred_at, str):
        occurred_at = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
    elif occurred_at is not None and not isinstance(occurred_at, datetime):
        raise TypeError("occurred_at must be a datetime or ISO-8601 string")

    reserved = {
        "type",
        "event_type",
        "source",
        "occurred_at",
        "confidence",
        "details",
        "event_id",
    }
    details = dict(value.get("details") or {})
    details.update({key: item for key, item in value.items() if key not in reserved})
    return ProctorEvent(
        type=EventType(event_type),
        source=str(value.get("source") or default_source),
        occurred_at=occurred_at or utc_now(),
        confidence=(
            float(value["confidence"])
            if value.get("confidence") is not None
            else None
        ),
        details=details,
        event_id=str(value.get("event_id") or uuid4().hex),
    )


def normalize_events(
    raw_events: RawEvent | Iterable[RawEvent] | None,
    *,
    default_source: str,
) -> list[ProctorEvent]:
    """Convert detector output into the shared event representation."""

    if raw_events is None:
        return []
    if isinstance(raw_events, (ProctorEvent, str, Mapping)):
        values: Iterable[RawEvent] = [raw_events]
    else:
        values = raw_events

    result: list[ProctorEvent] = []
    for value in values:
        if isinstance(value, ProctorEvent):
            result.append(value)
        elif isinstance(value, str):
            result.append(ProctorEvent.create(value, source=default_source))
        elif isinstance(value, Mapping):
            result.append(_from_mapping(value, default_source))
        else:
            raise TypeError(f"unsupported event value: {type(value).__name__}")
    return result

