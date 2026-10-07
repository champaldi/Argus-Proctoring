"""Правила корреляции: что произошло в сессии, а не сколько набрано баллов.

Каждое правило смотрит на сочетание событий во времени, как правило
детектирования в SIEM. Одно событие само по себе чаще всего лишь наблюдение;
нарушением считается сочетание, которое трудно объяснить случайностью.
Правила работают по уже записанному журналу и не меняют приложение студента.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Iterable, Sequence

from core.storage import StoredEvent

from .data import Session

SUSPICIOUS = "suspicious"
VIOLATION = "violation"
LEVEL_TITLES = {SUSPICIOUS: "Подозрительно", VIOLATION: "Признаки нарушения"}


@dataclass(frozen=True, slots=True)
class Finding:
    rule_id: str
    level: str
    title: str
    explanation: str
    occurred_at: datetime
    event_ids: tuple[str, ...]

    def as_text(self, session: Session) -> str:
        seconds = max(0.0, (self.occurred_at - session.started_at).total_seconds())
        minute = int(seconds // 60) + 1
        return (
            f"{LEVEL_TITLES[self.level]}: {self.title} ({self.rule_id}, "
            f"{minute}-я минута) — {self.explanation}"
        )


Events = Sequence[StoredEvent]
Match = tuple[StoredEvent, ...]


def _of(events: Events, *names: str) -> list[StoredEvent]:
    return [stored for stored in events if stored.event.type.value in names]


def _single(*names: str) -> Callable[[Events], Match | None]:
    def match(events: Events) -> Match | None:
        found = _of(events, *names)
        return tuple(found) if found else None

    return match


def _pair(first: Sequence[str], second: Sequence[str], seconds: float):
    """Событие из первой группы и событие из второй не дальше ``seconds``."""
    window = timedelta(seconds=seconds)

    def match(events: Events) -> Match | None:
        for left in _of(events, *first):
            for right in _of(events, *second):
                if abs(left.event.occurred_at - right.event.occurred_at) <= window:
                    return tuple(sorted((left, right), key=lambda s: s.event.occurred_at))
        return None

    return match


def _repeated(names: Sequence[str], count: int, seconds: float):
    """Не меньше ``count`` событий за ``seconds``."""
    window = timedelta(seconds=seconds)

    def match(events: Events) -> Match | None:
        found = _of(events, *names)
        for start in range(len(found) - count + 1):
            group = found[start : start + count]
            if group[-1].event.occurred_at - group[0].event.occurred_at <= window:
                return tuple(group)
        return None

    return match


@dataclass(frozen=True, slots=True)
class Rule:
    rule_id: str
    level: str
    title: str
    explanation: str
    match: Callable[[Events], Match | None]
    # Более слабые правила, которые это правило уже объясняет.
    covers: tuple[str, ...] = ()


RULES: tuple[Rule, ...] = (
    Rule(
        "V1", VIOLATION, "Фотографирование экрана",
        "телефон наведён на экран",
        _single("phone_aimed_at_screen"), covers=("S1",),
    ),
    Rule(
        "V2", VIOLATION, "Списывание с телефона",
        "телефон в кадре и взгляд вниз в пределах 30 секунд",
        _pair(("phone_detected",), ("gaze_down",), 30), covers=("S1", "S2"),
    ),
    Rule(
        "V3", VIOLATION, "Удалённое управление компьютером",
        "тест запущен через удалённую сессию",
        _single("remote_session"), covers=("S6", "S8"),
    ),
    Rule(
        "V4", VIOLATION, "Управление извне",
        "ввод не с физических устройств при запущенной запрещённой программе",
        _pair(("injected_input",), ("suspicious_process",), 120), covers=("S6", "S8"),
    ),
    Rule(
        "V5", VIOLATION, "Подсказка от постороннего",
        "второй человек в кадре и взгляд в сторону в пределах 30 секунд",
        _pair(("multiple_faces",), ("gaze_side",), 30), covers=("S3",),
    ),
    Rule(
        "V6", VIOLATION, "Подмена или выход из-под наблюдения",
        "студент пропал из кадра, и рядом по времени в кадре телефон или другой человек",
        _pair(("no_face",), ("phone_detected", "multiple_faces"), 30), covers=("S4",),
    ),
    Rule(
        "S1", SUSPICIOUS, "Телефон в кадре",
        "телефон замечен, но признаков использования рядом по времени нет",
        _single("phone_detected"),
    ),
    Rule(
        "S2", SUSPICIOUS, "Систематические взгляды вниз",
        "три и более долгих взгляда вниз за 5 минут",
        _repeated(("gaze_down",), 3, 300),
    ),
    Rule(
        "S3", SUSPICIOUS, "Посторонний в кадре",
        "в кадре было несколько лиц",
        _single("multiple_faces"),
    ),
    Rule(
        "S4", SUSPICIOUS, "Студент покидал кадр",
        "лица не было в кадре два раза и более за 10 минут",
        _repeated(("no_face",), 2, 600),
    ),
    Rule(
        "S5", SUSPICIOUS, "Систематические взгляды в сторону",
        "три и более долгих взгляда в сторону за 5 минут",
        _repeated(("gaze_side",), 3, 300),
    ),
    Rule(
        "S6", SUSPICIOUS, "Запрещённая программа",
        "во время теста запущена программа из списка запрещённых",
        _single("suspicious_process"),
    ),
    Rule(
        "S7", SUSPICIOUS, "Попытки выйти из окна теста",
        "три и более заблокированных сочетания или переключения окна за 2 минуты",
        _repeated(("hotkey_blocked", "window_switched"), 3, 120),
    ),
    Rule(
        "S8", SUSPICIOUS, "Ввод не с физических устройств",
        "клавиатура или мышь управлялись программно",
        _single("injected_input"),
    ),
    Rule(
        "S9", SUSPICIOUS, "Ненадёжное окружение",
        "несколько мониторов или не удалось скрыть окно от захвата экрана",
        _single("multiple_monitors", "capture_protection_failed"),
    ),
)


def evaluate_rules(
    session: Session, false_positive_ids: Iterable[str] = ()
) -> tuple[Finding, ...]:
    """Сработавшие правила: сначала нарушения, затем подозрительное, по времени."""
    excluded = set(false_positive_ids)
    events = sorted(
        (stored for stored in session.events if stored.event.event_id not in excluded),
        key=lambda stored: stored.event.occurred_at,
    )
    matched: list[tuple[Rule, Match]] = []
    for rule in RULES:
        found = rule.match(events)
        if found:
            matched.append((rule, found))
    covered = {
        rule_id for rule, _ in matched if rule.level == VIOLATION for rule_id in rule.covers
    }
    findings = [
        Finding(
            rule.rule_id, rule.level, rule.title, rule.explanation,
            found[0].event.occurred_at,
            tuple(stored.event.event_id for stored in found),
        )
        for rule, found in matched
        if rule.rule_id not in covered
    ]
    findings.sort(key=lambda item: (item.level != VIOLATION, item.occurred_at))
    return tuple(findings)


def classify(findings: Sequence[Finding]) -> str:
    """Итог по сессии: ``violation``, ``suspicious`` или ``clean``."""
    levels = {finding.level for finding in findings}
    return VIOLATION if VIOLATION in levels else SUSPICIOUS if levels else "clean"
