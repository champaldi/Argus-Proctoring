"""Короткое текстовое заключение по сессии для преподавателя.

Заключение строится по простым правилам из уже записанных событий. Это
подсказка, куда смотреть, а не вердикт: решение остаётся за преподавателем.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .data import Session, recalculate_risk, risk_zone
from .rules import VIOLATION, classify, evaluate_rules


@dataclass(frozen=True, slots=True)
class Conclusion:
    zone: str
    headline: str
    reasons: tuple[str, ...]
    recommendation: str
    # Сработавшие правила корреляции и итог: violation, suspicious или clean.
    findings: tuple[str, ...] = ()
    classification: str = "clean"

    def as_text(self) -> str:
        lines = [self.headline]
        if self.findings:
            lines.append("Сработавшие правила:")
            lines.extend(f"• {finding}" for finding in self.findings)
            lines.append("События:")
        lines.extend(f"• {reason}" for reason in self.reasons)
        lines.append(self.recommendation)
        return "\n".join(lines)


HEADLINES = {
    "low": "Существенных признаков нарушения не обнаружено.",
    "medium": "Есть отдельные подозрительные моменты.",
    "high": "Высокий риск нарушения.",
}
VIOLATION_HEADLINE = "Есть признаки нарушения: сработало правило корреляции."
VIOLATION_RECOMMENDATION = (
    "Рекомендуется проверка преподавателем: начните с кадров сработавшего правила."
)
RECOMMENDATIONS = {
    "low": "Проверка кадров не обязательна.",
    "medium": "Рекомендуется просмотреть отмеченные кадры.",
    "high": "Рекомендуется проверка преподавателем: начните с ключевых моментов.",
}
# Формулировка причины и порядок важности при равном вкладе в риск.
REASON_PHRASES = {
    "phone_aimed_at_screen": "телефон наведён на экран",
    "phone_detected": "телефон в кадре",
    "multiple_faces": "в кадре несколько лиц",
    "remote_session": "тест запущен через удалённую сессию",
    "injected_input": "ввод не с физических устройств",
    "protection_disabled": "защита отключена вручную",
    "suspicious_process": "запущены запрещённые программы",
    "multiple_monitors": "подключено несколько мониторов",
    "no_face": "студента не было в кадре",
    "gaze_down": "долгий взгляд вниз",
    "gaze_side": "долгий взгляд в сторону",
    "window_switched": "переключение на другое окно",
    "hotkey_blocked": "попытки нажать заблокированные сочетания",
    "too_close_to_camera": "слишком близко к камере",
    "capture_protection_failed": "не удалось скрыть окно от захвата экрана",
}
MAX_REASONS = 4
# Эти сигналы не бывают случайными: даже один такой эпизод стоит посмотреть,
# каким бы низким ни был суммарный риск.
STRONG_SIGNALS = frozenset({
    "protection_disabled",
    "phone_aimed_at_screen",
    "phone_detected",
    "multiple_faces",
    "remote_session",
    "injected_input",
})


def _times(count: int) -> str:
    """Русское согласование: 1 раз, 2 раза, 5 раз, 21 раз, 22 раза."""
    tail = count % 100
    if 11 <= tail <= 14:
        return f"{count} раз"
    return f"{count} раза" if count % 10 in (2, 3, 4) else f"{count} раз"


def _minute(session: Session, occurred_at) -> int:
    seconds = max(0.0, (occurred_at - session.started_at).total_seconds())
    return int(seconds // 60) + 1


def build_conclusion(
    session: Session, false_positive_ids: Iterable[str] = ()
) -> Conclusion:
    """Собирает заключение без событий, отмеченных преподавателем как ошибочные."""
    excluded = set(false_positive_ids)
    zone = risk_zone(recalculate_risk(session, excluded))

    grouped: dict[str, dict[str, object]] = {}
    for stored in session.events:
        if stored.event.event_id in excluded:
            continue
        name = stored.event.type.value
        entry = grouped.setdefault(
            name, {"count": 0, "weight": 0.0, "first": stored.event.occurred_at}
        )
        entry["count"] = int(entry["count"]) + 1
        entry["weight"] = float(entry["weight"]) + stored.weight
        if stored.event.occurred_at < entry["first"]:
            entry["first"] = stored.event.occurred_at

    order = list(REASON_PHRASES)
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            -float(item[1]["weight"]),
            order.index(item[0]) if item[0] in order else len(order),
        ),
    )
    reasons = []
    for name, entry in ranked[:MAX_REASONS]:
        phrase = REASON_PHRASES.get(name, name)
        count = int(entry["count"])
        minute = _minute(session, entry["first"])
        first = "на" if count == 1 else "впервые на"
        reasons.append(f"{phrase} — {_times(count)}, {first} {minute}-й минуте")
    hidden = len(ranked) - len(reasons)
    if hidden > 0:
        reasons.append(f"и ещё типов событий: {hidden}")

    if not grouped:
        headline = (
            "Нарушений не зафиксировано."
            if not excluded
            else "После снятия ошибочных событий нарушений не осталось."
        )
        return Conclusion(zone, headline, (), RECOMMENDATIONS["low"])
    found = evaluate_rules(session, excluded)
    findings = tuple(finding.as_text(session) for finding in found)
    classification = classify(found)
    if classification == VIOLATION:
        return Conclusion(
            zone, VIOLATION_HEADLINE, tuple(reasons), VIOLATION_RECOMMENDATION,
            findings, classification,
        )
    wording = "medium" if zone == "low" and (STRONG_SIGNALS & grouped.keys() or found) else zone
    return Conclusion(
        zone, HEADLINES[wording], tuple(reasons), RECOMMENDATIONS[wording],
        findings, classification,
    )
