"""Shared human-readable event details for both result screens."""

from collections.abc import Mapping

from events import EventType, ProctorEvent


EVENT_LABELS = {
    "phone_detected": "Обнаружен телефон",
    "phone_aimed_at_screen": "Телефон направлен на экран",
    "gaze_down": "Долгий взгляд вниз",
    "gaze_side": "Долгий взгляд в сторону",
    "no_face": "Лицо не обнаружено",
    "multiple_faces": "В кадре несколько лиц",
    "too_close_to_camera": "Слишком близко к камере",
    "hotkey_blocked": "Заблокирована комбинация клавиш",
    "window_switched": "Переключение окна",
    "suspicious_process": "Обнаружен запрещённый процесс",
    "capture_protection_failed": "Не удалось скрыть окно от захвата",
    "remote_session": "Тест запущен через удалённую сессию",
    "multiple_monitors": "Подключено несколько мониторов",
    "injected_input": "Обнаружен программно внедрённый ввод",
    "protection_disabled": "Защита отключена вручную (Ctrl+Alt+F12)",
}


def application_details(event: ProctorEvent) -> list[str]:
    """Describe the exact processes and services recorded by protection."""
    if event.type != EventType.SUSPICIOUS_PROCESS:
        return []
    lines = []
    for process in event.details.get("processes") or []:
        if not isinstance(process, Mapping) or not process.get("name"):
            continue
        line = f"Приложение: {process['name']}"
        if process.get("pid") is not None:
            line += f" (PID {process['pid']})"
        lines.append(line)
    for service in event.details.get("services") or []:
        if not isinstance(service, Mapping):
            continue
        name = service.get("name")
        display_name = service.get("display_name") or name
        if not display_name:
            continue
        line = f"Служба: {display_name}"
        if name and name != display_name:
            line += f" ({name})"
        lines.append(line)
    return lines


