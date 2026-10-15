"""Имитация студента для проверки живого режима без камеры и моделей.

    python tools/live_demo.py 192.168.1.5
    python tools/live_demo.py 100.101.102.103 --name "Карим" --fast

Отправляет на панель (``python -m teacher --live``) начало теста, несколько
событий, на которые срабатывают правила S5, V1 и V2, и конец теста.
Через ``--token`` передаётся ``PROCTOR_LIVE_TOKEN``, если он задан у панели.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.live import LiveReporter, normalize_url  # noqa: E402
from core.storage import StoredEvent  # noqa: E402
from events import ProctorEvent  # noqa: E402

SCRIPT = (
    ("gaze_side", 4.0, "долгий взгляд в сторону"),
    ("gaze_side", 4.0, "ещё один"),
    ("gaze_side", 4.0, "третий за 5 минут → правило S5"),
    ("phone_detected", 12.0, "телефон в кадре → S1"),
    ("gaze_down", 6.0, "взгляд вниз рядом с телефоном → V2"),
    ("phone_aimed_at_screen", 25.0, "телефон наведён на экран → V1"),
)


def check(base: str) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(base + "/api/health", timeout=3) as response:
            return bool(json.load(response).get("ok"))
    except Exception as exc:  # noqa: BLE001 — показать любую причину
        print(f"Панель не отвечает по {base}: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка живого режима без камеры")
    parser.add_argument("url", help="адрес панели, например 192.168.1.5")
    parser.add_argument("--name", default="Проверка связи", help="имя студента")
    parser.add_argument("--token", default=None, help="PROCTOR_LIVE_TOKEN панели")
    parser.add_argument("--fast", action="store_true", help="без пауз между событиями")
    args = parser.parse_args()

    base = normalize_url(args.url)
    print(f"Панель: {base}")
    if not check(base):
        print("Проверьте: панель запущена с --live, адрес и порт верные, "
              "брандмауэр Windows разрешил доступ, оба компьютера в одной сети.")
        return 1
    print("Связь есть.")

    reporter = LiveReporter(base, token=args.token, thumbnailer=lambda _: None)
    reporter.start()
    session = uuid4().hex
    reporter.session_started(session, {"student_name": args.name})
    risk = 0.0
    for kind, weight, note in SCRIPT:
        time.sleep(0 if args.fast else 3)
        risk = min(100.0, risk + weight)
        stored = StoredEvent(
            ProctorEvent.create(kind, source="live_demo"),
            weight=weight, risk_total=risk, screenshot_path=None,
        )
        reporter.event_recorded(session, stored)
        print(f"  отправлено: {note}")
    time.sleep(0 if args.fast else 2)
    reporter.session_finished(session, risk)
    reporter.close(timeout=5)
    if reporter.pending():
        print(f"Не доставлено: {reporter.pending()} ({reporter.last_error})")
        return 1
    print(f"Готово: {reporter.sent} сообщений. На панели должны быть уведомления S5, S1/V2, V1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
