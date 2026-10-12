"""Дослать преподавателю завершённые сессии с этого ноутбука.

    python tools/sync_sessions.py 192.168.1.5

То же самое Argus делает сам при каждом запуске теста с заданным адресом
преподавателя. Ручной запуск нужен, если тест больше запускать не будут, а
кадры должны оказаться в журнале преподавателя. Повтор безопасен: уже
полученное сервер узнаёт по ID.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import AppConfig  # noqa: E402
from core.live import normalize_url, resync_finished_sessions, teacher_url_from_env  # noqa: E402


def main() -> int:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Досылка сессий преподавателю")
    parser.add_argument("url", nargs="?", help="адрес панели; по умолчанию из data/teacher_url.txt")
    parser.add_argument("--token", default=os.getenv("PROCTOR_LIVE_TOKEN"))
    args = parser.parse_args()
    url = normalize_url(args.url or "") or teacher_url_from_env(config.data_dir)
    if not url:
        print("Адрес преподавателя не задан: укажите его аргументом или в data/teacher_url.txt")
        return 1
    print(f"База: {config.database_path}\nПанель: {url}")
    synced, error = resync_finished_sessions(config.database_path, url, token=args.token)
    print(f"Досланы сессий: {synced}")
    if error:
        print(f"Остановлено на ошибке, повторите позже: {error}")
        return 1
    print("Все завершённые сессии есть у преподавателя.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
