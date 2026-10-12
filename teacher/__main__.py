"""Запуск панели: python -m teacher [--demo]."""

from __future__ import annotations

import argparse
import sqlite3
import os
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from ui.fonts import load_app_fonts
from ui.theme import APP_NAME

from config import AppConfig

from .archive import SessionArchive
from .live import DEFAULT_PORT, LiveBoard, LiveServer
from .seed_demo import DEMO_DIR
from .ui import TeacherWindow


STATE_DIR = Path(__file__).resolve().parent / "state"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Панель преподавателя")
    parser.add_argument("--demo", action="store_true",
                        help="читать шесть сессий из teacher/demo_data")
    parser.add_argument("--db", type=Path, help="путь к существующей базе сессий")
    parser.add_argument("--reviews", type=Path,
                        help="отдельная база решений преподавателя")
    parser.add_argument("--live", action="store_true",
                        help="принимать события студентов по локальной сети")
    parser.add_argument("--alerts", action="store_true",
                        help="всплывающие уведомления о сработавших правилах (с --live)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"порт для --live, по умолчанию {DEFAULT_PORT}")
    args = parser.parse_args(argv)
    source = DEMO_DIR / "proctoring.db" if args.demo else AppConfig.from_env().database_path
    database = args.db or source
    reviews = args.reviews or (DEMO_DIR if args.demo else STATE_DIR) / "reviews.db"
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(f"{APP_NAME} · Преподаватель")
    load_app_fonts(app)
    server = None
    if args.live:
        try:
            # Сессии с других ноутбуков пишутся в ту же базу, что читает панель.
            server = LiveServer(
                LiveBoard(), port=args.port, token=os.getenv("PROCTOR_LIVE_TOKEN") or None,
                archive=SessionArchive(database), reviews_path=reviews,
            )
            server.start()
        except (OSError, sqlite3.Error) as exc:
            QMessageBox.warning(
                None, "Живой режим не запущен",
                f"Не удалось запустить приём на порту {args.port}: {exc}\n"
                "Панель откроется без живых уведомлений.",
            )
            server = None
    window = TeacherWindow(database, reviews, live_server=server, show_alerts=args.alerts)
    window.show()
    try:
        return app.exec()
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
