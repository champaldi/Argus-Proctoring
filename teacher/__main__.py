"""Запуск панели: python -m teacher [--demo]."""

from __future__ import annotations

import argparse
from pathlib import Path

from PySide6.QtWidgets import QApplication

from config import AppConfig

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
    args = parser.parse_args(argv)
    source = DEMO_DIR / "proctoring.db" if args.demo else AppConfig.from_env().database_path
    database = args.db or source
    reviews = args.reviews or (DEMO_DIR if args.demo else STATE_DIR) / "reviews.db"
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Proctoring · Преподаватель")
    window = TeacherWindow(database, reviews)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
