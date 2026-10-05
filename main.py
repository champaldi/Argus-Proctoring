"""Application entry point: ``python main.py``."""

from __future__ import annotations

import os
import sys

from config import AppConfig


def main() -> int:
    try:
        from ui.main_window import run_application
    except ImportError as exc:
        missing = getattr(exc, "name", None) or "dependency"
        print(
            f"Не удалось импортировать {missing}. "
            "Установите зависимости: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    try:
        os.environ.setdefault("PROCTOR_PHONE_MODULE", "detection.phone_detector")
        os.environ.setdefault("PROCTOR_GAZE_MODULE", "gaze_analyzer")
        config = AppConfig.from_env()
        config.ensure_directories()
        return run_application(config)
    except Exception as exc:
        print(f"Ошибка запуска: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

