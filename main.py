"""Application entry point: ``python main.py``."""

from __future__ import annotations

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
        config = AppConfig.from_env()
        config.ensure_directories()
        return run_application(config)
    except Exception as exc:
        print(f"Ошибка запуска: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

