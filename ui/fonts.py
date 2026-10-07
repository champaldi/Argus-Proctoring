"""Loads the bundled Inter typeface, so every computer shows the same text."""

from __future__ import annotations

from pathlib import Path

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FILES = ("Inter-Regular.ttf", "Inter-SemiBold.ttf", "Inter-Bold.ttf")
FONT_FAMILY = "Inter"


def font_paths() -> list[Path]:
    """Bundled font files that are actually present."""
    return [path for name in FONT_FILES if (path := FONT_DIR / name).is_file()]


def load_app_fonts(app) -> bool:
    """Register Inter and make it the application font; False keeps the system font."""
    # Imported here so this module can be checked without a display.
    from PySide6.QtGui import QFont, QFontDatabase

    loaded = False
    for path in font_paths():
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id >= 0 and FONT_FAMILY in QFontDatabase.applicationFontFamilies(font_id):
            loaded = True
    if loaded:
        font = QFont(FONT_FAMILY)
        font.setPointSize(10)
        app.setFont(font)
    return loaded
