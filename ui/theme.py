"""Name and look of the student application; free of any Qt import.

One light theme shared by the start dialog and the test window. The palette
is the team's: off-white page, deep blue brand, graphite text, three status colours.
"""

from __future__ import annotations

# Shown in window titles. Change the name of the product here only.
APP_NAME = "Argus"
APP_TAGLINE = "локальный прокторинг"

# Palette chosen by the team: calm off-white page, deep blue brand, graphite text.
PAGE = "#F8F9FA"
CARD = "#FFFFFF"
BORDER = "#E5E7EB"
TEXT = "#374151"
HEADING = "#1F2937"
MUTED = "#6B7280"
ACCENT = "#1E3A8A"
ACCENT_DARK = "#172E6E"
ACCENT_SOFT = "#EEF2FF"
# Status colours of the palette. Text uses the darker shade of each, because
# the base colours are too light to read on the off-white page.
SAFE = "#10B981"
SAFE_TEXT = "#047857"
WARNING = "#F59E0B"
WARNING_TEXT = "#B45309"
CRITICAL = "#EF4444"
CRITICAL_TEXT = "#B91C1C"
# Inter ships with the application (ui/fonts); Segoe UI is the fallback.
FONT = '"Inter", "Segoe UI", sans-serif'


def window_title(section: str) -> str:
    return f"{APP_NAME} · {section}"


_BUTTONS = f"""
QPushButton {{ background: {ACCENT}; color: white; border: 0; border-radius: 10px;
               padding: 12px 22px; font-size: 14px; font-weight: 600; }}
QPushButton:hover {{ background: {ACCENT_DARK}; }}
QPushButton:disabled {{ background: #D1D5DB; color: #6B7280; }}
QPushButton#secondary {{ background: #E5E7EB; color: {HEADING}; }}
QPushButton#secondary:hover {{ background: #D1D5DB; }}
QPushButton#secondary:disabled {{ background: #F3F4F6; color: #9CA3AF; }}
"""

MAIN_WINDOW_STYLE = f"""
QMainWindow, QWidget#page {{ background: {PAGE}; }}
QWidget {{ color: {TEXT}; font-family: {FONT}; }}
QLabel {{ background: transparent; }}
QMessageBox {{ background: {PAGE}; }}
QMessageBox QLabel {{ color: {HEADING}; font-size: 15px; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; }}
QLabel#eyebrow {{ color: {ACCENT}; font-size: 12px; font-weight: 700; }}
QLabel#heading {{ color: {HEADING}; font-size: 26px; font-weight: 700; }}
QLabel#question {{ color: {HEADING}; font-size: 21px; font-weight: 600; }}
QLabel#status {{ padding: 7px 12px; background: {ACCENT_SOFT}; color: {ACCENT};
                 border-radius: 10px; font-size: 13px; font-weight: 600; }}
QRadioButton {{ background: {PAGE}; border: 1px solid {BORDER}; border-radius: 12px;
                padding: 15px 16px; font-size: 15px; }}
QRadioButton:hover {{ border-color: {ACCENT}; }}
QRadioButton:checked {{ background: {ACCENT_SOFT}; border: 2px solid {ACCENT};
                        padding: 14px 15px; font-weight: 600; }}
QStatusBar {{ background: {PAGE}; color: {CRITICAL_TEXT}; }}
{_BUTTONS}
"""

START_DIALOG_STYLE = f"""
QDialog {{ background: {PAGE}; }}
QWidget {{ color: {TEXT}; font-family: {FONT}; }}
QLabel {{ background: transparent; font-size: 14px; }}
QMessageBox {{ background: {PAGE}; }}
QLabel#eyebrow {{ color: {ACCENT}; font-size: 12px; font-weight: 700; }}
QLabel#heading {{ color: {HEADING}; font-size: 24px; font-weight: 700; }}
QLabel#hint {{ color: {MUTED}; font-size: 13px; }}
QLineEdit {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
             padding: 11px 12px; font-size: 15px; }}
QLineEdit:focus {{ border: 2px solid {ACCENT}; padding: 10px 11px; }}
QCheckBox {{ font-size: 14px; }}
{_BUTTONS}
"""

MONITOR_NOTE_STYLE = f"color: {MUTED}; font-size: 13px;"
STATUS_OK_COLOR = SAFE_TEXT
STATUS_WARNING_COLOR = WARNING_TEXT
# Dark enough to survive a phone photo of the light page.
WATERMARK_RGB = (31, 41, 55)
