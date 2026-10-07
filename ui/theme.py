"""Name and look of the student application; free of any Qt import.

One light theme shared by the start dialog and the test window. The colours
match the project presentation, so the demo video and the slides look alike.
"""

from __future__ import annotations

# Shown in window titles. Change the name of the product here only.
APP_NAME = "Adal"
APP_TAGLINE = "честный экзамен"

PAGE = "#f7f6f0"
CARD = "#fffefa"
BORDER = "#dfe2d8"
TEXT = "#15231d"
MUTED = "#51605a"
ACCENT = "#1e7a55"
ACCENT_DARK = "#17603f"
ACCENT_SOFT = "#e4f1ea"
WARNING = "#9a5a00"
FONT = '"Segoe UI Variable Text", "Segoe UI", "Inter", sans-serif'


def window_title(section: str) -> str:
    return f"{APP_NAME} · {section}"


_BUTTONS = f"""
QPushButton {{ background: {ACCENT}; color: white; border: 0; border-radius: 10px;
               padding: 12px 22px; font-size: 14px; font-weight: 600; }}
QPushButton:hover {{ background: {ACCENT_DARK}; }}
QPushButton:disabled {{ background: #d5dad2; color: #8a958e; }}
QPushButton#secondary {{ background: #e9ebe3; color: {TEXT}; }}
QPushButton#secondary:hover {{ background: #dfe2d8; }}
QPushButton#secondary:disabled {{ background: #eef0e9; color: #a3aca6; }}
"""

MAIN_WINDOW_STYLE = f"""
QMainWindow, QWidget#page {{ background: {PAGE}; }}
QWidget {{ color: {TEXT}; font-family: {FONT}; }}
QLabel {{ background: transparent; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; }}
QLabel#eyebrow {{ color: {ACCENT}; font-size: 12px; font-weight: 700; }}
QLabel#heading {{ font-size: 26px; font-weight: 700; }}
QLabel#question {{ font-size: 21px; font-weight: 600; }}
QLabel#status {{ padding: 7px 12px; background: {ACCENT_SOFT}; color: {ACCENT_DARK};
                 border-radius: 10px; font-size: 13px; font-weight: 600; }}
QRadioButton {{ background: {PAGE}; border: 1px solid {BORDER}; border-radius: 12px;
                padding: 15px 16px; font-size: 15px; }}
QRadioButton:hover {{ border-color: {ACCENT}; }}
QRadioButton:checked {{ background: {ACCENT_SOFT}; border: 2px solid {ACCENT};
                        padding: 14px 15px; font-weight: 600; }}
QStatusBar {{ background: {PAGE}; color: {WARNING}; }}
{_BUTTONS}
"""

START_DIALOG_STYLE = f"""
QDialog {{ background: {PAGE}; }}
QWidget {{ color: {TEXT}; font-family: {FONT}; }}
QLabel {{ background: transparent; font-size: 14px; }}
QLabel#eyebrow {{ color: {ACCENT}; font-size: 12px; font-weight: 700; }}
QLabel#heading {{ font-size: 24px; font-weight: 700; }}
QLabel#hint {{ color: {MUTED}; font-size: 13px; }}
QLineEdit {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
             padding: 11px 12px; font-size: 15px; }}
QLineEdit:focus {{ border: 2px solid {ACCENT}; padding: 10px 11px; }}
QCheckBox {{ font-size: 14px; }}
{_BUTTONS}
"""

MONITOR_NOTE_STYLE = f"color: {MUTED}; font-size: 13px;"
STATUS_OK_COLOR = ACCENT_DARK
STATUS_WARNING_COLOR = WARNING
# Dark enough to survive a phone photo of the light page.
WATERMARK_RGB = (40, 70, 55)
