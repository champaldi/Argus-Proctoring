"""Text of the on-screen watermark; free of any UI dependency."""

from __future__ import annotations

from datetime import datetime

WATERMARK_FALLBACK_NAME = "Студент"


def watermark_text(
    student_name: str | None, session_id: str, moment: datetime | None = None
) -> str:
    """Who is taking the test, which session and when.

    A photo of the screen carries this line, so a leaked question can be
    traced to the session it was taken in.
    """
    name = " ".join((student_name or "").split()) or WATERMARK_FALLBACK_NAME
    stamp = (moment or datetime.now()).strftime("%d.%m.%Y %H:%M")
    return f"{name} · {session_id[:8]} · {stamp}"
