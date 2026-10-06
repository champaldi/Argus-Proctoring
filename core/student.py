"""Student identity entered before the test; free of any UI dependency."""

from __future__ import annotations

import os

STUDENT_NAME_ENV = "PROCTOR_STUDENT_NAME"
STUDENT_NAME_MIN_LENGTH = 2
STUDENT_NAME_MAX_LENGTH = 80


def normalize_student_name(value: str | None) -> str | None:
    """Collapse whitespace and reject names that are too short to identify anyone."""
    if value is None:
        return None
    name = " ".join(str(value).split())[:STUDENT_NAME_MAX_LENGTH].strip()
    return name if len(name) >= STUDENT_NAME_MIN_LENGTH else None


def student_name_from_env() -> str | None:
    """Name preset for automated runs; skips the start dialog when valid."""
    return normalize_student_name(os.getenv(STUDENT_NAME_ENV))


def session_metadata(student_name: str | None) -> dict[str, str]:
    """Metadata stored with a session; the teacher panel reads ``student_name``."""
    name = normalize_student_name(student_name)
    return {"student_name": name} if name else {}
