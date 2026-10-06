"""Создаёт шесть учебных сессий в обычной схеме EventStore."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from core.risk import RiskScorer
from core.storage import EventStore
from events import EventType, ProctorEvent


DEMO_DIR = Path(__file__).resolve().parent / "demo_data"
BASE_TIME = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
SCENARIOS = (
    ("demo · Анна", 18, (("gaze_side", 10, 3.0),)),
    ("demo · Борис", 22, (("phone_detected", 10, 2.0),)),
    ("demo · Вера", 26, (("gaze_down", 900, 2.0),
                          ("multiple_faces", 8, 4.0), ("gaze_side", 3, 2.0))),
    ("demo · Глеб", 30, (("window_switched", 900, 1.0),
                          ("phone_detected", 8, 3.0), ("gaze_down", 3, 2.0))),
    ("demo · Дарья", 24, (("gaze_side", 600, 2.0),
                           ("phone_aimed_at_screen", 8, 5.0),
                           ("multiple_faces", 3, 2.0))),
    ("demo · Егор", 28, (("multiple_monitors", 600, 2.0),
                          ("remote_session", 8, 4.0),
                          ("phone_detected", 3, 2.0))),
)


def _placeholder(index: int, event_name: str) -> np.ndarray:
    """Рисует простой кадр-заглушку без камеры и внешних изображений."""
    frame = np.full((360, 640, 3), (28, 20, 12), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (640, 16), (246, 125, 93), -1)
    cv2.putText(frame, "DEMO FRAME", (36, 105), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (235, 240, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Session {index}", (38, 165), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (157, 180, 212), 2, cv2.LINE_AA)
    cv2.putText(frame, event_name.replace("_", " "), (38, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (109, 213, 243), 2, cv2.LINE_AA)
    cv2.putText(frame, "Placeholder - no camera recording", (38, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (147, 160, 184), 1, cv2.LINE_AA)
    return frame


def seed_demo(database_path: Path, screenshots_dir: Path) -> None:
    """Перезаписывает только шесть demo-сессий в указанной базе."""
    store = EventStore(database_path, screenshots_dir)
    try:
        for index, (name, minutes, observations) in enumerate(SCENARIOS, start=1):
            session_id = f"teacher-demo-{index:02d}"
            started = BASE_TIME + timedelta(days=index - 1)
            ended = started + timedelta(minutes=minutes)
            store.connection.execute("DELETE FROM events WHERE session_id=?", (session_id,))
            store.connection.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            store.connection.commit()
            store.start_session(session_id, {
                "application": "proctoring", "student_name": name, "demo": True,
            })
            clock = [started.timestamp()]
            scorer = RiskScorer(clock=lambda: clock[0])
            for event_index, (event_type, seconds_before_end, duration) in enumerate(
                observations, start=1
            ):
                occurred = ended - timedelta(seconds=seconds_before_end)
                event = ProctorEvent(
                    type=EventType(event_type), source="teacher_demo",
                    occurred_at=occurred,
                    confidence=0.92,
                    details={"duration_seconds": duration, "demo": True},
                    event_id=f"teacher-demo-{index:02d}-{event_index:02d}",
                )
                clock[0] = occurred.timestamp()
                update = scorer.add(event)
                store.record_event(
                    event, session_id=session_id, weight=update.weight,
                    risk_total=update.total, frame=_placeholder(index, event_type),
                )
            clock[0] = ended.timestamp()
            final_risk = scorer.peak().total
            store.finish_session(session_id, final_risk=final_risk)
            store.connection.execute(
                """UPDATE sessions SET started_at=?, ended_at=?, status='completed',
                   final_risk=? WHERE id=?""",
                (started.isoformat(), ended.isoformat(), final_risk, session_id),
            )
            store.connection.commit()
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Создать шесть demo-сессий")
    parser.add_argument("--directory", type=Path, default=DEMO_DIR,
                        help="папка для учебной базы и кадров")
    args = parser.parse_args(argv)
    database = args.directory / "proctoring.db"
    seed_demo(database, args.directory / "screenshots")
    print(f"Созданы 6 demo-сессий: {database.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
