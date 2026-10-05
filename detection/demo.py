"""Демонстрация детектора на веб-камере; запуск: python -m detection.demo."""

from __future__ import annotations

import json
import sys
import time

import cv2

from events import EventType
from .phone_detector import PERSON_CLASS_ID, PhoneDetector


ALERT_SECONDS = 2.0


def main() -> int:
    """Читает кадры до нажатия q и показывает рамки объектов."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    camera = cv2.VideoCapture(0, backend)
    try:
        if not camera.isOpened():
            print("Не удалось открыть веб-камеру 0", file=sys.stderr)
            return 1
        detector = PhoneDetector()
        alert_until = 0.0
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Не удалось прочитать кадр с камеры", file=sys.stderr)
                return 1

            for event in detector.detect(frame):
                print(json.dumps(event.to_dict(), ensure_ascii=False), flush=True)
                if event.type == EventType.PHONE_AIMED_AT_SCREEN:
                    alert_until = time.monotonic() + ALERT_SECONDS

            # Рисуем поверх копии, чтобы детектор всегда видел исходный кадр.
            preview = frame.copy()
            for item in detector.last_detections:
                x1, y1, x2, y2 = item.bbox
                is_person = item.class_id == PERSON_CLASS_ID
                color = (0, 180, 0) if is_person else (0, 0, 255)
                label = "person" if is_person else "phone"
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    preview, f"{label} {item.confidence:.2f}",
                    (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA,
                )
            if time.monotonic() < alert_until:
                # Красная полоса остаётся видимой две секунды после события.
                cv2.rectangle(preview, (0, 0), (preview.shape[1], 58), (0, 0, 180), -1)
                cv2.putText(
                    preview, "PHONE AIMED AT SCREEN!", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
                )
            cv2.imshow("Phone detector - q to quit", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
