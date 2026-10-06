from datetime import datetime, timezone
import unittest

from events import EventType, ProctorEvent, normalize_events


class EventContractTests(unittest.TestCase):
    def test_normalizes_supported_detector_outputs(self) -> None:
        events = normalize_events(
            [
                "phone_detected",
                {"type": "gaze_side", "confidence": 0.75, "angle": 32},
                {"type": EventType.GAZE_DOWN},
                "too_close_to_camera",
                ProctorEvent.create(EventType.NO_FACE, source="face"),
            ],
            default_source="test",
        )

        self.assertEqual([event.type for event in events], [
            EventType.PHONE_DETECTED,
            EventType.GAZE_SIDE,
            EventType.GAZE_DOWN,
            EventType.TOO_CLOSE_TO_CAMERA,
            EventType.NO_FACE,
        ])
        self.assertEqual(events[1].details["angle"], 32)
        self.assertEqual(events[1].source, "test")

    def test_parses_iso_timestamp(self) -> None:
        event = normalize_events(
            {"type": "no_face", "occurred_at": "2026-10-05T12:00:00Z"},
            default_source="face",
        )[0]
        self.assertEqual(
            event.occurred_at,
            datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc),
        )

    def test_rejects_invalid_confidence(self) -> None:
        with self.assertRaises(ValueError):
            ProctorEvent.create("gaze_down", source="face", confidence=1.2)

    def test_capture_protection_failure_uses_shared_contract(self) -> None:
        event = normalize_events(
            {"type": "capture_protection_failed", "target_hwnd": 100},
            default_source="security",
        )[0]
        self.assertEqual(event.type, EventType.CAPTURE_PROTECTION_FAILED)
        self.assertEqual(event.details, {"target_hwnd": 100})

    def test_remote_session_uses_shared_contract(self) -> None:
        event = ProctorEvent.create("remote_session", source="security")
        self.assertEqual(event.type, EventType.REMOTE_SESSION)

    def test_multiple_monitors_uses_shared_contract(self) -> None:
        event = ProctorEvent.create("multiple_monitors", source="security")
        self.assertEqual(event.type, EventType.MULTIPLE_MONITORS)

    def test_injected_input_uses_shared_contract(self) -> None:
        event = ProctorEvent.create("injected_input", source="security")
        self.assertEqual(event.type, EventType.INJECTED_INPUT)


if __name__ == "__main__":
    unittest.main()

