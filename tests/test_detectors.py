import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from core.detectors import DetectorCollection
from core.security import SecurityAdapter


class DetectorCollectionTests(unittest.TestCase):
    def test_same_frame_object_goes_to_both_modules(self) -> None:
        phone = ModuleType("test_phone_module")
        gaze = ModuleType("test_gaze_module")
        received: list[tuple[str, object]] = []
        resets: list[str] = []

        phone.detect = lambda frame: received.append(("phone", frame)) or []
        phone.reset_default_detector = lambda: resets.append("phone")
        gaze.analyze = lambda frame: received.append(("gaze", frame)) or []
        gaze.reset_default_analyzer = lambda: resets.append("gaze")

        with patch.dict(
            sys.modules,
            {"test_phone_module": phone, "test_gaze_module": gaze},
        ):
            collection = DetectorCollection("test_phone_module", "test_gaze_module")
            statuses = collection.load()
            frame = object()
            events, errors = collection.analyze(frame)
            collection.close()

        self.assertTrue(all(status.loaded for status in statuses))
        self.assertEqual(events, [])
        self.assertEqual(errors, [])
        self.assertEqual([name for name, _ in received], ["phone", "gaze"])
        self.assertIs(received[0][1], frame)
        self.assertIs(received[1][1], frame)
        self.assertEqual(resets, ["phone", "gaze"])


class SecurityAdapterTests(unittest.TestCase):
    def test_passes_callback_and_explicit_test_window_handle(self) -> None:
        module = ModuleType("test_security_module")
        calls: list[tuple[object, int | None]] = []
        received = []

        def enable(callback=None, *, hwnd=None):
            calls.append((callback, hwnd))
            callback({"type": "window_switched", "details": {"focus_restored": True}})

        module.enable = enable
        module.disable = lambda: None
        with patch.dict(sys.modules, {"test_security_module": module}):
            adapter = SecurityAdapter("test_security_module", received.append)
            enabled, _ = adapter.enable(hwnd=321)
            adapter.disable()

        self.assertTrue(enabled)
        self.assertEqual(calls[0][1], 321)
        self.assertEqual(received[0].type.value, "window_switched")
        self.assertEqual(received[0].details["focus_restored"], True)


if __name__ == "__main__":
    unittest.main()
