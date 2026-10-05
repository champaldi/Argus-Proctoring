import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from core.detectors import DetectorCollection


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


if __name__ == "__main__":
    unittest.main()
