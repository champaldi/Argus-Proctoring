"""Проверки порогов без камеры и загрузки весов YOLO."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import detection.phone_detector as phone_module
from detection.phone_detector import PhoneDetector


FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


def box(kind, y=10, confidence=0.9):
    class Coordinates(list):
        def tolist(self):
            return list(self)

    return SimpleNamespace(
        cls=[kind],
        conf=[confidence],
        xyxy=[Coordinates([10, y, 30, y + 20])],
    )


class FakeModel:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.calls = 0
        self.options = []

    def predict(self, frame, **kwargs):
        self.calls += 1
        self.options.append(kwargs)
        return [SimpleNamespace(boxes=next(self.frames))]


class PhoneDetectorTests(unittest.TestCase):
    def setUp(self):
        self.every_frame = patch.object(phone_module, "DETECT_EVERY_N_FRAMES", 1)
        self.every_frame.start()

    def tearDown(self):
        self.every_frame.stop()

    def test_phone_needs_three_of_five_model_runs_and_repeats_after_two_seconds(self):
        model = FakeModel([[box(0), box(67, 60)], [box(0), box(67, 60)],
                           [box(0), box(67, 60)], [box(0), box(67, 60)],
                           [box(0), box(67, 60)]])
        detector = PhoneDetector(model=model)
        times = [0, 0.1, 0.2, 1.0, 2.2]
        actual = [[event.type.value for event in detector.detect(FRAME, timestamp=t)]
                  for t in times]
        self.assertEqual(actual, [[], [], ["phone_detected"], [], ["phone_detected"]])

    def test_phone_window_accepts_one_missed_detection(self):
        model = FakeModel([[box(0), box(67, 60)], [box(0)], [box(0), box(67, 60)],
                           [box(0), box(67, 60)], [box(0), box(67, 60)]])
        detector = PhoneDetector(model=model)
        actual = [[event.type.value for event in detector.detect(FRAME, timestamp=t)]
                  for t in range(5)]
        self.assertEqual(actual, [[], [], [], ["phone_detected"], []])

    def test_phone_window_expires_after_older_hits_leave_it(self):
        frames = [[box(0), box(67, 60)]] * 3 + [[box(0)]] * 3 + [[box(0), box(67, 60)]] * 3
        detector = PhoneDetector(model=FakeModel(frames))
        actual = [[event.type.value for event in detector.detect(FRAME, timestamp=t)]
                  for t in range(9)]
        self.assertEqual(actual, [[], [], ["phone_detected"], [], [], [],
                                  [], [], ["phone_detected"]])

    def test_phone_has_own_confidence_threshold(self):
        model = FakeModel([[box(0, confidence=0.4), box(67, 60, 0.4)] for _ in range(3)])
        detector = PhoneDetector(model=model)
        actual = [detector.detect(FRAME, timestamp=t) for t in range(3)]
        self.assertEqual([event.type.value for event in actual[2]], ["phone_detected"])
        self.assertEqual([item.class_id for item in detector.last_detections], [67])
        self.assertEqual(model.options[0]["conf"], 0.35)

    def test_phone_below_own_threshold_is_ignored(self):
        model = FakeModel([[box(67, 60, 0.34)] for _ in range(3)])
        detector = PhoneDetector(model=model)
        actual = [detector.detect(FRAME, timestamp=t) for t in range(3)]
        self.assertEqual(actual, [[], [], []])

    def test_upper_phone_needs_more_than_one_point_five_seconds(self):
        model = FakeModel([[box(0), box(67, 30)] for _ in range(4)])
        detector = PhoneDetector(model=model)
        actual = [[event.type.value for event in detector.detect(FRAME, timestamp=t)]
                  for t in [0, 1.5, 1.6, 1.7]]
        self.assertEqual(actual, [[], [], ["phone_detected", "phone_aimed_at_screen"], []])

    def test_lower_phone_does_not_count_as_aimed(self):
        model = FakeModel([[box(0), box(67, 60)] for _ in range(4)])
        detector = PhoneDetector(model=model)
        actual = [[event.type.value for event in detector.detect(FRAME, timestamp=t)]
                  for t in [0, 1, 2, 3]]
        self.assertEqual(actual, [[], [], ["phone_detected"], []])

    def test_aimed_timer_survives_brief_missing_detection(self):
        model = FakeModel([[box(0), box(67, 30)], [box(0), box(67, 30)],
                           [box(0)], [box(0), box(67, 30)], [box(0), box(67, 30)]])
        detector = PhoneDetector(model=model)
        actual = [[event.type.value for event in detector.detect(FRAME, timestamp=t)]
                  for t in [0, 0.6, 1.0, 1.2, 1.6]]
        self.assertIn("phone_aimed_at_screen", actual[4])

    def test_aimed_timer_resets_after_long_absence(self):
        model = FakeModel([[box(0), box(67, 30)], [box(0)], [box(0)],
                           [box(0), box(67, 30)], [box(0), box(67, 30)],
                           [box(0), box(67, 30)]])
        detector = PhoneDetector(model=model)
        actual = [[event.type.value for event in detector.detect(FRAME, timestamp=t)]
                  for t in [0, 0.4, 1.2, 1.3, 2.0, 2.9]]
        self.assertNotIn("phone_aimed_at_screen", actual[4])
        self.assertIn("phone_aimed_at_screen", actual[5])

    def test_no_person_is_disabled_by_default(self):
        model = FakeModel([[], [], [], [], [], [box(0)], []])
        detector = PhoneDetector(model=model)
        actual = [detector.detect(FRAME, timestamp=t) for t in [0, 3, 3.1, 4, 5.2, 6, 7]]
        self.assertEqual([[e.type.value for e in events] for events in actual],
                         [[], [], [], [], [], [], []])

    def test_no_person_can_be_enabled_explicitly(self):
        model = FakeModel([[], [], [], [], []])
        detector = PhoneDetector(model=model)
        with patch.object(phone_module, "ENABLE_NO_PERSON", True):
            actual = [detector.detect(FRAME, timestamp=t) for t in [0, 3, 3.1, 4, 5.2]]
        self.assertEqual([[e.type.value for e in events] for events in actual],
                         [[], [], ["no_face"], [], ["no_face"]])
        self.assertEqual(actual[2][0].details["observation"], "no_person")

    def test_skipped_frames_keep_aimed_timer_and_phone_streak(self):
        model = FakeModel([[box(0), box(67, 30)] for _ in range(3)])
        detector = PhoneDetector(model=model)
        with patch.object(phone_module, "DETECT_EVERY_N_FRAMES", 3):
            actual = [
                [event.type.value for event in detector.detect(FRAME, timestamp=t * 0.5)]
                for t in range(9)
            ]
        self.assertEqual(model.calls, 3)
        self.assertEqual(actual, [[], [], [], [], [], [],
                                  ["phone_detected", "phone_aimed_at_screen"], [], []])

    def test_model_path_is_next_to_module(self):
        with patch("ultralytics.YOLO", return_value=FakeModel([])) as yolo:
            PhoneDetector()
        path = Path(yolo.call_args.args[0])
        self.assertEqual(path.parent, Path(phone_module.__file__).resolve().parent)
        self.assertEqual(path.name, "yolov8s.pt")


if __name__ == "__main__":
    unittest.main()
