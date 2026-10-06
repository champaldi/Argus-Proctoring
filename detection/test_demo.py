"""Проверка сохранения исходных кадров из окна демонстрации."""

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from detection import demo


class Camera:
    def __init__(self):
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def isOpened(self):
        return True

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass


class DemoTests(unittest.TestCase):
    def test_p_and_n_save_original_frame_without_drawn_box(self):
        detector = SimpleNamespace(
            detect=lambda frame: [],
            close=Mock(),
            last_detections=[SimpleNamespace(class_id=67, confidence=0.9,
                                             bbox=(10, 10, 50, 50), aspect=1.0,
                                             verifier_score=0.8)],
            last_phone_candidates=[SimpleNamespace(
                detection=SimpleNamespace(bbox=(60, 10, 90, 40), verifier_score=0.19),
                rejected_reason="verifier_score 0.19 < 0.20",
            )],
        )
        with tempfile.TemporaryDirectory() as temporary:
            with (patch.object(demo, "SAMPLES_DIR", Path(temporary)),
                  patch.object(demo, "PhoneDetector", return_value=detector),
                  patch.object(demo.cv2, "VideoCapture", return_value=Camera()),
                  patch.object(demo.cv2, "imshow"),
                  patch.object(demo.cv2, "putText") as put_text,
                  patch.object(demo.cv2, "waitKey", side_effect=[ord("p"), ord("n"), ord("q")]),
                  patch.object(demo.cv2, "destroyAllWindows"),
                  patch("sys.stdout", new_callable=io.StringIO)):
                self.assertEqual(demo.main(), 0)
            self.assertTrue(any(call.args[1] == "phone 0.90 clip=0.80"
                                for call in put_text.call_args_list))
            self.assertTrue(any(call.args[1] == "rejected clip=0.19"
                                for call in put_text.call_args_list))
            detector.close.assert_called_once()
            for label in ("phone", "not_phone"):
                files = list((Path(temporary) / label).glob("*.jpg"))
                self.assertEqual(len(files), 1)
                saved = cv2.imread(str(files[0]))
                self.assertEqual(saved.shape, (100, 100, 3))
                self.assertTrue(np.all(saved == 0))


if __name__ == "__main__":
    unittest.main()
