"""Проверка оценки независимых кадров без настоящей модели и камеры."""

import io
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from detection.evaluate import evaluate_samples
from detection.test_phone_detector import FakeModel, box


class EvaluateTests(unittest.TestCase):
    def test_counts_hits_false_positives_and_reports_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder, names in (("phone", ["a.jpg"]),
                                  ("not_phone", ["b.jpg", "c.jpg"])):
                (root / folder).mkdir()
                for name in names:
                    self.assertTrue(cv2.imwrite(str(root / folder / name),
                                                np.zeros((100, 100, 3), dtype=np.uint8)))
            model = FakeModel([
                [box(67, confidence=0.9)],
                [box(67, confidence=0.47), box(65, confidence=0.76)],
                [box(67, confidence=0.6)],
            ])
            output = io.StringIO()
            result = evaluate_samples(model, root, output=output)

        self.assertEqual((result.phone_found, result.phone_total), (1, 1))
        self.assertEqual((result.false_positives, result.not_phone_total), (1, 2))
        self.assertEqual(model.options[0]["conf"], 0.05)
        report = output.getvalue()
        for text in ("a.jpg", "b.jpg", "c.jpg", "0.47", "a=1.00", "remote"):
            self.assertIn(text, report)

    def test_verify_and_save_debug(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Проверка"
            for folder in ("phone", "not_phone"):
                (root / folder).mkdir(parents=True)
                ok, encoded = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))
                self.assertTrue(ok)
                (root / folder / "a.jpg").write_bytes(encoded.tobytes())
            model = FakeModel([[box(67)], [box(67)]])
            scores = iter([0.8, 0.19])
            output = io.StringIO()
            result = evaluate_samples(
                model, root, use_verifier=True,
                verifier=lambda frame, bbox: next(scores),
                save_debug=True, output=output,
            )
            self.assertTrue((root / "_debug" / "phone_a.jpg").is_file())
            self.assertTrue((root / "_debug" / "not_phone_a.jpg").is_file())
        self.assertEqual((result.phone_found, result.false_positives), (1, 0))
        self.assertIn("verifier_score=0.800", output.getvalue())
        self.assertIn("verifier_score=0.190", output.getvalue())
        self.assertIn("(1 рамок)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
